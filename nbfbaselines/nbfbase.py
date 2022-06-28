
import os
import torch
import math
import numpy as np

from .tokenizer import tokenizer_from_config
from .models    import nbfmodel_from_config
from .config    import save_config_to_json, load_config_from_json

from .utils     import load_model_directory

from collections import OrderedDict


class NBFModel:

    def __init__(self, config):
        self.config    = config
        self.tokenizer = tokenizer_from_config(config.tokenizer_config)

        # Vocabulary size
        self.config.vocabulary_size = self.tokenizer.get_vocab_size()

        # Load target vocabulary
        self.target_vocabulary = self.tokenizer.target_vocab
        if self.target_vocabulary is not None:
            config.target_vocabulary_size = len(self.target_vocabulary)
        else:
            config.target_vocabulary_size = -1

        self.max_length = self.config.max_sequence_length

        self.model     = nbfmodel_from_config(config)
        self.device    = torch.device("cpu")

    # Load / Save utils --------------------------------
    @staticmethod
    def from_pretrained(name_or_path):
        path_to_pretrained = load_model_directory(name_or_path)

        config_path = os.path.join(path_to_pretrained, "config.json")
        config      = load_config_from_json(config_path)
        nbf_model   = NBFModel(config)
        
        # Load model
        model = nbf_model.model
        model_path = os.path.join(path_to_pretrained, "model.pt")
        state_dict = torch.load(model_path, 
                        map_location=torch.device("cpu"))
        model.load_state_dict(
            state_dict
        )

        return nbf_model


    def save_pretrained(self, path_to_pretrained):
        if os.path.exists(path_to_pretrained):
            raise ValueError("Path for pretrained model already exists: %s" % path_to_pretrained)

        # Create path
        os.makedirs(path_to_pretrained)
        model_path  = os.path.join(path_to_pretrained, "model.pt")

        save_config_to_json(self.config, path_to_pretrained)
        torch_model = self.model
        torch.save(torch_model.state_dict(), model_path)


    # Decode candidate

    def _decode_candidate(self, tokens, error_index, repair_index, logprob, return_tokens = False):
        new_tokens = list(tokens)

        if error_index > 0: # The code is predicted to be incorrect
            if self.target_vocabulary is not None:
                if repair_index < len(self.target_vocabulary):
                    repair = self.target_vocabulary[repair_index]

                    if repair == "not":
                        error_token = new_tokens[error_index]
                        if error_token == "is":
                            repair = "is not"
                        else:
                            repair = "not %s" % error_token

                    new_tokens[error_index] = repair
                else:
                    new_tokens[error_index] = tokens[repair_index - len(self.target_vocabulary)]
            else:
                new_tokens[error_index] = tokens[repair_index]

        output_tokens = [t for t in new_tokens if t not in ["[CLS]", "[EOS]", "[PAD]"]]

        output_code = self.tokenizer.code_tokenizer.decode(output_tokens)    

        before_token = tokens[error_index]
        after_token  = new_tokens[error_index]
        total_prob   = np.exp(logprob)

        if error_index == 0: #or before_token == after_token:
            after_token = "#norepair#"

        result = {
            "text"  : output_code["input_text"],
            "before": before_token,
            "after" : after_token,
            "prob"  : total_prob 
        }    

        if return_tokens:
            result["tokens"] = new_tokens
            result["token_error_loc"] = error_index

        return result


    # Main method --------------------------------------

    def __call__(self, 
                    input_text = None, 
                    input_tokens = None,
                    topk = 1,
                    beam_width = 1,
                    return_tokens = False,
                    return_logits = False,
                    **kwargs):

        assert input_text is not None or input_tokens is not None, "Either input_text or input_tokens are required"

        if input_text is None:
            kwargs["input_tokens"] = input_tokens
            kwargs["pre_tokenized"] = True
        else:
            assert input_tokens is None, "You can provide both input text and tokens"
            kwargs["input_text"] = input_text

        input_dict, tokens = self.tokenizer(
            format = "pt",
            return_tokens = True,
            **kwargs
        )

        input_dict = {k: v.unsqueeze(0).to(self.device) for k, v in input_dict.items()}

        # Run the model --------------------------------
        model = self.model.eval()
        with torch.no_grad():
            model_output = model(**input_dict)
            loc_logits, repair_logits = model_output.loc_logits, model_output.repair_logits
            loc_logits, repair_logits = loc_logits[0], repair_logits[0]

            loc_logprobs = torch.nn.LogSoftmax(dim = -1)(loc_logits)
            repair_logprobs = torch.nn.LogSoftmax(dim = -1)(repair_logits)

        # Assume repair_logprobs = [num_tokens x (num_tokens + target)]
        if len(repair_logprobs.shape) == 1:
            # Currently repair_logprobs = [1 x (num_tokens + target)]
            repair_logprobs = repair_logprobs.unsqueeze(0).expand(loc_logits.shape[0], -1)

        if "location_mask" in input_dict:
            loc_logprobs = _mask(loc_logprobs, input_dict["location_mask"])

        loc_logprobs, repair_logprobs = loc_logprobs.cpu(), repair_logprobs.cpu()

        # Beam search over possible locations / repairs
        cluster_assignment = compute_repair_cluster(tokens, self.target_vocabulary)
        cluster_assignment = torch.LongTensor(cluster_assignment)
        candidates         = cluster_beam_search(
            cluster_assignment, 
            loc_logprobs,
            repair_logprobs,
            beam_width = max(topk, beam_width)
        )

        # Decode candidates
        candidates = [self._decode_candidate(tokens, *candidate, return_tokens = return_tokens)
                         for candidate in candidates][:topk]

        #if len(candidates) == 1: candidates = candidates[0]
        if return_logits:        return candidates, (loc_logits, repair_logits)

        return candidates

    # Length adjustment ----------------------------------------------------

    def increase_accepted_length(self, new_length):
        assert new_length >= self.max_length, "You can only increase the accepted length: %d" % self.max_length

        try:
            self.model.increase_token_length(new_length)
            print("Warning: The model might not be trained for programs with %d tokens." % new_length)
            print("Evaluting the model on longer lengths might result into suboptimal performance")
            self.max_length = new_length
        except (ValueError, AttributeError):
            print("Warning: Model cannot be adjusted to %d tokens." % new_length)
            print("Keep old length of %d tokens" % self.max_length)

    # Representation --------------------------------

    def to(self, device):
        self.device = device
        self.model  = self.model.to(self.device)

    def __repr__(self):
        model_clazz = self.model.__class__.__name__
        num_params  = num_parameters(self.model)
        num_params  = format_num(num_params)

        return "%s(num_params = %s)" % (model_clazz, num_params)



# Beam search ----------------------------------------------------------------

@torch.no_grad()
def beam_search(loc_logprobs, repair_logprobs, beam_width):
    
    candidates = []
    loc_width = min(beam_width, loc_logprobs.shape[0])
    loc_tk_logprobs, loc_tk_index = loc_logprobs.topk(loc_width)

    for i, index in enumerate(loc_tk_index):
        loc_logprob = loc_tk_logprobs[i]
        repair_logprob = repair_logprobs[index]
        repair_width   = min(beam_width, repair_logprob.shape[0])
        repair_tk_logprobs, repair_tk_index = repair_logprob.topk(repair_width)

        for j, rindex in enumerate(repair_tk_index):
            candidates.append(
                (index.item(), rindex.item(), (repair_tk_logprobs[j] + loc_logprob).item())
            )

    candidates = sorted(candidates, key=lambda x: x[2], reverse=True)
    return candidates[:beam_width]



@torch.no_grad()
def cluster_beam_search(cluster_assign, loc_logprobs, repair_logprobs, beam_width):
    
    # Cluster accumulate --------------------------------
    repair_probs = torch.exp(repair_logprobs)
    max_clusters = cluster_assign.max() + 1
    
    cluster_assign_broadcast = cluster_assign.unsqueeze(0).repeat(repair_probs.shape[0], 1)
    repair_cluster_probs = torch.zeros((repair_probs.shape[0], max_clusters))
    repair_cluster_probs.scatter_add_(1, cluster_assign_broadcast, repair_probs)
    repair_cluster_logprobs = torch.log(repair_cluster_probs)

    # Beam search --------------------------------
    candidates = beam_search(loc_logprobs, repair_cluster_logprobs, beam_width)

    # Cluster to token ---------------------------
    output_candidates = []

    for loc, repair_cluster, loc_repair_logprob in candidates:
        cluster_mask     = cluster_assign == repair_cluster
        loc_repair_probs = repair_probs[loc].clone()
        loc_repair_probs[~cluster_mask] = 0.0
        repair_index = loc_repair_probs.argmax().item()
        output_candidates.append((loc, repair_index, loc_repair_logprob))
    
    return output_candidates


def compute_repair_cluster(tokens, target_vocabulary = None):
    cluster = {}
    cluster_assign = []

    if target_vocabulary is not None:
        for vocab_token in target_vocabulary:
            if vocab_token not in cluster: cluster[vocab_token] = len(cluster)
            cluster_assign.append(cluster[vocab_token])

    for token in tokens:
        if token not in cluster: cluster[token] = len(cluster)
        cluster_assign.append(cluster[token])
    
    return cluster_assign

# Helper ----------------------------------------------------------------
def num_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_num(number):
    if number == 0: return "0"
    magnitude = int(math.log10(number)) // 3
    number /= 10**(magnitude * 3)
    return "%.2f%s" % (number, ["", "K", "M", "G", "T", "P"][magnitude])

def convert_to_old_format(state_dict):
    output = []

    for key, value in state_dict.items():
        if key.startswith("loc_head"):
            key = "prediction_head.%s" % key
        elif key.startswith("repair_head"):
            key = "prediction_head.%s" % key
        output.append((key, value))
    
    output.append((
        "prediction_head.loc_head.repr_to_output.bias",
        torch.zeros((1,))
    ))

    return OrderedDict(output)

def _mask(logits, mask):
    return mask * logits - 1e3 * (1 - mask)
