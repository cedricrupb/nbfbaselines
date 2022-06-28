import torch
import math
from torch import nn
from torch.nn import functional as F

from collections import namedtuple

from .utils import init_weights


def nbf_head_from_config(config):
    head_type = config.nbf_head_type

    if head_type == "none":
        return None

    if head_type == "joint":
        return JointLocRepairHead(config)

    if head_type == "loc":
        loc_head = LocalizationModule(config)
        return ConditionalLocRepairHead(config, loc_head, None)

    if head_type == "conditional":
        loc_head = MLPLocalizationModule(config)
        repair_head = MLPRepairModule(config)
        return ConditionalLocRepairHead(config, loc_head, repair_head)
    
    if head_type.startswith("conditional_"):
        loc_head, repair_head = None, None

        if head_type == "conditional_custom":
            loc_head = LocalizationModule(config)
            repair_head = RepairModule(config)

        if loc_head is not None:
            return ConditionalLocRepairHead(config, loc_head, repair_head)


    raise ValueError("Unknown loc repair head type: %s" % head_type)

# Localization and Repair --------------------------------

NBFPredictionResult = namedtuple('NBFPredictionResult', 
                                    [
                                        "loc_loss", 
                                        "repair_loss", 
                                        "loc_logits", 
                                        "repair_logits", 
                                        "input_encoding"
                                    ])

def _mask(logits, mask):
    return mask * logits - 1e3 * (1 - mask)

@torch.no_grad()
def convert_target_to_mask(repair_target, repair_mask, target_vocabulary_size):
    if len(repair_target.size()) > 1: repair_target = repair_target.squeeze(-1)

    add_mask = F.one_hot(repair_target, target_vocabulary_size + 1)
    add_mask = add_mask[:, 1:].to(repair_mask.device)
    return torch.cat((add_mask, repair_mask), dim = 1)


class BaseLocRepairHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def convert_target_to_mask(self, repair_target, repair_mask):
        return convert_target_to_mask(repair_target, repair_mask, self.config.target_vocabulary_size)

    def localization_loss(self, logits, location_mask):
        loc_logprobs = nn.LogSoftmax(dim = -1)(logits)
        return (-location_mask * loc_logprobs).sum(dim=1).mean()

    def repair_loss(self, repair_logits, repair_mask):
        assert repair_mask.shape[1] == repair_logits.shape[1]

        repair_log_probs = nn.LogSoftmax(dim = 1)(repair_logits)
        norm = repair_mask.sum(dim = -1).clamp_(0, 1)

        # Collect log probs
        # log sum_(t_i = w)(P(t_i)) = log sum_(t_i = w)(exp log P(t_i))
        #      = LSE(log P(t_i))
        repair_log_probs = _mask(repair_log_probs, repair_mask)
        per_example_loss = -norm * torch.logsumexp(repair_log_probs, dim = 1)
    
        return per_example_loss.mean()

            

# Join Loc and Repair head based on Vasic'19

class JointLocRepairHead(BaseLocRepairHead):

    def __init__(self, config):
        super().__init__(config)

        assert self.config.target_vocabulary_size == -1, \
            "Joint loc head does not support a target vocabulary"

        self.prediction_head = nn.Linear(config.hidden_size, 2)

    def forward(self, 
                hidden_encoding, 
                input_encoding = None,
                input_mask     = None,
                location_index = None,
                location_mask  = None,
                repair_mask    = None,
                repair_target  = None):

        if location_index is not None:
            assert location_mask is None, "Cannot provide both location index and mask"
            location_mask = F.one_hot(location_index.squeeze(1), 
                                     num_classes=hidden_encoding.shape[1])

        loc_repair_logits = self.prediction_head(hidden_encoding)
        
        if input_mask is not None:
            input_mask = input_mask.unsqueeze(2).repeat(1, 1, 2)
            loc_repair_logits = _mask(loc_repair_logits, input_mask)

        loc_logits    = loc_repair_logits[:, :, 0]
        repair_logits = loc_repair_logits[:, :, 1]

        loc_loss, repair_loss = None, None

        if location_mask is not None:
            loc_loss = self.localization_loss(loc_logits, location_mask)
        
        if repair_mask is not None:

            if repair_target is not None:
                repair_mask = self.convert_target_to_mask(repair_target, repair_mask)

            repair_loss = self.repair_loss(repair_logits, repair_mask)
        
        return NBFPredictionResult(loc_loss, repair_loss, loc_logits, 
                                    repair_logits, hidden_encoding)


# Conditional Loc Repair --------------------------------

class ConditionalLocRepairHead(BaseLocRepairHead):

    def __init__(self, config, loc_head, repair_head):
        super().__init__(config)
        self.loc_head    = loc_head
        self.repair_head = repair_head

    def forward(self, 
                hidden_encoding, 
                input_encoding = None,
                input_mask     = None,
                location_index = None,
                location_mask  = None,
                repair_mask    = None,
                repair_target  = None):

        loc_loss, repair_loss = None, None

        if location_index is not None:
            assert location_mask is None, "Cannot provide both location index and mask"
            location_mask = F.one_hot(location_index.squeeze(1), 
                                     num_classes=hidden_encoding.shape[1])

        # Localization
        loc_logits = self.loc_head(hidden_encoding, input_encoding, input_mask)

        if location_mask is not None:
            loc_loss = self.localization_loss(loc_logits, location_mask)
            error_hidden = hidden_encoding[location_mask.bool()]
        else:
            error_hidden = hidden_encoding

        # Repair
        repair_logits = None
        if self.repair_head is not None:
            repair_logits = self.repair_head(error_hidden, 
                                                hidden_encoding, 
                                                input_encoding = input_encoding,
                                                input_mask = input_mask)

            if repair_target is not None:
                repair_mask = self.convert_target_to_mask(repair_target, repair_mask)
        
            if repair_mask is not None:
                repair_loss = self.repair_loss(repair_logits, repair_mask)
        
        return NBFPredictionResult(loc_loss, repair_loss, loc_logits, 
                                    repair_logits, hidden_encoding)


# General model that works with inner repairs and localization --------------------------------

class LocalizationModule(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = self.config.hidden_size

        self.ffn = nn.Linear(3 * hidden_size, hidden_size, bias = True)
        self.repr_to_output = nn.Linear(hidden_size, 1, bias = True)

    def forward(self, input_embeds, target_embeds, input_mask = None):
        diff_embed = input_embeds - target_embeds
        full_vector = torch.cat([input_embeds, target_embeds, diff_embed], dim = -1)
        hidden_repr = torch.tanh(self.ffn(full_vector))
        output = self.repr_to_output(hidden_repr).squeeze(-1)

        if input_mask is not None: output = _mask(output, input_mask)

        return output


# Repair ---------------------------------------------------------------

class RepairModule(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.repair_query = nn.Linear(config.hidden_size, config.hidden_size, bias = False)
        self.repair_target = nn.Linear(config.hidden_size, config.hidden_size, bias = False)

        if self.config.target_vocabulary_size > 0:
            self.decoder = nn.Linear(config.hidden_size, config.target_vocabulary_size, bias = False)
            self.apply(init_weights)


    def forward(self, input_embeds, target_embeds, input_mask = None, input_encoding = None):

        if len(input_embeds.shape) < 3:
            input_embeds = input_embeds.unsqueeze(1)

        assert len(input_embeds.shape) == 3 and len(target_embeds.shape) == 3, "Input or target embedding is not in batch!"

        # Compute pointer from query to target token
        repair_query = self.repair_query(input_embeds)
        repair_targets = self.repair_target(target_embeds)

        assert repair_query.shape[0] == repair_targets.shape[0], "%s and %s does not match." % (str(repair_query.shape), str(repair_targets.shape))

        query_to_target = torch.bmm(repair_query, repair_targets.transpose(2, 1))
        query_to_target /= math.sqrt(self.config.hidden_size)

        repair_logits = query_to_target.squeeze(1)

        if input_mask is not None: 
            # There are two options:
            # 1) An error mask exists than repair_logits = [Batch, Seq Len]
            # 2) An error mask do not exist than repair_logits = [Batch, Seq Len, Seq Len]
            # For the second case, we have to adapt the logit mask

            if len(input_mask.shape) == len(repair_logits.shape) - 1: # Second case
                input_mask = input_mask.unsqueeze(1)

            repair_logits = _mask(repair_logits, input_mask)

        if hasattr(self, "decoder"):
            decoder_logits = self.decoder(input_embeds.squeeze(1))
            repair_logits = torch.cat([decoder_logits, repair_logits], dim = -1)

        return repair_logits


# Similar to Allamanis et al. 21 --------------------------------

class MLP(nn.Module):

    def __init__(self, config, activation):
        super().__init__()
        hidden_size = config.hidden_size
        self._l1 = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size, bias = True)
        self._l2 = nn.Linear(in_features=hidden_size, out_features=1, bias=False)
        self._activation = activation

    def forward(self, x):
        logits = self._l1(x)
        logits = self._activation(logits)
        return self._l2(logits).squeeze(-1)


class MLPLocalizationModule(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = self.config.hidden_size

        self._loc_query = nn.Linear(hidden_size, hidden_size, bias=True)
        self._mlp = MLP(config, nn.Sigmoid())

    def forward(self, input_embeds, target_embeds = None, input_mask = None):
        
        # Compute localization query --------------------------------
        query_embeds = self._loc_query(input_embeds)
        query_embeds = query_embeds.max(dim = 1, keepdim = True).values
        query_embeds = query_embeds.expand_as(input_embeds)

        # Compute localization --------------------------------
        loc_logits = torch.cat([input_embeds, query_embeds], dim = 2)
        output = self._mlp(loc_logits)

        if input_mask is not None: output = _mask(output, input_mask)

        return output


class MLPRepairModule(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self._mlp = MLP(config, nn.ReLU())

        if self.config.target_vocabulary_size > 0:
            self.decoder = nn.Linear(config.hidden_size, config.target_vocabulary_size, bias = False)
            self.apply(init_weights)


    def forward(self, input_embeds, target_embeds, input_mask = None, input_encoding = None):

        if len(input_embeds.shape) < 3:
            input_embeds = input_embeds.unsqueeze(1)

        assert len(input_embeds.shape) == 3 and len(target_embeds.shape) == 3, "Input or target embedding is not in batch!"

        # Token repair --------------------------------
        repair_query  = input_embeds.unsqueeze(2)
        repair_target = target_embeds.unsqueeze(1).repeat(1, repair_query.shape[1], 1, 1)
        repair_query = repair_query.expand_as(repair_target)
        repair_logits = torch.cat([repair_query, repair_target], dim = 3)
        repair_logits = self._mlp(repair_logits)

        if input_mask is not None: 
            # There are two options:
            # 1) An error mask exists than repair_logits = [Batch, Seq Len]
            # 2) An error mask do not exist than repair_logits = [Batch, Seq Len, Seq Len]
            # For the second case, we have to adapt the logit mask

            if len(input_mask.shape) == len(repair_logits.shape) - 1: # Second case
                input_mask = input_mask.unsqueeze(1)

            repair_logits = _mask(repair_logits, input_mask)

        if hasattr(self, "decoder"):
            decoder_weight = self.decoder.weight
            decoder_weight = decoder_weight.unsqueeze(0).unsqueeze(0)\
                                .repeat(input_embeds.shape[0], input_embeds.shape[1], 1, 1)
            repair_query  = input_embeds.unsqueeze(2).expand_as(decoder_weight)
            
            decoder_logits = torch.cat([repair_query, decoder_weight], dim = 3)
            decoder_logits = self._mlp(decoder_logits)
            repair_logits = torch.cat([decoder_logits, repair_logits], dim = 2)

        return repair_logits.squeeze(1)