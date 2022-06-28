import torch

from .config import TokenizerConfiguration
from .code_tokenizer import code_tokenizer_from_config
from .preprocessors  import preprocessor_from_config

from .vocabulary import (
    Vocabulary,
    BPEEncoder
)

from .code_tokenizer import (
    CodeTokenizer
)

from .code_tokenizer import pretty_print


from abc import ABC, abstractmethod


def tokenizer_from_config(config):
    # Setup vocabulary 
    vocabulary = Vocabulary()
    vocabulary.load(config.vocabulary_path)
    vocabulary.close()

    if not config.bpe_encoding:
        return BasicTokenizer(config, vocabulary)

    bpe_encoder = BPEEncoder(vocabulary, eot = config.bpe_eot)

    if config.bpe_alignment == "scatter":
        return BPETokenizer(config, bpe_encoder)

    raise ValueError("Unknown BPE alignment: %s" % config.bpe_alignment)


# Tokenizer ----------------------------------------------------------------

class Tokenizer(ABC):

    @staticmethod
    def from_config(config):
        return tokenizer_from_config(config)

    def __init__(self, config):
        self.config         = config
        self.code_tokenizer = code_tokenizer_from_config(config)

        self.target_vocab = None
        if config.target_vocab_path is not None:
            self.target_vocab = Vocabulary()
            self.target_vocab.load(config.target_vocab_path)
            self.target_vocab.close()

        self.preprocess     = preprocessor_from_config(config, self.target_vocab)

        self.special_tokens = []

    # Abstract methdods -----------------------------------------------------

    @abstractmethod
    def get_vocab_size(self):
        pass

    @abstractmethod
    def convert_tokens_to_ids(self, tokens):
        pass

    @abstractmethod
    def convert_ids_to_tokens(self, token_ids):
        pass

    # API ----------------------------------------------------------

    def __call__(self, input_text = None, input_tokens = None, pre_tokenized = False, **kwargs):
        return self.tokenize(input_text, input_tokens, pre_tokenized, **kwargs)

    def tokenize(self, 
                    input_text    = None,
                    input_tokens  = None,
                    pre_tokenized = False, 
                    return_tokens = False, 
                    format = "dict",
                    **kwargs):

        if pre_tokenized:
            assert input_tokens is not None, "Please set input_tokens = [...] to enable processing pre tokenized code."
            source_input = {"input_tokens": input_tokens}
            source_input.update(kwargs)
        else:
            assert input_text is not None, "Please set input_text = ... to enable processing untokenized code."
            source_input = self.code_tokenizer(input_text, **kwargs)
       
        # Preprocess input
        source_input = self.preprocess(source_input)

        input_tokens = source_input["input_tokens"]
        input_labels = {k: v  for k, v in source_input.items() if k != "input_tokens"}

        input_ids = self.convert_tokens_to_ids(input_tokens)
        input_labels["input_mask"] = [1] * len(input_ids)
        
        result = format_output(format, input_ids, input_labels)

        if return_tokens:
            return result, input_tokens

        return result
       

    def decode(self, input_ids, return_tokens = False, ignore_special_tokens = False, **kwargs):    
        tokens = self.convert_ids_to_tokens(input_ids)

        if return_tokens:
            token_output = {"input_tokens": tokens}
            token_output.update(kwargs)
            return token_output

        if ignore_special_tokens:
            tokens = [t for t in tokens if t not in self.special_tokens]

        return self.code_tokenizer.decode(tokens, **kwargs)

    # Interface ------------------------------------------------------------

    def __repr__(self):
        name   = self.__class__.__name__
        config = self.config.__dict__

        return "%s(%s)" % (name, ", ".join(["%s=%s" % (k, str(v)) for k, v in config.items()]))


# Tokenizer implementation ----------------------------------------------------

class BasicTokenizer(Tokenizer):

    def __init__(self, config, vocabulary):
        super().__init__(config)
        self.vocabulary = vocabulary
        self.special_tokens = [t for t in vocabulary
                                 if t.startswith('[') and t.endswith(']')]

    def get_vocab_size(self):
        return len(self.vocabulary)
    
    def convert_tokens_to_ids(self, tokens):
        return [self.vocabulary.index(t) if t in self.vocabulary else 0
                 for t in tokens]
    
    def convert_ids_to_tokens(self, token_ids):
        return [self.vocabulary[t] for t in token_ids]

# BPE tokenizer --------------------------------

class BPETokenizer(Tokenizer):

    def __init__(self, config, bpe_encoder):
        super().__init__(config)
        self.bpe_encoder = bpe_encoder

        self.vocabulary = self.bpe_encoder.vocab
        
        special_tokens = filter(
            lambda x: x.startswith("[") and any(x.endswith(t) for t in ["]", "]#"]),
            self.vocabulary
        )
        special_tokens = map(
            lambda x: x[:-1] if x.endswith("#") else x,
            special_tokens
        )

        self.special_tokens = list(special_tokens)


    def get_vocab_size(self):
        return len(self.vocabulary)
    
    def convert_tokens_to_ids(self, tokens):
        cutoff = self.config.bpe_cutoff

        def prepare_token(token):
            bpe_encoding = self.bpe_encoder(token)
            return bpe_encoding[:cutoff] + (0,) * (cutoff - len(bpe_encoding))

        return list(map(prepare_token, tokens))
    
    def convert_ids_to_tokens(self, token_ids):
        vocabulary = self.vocabulary

        def decode_token(subtoken_ids):
            subtoken_ids = (x for x in subtoken_ids if x != 0)
            subtokens = map(lambda x: vocabulary[x], subtoken_ids)
            return "".join(subtokens)[:-1]

        return list(map(decode_token, token_ids))

    
# Formatter -------------------------------------------------------------------

def format_output(format, token_ids, token_labels):
    output = {"input_ids": token_ids}
    output.update(token_labels)

    if format == "dict":
        return output

    if format == "pt": 
        return {
            k: torch.LongTensor(v) for k, v in output.items()
        }

    raise ValueError("Unknown format: %s" % format)


# Helper ----------------------------------------------------------------------

def is_iterable(obj):
    if isinstance(obj, str): return False
    try:
        for _ in obj: return True
        return True
    except TypeError:
        return False