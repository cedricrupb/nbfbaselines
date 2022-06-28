import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel, BertConfig

from dataclasses import dataclass

from .utils import init_weights


@dataclass
class TransformerConfig:
    embedding_size : int = 512
    hidden_size : int = 512
    intermediate_size : int = 2048
    num_layers : int = 6
    num_heads : int = 8
    max_length : int = 512
    dropout : float = 0.1
    pad_token_id : int = 0
    sinoid             : bool = True
    relative_attention : bool = False

    def copy_from_nbfconfig(self, nbf_config):
        self.embedding_size = nbf_config.embedding_size
        self.max_length     = nbf_config.max_sequence_length
        self.hidden_size    = nbf_config.hidden_size
        self.pad_token_id   = nbf_config.pad_token_id

# Predefined model size --------------------------------

def small_model_config():
    # Values equivalent to BERT small
    return TransformerConfig(
        embedding_size = 512,
        hidden_size = 512,
        intermediate_size = 2048,
        num_layers = 4,
        num_heads = 8
    )

def medium_model_config():
    return TransformerConfig()


def config_from_model_size(model_size):
    if model_size == "small":
        return small_model_config()
    
    if model_size == "medium":
        return medium_model_config()

    raise ValueError("Unknown model size: %s" % model_size)


# NBF model -----------------------------------------------------

class TransformerNBF(nn.Module):
    
    def __init__(self, config, embeddings, prediction_head):
        super().__init__()
        self.config = config
        self.encoder = TransformerEncoder(config, embeddings)
        self.prediction_head = prediction_head

        self.init_weights()
        
    def init_weights(self):
        self.prediction_head.apply(init_weights)

    def increase_token_length(self, max_length):
        if self.config.relative_attention:
            raise AttributeError("Issue #17026: Relative atttention do not support sequences larger than max length tokens.")

        try:
            self.encoder.encoder.embeddings.position_embeddings.expand_length(max_length)
            position_ids = torch.arange(max_length).expand((1, -1))
            self.encoder.encoder.embeddings.position_ids = position_ids
            self.encoder.encoder.embeddings.token_type_ids = torch.zeros(position_ids.size(), dtype=torch.long)
        except AttributeError:
            raise ValueError("Transformer used a learned embedding and can therefore not be adjusted")

    def forward(self, 
                input_ids,
                input_mask = None,
                position_ids = None,
                location_index = None,
                location_mask  = None,
                repair_mask  = None,
                repair_target = None):

        hidden, target_embed = self.encoder(input_ids, 
                                            input_mask, 
                                            position_ids)

        return self.prediction_head(
            hidden,
            target_embed,
            input_mask     = input_mask,
            location_index = location_index,
            location_mask  = location_mask,
            repair_mask    = repair_mask,
            repair_target = repair_target
        )


# Implementation ------------------------------------------------


def detect_to_hf_config(config):
    return BertConfig(
        vocab_size=1, #We delete the word embeddings (Never used)
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_layers,
        num_attention_heads=config.num_heads,
        max_position_embeddings=config.max_length,
        hidden_dropout_prob=config.dropout,
        attention_probs_dropout_prob=config.dropout,
        pad_token_id=config.pad_token_id,
        position_embedding_type = ("relative_key" 
                                    if config.relative_attention 
                                    else "absolute")
    )


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.embed_size = d_model
        self.expand_length(max_len)

    def expand_length(self, max_length):
        pe = torch.zeros(max_length, self.embed_size)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_size, 2).float() * (-math.log(10000.0) / self.embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.pe.requires_grad = False

    def forward(self, position_ids):
        return F.embedding(position_ids, self.pe)


class TransformerEncoder(nn.Module):

    def __init__(self, config, embeddings):
        super().__init__()

        self.config = config
        self.embeddings = embeddings

        if config.hidden_size != config.embedding_size:
            self.embedding_proj = nn.Linear(config.embedding_size, config.hidden_size)

        # Bert encoder
        hf_config = detect_to_hf_config(config)
        model = BertModel(hf_config, False)
        model.embeddings.word_embeddings = None

        if config.sinoid:
            encoding = PositionalEncoding(config.hidden_size)
            model.embeddings.position_embeddings = encoding

        self.encoder = model

        self.init_weights()

    def init_weights(self):
        self.embeddings.apply(init_weights)

        if hasattr(self, "embedding_proj"):
            self.embedding_proj.apply(init_weights)

    def get_embedding(self):
        return self.embeddings

    def forward(self, tokens,
                    attention_mask=None,
                    position_ids=None,
                    token_type_ids=None):

        subtoken = self.embeddings(tokens)

        if hasattr(self, "embedding_proj"):
            subtoken = self.embedding_proj(subtoken)

        token_encoding = self.encoder(
            inputs_embeds = subtoken,
            attention_mask = attention_mask,
            position_ids = position_ids,
            token_type_ids = token_type_ids
        )

        return token_encoding[0], subtoken

