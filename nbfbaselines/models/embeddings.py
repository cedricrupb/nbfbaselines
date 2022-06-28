import torch
from torch import nn

# API method -------------------------------

def embedding_from_config(config):
    
    if config.embedding_type == "none":
        return nn.Embedding(config.vocab_size, config.embedding_size, padding_idx = config.pad_token_id)

    return SubtokenEmbeddings(config)


# Modules --------------------------------------------------------

class SubtokenEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocabulary_size,
                                                config.embedding_size,
                                                padding_idx=config.pad_token_id)

    def forward(self, subtokens):
        assert subtokens.max() < self.config.vocabulary_size, "Found token %d not in vocabulary" % subtokens.max()
        subtoken_embeddings = self.word_embeddings(subtokens)

        if self.config.embedding_type == "max_pool":
            return subtoken_embeddings.max(dim = -2).values

        with torch.no_grad():
            token_weights = subtokens.ne(0).float()
            token_norm    = token_weights.sum(dim=-1).unsqueeze(-1)
            token_norm    = token_norm.expand_as(token_weights)
            token_weights /= (token_norm + 1e-9)

        subtoken_embeddings *= token_weights.unsqueeze(-1)
        subtoken_embeddings = subtoken_embeddings.sum(dim=-2)

        return subtoken_embeddings