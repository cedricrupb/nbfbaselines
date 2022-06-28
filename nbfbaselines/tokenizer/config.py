from dataclasses import dataclass



@dataclass
class TokenizerConfiguration:
    vocabulary_path : str

    add_special_tokens : bool = True

    # BPE config
    bpe_encoding : bool = True
    bpe_alignment : str = "scatter" # Options: scatter, sequence
    bpe_cutoff : int = 10    # Only if bpe_alignment = scatter
    bpe_eot    : str = "#"

    # Pre tokenization
    lang : str = "python"
    ignore_syntax_errors : bool = True
    str_cutoff : int = -1

    target_vocab_path : str = None

    # Graph preprocessing
    construct_graph  : bool                 = False