"""
Test the code tokenizer: source_code <-> source_tokens

To allow end-to-end processing, we require that the tokenizer is bijective, i.e.:
    x = tokenizer.decode(tokenizer.tokenize(x))

We test the tokenizer for bijectivity.

For tokens, we ignore formatting and comments.
"""

from nbfbaselines.tokenizer import code_tokenizer 
from nbfbaselines.tokenizer import TokenizerConfiguration

from .utils import assert_tokenizer_bijective, assert_decoder_bijective

# Utils ------------------------------------------------

def new_tokenizer():
    config = TokenizerConfiguration(
        "", # No vocab is needed
        lang = "java",
        ignore_syntax_errors = True
    )

    return code_tokenizer.CodeTokenizer(config)


# Tests ------------------------------------------------

def test_tokenizer_pure_code_1():
    tokenizer = new_tokenizer()

    code = """
    
    public int f(int x, int y){
        return x + y;
    }

    """

    assert_tokenizer_bijective(tokenizer, {"input_text": code})


def test_tokenizer_code_marker_1():
    tokenizer = new_tokenizer()

    code = """public class Test {
    public int f(int x, int y){
        return x + y;
    }
}
    """

    assert_tokenizer_bijective(tokenizer, {"input_text": code, 
                            "test_marker": [(0, 0, 0, 6)]})


# Invert tokens test ----------------------------------------------------------------

def test_decoder_pure_code_1():
    tokenizer = new_tokenizer()

    tokens = [
        "public", "class", "Test", "{",
            "public", "int", "f", "(", "int", "x", ",", "int", "y", ")", "{",
                "return", "x", "+", "y", ";",
            "}",
        "}"
    ]

    assert_decoder_bijective(tokenizer, {"input_tokens": tokens})


def test_decoder_code_marker_1():
    tokenizer = new_tokenizer()

    tokens = [
        "public", "class", "Test", "{",
            "public", "int", "f", "(", "int", "x", ",", "int", "y", ")", "{",
                "return", "x", "+", "y", ";",
            "}",
        "}"
    ]

    mask = [0] * len(tokens)
    mask[9] = 1

    assert_decoder_bijective(tokenizer, {"input_tokens": tokens, "location_mask": mask})


