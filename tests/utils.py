from nbfbaselines.tokenizer.code_tokenizer import graph_to_tokens

def _compress_code(code):
    print(code)
    return "".join(code.split())

def is_code_equal(code, token_return):

    assert type(code) == type(token_return), "The result of tokenization result has to be the same type as the input: %s != %s" % (type(code), type(token_return))

    if isinstance(code, str):
        assert _compress_code(code) == _compress_code(token_return), "Code returned by tokenizer is not the same as input code: \n %s\n-----\n %s" % (code, token_return)
        return

    if isinstance(code, dict):
        for key, value in code.items():
            assert key in token_return, "Tokenizer does not return key: %s" % key
            is_code_equal(value, token_return[key])
        return

    assert code == token_return, "Object returned by tokenizer is not equal to input object:\n %s\n-----\n %s" % (code, token_return)


def assert_tokenizer_bijective(tokenizer, code, decoder_args = {}):
    token_output = tokenizer.tokenize(**code)
    token_output.update(decoder_args)
    token_return = tokenizer.decode(**token_output)

    is_code_equal(code, token_return)


def assert_decoder_bijective(tokenizer, code, ignore_graph = False):
    token_output = tokenizer.decode(**code)
    token_return = tokenizer.tokenize(**token_output)

    if ignore_graph:
        token_input, kwargs = graph_to_tokens(**token_return)
        token_return["input_tokens"] = token_input
        token_return.update(kwargs)

    is_code_equal(code, token_return)