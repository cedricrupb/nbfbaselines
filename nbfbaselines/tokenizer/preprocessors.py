from .vocabulary import Vocabulary

# API method ----------------------------------------------------------------

def preprocessor_from_config(config, target_vocab = None):
    preprocessors = [
        IndexToMask()
    ]

    if config.add_special_tokens:
        preprocessors.append(AddSpecialTokens())
    
    if target_vocab is not None:
        preprocessors.append(TokenToMask(target_vocab))
    else:
        preprocessors.append(TokenToMask())

    if config.str_cutoff > 0:
        preprocessors.append(
            TokenPreprocessor(
                StringTruncate(config.str_cutoff)
            )
        )    

    return Preprocessor(preprocessors)


class Preprocessor:

    def __init__(self, preprocessors):
        self.preprocessors = preprocessors

    def __call__(self, source_input):
        source_input = {k: v for k, v in source_input.items() if v is not None}

        for preprocessor in self.preprocessors:
            source_input = preprocessor(source_input)
        
        return source_input

# Available preprocessors -------------------------------------------------


class TokenPreprocessor:

    def __init__(self, prepare_fn):
        self.prepare_fn = prepare_fn

    def __call__(self, source_input):
        tokens = source_input["input_tokens"]
        original_length = len(tokens)

        ptokens = self.prepare_fn(tokens)

        assert len(ptokens) == original_length, "Expected that the token seuqence length remains unchange, but %d != %d" % (original_length, len(ptokens))
        source_input["input_tokens"] = ptokens
        return source_input


class StringTruncate:

    def __init__(self, max_string_length):
        self.max_string_length = max_string_length

    def __call__(self, tokens):

        def truncate_string(token):
            
            if token.startswith("\"") and len(token) > self.max_string_length:
                return "\"\""
            
            if token.startswith("\'") and len(token) > self.max_string_length:
                return "\'\'"
            
            return token

        return list(map(truncate_string, tokens))


# Full preprocessors ----------------------------------------------------------------

class IndexToMask:

    def __call__(self, source_input):
        index_keys = [k for k in source_input.keys() if k.endswith("_index")]
        if len(index_keys) == 0: return source_input

        for index_key in index_keys:
            index = source_input[index_key]
            mask  = [0] * len(source_input["input_tokens"])
            mask[index] = 1

            source_input[index_key.replace("_index", "_mask")] = mask
            del source_input[index_key]
        
        return source_input


class TokenToMask:

    def __init__(self, vocabulary = None):
        self.vocabulary = vocabulary

    def __call__(self, source_input):
        index_keys = [k for k in source_input.keys() if k.endswith("_token") and type(source_input[k]) == str]
        if len(index_keys) == 0: return source_input

        tokens = source_input["input_tokens"]

        for index_key in index_keys:
            token = source_input[index_key]
            mask = [1 if t == token else 0 for t in tokens]
            source_input[index_key.replace("_token", "_mask")] = mask
            del source_input[index_key]
        
        if self.vocabulary is not None:
            if token in self.vocabulary:
                repair_ix = self.vocabulary.index(token) + 1
            else:
                repair_ix = 0

            source_input["repair_target"] = [repair_ix]

        return source_input


# Add special tokens --------------------------------

class AddSpecialTokens:

    def __init__(self, sos = "[CLS]", eos = "[EOS]"):
        self.sos = sos
        self.eos = eos

    def __call__(self, source_input):
        input_tokens = source_input["input_tokens"]
        if len(input_tokens) == 0: return source_input

        if input_tokens[0] != self.sos:
            input_tokens.insert(0, self.sos)

            for key in source_input.keys():
                if key.endswith("_mask"):
                    source_input[key].insert(0, 0)
        
        if input_tokens[-1] != self.eos:
            input_tokens.append(self.eos)
            for key in source_input.keys():
                if key.endswith("_mask"):
                    source_input[key].append(0)

        return source_input

