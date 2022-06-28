"""
Program code can have many forms dependent on the language.
Code tokenizer translate source code in a unified form as tokens.

Additional token level analyzes are possible. They will be carried through
"""
import re

import code_tokenize as ctok

# API method ----------------------------------------------------------------

def code_tokenizer_from_config(config):

    if config.construct_graph:
        raise NotImplementedError("Graph generation is currently unsupported")

    return CodeTokenizer(config)
    
    
# Available code tokenizer ----------------------------------------------------------------

class CodeTokenizer:

    def __init__(self, config):
        self.config = config

    # Helper methods --------------------------------------------------------------

    def _prepare_input(self, arguments):

        markers, kwargs = {}, {}

        for key, value in arguments.items():
            if key.endswith("_marker"): markers[key] = value
            else                      : kwargs[key]  = value

        return markers, kwargs

    def _preprocess_text(self, input_text):

        # Remove non-ascii
        return input_text.encode('ascii', 'replace').decode()

    # API methods ----------------------------------------------------------------

    def tokenize(self, input_text, **kwargs):
        markers, kwargs = self._prepare_input(kwargs)
        input_text = self._preprocess_text(input_text)

        tokens = ctok.tokenize(input_text, 
                                lang = self.config.lang,
                                syntax_error = "ignore" if self.config.ignore_syntax_errors else "raise")

        output = {"input_tokens": [t.text for t in tokens]}

        for marker_key, marker in markers.items():
            output[marker_key.replace("_marker", "_mask")] = align_tokens_with_marker(tokens, marker)

        output.update(kwargs)
        return output


    def decode(self, input_tokens, **kwargs):
        return decoder_by_lang(self.config.lang)(input_tokens, **kwargs)
    
    def __call__(self, input_text, **kwargs):
        return self.tokenize(input_text, **kwargs)


# Pretty print -----------------------------

def pretty_print(tokens, lang):
    try:
        return decoder_by_lang(lang)(tokens)["input_text"]
    except ValueError:
        return pprint_simple(tokens)

def pprint_simple(tokens):
    return " ".join(tokens)


# Token decoder ----------------------------------------------------------

def decoder_by_lang(lang):
    if lang == "java":   return JavaDecoder()
    if lang == "python": return PythonDecoder()

    raise ValueError("Language %s is not supported" % lang)


class TokenDecoder:

    def __init__(self):
        self._output_text = [""]

        self._masks  = None
        self._marker = {}

        self._text_line   = 0
        self._text_pos    = 0
        self._text_indent = 0
        self._token_pos   = 0

    def _prepare_args(self, kwargs):
        masks, out_args = {}, {}

        for key, value in kwargs.items():
            if key.endswith("_mask")  : masks[key]     = value
            else                      : out_args[key]  = value

        return masks, out_args

    def _handle_masks(self, start_line, start_pos, end_line, end_pos):
        current_token_ix = self._token_pos

        for key, value in self._masks.items():
            if value[current_token_ix] == 1:
                marker_key = key.replace("_mask", "_marker")
                self._marker[marker_key].append(
                    (start_line, start_pos, end_line, end_pos)
                )

    # API methods --------------------------------------------------------

    def inc_indent(self):
        self._text_indent += 1

    def dec_indent(self):
        self._text_indent -= 1

    def indent(self):
        indent_size = 4 * self._text_indent
        self._output_text.append(" "*indent_size)
        self._text_pos += indent_size

    def newline(self):
        self._output_text.append("\n")
        self._text_line += 1
        self._text_pos  = 0

    def newline_indent(self):
        self.newline()
        self.indent()

    def add_text(self, text):
        self._output_text.append(text)
        self._text_pos += len(text)

    def add_token_text(self, text):
        lines = text.splitlines()
        for i, line in enumerate(lines):
            self.add_text(line)
            if i < len(lines) - 1: 
                self.newline()

    def handle_token(self, token):
        raise NotImplementedError()

    def __call__(self, tokens, **kwargs):
        self._masks, kwargs = self._prepare_args(kwargs)

        self._marker = {k.replace("_mask", "_marker"): [] for k in self._masks.keys()}
        
        # Main loop
        for token in tokens: 
            output = self.handle_token(token)
            start_line, start_pos = self._text_line, self._text_pos
            self.add_token_text(output)
            end_line, end_pos     = self._text_line, self._text_pos

            self._handle_masks(start_line, start_pos, end_line, end_pos)
            self._token_pos += 1
            
        output_text = "".join(self._output_text)

        # Remove non-ascii from text
        output_text = output_text.encode('ascii', 'replace').decode()

        output = {"input_text": output_text}
        output.update(kwargs)
        output.update(self._marker)
        
        return output


# Java decoding ----------------------------------------------------------------
identifier_pattern = re.compile("(?:\b[_a-zA-Z]|\\B\\$)[_$a-zA-Z0-9]*")


class JavaDecoder(TokenDecoder):

    def is_closing(self, token):
        return token in [")", "]", "}"]

    def isop(self, token):
        return any(t in token for t in ["<", ">", "!", "=", "+", "-", "*", "/", "^"])

    def must_separate(self, t1, t2):
        if len(t1.strip()) == 0: return False

        if self.isop(t1) or self.isop(t2):
            return True

        return not identifier_pattern.match(t1) and not identifier_pattern.match(t2)

    def handle_blocks(self, token):
        if token == "{":
            self.inc_indent()
            return

        last_token = self._output_text[-1]
        if last_token == "{" and token != "}":
            self.newline_indent()
        
        if last_token == ";" and token != "}": 
            self.newline_indent()

        if token == "}": 
            self.dec_indent()
            self.newline_indent()

    def handle_token(self, token):
        output = self._output_text
        separate = self.must_separate(output[-1], token)
        separate = separate or token == "{"
        separate = separate or output[-1] == ","
        separate = separate or (self.is_closing(output[-1]) and token != ";")

        # Handle blocks
        self.handle_blocks(token)

        if self._text_pos + len(token) > 100:
            self.newline_indent()
            separate = False

        if output[-1].startswith("//"):
            self.newline_indent()
            separate = False

        if separate: self.add_text(" ")
        return token


# Python decoding -------------------------------------------------------

class PythonDecoder(TokenDecoder):

    def __init__(self):
        super().__init__()
        self._last_indent = False
        self._last_dedent = False

    def reset_indent(self):
        self._last_dedent = False
        self._last_indent = False

    def handle_token(self, token):
        if token == "#NEWLINE#": 
            self.newline_indent()
            self.reset_indent()
            return ""
        elif token == "#INDENT#": 
            self.inc_indent()
            if not self._last_indent: self.newline_indent()
            self._last_indent = True
            return ""
        elif token == "#DEDENT#":
            self.dec_indent()
            self._last_dedent = True
            return ""

        if self._last_dedent: self.newline_indent()

        self.reset_indent()

        if len(self._output_text[-1].strip()) > 0:
            self.add_text(" ")
        return token


# Helper ----------------------------------------------------------------

def align_tokens_with_marker(tokens, markers):

    def enclose_marker(marker_start_point, marker_end_point, start_point, end_point):
        return (
            marker_start_point[0] <= start_point[0]
            and marker_end_point[0] >= end_point[0] # Marker is enclosed in lines
            and (
                marker_start_point[0] != start_point[0]
                or marker_start_point[1] <= start_point[1]
            )
            and (
                marker_end_point[0] != end_point[0]
                or marker_end_point[1] >= end_point[1]
            )
        )

    def any_enclosed(start_point, end_point):
        return any(enclose_marker(m[:2], m[2:], start_point, end_point) for m in markers)
    
    def is_token_marked(token):
        if not hasattr(token, 'ast_node'): return False
        ast_node = token.ast_node
        if ast_node is None: return False 
        start_point, end_point = ast_node.start_point, ast_node.end_point
        return any_enclosed(start_point, end_point)

    return [1 if is_token_marked(token) else 0 for token in tokens]

# Graph helper --------------------------------

def _syntax_childs(node):
    for _, edge_type, next in node.successors():
        if edge_type == "child": yield next

def preorder_traversal(graph):
    root_node = graph.root_node
    
    stack = [root_node]
    while len(stack) > 0:
        current_node = stack.pop(-1)
        yield current_node
        children = list(_syntax_childs(current_node))
        stack.extend(children[::-1])


def graph_to_tokens(input_tokens, **kwargs):
    sep_token_ix = min((i for i, tok in enumerate(input_tokens) if tok == "[SEP]"), default = len(input_tokens))
    input_tokens = input_tokens[:sep_token_ix]

    trunc_kwargs = {}
    for key in filter(lambda key: "edge" not in key, kwargs.keys()):
        if key == "preorder_traversal": continue
        if key.endswith("_mask"):
            trunc_kwargs[key] = kwargs[key][:sep_token_ix]
        else:
            trunc_kwargs[key] = kwargs[key]

    return input_tokens, trunc_kwargs
