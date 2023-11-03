import json
import random
import torch

from .util import Data

class TextTransform:
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _preprocess_not(self, output):
        """Handles unary operator based on not"""
        if output["repair_token"].startswith("not "):
            # x -> not x
            output["repair_token"] = "not"
            location_mask = output["location_mask"]
            error_location = min(i for i, b in enumerate(location_mask) if b == 1)
            location_mask  = [0] * len(location_mask)
            location_mask[error_location] = 1
            output["location_mask"] = location_mask
            return output

        location_mask = output["location_mask"]
        error_tokens = [(i, t) for i, t in enumerate(output["input_tokens"]) if location_mask[i] == 1]
        if len(error_tokens) <= 1: return output

        if len(error_tokens) > 2:
            return self._preprocess_binop(output)
            #raise ValueError("Something went wrong: %s" % error_tokens)

        if error_tokens[0][1] == "not":
            other_pos, _ = error_tokens[1]
            location_mask[other_pos] = 0
            output["repair_token"] = ""
            return output

        if error_tokens[-1][1] == "not":
            # is not -> is
            other_pos, _ = error_tokens[0]
            location_mask[other_pos] = 0
            output["repair_token"] = ""
            return output

        print(output)
        return {}

        raise ValueError("Not expected to ever reach here: %s, %s" % (error_tokens, output["repair_token"]))

    def _preprocess_binop(self, output):
        location_mask = output["location_mask"]
        error_tokens = [(i, t) for i, t in enumerate(output["input_tokens"]) if location_mask[i] == 1]
        repair_tokens = output["repair_token"].split()

        if len(error_tokens) != len(repair_tokens):
            return self._preprocess_isin(output)

        binop  = []
        repair = []

        for p, (ix, token) in enumerate(error_tokens):
            if repair_tokens[p] != token:
                binop.append((ix, token))
                repair.append(repair_tokens[p])
                continue

            location_mask[ix] = 0
        
        if len(repair) == 0: return {}
        
        output["repair_token"] = repair[0]        
        return output


    def _preprocess_isin(self, output):
        location_mask = output["location_mask"]
        error_tokens = [(i, t) for i, t in enumerate(output["input_tokens"]) if location_mask[i] == 1]

        binop = []
        for p, (ix, token) in enumerate(error_tokens):
            if token in ["is", "in"]:
                binop.append((ix, token))
                continue

            if token == "not":
                if error_tokens[p - 1][1] == "is" or error_tokens[p + 1][1] == "in":
                    binop.append((ix, token))
                    continue
            location_mask[ix] = 0
        
        binop_tokens = [b[1] for b in binop]

        if binop_tokens == ["is"]:
            output["repair_token"] = "is not"
        elif binop_tokens == ["in"]:
            output["repair_token"] = "not in"
        elif binop_tokens == ["is", "not"]:
            location_mask[binop[0][0]] = 0
            output["repair_token"] = ""
        elif binop_tokens == ["not", "in"]:
            location_mask[binop[1][0]] = 0
            output["repair_token"] = ""
        
        return output

    def __call__(self, code_dict):
        assert "input_text" in code_dict

        try:
            output = self.tokenizer(
                **code_dict, 
                format = "dict"
            )
        except Exception:
            return {}

        if "location_marker" not in code_dict: return output

        if code_dict["location_marker"][0] == (0, 0, 0, 0):
            # Program is correct
            output["location_mask"][0] = 1

        output = self._preprocess_not(output)
        if len(output) == 0: return output
        
        try:
            assert sum(output["location_mask"]) == 1, "Expected at least one position to be marked as incorrect but got none."
        except AssertionError:
            return {} # TODO: Fix current problem are f-strings

        return output