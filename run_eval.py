import os
import json
import argparse
import copy
import torch
import time
import traceback

import code_diff as cd

from glob import glob
from tqdm import tqdm

from nbfbaselines import NBFModel
from nbfbaselines.data.transforms import TextTransform


def iterate_examples(eval_dir):
    if os.path.isfile(eval_dir): 
        files = [eval_dir]
    else:
        files = glob(os.path.join(eval_dir, "*.jsonl"))

    for file in files:
        with open(file, "r") as lines:
            for line in lines:
                yield json.loads(line)


def _rm_additional_indent(code):
    lines = code.splitlines(True)
    white_space_offset = len(lines[0]) - len(lines[0].lstrip())
    return "".join([l[white_space_offset:] for l in lines])


def compute_difference(prediction):

    input_code   = _rm_additional_indent(prediction["source_code"])
    changed_code = prediction["repair_cand"] 

    try:
        diff = cd.difference(input_code, changed_code, lang = "python")
    except Exception:
        return input_code
   
    source_line = diff.source_ast.position[0][0]
    target_line = diff.target_ast.position[0][0]
    
    input_code_lines = input_code.splitlines(True)
    changed_code_lines = changed_code.splitlines(True)

    diff = []
    for i, line in enumerate(input_code_lines):
        if i == source_line:
            diff.append("-"+line)
            diff.append("+"+changed_code_lines[target_line])
        else:
            diff.append(line)

    return "".join(diff)


# Load model ----------------------------------------------------------------

def load_model(args, checkpoint_dir):
    nbf_model = NBFModel.from_pretrained(checkpoint_dir)
    nbf_model.model.eval()
    nbf_model.increase_accepted_length(args.max_length)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    nbf_model.to(device)
    return nbf_model


# Preprocess -----------------------------------------------------------

def prepare_input(tokenizer, example):
    source_code   = example["input_text"]
    error_loc     = example["error_marker"]
    repair_target = example["repair"]

    input_dict = TextTransform(tokenizer)({
        "input_text": source_code, 
        "location_marker": [error_loc],
        "repair_token": repair_target
    })

    assert len(input_dict) != 0, "Expected a preprocessing result but got None."

    error_mask = input_dict["location_mask"]
    repair_target = input_dict["repair_token"]

    del input_dict["location_mask"]
    del input_dict["repair_token"]
    del input_dict["format"]

    assert sum(error_mask) == 1, "Cannot predict an error location for more than one location"
    
    return input_dict, error_mask, repair_target

# Prediction ----------------------------------------------------------------

def predict_loc_repair(model, 
                        input_dict, 
                        location_mask = None, 
                        topk = 1, 
                        search_for = None):

    input_dict = copy.deepcopy(input_dict)

    predictions = model(
        **input_dict,
        location_mask = location_mask,
        pre_tokenized = True,
        topk = topk,
        return_tokens = True,
    )

    search_result = None
    if search_for:
        found = False
        for pos, prediction in enumerate(predictions):
            error_loc   = prediction["token_error_loc"] - 1
            repair      = prediction["after"]
            if error_loc == search_for[0] and _is_repair_correct(repair, search_for[1]):
                search_result = (pos, error_loc, repair, prediction["prob"])
                found = True
        if not found: search_result = (topk+1,) + search_for + (0.0,)

    for prediction in predictions:
        if  (prediction["before"] == "[CLS]"
                or prediction["before"] != prediction["after"]):
            break

    if search_result: prediction["search_result"] = search_result

    return prediction


# Evaluation ----------------------------------------------------------------

def classify_bug(example):
    bug_type = example["bug_type"]
    repair   = example["repair"]

    if bug_type == "binop":
        if repair in {"+","-", "*", "**", "/", "//", "%", "@", "<<", ">>", "|", "&", "^"}: return "binary_op"
        if repair in {"<", "<=", ">=", ">", "==", "!="}  : return "comp_op"
        if "=" in repair: return "assign_op"
        if repair in {"and", "or"}: return "boolean_op"
        if any(" %s " % op in repair for op in ["in", "not in", "is", "is not"]): return "comp_op"

        return "comp_op"

    if bug_type == "not": return "boolean_op"
    if bug_type in ["litbool", "numeric"]: return "literal"

    return example["bug_type"]


def _is_repair_correct(repair_prediction, repair_target):

    if repair_target == "not":
        return repair_prediction.startswith("not")

    return repair_prediction == repair_target


def _eval_loc_repair_accuracy(model, 
                                input_dict, 
                                error_mask, 
                                repair, 
                                topk = 1):

    prediction = predict_loc_repair(model, 
                                    input_dict, 
                                    topk = topk)

    error_loc   = prediction["token_error_loc"] - 1
    correct_loc = error_mask[error_loc] == 1
    correct_repair = False

    if "allowed_repairs" in input_dict:
        # Test allowed repairs
        allowed_repairs = input_dict["allowed_repairs"]
        allowed_locs    = set([r[0] - 1 for r in allowed_repairs])

        if max([error_mask[r] for r in allowed_locs], default = 0) == 0:
            error_index = min(i for i, e in enumerate(error_mask) if e == 1)
            print("WARNING: Cannot repair '%s' as the location is blocked by mask." % input_dict["input_tokens"][error_index])
            print("CONTEXT:", input_dict["input_tokens"][error_index - 5: error_index + 5])
            print("ALLOWED:", [input_dict["input_tokens"][i - 1] for i in sorted(allowed_locs)])

    if correct_loc: correct_repair = _is_repair_correct(prediction["after"], repair)

    return correct_loc, correct_repair, prediction


def _eval_repair_only(model, 
                        input_dict, 
                        error_mask, 
                        repair, 
                        topk = 1):

    prediction = predict_loc_repair(model, 
                                    input_dict,
                                    location_mask = error_mask, 
                                    topk = topk)

    return _is_repair_correct(prediction["after"], repair)


def eval_model(args, model, example, force_cpu = False):

    # Preprocess
    input_dict, error_mask, repair_target = prepare_input(
        model.tokenizer.code_tokenizer,
        example
    )

    if len(input_dict["input_tokens"]) > model.max_length - 2:
        print("Warning: Example has too many tokens. Assumes that the model fails to find the bug.")
        return {
            "bug_type"   : classify_bug(example),
            "source_code": example["input_text"],
            "repair_cand": example["input_text"],
            "prob"       : 1.0,
            "localized"  : False,
            "repaired"   : False,
            "loc_repair" : False
        }

    current_device = None
    if force_cpu:
        print(f'Example of {len(input_dict["input_tokens"])} tokens moved to CPU since GPU could not process the input.')
        current_device = model.device
        cpu_device     = torch.device("cpu")
        if current_device != cpu_device:
            model.to(cpu_device)
        else:
            current_device = None
    
    # Loc & Repair
    correct_loc, correct_loc_repair, repair_cand = _eval_loc_repair_accuracy(
        model,
        input_dict,
        error_mask,
        repair_target,
        topk = args.topk
    )

    # Repair only
    correct_repair = _eval_repair_only(
        model, 
        input_dict,
        error_mask,
        repair_target,
        topk = args.topk
    )

    if current_device:
        model.to(current_device)

    return {
        "bug_type"   : classify_bug(example),
        "source_code": example["input_text"],
        "repair_cand": repair_cand["text"],
        "prob"       : repair_cand["prob"],
        "localized"  : correct_loc,
        "repaired"   : correct_repair,
        "loc_repair" : correct_loc_repair
    }

# Exceptions ----------------------------------------------------------------

def handle_exception(exception, args, detector, example):
    
    if exception.__class__.__name__ == "RuntimeError":
        try:
            return eval_model(args, detector, example, force_cpu = True)
        except Exception as e:
            exception = e
        
    if args.ignore_error:
        print("Warning: Unknown error. Handle as if the model could not perform a prediction.")
        return {
            "bug_type"   : classify_bug(example),
            "source_code": example["input_text"],
            "repair_cand": example["input_text"],
            "prob"       : 1.0,
            "localized"  : False,
            "repaired"   : False,
            "loc_repair" : False
        }

    print("################ %s #####################" % str(exception.__class__.__name__))
    print("Model type: %s" % str(detector))

    project = example["project_url"].replace("https://:@github.com/", "").replace(".git", "").replace("/", "_")

    print("Run into exception for example: %s [%s]" % (project, example["commit_sha"]))
    example_path = "error_%s_%s.json" % (project, example["commit_sha"])

    with open(example_path, "w") as o:
        o.write(json.dumps(example))

    print("Reproduce error with running:")
    print(f"python run_eval.py {args.checkpoint_dir} {args.eval_dir} --topk {args.topk} --debug {example_path}")

    print("Exception:")
    traceback.print_exc()
    exit()


# Eval helper --------------------------------

class Statistics:

    def __init__(self):
        self._subcategories = {}

    def _get_or_create_counter(self, key):
        if key not in self._subcategories:
            self._subcategories[key] = {
                "total"     : 0,
                "loc"       : 0,
                "repair"    : 0,
                "loc_repair": 0,
                "runtime"   : 0.0,
            }
        return self._subcategories[key]

    def _update(self, key, result):
        subcounter = self._get_or_create_counter(key)
        subcounter["total"] += 1
        if result["localized"]:  subcounter["loc"] += 1
        if result["repaired"]:   subcounter["repair"] += 1
        if result["loc_repair"]: subcounter["loc_repair"] += 1
        if "runtime" in result : subcounter["runtime"] += result["runtime"]

    def add_result(self, result):
        self._update("all", result)
        self._update(result["bug_type"], result)

    def loc_repair(self, key):
        counter = self._get_or_create_counter(key)
        return counter["loc_repair"] / counter["total"]

    def __repr__(self):
        lines = ["Bug type\tTotal\tLoc & Repair\tLocalization\tRepair"]

        for key in [k for k in self._subcategories if k != "all"] + ["all"]:
            counter = self._subcategories[key]

            sub_stats = [f"{counter[k] / counter['total']}% ({counter[k]})" for k in ["loc_repair", "loc", "repair"]]
            sub_stats.insert(0, counter["total"])
            sub_stats.insert(0, key)
            lines.append("\t".join([str(s) for s in sub_stats]))

        return "\n".join(lines)

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir")
    parser.add_argument("eval_dir")
    parser.add_argument("output_file")

    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--debug", type = str)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max_length", type=int, default=5000)
    parser.add_argument("--ignore_error", action="store_true")

    args = parser.parse_args()

    stats = Statistics()

    detector = load_model(args, args.checkpoint_dir)

    if args.debug:
        print("Debug: %s" % args.debug)
        with open(args.debug, "r") as i:
            example = json.load(i)
        print(eval_model(args, detector, example))
        exit()

    if os.path.isfile(args.eval_dir):
        file_paths = [args.eval_dir]
    else:
        file_paths = glob(os.path.join(args.eval_dir, "*.jsonl"))

    pbar = tqdm(iterate_examples(args.eval_dir), total = len(file_paths) * 100_000)

    with open(args.output_file, "w") as o:
        try:
            for example in pbar:
                if "error_marker" in example and len(example["error_marker"]) == 0: continue
                start_time = time.time()

                try:
                    result = eval_model(args, detector, example)
                except (Exception, torch.cuda.OutOfMemoryError) as e:
                    result = handle_exception(e, args, detector, example)
                
                runtime = time.time() - start_time
                
                diff = compute_difference(result)

                output = {"code_diff": diff, "runtime": runtime}
                output.update(result)
                o.write(json.dumps(output) + "\n")

                stats.add_result(output)

                pbar.set_description("Loc & Repair: %f" % (stats.loc_repair("all")))
        except KeyboardInterrupt:
            pass
    
    print(stats)


if __name__ == '__main__':
    main()