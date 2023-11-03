# NBFBaselines
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cedricrupb/nbfbaselines/blob/main/demo.ipynb) 
[[**PAPER**](http://arxiv.org/abs/2207.00301)]
> Neural baselines for finding and fixing single token bugs in Python

Software bugs can interrupt our workflow and finding them can often be very time-consuming. Especially smaller bugs are often hard to spot since they affect only a few lines of code out of thousands. For example, recognizing that a variable was not renamed after refactoring or that a loop will run in an out-of-bounds error because of the comparison operator is often difficult.

NBFBaselines collects baseline algorithm for learning to localize and repair software bugs. These algorihtms are designed to assist a developer in finding and fixing simple software bugs. They learn from millionth of bugs and their fixes to detect and repair new bugs in unseen code. Currently, bug finders often support a wide range of simple bugs such as:
* **Variable Misuses:** The developer uses a variable although another was meant
```python
def compare(x1, x2):
    s1 = str(x1)
    s2 = str(x1) # Bug: x2 instead of x1
    return s1 == s2
```
* **Binary Operator Bugs:** The wrong binary operator was used
```python
def add_one(L):
    i = 0
    while i <= len(L): 
        L[i] = L[i] + 1
        i += 1
```
* **Unary Operator Bugs:** A unary operator was used mistakenly or was forgotten
```python
if namespace: # Bug: not namespace instead of namespace
    self.namespacesFilter = [ "prymatex", "user" ] 
else:
    self.namespacesFilter = namespace.split()
```
* **Wrong Literal Bugs:** The wrong literal was used
```python
def add_one(L):
    i = 0
    while i < len(L): 
        L[i] = L[i] + 1
        i += 2 # Bug: 1 instead of 2
```

# RealiT: Training on mutants and real bugs
RealiT is a neural bug localization and repair model trained on mutants and real bug fixes obtained from open source Python projects. Due to this combined training RealiT achieves a high performance in the localization and repair of real bugs. RealiT currently support the localization and repair of the four types of single statement bugs discussed before. You can try RealiT yourself by following our quick start guide.

## Quick start
You can try `RealiT` with only a few lines of code:
```python
from nbfbaselines import NBFModel

realit_model = NBFModel.from_pretrained("realit")

realit_model("""

def f(x, y):
    return x + x

""")

# Output:
# [{'text': 'def f ( x , y ) :\n    return x + y\n    ',
#  'before': 'x', 'after': 'y',
#  'prob': 0.981337789216316}]

```
`RealiT` is mostly trained on function implementation written in Python 3. Therefore, it will work best if provided with a complete function implementation. It will likely also work on partial code snippets (e.g. partial function implementation or statement outside of a function).

You can also test `RealiT` in your browser without cloning this project. For this, open the following [Colab link](https://colab.research.google.com/github/cedricrupb/nbfbaselines/blob/main/demo.ipynb).


## Models

We trained and publish several baseline models for neural bug detection. All models were evaluated on [PyPIBugs](https://github.com/microsoft/neurips21-self-supervised-bug-detection-and-repair) and on the test portion of [ETH Py150k](https://www.sri.inf.ethz.ch/py150). We used PyPIBugs to estimate the localization (Loc), repair (Rep) and Joint localization and repair performance (Joint). ETH Py150k was used to estimate the false positive rate (FPR).

**RealiT models:**
| model_id | FPR | Joint | Loc | Rep | description |
|----------|-----|-------|-----|-----|-------------|
| `realit5x-noreal` | 25.2 | 21.4 | 25.8 | 59.9 | RealiT without fine-tuning on real bugs and 5x mutants injected during pre-training |
| `realit-noreal` | 30.0 | 24.9 | 30.4 | 65.6 | RealiT without fine-tuning on real bugs |
| `realit` | 22.0 | 36.7 | 41.9 | 73.5 | Base RealiT model (Transformer) trained on 100x mutants and fine-tuned on real bugs. |
| `csnrealit` | 16.5 | 38.7 | 42.7 | 76.7 | RealiT further pre-trained on the Python corpus of CodeSearchNet. |

You can try these models easily yourself:
```python
from nbfbaselines import NBFModel

nbf_model = NBFModel.from_pretrained([model_id])

```

## Evaluation
The models were evaluated with the following Python script:
```bash
$ python run_eval.py [model_id] [test_file] [output_file] [options...]
```
For `model_id`, you can use any of the model ids specified before.
The format of the test file is described below. The script accepts further options:
| option | description | default |
|--------|-------------|---------|
| `--topk` | Activates beam search over the topk most likely repairs | 1 |
| `--cpu` | The script tries to run the code on an Nvidia GPU if availale. This forces the script to run on CPU. | False |
| `--max_length` | The maximal number of tokens to be accepted by the model. If an example has more token, we assume that no bug is detected. Tune this if the model exceeds the available memory.| 5000 |
| `--ignore_error` | The script stops if the evaluation runs into an error (e.g. because the code contains a syntax error). This ignores the error and continues the evaluation. | False |

**Test file format:**
As a test
file, we expect a file in json lines format where each JSON object has the following
format:
```json
{
    "input_text": [buggy_code_or_correct_code],
    "error_marker": [sl, sc, el, ec],
    "repair": [repair_text]
}
```
Note that `input_text` is required. The fields `error_marker` and `repair` are required if the code contains a bug. `error_marker` specifies the start line (sl), start char in that line (sc), end line (el) and the end char in that line (ec). The `repair` can be any string. The JSON object can contain further meta data. An example for a test_file is given in `data/real_bugs_dummy.jsonl`.

## Project info

Cedric Richter - [@cedrichter](https://twitter.com/cedrichter) - cedric.richter@uol.de

Distributed under the MIT license. See `LICENSE` for more information.

This project is currently incomplete and mainly designed to showcase `RealiT`. However, we plan to release the full code base including further pre-trained models, training script and evaluation code.

Feel free to open an issue if anything unexpected happens.
