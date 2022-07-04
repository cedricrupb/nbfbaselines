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


## Project info

Cedric Richter - [@cedrichter](https://twitter.com/cedrichter) - cedric.richter@uol.de

Distributed under the MIT license. See `LICENSE` for more information.

This project is currently incomplete and mainly designed to showcase `RealiT`. However, we plan to release the full code base including further pre-trained models, training script and evaluation code.

Feel free to open an issue if anything unexpected happens.
