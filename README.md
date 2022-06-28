# NBFBaselines
> Neural baseline for finding and fixing single token bugs in Python

Software bugs are nasty. They interrupt our workflow and finding software bugs can often be very time-consuming. Especially smaller bugs are often hard to spot since they affect only a few lines of code out of thousands. For example, recognizing that you forgot to rename a variable after refactoring or that your loop will run in an out-of-bounds error because of the comparison operator is often difficult.

NBFBaselines collects baseline algorithm for learning to localize and repair software bugs. These algorihtms are designed to assist you in finding and fixing simple software bugs. They learn from millionth of bugs and their fixes to detect and repair new bugs in unseen code. Currently, bug finders often support a wide range of simple bugs such as:
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

## Quick start: RealiT
Want to use a NBFBaselines for debugging your Python code? You can try `RealiT` with only a few lines of code:
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

