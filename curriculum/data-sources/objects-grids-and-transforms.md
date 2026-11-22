# Python objects, grids and transforms

Large sets of numbers, in the notation itself. Python reads the files, in its own
cells, with any library. The language reads the Python objects by name:
`\texttt{prices}` is the object called `prices` in the Python scope the workbook was
given. Attributes and keys follow, as in `\texttt{table.x}` and `\texttt{data["AAPL"]}`.
What an object becomes depends on its type:

- a number is a scalar;
- an array or a DataFrame is data, labelled where it has labels;
- a function is callable;
- a SymPy expression is an expression.

Grids are written as mathematics writes them, and transforms by name or as their sums.
A case's `python-scope` block is that scope: run first, in this folder, with the tables
in `tables/`.

## A column of a table

**Level 1** · `data.objects.column` · standard

A NumPy vector from the Python scope becomes a vector over its entries.

```python-scope
import numpy as np

signal = np.genfromtxt("tables/signal.csv", delimiter=",", names=True)["x"]
```

```latex
x_t := \texttt{signal}
\sum_t x_t
```

```python
table = np.genfromtxt("tables/signal.csv", delimiter=",", names=True)
expected = float(table["x"].sum())
```

## A whole DataFrame, labelled

**Level 1** · `data.objects.dataframe` · standard

A pandas DataFrame becomes a tensor. Its index labels the rows, the basis of `t`, and
its columns are the basis of `a`.

```python-scope
import pandas as pd

prices = pd.read_csv("tables/prices.csv", index_col="date")
```

```latex
P_{ta} := \texttt{prices}
\mathbb{E}_t[P_{ta}]
```

```python
import csv

with open("tables/prices.csv") as file:
    rows = list(csv.reader(file))
columns = list(zip(*[[float(v) for v in row[1:]] for row in rows[1:]]))
expected = [sum(column) / len(column) for column in columns]
```

## A key of an object

**Level 1** · `data.objects.key` · standard

Attributes and keys reach inside an object: one column of the DataFrame.

```python-scope
import pandas as pd

prices = pd.read_csv("tables/prices.csv", index_col="date")
```

```latex
x_t := \texttt{prices["NVDA"]}
\max_t x_t
```

```python
import csv

with open("tables/prices.csv") as file:
    rows = list(csv.reader(file))
expected = max(float(row[3]) for row in rows[1:])
```

## A constant from Python

**Level 2** · `data.objects.constants` · standard

The Schwarzschild radius of the Sun, from SciPy's constants and a mass defined in Python.

```python-scope
from scipy import constants

G, c = constants.G, constants.c
M_sun = 1.98847e30  # kg
```

```latex
r_s := \frac{2 \texttt{G} \texttt{M_sun}}{\texttt{c}^2}
r_s
```

```python
expected = 2 * 6.67430e-11 * 1.98847e30 / 299792458.0**2
tolerance = 1e-12
```

## A Python function, called

**Level 3** · `data.objects.function` · Harris (1978)

A window defined as a Python function, applied to a grid written in the notation.

```python-scope
import numpy as np


def hann(n, N):
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))
```

```latex
N := 8
w_n := \texttt{hann}(n, N), \quad n = 0, \ldots, N-1
\sum_n w_n
```

```python
n = np.arange(8)
expected = float(np.sum(0.5 - 0.5 * np.cos(2 * np.pi * n / 7)))
```

## A SymPy expression, differentiated

**Level 2** · `data.objects.sympy` · standard

An expression built in Python with SymPy is an expression in the language.

```python-scope
import sympy

x = sympy.Symbol("x")
f = sympy.sin(x) ** 2
```

```latex
\frac{d}{dx} \texttt{f}
```

```python
x = symbols("x")
expected = 2 * sin(x) * cos(x)
```

## The transform by name

**Level 2** · `data.fft.named` · Cooley and Tukey (1965)

The discrete Fourier transform of sampled data, as `fft`.

```python-scope
import numpy as np

signal = np.genfromtxt("tables/signal.csv", delimiter=",", names=True)["x"]
```

```latex
x_t := \texttt{signal}
\operatorname{fft}(x_t)
```

```python
table = np.genfromtxt("tables/signal.csv", delimiter=",", names=True)
expected = list(np.fft.fft(table["x"]))
tolerance = 1e-9
```

## A uniform grid

**Level 2** · `data.grids.uniform` · standard

A grid defined as mathematics defines it: a spacing and a range of the index.

```latex
N := 8
\Delta x := \frac{1}{N}
x_n := n\,\Delta x, \quad n = 0, \ldots, N-1
\sum_n x_n
```

```python
expected = Rational(7, 2)
```

## The transform written as its sum

**Level 3** · `data.fft.dft-sum` · Bracewell (2000), ch. 9

The discrete Fourier transform written out, with the range of its new index: the
language should compute it as a fast transform.

```python-scope
import numpy as np

signal = np.genfromtxt("tables/signal.csv", delimiter=",", names=True)["x"]
```

```latex
x_t := \texttt{signal}
N := 16
X_k := \sum_{t=0}^{N-1} x_t\, e^{-2\pi i k t/N}, \quad k = 0, \ldots, N-1
X_k
```

```python
table = np.genfromtxt("tables/signal.csv", delimiter=",", names=True)
expected = list(np.fft.fft(table["x"]))
tolerance = 1e-9
```

## The tone in a sampled signal

**Level 3** · `data.fft.sampled-tone` · standard

Sample a function on a grid, transform it, and find its frequency.

```latex
N := 64
t_n := \frac{n}{N}, \quad n = 0, \ldots, N-1
f_n := \sin(2\pi \cdot 5\, t_n)
\operatorname*{arg\,max}_k \left| \operatorname{fft}(f)_k \right|
```

```python
n = np.arange(64)
spectrum = np.abs(np.fft.fft(np.sin(2 * np.pi * 5 * n / 64)))
expected = int(np.argmax(spectrum))
```

## Two objects, aligned by label

**Level 3** · `data.objects.aligned` · standard

The weights of a portfolio are a pandas Series in another order. Contracting the prices
with them must pair each asset with its own weight, by name.

```python-scope
import pandas as pd

prices = pd.read_csv("tables/prices.csv", index_col="date")
weights = pd.read_csv("tables/weights.csv", index_col="asset")["weight"]
```

```latex
P_{ta} := \texttt{prices}
w^a := \texttt{weights}
V_t := P_{ta} w^a
V_t
```

```python
import csv

with open("tables/prices.csv") as file:
    rows = list(csv.reader(file))
with open("tables/weights.csv") as file:
    weights = {asset: float(weight) for asset, weight in list(csv.reader(file))[1:]}
assets = rows[0][1:]
expected = [sum(float(p) * weights[a] for p, a in zip(row[1:], assets)) for row in rows[1:]]
```
