# Tables, grids and transforms

Large sets of numbers, written in the notation itself, not bound from Python: taken from a
table, laid out on a grid, or transformed with the FFT. These cases fix the notation the
language should read.

- A file named in `\texttt{...}` is a table, as LaTeX typesets a path.
- `[x]` takes a column.
- A tensor defined from a whole table takes its spaces from the table's header and row
  labels.

The tables are in `tables/`, beside this file.

## One column of a table

**Level 1** · `data.tables.column` · standard

A column of a CSV file becomes a vector over the table's rows.

```latex
x_t := \texttt{tables/signal.csv}[x]
\sum_t x_t
```

```python
table = np.genfromtxt("tables/signal.csv", delimiter=",", names=True)
expected = float(table["x"].sum())
```

## A whole table, labelled

**Level 1** · `data.tables.whole` · standard

A table whose first column labels its rows becomes a tensor: its rows are one space,
named by those labels, and its other columns another.

```latex
P_{ta} := \texttt{tables/prices.csv}
\mathbb{E}_t[P_{ta}]
```

```python
import csv

with open("tables/prices.csv") as file:
    rows = list(csv.reader(file))
columns = list(zip(*[[float(v) for v in row[1:]] for row in rows[1:]]))
expected = [sum(column) / len(column) for column in columns]
```

## A transform by name

**Level 2** · `data.fft.named` · Cooley and Tukey (1965)

The discrete Fourier transform of sampled data, as `fft`.

```latex
x_t := \texttt{tables/signal.csv}[x]
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

```latex
x_t := \texttt{tables/signal.csv}[x]
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

## Two tables, aligned by label

**Level 3** · `data.tables.aligned` · standard

The weights of a portfolio come from another table, listed in another order. Contracting
the prices with them must pair each asset with its own weight, by name.

```latex
P_{ta} := \texttt{tables/prices.csv}
w^a := \texttt{tables/weights.csv}[weight]
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
