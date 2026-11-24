# Beyond spacetime

> **For:** anyone with data in NumPy or pandas; no General Relativity needed. **You will
> learn:** how data is bound to tensors, what spaces and indices are, and how to write
> statistics and linear algebra in index notation.
> **Before this:** [Getting started](getting-started.md), with the `data` extra.

Index notation never needed spacetime. An index ranges over a **space**: the four
coordinates of a metric, 500 assets, 2,520 trading days, the two states of a qubit.
A tensor is an array whose axes range over spaces, and RelativisticPy reads the same
notation in every case. It computes symbolic components with SymPy and numerical ones
with NumPy, SciPy and scikit-learn.

```python
>>> import numpy as np
>>> from relativisticpy import Workbook
>>> wb = Workbook()
>>> wb["R_{ta}"] = np.array([[0.01, 0.02, -0.01],
...                          [0.03, -0.01, 0.00],
...                          [-0.02, 0.01, 0.02],
...                          [0.00, 0.02, 0.03]])        # 4 days of returns on 3 assets
>>> wb.exe(r"""
...     \mu_a := \mathbb{E}_t[R_{ta}]
...     \Sigma_{ab} := \mathbb{E}_t[(R_{ta} - \mu_a)(R_{tb} - \mu_b)]
... """)
[]
>>> wb["mu"].to_numpy().round(4).tolist()
[0.005, 0.01, 0.01]
>>> wb["Sigma"].to_numpy().round(5).tolist()[0]
[0.00032, -0.00012, -0.00018]

```

## Five words

- **Index.** A letter written on a tensor, as `t` and `a` are on `R_{ta}`, which counts
  along one of its axes: here `t` counts the days and `a` the assets.
- **Space.** What an index counts over: the days, the assets. Data brings its own
  spaces, and a space can be declared.
- **Upper and lower.** An index is written up, as in `w^a`, or down, as in `\mu_a`.
  Upper indices are vectors, such as the weights of a portfolio; lower ones are
  covectors, such as the returns that weights are weighed against.
- **Contraction.** An index written twice in a term, once up and once down, is summed
  over: `w^a \mu_a` is the sum of each weight times its expected return. This is
  Einstein's convention, and it is how every product of tensors is written.
- **Metric.** A tensor `g_{ab}` that turns upper indices into lower ones,
  `w_a = g_{ab} w^b`, and so measures lengths. Until a space is given one, its metric is
  the identity, and an index may be summed in any position.

## Spaces

A space is declared in the notation, with its dimension, and the index labels that
range over it are declared with `\in`:

```python
>>> spins = Workbook()
>>> spins.exe(r"""
...     \mathcal{H} := \mathbb{C}^{2}
...     i, j \in \mathcal{H}
...     \sigma^{i}{}_{j} := [[0, 1], [1, 0]]
... """)
[]
>>> spins["sigma"].spaces[0]
Space(name='H', dim=2, basis=None, field='complex', spacetime=False)

```

Data brings its own spaces, by these rules:

1. **Binding names the axes.** `wb["R_{ta}"] = returns` says that `t` ranges over the
   rows of `returns` and `a` over its columns.
2. **A label written on a tensor takes that tensor's space** for the rest of the
   statement, so it need not be declared: in `\mathbb{E}_t[R_{ta}]`, `t` counts days.
3. **Binding declares nothing else**, so `t` can still be a coordinate elsewhere, as in
   `\Gamma^{t}{}_{tr}`.
4. **Data bound later shares a space** with the same index of a tensor already bound,
   if it has as many components: `w^a`, bound after `R_{ta}`, ranges over its assets.
5. **Like indices of one tensor share one space**, as the two of a covariance
   `\Sigma_{ab}` do. `t` and `a` are not alike, even when there are as many days as
   assets.
6. **pandas names spaces and their bases.** An axis's name names its space, and its
   labels the space's basis, by which components can be taken, as coordinates name the
   components of a metric. Labelled data meets the space whose elements its labels are,
   whatever its axis is named, and is aligned with that space's basis by label, never by
   position; it is refused where its labels are not the space's:

```python
>>> import pandas as pd
>>> prices = pd.Series([0.5, 0.3, 0.2], index=pd.Index(["AAPL", "MSFT", "NVDA"], name="assets"))
>>> book = Workbook()
>>> book["w^a"] = prices
>>> book.expr(r"w^{a:MSFT}")
0.3

```

An index ranges over one space, and RelativisticPy says so when it does not:

```python
>>> wb.expr(r"R_{ta} R_{at}")
Traceback (most recent call last):
  ...
relativisticpy.errors.IndexNotationError: the index a ranges over the space of a, but is written here on an axis over the space of t
 --> <input>:1:8
  |
1 | R_{ta} R_{at}
  |        ^^^^^^

```

## Data in, results out

`wb["R_{ta}"] = data` binds a NumPy array, a pandas DataFrame or Series, a SciPy
sparse matrix, a SymPy array or nested lists. A name without braces works too:
binding to `y_t` binds the tensor `y` with the index `t`. Results are
[`Tensor`](python.md#tensor) values with `to_numpy()` and `to_pandas()`; numbers
are Python numbers. What was bound reads back as it was written, `wb["R_{ta}"]`,
and `"R" in wb` says whether it was. Data that cannot be bound is refused with a
`DataError`, which is also a `ValueError`, saying why:

```python
>>> wb["v^a"] = [0.5, 0.25, [0.25]]
Traceback (most recent call last):
  ...
relativisticpy.errors.DataError: the data for v^a is ragged: entry 3 is a list, the others are numbers

```

A name takes a number, `wb["J"] = 1.5`, or text, which is read as a program reads
it, never run as Python: `wb["J"] = "x^2 + 1"`.

## Python objects, by name

Python reads the files, in its own cells, with any library there is, and the language
reads the objects by name. `\texttt{prices}` is the object called `prices` in the scope
the workbook is given, and attributes, keys and positions follow, as in
`\texttt{table.x}`, `\texttt{prices["NVDA"]}` and `\texttt{data[0]}`. The scope is a
mapping, as `globals()` or a dict, or a module or any object whose attributes are read:

```python
>>> import pandas as pd
>>> prices = pd.DataFrame(
...     {"AAPL": [231.5, 233.0, 229.75], "NVDA": [141.2, 143.8, 140.1]},
...     index=pd.Index(["2026-11-16", "2026-11-17", "2026-11-18"], name="date"),
... )
>>> weights = pd.Series([0.25, 0.75], index=pd.Index(["NVDA", "AAPL"], name="asset"))
>>> market = Workbook(scope={"prices": prices, "weights": weights})
>>> market.exe(r"""
...     P_{ta} := \texttt{prices}
...     w^a := \texttt{weights}
...     x_t := \texttt{prices["NVDA"]}
... """)
[]
>>> market.expr(r"P_{ta} w^a").to_numpy().round(2).tolist()   # each asset by its own weight
[208.92, 210.7, 207.34]
>>> market.expr(r"\mathbb{E}_t[x_t]")
141.7

```

What an object becomes depends on its type, as an object integration of
`relativisticpy.integrations` declares it:

- **A number** is a scalar, exact where Python's is: an integer or a fraction stays one.
- **An array**, from NumPy, or anything with `__array__` or the array API, as a JAX or
  PyTorch array, is data, each axis over a space of its size.
- **A pandas DataFrame or Series** is labelled data. Defined with indices, as
  `P_{ta} := \texttt{prices}`, it is bound as `wb["P_{ta}"] = prices` binds it: its index
  and columns are the bases of spaces, named after its axes.
- **A SciPy sparse matrix** is sparse data, and **an xarray DataArray** data over its
  named dimensions, labelled by their coordinates.
- **A SymPy expression**, or anything with `_sympy_`, is an expression, its symbols the
  program's by name, and a SymPy matrix is a matrix.
- **A list or tuple of numbers** is a vector or a matrix, as `[1, 2]` is.
- **A function**, or anything else Python calls, is called: on numbers, and on data
  elementwise. One written with NumPy is given whole arrays; any other, one number at a
  time.

Data from two objects meets by label, never by position: above, `weights` lists the
assets in another order than `prices`, and each asset is weighed by its own weight.
Labels that are not the same elements are refused where they meet. Written with indices,
as `\texttt{signal}_t`, an object is data with those indices, without a definition. A
function and an expression are used as the program's own:

```python
>>> import numpy as np
>>> import sympy
>>> def hann(n, N):
...     return 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))
>>> windowed = Workbook(scope={"hann": hann, "signal": np.array([4.0, 2.0, 6.0, 8.0])})
>>> windowed.expr(r"\mathbb{E}_t[\texttt{signal}_t]")
5.0
>>> windowed.expr(r"\texttt{hann}(\texttt{signal}_t, 9)").to_numpy().round(3).tolist()
[1.0, 0.5, 0.5, 0.0]
>>> x = sympy.Symbol("x")
>>> Workbook(scope={"f": sympy.sin(x) ** 2}).expr(r"\frac{d}{dx} \texttt{f}")
2*sin(x)*cos(x)

```

A reference is read by a grammar of its own, a name, then attributes, keys in quotes and
positions, and is never Python to run: anything else is refused, saying what may be
written. An object is read anew each time a program names it, and what it becomes is
the workbook's own, so that changing the object afterwards changes nothing already read.
A workbook given no scope reaches nothing in Python, nor does a document checked with
`relativisticpy check`, nor a program sent to the [MCP server](mcp.md):

```python
>>> Workbook().expr(r"\texttt{prices}")
Traceback (most recent call last):
  ...
relativisticpy.errors.CompileError: \texttt{prices} names a Python object, but no Python scope was given to read it from
 --> <input>:1:1
  |
1 | \texttt{prices}
  | ^^^^^^^^^^^^^^^
note: a document checked with relativisticpy check, and a program sent to the MCP server, are given no Python scope, so that they never reach into Python
hint: give the workbook one, as Workbook(scope=globals()), or the object to show, as show(source, prices=prices)

```

In a notebook, `show(source, prices=prices)` gives a program its objects, and the editor
takes a scope: see [Live mathematics](live.md#python-objects).

## Positions, metrics and duals

Upper indices are vectors and lower ones covectors, in every domain. In a space whose
metric is the identity, as is every space until it is given another, an index may be
summed in any position, as in Euclidean tensor calculus. Its Kronecker delta and the
covector of ones, `1_a` or `\mathbb{1}_a`, are written as usual:

```python
>>> wb["w^a"] = np.array([0.5, 0.25, 0.25])
>>> wb.expr(r"1_a w^a")
1.0
>>> wb.expr(r"w^a w^b \delta_{ab}")
0.375

```

Any symmetric positive definite tensor can be made a space's metric, as the line
element makes `g_{\mu\nu}` spacetime's; anything else is refused where it is first
used. Then indices pair up, and raising and lowering use it. With the covariance as
the metric of the assets, portfolio theory is geometry: `w_a` is each asset's
marginal risk, `w^a w_a` the variance of the portfolio, and `\mu^a` the direction of
Markowitz's portfolios:

```python
>>> wb.exe(r"g_{ab} := \Sigma_{ab}")
[]
>>> round(wb.expr(r"w^a w_a"), 8) == round(wb.expr(r"w^a \Sigma_{ab} w^b"), 8)
True
>>> wb.expr(r"w_{a} w_{a}")
Traceback (most recent call last):
  ...
relativisticpy.errors.IndexNotationError: index a appears twice as a lower index; a summed index must appear once up and once down
 --> <input>:1:1
  |
1 | w_{a} w_{a}
  | ^^^^^^^^^^^

```

## Statistics

`\mathbb{E}_t[\ldots]` binds `t` as a sum does: inside it every factor shares `t`, and
a term without `t`, such as `\mu_a` in `R_{ta} - \mu_a`, is constant in it. It is a
contraction with the measure of `t`'s space, uniform unless `wb.space(...,
measure=weights)` gives it weights, and RelativisticPy computes it as one contraction
with whatever it multiplies, never forming the product of everything averaged. With
2,520 days and 500 assets, that product would take 5 GB.

| Statistic | Meaning |
|---|---|
| `\mathbb{E}_t[X]` | the expectation over `t` |
| `\operatorname{Var}_t(X)` | the variance of each component, of \(\lvert X - \mathbb{E}_t X \rvert^2\) for complex data |
| `\operatorname{Cov}_t(X, Y)` | the covariance, with the free indices of both; of complex data, with \(Y\) conjugated, as NumPy's |
| `\operatorname{LedoitWolf}_t(R_{ta})` | the Ledoit–Wolf shrinkage estimate of a covariance, from scikit-learn, which weighs every sample alike |

Without an index, `\mathbb{E}[X]`, `\operatorname{Var}(X)` and `\operatorname{Cov}(X, Y)`
are of random variables, declared as [Random variables](probability.md) says; of
data they are refused, saying which index to write.

```python
>>> np.allclose(wb["Sigma"].to_numpy(), wb.expr(r"\operatorname{Cov}_t(R_{ta}, R_{tb})").to_numpy())
True

```

## Linear algebra

| Written | Meaning |
|---|---|
| `(\Sigma^{-1})^{ab}` | the inverse of a two-index tensor: of a form, its indices in the other position; of a map, a map, `(A^{-1})^a{}_b` |
| `\operatorname{eig}(A^a{}_b)` | eigenvalues and eigenvectors: `\lambda_k, v^a{}_k := \operatorname{eig}(A^a{}_b)` |
| `\operatorname{spec}(A)` | the eigenvalues; with a second argument `n`, the `n` largest, or smallest if negative, as for a large sparse matrix |
| `\operatorname{svd}(X_{ta})` | `U_{tk}, s_k, V_{ak} := \operatorname{svd}(X_{ta})`, so that \(X_{ta} = U_{tk} \operatorname{diag}(s)_{kl} V_{al}\) |
| `\det`, `\operatorname{tr}` | of a map, or of a matrix of components |
| `\operatorname{rank}`, `\rho`, `\lambda_{\max}`, `\kappa`, `expm`, `norm` | the [functions of a matrix](matrices.md), of a map's matrix of components: a number, or a map of the same space |
| `\|v\|` | the length of a vector, with the metric of its space: \(\sqrt{g_{ab} \bar v^a v^b}\) |
| `\operatorname{diag}(d)_{ab}` | the diagonal matrix of a vector |
| `A_{ta} \odot B_{ta}` | the elementwise product, which index notation has no other way to write |
| `\exp(H)`, `\sqrt{\Sigma}`, `\log` | functions of a matrix, for a tensor named without indices: `\exp(-i H t)` |

An inverse multiplied by what it acts on is never formed: it is solved for, by a
Cholesky factorisation when the matrix is symmetric positive definite. The tangency
portfolio is written as in a textbook and computed with two solves:

```python
>>> m = wb.expr(r"\frac{(\Sigma^{-1})^{ab} \mu_b}{1_c (\Sigma^{-1})^{cd} \mu_d}")
>>> round(float(m.to_numpy().sum()), 10)
1.0
>>> "solve" in wb.ir(r"(\Sigma^{-1})^{ab} \mu_b")
True

```

A map, or linear operator, has one index up and one down, `A^a{}_b`: it takes a vector
`v^b` to the vector `A^a{}_b v^b`. Its eigenvalues come with a new index, over a space
of their own. A bilinear form, with both indices down, has eigenvalues only relative to
a metric: \(S_{ab} v^b = \lambda g_{ab} v^b\). They are computed relative to the metric
of the form's space. Symbolic components give exact results:

```python
>>> spins.exe(r"H^{i}{}_{j} := [[\epsilon, \Delta], [\Delta, -\epsilon]]")
[]
>>> spins.expr(r"\operatorname{spec}(H^i{}_j)")
spec(H^i{}_j) over eigenvectors of H^i{}_j: 2 components non-zero
  (spec(H^i{}_j))_0 = −√(Δ² + ε²)
  (spec(H^i{}_j))_1 = √(Δ² + ε²)

```

## Grids and the Fourier transform

A range written beside a definition declares the space of its index, of one element for
each value, labelled by it, and the definition is made for each value: `n`, used as a
number, is the value. A sum whose body writes its index on data runs over the whole of
the data's space, from its lower limit, and binds the index, as a statistic does:

```python
>>> grid = Workbook()
>>> grid.exe(r"""
...     N := 8
...     h := \frac{1}{N}
...     x_n := n\,h, \quad n = 0, \ldots, N-1
... """)
[]
>>> grid["x"].spaces[0].basis
('0', '1', '2', '3', '4', '5', '6', '7')
>>> grid.expr(r"\sum_n x_n")
7/2

```

`\operatorname{fft}` is the discrete Fourier transform, \(X_k = \sum_n x_n e^{-2\pi i k
n/N}\), by the fast algorithm, with NumPy, or JAX. Of data over an index, it gives data
over a new space, the frequencies of the index, labelled by their bins, k = 0, …, N − 1,
as X_k is written: bin k holds the frequency k/(N d), for samples d apart, and (k − N)/(N d)
from N/2 on. `\operatorname{ifft}` gives back data over the samples' own space:

```python
>>> signal = Workbook()
>>> signal["y_t"] = np.array([1.0, 2.0, 0.0, -1.0])
>>> spectrum = signal.expr(r"\operatorname{fft}(y_t)")
>>> spectrum.spaces[0].basis
('0', '1', '2', '3')
>>> spectrum.to_numpy().round(10).tolist()
[(2+0j), (1-3j), 0j, (1+3j)]
>>> signal.expr(r"\operatorname{ifft}(\operatorname{fft}(y_t))").spaces == signal["y"].spaces
True

```

The transform written as its sum is computed as an FFT too: a rewrite, which gives the
numbers the sum gives, in \(N \log N\) steps rather than \(N^2\):

```python
>>> grid.exe(r"X_k := \sum_{n=0}^{N-1} x_n e^{-2\pi i k n/N}, \quad k = 0, \ldots, N-1")
[]
>>> "call fft" in grid.ir("X_k")
True

```

## Optima over an index

An optimum over an index of data binds the index, as a sum does, and is the greatest or the
least value along it, for each value of any other index; `\arg\max` and `\arg\min` give the
label of the element where it is, a number where the label is one, or else its position.
Where several elements share it, the first is given, and a note says so. An optimum over
variables is found by the optimizers (see [Optima](optima.md)):

```python
>>> signal.exe(r"""
...     \max_t y_t
...     \arg\min_t y_t
... """)
[2.0, 3]
>>> grid.expr(r"\arg\max_n \sin(\pi x_n)")
4
>>> assets = Workbook()
>>> assets["R_{ta}"] = np.array([[0.01, 0.02, -0.01], [0.03, -0.02, 0.0]])
>>> assets.expr(r"\max_t R_{ta}").to_numpy().tolist()
[0.03, 0.02, 0.0]

```

## Domains

**Portfolios**: expected returns, covariances, the tangency portfolio, risk as a metric,
and the spectrum of the correlation matrix against the Marchenko–Pastur law of random
matrices, which tells real factors from noise: see the
[portfolio of 500 assets](examples.md#a-portfolio-of-500-assets-in-index-notation).

**Quantum systems**: an operator on two spins is a tensor with two indices of each kind,
\(H^{ik}{}_{jl}\) on \(\mathcal{H} \otimes \mathcal{H}\): see
[two spins](examples.md#two-spins).

**Graphs**: a graph of six nodes, two triangles joined by one edge, has the adjacency
matrix `A_{ij}`. Its degrees are `k_i = A_{ij} 1^j`, its Laplacian is
`L_{ij} = \operatorname{diag}(k)_{ij} - A_{ij}`, and the eigenvector of its second
eigenvalue, Fiedler's vector, cuts the graph in two where it is thinnest:

```python
>>> A = np.zeros((6, 6))
>>> for i, j in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]:
...     A[i, j] = A[j, i] = 1.0
>>> graph = Workbook()
>>> graph["A_{ij}"] = A
>>> graph.exe(r"""
...     k_i := A_{ij} 1^j
...     L_{ij} := \operatorname{diag}(k)_{ij} - A_{ij}
...     \lambda_n, v^i{}_n := \operatorname{eig}(L^i{}_j)
... """)
[]
>>> round(float(graph["lambda"].to_numpy()[1]), 4)                # the second eigenvalue
0.4384
>>> fiedler = graph["v"].to_numpy()[:, 1]
>>> (np.sign(fiedler) == np.sign(fiedler[0])).tolist()   # each node's side of the cut
[True, True, True, False, False, False]

```

**Regression**: least squares is the normal equations as they are written,
\(\beta^k = (G^{-1})^{kl} X_{tl} y_t\) with \(G_{kl} = X_{tk} X_{tl}\), and the inverse is
solved for, never formed. Define `G` with `:=`, since `G_{kl}` with two indices is
otherwise the Einstein tensor:

```python
>>> fit = Workbook()
>>> fit["X_{tk}"] = np.column_stack([np.ones(5), np.arange(5.0)])   # a constant, and t
>>> fit["y_t"] = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
>>> fit.exe(r"""
...     G_{kl} := X_{tk} X_{tl}
...     \beta^k := (G^{-1})^{kl} X_{tl} y_t
... """)
[]
>>> fit["beta"].to_numpy().round(10).tolist()                     # y = 1 + 2t
[1.0, 2.0]

```

## Scale

- **Planning.** Contractions are ordered by cost with `numpy.einsum`'s optimiser,
  and the plan is kept for the next contraction of the same shapes.
- **Memory.** A contraction whose operands are too large to hold at once is done in
  slices along an index it sums over, and the slices added.
- **Sparse data.** A SciPy sparse matrix is a tensor like any other, and is never
  densified: it multiplies vectors, dense matrices and other sparse matrices as a
  matrix, its products with sparse matrices stay sparse, and its few largest or
  smallest eigenvalues come from an iterative solver:
  `\operatorname{spec}(L^i{}_j, -6)`. Without a count, a sparse matrix of up to 4,096
  rows has its whole spectrum computed as a dense one.
- **JAX.** `Workbook(backend="jax")` keeps bound data as JAX arrays, computed with JAX,
  on a GPU if there is one; a sparse matrix stays SciPy's, and multiplies JAX arrays
  as one of JAX's. Statistics need 64-bit numbers, so it turns on JAX's
  `jax_enable_x64` for the whole process. `wb.grad(expression, "w")` differentiates
  any scalar expression with respect to bound data, through contractions, sparse
  matrices and linear solves:

```python
>>> grad = Workbook()
>>> grad["w^a"] = np.array([1.0, 2.0])
>>> grad["S_{ab}"] = np.array([[2.0, 0.0], [0.0, 3.0]])
>>> grad.grad(r"w^a S_{ab} w^b", "w").to_numpy()
array([ 4., 12.])

```

Numerical work needs `pip install "relativisticpy[numeric]"`; pandas and
scikit-learn come with `relativisticpy[data]`, and JAX with `relativisticpy[jax]`.

**Next:** [Python API](python.md#tensor), for every method of a tensor and of its data,
or the [examples](examples.md), worked through.
