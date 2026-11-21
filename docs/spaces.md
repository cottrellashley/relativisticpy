# Beyond spacetime

Index notation never needed spacetime. An index ranges over a **space**: the four
coordinates of a metric, 500 assets, 2,520 trading days, the two states of a qubit.
A tensor is an array whose axes range over spaces, Einstein summation is a
contraction, and a metric identifies a space with its dual. RelativisticPy reads the
same notation in every case. It computes symbolic components with SymPy and
numerical ones with NumPy, SciPy and scikit-learn.

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

Data brings its own spaces. Binding `R_{ta}` says that `t` ranges over the rows and
`a` over the columns, and a label used on a tensor ranges over that tensor's space
for the rest of the statement, so it need not be declared. Binding declares nothing
beyond that, so `t` can still be a coordinate in \(\Gamma^{t}{}_{tr}\). Data bound
later shares the space of the same index of a tensor already bound, as `w^a` would
share the assets of `R_{ta}`, and like indices of one tensor share one space, as both
of a covariance's do; `t` and `a` are not alike, even when there are as many days as
assets. With pandas, an axis's name names its space and its labels name the space's
basis, by which components can be taken, as coordinates name the components of a
metric. Data is aligned with a space's basis by label, and refused when its labels
are not the space's:

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
are Python numbers.

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
element makes `g_{\mu\nu}` spacetime's. Then indices pair up, and raising and
lowering use it. With the covariance as the metric of the assets, portfolio theory is
geometry: `w_a` is each asset's marginal risk, `w^a w_a` the variance of the
portfolio, and `\mu^a` the direction of Markowitz's portfolios:

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

| | |
|---|---|
| `\mathbb{E}_t[X]` | the expectation over `t` |
| `\operatorname{Var}_t(X)` | the variance of each component |
| `\operatorname{Cov}_t(X, Y)` | the covariance, with the free indices of both |
| `\operatorname{LedoitWolf}_t(R_{ta})` | the Ledoit–Wolf shrinkage estimate of a covariance, from scikit-learn |

```python
>>> np.allclose(wb["Sigma"].to_numpy(), wb.expr(r"\operatorname{Cov}_t(R_{ta}, R_{tb})").to_numpy())
True

```

## Linear algebra

| | |
|---|---|
| `(\Sigma^{-1})^{ab}` | the inverse of a two-index tensor, its indices in the other position |
| `\operatorname{eig}(A^a{}_b)` | eigenvalues and eigenvectors: `\lambda_k, v^a{}_k := \operatorname{eig}(A^a{}_b)` |
| `\operatorname{spec}(A)` | the eigenvalues; with a second argument `n`, the `n` largest, or smallest if negative, as for a large sparse matrix |
| `\operatorname{svd}(X_{ta})` | `U_{tk}, s_k, V_{ak} := \operatorname{svd}(X_{ta})` |
| `\det`, `\operatorname{tr}` | of a map, or of a matrix of components |
| `\|v\|` | the length of a vector, with the metric of its space |
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

A map is a tensor with upper indices that it takes to lower ones; its eigenvalues
come with a new index, over a space of their own. A bilinear form, with both indices
down, has eigenvalues only relative to a metric:
\(S_{ab} v^b = \lambda g_{ab} v^b\). They are computed relative to the metric of the
form's space. Symbolic components give exact results:

```python
>>> spins.exe(r"H^{i}{}_{j} := [[\epsilon, \Delta], [\Delta, -\epsilon]]")
[]
>>> spins.expr(r"\operatorname{spec}(H^i{}_j)")
Tensor('_{k}', [-sqrt(Delta**2 + epsilon**2), sqrt(Delta**2 + epsilon**2)])

```

## Domains

- **Portfolios**: expected returns, covariances, the tangency portfolio, risk as a
  metric, and the spectrum of the correlation matrix against the Marchenko–Pastur
  law of random matrices, which tells real factors from noise. See
  [examples/portfolio.py](https://github.com/cottrellashley/relativisticpy/blob/main/examples/portfolio.py).
- **Quantum systems**: an operator on two spins is a tensor with two indices of each
  kind, \(H^{ik}{}_{jl}\) on \(\mathcal{H} \otimes \mathcal{H}\). See
  [examples/spins.md](https://github.com/cottrellashley/relativisticpy/blob/main/examples/spins.md).
- **Graphs**: the Laplacian \(L_{ij} = \operatorname{diag}(k)_{ij} - A_{ij}\) with
  \(k_i = A_{ij} 1^j\), and the eigenvector of its second eigenvalue, which cuts the
  graph in two.
- **Regression**: the normal equations, \(\beta^k = (G^{-1})^{kl} X_{tl} y^t\) with
  \(G_{kl} = X_{tk} X_{tl}\), solved rather than inverted.

## Scale

- **Planning.** Contractions are ordered by cost with `numpy.einsum`'s optimiser,
  and the plan is kept for the next contraction of the same shapes.
- **Memory.** A contraction whose operands are too large to hold at once is done in
  slices along an index it sums over, and the slices added.
- **Sparse data.** A SciPy sparse matrix is a tensor like any other; it multiplies
  vectors and dense matrices as a matrix, and its few largest or smallest eigenvalues
  come from an iterative solver: `\operatorname{spec}(L^i{}_j, -6)`. Without a count,
  a sparse matrix of up to 4,096 rows has its whole spectrum computed as a dense one.
- **JAX.** `Workbook(backend="jax")` keeps bound data as JAX arrays, computed with JAX,
  on a GPU if there is one; a sparse matrix stays SciPy's, and multiplies JAX arrays
  as one of JAX's. `wb.grad(expression, "w")` differentiates any scalar expression
  with respect to bound data, through contractions, sparse matrices and linear solves:

```python
>>> grad = Workbook()
>>> grad["w^a"] = np.array([1.0, 2.0])
>>> grad["S_{ab}"] = np.array([[2.0, 0.0], [0.0, 3.0]])
>>> grad.grad(r"w^a S_{ab} w^b", "w").to_numpy()
array([ 4., 12.])

```

Numerical work needs `pip install "relativisticpy[numeric]"`; pandas and
scikit-learn come with `relativisticpy[data]`, and JAX with `relativisticpy[jax]`.
