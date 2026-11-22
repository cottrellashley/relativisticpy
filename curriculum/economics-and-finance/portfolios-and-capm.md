# Portfolios and the CAPM

Jensen's alpha from a table of returns read in Python, and a point on Markowitz's
frontier from Merton's closed form. The two-asset portfolios of Bodie, Kane and Marcus,
and the beta of a stock, are in the `data/` domain.

## Jensen's alpha

**Level 3** · `econ.capm.alpha` · Jensen (1968)

Six months of returns of a stock and of the market, from a CSV file, and a risk-free
rate of 0.2 percent a month. The stock's beta is \(\operatorname{Cov}(r, m)/\operatorname{Var}(m)
= 1.23\), and its alpha, the mean excess return the CAPM does not explain, is
\(\mathbb{E}[r - r_f] - \beta\,\mathbb{E}[m - r_f] = 0.271\) percent a month.

```python-scope
import pandas as pd

returns = pd.read_csv("tables/returns.csv", index_col="month")
```

```latex
r_t := \texttt{returns["stock"]}
m_t := \texttt{returns["market"]}
r_f := 0.002
\beta := \frac{\operatorname{Cov}_t(r_t, m_t)}{\operatorname{Var}_t(m_t)}
\mathbb{E}_t[r_t - r_f] - \beta \, \mathbb{E}_t[m_t - r_f]
```

```python
import csv

with open("tables/returns.csv") as file:
    rows = list(csv.reader(file))[1:]
r = [Rational(row[1]) for row in rows]
m = [Rational(row[2]) for row in rows]
mr, mm = sum(r) / len(r), sum(m) / len(m)
beta = sum((a - mr) * (b - mm) for a, b in zip(r, m)) / sum((b - mm) ** 2 for b in m)
assert beta == Rational(123, 100)
expected = (mr - Rational(2, 1000)) - beta * (mm - Rational(2, 1000))
assert expected == Rational(271, 100000)
slope, intercept = np.polyfit([float(v) - 0.002 for v in m], [float(v) - 0.002 for v in r], 1)
assert abs(slope - 1.23) < 1e-12 and abs(intercept - float(expected)) < 1e-12  # the regression
```

## A point on the efficient frontier

**Level 4** · `econ.portfolio.frontier` · Merton (1972)

Merton's closed form for the least variance of a portfolio with expected return \(\mu\),
\(\sigma^2(\mu) = \frac{C\mu^2 - 2A\mu + B}{D}\), with \(A = \mathbf{1}^T\Sigma^{-1}\bar{\mathbf{e}}\),
\(B = \bar{\mathbf{e}}^T\Sigma^{-1}\bar{\mathbf{e}}\), \(C = \mathbf{1}^T\Sigma^{-1}\mathbf{1}\) and
\(D = BC - A^2\). For three uncorrelated assets with returns of 10, 15 and 20 percent and
volatilities of 20, 30 and 40 percent, the frontier at \(\mu = 0.15\) has variance
\(\frac{9}{280}\).

```latex
\Sigma := \operatorname{diag}(0.04, 0.09, 0.16)
\bar{\mathbf{e}} := \begin{pmatrix} 0.10 \\ 0.15 \\ 0.20 \end{pmatrix}
\mathbf{1} := \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix}
A := \mathbf{1}^T \Sigma^{-1} \bar{\mathbf{e}}
B := \bar{\mathbf{e}}^T \Sigma^{-1} \bar{\mathbf{e}}
C := \mathbf{1}^T \Sigma^{-1} \mathbf{1}
D := B C - A^2
\sigma^2(\mu) := \frac{C \mu^2 - 2 A \mu + B}{D}
\sigma^2(0.15)
```

```python
from scipy.optimize import minimize

Sigma = diag(Rational(4, 100), Rational(9, 100), Rational(16, 100))
e = Matrix([Rational(10, 100), Rational(15, 100), Rational(20, 100)])
one = ones(3, 1)
A = (one.T * Sigma.inv() * e)[0]
B = (e.T * Sigma.inv() * e)[0]
C = (one.T * Sigma.inv() * one)[0]
D = B * C - A**2
mu = Rational(15, 100)
expected = (C * mu**2 - 2 * A * mu + B) / D
assert expected == Rational(9, 280)
covariance, means = np.diag([0.04, 0.09, 0.16]), np.array([0.10, 0.15, 0.20])
found = minimize(
    lambda w: w @ covariance @ w,
    np.ones(3) / 3,
    constraints=[
        {"type": "eq", "fun": lambda w: w.sum() - 1},
        {"type": "eq", "fun": lambda w: w @ means - 0.15},
    ],
    tol=1e-15,
)
assert abs(found.fun - 9 / 280) < 1e-10  # the least variance, found directly
```
