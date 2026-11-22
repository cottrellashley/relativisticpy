# Portfolios

Mean–variance portfolio theory in index notation, over a space of assets: the variance
of a portfolio, the minimum-variance and tangency portfolios, and the beta of a stock.
The two assets are the bond and stock funds of Bodie, Kane and Marcus: expected returns
of 8 and 13 percent, standard deviations of 12 and 20 percent, a correlation of 0.3, and
a risk-free rate of 5 percent.

## The variance of a portfolio

**Level 2** · `data.portfolio.variance` · Bodie, Kane and Marcus (2014), chapter 7

\(\sigma_P^2 = w^a \Sigma_{ab} w^b\) for 40 percent in the bond fund and 60 percent in
the stock fund: a standard deviation of 14.2 percent.

```latex
\mathcal{A} := \mathbb{R}^{2}
a, b \in \mathcal{A}
\Sigma_{ab} := [[0.0144, 0.0072], [0.0072, 0.04]]
w^{a} := [0.4, 0.6]
w^{a} \Sigma_{ab} w^{b}
```

```python
sigma = np.array([0.12, 0.20])
Sigma = np.outer(sigma, sigma) * np.array([[1, 0.3], [0.3, 1]])
w = np.array([0.4, 0.6])
expected = float(w @ Sigma @ w)
assert abs(expected - 0.02016) < 1e-12
assert abs(np.sqrt(expected) - 0.142) < 5e-4  # the book's 14.2 percent
```

## The beta of a stock

**Level 2** · `data.portfolio.capm-beta` · Bodie, Kane and Marcus (2014), chapter 9

The beta of a stock against the market, \(\beta = \operatorname{Cov}(r, m) /
\operatorname{Var}(m)\), from six periods of returns given inline, each series indexed
by the period \(t\) as \(r_t\) and \(m_t\).

```latex
\mathcal{T} := \mathbb{R}^{6}
t \in \mathcal{T}
r_t := [0.05, -0.02, 0.03, 0.04, -0.01, 0.02]
m_t := [0.03, -0.01, 0.02, 0.03, -0.02, 0.01]
\frac{\operatorname{Cov}_t(r_t, m_t)}{\operatorname{Var}_t(m_t)}
```

```python
r = np.array([0.05, -0.02, 0.03, 0.04, -0.01, 0.02])
m = np.array([0.03, -0.01, 0.02, 0.03, -0.02, 0.01])
expected = Rational(14, 11)
assert abs(np.cov(r, m)[0, 1] / np.var(m, ddof=1) - float(expected)) < 1e-12
assert abs(np.polyfit(m, r, 1)[0] - float(expected)) < 1e-9  # the slope of r on m
```

## The minimum-variance portfolio

**Level 3** · `data.portfolio.minimum-variance` · Bodie, Kane and Marcus (2014), chapter 7

The weights \(w^a \propto (\Sigma^{-1})^{ab} 1_b\), normalised, put 82 percent in the
bond fund; the risk of the portfolio is then a standard deviation of 11.45 percent.

```latex
\mathcal{A} := \mathbb{R}^{2}
a, b, c, d \in \mathcal{A}
\Sigma_{ab} := [[0.0144, 0.0072], [0.0072, 0.04]]
w^{a} := \frac{(\Sigma^{-1})^{ab} 1_{b}}{1_{c} (\Sigma^{-1})^{cd} 1_{d}}
\sqrt{w^{a} \Sigma_{ab} w^{b}}
```

```python
Sigma = np.array([[0.0144, 0.0072], [0.0072, 0.04]])
w = np.linalg.solve(Sigma, np.ones(2))
w /= w.sum()
assert np.allclose(w, [0.82, 0.18])
expected = float(np.sqrt(w @ Sigma @ w))
assert abs(expected - 0.1145) < 5e-5  # the book's 11.45 percent
```

## The tangency portfolio

**Level 3** · `data.portfolio.tangency` · Bodie, Kane and Marcus (2014), chapter 7

The risky portfolio with the steepest capital allocation line, \(w^a \propto
(\Sigma^{-1})^{ab} (\mu_b - r_f 1_b)\): 40 percent in bonds, 60 in stocks, and a
Sharpe ratio of 0.42.

```latex
\mathcal{A} := \mathbb{R}^{2}
a, b, c, d \in \mathcal{A}
\Sigma_{ab} := [[0.0144, 0.0072], [0.0072, 0.04]]
\mu_{a} := [0.08, 0.13]
r_f := 0.05
w^{a} := \frac{(\Sigma^{-1})^{ab} (\mu_{b} - r_f 1_{b})}{1_{c} (\Sigma^{-1})^{cd} (\mu_{d} - r_f 1_{d})}
\frac{w^{a} \mu_{a} - r_f}{\sqrt{w^{a} \Sigma_{ab} w^{b}}}
```

```python
Sigma = np.array([[0.0144, 0.0072], [0.0072, 0.04]])
mu, rf = np.array([0.08, 0.13]), 0.05
w = np.linalg.solve(Sigma, mu - rf)
w /= w.sum()
assert np.allclose(w, [0.4, 0.6])
expected = float((w @ mu - rf) / np.sqrt(w @ Sigma @ w))
assert abs(expected - 0.42) < 5e-3  # the book's 0.42
```

## The minimum-variance portfolio as an optimisation

**Level 4** · `data.portfolio.argmin` · Bodie, Kane and Marcus (2014), chapter 7

The same portfolio as the solution of the problem Markowitz posed: the weights that
minimise the variance, subject to their summing to one.

```latex
\mathcal{A} := \mathbb{R}^{2}
a, b \in \mathcal{A}
\Sigma_{ab} := [[0.0144, 0.0072], [0.0072, 0.04]]
\arg\min_{w} \; w^{a} \Sigma_{ab} w^{b} \quad \text{subject to} \quad 1_{a} w^{a} = 1
```

```python
from scipy.optimize import minimize

Sigma = np.array([[0.0144, 0.0072], [0.0072, 0.04]])
found = minimize(
    lambda w: w @ Sigma @ w,
    [0.5, 0.5],
    constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
    tol=1e-14,
)
assert np.allclose(found.x, [0.82, 0.18], atol=1e-6)
expected = [Rational(41, 50), Rational(9, 50)]
```
