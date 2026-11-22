# Regression

Linear regression in closed form on data from Python: ridge regression and its effective
degrees of freedom, the coefficient of determination of a fitted line, and the lasso's
soft thresholding.

## Ridge regression

**Level 2** · `ml.regression.ridge` · Hastie, Tibshirani and Friedman (2009), section 3.4.1

The ridge coefficients \((X^T X + \lambda I)^{-1} X^T \mathbf{y}\), with the design matrix and
the responses given as NumPy arrays in Python, and \(\lambda = 1\):
\((\frac{72}{67}, \frac{273}{335})\).

```python-scope
import numpy as np

X = np.array([[1, 2], [2, 1], [3, 3], [4, 3], [5, 6]])
y = np.array([3, 3, 6, 7, 10])
```

```latex
X := \texttt{X}
\mathbf{y} := \texttt{y}
\lambda := 1
(X^T X + \lambda I)^{-1} X^T \mathbf{y}
```

```python
from scipy.optimize import minimize

X = Matrix([[1, 2], [2, 1], [3, 3], [4, 3], [5, 6]])
y = Matrix([3, 3, 6, 7, 10])
expected = (X.T * X + eye(2)).LUsolve(X.T * y)
assert expected == Matrix([Rational(72, 67), Rational(273, 335)])
A, b = np.array(X.tolist(), float), np.array(y.tolist(), float).ravel()
penalised = minimize(lambda w: np.sum((b - A @ w) ** 2) + w @ w, np.zeros(2), tol=1e-12)
assert np.allclose(penalised.x, [float(v) for v in expected], atol=1e-6)
```

## The effective degrees of freedom of ridge regression

**Level 3** · `ml.regression.effective-df` · Hastie, Tibshirani and Friedman (2009), section 3.4.1

\(\operatorname{df}(\lambda) = \operatorname{tr}[X(X^T X + \lambda I)^{-1} X^T]\), the trace of
the ridge smoother. Its columns orthogonal with squared lengths 2 and 4, this \(X\) has
\(\operatorname{df}(2) = \frac{2}{2 + 2} + \frac{4}{4 + 2} = \frac{7}{6}\), the sum
\(\sum_j \frac{d_j^2}{d_j^2 + \lambda}\) over its singular values.

```latex
X := \begin{pmatrix} 1 & 0 \\ 1 & 0 \\ 0 & 2 \\ 0 & 0 \end{pmatrix}
\lambda := 2
\operatorname{tr}\left[X (X^T X + \lambda I)^{-1} X^T\right]
```

```python
X = Matrix([[1, 0], [1, 0], [0, 2], [0, 0]])
expected = (X * (X.T * X + 2 * eye(2)).inv() * X.T).trace()
assert expected == Rational(7, 6)
singular = np.linalg.svd(np.array(X.tolist(), float), compute_uv=False)
assert abs(np.sum(singular**2 / (singular**2 + 2)) - 7 / 6) < 1e-12
```

## The coefficient of determination

**Level 3** · `ml.regression.r-squared` · James, Witten, Hastie and Tibshirani (2013), section 3.1

A line fitted to eight students' hours of study and their scores, from a pandas
DataFrame: the slope \(\operatorname{Cov}(x, y)/\operatorname{Var}(x)\), the intercept, and
\(R^2 = 1 - \text{RSS}/\text{TSS} = 0.98847\), the share of the variance of the scores the line
explains.

```python-scope
import pandas as pd

students = pd.read_csv("tables/students.csv")
```

```latex
x_t := \texttt{students["hours"]}
y_t := \texttt{students["score"]}
\hat\beta_1 := \frac{\operatorname{Cov}_t(x_t, y_t)}{\operatorname{Var}_t(x_t)}
\hat\beta_0 := \mathbb{E}_t[y_t] - \hat\beta_1 \mathbb{E}_t[x_t]
1 - \frac{\sum_t (y_t - \hat\beta_0 - \hat\beta_1 x_t)^2}{\sum_t (y_t - \mathbb{E}_t[y_t])^2}
```

```python
import csv

with open("tables/students.csv") as file:
    rows = [(int(h), int(s)) for h, s in list(csv.reader(file))[1:]]
x = [Integer(h) for h, _ in rows]
y = [Integer(s) for _, s in rows]
mx, my = sum(x) / len(x), sum(y) / len(y)
slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / sum((a - mx) ** 2 for a in x)
intercept = my - slope * mx
rss = sum((b - intercept - slope * a) ** 2 for a, b in zip(x, y))
expected = 1 - rss / sum((b - my) ** 2 for b in y)
assert expected == Rational(33856, 34251)
assert (
    abs(np.corrcoef([float(v) for v in x], [float(v) for v in y])[0, 1] ** 2 - float(expected))
    < 1e-12
)
```

## The lasso's soft thresholding

**Level 4** · `ml.regression.lasso` · Hastie, Tibshirani and Friedman (2009), section 3.4

With orthonormal columns, the lasso shrinks each least-squares coefficient towards zero by
\(\lambda\) and stops at zero: \(\operatorname{sign}(\hat\beta_j)(|\hat\beta_j| - \lambda)_+\),
with the subscript \(+\) for the positive part. For \(\hat\beta = (3, -0.5, 1.2)\) and
\(\lambda = 1\) it gives \((2, 0, 0.2)\).

```latex
\hat\beta := \begin{pmatrix} 3 \\ -0.5 \\ 1.2 \end{pmatrix}
\lambda := 1
\operatorname{sign}(\hat\beta_j)\left(|\hat\beta_j| - \lambda\right)_+, \quad j = 1, 2, 3
```

```python
from scipy.optimize import minimize_scalar

estimates = [Integer(3), Rational(-1, 2), Rational(6, 5)]
expected = [sign(b) * Max(Abs(b) - 1, 0) for b in estimates]
assert expected == [2, 0, Rational(1, 5)]
for b, shrunk in zip(estimates, expected, strict=True):  # each minimises (z - b)^2/2 + |z|
    best = minimize_scalar(
        lambda z, b=float(b): (z - b) ** 2 / 2 + abs(z), bounds=(-5, 5), method="bounded"
    )
    assert abs(best.x - float(shrunk)) < 1e-5
```
