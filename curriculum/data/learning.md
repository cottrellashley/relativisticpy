# Learning from data

The softmax and the cross-entropy of classifiers, the least-squares loss and its
gradient, and principal components, with small datasets given inline in index
notation: Anscombe's first set and the ten points of Smith's tutorial on principal
components.

## The cross-entropy of a prediction

**Level 1** · `data.learning.cross-entropy` · Goodfellow, Bengio and Courville (2016), section 6.2.2.3

A classifier gives three classes the scores 1, 2 and 3, and the true class is the
third: the cross-entropy loss is the negative logarithm of its softmax probability.

```latex
-\ln \frac{e^{3}}{e^{1} + e^{2} + e^{3}}
```

```python
expected = log(E + E**2 + E**3) - 3
z = np.array([1.0, 2.0, 3.0])
assert abs(float(expected) - (-np.log(np.exp(z[2]) / np.exp(z).sum()))) < 1e-12
assert abs(float(expected) - 0.4076) < 1e-4
```

## The softmax

**Level 2** · `data.learning.softmax` · Goodfellow, Bengio and Courville (2016), section 6.2.2.3

\(\operatorname{softmax}(z)_i = \exp(z_i) / \sum_j \exp(z_j)\), defined as the book
defines it and applied to the scores \((1, 2, 3)\).

```latex
\mathcal{K} := \mathbb{R}^{3}
i, j \in \mathcal{K}
z_i := [1, 2, 3]
\operatorname{softmax}(z)_i := \frac{\exp(z_i)}{\sum_j \exp(z_j)}
\operatorname{softmax}(z)_i
```

```python
total = E + E**2 + E**3
expected = [E / total, E**2 / total, E**3 / total]
assert np.allclose([float(p) for p in expected], [0.09003057, 0.24472847, 0.66524096])
```

## The gradient of the least-squares loss

**Level 3** · `data.learning.least-squares-gradient` · Hastie, Tibshirani and Friedman (2009), section 3.2

The residual sum of squares of a straight line through four points, and its gradient
with respect to the coefficients \(\beta^k\), written in index notation:
\(-2 X_{tk} (y_t - X_{tl} \beta^l)\).

```latex
\mathcal{T} := \mathbb{R}^{4}
\mathcal{K} := \mathbb{R}^{2}
t \in \mathcal{T}
k, l \in \mathcal{K}
X_{tk} := [[1, 1], [1, 2], [1, 3], [1, 4]]
y_{t} := [6, 5, 7, 10]
\beta^{k} := [\beta_0, \beta_1]
L := (y_{t} - X_{tk} \beta^{k})(y_{t} - X_{tl} \beta^{l})
\frac{\partial L}{\partial \beta^{k}}
```

```python
beta_0, beta_1 = symbols("beta_0 beta_1")
X = Matrix([[1, 1], [1, 2], [1, 3], [1, 4]])
y = Matrix([6, 5, 7, 10])
b = Matrix([beta_0, beta_1])
residual = y - X * b
loss = (residual.T * residual)[0, 0]
expected = [diff(loss, beta_0), diff(loss, beta_1)]
assert Matrix(expected) == (-2 * X.T * residual).applyfunc(expand)  # -2 X^T (y - X beta)
assert solve(expected, [beta_0, beta_1]) == {beta_0: Rational(7, 2), beta_1: Rational(7, 5)}
```

## Principal components as an eigen-decomposition

**Level 3** · `data.learning.pca` · Smith (2002), section 3.1

Ten points in the plane, their mean and their sample covariance, with the sums over the
points written out as the tutorial writes them, and the eigenvalues of the covariance:
the variances along the principal axes, 0.0490833989 and 1.28402771.

```latex
\mathcal{T} := \mathbb{R}^{10}
\mathcal{A} := \mathbb{R}^{2}
t \in \mathcal{T}
a, b \in \mathcal{A}
X_{ta} := [[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], [2.3, 2.7], [2, 1.6], [1, 1.1], [1.5, 1.6], [1.1, 0.9]]
n := 10
\mu_{a} := \frac{1}{n} \sum_{t=1}^{n} X_{ta}
C_{ab} := \frac{1}{n - 1} \sum_{t=1}^{n} (X_{ta} - \mu_{a})(X_{tb} - \mu_{b})
\operatorname{spec}(C^{a}{}_{b})
```

```python
x = [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1]
y = [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9]
points = np.column_stack([x, y])
covariance = np.cov(points.T)
assert np.allclose(covariance, [[0.616555556, 0.615444444], [0.615444444, 0.716555556]])
expected = list(np.linalg.eigvalsh(covariance))
assert np.allclose(expected, [0.0490833989, 1.28402771])  # the tutorial's eigenvalues
```

## Anscombe's regression line

**Level 3** · `data.learning.anscombe-regression` · Anscombe (1973)

The first of Anscombe's four datasets, and its least-squares line by the normal
equations, \(\beta^k = (G^{-1})^{kl} X_{tl} y_t\) with \(G_{kl} = X_{tk} X_{tl}\): the
line \(y = 3.00 + 0.500 x\) that all four sets share.

```latex
\mathcal{T} := \mathbb{R}^{11}
\mathcal{K} := \mathbb{R}^{2}
t \in \mathcal{T}
k, l \in \mathcal{K}
X_{tk} := [[1, 10], [1, 8], [1, 13], [1, 9], [1, 11], [1, 14], [1, 6], [1, 4], [1, 12], [1, 7], [1, 5]]
y_{t} := [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
G_{kl} := X_{tk} X_{tl}
(G^{-1})^{kl} X_{tl} y_{t}
```

```python
x = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], float)
y = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])
X = np.column_stack([np.ones_like(x), x])
expected = list(np.linalg.solve(X.T @ X, X.T @ y))
assert np.allclose(expected, [3.0, 0.5], atol=5e-4)  # y = 3.00 + 0.500 x
assert abs(np.corrcoef(x, y)[0, 1] - 0.816) < 5e-4
```
