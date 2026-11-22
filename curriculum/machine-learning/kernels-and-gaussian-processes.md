# Kernels and Gaussian processes

Kernels as inner products of feature maps: the inner product written with angle brackets,
the squared-exponential kernel, a polynomial kernel as the inner product of explicit
features, its Gram matrix, and the posterior mean of a Gaussian process.

## An inner product

**Level 1** · `ml.kernels.inner-product` · Schölkopf and Smola (2002), chapter 2

The inner product of two vectors written with angle brackets, as kernel methods write it:
\(\langle (1, 2), (3, 4) \rangle = 11\).

```latex
\langle (1, 2), (3, 4) \rangle
```

```python
expected = 1 * 3 + 2 * 4
assert expected == 11 == np.dot([1, 2], [3, 4])
```

## The squared-exponential kernel

**Level 1** · `ml.kernels.rbf` · Rasmussen and Williams (2006), section 4.2

The squared-exponential kernel with unit length scale, \(\exp(-\|x - x'\|^2/2)\), between the
points \((1, 2)\) and \((2, 0)\), a squared distance of 5 apart: \(e^{-5/2}\).

```latex
\exp\left(-\frac{\|(1, 2) - (2, 0)\|^2}{2}\right)
```

```python
expected = exp(-Rational(5, 2))
distance = np.linalg.norm(np.array([1, 2]) - np.array([2, 0]))
assert abs(np.exp(-(distance**2) / 2) - float(expected)) < 1e-15
```

## A kernel as an inner product of features

**Level 2** · `ml.kernels.feature-map` · Bishop (2006), section 6.2

The kernel \((\mathbf{x}^T\mathbf{z})^2\) in two dimensions is the inner product of the
features \(\phi(\mathbf{x}) = (x_1^2, \sqrt{2}x_1x_2, x_2^2)\): a claim for every pair of
points, which holds.

```latex
\phi(x_1, x_2) := \begin{pmatrix} x_1^2 \\ \sqrt{2} x_1 x_2 \\ x_2^2 \end{pmatrix}
\phi(x_1, x_2)^T \phi(z_1, z_2) = (x_1 z_1 + x_2 z_2)^2
```

```python
x1, x2, z1, z2 = symbols("x_1 x_2 z_1 z_2")
phi = lambda a, b: Matrix([a**2, sqrt(2) * a * b, b**2])
assert expand((phi(x1, x2).T * phi(z1, z2))[0] - (x1 * z1 + x2 * z2) ** 2) == 0
rng = np.random.default_rng(0)
for _ in range(3):
    p, q = rng.normal(size=2), rng.normal(size=2)
    features = lambda v: np.array([v[0] ** 2, np.sqrt(2) * v[0] * v[1], v[1] ** 2])
    assert abs(features(p) @ features(q) - (p @ q) ** 2) < 1e-12
expected = True
```

## A Gram matrix

**Level 3** · `ml.kernels.gram-matrix` · Bishop (2006), section 6.1

The Gram matrix \(K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)\) of the kernel
\(k(\mathbf{x}, \mathbf{y}) = (1 + \mathbf{x}^T\mathbf{y})^2\) on three points, defined entry by
entry over the range of its indices. Like every Gram matrix of a valid kernel it is
positive semidefinite.

```latex
\mathbf{x}_1 := \begin{pmatrix} 0 \\ 1 \end{pmatrix}
\mathbf{x}_2 := \begin{pmatrix} 1 \\ 0 \end{pmatrix}
\mathbf{x}_3 := \begin{pmatrix} 1 \\ 1 \end{pmatrix}
k(\mathbf{x}, \mathbf{y}) := \left(1 + \mathbf{x}^T \mathbf{y}\right)^2
K_{ij} := k(\mathbf{x}_i, \mathbf{x}_j), \quad i, j = 1, 2, 3
K_{ij}
```

```python
points = [Matrix([0, 1]), Matrix([1, 0]), Matrix([1, 1])]
K = Matrix(3, 3, lambda i, j: (1 + (points[i].T * points[j])[0]) ** 2)
assert K == Matrix([[4, 1, 4], [1, 4, 4], [4, 4, 9]])
assert all(value >= 0 for value in K.eigenvals())
assert np.all(np.linalg.eigvalsh(np.array(K.tolist(), float)) > -1e-12)
expected = K
```

## The posterior mean of a Gaussian process

**Level 4** · `ml.gp.posterior-mean` · Rasmussen and Williams (2006), section 2.2

Regression with a Gaussian process: four noisy observations from Python, the
squared-exponential kernel, noise of standard deviation \(\sigma_n = 0.1\), and the mean of
the prediction at \(x_* = 0\), \(\bar f_* = \mathbf{k}_*^T(K + \sigma_n^2 I)^{-1}\mathbf{y}\),
with \(K\) and \(\mathbf{k}_*\) defined entry by entry from the kernel: 0.65912.

```python-scope
import numpy as np

x_train = np.array([-1.5, -0.5, 0.5, 1.5])
y_train = np.array([-1.0, 0.0, 1.0, 0.5])
```

```latex
\mathbf{x} := \texttt{x_train}
\mathbf{y} := \texttt{y_train}
x_* := 0
\sigma_n := 0.1
k(x, x') := \exp\left(-\frac{(x - x')^2}{2}\right)
K_{ij} := k(x_i, x_j), \quad i, j = 1, \ldots, 4
(\mathbf{k}_*)_i := k(x_*, x_i), \quad i = 1, \ldots, 4
\mathbf{k}_*^T \left(K + \sigma_n^2 I\right)^{-1} \mathbf{y}
```

```python
from scipy.linalg import cho_factor, cho_solve

x = np.array([-1.5, -0.5, 0.5, 1.5])
y = np.array([-1.0, 0.0, 1.0, 0.5])
kernel = lambda a, b: np.exp(-((a - b) ** 2) / 2)
K = kernel(x[:, None], x[None, :])
k_star = kernel(0.0, x)
expected = float(k_star @ np.linalg.solve(K + 0.01 * np.eye(4), y))
alpha = cho_solve(cho_factor(K + 0.01 * np.eye(4)), y)  # the book's algorithm 2.1
assert abs(k_star @ alpha - expected) < 1e-12
assert abs(expected - 0.65912) < 5e-6
```
