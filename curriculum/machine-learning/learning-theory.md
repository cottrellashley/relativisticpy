# Learning theory

What the best possible classifier can do, and how rich a class of classifiers is: a
minimum over all functions of a probability under a joint distribution, and the
Vapnik–Chervonenkis dimension of half-planes. Both quantify over sets the language has
no design for yet.

## The Bayes error rate

**Level 5** · `ml.theory.bayes-error` · Hastie, Tibshirani and Friedman (2009), section 2.4

Two equally likely classes, each normal with unit variance, centred at \(-1\) and \(1\). No
classifier errs less often than the Bayes classifier, which thresholds at 0; its error
rate is \(\Phi(-1) = 0.1587\). The minimum ranges over every function \(h\), and the
probability is under a joint distribution given by its conditionals.

```latex
Y \sim \operatorname{Bernoulli}\left(\frac{1}{2}\right)
X \mid Y = 0 \sim \mathcal{N}(-1, 1)
X \mid Y = 1 \sim \mathcal{N}(1, 1)
\min_{h} P\left(h(X) \neq Y\right)
```

```python
from scipy.stats import norm

expected = (1 - erf(1 / sqrt(2))) / 2  # Phi(-1)
assert abs(float(expected) - norm.cdf(-1)) < 1e-15
threshold = lambda c: 0.5 * norm.sf(c, loc=-1) + 0.5 * norm.cdf(c, loc=1)
assert all(threshold(c) > threshold(0) for c in (-0.5, 0.2, 1.0))  # 0 is the best threshold
assert abs(threshold(0) - float(expected)) < 1e-15
rng = np.random.default_rng(0)
labels = rng.integers(0, 2, 200_000)
samples = rng.normal(2 * labels - 1, 1)
assert abs(np.mean((samples > 0) != labels) - float(expected)) < 0.005
```

## The VC dimension of half-planes

**Level 5** · `ml.theory.vc-dimension` · Hastie, Tibshirani and Friedman (2009), section 7.9

The classifiers that split the plane by a line shatter three points in general position,
realising all eight labellings, but no four points: the Vapnik–Chervonenkis dimension of
the class is 3. A class of functions written in set-builder notation, and a dimension
defined by quantifiers over point sets and labellings.

```latex
\mathcal{H} := \left\{ \mathbf{x} \mapsto \operatorname{sign}(\mathbf{w}^T \mathbf{x} + b) : \mathbf{w} \in \mathbb{R}^2, \, b \in \mathbb{R} \right\}
\operatorname{VCdim}(\mathcal{H})
```

```python
from itertools import product

from scipy.optimize import linprog


def separable(points, labels):  # is there (w, b) with y (w.x + b) >= 1 for every point?
    rows = [[-y * x[0], -y * x[1], -y] for x, y in zip(points, labels, strict=True)]
    found = linprog([0, 0, 0], A_ub=rows, b_ub=[-1] * len(points), bounds=[(None, None)] * 3)
    return found.status == 0


triangle = [(0, 0), (1, 0), (0, 1)]
assert all(separable(triangle, labels) for labels in product([-1, 1], repeat=3))
# Four points cannot be shattered (Radon): in convex position the diagonals cross, and
# otherwise one point lies in the triangle of the others.
assert not separable([(0, 0), (1, 1), (1, 0), (0, 1)], [1, 1, -1, -1])
assert not separable([(0, 0), (4, 0), (0, 4), (1, 1)], [1, 1, 1, -1])
expected = 3
```
