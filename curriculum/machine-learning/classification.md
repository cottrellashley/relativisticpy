# Classification

The logistic sigmoid and the rectifier by their book names, the derivative that makes
logistic regression's gradient simple, the margin of a linear classifier, the gradient of
the cross-entropy of logistic regression on data from Python, and of the softmax in index
notation.

## The logistic sigmoid

**Level 1** · `ml.functions.sigmoid` · Bishop (2006), section 4.2

\(\sigma(a) = \frac{1}{1 + e^{-a}}\), the logistic sigmoid, by the name machine learning gives
it: \(\sigma(\ln 3) = \frac{3}{4}\).

```latex
\sigma(\ln 3)
```

```python
from scipy.special import expit

a = symbols("a")
expected = (1 / (1 + exp(-a))).subs(a, log(3))
assert expected == Rational(3, 4)
assert abs(expit(np.log(3)) - 0.75) < 1e-15
```

## The rectifier

**Level 1** · `ml.functions.relu` · Goodfellow, Bengio and Courville (2016), section 6.3.1

The rectified linear unit, \(\operatorname{ReLU}(x) = \max(0, x)\), by its name.

```latex
\operatorname{ReLU}(-2) + \operatorname{ReLU}(3)
```

```python
expected = Max(0, -2) + Max(0, 3)
assert expected == 3 == np.maximum(0, -2) + np.maximum(0, 3)
```

## The derivative of the sigmoid

**Level 2** · `ml.logistic.derivative` · Bishop (2006), section 4.3.2

The identity \(\frac{d\sigma}{da} = \sigma(1 - \sigma)\), which makes the gradient of logistic
regression's error so simple: a claim, which holds for every \(a\).

```latex
\sigma(a) := \frac{1}{1 + e^{-a}}
\frac{d\sigma}{da} = \sigma(a)\left(1 - \sigma(a)\right)
```

```python
from scipy.special import expit

a = symbols("a")
sigma = 1 / (1 + exp(-a))
assert simplify(diff(sigma, a) - sigma * (1 - sigma)) == 0
h = 1e-6
for point in (-2.0, 0.3, 4.0):
    slope = (expit(point + h) - expit(point - h)) / (2 * h)
    assert abs(slope - expit(point) * (1 - expit(point))) < 1e-9
expected = True
```

## The margin of a linear classifier

**Level 2** · `ml.classification.margin` · Bishop (2006), section 7.1

With the canonical scaling, the nearest points lie on \(\mathbf{w}^T\mathbf{x} + b = \pm 1\), at
a distance \(1/\|\mathbf{w}\|\) from the boundary, so the band between them is
\(2/\|\mathbf{w}\|\) wide: \(\frac{2}{5}\) for \(\mathbf{w} = (3, 4)\).

```latex
\mathbf{w} := \begin{pmatrix} 3 \\ 4 \end{pmatrix}
\frac{2}{\|\mathbf{w}\|}
```

```python
expected = 2 / sqrt(3**2 + 4**2)
assert expected == Rational(2, 5)
assert abs(2 / np.linalg.norm([3, 4]) - 0.4) < 1e-15
```

## The gradient of logistic regression's error

**Level 3** · `ml.logistic.gradient` · Bishop (2006), section 4.3

The cross-entropy error of logistic regression has the gradient
\(\Phi^T(\mathbf{y} - \mathbf{t})\), where \(\mathbf{y} = \sigma(\Phi\mathbf{w})\) applies the
sigmoid to each component. The design matrix and the labels come from Python; at
\(\mathbf{w} = 0\) every prediction is \(\frac{1}{2}\), and the gradient is \((0, -2)\).

```python-scope
import numpy as np

features = np.array([[1, 0.5], [1, 1.5], [1, 2.5], [1, 3.5]])
labels = np.array([0, 0, 1, 1])
```

```latex
\sigma(a) := \frac{1}{1 + e^{-a}}
\Phi := \texttt{features}
\mathbf{t} := \texttt{labels}
\mathbf{w} := \begin{pmatrix} 0 \\ 0 \end{pmatrix}
\mathbf{y} := \sigma(\Phi \mathbf{w})
\Phi^T (\mathbf{y} - \mathbf{t})
```

```python
from scipy.special import expit

features = np.array([[1, 0.5], [1, 1.5], [1, 2.5], [1, 3.5]])
labels = np.array([0, 0, 1, 1])
gradient = features.T @ (expit(features @ np.zeros(2)) - labels)
assert np.allclose(gradient, [0, -2])


def cross_entropy(w):
    p = expit(features @ w)
    return -np.sum(labels * np.log(p) + (1 - labels) * np.log(1 - p))


h = 1e-6  # the gradient of the error itself, by differences
numeric = [(cross_entropy(h * e) - cross_entropy(-h * e)) / (2 * h) for e in np.eye(2)]
assert np.allclose(numeric, gradient, atol=1e-6)
expected = [0, -2]
```

## The gradient of the softmax cross-entropy

**Level 4** · `ml.softmax.cross-entropy-gradient` · Bishop (2006), section 4.3.4

For the softmax \(y_k = e^{a_k}/\sum_j e^{a_j}\) and a one-hot target \(t_k\), the derivative of
the cross-entropy \(E = -\sum_k t_k \ln y_k\) with respect to each activation is
\(y_j - t_j\): in index notation, over a space of three classes.

```latex
\mathcal{K} := \mathbb{R}^{3}
j, k \in \mathcal{K}
a_k := [a_1, a_2, a_3]
t_k := [0, 0, 1]
y_k := \frac{e^{a_k}}{\sum_j e^{a_j}}
E := -\sum_k t_k \ln y_k
\frac{\partial E}{\partial a_j}
```

```python
a = symbols("a_1 a_2 a_3")
total = sum(exp(v) for v in a)
softmax = [exp(v) / total for v in a]
cross_entropy = -log(softmax[2])  # the target is the third class
expected = [simplify(diff(cross_entropy, v)) for v in a]
target = [0, 0, 1]
assert all(simplify(g - (p - t)) == 0 for g, p, t in zip(expected, softmax, target, strict=True))
```
