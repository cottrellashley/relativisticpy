# Neural networks

A small network with one hidden layer of two sigmoid units: its output, with the sigmoid
applied to each component of a vector; the derivative of a loss with respect to a
weight, by the chain rule; and backpropagation, the errors \(\boldsymbol{\delta}\) passed
back through the layers.

## A forward pass

**Level 2** · `ml.network.forward-pass` · Goodfellow, Bengio and Courville (2016), chapter 6

\(\mathbf{w}_2^T\sigma(W_1\mathbf{x} + \mathbf{b}_1) + b_2\), with the sigmoid applied component
by component: the hidden layer is \(\sigma((0, 1)) = (\frac{1}{2}, \sigma(1))\), and the
output \(\frac{1}{2} - \sigma(1) = -0.2311\).

```latex
\sigma(a) := \frac{1}{1 + e^{-a}}
W_1 := \begin{pmatrix} 1 & -1 \\ 2 & 0 \end{pmatrix}
\mathbf{b}_1 := \begin{pmatrix} 0 \\ -1 \end{pmatrix}
\mathbf{w}_2 := \begin{pmatrix} 1 \\ -1 \end{pmatrix}
b_2 := 0
\mathbf{x} := \begin{pmatrix} 1 \\ 1 \end{pmatrix}
\mathbf{w}_2^T \sigma(W_1 \mathbf{x} + \mathbf{b}_1) + b_2
```

```python
from scipy.special import expit

sigma = lambda a: 1 / (1 + exp(-a))
hidden = (Matrix([[1, -1], [2, 0]]) * Matrix([1, 1]) + Matrix([0, -1])).applyfunc(sigma)
expected = (Matrix([[1, -1]]) * hidden)[0]
assert simplify(expected - (Rational(1, 2) - sigma(1))) == 0
layer = expit(np.array([[1, -1], [2, 0]]) @ np.array([1, 1]) + np.array([0, -1]))
assert abs(np.array([1, -1]) @ layer - float(expected)) < 1e-15
```

## The derivative of a loss with respect to a weight

**Level 3** · `ml.network.chain-rule` · Goodfellow, Bengio and Courville (2016), section 6.5

The squared error of a network with one hidden unit, \(\frac{1}{2}(w_2\sigma(w_1x) - t)^2\),
differentiated with respect to the first weight and evaluated: at \(w_1 = 0\), \(w_2 = 1\),
\(x = 1\) and \(t = 0\) the chain rule gives
\((w_2\sigma - t) \cdot w_2 \cdot \sigma(1 - \sigma) \cdot x = \frac{1}{8}\).

```latex
\sigma(a) := \frac{1}{1 + e^{-a}}
L(w_1, w_2) := \frac{1}{2}\left(w_2 \sigma(w_1 x) - t\right)^2
\left. \frac{\partial L}{\partial w_1} \right|_{w_1 = 0, w_2 = 1, x = 1, t = 0}
```

```python
w1, w2, x, t = symbols("w_1 w_2 x t")
sigma = 1 / (1 + exp(-w1 * x))
L = (w2 * sigma - t) ** 2 / 2
expected = diff(L, w1).subs({w1: 0, w2: 1, x: 1, t: 0})
assert expected == Rational(1, 8)
h = 1e-6
loss = lambda a: 0.5 * (1 / (1 + np.exp(-a))) ** 2
assert abs((loss(h) - loss(-h)) / (2 * h) - 0.125) < 1e-9
```

## Backpropagation

**Level 4** · `ml.network.backpropagation` · Bishop (2006), section 5.3

The errors of the output and of the hidden layer, labelled by layer with superscripts in
parentheses: \(\delta^{(2)} = y - t\) for the squared error, and
\(\boldsymbol{\delta}^{(1)} = \sigma'(\mathbf{z}^{(1)}) \odot \mathbf{w}^{(2)}\delta^{(2)}\),
with \(\odot\) the product component by component. The gradient of the loss with respect to
the first layer's weights is then the outer product \(\boldsymbol{\delta}^{(1)}\mathbf{x}^T\).

```latex
\sigma(a) := \frac{1}{1 + e^{-a}}
W^{(1)} := \begin{pmatrix} 1 & -1 \\ 2 & 0 \end{pmatrix}
\mathbf{b}^{(1)} := \begin{pmatrix} 0 \\ -1 \end{pmatrix}
\mathbf{w}^{(2)} := \begin{pmatrix} 1 \\ -1 \end{pmatrix}
\mathbf{x} := \begin{pmatrix} 1 \\ 1 \end{pmatrix}
t := 1
\mathbf{z}^{(1)} := W^{(1)} \mathbf{x} + \mathbf{b}^{(1)}
y := (\mathbf{w}^{(2)})^T \sigma(\mathbf{z}^{(1)})
L := \frac{1}{2}(y - t)^2
\delta^{(2)} := y - t
\boldsymbol{\delta}^{(1)} := \sigma'(\mathbf{z}^{(1)}) \odot \mathbf{w}^{(2)} \delta^{(2)}
\boldsymbol{\delta}^{(1)} \mathbf{x}^T
```

```python
w = symbols("w_11 w_12 w_21 w_22")
W = Matrix([[w[0], w[1]], [w[2], w[3]]])
sigma = lambda a: 1 / (1 + exp(-a))
output = (Matrix([[1, -1]]) * (W * Matrix([1, 1]) + Matrix([0, -1])).applyfunc(sigma))[0]
loss = (output - 1) ** 2 / 2
values = dict(zip(w, [1, -1, 2, 0], strict=True))
expected = Matrix(2, 2, lambda i, j: diff(loss, w[2 * i + j]).subs(values))
z = [0, 1]  # the hidden pre-activations
delta2 = Rational(1, 2) - sigma(1) - 1
delta1 = [sigma(z[i]) * (1 - sigma(z[i])) * [1, -1][i] * delta2 for i in range(2)]
assert simplify(expected - Matrix(2, 2, lambda i, j: delta1[i] * 1)) == zeros(2, 2)
assert np.allclose(
    np.array(expected.evalf().tolist(), float), [[-0.3077646, -0.3077646], [0.2420408, 0.2420408]]
)
```
