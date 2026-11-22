# Unconstrained minimisation

Minima and maximisers written with \(\min\) and \(\arg\max\) under the variable, the
Hessian written \(\nabla^2 f\) as optimisation books write it, Newton's step, and
gradient descent with an exact line search.

## The minimum of a quadratic

**Level 1** · `opt.unconstrained.minimum-value` · standard

The least value of \(x^2 - 4x + 7 = (x - 2)^2 + 3\) over the real line, with the variable
and its set under the \(\min\).

```latex
\min_{x \in \mathbb{R}} \left(x^2 - 4x + 7\right)
```

```python
x = symbols("x", real=True)
expected = minimum(x**2 - 4 * x + 7, x, S.Reals)
assert expected == 3 == ((x - 2) ** 2 + 3).subs(x, 2)
```

## Where a function is greatest

**Level 1** · `opt.unconstrained.argmax` · standard

The point, not the value: \(x e^{-x}\) is greatest on \(x \geq 0\) at \(x = 1\), where its
derivative \((1 - x)e^{-x}\) changes sign.

```latex
\arg\max_{x \geq 0} \; x e^{-x}
```

```python
x = symbols("x", nonnegative=True)
assert solve(diff(x * exp(-x), x), x) == [1]
assert diff(x * exp(-x), x, 2).subs(x, 1) < 0
grid = np.linspace(0, 10, 100001)
assert abs(grid[np.argmax(grid * np.exp(-grid))] - 1) < 1e-4
expected = 1
```

## The Hessian of Rosenbrock's function

**Level 2** · `opt.unconstrained.rosenbrock-hessian` · Nocedal and Wright (2006), chapter 2

In optimisation \(\nabla^2 f\) is the Hessian, the matrix of second derivatives, not the
Laplacian. At the minimiser \((1, 1)\) of Rosenbrock's function it is positive
definite.

```latex
f(x_1, x_2) := 100(x_2 - x_1^2)^2 + (1 - x_1)^2
\nabla^2 f(1, 1)
```

```python
x1, x2 = symbols("x_1 x_2")
f = 100 * (x2 - x1**2) ** 2 + (1 - x1) ** 2
assert solve([diff(f, x1), diff(f, x2)], [x1, x2], dict=True) == [{x1: 1, x2: 1}]
expected = hessian(f, (x1, x2)).subs({x1: 1, x2: 1})
assert expected == Matrix([[802, -400], [-400, 200]])
assert all(value > 0 for value in expected.eigenvals())
```

## Newton's step

**Level 2** · `opt.unconstrained.newton-step` · Boyd and Vandenberghe (2004), section 9.5

The Newton step \(-[\nabla^2 f(x)]^{-1} \nabla f(x)\), with the gradient and the Hessian
evaluated at a point: from the origin it reaches the minimiser \((2, -1)\) of a convex
quadratic in one step.

```latex
f(x, y) := x^2 + x y + y^2 - 3x
\begin{pmatrix} 0 \\ 0 \end{pmatrix} - \left[\nabla^2 f(0, 0)\right]^{-1} \nabla f(0, 0)
```

```python
x, y = symbols("x y")
f = x**2 + x * y + y**2 - 3 * x
gradient = Matrix([diff(f, x), diff(f, y)])
expected = -hessian(f, (x, y)).inv() * gradient.subs({x: 0, y: 0})
assert expected == Matrix([2, -1])
assert solve(list(gradient), [x, y]) == {x: 2, y: -1}
```

## Gradient descent with an exact line search

**Level 3** · `opt.unconstrained.exact-line-search` · Boyd and Vandenberghe (2004), section 9.3

One step of gradient descent on \(f(x) = \frac{1}{2}(x_1^2 + \gamma x_2^2)\) from
\((\gamma, 1)\), with the step length \(t^*\) that minimises \(f\) along the ray: for
\(\gamma = 10\) it is \(\frac{2}{11}\), and the step ends at
\(\left(\gamma\frac{\gamma - 1}{\gamma + 1}, -\frac{\gamma - 1}{\gamma + 1}\right) =
\left(\frac{90}{11}, -\frac{9}{11}\right)\), as the book's closed form says.

```latex
\gamma := 10
f(x_1, x_2) := \frac{1}{2}\left(x_1^2 + \gamma x_2^2\right)
\mathbf{x} := \begin{pmatrix} \gamma \\ 1 \end{pmatrix}
t^* := \arg\min_{t \geq 0} f(\mathbf{x} - t \nabla f(\mathbf{x}))
\mathbf{x} - t^* \nabla f(\mathbf{x})
```

```python
t = symbols("t", nonnegative=True)
gamma = 10
f = lambda v: (v[0] ** 2 + gamma * v[1] ** 2) / 2
start = Matrix([gamma, 1])
gradient = Matrix([start[0], gamma * start[1]])
steps = solve(diff(f(start - t * gradient), t), t)
assert steps == [Rational(2, 11)]
expected = start - steps[0] * gradient
ratio = Rational(gamma - 1, gamma + 1)
assert expected == Matrix([gamma * ratio, -ratio])  # the book's closed form, k = 1
```
