# Constraints, multipliers and duality

Extrema under constraints: on an interval, on a circle, under an inequality, as a
quadratic program, and over functions. The Karush–Kuhn–Tucker conditions, the KKT system
of an equality-constrained quadratic program, the Lagrange dual function, and positive
definiteness written as Boyd and Vandenberghe write it.

## The closed interval method

**Level 1** · `opt.constrained.closed-interval` · Stewart (2015), section 4.1

The greatest value of \(x^3 - 3x\) on \([0, 3]\) is at the end of the interval, 18, not at
a critical point: \(x = 1\) is the least.

```latex
\max_{0 \leq x \leq 3} \left(x^3 - 3x\right)
```

```python
x = symbols("x", real=True)
f = x**3 - 3 * x
expected = maximum(f, x, Interval(0, 3))
candidates = [f.subs(x, point) for point in [0, 3, *solve(diff(f, x), x)] if 0 <= point <= 3]
assert expected == max(candidates) == 18
```

## A positive definite matrix

**Level 1** · `opt.convex.positive-definite` · Boyd and Vandenberghe (2004), section 2.4

\(A \succ 0\): the generalized inequality of the positive semidefinite cone, which says that
\(A\) is positive definite. This one is, with eigenvalues 1 and 3.

```latex
\begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix} \succ 0
```

```python
A = Matrix([[2, -1], [-1, 2]])
assert A.eigenvals() == {1: 1, 3: 1}
assert A.is_positive_definite
assert np.all(np.linalg.eigvalsh(np.array([[2.0, -1.0], [-1.0, 2.0]])) > 0)
expected = True
```

## A minimum on a circle

**Level 2** · `opt.constrained.circle` · Nocedal and Wright (2006), section 12.1

The least value of \(x_1 + x_2\) on the circle \(x_1^2 + x_2^2 = 2\), at \((-1, -1)\), written
as the problem is posed, with the constraint after "subject to".

```latex
\min_{x_1, x_2} \; x_1 + x_2 \quad \text{subject to} \quad x_1^2 + x_2^2 = 2
```

```python
x1, x2, lam = symbols("x_1 x_2 lambda")
lagrangian = x1 + x2 - lam * (x1**2 + x2**2 - 2)
points = solve([diff(lagrangian, v) for v in (x1, x2, lam)], [x1, x2, lam])
assert sorted(points) == [(-1, -1, Rational(-1, 2)), (1, 1, Rational(1, 2))]
expected = min(p[0] + p[1] for p in points)
assert expected == -2
theta = np.linspace(0, 2 * np.pi, 100001)
assert abs(np.min(np.sqrt(2) * (np.cos(theta) + np.sin(theta))) - expected) < 1e-8
```

## The Karush–Kuhn–Tucker conditions

**Level 3** · `opt.constrained.kkt` · Nocedal and Wright (2006), section 12.1

The same objective inside the disk, \(c(x) = 2 - x_1^2 - x_2^2 \geq 0\). The KKT
conditions, stationarity of the Lagrangian, complementarity, and the signs of the
multiplier and the constraint, written as a system of equations and inequalities: one
point satisfies them all, \((-1, -1)\) with \(\lambda = \frac{1}{2}\). Without the
inequalities, \((1, 1)\) with \(\lambda = -\frac{1}{2}\) would solve the equations too.

```latex
f(x_1, x_2) := x_1 + x_2
c(x_1, x_2) := 2 - x_1^2 - x_2^2
\mathcal{L}(x_1, x_2, \lambda) := f(x_1, x_2) - \lambda c(x_1, x_2)
solve([\frac{\partial \mathcal{L}}{\partial x_1} = 0, \frac{\partial \mathcal{L}}{\partial x_2} = 0, \lambda c(x_1, x_2) = 0, \lambda \geq 0, c(x_1, x_2) \geq 0], [x_1, x_2, \lambda])
```

```python
x1, x2, lam = symbols("x_1 x_2 lambda")
c = 2 - x1**2 - x2**2
lagrangian = x1 + x2 - lam * c
points = solve([diff(lagrangian, x1), diff(lagrangian, x2), lam * c], [x1, x2, lam], dict=True)
kkt = [p for p in points if p[lam] >= 0 and c.subs(p) >= 0]
expected = [(p[x1], p[x2], p[lam]) for p in kkt]
assert expected == [(-1, -1, Rational(1, 2))]
```

## A quadratic program: the long-only minimum-variance portfolio

**Level 3** · `opt.constrained.long-only` · Boyd and Vandenberghe (2004), section 4.4

Markowitz's problem with short sales forbidden, \(w \succeq 0\), for three assets. Without
that constraint the least variance would sell the second asset short; with it, the
second weight is zero and the others are \(\frac{4}{5}\) and \(\frac{1}{5}\).

```latex
\Sigma := \begin{pmatrix} 1 & 1.8 & 0 \\ 1.8 & 4 & 0 \\ 0 & 0 & 4 \end{pmatrix}
\arg\min_{w} \; w^T \Sigma w \quad \text{subject to} \quad \mathbf{1}^T w = 1, \quad w \succeq 0
```

```python
from scipy.optimize import minimize

Sigma = np.array([[1, 1.8, 0], [1.8, 4, 0], [0, 0, 4]])
unconstrained = np.linalg.solve(Sigma, np.ones(3))
assert (unconstrained / unconstrained.sum())[1] < 0  # it would sell short
expected = [Rational(4, 5), 0, Rational(1, 5)]
w = np.array([0.8, 0, 0.2])
gradient = 2 * Sigma @ w  # KKT: equal on the support, no smaller off it
assert np.isclose(gradient[0], gradient[2]) and gradient[1] > gradient[0]
found = minimize(
    lambda v: v @ Sigma @ v,
    np.ones(3) / 3,
    method="SLSQP",
    bounds=[(0, None)] * 3,
    constraints={"type": "eq", "fun": lambda v: v.sum() - 1},
    tol=1e-14,
)
assert np.allclose(found.x, w, atol=1e-6)
tolerance = 1e-6
```

## The KKT system of a quadratic program

**Level 4** · `opt.constrained.kkt-system` · Boyd and Vandenberghe (2004), section 10.1

Minimising \(\frac{1}{2}x^T P x + q^T x\) subject to \(Ax = b\) is one linear system, with
the multiplier \(\nu\) beside \(x\): a block matrix, with \(A^T\) and a zero block in it.
The solution is \(x = (\frac{1}{3}, \frac{2}{3})\) and \(\nu = \frac{4}{3}\).

```latex
P := \begin{pmatrix} 2 & 0 \\ 0 & 4 \end{pmatrix}
q := \begin{pmatrix} -2 \\ -4 \end{pmatrix}
A := \begin{pmatrix} 1 & 1 \end{pmatrix}
b := 1
\begin{pmatrix} P & A^T \\ A & 0 \end{pmatrix}^{-1} \begin{pmatrix} -q \\ b \end{pmatrix}
```

```python
P = Matrix([[2, 0], [0, 4]])
q = Matrix([-2, -4])
A = Matrix([[1, 1]])
system = Matrix(BlockMatrix([[P, A.T], [A, zeros(1, 1)]]))
expected = system.inv() * Matrix([2, 4, 1])
assert expected == Matrix([Rational(1, 3), Rational(2, 3), Rational(4, 3)])
x1, x2, nu = symbols("x_1 x_2 nu")
lagrangian = x1**2 - 2 * x1 + 2 * x2**2 - 4 * x2 + nu * (x1 + x2 - 1)
assert solve([diff(lagrangian, v) for v in (x1, x2, nu)], [x1, x2, nu]) == {
    x1: Rational(1, 3),
    x2: Rational(2, 3),
    nu: Rational(4, 3),
}
```

## The Lagrange dual function

**Level 4** · `opt.constrained.dual-function` · Boyd and Vandenberghe (2004), section 5.1

The dual function \(g(\nu) = \inf_x L(x, \nu)\) of the least-norm problem: minimise
\(x_1^2 + x_2^2\) subject to \(x_1 + x_2 = 1\). The infimum over \(x\) is attained where
\(x_1 = x_2 = -\nu/2\), so \(g(\nu) = -\frac{\nu^2}{2} - \nu\), whose greatest value,
\(\frac{1}{2}\), is the primal optimum.

```latex
L(x_1, x_2, \nu) := x_1^2 + x_2^2 + \nu (x_1 + x_2 - 1)
g(\nu) := \inf_{x_1, x_2} L(x_1, x_2, \nu)
g(\nu)
```

```python
x1, x2, nu = symbols("x_1 x_2 nu", real=True)
L = x1**2 + x2**2 + nu * (x1 + x2 - 1)
assert hessian(L, (x1, x2)).is_positive_definite  # so the stationary point is the infimum
at = solve([diff(L, x1), diff(L, x2)], [x1, x2])
expected = expand(L.subs(at))
assert expected == -(nu**2) / 2 - nu
assert maximum(expected, nu, S.Reals) == Rational(1, 2)  # strong duality
```

## A problem of optimal control

**Level 5** · `opt.control.minimum-energy` · Liberzon (2012), chapter 4

The least energy \(\int_0^1 u^2\,dt\) that moves \(x\) from 0 to 1 when \(\dot x = u\): a
minimum over functions, under a differential equation. The maximum principle gives a
constant control, \(u = 1\), and the energy 1. No design of the language reads a
minimum over a space of functions.

```latex
\min_{u} \int_0^1 u(t)^2 \, dt \quad \text{subject to} \quad \dot{x}(t) = u(t), \quad x(0) = 0, \quad x(1) = 1
```

```python
t, p = symbols("t p")
u = symbols("u")
hamiltonian = u**2 + p * u  # the costate p is constant, since H does not depend on x
control = solve(diff(hamiltonian, u), u)[0]  # u = -p/2, a constant
constant = solve(integrate(control, (t, 0, 1)) - 1, p)[0]  # x(1) - x(0) = 1
assert control.subs(p, constant) == 1
expected = integrate(control.subs(p, constant) ** 2, (t, 0, 1))
assert expected == 1
rng = np.random.default_rng(0)
for _ in range(5):  # any other control that reaches 1 costs more (Cauchy-Schwarz)
    other = 1 + rng.normal(size=1000)
    other += 1 - other.mean()
    assert np.mean(other**2) >= 1
```
