# Lagrangian mechanics

Lagrangians written as mechanics books write them, with dots for time derivatives and
the coordinates implicitly functions of time: the Euler–Lagrange equations they give,
equilibria and small oscillations, the conserved quantities that symmetries bring, and
Hamilton's principle behind them all.

## The simple pendulum

**Level 2** · `cm.lagrangian.pendulum` · Taylor (2005), section 7.2

The Euler–Lagrange equation of a pendulum, from a Lagrangian written with
\(\dot\theta\) and no mention that \(\theta\) depends on \(t\): the dot says so.

```latex
L = \frac{1}{2} m l^2 \dot\theta^2 + m g l \cos\theta
\frac{d}{dt} \frac{\partial L}{\partial \dot\theta} - \frac{\partial L}{\partial \theta}
```

```python
t, m, l, g = symbols("t m l g")
theta = Function("theta")(t)
L = Rational(1, 2) * m * l**2 * diff(theta, t) ** 2 + m * g * l * cos(theta)
expected = diff(diff(L, diff(theta, t)), t) - diff(L, theta)
assert simplify(expected - (m * l**2 * diff(theta, t, 2) + m * g * l * sin(theta))) == 0
```

## A bead on a spinning hoop

**Level 3** · `cm.lagrangian.bead-on-hoop` · Taylor (2005), section 7.5

A bead slides on a hoop of radius \(R\) spinning at \(\omega\) about its vertical
diameter. Its effective potential, read off the Lagrangian at \(\dot\theta = 0\), is
curved at the bottom by \(m R^2 (g/R - \omega^2)\): the bottom is stable only while
\(\omega^2 < g/R\), and \(g/R - \omega^2\) is the square of the frequency of small
oscillations there.

```latex
L = \frac{1}{2} m R^2 \left(\dot\theta^2 + \omega^2 \sin^2\theta\right) - m g R (1 - \cos\theta)
U_{\text{eff}} = -\left. L \right|_{\dot\theta = 0}
\left. \frac{1}{m R^2} \frac{d^2 U_{\text{eff}}}{d\theta^2} \right|_{\theta = 0}
```

```python
m, R, omega, g, theta, theta_dot = symbols("m R omega g theta theta_dot")
L = m * R**2 / 2 * (theta_dot**2 + omega**2 * sin(theta) ** 2) - m * g * R * (1 - cos(theta))
U_eff = -L.subs(theta_dot, 0)
expected = simplify(diff(U_eff, theta, 2).subs(theta, 0) / (m * R**2))
assert simplify(expected - (g / R - omega**2)) == 0
# the same from the equation of motion, m R^2 theta'' = dL/dtheta, linearised about theta = 0
acceleration = diff(L, theta) / (m * R**2)
assert simplify(-diff(acceleration, theta).subs(theta, 0) - expected) == 0
```

## The normal modes of a double pendulum

**Level 3** · `cm.lagrangian.double-pendulum` · Taylor (2005), section 11.4

A double pendulum of two equal masses on equal strings. The mass and stiffness matrices
of its small oscillations are second derivatives of its kinetic and potential energies
at rest, written with the indices \(j, k\) of the generalised coordinates; their
characteristic equation gives \(\omega^2 = (2 \mp \sqrt{2})\, g/l\).

```latex
T = m l^2 \dot\phi_1^2 + \frac{1}{2} m l^2 \dot\phi_2^2 + m l^2 \dot\phi_1 \dot\phi_2 \cos(\phi_1 - \phi_2)
U = -2 m g l \cos\phi_1 - m g l \cos\phi_2
M_{jk} = \left. \frac{\partial^2 T}{\partial \dot\phi_j \, \partial \dot\phi_k} \right|_{\phi = 0}, \quad K_{jk} = \left. \frac{\partial^2 U}{\partial \phi_j \, \partial \phi_k} \right|_{\phi = 0}
solve(\det(K - \omega^2 M) = 0, \omega^2)
```

```python
m, l, g, omega = symbols("m l g omega")
phi = symbols("phi_1 phi_2")
phi_dot = symbols("phi_dot_1 phi_dot_2")
T = (
    m * l**2 * phi_dot[0] ** 2
    + m * l**2 * phi_dot[1] ** 2 / 2
    + m * l**2 * phi_dot[0] * phi_dot[1] * cos(phi[0] - phi[1])
)
U = -2 * m * g * l * cos(phi[0]) - m * g * l * cos(phi[1])
rest = {phi[0]: 0, phi[1]: 0}
M = Matrix(2, 2, lambda j, k: diff(T, phi_dot[j], phi_dot[k]).subs(rest))
K = Matrix(2, 2, lambda j, k: diff(U, phi[j], phi[k]).subs(rest))
assert M == m * l**2 * Matrix([[2, 1], [1, 1]]) and K == m * g * l * Matrix([[2, 0], [0, 1]])
expected = solve(Eq((K - omega**2 * M).det(), 0), omega**2)
assert expected == [g * (2 - sqrt(2)) / l, g * (sqrt(2) + 2) / l]
```

## Lagrange's equations in index notation

**Level 4** · `cm.lagrangian.index-form` · standard

A free particle on a sphere, its kinetic energy a quadratic form in the velocities,
\(\tfrac12 m\, g_{ij} \dot q^i \dot q^j\), with the metric of the sphere evaluated along
the path \(q^i(t)\). Lagrange's equations, with an index on
\(\partial L / \partial \dot q^i\), are the geodesic equations of the sphere.

```latex
Coordinates := [\theta, \phi]
ds^2 = R^2 \left(d\theta^2 + \sin^2\theta\, d\phi^2\right)
q^i := (\theta(t), \phi(t))
L := \frac{1}{2} m\, g_{ij} \dot{q}^i \dot{q}^j
\frac{d}{dt} \frac{\partial L}{\partial \dot{q}^i} - \frac{\partial L}{\partial q^i}
```

```python
t, m, R = symbols("t m R")
theta, phi = Function("theta")(t), Function("phi")(t)
L = m * R**2 / 2 * (diff(theta, t) ** 2 + sin(theta) ** 2 * diff(phi, t) ** 2)
equations = [diff(diff(L, diff(q, t)), t) - diff(L, q) for q in (theta, phi)]
expected = [
    m * R**2 * (diff(theta, t, 2) - sin(theta) * cos(theta) * diff(phi, t) ** 2),
    m * R**2 * (sin(theta) ** 2 * diff(phi, t, 2) + 2 * sin(theta) * cos(theta) * diff(theta, t) * diff(phi, t)),
]
assert all(simplify(equation - e) == 0 for equation, e in zip(equations, expected))
```

## Noether's theorem for rotations

**Level 4** · `cm.lagrangian.noether` · Noether (1918); Landau and Lifshitz (1976), section 9

A Lagrangian invariant under rotations about the origin, \(\delta q_i = \epsilon K_i\)
with \(K = (-y, x)\), and its Noether charge
\(\sum_i (\partial L / \partial \dot q_i) K_i\): the angular momentum
\(m (x \dot y - y \dot x)\).

```latex
L = \frac{1}{2} m \left(\dot x^2 + \dot y^2\right) - V\left(\sqrt{x^2 + y^2}\right)
q = (x, y), \quad K = (-y, x)
\sum_i \frac{\partial L}{\partial \dot q_i} K_i
```

```python
t, m = symbols("t m")
x, y = Function("x")(t), Function("y")(t)
V = Function("V")
L = m / 2 * (diff(x, t) ** 2 + diff(y, t) ** 2) - V(sqrt(x**2 + y**2))
charge = sum(diff(L, diff(q, t)) * K for q, K in zip((x, y), (-y, x)))
expected = simplify(charge)
assert simplify(expected - m * (x * diff(y, t) - y * diff(x, t))) == 0
# it is conserved on the equations of motion
accelerations = {
    diff(q, t, 2): solve(diff(diff(L, diff(q, t)), t) - diff(L, q), diff(q, t, 2))[0]
    for q in (x, y)
}
assert simplify(diff(charge, t).subs(accelerations)) == 0
```

## Hamilton's principle

**Level 5** · `cm.lagrangian.hamilton-principle` · Goldstein (2002), section 2.3; Landau and Lifshitz (1976), section 2

The action as a functional of the path, and its functional derivative: the equation of
motion of the oscillator, \(\delta S / \delta q(t) = -m \ddot q - k q\). No design of the
language reads functionals or their derivatives yet.

```latex
S[q] = \int_{t_1}^{t_2} \left( \frac{1}{2} m \dot q^2 - \frac{1}{2} k q^2 \right) dt
\frac{\delta S}{\delta q(t)}
```

```python
t, m, k, epsilon = symbols("t m k epsilon")
q = Function("q")(t)
lagrangian = m / 2 * diff(q, t) ** 2 - k / 2 * q**2
expected = diff(lagrangian, q) - diff(diff(lagrangian, diff(q, t)), t)
assert simplify(expected - (-m * diff(q, t, 2) - k * q)) == 0
# The definition: the first variation of S along any eta vanishing at the ends is the
# integral of (delta S / delta q) eta. Checked for one path and one variation on [0, 1].
path, eta = sin(3 * t) + t**2, t**2 * (1 - t)
values = {m: 2, k: 5}


def action(curve):
    return integrate(lagrangian.subs(values).subs(q, curve).doit(), (t, 0, 1))


variation = diff(action(path + epsilon * eta), epsilon).subs(epsilon, 0)
pairing = integrate((expected.subs(values).subs(q, path).doit()) * eta, (t, 0, 1))
assert abs(N(variation - pairing)) < 1e-12
```
