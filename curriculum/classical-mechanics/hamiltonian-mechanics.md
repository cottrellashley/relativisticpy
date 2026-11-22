# Hamiltonian mechanics

Phase space as mechanics books write it: Poisson brackets, the Legendre transform from
velocities to momenta, canonical transformations, action–angle variables and the
Hamilton–Jacobi equation.

## The Poisson brackets of angular momentum

**Level 2** · `cm.hamiltonian.angular-momentum` · Goldstein (2002), section 9.7

The Poisson bracket written with braces, over the canonical pairs \((x, p_x)\),
\((y, p_y)\) and \((z, p_z)\): the components of the angular momentum satisfy
\(\{L_x, L_y\} = L_z\).

```latex
L_x = y p_z - z p_y, \quad L_y = z p_x - x p_z, \quad L_z = x p_y - y p_x
\{L_x, L_y\} = L_z
```

```python
x, y, z, px, py, pz = symbols("x y z p_x p_y p_z")


def bracket(f, g):
    pairs = ((x, px), (y, py), (z, pz))
    return sum(diff(f, q) * diff(g, p) - diff(f, p) * diff(g, q) for q, p in pairs)


Lx, Ly, Lz = y * pz - z * py, z * px - x * pz, x * py - y * px
assert expand(bracket(Lx, Ly) - Lz) == 0
expected = True
```

## A canonical transformation of the oscillator

**Level 4** · `cm.hamiltonian.canonical-bracket` · Goldstein (2002), sections 9.3 and 9.5

The transformation that turns the oscillator's Hamiltonian into \(\omega P\) is
canonical: the bracket of \(q\) and \(p\), taken with respect to the new variables
\((Q, P)\) named in its subscript, is one.

```latex
q = \sqrt{\frac{2P}{m\omega}} \sin Q, \quad p = \sqrt{2 m \omega P} \cos Q
\{q, p\}_{Q, P} = 1
```

```python
Q, P, m, omega = symbols("Q P m omega", positive=True)
q = sqrt(2 * P / (m * omega)) * sin(Q)
p = sqrt(2 * m * omega * P) * cos(Q)
assert simplify(diff(q, Q) * diff(p, P) - diff(q, P) * diff(p, Q)) == 1
# and in the new variables the Hamiltonian is omega P
assert simplify(p**2 / (2 * m) + m * omega**2 * q**2 / 2 - omega * P) == 0
expected = True
```

## The Legendre transform of a relativistic particle

**Level 4** · `cm.hamiltonian.legendre` · Arnold (1989), section 14; Goldstein (2002), section 8.4

The Legendre transform written as convex analysis writes it, a maximum over the velocity:
the Hamiltonian of a free relativistic particle is \(\sqrt{p^2 c^2 + m^2 c^4}\).

```latex
L(v) = -m c^2 \sqrt{1 - \frac{v^2}{c^2}}
H(p) = \max_v \left(p v - L(v)\right)
H(p)
```

```python
v, p, m, c = symbols("v p m c", positive=True)
L = -m * c**2 * sqrt(1 - v**2 / c**2)
stationary = solve(Eq(diff(p * v - L, v), 0), v)
assert len(stationary) == 1  # and p v - L is concave in v, since L'' = m (1 - v^2/c^2)^(-3/2) > 0
assert simplify(diff(L, v, 2) - m / (1 - v**2 / c**2) ** Rational(3, 2)) == 0
expected = simplify((p * v - L).subs(v, stationary[0]))
assert simplify(expected - sqrt(p**2 * c**2 + m**2 * c**4)) == 0
```

## The action of an oscillator

**Level 4** · `cm.hamiltonian.action-oscillator` · Goldstein (2002), section 10.6

The action variable is the integral of \(p\, dq\) around the closed orbit of energy
\(E\) in phase space, \(\oint p\, dq = 2\pi E/\omega\): both branches of
\(p = \pm\sqrt{2mE - m^2\omega^2 q^2}\), between the turning points.

```latex
\frac{p^2}{2m} + \frac{1}{2} m \omega^2 q^2 = E
J = \oint p\, dq
J
```

```python
q, m, omega, E = symbols("q m omega E", positive=True)
turning = sqrt(2 * E / (m * omega**2))
momentum = sqrt(2 * m * E - m**2 * omega**2 * q**2)
expected = simplify(2 * integrate(momentum, (q, -turning, turning)))
assert simplify(expected - 2 * pi * E / omega) == 0
```

## The Hamilton–Jacobi equation of the oscillator

**Level 5** · `cm.hamiltonian.hamilton-jacobi` · Goldstein (2002), section 10.2

The Hamilton–Jacobi equation, a nonlinear partial differential equation for Hamilton's
principal function \(S\), separated as \(S = W(q) - \alpha t\); the constant \(\beta =
\partial S / \partial \alpha\) then gives the motion,
\(q = \sqrt{2\alpha / m\omega^2} \sin \omega (t + \beta)\).

```latex
\frac{1}{2m} \left(\frac{\partial S}{\partial q}\right)^2 + \frac{1}{2} m \omega^2 q^2 + \frac{\partial S}{\partial t} = 0
S = W(q) - \alpha t
solve\left(\beta = \frac{\partial S}{\partial \alpha}, q\right)
```

```python
q, t, m, omega, alpha, beta = symbols("q t m omega alpha beta", positive=True)
W_prime = sqrt(2 * m * alpha - m**2 * omega**2 * q**2)
# S = W(q) - alpha t solves the equation when W' is this
assert simplify(W_prime**2 / (2 * m) + m * omega**2 * q**2 / 2 - alpha) == 0
# dW/dalpha is the antiderivative in q of dW'/dalpha, up to a function of alpha alone,
# which shifts beta
dW_dalpha = asin(q * omega * sqrt(m / (2 * alpha))) / omega
assert simplify(diff(dW_dalpha, q) - diff(W_prime, alpha)) == 0
expected = solve(Eq(dW_dalpha - t, beta), q)
motion = sqrt(2 * alpha / (m * omega**2)) * sin(omega * (t + beta))
assert len(expected) == 1 and simplify(expected[0] - motion) == 0
```
