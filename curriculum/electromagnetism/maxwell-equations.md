# Maxwell's equations

Maxwell's equations in vector form and in tensor form, the gauge freedom of the
potential, and the plane waves the equations admit. Tensor cases use the signature
\((-,+,+,+)\), units with \(c = 1\), and \(A_\mu = (-\phi, \mathbf{A})\).

## Mismatched free indices

**Level 1** · `em.maxwell.mismatched-indices` · standard

Gauss's law in tensor form with the current written with the wrong free index: the left
side is free in \(\nu\), the right in \(\mu\). The notation is wrong and must be
refused.

```latex
ds^2 = -dt^2 + dx^2 + dy^2 + dz^2
A_\mu := (-\phi(x, y, z), 0, 0, 0)
J^\mu := (\rho(x, y, z), 0, 0, 0)
F_{\mu\nu} := \partial_\mu A_\nu - \partial_\nu A_\mu
\partial_\mu F^{\mu\nu} = J^\mu
```

```python
error = "IndexNotationError"
message = "free indices"
```

## Gauss's law inside a charged ball

**Level 2** · `em.maxwell.gauss` · Griffiths (2013), section 2.2

The divergence written with \(\nabla \cdot\), of the field inside a uniformly charged
ball, \(\mathbf{E} = \rho\, \mathbf{r} / 3\epsilon_0\): Gauss's law in differential
form.

```latex
\mathbf{r} := (x, y, z)
\mathbf{E} := \frac{\rho}{3 \epsilon_0} \mathbf{r}
\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}
```

```python
x, y, z, rho, epsilon_0 = symbols("x y z rho epsilon_0")
field = [rho / (3 * epsilon_0) * coordinate for coordinate in (x, y, z)]
divergence = diff(field[0], x) + diff(field[1], y) + diff(field[2], z)
assert simplify(divergence - rho / epsilon_0) == 0
expected = True
```

## Gauge invariance of the field tensor

**Level 3** · `em.maxwell.gauge-invariance` · Jackson (1999), section 6.3

A general potential, a gauge transformation \(A'_\mu = A_\mu + \partial_\mu \chi\), and
the claim that the field tensor does not change.

```latex
ds^2 = -dt^2 + dx^2 + dy^2 + dz^2
A_\mu := (-\phi(t, x, y, z), A_x(t, x, y, z), A_y(t, x, y, z), A_z(t, x, y, z))
A'_\mu := A_\mu + \partial_\mu \chi(t, x, y, z)
\partial_\mu A'_\nu - \partial_\nu A'_\mu = \partial_\mu A_\nu - \partial_\nu A_\mu
```

```python
X = symbols("t x y z")
potential = [-Function("phi")(*X)] + [Function(name)(*X) for name in ("A_x", "A_y", "A_z")]
chi = Function("chi")(*X)
gauged = [potential[mu] + diff(chi, X[mu]) for mu in range(4)]


def field(A):
    return Matrix(4, 4, lambda mu, nu: diff(A[nu], X[mu]) - diff(A[mu], X[nu]))


assert simplify(field(gauged) - field(potential)) == zeros(4, 4)
expected = True
```

## The dispersion relation of light

**Level 3** · `em.maxwell.dispersion` · Jackson (1999), section 7.1

A plane wave polarised along \(y\), its field tensor, and the vacuum Maxwell equations
\(\partial_\mu F^{\mu\nu} = 0\): they hold only if \(\omega = \pm k\).

```latex
ds^2 = -dt^2 + dx^2 + dy^2 + dz^2
A_\mu := (0, 0, a e^{i(k x - \omega t)}, 0)
F_{\mu\nu} := \partial_\mu A_\nu - \partial_\nu A_\mu
solve(\partial_\mu F^{\mu y} = 0, \omega)
```

```python
X = symbols("t x y z")
t, x = X[0], X[1]
a, k, omega = symbols("a k omega")
eta = diag(-1, 1, 1, 1)
A = [0, 0, a * exp(I * (k * x - omega * t)), 0]
F = Matrix(4, 4, lambda mu, nu: diff(A[nu], X[mu]) - diff(A[mu], X[nu]))
F_up = eta * F * eta
divergence = [sum(diff(F_up[mu, nu], X[mu]) for mu in range(4)) for nu in range(4)]
assert divergence[0] == divergence[1] == divergence[3] == 0
expected = solve(Eq(divergence[2], 0), omega)
assert expected == [-k, k]
```

## The homogeneous Maxwell equations hold identically

**Level 4** · `em.maxwell.bianchi` · Carroll (2004), section 1.8

Antisymmetrisation brackets that span a derivative and a tensor:
\(\partial_{[\lambda} F_{\mu\nu]} = 0\) holds for every potential.

```latex
ds^2 = -dt^2 + dx^2 + dy^2 + dz^2
A_\mu := (-\phi(t, x, y, z), A_x(t, x, y, z), A_y(t, x, y, z), A_z(t, x, y, z))
F_{\mu\nu} := \partial_\mu A_\nu - \partial_\nu A_\mu
\partial_{[\lambda} F_{\mu\nu]} = 0
```

```python
X = symbols("t x y z")
A = [-Function("phi")(*X)] + [Function(name)(*X) for name in ("A_x", "A_y", "A_z")]
F = Matrix(4, 4, lambda mu, nu: diff(A[nu], X[mu]) - diff(A[mu], X[nu]))
for l in range(4):
    for m in range(4):
        for n in range(4):
            cyclic = diff(F[m, n], X[l]) + diff(F[n, l], X[m]) + diff(F[l, m], X[n])
            assert expand(cyclic) == 0
expected = True
```
