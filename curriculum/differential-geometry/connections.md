# Connections and parallel transport

The Levi-Civita connection: its Christoffel symbols in curvilinear coordinates, the
transport of a vector along a curve, the Bianchi identity, and the spin connection of an
orthonormal frame from Cartan's structure equation.

## The Christoffel symbols of polar coordinates

**Level 2** · `dg.connections.polar-christoffel` · Carroll (2004), section 3.2

The flat plane in polar coordinates has a connection that does not vanish:
\(\Gamma^r_{\theta\theta} = -r\) and \(\Gamma^\theta_{r\theta} = 1/r\).

```latex
ds^2 = dr^2 + r^2 d\theta^2
\Gamma^r_{\theta\theta}
\Gamma^\theta_{r\theta}
```

```python
r, theta = symbols("r theta")
X = [r, theta]
g = diag(1, r**2)
gi = g.inv()
n = 2
Gam = [[[simplify(sum(gi[a, d] * (diff(g[d, b], X[c]) + diff(g[d, c], X[b]) - diff(g[b, c], X[d]))
                      for d in range(n)) / 2) for c in range(n)] for b in range(n)] for a in range(n)]
outputs = [Gam[0][1][1], Gam[1][0][1]]
assert outputs == [-r, 1 / r]
```

## Parallel transport around a circle of latitude

**Level 4** · `dg.connections.parallel-transport` · Carroll (2004), section 3.3

A vector carried around the circle \(\theta = \theta_0\) of the unit sphere obeys
\(\frac{dV^\mu}{d\lambda} + \Gamma^\mu_{\nu\rho} \frac{dx^\nu}{d\lambda} V^\rho = 0\).
It turns at the rate \(\cos\theta_0\) against the coordinate basis, so it comes back
rotated: the holonomy of the sphere.

```latex
ds^2 = d\theta^2 + \sin^2\theta\, d\phi^2
x^\mu(\lambda) := [\theta_0, \lambda]
V^\mu(\lambda) := \left[\cos(\lambda \cos\theta_0), -\frac{\sin(\lambda \cos\theta_0)}{\sin\theta_0}\right]
\frac{dV^\mu}{d\lambda} + \Gamma^\mu_{\nu\rho} \frac{dx^\nu}{d\lambda} V^\rho = 0
```

```python
theta, phi, theta0, lam = symbols("theta phi theta_0 lambda")
X = [theta, phi]
g = diag(1, sin(theta) ** 2)
gi = g.inv()
n = 2
Gam = [[[sum(gi[a, d] * (diff(g[d, b], X[c]) + diff(g[d, c], X[b]) - diff(g[b, c], X[d]))
             for d in range(n)) / 2 for c in range(n)] for b in range(n)] for a in range(n)]
curve = [theta0, lam]
V = [cos(lam * cos(theta0)), -sin(lam * cos(theta0)) / sin(theta0)]
on_curve = dict(zip(X, curve))
transport = [diff(V[m], lam) + sum(Gam[m][a][b].subs(on_curve) * diff(curve[a], lam) * V[b]
                                   for a in range(n) for b in range(n)) for m in range(n)]
expected = all(simplify(component) == 0 for component in transport)
assert expected
```

## The contracted Bianchi identity

**Level 4** · `dg.connections.contracted-bianchi` · Carroll (2004), section 3.7

The Einstein tensor is divergence-free for any metric, \(\nabla_\mu G^{\mu\nu} = 0\):
here a static, spherically symmetric one with two unknown functions.

```latex
ds^2 = -e^{2\alpha(r)} dt^2 + e^{2\beta(r)} dr^2 + r^2 d\Omega^2
\nabla_\mu G^{\mu\nu} = 0
```

```python
t, r, theta, phi = symbols("t r theta phi")
alpha, beta = Function("alpha")(r), Function("beta")(r)
X = [t, r, theta, phi]
g = diag(-exp(2 * alpha), exp(2 * beta), r**2, r**2 * sin(theta) ** 2)
gi = g.inv()
n = 4
Gam = [[[sum(gi[a, d] * (diff(g[d, b], X[c]) + diff(g[d, c], X[b]) - diff(g[b, c], X[d]))
             for d in range(n)) / 2 for c in range(n)] for b in range(n)] for a in range(n)]
ricci = Matrix(n, n, lambda b, d: sum(
    diff(Gam[a][d][b], X[a]) - diff(Gam[a][a][b], X[d])
    + sum(Gam[a][a][e] * Gam[e][d][b] - Gam[a][d][e] * Gam[e][a][b] for e in range(n))
    for a in range(n)))
scalar = sum(gi[b, d] * ricci[b, d] for b in range(n) for d in range(n))
G_up = gi * (ricci - scalar * g / 2) * gi
divergence = [sum(diff(G_up[m, v], X[m]) for m in range(n))
              + sum(Gam[m][m][l] * G_up[l, v] + Gam[v][m][l] * G_up[m, l]
                    for m in range(n) for l in range(n)) for v in range(n)]
expected = all(simplify(component) == 0 for component in divergence)
assert expected
```

## The spin connection of the sphere

**Level 4** · `dg.connections.spin-connection` · Carroll (2004), appendix J

With the orthonormal coframe \(e^1 = d\theta\), \(e^2 = \sin\theta\, d\phi\) of the unit
sphere, Cartan's first structure equation, without torsion, fixes the connection one-form:
\(\omega^1{}_2 = -\cos\theta\, d\phi\), whose components in \((\theta, \phi)\) are
\((0, -\cos\theta)\).

```latex
e^1 := d\theta
e^2 := \sin\theta\, d\phi
de^a + \omega^a{}_b \wedge e^b = 0
\omega^1{}_2
```

```python
theta, phi, A, B = symbols("theta phi A B")
X = [theta, phi]
e = [Matrix([1, 0]), Matrix([0, sin(theta)])]  # components of e^1 and e^2

def d(form):  # the exterior derivative of a one-form, as its (theta, phi) component
    return diff(form[1], theta) - diff(form[0], phi)

def wedge(u, v):  # the (theta, phi) component of u ^ v
    return u[0] * v[1] - u[1] * v[0]

# with an orthonormal frame, omega^1_2 = -omega^2_1: try omega^1_2 = A d theta + B d phi
omega12 = Matrix([A, B])
structure = [d(e[0]) + wedge(omega12, e[1]), d(e[1]) + wedge(-omega12, e[0])]
solution = solve(structure, [A, B], dict=True)
assert solution == [{A: 0, B: -cos(theta)}]
expected = [solution[0][A], solution[0][B]]
```
