# The 3+1 split, null tetrads and conformal diagrams

The formalisms of specialist texts: spacetime sliced into space and time (the lapse, the
extrinsic curvature and the Hamiltonian constraint of ADM), the Newman–Penrose scalars of
a null tetrad, and the conformal compactification behind a Penrose–Carter diagram.

## The lapse of Painlevé–Gullstrand slices

**Level 2** · `gr.formalisms.painleve-gullstrand-lapse` · Poisson (2004), chapter 4

Schwarzschild spacetime sliced by the proper time of observers falling from rest at
infinity: written in the ADM form, with a square of differentials, the lapse
\(N = (-g^{tt})^{-1/2}\) is one.

```latex
ds^2 = -dt^2 + \left(dr + \sqrt{\frac{2M}{r}}\, dt\right)^2 + r^2 d\Omega^2
N := \frac{1}{\sqrt{-g^{tt}}}
N
```

```python
t, r, theta, phi, M = symbols("t r theta phi M", positive=True)
X = [t, r, theta, phi]
beta = sqrt(2 * M / r)  # the shift, N^r
g = Matrix([[-1 + beta**2, beta, 0, 0], [beta, 1, 0, 0], [0, 0, r**2, 0],
            [0, 0, 0, r**2 * sin(theta) ** 2]])
gi = simplify(g.inv())
expected = simplify(1 / sqrt(-gi[0, 0]))
assert expected == 1
# and it is Schwarzschild spacetime: a vacuum
n = 4
Gam = [[[sum(gi[a, d] * (diff(g[d, b], X[c]) + diff(g[d, c], X[b]) - diff(g[b, c], X[d]))
             for d in range(n)) / 2 for c in range(n)] for b in range(n)] for a in range(n)]
ricci = [simplify(sum(diff(Gam[a][d][b], X[a]) - diff(Gam[a][a][b], X[d])
                      + sum(Gam[a][a][e] * Gam[e][d][b] - Gam[a][d][e] * Gam[e][a][b] for e in range(n))
                      for a in range(n))) for b in range(n) for d in range(n)]
assert all(component == 0 for component in ricci)
```

## The mean curvature of a cosmological slice

**Level 4** · `gr.formalisms.extrinsic-curvature-trace` · Poisson (2004), chapter 3

The trace of the extrinsic curvature of the slices \(t = \text{const}\) is the divergence
of their unit normal, \(K = \nabla_\mu n^\mu\): for a flat FLRW universe, three times the
Hubble rate.

```latex
ds^2 = -dt^2 + a(t)^2 \left(dx^2 + dy^2 + dz^2\right)
n^\mu := [1, 0, 0, 0]
K := \nabla_\mu n^\mu
K
```

```python
t, x, y, z = symbols("t x y z")
a = Function("a")(t)
X = [t, x, y, z]
g = diag(-1, a**2, a**2, a**2)
root = sqrt(-g.det())  # divergence: (1/sqrt(-g)) d_mu (sqrt(-g) n^mu)
n_up = [1, 0, 0, 0]
expected = simplify(sum(diff(root * n_up[m], X[m]) for m in range(4)) / root)
assert simplify(expected - 3 * diff(a, t) / a) == 0
```

## The Coulomb part of the Weyl tensor

**Level 4** · `gr.formalisms.newman-penrose-psi2` · Teukolsky (1973), ApJ 185, 635

With Kinnersley's null tetrad, the only nonzero Weyl scalar of Schwarzschild spacetime is
\(\Psi_2 = C_{\mu\nu\rho\sigma} l^\mu m^\nu \bar m^\rho n^\sigma = -M/r^3\). The bar is
complex conjugation.

```latex
g_{\mu\nu} := Schwarzschild(M)
l^\mu := \left[\frac{1}{1 - 2M/r}, 1, 0, 0\right]
n^\mu := \frac{1}{2} \left[1, -\left(1 - \frac{2M}{r}\right), 0, 0\right]
m^\mu := \frac{1}{\sqrt{2}\, r} \left[0, 0, 1, \frac{i}{\sin\theta}\right]
\Psi_2 := C_{\mu\nu\rho\sigma} l^\mu m^\nu \bar{m}^\rho n^\sigma
\Psi_2
```

```python
t, r, theta, phi, M = symbols("t r theta phi M", positive=True)
X = [t, r, theta, phi]
f = 1 - 2 * M / r
g = diag(-f, 1 / f, r**2, r**2 * sin(theta) ** 2)
gi = g.inv()
n = 4
Gam = [[[sum(gi[a, d] * (diff(g[d, b], X[c]) + diff(g[d, c], X[b]) - diff(g[b, c], X[d]))
             for d in range(n)) / 2 for c in range(n)] for b in range(n)] for a in range(n)]

def riemann_up(a, b, c, d):  # R^a_{bcd}
    return (diff(Gam[a][d][b], X[c]) - diff(Gam[a][c][b], X[d])
            + sum(Gam[a][c][e] * Gam[e][d][b] - Gam[a][d][e] * Gam[e][c][b] for e in range(n)))

down = MutableDenseNDimArray.zeros(n, n, n, n)
for a in range(n):
    for b in range(n):
        for c in range(n):
            for d in range(n):
                down[a, b, c, d] = simplify(sum(g[a, e] * riemann_up(e, b, c, d) for e in range(n)))
ricci = [simplify(sum(riemann_up(a, b, a, d) for a in range(n))) for b in range(n) for d in range(n)]
assert all(component == 0 for component in ricci)  # a vacuum: the Weyl tensor is the Riemann tensor
l = [1 / f, 1, 0, 0]
nn = [Rational(1, 2), -f / 2, 0, 0]
m = [0, 0, 1 / (sqrt(2) * r), I / (sqrt(2) * r * sin(theta))]
mbar = [conjugate(component) for component in m]
dot = lambda u, v: simplify(sum(g[a, b] * u[a] * v[b] for a in range(n) for b in range(n)))
assert dot(l, nn) == -1 and dot(m, mbar) == 1 and dot(l, l) == 0 and dot(m, m) == 0
expected = simplify(sum(down[a, b, c, d] * l[a] * m[b] * mbar[c] * nn[d]
                        for a in range(n) for b in range(n) for c in range(n) for d in range(n)))
assert expected == -M / r**3
```

## The Hamiltonian constraint on a cosmological slice

**Level 5** · `gr.formalisms.hamiltonian-constraint` · Poisson (2004), chapter 4

The time–time part of Einstein's equation as a constraint on a slice:
\({}^{(3)}R + K^2 - K_{\mu\nu} K^{\mu\nu} = 16\pi\rho\), with \({}^{(3)}R\) the curvature
of the slice's own metric \(h_{\mu\nu}\). On the slices of FLRW it is the Friedmann
equation. The case asks for a second metric, induced on a hypersurface, and its
curvature.

```latex
ds^2 = -dt^2 + a(t)^2 \left(\frac{dr^2}{1 - k r^2} + r^2 d\Omega^2\right)
n^\mu := [1, 0, 0, 0]
h_{\mu\nu} := g_{\mu\nu} + n_\mu n_\nu
K_{\mu\nu} := h_\mu{}^\alpha \nabla_\alpha n_\nu
solve({}^{(3)}R + K^2 - K_{\mu\nu} K^{\mu\nu} = 16 \pi \rho, \rho)
```

```python
t, r, theta, phi, k, rho = symbols("t r theta phi k rho")
a = Function("a")(t)
# the slice t = const, with its own metric and curvature
Y = [r, theta, phi]
h = diag(a**2 / (1 - k * r**2), a**2 * r**2, a**2 * r**2 * sin(theta) ** 2)
hi = h.inv()
n = 3
Gam = [[[sum(hi[A, d] * (diff(h[d, b], Y[c]) + diff(h[d, c], Y[b]) - diff(h[b, c], Y[d]))
             for d in range(n)) / 2 for c in range(n)] for b in range(n)] for A in range(n)]
ricci3 = Matrix(n, n, lambda b, d: sum(
    diff(Gam[A][d][b], Y[A]) - diff(Gam[A][A][b], Y[d])
    + sum(Gam[A][A][e] * Gam[e][d][b] - Gam[A][d][e] * Gam[e][A][b] for e in range(n))
    for A in range(n)))
R3 = simplify(sum(hi[b, d] * ricci3[b, d] for b in range(n) for d in range(n)))
assert simplify(R3 - 6 * k / a**2) == 0
# K_ij = nabla_i n_j = -Gamma^t_ij n_t = (1/2) d_t h_ij, for n = d_t
K = h.diff(t) / 2
trace = simplify(sum(hi[i, j] * K[i, j] for i in range(n) for j in range(n)))
square = simplify(sum(hi[i, p] * hi[j, q] * K[i, j] * K[p, q]
                      for i in range(n) for j in range(n) for p in range(n) for q in range(n)))
expected = solve(Eq(R3 + trace**2 - square, 16 * pi * rho), rho)
assert len(expected) == 1
assert simplify(expected[0] - 3 * (diff(a, t) ** 2 + k) / (8 * pi * a**2)) == 0
```

## Minkowski space compactified

**Level 5** · `gr.formalisms.penrose-compactification` · Hawking & Ellis (1973), section 5.1

The coordinates of the Penrose–Carter diagram, \(t \pm r = \tan\frac{T \pm R}{2}\), bring
infinity to a finite place, and the metric becomes conformal to the Einstein static
universe: \(ds^2 = \omega^{-2}(-dT^2 + dR^2 + \sin^2 R\, d\Omega^2)\) with
\(\omega = 2\cos\frac{T+R}{2}\cos\frac{T-R}{2}\). The case asks for a coordinate
transformation given implicitly, with \(\pm\).

```latex
ds^2 = -dt^2 + dr^2 + r^2 d\Omega^2
t \pm r = \tan\frac{T \pm R}{2}
Coordinates := [T, R, \theta, \phi]
g_{RR}
```

```python
T, R = symbols("T R")
v, u = tan((T + R) / 2), tan((T - R) / 2)  # t + r and t - r
t, r = (v + u) / 2, (v - u) / 2
g_RR = simplify(-diff(t, R) ** 2 + diff(r, R) ** 2)
g_TT = simplify(-diff(t, T) ** 2 + diff(r, T) ** 2)
g_TR = simplify(-diff(t, T) * diff(t, R) + diff(r, T) * diff(r, R))
omega = 2 * cos((T + R) / 2) * cos((T - R) / 2)
assert simplify(g_RR - 1 / omega**2) == 0 and simplify(g_TT + 1 / omega**2) == 0 and g_TR == 0
assert simplify(r**2 - sin(R) ** 2 / omega**2) == 0
expected = 1 / omega**2
```
