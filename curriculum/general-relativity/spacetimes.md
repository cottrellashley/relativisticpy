# Spacetimes in their usual coordinates

The metrics a course meets first, written as line elements in the coordinates books use:
flat spacetime in spherical and null coordinates, the uniformly accelerated frame, de
Sitter and anti-de Sitter space, a traversable wormhole, and Schwarzschild in coordinates
that cross its horizon.

## Minkowski space in spherical coordinates

**Level 1** · `gr.spacetimes.minkowski-spherical` · Carroll (2004), section 2.5

A component of flat spacetime's metric, written with the solid angle element.

```latex
ds^2 = -dt^2 + dr^2 + r^2 d\Omega^2
g_{\phi\phi}
```

```python
r, theta, phi = symbols("r theta phi")
# d\Omega^2 = d\theta^2 + \sin^2\theta\, d\phi^2: the coefficient of d\phi^2
expected = r**2 * sin(theta) ** 2
```

## Minkowski space in null coordinates

**Level 1** · `gr.spacetimes.minkowski-null` · standard

With \(u = t - x\) and \(v = t + x\), \(-dt^2 + dx^2 = -du\,dv\): a line element with a
cross term, whose metric component is half its coefficient.

```latex
ds^2 = -du\,dv + dy^2 + dz^2
g_{uv}
```

```python
u, v = symbols("u v")
t, x = (u + v) / 2, (v - u) / 2
J = Matrix([[diff(t, u), diff(t, v)], [diff(x, u), diff(x, v)]])
null = J.T * diag(-1, 1) * J  # the metric of (t, x) in the coordinates (u, v)
assert null == Matrix([[0, -Rational(1, 2)], [-Rational(1, 2), 0]])
expected = null[0, 1]
```

## The Rindler wedge is flat

**Level 2** · `gr.spacetimes.rindler-flat` · Rindler (1966), Am. J. Phys. 34, 1174

The frame of a uniformly accelerated observer has a metric that depends on \(x\), and
still no curvature: every component of the Riemann tensor vanishes.

```latex
ds^2 = -\alpha^2 x^2 dt^2 + dx^2 + dy^2 + dz^2
R^\rho{}_{\sigma\mu\nu} = 0
```

```python
t, x, y, z, alpha = symbols("t x y z alpha")
X = [t, x, y, z]
g = diag(-(alpha**2) * x**2, 1, 1, 1)
gi = g.inv()
n = 4
Gam = [
    [
        [
            sum(
                gi[a, d] * (diff(g[d, b], X[c]) + diff(g[d, c], X[b]) - diff(g[b, c], X[d]))
                for d in range(n)
            )
            / 2
            for c in range(n)
        ]
        for b in range(n)
    ]
    for a in range(n)
]
riemann = [
    simplify(
        diff(Gam[a][d][b], X[c])
        - diff(Gam[a][c][b], X[d])
        + sum(Gam[a][c][e] * Gam[e][d][b] - Gam[a][d][e] * Gam[e][c][b] for e in range(n))
    )
    for a in range(n)
    for b in range(n)
    for c in range(n)
    for d in range(n)
]
expected = all(component == 0 for component in riemann)
assert expected
```

## De Sitter space is maximally symmetric

**Level 2** · `gr.spacetimes.de-sitter-maximally-symmetric` · Carroll (2004), section 8.1

In the static patch of de Sitter space the Riemann tensor is built from the metric alone,
\(R_{\mu\nu\rho\sigma} = L^{-2}(g_{\mu\rho} g_{\nu\sigma} - g_{\mu\sigma} g_{\nu\rho})\).

```latex
ds^2 = -\left(1 - \frac{r^2}{L^2}\right) dt^2 + \left(1 - \frac{r^2}{L^2}\right)^{-1} dr^2 + r^2 d\Omega^2
R_{\mu\nu\rho\sigma} = \frac{1}{L^2} \left( g_{\mu\rho} g_{\nu\sigma} - g_{\mu\sigma} g_{\nu\rho} \right)
```

```python
t, r, theta, phi, L = symbols("t r theta phi L")
X = [t, r, theta, phi]
f = 1 - r**2 / L**2
g = diag(-f, 1 / f, r**2, r**2 * sin(theta) ** 2)
gi = g.inv()
n = 4
Gam = [
    [
        [
            sum(
                gi[a, d] * (diff(g[d, b], X[c]) + diff(g[d, c], X[b]) - diff(g[b, c], X[d]))
                for d in range(n)
            )
            / 2
            for c in range(n)
        ]
        for b in range(n)
    ]
    for a in range(n)
]


def riemann_up(a, b, c, d):  # R^a_{bcd}
    return (
        diff(Gam[a][d][b], X[c])
        - diff(Gam[a][c][b], X[d])
        + sum(Gam[a][c][e] * Gam[e][d][b] - Gam[a][d][e] * Gam[e][c][b] for e in range(n))
    )


expected = all(
    simplify(
        sum(g[m, a] * riemann_up(a, b, c, d) for a in range(n))
        - (g[m, c] * g[b, d] - g[m, d] * g[b, c]) / L**2
    )
    == 0
    for m in range(n)
    for b in range(n)
    for c in range(n)
    for d in range(n)
)
assert expected
```

## Anti-de Sitter space and a negative cosmological constant

**Level 2** · `gr.spacetimes.anti-de-sitter-lambda` · Carroll (2004), section 8.1

Global anti-de Sitter space solves Einstein's equation in vacuum with
\(\Lambda = -3/L^2\).

```latex
g_{\mu\nu} := AntiDeSitter(L)
\Lambda := -\frac{3}{L^2}
G_{\mu\nu} + \Lambda g_{\mu\nu} = 0
```

```python
t, r, theta, phi, L = symbols("t r theta phi L")
X = [t, r, theta, phi]
f = 1 + r**2 / L**2
g = diag(-f, 1 / f, r**2, r**2 * sin(theta) ** 2)
gi = g.inv()
n = 4
Gam = [
    [
        [
            sum(
                gi[a, d] * (diff(g[d, b], X[c]) + diff(g[d, c], X[b]) - diff(g[b, c], X[d]))
                for d in range(n)
            )
            / 2
            for c in range(n)
        ]
        for b in range(n)
    ]
    for a in range(n)
]
ricci = Matrix(
    n,
    n,
    lambda b, d: simplify(
        sum(
            diff(Gam[a][d][b], X[a])
            - diff(Gam[a][a][b], X[d])
            + sum(Gam[a][a][e] * Gam[e][d][b] - Gam[a][d][e] * Gam[e][a][b] for e in range(n))
            for a in range(n)
        )
    ),
)
scalar = simplify(sum(gi[b, d] * ricci[b, d] for b in range(n) for d in range(n)))
assert scalar == -12 / L**2
einstein = ricci - scalar * g / 2
expected = simplify(einstein - 3 * g / L**2) == zeros(n, n)
assert expected
```

## The energy density of a wormhole

**Level 2** · `gr.spacetimes.wormhole-energy-density` · Morris & Thorne (1988), Am. J. Phys. 56, 395

The simplest traversable wormhole, with throat radius \(b\) and proper radial distance
\(l\). Einstein's equation gives the energy density its throat needs, \(G_{tt} = 8\pi
\rho\), which is negative.

```latex
ds^2 = -dt^2 + dl^2 + (b^2 + l^2) d\Omega^2
G_{tt}
```

```python
t, l, theta, phi, b = symbols("t l theta phi b")
X = [t, l, theta, phi]
g = diag(-1, 1, b**2 + l**2, (b**2 + l**2) * sin(theta) ** 2)
gi = g.inv()
n = 4
Gam = [
    [
        [
            sum(
                gi[a, d] * (diff(g[d, B], X[c]) + diff(g[d, c], X[B]) - diff(g[B, c], X[d]))
                for d in range(n)
            )
            / 2
            for c in range(n)
        ]
        for B in range(n)
    ]
    for a in range(n)
]
ricci = Matrix(
    n,
    n,
    lambda B, d: simplify(
        sum(
            diff(Gam[a][d][B], X[a])
            - diff(Gam[a][a][B], X[d])
            + sum(Gam[a][a][e] * Gam[e][d][B] - Gam[a][d][e] * Gam[e][a][B] for e in range(n))
            for a in range(n)
        )
    ),
)
scalar = simplify(sum(gi[B, d] * ricci[B, d] for B in range(n) for d in range(n)))
expected = simplify(ricci[0, 0] - scalar * g[0, 0] / 2)
assert simplify(expected + b**2 / (b**2 + l**2) ** 2) == 0  # rho = -b^2 / (8 pi (b^2 + l^2)^2)
```

## Schwarzschild in ingoing Eddington–Finkelstein coordinates

**Level 3** · `gr.spacetimes.eddington-finkelstein` · Carroll (2004), section 5.6

The tortoise coordinate \(r^*\) and the null coordinate \(v = t + r^*\) take the metric
to coordinates that cross the horizon, where \(ds^2 = -(1 - 2M/r)\,dv^2 + 2\,dv\,dr +
r^2 d\Omega^2\). The case asks for a coordinate transformation.

```latex
ds^2 = -\left(1 - \frac{2M}{r}\right) dt^2 + \left(1 - \frac{2M}{r}\right)^{-1} dr^2 + r^2 d\Omega^2
r^* := r + 2M \ln\left(\frac{r}{2M} - 1\right)
v := t + r^*
Coordinates := [v, r, \theta, \phi]
g_{vr}
```

```python
v, r, M = symbols("v r M")
f = 1 - 2 * M / r
rstar = r + 2 * M * log(r / (2 * M) - 1)
t = v - rstar  # the old time coordinate as a function of (v, r)
J = Matrix([[diff(t, v), diff(t, r)], [0, 1]])  # d(t, r)/d(v, r)
metric = simplify(J.T * diag(-f, 1 / f) * J)
assert simplify(metric[0, 0] + f) == 0 and metric[1, 1] == 0
expected = metric[0, 1]
assert expected == 1
```
