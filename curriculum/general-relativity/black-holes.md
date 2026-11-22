# Black holes

Horizons and what they radiate: where the horizons of charged and rotating black holes
are, their surface gravity, their temperature and area, and the energy conditions that
ordinary matter obeys and a wormhole cannot.

## The Hawking temperature of a solar-mass black hole

**Level 1** · `gr.black-holes.hawking-temperature-sun` · Hawking (1975), Commun. Math. Phys. 43, 199

\(T = \hbar c^3 / 8\pi G M k_B\) for the mass of the Sun, in kelvin: about
\(6 \times 10^{-8}\), far colder than the microwave background.

```latex
\hbar := 1.0546 \times 10^{-34}
c := 2.998 \times 10^{8}
G := 6.674 \times 10^{-11}
k_B := 1.381 \times 10^{-23}
M_\odot := 1.989 \times 10^{30}
\frac{\hbar c^3}{8 \pi G M_\odot k_B}
```

```python
hbar, c, G, kB, M = 1.0546e-34, 2.998e8, 6.674e-11, 1.381e-23, 1.989e30
expected = hbar * c**3 / (8 * pi * G * M * kB)
assert abs(float(expected) / 6.17e-8 - 1) < 0.01
tolerance = 1e-6
```

## The horizons of a charged black hole

**Level 2** · `gr.black-holes.reissner-nordstrom-horizons` · Carroll (2004), section 6.5

The horizons of the Reissner–Nordström metric are where \(g^{rr}\) vanishes:
\(r_\pm = M \pm \sqrt{M^2 - Q^2}\).

```latex
ds^2 = -\left(1 - \frac{2M}{r} + \frac{Q^2}{r^2}\right) dt^2 + \left(1 - \frac{2M}{r} + \frac{Q^2}{r^2}\right)^{-1} dr^2 + r^2 d\Omega^2
solve(g^{rr} = 0, r)
```

```python
M, Q, r = symbols("M Q r")
expected = solve(1 - 2 * M / r + Q**2 / r**2, r)  # g^{rr} = 1/g_{rr}
stated = [M - sqrt(M**2 - Q**2), M + sqrt(M**2 - Q**2)]
assert all(simplify(a - b) == 0 for a, b in zip(expected, stated))
```

## The ergosphere of a rotating black hole

**Level 2** · `gr.black-holes.kerr-ergosphere` · Carroll (2004), section 6.6

The static limit of the Kerr metric, where \(\partial_t\) becomes null and nothing can
stay at rest: \(g_{tt} = 0\), at \(r = M + \sqrt{M^2 - a^2 \cos^2\theta}\), outside the
horizon except at the poles.

```latex
g_{\mu\nu} := Kerr(M, a)
solve(g_{tt} = 0, r)
```

```python
M, a, r, theta = symbols("M a r theta")
Sigma = r**2 + a**2 * cos(theta) ** 2
g_tt = -(1 - 2 * M * r / Sigma)  # Boyer-Lindquist coordinates
expected = solve(g_tt, r)
stated = [M - sqrt(M**2 - a**2 * cos(theta) ** 2), M + sqrt(M**2 - a**2 * cos(theta) ** 2)]
assert all(simplify(x - y) == 0 for x, y in zip(expected, stated))
```

## The temperature of a charged black hole

**Level 3** · `gr.black-holes.reissner-nordstrom-temperature` · Poisson (2004), chapter 5

For a metric \(-f\,dt^2 + f^{-1} dr^2 + r^2 d\Omega^2\) the surface gravity of the
horizon is \(\kappa = \frac{1}{2} f'(r_+)\), and the temperature \(\kappa / 2\pi\). For
Reissner–Nordström, \(\kappa = (r_+ - r_-)/2r_+^2\), which vanishes when \(|Q| = M\).

```latex
f(r) := 1 - \frac{2M}{r} + \frac{Q^2}{r^2}
r_+ := M + \sqrt{M^2 - Q^2}
\kappa := \frac{1}{2} f'(r_+)
T_H := \frac{\kappa}{2 \pi}
T_H
```

```python
M, Q, r = symbols("M Q r", positive=True)
f = 1 - 2 * M / r + Q**2 / r**2
r_plus, r_minus = M + sqrt(M**2 - Q**2), M - sqrt(M**2 - Q**2)
kappa = diff(f, r).subs(r, r_plus) / 2
assert simplify(kappa - (r_plus - r_minus) / (2 * r_plus**2)) == 0
expected = simplify(kappa / (2 * pi))
```

## The area of the Kerr horizon

**Level 3** · `gr.black-holes.kerr-horizon-area` · Poisson (2004), chapter 5

The horizon's area is the integral of \(\sqrt{g_{\theta\theta} g_{\phi\phi}}\) over a
section \(t = \text{const}\), \(r = r_+\): \(A = 4\pi(r_+^2 + a^2)\).

```latex
g_{\mu\nu} := Kerr(M, a)
r_+ := M + \sqrt{M^2 - a^2}
A := \int_0^{2\pi} \int_0^{\pi} \left. \sqrt{g_{\theta\theta}\, g_{\phi\phi}} \right|_{r = r_+} d\theta\, d\phi
A
```

```python
a, r, theta, rp = symbols("a r theta r_p", positive=True)
M = (rp**2 + a**2) / (2 * rp)  # the mass whose outer horizon is at r_p: Delta(r_p) = 0
Sigma = r**2 + a**2 * cos(theta) ** 2
g_thth = Sigma
g_phph = (r**2 + a**2 + 2 * M * a**2 * r * sin(theta) ** 2 / Sigma) * sin(theta) ** 2
on_horizon = simplify((g_thth * g_phph).subs(r, rp))
assert simplify(on_horizon - (rp**2 + a**2) ** 2 * sin(theta) ** 2) == 0
area = integrate((rp**2 + a**2) * sin(theta), (theta, 0, pi), (symbols("phi"), 0, 2 * pi))
Ms = symbols("M")
r_plus = Ms + sqrt(Ms**2 - a**2)
expected = area.subs(rp, r_plus)
assert simplify(expected - 4 * pi * (r_plus**2 + a**2)) == 0
```

## The surface gravity of a Killing horizon

**Level 4** · `gr.black-holes.surface-gravity-killing` · Carroll (2004), section 6.3

The surface gravity from the Killing vector \(\xi = \partial_t\) that is null on the
horizon: \(\kappa^2 = -\frac{1}{2} (\nabla_\mu \xi_\nu)(\nabla^\mu \xi^\nu)\) at
\(r = 2M\), which gives \(\kappa = 1/4M\).

```latex
g_{\mu\nu} := Schwarzschild(M)
\xi^\mu := [1, 0, 0, 0]
\kappa := \left. \sqrt{-\frac{1}{2} (\nabla_\mu \xi_\nu)(\nabla^\mu \xi^\nu)} \right|_{r = 2M}
\kappa
```

```python
t, r, theta, phi, M = symbols("t r theta phi M", positive=True)
X = [t, r, theta, phi]
g = diag(-(1 - 2 * M / r), 1 / (1 - 2 * M / r), r**2, r**2 * sin(theta) ** 2)
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
xi_down = [g[m, 0] for m in range(n)]  # xi_mu = g_{mu t}
D = Matrix(
    n, n, lambda m, v: diff(xi_down[v], X[m]) - sum(Gam[l][m][v] * xi_down[l] for l in range(n))
)
D_up = gi * D * gi  # nabla^mu xi^nu
square = simplify(sum(D[m, v] * D_up[m, v] for m in range(n) for v in range(n)))
assert simplify(square + 2 * M**2 / r**4) == 0
expected = simplify(sqrt(-square / 2).subs(r, 2 * M))
assert expected == 1 / (4 * M)
```

## The temperature from the Euclidean section

**Level 5** · `gr.black-holes.euclidean-period` · Gibbons & Hawking (1977), Phys. Rev. D 15, 2752

With \(t = -i\tau\) the Schwarzschild metric is Riemannian, and smooth at \(r = 2M\) only
if \(\tau\) is periodic, with period \(\beta = 8\pi M\): the inverse Hawking temperature.
The case asks the language to read an identification and find the period that removes a
conical singularity.

```latex
ds^2 = \left(1 - \frac{2M}{r}\right) d\tau^2 + \left(1 - \frac{2M}{r}\right)^{-1} dr^2 + r^2 d\Omega^2
\tau \sim \tau + \beta
\beta
```

```python
M, r, eps = symbols("M r epsilon", positive=True)
f = 1 - 2 * M / r
kappa = diff(f, r).subs(r, 2 * M) / 2
# near the horizon, with proper distance rho: f d\tau^2 + d\rho^2 = rho^2 (kappa d\tau)^2 + d\rho^2
rho = integrate(1 / sqrt(f.subs(r, 2 * M + eps).series(eps, 0, 2).removeO()), (eps, 0, eps))
assert simplify(f.subs(r, 2 * M + eps).series(eps, 0, 2).removeO() - (kappa * rho) ** 2) == 0
expected = 2 * pi / kappa  # kappa tau must have period 2 pi
assert simplify(expected - 8 * pi * M) == 0
```

## A wormhole violates the null energy condition

**Level 5** · `gr.black-holes.null-energy-condition` · Carroll (2004), section 4.6

The null energy condition, \(T_{\mu\nu} k^\mu k^\nu \ge 0\) for every null vector, is a
claim about all vectors at once. Through the Einstein tensor, it fails for the
traversable wormhole of Morris and Thorne: along a radial null ray,
\(G_{\mu\nu} k^\mu k^\nu = -2b^2/(b^2 + l^2)^2\).

```latex
ds^2 = -dt^2 + dl^2 + (b^2 + l^2) d\Omega^2
G_{\mu\nu} k^\mu k^\nu \geq 0 \quad \forall\, k^\mu : g_{\mu\nu} k^\mu k^\nu = 0
```

```python
t, l, theta, phi, b = symbols("t l theta phi b", positive=True)
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
einstein = simplify(ricci - scalar * g / 2)
k = Matrix([1, 1, 0, 0])  # a radial null vector
assert (k.T * g * k)[0] == 0
along = simplify((k.T * einstein * k)[0])
assert simplify(along + 2 * b**2 / (b**2 + l**2) ** 2) == 0
expected = False  # negative for every l: the condition fails
```
