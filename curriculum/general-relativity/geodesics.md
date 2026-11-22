# Geodesics and orbits

Free fall in curved spacetime: the effective potential of the Schwarzschild geometry, its
circular orbits and the innermost stable one, the geodesic equation in index notation,
and the two classic tests, the bending of light and the advance of the perihelion.

## The deflection of starlight by the Sun

**Level 1** · `gr.geodesics.light-deflection-sun` · Carroll (2004), section 7.3

Einstein's angle \(4GM/c^2 b\) for a ray grazing the Sun, in seconds of arc: one formula
in SI units, with numbers in scientific notation.

```latex
G := 6.674 \times 10^{-11}
M_\odot := 1.989 \times 10^{30}
R_\odot := 6.957 \times 10^{8}
c := 2.998 \times 10^{8}
\frac{4 G M_\odot}{c^2 R_\odot} \cdot \frac{180}{\pi} \cdot 3600
```

```python
G, M, R, c = 6.674e-11, 1.989e30, 6.957e8, 2.998e8
expected = 4 * G * M / (c**2 * R) * 180 / pi * 3600
assert abs(float(expected) - 1.75) < 0.01  # 1.75 arcseconds
tolerance = 1e-6
```

## The perihelion of Mercury

**Level 1** · `gr.geodesics.mercury-perihelion` · Will (2014), Living Rev. Relativ. 17, 4

The advance \(6\pi GM/c^2 a (1 - e^2)\) per orbit, over the 415 orbits of a century, in
seconds of arc: about 43. The eccentricity is \(e\), as astronomers write it.

```latex
G := 6.674 \times 10^{-11}
M_\odot := 1.989 \times 10^{30}
c := 2.998 \times 10^{8}
a := 5.791 \times 10^{10}
e := 0.2056
T := 87.969
\Delta\phi := \frac{6 \pi G M_\odot}{c^2 a (1 - e^2)}
\Delta\phi \cdot \frac{36525}{T} \cdot \frac{180}{\pi} \cdot 3600
```

```python
G, M, c, a, e, T = 6.674e-11, 1.989e30, 2.998e8, 5.791e10, 0.2056, 87.969
per_orbit = 6 * pi * G * M / (c**2 * a * (1 - e**2))
expected = per_orbit * 36525 / T * 180 / pi * 3600
assert abs(float(expected) - 42.98) < 0.05  # arcseconds per century
tolerance = 1e-6
```

## Circular orbits

**Level 2** · `gr.geodesics.circular-orbits` · Carroll (2004), section 5.4

A massive particle in the Schwarzschild geometry moves in the effective potential
\(V(r)\), per unit mass, with \(G = 1\). Circular orbits sit where it is extremal, at
\(r_c = (L^2 \pm \sqrt{L^4 - 12 M^2 L^2})/2M\).

```latex
V(r) := \frac{1}{2} - \frac{M}{r} + \frac{L^2}{2 r^2} - \frac{M L^2}{r^3}
solve(V'(r) = 0, r)
```

```python
M, L, r = symbols("M L r")
V = Rational(1, 2) - M / r + L**2 / (2 * r**2) - M * L**2 / r**3
expected = solve(diff(V, r), r)
stated = [
    (L**2 - sqrt(L**4 - 12 * M**2 * L**2)) / (2 * M),
    (L**2 + sqrt(L**4 - 12 * M**2 * L**2)) / (2 * M),
]
Lp, Mp = symbols("L M", positive=True)
assert all(simplify((a - b).subs({L: Lp, M: Mp})) == 0 for a, b in zip(expected, stated))
```

## The innermost stable circular orbit

**Level 3** · `gr.geodesics.isco` · Carroll (2004), section 5.4

The two circular orbits merge where \(V'(r) = V''(r) = 0\): at \(r = 6M\), with
\(L = 2\sqrt{3}\,M\).

```latex
V(r) := \frac{1}{2} - \frac{M}{r} + \frac{L^2}{2 r^2} - \frac{M L^2}{r^3}
solve([V'(r) = 0, V''(r) = 0], [r, L])
```

```python
M, L, r = symbols("M L r")
V = Rational(1, 2) - M / r + L**2 / (2 * r**2) - M * L**2 / r**3
expected = solve([diff(V, r), diff(V, r, 2)], [r, L])
assert expected == [(6 * M, -2 * sqrt(3) * M), (6 * M, 2 * sqrt(3) * M)]
```

## The energy of the last stable orbit

**Level 3** · `gr.geodesics.isco-energy` · Bardeen, Press & Teukolsky (1972), ApJ 178, 347, eq. (2.12)

A circular orbit turns at \(\Omega = \sqrt{M/r^3}\); its four-velocity, normalised, gives
the conserved energy \(E = -u_t\). At \(r = 6M\) it is \(\sqrt{8/9}\): matter falling in
from there has radiated 5.7% of its rest mass.

```latex
g_{\mu\nu} := Schwarzschild(M)
\Omega := \sqrt{\frac{M}{r^3}}
u^\mu := \frac{1}{\sqrt{1 - 3M/r}} \left[1, 0, 0, \Omega\right]
E := -u_t
E |_{r = 6M}
```

```python
M, r = symbols("M r", positive=True)
f = 1 - 2 * M / r
ut = 1 / sqrt(1 - 3 * M / r)
Omega = sqrt(M / r**3)
assert simplify(ut**2 * (-f + r**2 * Omega**2)) == -1  # normalised, in the equatorial plane
E = f * ut  # -u_t = -g_tt u^t
expected = simplify(E.subs(r, 6 * M))
assert simplify(expected - sqrt(Rational(8, 9))) == 0
```

## A circular orbit solves the geodesic equation

**Level 4** · `gr.geodesics.circular-geodesic` · Carroll (2004), section 3.3

The geodesic equation in index notation, for a worldline \(x^\mu(\tau)\) given
explicitly: the circular orbit at \(r_0\) in the equatorial plane, with
\(\Omega^2 = M/r_0^3\). The Christoffel symbols are evaluated along the curve.

```latex
g_{\mu\nu} := Schwarzschild(M)
\Omega := \sqrt{\frac{M}{r_0^3}}
\gamma := \frac{1}{\sqrt{1 - 3M/r_0}}
x^\mu(\tau) := \left[\gamma \tau, r_0, \frac{\pi}{2}, \gamma \Omega \tau\right]
\frac{d^2 x^\mu}{d\tau^2} + \Gamma^\mu_{\rho\sigma} \frac{dx^\rho}{d\tau} \frac{dx^\sigma}{d\tau} = 0
```

```python
t, r, theta, phi, M, r0, tau = symbols("t r theta phi M r_0 tau", positive=True)
X = [t, r, theta, phi]
f = 1 - 2 * M / r
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
ut = 1 / sqrt(1 - 3 * M / r0)
curve = [ut * tau, r0, pi / 2, ut * sqrt(M / r0**3) * tau]
velocity = [diff(x, tau) for x in curve]
on_curve = dict(zip(X, curve))
equation = [
    diff(curve[m], tau, 2)
    + sum(
        Gam[m][a][b].subs(on_curve) * velocity[a] * velocity[b] for a in range(n) for b in range(n)
    )
    for m in range(n)
]
expected = all(simplify(component) == 0 for component in equation)
assert expected
```

## The bending of light, to first order

**Level 5** · `gr.geodesics.light-bending` · Weinberg (1972), section 8.5

A ray that comes closest at \(r_0\) turns through the integral of its orbit, less \(\pi\).
To first order in \(M\) the deflection is \(4M/r_0\): the case asks for the asymptotic
expansion of an integral whose integrand and impact parameter both depend on \(M\).

```latex
b := \sqrt{\frac{r_0^3}{r_0 - 2M}}
\Delta\phi := 2 \int_{r_0}^{\infty} \frac{dr}{r^2 \sqrt{\frac{1}{b^2} - \frac{1}{r^2}\left(1 - \frac{2M}{r}\right)}} - \pi
\Delta\phi = \frac{4M}{r_0} + O(M^2)
```

```python
from scipy.integrate import quad


def deflection(M, r0=1.0):
    # u = 1/r, then u = u0 - w^2 removes the square-root singularity at the turning point
    u0 = 1 / r0
    F = lambda u: u0**2 * (1 - 2 * M * u0) - u**2 * (1 - 2 * M * u)
    value, _ = quad(
        lambda w: 2 * w / np.sqrt(F(u0 - w**2)),
        0,
        np.sqrt(u0),
        epsabs=1e-15,
        epsrel=1e-13,
        limit=200,
    )
    return 2 * value - np.pi


# (Δφ - 4M/r0)/M^2 tends to a constant, 15π/4 - 4: the remainder is O(M^2)
second = [(deflection(M) - 4 * M) / M**2 for M in (1e-3, 1e-4, 1e-5)]
assert all(abs(s - (15 * np.pi / 4 - 4)) < 0.05 for s in second)
expected = True
```

## The advance of the perihelion

**Level 5** · `gr.geodesics.perihelion-advance` · Carroll (2004), section 5.5

The orbit equation for \(u = 1/r\) has a relativistic term \(3Mu^2\). Solved by
perturbation theory, the orbit is an ellipse whose perihelion turns: \(\cos\) of
\((1 - \alpha)\phi\) rather than \(\phi\), with \(\alpha = 3M^2/L^2\), so that it advances
by \(6\pi M^2/L^2\) per orbit. The case asks the language to find \(\alpha\).

```latex
\frac{d^2 u}{d\phi^2} + u = \frac{M}{L^2} + 3 M u^2
u(\phi) \approx \frac{M}{L^2} \left[1 + e \cos\left((1 - \alpha)\phi\right)\right]
\alpha
```

```python
M, L = symbols("M L", positive=True)
# the circular orbit, u_c = M/L^2 + 3 M u_c^2, and small oscillations about it,
# delta'' + (1 - 6 M u_c) delta = 0: the perihelion returns after 2 pi / omega
u_c = (1 - sqrt(1 - 12 * M**2 / L**2)) / (6 * M)
assert simplify(u_c - M / L**2 - 3 * M * u_c**2) == 0
omega = sqrt(1 - 6 * M * u_c)
expected = series(1 - omega, M, 0, 3).removeO()  # to first order in M^2/L^2
assert simplify(expected - 3 * M**2 / L**2) == 0
```
