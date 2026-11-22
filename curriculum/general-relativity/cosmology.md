# Cosmology

The homogeneous, isotropic universe: the Hubble parameter, the critical density, the
Friedmann equations from the FLRW metric, the radiation era, the event horizon of de
Sitter expansion, and the conservation of energy.

## The Hubble parameter of the matter era

**Level 1** · `gr.cosmology.hubble-matter` · Carroll (2004), section 8.4

In a flat universe of dust the scale factor grows as \(t^{2/3}\); the Hubble parameter
\(\dot a / a\) is \(2/3t\). One expression, with Newton's dot for the time derivative.

```latex
a(t) := t^{2/3}
\frac{\dot a}{a}
```

```python
t = symbols("t")
a = t ** Rational(2, 3)
expected = simplify(diff(a, t) / a)
assert expected == 2 / (3 * t)
```

## The critical density

**Level 1** · `gr.cosmology.critical-density` · Carroll (2004), section 8.3

The density \(3H_0^2/8\pi G\) of a flat universe, for \(H_0 = 70\) km/s/Mpc, in kilograms
per cubic metre: about \(9.2 \times 10^{-27}\).

```latex
G := 6.674 \times 10^{-11}
H_0 := \frac{70 \times 10^{3}}{3.086 \times 10^{22}}
\rho_c := \frac{3 H_0^2}{8 \pi G}
\rho_c
```

```python
G = 6.674e-11
H0 = 70e3 / 3.086e22  # 70 km/s/Mpc, in 1/s
expected = 3 * H0**2 / (8 * pi * G)
assert abs(float(expected) / 9.2e-27 - 1) < 0.01
tolerance = 1e-6
```

## The Friedmann equation

**Level 3** · `gr.cosmology.friedmann-equations` · Carroll (2004), section 8.3

The FLRW metric filled with a perfect fluid at rest: the time component of Einstein's
equation, solved for the density, is the first Friedmann equation,
\(8\pi G \rho = 3(\dot a^2 + k)/a^2\).

```latex
ds^2 = -dt^2 + a(t)^2 \left(\frac{dr^2}{1 - k r^2} + r^2 d\Omega^2\right)
u^\mu := [1, 0, 0, 0]
T^{\mu\nu} := (\rho + p) u^\mu u^\nu + p g^{\mu\nu}
solve(G_{tt} = 8 \pi G T_{tt}, \rho)
```

```python
t, r, theta, phi, k, G, rho, p = symbols("t r theta phi k G rho p")
a = Function("a")(t)
X = [t, r, theta, phi]
g = diag(-1, a**2 / (1 - k * r**2), a**2 * r**2, a**2 * r**2 * sin(theta) ** 2)
gi = g.inv()
n = 4
Gam = [[[sum(gi[A, d] * (diff(g[d, b], X[c]) + diff(g[d, c], X[b]) - diff(g[b, c], X[d]))
             for d in range(n)) / 2 for c in range(n)] for b in range(n)] for A in range(n)]
ricci = Matrix(n, n, lambda b, d: simplify(sum(
    diff(Gam[A][d][b], X[A]) - diff(Gam[A][A][b], X[d])
    + sum(Gam[A][A][e] * Gam[e][d][b] - Gam[A][d][e] * Gam[e][A][b] for e in range(n))
    for A in range(n))))
scalar = simplify(sum(gi[b, d] * ricci[b, d] for b in range(n) for d in range(n)))
G_tt = simplify(ricci[0, 0] - scalar * g[0, 0] / 2)
T_tt = rho  # (rho + p) u_t u_t + p g_tt, with u_t = -1 and g_tt = -1
expected = solve(Eq(G_tt, 8 * pi * G * T_tt), rho)
assert len(expected) == 1
assert simplify(expected[0] - 3 * (diff(a, t) ** 2 + k) / (8 * pi * G * a**2)) == 0
```

## The radiation era

**Level 3** · `gr.cosmology.radiation-era` · Carroll (2004), section 8.4

Radiation dilutes as \(a^{-4}\), and a flat universe full of it expands as \(t^{1/2}\):
the scale factor, with its constant fixed, solves the Friedmann equation.

```latex
a(t) := \left(\frac{32 \pi G \rho_0}{3}\right)^{1/4} t^{1/2}
\rho := \frac{\rho_0}{a^4}
\left(\frac{\dot a}{a}\right)^2 = \frac{8 \pi G}{3} \rho
```

```python
t, G, rho0 = symbols("t G rho_0", positive=True)
a = (32 * pi * G * rho0 / 3) ** Rational(1, 4) * sqrt(t)
rho = rho0 / a**4
expected = simplify((diff(a, t) / a) ** 2 - 8 * pi * G * rho / 3) == 0
assert expected
```

## The event horizon of de Sitter expansion

**Level 3** · `gr.cosmology.de-sitter-horizon` · Rindler (1956), MNRAS 116, 662

In a universe expanding exponentially, light emitted now reaches only a finite proper
distance, however long one waits: the event horizon, at \(1/H\).

```latex
Positive := [H]
a(t) := e^{H t}
d_E := a(t) \int_t^\infty \frac{dt'}{a(t')}
d_E
```

```python
t, s, H = symbols("t s H", positive=True)
a = exp(H * t)
expected = simplify(a * integrate(1 / a.subs(t, s), (s, t, oo)))
assert expected == 1 / H
```

## The conservation of energy

**Level 4** · `gr.cosmology.continuity-equation` · Carroll (2004), section 8.3

The time component of \(\nabla_\mu T^{\mu\nu} = 0\) for a perfect fluid in FLRW is the
continuity equation, \(\dot\rho + 3 \frac{\dot a}{a} (\rho + p) = 0\). Its left-hand side:

```latex
ds^2 = -dt^2 + a(t)^2 \left(\frac{dr^2}{1 - k r^2} + r^2 d\Omega^2\right)
u^\mu := [1, 0, 0, 0]
T^{\mu\nu} := (\rho(t) + p(t)) u^\mu u^\nu + p(t) g^{\mu\nu}
\nabla_\mu T^{\mu t}
```

```python
t, r, theta, phi, k = symbols("t r theta phi k")
a, rho, p = Function("a")(t), Function("rho")(t), Function("p")(t)
X = [t, r, theta, phi]
g = diag(-1, a**2 / (1 - k * r**2), a**2 * r**2, a**2 * r**2 * sin(theta) ** 2)
gi = g.inv()
n = 4
Gam = [[[sum(gi[A, d] * (diff(g[d, b], X[c]) + diff(g[d, c], X[b]) - diff(g[b, c], X[d]))
             for d in range(n)) / 2 for c in range(n)] for b in range(n)] for A in range(n)]
u = [1, 0, 0, 0]
T = Matrix(n, n, lambda m, v: (rho + p) * u[m] * u[v] + p * gi[m, v])
nu = 0  # the t component of the divergence
divergence = sum(diff(T[m, nu], X[m]) for m in range(n)) + sum(
    Gam[m][m][l] * T[l, nu] + Gam[nu][m][l] * T[m, l] for m in range(n) for l in range(n))
expected = simplify(divergence)
assert simplify(expected - (diff(rho, t) + 3 * diff(a, t) / a * (rho + p))) == 0
```
