# Viscous flow

Steady flows of a viscous fluid that solve the Navier–Stokes equations exactly, plane
Couette flow and Poiseuille flow in a pipe, the Reynolds number that says when such flows
are seen, and Stokes's drag on a slowly moving sphere. Landau and Lifshitz (1987) is
*Fluid Mechanics*, 2nd edition.

## The Reynolds number of water in a pipe

**Level 1** · `fluid.viscous.reynolds-number` · Reynolds (1883), Phil. Trans. R. Soc. 174, 935

\(\rho U d / \mu\) for water at 20 °C (\(\rho = 998.2\) kg/m³, \(\mu = 1.002 \times
10^{-3}\) Pa s) flowing at 0.1 m/s through a pipe 2 cm across: about 1992, just below the
value of about 2000 above which flow in a pipe usually turns turbulent. One expression, its
values given by an evaluation bar.

```latex
\frac{\rho U d}{\mu} \Big|_{\rho = 998.2,\ U = 0.1,\ d = 0.02,\ \mu = 1.002 \times 10^{-3}}
```

```python
rho, U, d, mu = 998.2, 0.1, 0.02, 1.002e-3
expected = rho * U * d / mu
nu = mu / rho  # the kinematic viscosity, about 1.004e-6 m^2/s
assert abs(U * d / nu - expected) < 1e-9
assert abs(expected - 1992.4) < 0.1
```

## The flow rate of Poiseuille flow

**Level 1** · `fluid.viscous.flow-rate` · Landau and Lifshitz (1987), §17

The volume crossing a section of a pipe of radius \(a\) each second: the parabolic profile
\(u = G(a^2 - r^2)/4\mu\), with \(G\) the fall of pressure per unit length, integrated over
the cross-section. Poiseuille's law, \(Q = \pi G a^4 / 8\mu\).

```latex
\int_0^a \frac{G}{4\mu}\left(a^2 - r^2\right) 2\pi r \, dr
```

```python
from scipy.integrate import quad

r, a, G, mu = symbols("r a G mu", positive=True)
expected = integrate(G / (4 * mu) * (a**2 - r**2) * 2 * pi * r, (r, 0, a))
assert expected == pi * G * a**4 / (8 * mu)
# and by quadrature, at G = 3, mu = 0.7, a = 1.3
numerical = quad(lambda s: 3 / (4 * 0.7) * (1.3**2 - s**2) * 2 * np.pi * s, 0, 1.3)[0]
assert abs(numerical - float(expected.subs({G: 3, mu: 0.7, a: 1.3}))) < 1e-12
```

## Plane Couette flow

**Level 2** · `fluid.viscous.couette` · Landau and Lifshitz (1987), §17

Fluid between a plate at rest at \(y = 0\) and a plate sliding at speed \(U\) at
\(y = h\), with no pressure gradient along the plates: the Navier–Stokes equations reduce
to \(\mu u'' = 0\), and with no slip at both plates the velocity grows linearly across the
gap, \(u = Uy/h\). The unknown is named without its argument, and the two boundary
conditions, at 0 and at a symbolic \(h\), sit beside the equation.

```latex
\mu, h > 0
dsolve\left(\mu \frac{d^2 u}{dy^2} = 0, u(0) = 0, u(h) = U\right)
```

```python
y, U = symbols("y U")
mu, h = symbols("mu h", positive=True)
u = Function("u")
expected = dsolve(Eq(mu * u(y).diff(y, 2), 0), u(y), ics={u(0): 0, u(h): U})
assert expected.rhs == U * y / h
profile = expected.rhs  # it solves the equation and meets both conditions
assert diff(profile, y, 2) == 0 and profile.subs(y, 0) == 0 and profile.subs(y, h) == U
```

## The terminal velocity of a small sphere

**Level 2** · `fluid.viscous.terminal-velocity` · Stokes (1851), Trans. Cambridge Philos. Soc. 9, 8

A sphere of radius \(a\) and density \(\rho_s\) settling slowly through a fluid of density
\(\rho\) and viscosity \(\mu\) falls at the speed at which Stokes's drag \(6\pi\mu a U\)
balances its weight less its buoyancy: \(U = 2a^2(\rho_s - \rho)g/9\mu\).

```latex
\mu, a > 0
solve\left(6\pi\mu a U = \frac{4}{3}\pi a^3 (\rho_s - \rho) g, U\right)
```

```python
mu, a = symbols("mu a", positive=True)
U, rho_s, rho, g = symbols("U rho_s rho g")
expected = solve(Eq(6 * pi * mu * a * U, Rational(4, 3) * pi * a**3 * (rho_s - rho) * g), U)
assert len(expected) == 1
assert simplify(expected[0] - 2 * a**2 * (rho_s - rho) * g / (9 * mu)) == 0
# a grain of silt of radius 10 micrometres in water: about 0.36 mm/s, and the forces balance
speed = 2 * (1e-5) ** 2 * (2650 - 1000) * 9.81 / (9 * 1.0e-3)
drag = 6 * np.pi * 1.0e-3 * 1e-5 * speed
weight_less_buoyancy = 4 / 3 * np.pi * (1e-5) ** 3 * (2650 - 1000) * 9.81
assert abs(drag / weight_less_buoyancy - 1) < 1e-12 and abs(speed - 3.6e-4) < 1e-5
assert 1000 * speed * 2e-5 / 1.0e-3 < 0.01  # a Reynolds number far below 1, where the law holds
```

## The drag coefficient of Stokes flow

**Level 3** · `fluid.viscous.drag-coefficient` · standard

Stokes's drag on a sphere, written as a drag coefficient, the force over \(\frac12\rho U^2\)
times the frontal area, is \(C_D = 24/\mathrm{Re}\), with the Reynolds number of the
diameter. \(\mathrm{Re}\) in roman type is the Reynolds number, not the real part.

```latex
F := 6\pi\mu a U
d := 2a
\mathrm{Re} := \frac{\rho U d}{\mu}
C_D := \frac{F}{\frac{1}{2}\rho U^2 \pi a^2}
C_D = \frac{24}{\mathrm{Re}}
```

```python
mu, a, U, rho = symbols("mu a U rho", positive=True)
F = 6 * pi * mu * a * U
Re = rho * U * 2 * a / mu
C_D = F / (rho * U**2 * pi * a**2 / 2)
assert simplify(C_D - 24 / Re) == 0
assert C_D.subs({mu: 1, a: 1, U: 1, rho: 1}) == 12  # 24/Re with Re = 2
expected = True
```

## The mean speed of Poiseuille flow

**Level 3** · `fluid.viscous.mean-speed` · Landau and Lifshitz (1987), §17

From the profile to the flow rate, and to the mean speed \(\bar u = Q/\pi a^2\), a name with
an accent defined by `:=`: the mean speed is half the speed on the axis.

```latex
u(r) := \frac{G}{4\mu}\left(a^2 - r^2\right)
Q := \int_0^a u(r)\, 2\pi r \, dr
\bar{u} := \frac{Q}{\pi a^2}
\frac{\bar{u}}{u(0)}
```

```python
r, a, G, mu = symbols("r a G mu", positive=True)
u = G / (4 * mu) * (a**2 - r**2)
Q = integrate(u * 2 * pi * r, (r, 0, a))
expected = simplify(Q / (pi * a**2) / u.subs(r, 0))
assert expected == Rational(1, 2)
# the volume under a paraboloid of revolution is half that of its cylinder
assert integrate((1 - (r / a) ** 2) * 2 * pi * r, (r, 0, a)) == pi * a**2 / 2
```

## The profile of Poiseuille flow

**Level 4** · `fluid.viscous.poiseuille-profile` · Landau and Lifshitz (1987), §17

In a pipe of radius \(a\) the axial velocity solves \(\frac{\mu}{r}(r u')' = -G\). Its
general solution has a term in \(\ln r\); the velocity must stay finite on the axis, a
condition written \(|u(0)| < \infty\), which removes it, and no slip at the wall then gives
\(u = G(a^2 - r^2)/4\mu\).

```latex
a, \mu > 0
dsolve\left(\frac{\mu}{r} \frac{d}{dr}\left(r \frac{du}{dr}\right) = -G, u(a) = 0, |u(0)| < \infty\right)
```

```python
r, a, mu = symbols("r a mu", positive=True)
G, C1, C2 = symbols("G C1 C2")
u = Function("u")
general = dsolve(Eq(mu / r * diff(r * diff(u(r), r), r), -G), u(r)).rhs
assert simplify(general - (C1 + C2 * log(r) - G * r**2 / (4 * mu))) == 0
assert limit(log(r), r, 0, "+") == -oo  # so a velocity finite on the axis needs C2 = 0
bounded = general.subs(C2, 0)
profile = bounded.subs(C1, solve(bounded.subs(r, a), C1)[0])
expected = Eq(u(r), profile)
assert simplify(profile - G * (a**2 - r**2) / (4 * mu)) == 0
assert simplify(mu / r * diff(r * diff(profile, r), r) + G) == 0 and profile.subs(r, a) == 0
```
