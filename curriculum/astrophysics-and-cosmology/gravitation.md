# Gravitation and orbits

Newtonian gravitation as astronomers use it: the escape velocity, Kepler's third law
weighing a planet and placing a satellite, the gravitational energy of a uniform sphere, the
virial theorem on an elliptical orbit, and Kepler's equation for where a body is on its
orbit. Constants are in SI units; G is the CODATA 2018 value, and the Earth's equatorial
radius and gravitational parameter are the nominal values of IAU (2015) Resolution B3.

## The escape velocity of the Earth

**Level 1** · `astro.gravitation.escape-velocity` · standard; IAU (2015) Resolution B3

The speed that escapes from the Earth's surface, \(\sqrt{2GM_\oplus/R_\oplus}\), about
11.2 km/s, with the Earth's mass and radius named as astronomers name them, by the
symbol \(\oplus\).

```latex
G := 6.6743 \times 10^{-11}
M_\oplus := 5.9722 \times 10^{24}
R_\oplus := 6.3781 \times 10^{6}
\sqrt{\frac{2 G M_\oplus}{R_\oplus}}
```

```python
G, M, R = Float("6.6743e-11"), Float("5.9722e24"), Float("6.3781e6")  # SI units
expected = sqrt(2 * G * M / R)
assert abs(expected - 11179.94) < 0.01  # metres per second
# From the nominal GM of the Earth, 3.986004e14 m^3/s^2, of which M = GM / G is the mass
assert abs(np.sqrt(2 * 3.986004e14 / 6.3781e6) / float(expected) - 1) < 1e-5
```

## Weighing Jupiter with Io

**Level 1** · `astro.gravitation.jupiter-mass` · standard (Kepler's third law)

Kepler's third law, \(P^2 = 4\pi^2 a^3/GM\), gives the mass of the planet that Io
circles every 1.769138 days at 421 700 km: Jupiter's, \(1.90 \times 10^{27}\) kg.

```latex
G := 6.6743 \times 10^{-11}
a := 4.217 \times 10^{8}
P := 1.769138 \times 86\,400
M_J := \frac{4\pi^2 a^3}{G P^2}
M_J
```

```python
G, a, P = Float("6.6743e-11"), Float("4.217e8"), Float("1.769138") * 86400  # SI units
expected = 4 * pi**2 * a**3 / (G * P**2)
assert abs(float(expected) / 1.8985e27 - 1) < 1e-4  # kilograms
# Jupiter's mass from its GM, 1.26686534e17 m^3/s^2 (JPL), within 0.05 per cent: Io's own
# mass, 5e-5 of Jupiter's, and the rounding of its orbit are all the difference
assert abs(float(expected) / (1.26686534e17 / 6.6743e-11) - 1) < 5e-4
```

## A geostationary orbit

**Level 2** · `astro.gravitation.geostationary-orbit` · standard (Kepler's third law)

The orbit whose period is one sidereal day, 86 164.1 s, from Kepler's third law solved
for its radius: the one positive root of a cubic, 42 164 km from the Earth's centre. The
radius is declared positive first, as a book takes it; \(M_E\) is the Earth's mass, as
physics books write it.

```latex
a > 0
G := 6.6743 \times 10^{-11}
M_E := 5.9722 \times 10^{24}
T := 86\,164.1
solve\left(T^2 = \frac{4\pi^2 a^3}{G M_E}, a\right)
```

```python
G, M, T = Float("6.6743e-11"), Float("5.9722e24"), Float("86164.1")  # SI units
expected = [N((G * M * T**2 / (4 * pi**2)) ** Rational(1, 3), 15)]
assert abs(expected[0] - 42164247) < 1  # metres
a = symbols("a", positive=True)
roots = solve(Eq(T**2, 4 * pi**2 * a**3 / (G * M)), a)
assert len(roots) == 1  # the other two roots of the cubic are complex
assert abs(roots[0] / expected[0] - 1) < 1e-12
```

## The gravitational energy of a uniform sphere

**Level 2** · `astro.gravitation.binding-energy` · Carroll and Ostlie (2007), chapter 10

The gravitational energy of a star of uniform density, assembled shell by shell,
\(U = -4\pi G\int_0^R M_r\,\rho\,r\,dr = -\frac{3}{5}\frac{GM^2}{R}\), with \(M_r\), the
astronomers' name for the mass inside the radius \(r\).

```latex
\rho := \frac{3M}{4\pi R^3}
M_r := \frac{4}{3}\pi r^3 \rho
U := -4\pi G \int_0^R M_r \rho\, r \, dr
U
```

```python
G, M, R, r = symbols("G M R r", positive=True)
rho = 3 * M / (4 * pi * R**3)
M_r = Rational(4, 3) * pi * r**3 * rho
expected = -4 * pi * G * integrate(M_r * rho * r, (r, 0, R))
assert simplify(expected + 3 * G * M**2 / (5 * R)) == 0
# Half the integral of the density times the potential inside, -G M (3 R^2 - r^2) / 2 R^3
potential = -G * M * (3 * R**2 - r**2) / (2 * R**3)
assert simplify(integrate(rho * potential * 4 * pi * r**2, (r, 0, R)) / 2 - expected) == 0
```

## The virial theorem on an elliptical orbit

**Level 3** · `astro.gravitation.virial-ellipse` · Goldstein (2002), sections 3.4 and 3.8

Averages over time on a Kepler orbit, taken with the eccentric anomaly \(E\), in which
\(r = a(1 - \epsilon\cos E)\) and \(dt\) is proportional to \((1 - \epsilon\cos E)\,dE\).
With the speed from the vis-viva equation, \(v^2 = GM(2/r - 1/a)\), twice the mean kinetic
energy and the mean potential energy add to zero: the virial theorem.

```latex
a > 0
0 \leq \epsilon < 1
r(E) := a (1 - \epsilon \cos E)
\langle K \rangle := \frac{1}{2\pi} \int_0^{2\pi} \frac{m}{2} G M \left(\frac{2}{r(E)} - \frac{1}{a}\right) (1 - \epsilon \cos E) \, dE
\langle U \rangle := \frac{1}{2\pi} \int_0^{2\pi} \left(-\frac{G M m}{r(E)}\right) (1 - \epsilon \cos E) \, dE
2 \langle K \rangle + \langle U \rangle = 0
```

```python
a, G, M, m = symbols("a G M m", positive=True)
epsilon, E = symbols("epsilon E", real=True)
r = a * (1 - epsilon * cos(E))
weight = (1 - epsilon * cos(E)) / (2 * pi)  # dt / P, with dt = (P / 2 pi)(1 - epsilon cos E) dE
K = integrate(cancel(m / 2 * G * M * (2 / r - 1 / a) * weight), (E, 0, 2 * pi))
U = integrate(cancel(-G * M * m / r * weight), (E, 0, 2 * pi))
assert simplify(K - G * M * m / (2 * a)) == 0 and simplify(U + G * M * m / a) == 0
# The same averages over time itself: solve Kepler's equation at equally spaced mean
# anomalies, by Newton's method, on an orbit with a = G M = m = 1 and eccentricity 0.6
mean = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
eccentric = mean.copy()
for _ in range(60):
    eccentric -= (eccentric - 0.6 * np.sin(eccentric) - mean) / (1 - 0.6 * np.cos(eccentric))
radius = 1 - 0.6 * np.cos(eccentric)
assert abs(np.mean(1 / radius) - 1) < 1e-9  # <1/r> = 1/a, so <U> = -G M m / a
assert abs(np.mean(2 / radius - 1) - 1) < 1e-9  # <v^2> = G M / a, so <K> = G M m / 2a
expected = True
```

## Kepler's equation

**Level 4** · `astro.gravitation.kepler-equation` · Goldstein (2002), section 3.8; Murray and Dermott (1999), chapter 2

Where a body is on its orbit at a given time: Kepler's equation \(E - e\sin E = M\), for
the eccentric anomaly \(E\) at mean anomaly \(M = 1\) on an orbit of eccentricity
\(e = 0.5\). Celestial mechanics writes \(e\) for the eccentricity, a definition that here
replaces Euler's number, and the equation is transcendental: its one real root is found
numerically.

```latex
E \in \mathbb{R}
e := 0.5
M := 1
solve(E - e \sin E = M, E)
```

```python
from scipy.optimize import brentq
from scipy.special import jv

root = brentq(lambda x: x - 0.5 * np.sin(x) - 1, 0, np.pi)  # E - e sin E increases with E
expected = [Float(root)]
assert abs(root - 1.4987011335178484) < 1e-14
# Bessel's series for the eccentric anomaly, E = M + sum over n of 2 J_n(n e) sin(n M) / n
series = 1 + sum(2 / n * jv(n, n * 0.5) * np.sin(n) for n in range(1, 60))
assert abs(series - root) < 1e-12
```
