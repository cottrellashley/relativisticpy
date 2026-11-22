# Stellar structure

A star in hydrostatic equilibrium: the pressure that holds up a star of uniform density,
the luminosity at which the pressure of its radiation balances its gravity, and the
polytropes of the Lane–Emden equation,
\(\frac{1}{\xi^2}\frac{d}{d\xi}\left(\xi^2\frac{d\theta}{d\xi}\right) = -\theta^n\), with
its exact solutions for \(n = 0\), 1 and 5, and the polytrope of index 3 behind the
Chandrasekhar mass. Constants are CODATA 2018 values; the Sun's luminosity is the nominal
value of IAU (2015) Resolution B3, and its mass the nominal \((GM)_\odot\) of that
resolution divided by G.

## The Eddington luminosity of the Sun

**Level 1** · `astro.stars.eddington-luminosity` · Carroll and Ostlie (2007), chapter 10

The luminosity at which the pressure of radiation on the electrons of ionised hydrogen
balances gravity, \(4\pi G M m_p c/\sigma_T\), for one solar mass, in solar luminosities:
about 33 000. Names with the solar symbol \(\odot\) and a Greek letter with a label,
\(\sigma_T\), the Thomson cross-section.

```latex
G := 6.6743 \times 10^{-11}
M_\odot := 1.98841 \times 10^{30}
m_p := 1.67262192 \times 10^{-27}
c := 299\,792\,458
\sigma_T := 6.6524587 \times 10^{-29}
L_\odot := 3.828 \times 10^{26}
\frac{4\pi G M_\odot m_p c}{\sigma_T L_\odot}
```

```python
G, M_sun, m_p = 6.6743e-11, 1.98841e30, 1.67262192e-27
c, sigma_T, L_sun = 299792458, 6.6524587e-29, 3.828e26  # SI units
expected = Float(4 * np.pi * G * M_sun * m_p * c / (sigma_T * L_sun))
assert abs(expected - 32838.7) < 0.1
# With the nominal GM of the Sun, 1.3271244e20 m^3/s^2: 1.257e31 W, the same within 1e-6
eddington = 4 * np.pi * 1.3271244e20 * m_p * c / sigma_T
assert abs(eddington / 1.2571e31 - 1) < 1e-4 and abs(eddington / L_sun / expected - 1) < 1e-6
```

## Hydrostatic equilibrium in a uniform star

**Level 2** · `astro.stars.hydrostatic-pressure` · Carroll and Ostlie (2007), chapter 10

The equation of hydrostatic equilibrium, \(dP/dr = -GM_r\rho/r^2\), solved for a star of
uniform density with no pressure at its surface:
\(P = \frac{3GM^2}{8\pi R^4}\left(1 - \frac{r^2}{R^2}\right)\), whose value at the centre is
the central pressure. The unknown is named without its argument, as books write it.

```latex
\rho := \frac{3M}{4\pi R^3}
M_r := \frac{4}{3}\pi r^3 \rho
dsolve\left(\frac{dP}{dr} = -\frac{G M_r \rho}{r^2}, P(R) = 0\right)
```

```python
G, M, R, r, s = symbols("G M R r s", positive=True)
P = Function("P")
rho = 3 * M / (4 * pi * R**3)
M_r = Rational(4, 3) * pi * r**3 * rho
expected = dsolve(Eq(P(r).diff(r), -G * M_r * rho / r**2), P(r), ics={P(R): 0})
profile = 3 * G * M**2 / (8 * pi * R**4) * (1 - r**2 / R**2)
assert simplify(expected.rhs - profile) == 0
# Integrating inward from the surface instead: P(r) = int_r^R G M_s rho / s^2 ds
assert simplify(integrate((G * M_r * rho / r**2).subs(r, s), (s, r, R)) - profile) == 0
assert profile.subs(r, 0) == 3 * G * M**2 / (8 * pi * R**4)  # the central pressure
```

## The polytrope of index one

**Level 2** · `astro.stars.lane-emden-n1` · Chandrasekhar (1939), chapter IV

\(\theta = \sin\xi/\xi\) solves the Lane–Emden equation of index 1, written as the books
write it, with \(\theta\) named without its argument after it is defined.

```latex
\xi > 0
\theta(\xi) := \frac{\sin\xi}{\xi}
\frac{1}{\xi^2} \frac{d}{d\xi}\left(\xi^2 \frac{d\theta}{d\xi}\right) = -\theta
```

```python
from scipy.integrate import solve_ivp

xi = symbols("xi", positive=True)
theta = sin(xi) / xi
assert simplify(diff(xi**2 * diff(theta, xi), xi) / xi**2 + theta) == 0
# Numerically: integrate the equation outward from the series 1 - xi^2/6 near the centre
start, points = 1e-4, np.array([1.0, 2.0, 3.0])
solution = solve_ivp(
    lambda x, y: [y[1], -y[0] - 2 * y[1] / x],
    (start, 3),
    [1 - start**2 / 6, -start / 3],
    t_eval=points,
    rtol=1e-11,
    atol=1e-13,
)
assert np.allclose(solution.y[0], np.sin(points) / points, rtol=0, atol=1e-9)
expected = True
```

## The polytrope of index five has a finite mass

**Level 2** · `astro.stars.lane-emden-n5-mass` · Chandrasekhar (1939), chapter IV

The solution of index 5, \(\theta = (1 + \xi^2/3)^{-1/2}\), never vanishes, so the
polytrope has no surface; yet its mass, proportional to \(-\xi^2\theta'\) at the edge, is
finite: the limit is \(\sqrt{3}\).

```latex
\theta(\xi) := \left(1 + \frac{\xi^2}{3}\right)^{-1/2}
\lim_{\xi \to \infty} \left(-\xi^2 \theta'(\xi)\right)
```

```python
xi = symbols("xi", positive=True)
theta = (1 + xi**2 / 3) ** Rational(-1, 2)
assert simplify(diff(xi**2 * diff(theta, xi), xi) / xi**2 + theta**5) == 0  # index 5
expected = limit(-(xi**2) * diff(theta, xi), xi, oo)
assert expected == sqrt(3)
assert abs(float((-(xi**2) * diff(theta, xi)).subs(xi, 10**6)) - np.sqrt(3)) < 1e-9
```

## The central concentration of a polytrope of index one

**Level 3** · `astro.stars.density-contrast` · Chandrasekhar (1939), chapter IV

From the solution \(\theta = \sin\xi/\xi\), whose first zero \(\xi_1 = \pi\) is the
star's surface, to its mean density,
\(\bar\rho = \frac{3\rho_c}{\xi_1^3}\int_0^{\xi_1}\theta\,\xi^2\,d\xi\), and the ratio of
the central density to the mean, \(\pi^2/3 \approx 3.29\). A Greek letter with a label,
\(\xi_1\), and a barred name are defined.

```latex
\theta(\xi) := \frac{\sin\xi}{\xi}
\xi_1 := \pi
\bar\rho := \frac{3 \rho_c}{\xi_1^3} \int_0^{\xi_1} \theta(\xi)\, \xi^2 \, d\xi
\frac{\rho_c}{\bar\rho}
```

```python
xi = symbols("xi", positive=True)
theta = sin(xi) / xi
xi_1 = pi
mean = 3 / xi_1**3 * integrate(theta * xi**2, (xi, 0, xi_1))  # the mean density over rho_c
expected = simplify(1 / mean)
assert expected == pi**2 / 3
# Chandrasekhar's form of the same ratio, -xi_1 / (3 theta'(xi_1))
assert simplify(-xi_1 / (3 * diff(theta, xi).subs(xi, xi_1)) - expected) == 0
assert abs(float(expected) - 3.29) < 0.001
```

## The polytrope of index zero, from its central conditions

**Level 4** · `astro.stars.lane-emden-n0` · Chandrasekhar (1939), chapter IV

The Lane–Emden equation of index 0, a star of constant density, solved with the
conditions at its centre, \(\theta(0) = 1\) and \(\theta'(0) = 0\). The centre is a
singular point of the equation: its general solution \(C_1 + C_2/\xi - \xi^2/6\) is finite
there only when \(C_2 = 0\), so the conditions select \(\theta = 1 - \xi^2/6\), whose
first zero, the surface, is \(\sqrt{6}\). The notation is that of the index-one case; the
level is for the conditions, given at a singular point, as stellar-structure books give
them.

```latex
dsolve\left(\frac{1}{\xi^2} \frac{d}{d\xi}\left(\xi^2 \frac{d\theta}{d\xi}\right) = -1, \theta(0) = 1, \theta'(0) = 0\right)
```

```python
xi, C1, C2 = symbols("xi C1 C2")
theta = Function("theta")
general = dsolve(Eq(diff(xi**2 * theta(xi).diff(xi), xi) / xi**2, -1), theta(xi))
assert simplify(general.rhs - (C1 + C2 / xi - xi**2 / 6)) == 0
# Finite at the centre only if C2 = 0; then theta(0) = 1 gives C1 = 1, and theta'(0) = 0
solution = general.rhs.subs({C2: 0, C1: 1})
assert solution.subs(xi, 0) == 1 and diff(solution, xi).subs(xi, 0) == 0
expected = Eq(theta(xi), solution)
assert solve(solution, xi) == [-sqrt(6), sqrt(6)]  # the surface, at xi_1 = sqrt(6)
```

## The polytrope of index three, numerically

**Level 5** · `astro.stars.lane-emden-n3` · Chandrasekhar (1939), chapter IV

The Lane–Emden equation of index 3 has no solution in closed form. Books define
\(\theta\) by the equation and its conditions at the centre, call its first zero
\(\xi_1\), and tabulate \(-\xi_1^2\theta'(\xi_1) = 2.01824\), the constant in the
Chandrasekhar mass. A function defined by a differential equation, to be integrated
numerically from a singular point, and a number defined in words: no design covers them
yet.

```latex
\frac{1}{\xi^2} \frac{d}{d\xi}\left(\xi^2 \frac{d\theta}{d\xi}\right) = -\theta^3, \quad \theta(0) = 1, \quad \theta'(0) = 0
\xi_1 := \text{the first zero of } \theta
-\xi_1^2 \, \theta'(\xi_1)
```

```python
from scipy.integrate import solve_ivp


def lane_emden(x, y):  # theta'' = -theta^3 - 2 theta' / xi
    return [y[1], -(y[0] ** 3) - 2 * y[1] / x]


def surface(x, y):
    return y[0]


surface.terminal, surface.direction = True, -1
start = 1e-4  # the series theta = 1 - xi^2/6 + n xi^4/120 starts the integration
solution = solve_ivp(
    lane_emden,
    (start, 10),
    [1 - start**2 / 6 + 3 * start**4 / 120, -start / 3 + 3 * start**3 / 30],
    events=surface,
    method="DOP853",
    rtol=1e-12,
    atol=1e-14,
)
xi_1, slope = solution.t_events[0][0], solution.y_events[0][0][1]
expected = -(xi_1**2) * slope
# Chandrasekhar's table: xi_1 = 6.89685 and -xi_1^2 theta'(xi_1) = 2.01824
assert abs(xi_1 - 6.89685) < 5e-6 and abs(expected - 2.01824) < 5e-6
tolerance = 1e-6
```
