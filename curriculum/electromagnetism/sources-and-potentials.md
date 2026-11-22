# Fields of charges and currents

Coulomb's law and the Biot–Savart law, the multipole expansion of a potential, and the
retarded potential of a moving charge.

## Coulomb's law in hydrogen

**Level 1** · `em.sources.coulomb` · Griffiths (2013), section 2.1.2; CODATA 2018

The force between the proton and the electron of hydrogen at the Bohr radius, with the
constants given in scientific notation; \(e\) here is the elementary charge.

```latex
\frac{1}{4\pi\epsilon_0} \frac{e^2}{a_0^2} \Big|_{\epsilon_0 = 8.8541878128\times10^{-12},\ e = 1.602176634\times10^{-19},\ a_0 = 5.29177210903\times10^{-11}}
```

```python
epsilon_0, e, a_0 = 8.8541878128e-12, 1.602176634e-19, 5.29177210903e-11
expected = e**2 / (4 * np.pi * epsilon_0 * a_0**2)
assert abs(expected - 8.2387e-8) < 1e-11  # newtons
```

## The magnetic moment of a current loop

**Level 1** · `em.sources.loop-moment` · Griffiths (2013), section 5.4.3

The moment of a loop is its current times its area, here written with the loop's
diameter as \(\pi d^2 / 4\): a fraction whose numerator holds \(d^2\), not a second
derivative. A loop 10 cm across carrying 2 A.

```latex
\left. I \frac{\pi d^2}{4} \right|_{I = 2,\ d = 0.1}
```

```python
expected = 2 * np.pi * 0.1**2 / 4
assert abs(expected - 0.0157) < 1e-4  # ampere square metres
```

## The potential of a pure dipole

**Level 2** · `em.sources.dipole` · Griffiths (2013), section 3.4

Charges \(\pm q\) a distance \(d\) apart. In the limit \(d \to 0\) with the moment
\(p = qd\) held fixed, written as an evaluation bar inside a limit, their potential is
that of a pure dipole, \(p \cos\theta / 4\pi\epsilon_0 r^2\).

```latex
V := \frac{q}{4\pi\epsilon_0} \left( \frac{1}{\sqrt{r^2 + (d/2)^2 - r d \cos\theta}} - \frac{1}{\sqrt{r^2 + (d/2)^2 + r d \cos\theta}} \right)
\lim_{d \to 0} \left. V \right|_{q = p/d}
```

```python
q, epsilon_0, r, d, theta, p = symbols("q epsilon_0 r d theta p", positive=True)
plus = 1 / sqrt(r**2 + (d / 2) ** 2 - r * d * cos(theta))
minus = 1 / sqrt(r**2 + (d / 2) ** 2 + r * d * cos(theta))
V = q / (4 * pi * epsilon_0) * (plus - minus)
expected = limit(V.subs(q, p / d), d, 0)
assert simplify(expected - p * cos(theta) / (4 * pi * epsilon_0 * r**2)) == 0
```

## The field on the axis of a current loop

**Level 3** · `em.sources.current-loop` · Griffiths (2013), Example 5.6

The Biot–Savart law for a circular loop of radius \(a\), written with the cross product
of two three-vectors, the length of a vector, and an integral of a vector around the
loop.

```latex
\mathbf{R} := (-a \cos\phi,\ -a \sin\phi,\ z)
\frac{\mu_0 I}{4\pi} \int_0^{2\pi} \frac{a\,(-\sin\phi,\ \cos\phi,\ 0) \times \mathbf{R}}{|\mathbf{R}|^3}\, d\phi
```

```python
mu_0, current, a, z, phi = symbols("mu_0 I a z phi", positive=True)
R = Matrix([-a * cos(phi), -a * sin(phi), z])
assert simplify(R.norm() - sqrt(a**2 + z**2)) == 0
integrand = (a * Matrix([-sin(phi), cos(phi), 0])).cross(R) / (a**2 + z**2) ** Rational(3, 2)
field = mu_0 * current / (4 * pi) * integrand.applyfunc(lambda f: integrate(f, (phi, 0, 2 * pi)))
expected = [simplify(component) for component in field]
on_axis = mu_0 * current * a**2 / (2 * (a**2 + z**2) ** Rational(3, 2))
assert expected[0] == 0 and expected[1] == 0 and simplify(expected[2] - on_axis) == 0
```

## The expansion of the inverse distance in Legendre polynomials

**Level 4** · `em.sources.legendre` · Jackson (1999), section 3.3

The generating function of the Legendre polynomials, which underlies the multipole
expansion: an identity between a closed form and an infinite series of special
functions, for \(r' < r\).

```latex
r' < r
\frac{1}{\sqrt{r^2 + r'^2 - 2 r r' \cos\gamma}} = \sum_{l=0}^{\infty} \frac{r'^l}{r^{l+1}} P_l(\cos\gamma)
```

```python
from scipy.special import eval_legendre

h, x = symbols("h x")
closed = series(1 / sqrt(1 - 2 * x * h + h**2), h, 0, 12).removeO()
assert expand(closed - sum(legendre(l, x) * h**l for l in range(12))) == 0
for r, r_prime, angle in ((2.0, 0.7, 1.1), (1.0, 0.9, 2.5), (5.0, 1.0, 0.3)):
    left = 1 / np.sqrt(r**2 + r_prime**2 - 2 * r * r_prime * np.cos(angle))
    terms = (r_prime**l / r ** (l + 1) * eval_legendre(l, np.cos(angle)) for l in range(400))
    assert abs(left - sum(terms)) < 1e-12
expected = True
```

## The retarded potential of a uniformly moving charge

**Level 5** · `em.sources.lienard-wiechert` · Jackson (1999), section 14.1

The Liénard–Wiechert potential, written with Jackson's bracket \([\ldots]_{\text{ret}}\):
evaluated at the retarded time \(t'\) at which \(t - t' = R(t')\) (\(c = 1\), Gaussian
units). For a charge in uniform motion it is Heaviside's potential,
\(q / \sqrt{(x - vt)^2 + (1 - v^2)(y^2 + z^2)}\).

```latex
\mathbf{x} := (x, y, z), \quad \mathbf{r}(t') := (v t', 0, 0), \quad \boldsymbol{\beta} := (v, 0, 0)
\mathbf{R} := \mathbf{x} - \mathbf{r}(t'), \quad R := |\mathbf{R}|, \quad \mathbf{n} := \frac{\mathbf{R}}{R}
\Phi := \left[ \frac{q}{(1 - \boldsymbol{\beta} \cdot \mathbf{n}) R} \right]_{\text{ret}}
\Phi
```

```python
x, y, z, t, v, q = symbols("x y z t v q", real=True)
rho2 = y**2 + z**2
# (t - t')^2 = (x - v t')^2 + y^2 + z^2 is quadratic in t'; the root with t' < t is retarded
discriminant = (t - v * x) ** 2 - (1 - v**2) * (t**2 - x**2 - rho2)
retarded = ((t - v * x) - sqrt(discriminant)) / (1 - v**2)
R = t - retarded
potential = q / (R - v * (x - v * retarded))  # (1 - beta . n) R = R - beta . R-vector
expected = q / sqrt((x - v * t) ** 2 + (1 - v**2) * rho2)
points = [(0.3, -1.2, 0.5, 2.0, 0.6), (-2.0, 0.4, 1.5, -1.0, -0.8), (1.0, 1.0, 0.0, 0.5, 0.95)]
for px, py, pz, pt, pv in points:
    at = {x: px, y: py, z: pz, t: pt, v: pv, q: 1.3}
    t_ret, distance = float(retarded.subs(at)), float(R.subs(at))
    assert t_ret < pt  # retarded, and on the past light cone of the field point
    assert abs(distance - np.sqrt((px - pv * t_ret) ** 2 + py**2 + pz**2)) < 1e-12
    assert abs(float(potential.subs(at)) - float(expected.subs(at))) < 1e-12
```
