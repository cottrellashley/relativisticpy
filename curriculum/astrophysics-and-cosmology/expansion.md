# The expansion of the universe

Friedmann models in the notation of cosmology books: the deceleration parameter today,
the age of a universe of matter and of one with a cosmological constant, the scale factor
of the radiation era, the temperature of the cosmic microwave background as space
expands, and the Hubble time in the units astronomers quote it in. The density parameters
are \(\Omega_m\) and \(\Omega_\Lambda\), the Hubble parameter today is \(H_0\), and the
scale factor today is 1.

## The deceleration parameter today

**Level 1** · `astro.expansion.deceleration-parameter` · standard

For a universe of matter and a cosmological constant, \(q_0 = \Omega_m/2 - \Omega_\Lambda\):
with \(\Omega_m = 0.3\) and \(\Omega_\Lambda = 0.7\) it is negative, and the expansion
accelerates. Greek letters with labels, given values by an evaluation bar.

```latex
\frac{\Omega_m}{2} - \Omega_\Lambda \Big|_{\Omega_m = 0.3,\ \Omega_\Lambda = 0.7}
```

```python
a, H_0, Omega_m, Omega_L = symbols("a H_0 Omega_m Omega_Lambda", positive=True)
values = {Omega_m: Rational(3, 10), Omega_L: Rational(7, 10)}
expected = (Omega_m / 2 - Omega_L).subs(values)
assert expected == Rational(-11, 20)
# From its definition, q = -a a''/a'^2 = -1 - a H'(a)/H(a), with H^2 = H_0^2 (Omega_m/a^3 + Omega_L)
H = H_0 * sqrt(Omega_m / a**3 + Omega_L)
q = -1 - a * diff(H, a) / H
assert simplify(q.subs(a, 1).subs(values) - expected) == 0
```

## The age of a universe of matter

**Level 2** · `astro.expansion.matter-age` · Ryden (2017), chapter 5

In a flat universe of matter alone \(H = H_0 a^{-3/2}\), and the time since \(a = 0\) is
\(\int_0^1 da/aH\): \(t_0 = 2/3H_0\), two thirds of the Hubble time.

```latex
H_0 > 0
H(a) := H_0 a^{-3/2}
\int_0^1 \frac{da}{a H(a)}
```

```python
a, t, H_0 = symbols("a t H_0", positive=True)
expected = integrate(1 / (a * H_0 * a ** Rational(-3, 2)), (a, 0, 1))
assert expected == 2 / (3 * H_0)
# a = (t / t_0)^(2/3) with t_0 = 2 / 3 H_0 has a'/a = H_0 a^(-3/2), and a = 1 at t_0
t_0 = 2 / (3 * H_0)
scale = (t / t_0) ** Rational(2, 3)
assert simplify(diff(scale, t) / scale - H_0 * scale ** Rational(-3, 2)) == 0
assert scale.subs(t, t_0) == 1
```

## The age of a flat universe with a cosmological constant

**Level 2** · `astro.expansion.lambda-cdm-age` · Carroll, Press and Turner (1992), ARA&A 30, 499

Matter and a cosmological constant, \(H = H_0\sqrt{\Omega_m a^{-3} + \Omega_\Lambda}\): the
age \(\int_0^1 da/aH\) in closed form,
\(\frac{2}{3H_0\sqrt{\Omega_\Lambda}}\operatorname{arsinh}\sqrt{\Omega_\Lambda/\Omega_m}\),
0.964 Hubble times for \(\Omega_m = 0.3\) and \(\Omega_\Lambda = 0.7\).

```latex
H_0, \Omega_m, \Omega_\Lambda > 0
H(a) := H_0 \sqrt{\Omega_m a^{-3} + \Omega_\Lambda}
\int_0^1 \frac{da}{a H(a)}
```

```python
from scipy.integrate import quad

a, H_0, Omega_m, Omega_L = symbols("a H_0 Omega_m Omega_Lambda", positive=True)
H = H_0 * sqrt(Omega_m / a**3 + Omega_L)
expected = 2 * asinh(sqrt(Omega_L / Omega_m)) / (3 * H_0 * sqrt(Omega_L))
assert simplify(integrate(1 / (a * H), (a, 0, 1)) - expected) == 0
# Numerically, for Omega_m = 0.3 and Omega_Lambda = 0.7, in units of 1/H_0
numeric = quad(lambda x: 1 / (x * np.sqrt(0.3 / x**3 + 0.7)), 0, 1)[0]
assert abs(numeric - float(expected.subs({H_0: 1, Omega_m: 0.3, Omega_L: 0.7}))) < 1e-10
# Carroll, Press and Turner's form when Omega_m + Omega_Lambda = 1
flat = 2 / (3 * np.sqrt(0.7)) * np.log((1 + np.sqrt(0.7)) / np.sqrt(0.3))
assert abs(numeric - flat) < 1e-10 and abs(numeric - 0.964) < 0.001
```

## The scale factor of the radiation era

**Level 2** · `astro.expansion.radiation-scale-factor` · Ryden (2017), chapter 5

In a flat universe of radiation alone the Friedmann equation, \((\dot a/a)^2 = H_0^2/a^4\),
reads \(\dot a = H_0/a\) while the universe expands, with Newton's dot and the unknown named
without its argument. Solved from \(a(0) = 0\), for a scale factor that cannot be negative:
\(a = \sqrt{2H_0 t}\), so that \(H = 1/2t\).

```latex
H_0 > 0
t \geq 0
a \geq 0
dsolve\left(\dot a = \frac{H_0}{a}, a(0) = 0\right)
```

```python
t, H_0 = symbols("t H_0", positive=True)
a = Function("a")
solutions = dsolve(Eq(a(t).diff(t), H_0 / a(t)), a(t), ics={a(0): 0})
growing = [s for s in solutions if s.rhs.subs(t, 1).is_nonnegative]
assert len(solutions) == 2 and len(growing) == 1  # a = -sqrt(2 H_0 t) is excluded
expected = growing[0]
assert simplify(expected.rhs - sqrt(2 * H_0 * t)) == 0
assert simplify(diff(expected.rhs, t) / expected.rhs - 1 / (2 * t)) == 0  # H = 1/2t
```

## The temperature of the cosmic microwave background

**Level 4** · `astro.expansion.cmb-temperature` · Fixsen (2009), ApJ 707, 916

Radiation cools as space expands: \(T \propto 1/a\), the proportion fixed by today's
temperature, 2.7255 K at \(a = 1\). At redshift \(z\), where \(a = 1/(1 + z)\), the
temperature is \(T_0(1 + z)\): about 3000 K when the background was released, at
\(z \approx 1100\). A proportionality and its normalisation, as cosmology books state them.

```latex
T(a) \propto \frac{1}{a}
T(1) = 2.7255
T\left(\frac{1}{1 + z}\right)
```

```python
a, z, C = symbols("a z C", positive=True)
T = C / a  # T is proportional to 1/a
normalised = T.subs(C, solve(Eq(T.subs(a, 1), Float("2.7255")), C)[0])
expected = normalised.subs(a, 1 / (1 + z))
assert simplify(expected - Float("2.7255") * (1 + z)) == 0
assert abs(float(expected.subs(z, 1100)) - 3000.78) < 0.01  # at recombination
```

## The Hubble time in gigayears

**Level 5** · `astro.expansion.hubble-time-units` · standard; IAU (2012) Resolution B2

\(H_0 = 70\) km s\(^{-1}\) Mpc\(^{-1}\), written with its units as astronomers write it, and
the Hubble time \(1/H_0\) as a number of gigayears: the quantity divided by its unit, as
ISO 80000-1 writes a numerical value, 13.97. Kilometres, megaparsecs and gigayears (of
Julian years) must cancel to a number; no design of the language covers physical units
yet.

```latex
H_0 := 70\,\mathrm{km\,s^{-1}\,Mpc^{-1}}
\frac{H_0^{-1}}{\mathrm{Gyr}}
```

```python
au = 149_597_870_700  # metres, exactly: IAU (2012) Resolution B2
Mpc = 1e6 * 648_000 / np.pi * au  # a parsec is 648 000 / pi astronomical units
Gyr = 1e9 * 365.25 * 86_400  # seconds in a billion Julian years
expected = Mpc / 70e3 / Gyr
assert abs(expected - 13.968) < 0.001
# Hogg (1999), arXiv:astro-ph/9905116: 1/H_0 = 9.78 / h Gyr, with H_0 = 100 h km/s/Mpc
assert abs(expected - 9.78 / 0.7) < 0.01
tolerance = 1e-6
```
