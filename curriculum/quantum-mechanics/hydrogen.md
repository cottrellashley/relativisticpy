# The hydrogen atom

The Bohr levels \(E_n = E_1/n^2\), with \(E_1 = -\hbar^2/2ma^2\) and \(a\) the Bohr radius,
the spectrum they give, their degeneracy, and the wave functions
\(\psi_{nlm} = R_{nl}(r) Y_l^m(\theta, \phi)\) in spherical coordinates.

## The Balmer α line

**Level 1** · `qm.hydrogen.balmer` · Griffiths (2005), section 4.2.2

The Rydberg formula, \(1/\lambda = R(1/n_f^2 - 1/n_i^2)\) with
\(R = 1.097 \times 10^7\,\mathrm{m^{-1}}\), for the transition from \(n = 3\) to
\(n = 2\): red light, at 656 nm, with a number in scientific notation.

```latex
\frac{1}{1.097 \times 10^{7} \left(\frac{1}{2^2} - \frac{1}{3^2}\right)}
```

```python
R = 1.097e7  # the Rydberg constant, per metre
expected = 1 / (R * (Rational(1, 2**2) - Rational(1, 3**2)))
assert abs(expected - 656.3e-9) < 0.1e-9
```

## The degeneracy of a level

**Level 1** · `qm.hydrogen.degeneracy` · Griffiths (2005), section 4.2.1

Level \(n\) holds \(2l + 1\) states for each \(l < n\): a finite sum with a symbolic
limit, \(n^2\).

```latex
\sum_{l=0}^{n-1} (2l + 1)
```

```python
l, n = symbols("l n", integer=True, positive=True)
expected = factor(summation(2 * l + 1, (l, 0, n - 1)))
assert expected == n**2
```

## The mean radius of the ground state

**Level 2** · `qm.hydrogen.mean-radius` · Griffiths (2005), Problem 4.13

\(\langle r\rangle = 3a/2\) for \(\psi_{100} = e^{-r/a}/\sqrt{\pi a^3}\), integrated over all
of space in spherical coordinates.

```latex
\psi_{100}(r) := \frac{1}{\sqrt{\pi a^3}} e^{-r/a}
\int_0^{2\pi} \int_0^{\pi} \int_0^{\infty} r\, |\psi_{100}(r)|^2 \, r^2 \sin\theta \, dr \, d\theta \, d\phi
```

```python
r = symbols("r", positive=True)
theta, phi = symbols("theta phi", real=True)
a = symbols("a", positive=True)
psi = exp(-r / a) / sqrt(pi * a**3)
volume = r**2 * sin(theta)
limits = ((r, 0, oo), (theta, 0, pi), (phi, 0, 2 * pi))
assert integrate(psi**2 * volume, *limits) == 1
expected = integrate(r * psi**2 * volume, *limits)
assert expected == 3 * a / 2  # Griffiths, Problem 4.13
```

## A radial function solves the radial equation

**Level 3** · `qm.hydrogen.radial-equation` · Griffiths (2005), section 4.2.1

\(R_{21} = \frac{1}{\sqrt{24}} a^{-3/2} \frac{r}{a} e^{-r/2a}\), from Griffiths' table of
radial functions, a function named with two subscripts: \(u = rR_{21}\), built from it,
solves the radial equation with \(l = 1\) and energy \(E_1/2^2\). The Coulomb potential is
written with the Bohr radius, \(e^2/4\pi\epsilon_0 = \hbar^2/ma\).

```latex
l := 1
E_1 := -\frac{\hbar^2}{2ma^2}
R_{21}(r) := \frac{1}{\sqrt{24}} a^{-3/2} \frac{r}{a} e^{-\frac{r}{2a}}
u(r) := r R_{21}(r)
-\frac{\hbar^2}{2m} \frac{d^2u}{dr^2} + \left[-\frac{\hbar^2}{ma}\frac{1}{r} + \frac{\hbar^2}{2m}\frac{l(l+1)}{r^2}\right] u = \frac{E_1}{2^2} u
```

```python
r = symbols("r", positive=True)
a, m, hbar = symbols("a m hbar", positive=True)
l = 1
E_1 = -(hbar**2) / (2 * m * a**2)
R_21 = 1 / sqrt(24) * a ** Rational(-3, 2) * r / a * exp(-r / (2 * a))
assert integrate(R_21**2 * r**2, (r, 0, oo)) == 1  # normalised
u = r * R_21
V_eff = -(hbar**2) / (m * a * r) + hbar**2 / (2 * m) * l * (l + 1) / r**2
assert simplify(-(hbar**2) / (2 * m) * diff(u, r, 2) + V_eff * u - E_1 / 2**2 * u) == 0
expected = True
```
