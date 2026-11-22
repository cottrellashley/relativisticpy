# Quantum statistics and black-body radiation

The Fermi–Dirac and Bose–Einstein distributions from the grand and the canonical
partition functions of a single orbital, and black-body radiation: Planck's
distribution, its integral and the Stefan–Boltzmann law.

## The integral in Planck's law

**Level 1** · `statmech.quantum.planck-integral` · Pathria and Beale (2011), section 7.3

The integral behind the Stefan–Boltzmann law, \(\Gamma(4) \zeta(4) = \pi^4 / 15\).

```latex
\int_0^\infty \frac{x^3}{e^x - 1} \, dx
```

```python
from scipy.integrate import quad

expected = gamma(4) * zeta(4)
assert expected == pi**4 / 15
assert abs(quad(lambda x: x**3 / np.expm1(x), 1e-12, 200)[0] - float(expected)) < 1e-9
```

## The Fermi–Dirac distribution

**Level 2** · `statmech.quantum.fermi-dirac` · Kittel and Kroemer (1980), chapter 6

An orbital holds no fermion or one. Its Gibbs sum is \(\mathcal{Z} = 1 +
e^{(\mu - \varepsilon)/\tau}\), and its mean occupancy, \(\tau \partial \ln \mathcal{Z}
/ \partial \mu\), is the Fermi–Dirac distribution.

```latex
\mathcal{Z} := 1 + e^{(\mu - \varepsilon)/\tau}
\tau \frac{\partial}{\partial \mu} \ln \mathcal{Z}
```

```python
mu, epsilon, tau = symbols("mu epsilon tau", positive=True)
expected = 1 / (exp((epsilon - mu) / tau) + 1)
assert simplify(tau * diff(log(1 + exp((mu - epsilon) / tau)), mu) - expected) == 0
```

## The Planck distribution

**Level 2** · `statmech.quantum.planck-distribution` · Kittel and Kroemer (1980), chapter 4

The mean number of photons in a mode of frequency \(\omega\), from the canonical sums
over \(s\) photons: Bose–Einstein's distribution with no chemical potential,
\(1 / (e^{\hbar\omega/\tau} - 1)\).

```latex
\frac{\sum_{s=0}^{\infty} s \, e^{-s \hbar \omega / \tau}}{\sum_{s=0}^{\infty} e^{-s \hbar \omega / \tau}}
```

```python
hbar, omega, tau = symbols("hbar omega tau", positive=True)
r = exp(-hbar * omega / tau)
expected = 1 / (exp(hbar * omega / tau) - 1)
ratio = (r / (1 - r) ** 2) / (1 / (1 - r))  # the two geometric series, for r < 1
assert simplify(ratio - expected) == 0
x = 0.6  # and numerically, at hbar omega / tau = 0.6
terms = [np.exp(-x * s) for s in range(400)]
assert abs(sum(s * w for s, w in enumerate(terms)) / sum(terms) - 1 / np.expm1(x)) < 1e-12
```

## The Stefan–Boltzmann law

**Level 3** · `statmech.quantum.stefan-boltzmann` · Kittel and Kroemer (1980), chapter 4

The energy density of black-body radiation from Planck's distribution and the density
of modes of the field: \(u = \pi^2 \tau^4 / (15 \hbar^3 c^3)\).

```latex
\langle s \rangle := \frac{1}{e^{\hbar \omega / \tau} - 1}
D(\omega) := \frac{\omega^2}{\pi^2 c^3}
\int_0^\infty \hbar \omega \, \langle s \rangle \, D(\omega) \, d\omega
```

```python
from scipy.integrate import quad

hbar, tau, c = symbols("hbar tau c", positive=True)
# Substituting x = hbar omega / tau leaves tau^4 / (pi^2 hbar^3 c^3) times the integral of
# x^3 / (e^x - 1), which is pi^4 / 15.
expected = tau**4 / (pi**2 * hbar**3 * c**3) * gamma(4) * zeta(4)
assert simplify(expected - pi**2 * tau**4 / (15 * hbar**3 * c**3)) == 0
values = {hbar: 1.3, tau: 0.9, c: 0.7}
density = lambda w: 1.3 * w * (w**2 / (np.pi**2 * 0.7**3)) / np.expm1(1.3 * w / 0.9)
assert abs(quad(density, 1e-12, 200)[0] - float(expected.subs(values))) < 1e-9
```
