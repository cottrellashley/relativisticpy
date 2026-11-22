# Scattering

A particle scattered by a potential, described by its scattering amplitude \(f(\theta)\)
and cross-section \(|f(\theta)|^2\): partial waves at low energy, and the first Born
approximation, which makes the amplitude the Fourier transform of the potential, in the
radial form of Griffiths and the vector form of Sakurai.

## Low-energy scattering from a hard sphere

**Level 2** · `qm.scattering.hard-sphere` · Griffiths (2005), Example 11.3

At low energy only the s-wave scatters, with the phase shift \(\delta_0 = -ka\) of a hard
sphere: \(\sigma = \frac{4\pi}{k^2}\sin^2\delta_0 \to 4\pi a^2\), four times the
geometrical cross-section.

```latex
\delta_0 := -ka
\sigma := \frac{4\pi}{k^2} \sin^2\delta_0
\lim_{k \to 0} \sigma
```

```python
k, a = symbols("k a", positive=True)
delta_0 = -k * a
expected = limit(4 * pi / k**2 * sin(delta_0) ** 2, k, 0)
assert expected == 4 * pi * a**2  # Griffiths, Example 11.3
```

## The Born amplitude of a soft sphere

**Level 3** · `qm.scattering.soft-sphere` · Griffiths (2005), Example 11.4

The Born approximation for a spherically symmetric potential,
\(f(\theta) = -\frac{2m}{\hbar^2\kappa}\int_0^\infty rV(r)\sin(\kappa r)\,dr\) with
\(\kappa = 2k\sin(\theta/2)\), for a spherical well of depth \(V_0\) written piecewise; its
low-energy limit is \(-\tfrac{2}{3}mV_0a^3/\hbar^2\), the same in every direction.

```latex
V(r) := \begin{cases} V_0 & \text{if } r \leq a \\ 0 & \text{otherwise} \end{cases}
\kappa := 2k \sin\left(\frac{\theta}{2}\right)
f(\theta) := -\frac{2m}{\hbar^2 \kappa} \int_0^\infty r V(r) \sin(\kappa r) \, dr
\lim_{k \to 0} f(\theta)
```

```python
r, kappa, k, theta = symbols("r kappa k theta", positive=True)
a, m, hbar, V_0 = symbols("a m hbar V_0", positive=True)
V = Piecewise((V_0, r <= a), (0, True))
f = -2 * m / (hbar**2 * kappa) * integrate(r * V * sin(kappa * r), (r, 0, oo))
born = -2 * m * V_0 / (hbar**2 * kappa**3) * (sin(kappa * a) - kappa * a * cos(kappa * a))
assert simplify(f - born) == 0
expected = limit(born.subs(kappa, 2 * k * sin(theta / 2)), k, 0)
assert simplify(expected + Rational(2, 3) * m * V_0 * a**3 / hbar**2) == 0  # Griffiths, Example 11.4
```

## Rutherford scattering

**Level 3** · `qm.scattering.rutherford` · Griffiths (2005), Example 11.6

The Born amplitude of the Yukawa potential \(\beta e^{-\mu r}/r\), with the screening
removed, \(\mu \to 0\): the cross-section of the Coulomb potential is Rutherford's, in
terms of the energy \(E = \hbar^2k^2/2m\).

```latex
\kappa := 2k \sin\left(\frac{\theta}{2}\right)
f(\theta) := -\frac{2m\beta}{\hbar^2 \left(\mu^2 + \kappa^2\right)}
E := \frac{\hbar^2 k^2}{2m}
\lim_{\mu \to 0} |f(\theta)|^2 = \left(\frac{\beta}{4E\sin^2(\theta/2)}\right)^2
```

```python
theta, k, m, hbar, beta, mu = symbols("theta k m hbar beta mu", positive=True)
kappa = 2 * k * sin(theta / 2)
f = -2 * m * beta / (hbar**2 * (mu**2 + kappa**2))  # Griffiths, Example 11.5
E = hbar**2 * k**2 / (2 * m)
cross_section = limit(f**2, mu, 0)
assert simplify(cross_section - (beta / (4 * E * sin(theta / 2) ** 2)) ** 2) == 0
expected = True
```

## The Born approximation for a Yukawa potential

**Level 4** · `qm.scattering.born-yukawa` · Sakurai (2017), section 6.3

The first Born amplitude as Sakurai writes it, an integral over all of space with wave
vectors and a dot product, \(f^{(1)}(\mathbf{k}', \mathbf{k}) = -\frac{1}{4\pi}
\frac{2m}{\hbar^2} \int d^3x'\, e^{i(\mathbf{k} - \mathbf{k}')\cdot\mathbf{x}'}
V(\mathbf{x}')\), for the Yukawa potential \(V_0 e^{-\mu r}/\mu r\).

```latex
V(r) := \frac{V_0 e^{-\mu r}}{\mu r}
f^{(1)}(\mathbf{k}', \mathbf{k}) := -\frac{1}{4\pi} \frac{2m}{\hbar^2} \int d^3x' \, e^{i(\mathbf{k} - \mathbf{k}') \cdot \mathbf{x}'} \, V(r')
f^{(1)}(\mathbf{k}', \mathbf{k}) = -\frac{2mV_0}{\mu\hbar^2} \frac{1}{|\mathbf{k} - \mathbf{k}'|^2 + \mu^2}
```

```python
r = symbols("r", positive=True)
theta = symbols("theta", real=True)
q, m, hbar, V_0, mu = symbols("q m hbar V_0 mu", positive=True)
V = V_0 * exp(-mu * r) / (mu * r)
# With q = k - k' along the z-axis, q.x' = q r cos(theta); the azimuth gives 2 pi.
angular = simplify(integrate(exp(I * q * r * cos(theta)) * sin(theta), (theta, 0, pi)))
assert simplify(angular - 2 * sin(q * r) / (q * r)) == 0
f = -1 / (4 * pi) * 2 * m / hbar**2 * 2 * pi * integrate(angular * V * r**2, (r, 0, oo))
stated = -2 * m * V_0 / (mu * hbar**2) / (q**2 + mu**2)  # Sakurai, section 6.3
assert simplify(f - stated) == 0
expected = True
```
