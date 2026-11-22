# The canonical ensemble

Partition functions of the systems every course begins with, a two-level system, the
harmonic oscillator, the ideal gas and the rigid rotor, and the thermodynamics that
follows from them: the energy, the free energy, the entropy and the heat capacity.

## The energy of a two-level system

**Level 1** · `statmech.canonical.two-level-energy` · Kittel and Kroemer (1980), chapter 3

\(U = \tau^2 \partial \ln Z / \partial \tau\) for two states of energy 0 and
\(\varepsilon\), with \(\tau = k_B T\): \(\varepsilon / (e^{\varepsilon/\tau} + 1)\).

```latex
\tau^2 \frac{\partial}{\partial \tau} \ln\left(1 + e^{-\varepsilon/\tau}\right)
```

```python
tau, epsilon = symbols("tau epsilon", positive=True)
expected = epsilon / (exp(epsilon / tau) + 1)
assert simplify(tau**2 * diff(log(1 + exp(-epsilon / tau)), tau) - expected) == 0
```

## The partition function of a harmonic oscillator

**Level 2** · `statmech.canonical.oscillator` · Pathria and Beale (2011), section 3.8

The sum over the levels \((n + \frac{1}{2}) \hbar\omega\), a geometric series:
\(Z = 1 / (2 \sinh(\beta\hbar\omega/2))\).

```latex
\sum_{n=0}^{\infty} e^{-\beta \hbar \omega \left(n + \frac{1}{2}\right)}
```

```python
beta, hbar, omega = symbols("beta hbar omega", positive=True)
r = exp(-beta * hbar * omega)
expected = 1 / (2 * sinh(beta * hbar * omega / 2))
series = exp(-beta * hbar * omega / 2) / (1 - r)  # the geometric series, for r < 1
assert simplify((series - expected).rewrite(exp)) == 0
x = 0.8  # and numerically, at beta hbar omega = 0.8
assert abs(sum(np.exp(-x * (n + 0.5)) for n in range(200)) - 1 / (2 * np.sinh(x / 2))) < 1e-12
```

## The partition function of a particle in a box

**Level 2** · `statmech.canonical.ideal-gas` · Pathria and Beale (2011), section 3.5

The translational partition function of one molecule of an ideal gas: the volume times
a Gaussian integral over each component of the momentum, in units of \(h^3\),
\(V (2\pi m k T)^{3/2} / h^3\).

```latex
\frac{V}{h^3} \left(\int_{-\infty}^{\infty} e^{-p^2/(2 m k T)} \, dp\right)^3
```

```python
V, h, m, k, T, p = symbols("V h m k T p", positive=True)
expected = V * (2 * pi * m * k * T) ** Rational(3, 2) / h**3
found = V / h**3 * integrate(exp(-(p**2) / (2 * m * k * T)), (p, -oo, oo)) ** 3
assert simplify(found - expected) == 0
```

## The rotational partition function at high temperature

**Level 2** · `statmech.canonical.rigid-rotor` · Pathria and Beale (2011), section 6.5

The levels of a rigid rotor, \(l(l + 1) k \Theta_r\), each \((2l + 1)\)-fold degenerate.
Well above \(\Theta_r\) the sum over them becomes an integral, and \(Z_{\mathrm{rot}}
\approx T / \Theta_r\).

```latex
\int_0^\infty (2l + 1) e^{-l(l + 1) \Theta_r / T} \, dl
```

```python
l, T, Theta = symbols("l T Theta_r", positive=True)
expected = T / Theta
assert simplify(integrate((2 * l + 1) * exp(-l * (l + 1) * Theta / T), (l, 0, oo)) - expected) == 0
exact = sum((2 * j + 1) * np.exp(-j * (j + 1) * 0.01) for j in range(400))  # T = 100 Theta_r
assert abs(exact - (100 + 1 / 3)) < 0.01  # the sum exceeds T / Theta_r by about 1/3
```

## The heat capacity of a two-level system

**Level 3** · `statmech.canonical.schottky` · Kittel and Kroemer (1980), chapter 3

From the partition function to the free energy \(F = -\tau \ln Z\), the entropy
\(\sigma = -\partial F / \partial \tau\) and the heat capacity \(C = \tau \partial \sigma
/ \partial \tau\): the Schottky anomaly, a peak near \(\tau \approx 0.42 \varepsilon\).

```latex
Z := 1 + e^{-\varepsilon/\tau}
F := -\tau \ln Z
\sigma := -\frac{\partial F}{\partial \tau}
C := \tau \frac{\partial \sigma}{\partial \tau}
C
```

```python
tau, epsilon = symbols("tau epsilon", positive=True)
Z = 1 + exp(-epsilon / tau)
F = -tau * log(Z)
entropy = -diff(F, tau)
C = tau * diff(entropy, tau)
expected = (epsilon / tau) ** 2 * exp(epsilon / tau) / (exp(epsilon / tau) + 1) ** 2
assert simplify(C - expected) == 0
assert simplify(C - diff(tau**2 * diff(log(Z), tau), tau)) == 0  # C = dU/dtau, too
grid = np.linspace(0.1, 2, 19001)
heat = lambdify(tau, expected.subs(epsilon, 1), "numpy")(grid)
assert abs(grid[np.argmax(heat)] - 0.42) < 0.01  # the peak, in units of epsilon
```
