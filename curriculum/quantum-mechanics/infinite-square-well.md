# The infinite square well

A particle confined to \(0 \le x \le a\), with stationary states
\(\psi_n(x) = \sqrt{2/a} \sin(n\pi x/a)\) and energies \(E_n = n^2\pi^2\hbar^2/2ma^2\):
normalisation, orthogonality, expansions in the stationary states, and expectation
values, with momentum as a derivative.

## Normalisation

**Level 1** · `qm.square-well.normalisation` · Griffiths (2005), section 2.2

The ground state of the well is normalised: one integral of a squared modulus.

```latex
\int_0^a \left| \sqrt{\frac{2}{a}} \sin\left(\frac{\pi x}{a}\right) \right|^2 dx
```

```python
x = symbols("x", real=True)
a = symbols("a", positive=True)
expected = integrate((sqrt(2 / a) * sin(pi * x / a)) ** 2, (x, 0, a))
assert expected == 1
```

## Orthogonality

**Level 2** · `qm.square-well.orthogonality` · Griffiths (2005), section 2.2

A family of functions indexed by \(n\), and two of its members: stationary states of
different energies are orthogonal.

```latex
\psi_n(x) := \sqrt{\frac{2}{a}} \sin\left(\frac{n\pi x}{a}\right)
\int_0^a \psi_1(x)^* \psi_2(x) \, dx
```

```python
x = symbols("x", real=True)
a = symbols("a", positive=True)
psi = lambda n: sqrt(2 / a) * sin(n * pi * x / a)
expected = integrate(conjugate(psi(1)) * psi(2), (x, 0, a))
assert expected == 0
assert integrate(psi(2) ** 2, (x, 0, a)) == 1
```

## Expanding a state in stationary states

**Level 3** · `qm.square-well.expansion` · Griffiths (2005), Example 2.2

A parabolic state \(A x(a - x)\), normalised, and its component along the ground state:
\(c_1 = 8\sqrt{15}/\pi^3\), so that \(|c_1|^2 \approx 0.9986\).

```latex
A := \left(\int_0^a x^2 (a - x)^2 \, dx\right)^{-1/2}
\Psi(x) := A\, x (a - x)
\psi_n(x) := \sqrt{\frac{2}{a}} \sin\left(\frac{n\pi x}{a}\right)
c_n := \int_0^a \psi_n(x)^* \Psi(x) \, dx
c_1
```

```python
x = symbols("x", real=True)
a = symbols("a", positive=True)
A = 1 / sqrt(integrate(x**2 * (a - x) ** 2, (x, 0, a)))
assert simplify(A - sqrt(30 / a**5)) == 0  # Griffiths: A = sqrt(30 / a^5)
Psi = A * x * (a - x)
c = lambda n: integrate(sqrt(2 / a) * sin(n * pi * x / a) * Psi, (x, 0, a))
expected = simplify(c(1))
assert simplify(expected - 8 * sqrt(15) / pi**3) == 0  # c_n = 8 sqrt(15) / (n pi)^3, n odd
assert simplify(c(2)) == 0
```

## The uncertainty product of the ground state

**Level 3** · `qm.square-well.uncertainty` · Griffiths (2005), Problem 2.4

Expectation values of \(x\), \(x^2\) and \(p^2\), built on one another into the
uncertainties: \(\sigma_x\sigma_p = \frac{\hbar}{2}\sqrt{\pi^2/3 - 2}\), above the bound.

```latex
\psi_1(x) := \sqrt{\frac{2}{a}} \sin\left(\frac{\pi x}{a}\right)
\langle x \rangle := \int_0^a x\, |\psi_1(x)|^2 \, dx
\langle x^2 \rangle := \int_0^a x^2 |\psi_1(x)|^2 \, dx
\langle p^2 \rangle := \int_0^a \psi_1(x) \left(-\hbar^2 \frac{d^2}{dx^2}\right) \psi_1(x) \, dx
\sigma_x := \sqrt{\langle x^2 \rangle - \langle x \rangle^2}
\sigma_p := \sqrt{\langle p^2 \rangle}
\sigma_x \sigma_p
```

```python
x = symbols("x", real=True)
a, hbar = symbols("a hbar", positive=True)
psi = sqrt(2 / a) * sin(pi * x / a)
mean_x = integrate(x * psi**2, (x, 0, a))
mean_x2 = integrate(x**2 * psi**2, (x, 0, a))
mean_p = integrate(psi * (-I * hbar) * diff(psi, x), (x, 0, a))
mean_p2 = integrate(psi * (-(hbar**2)) * diff(psi, x, 2), (x, 0, a))
assert simplify(mean_x - a / 2) == 0 and mean_p == 0
computed = sqrt(mean_x2 - mean_x**2) * sqrt(mean_p2)
stated = hbar / 2 * sqrt(pi**2 / 3 - 2)  # Griffiths, Problem 2.4, with n = 1
assert simplify(computed - stated) == 0
assert N(stated / (hbar / 2)) > 1  # the uncertainty principle holds
expected = stated
```
