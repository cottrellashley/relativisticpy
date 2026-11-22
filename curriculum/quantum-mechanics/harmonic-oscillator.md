# The harmonic oscillator

The quantum oscillator two ways: analytically, with wave functions built from Hermite
polynomials, and algebraically, with the ladder operators that raise and lower its
evenly spaced levels, \(E_n = (n + \tfrac{1}{2})\hbar\omega\).

## Rodrigues' formula

**Level 1** · `qm.oscillator.rodrigues` · Griffiths (2005), section 2.3.2

A Hermite polynomial from Rodrigues' formula,
\(H_n(\xi) = (-1)^n e^{\xi^2} \frac{d^n}{d\xi^n} e^{-\xi^2}\), for \(n = 3\).

```latex
(-1)^3 e^{\xi^2} \frac{d^3}{d\xi^3} e^{-\xi^2}
```

```python
xi = symbols("xi")
expected = expand((-1) ** 3 * exp(xi**2) * diff(exp(-(xi**2)), xi, 3))
assert expected == expand(hermite(3, xi))
assert expected == 8 * xi**3 - 12 * xi  # Griffiths, Table 2.1
```

## The ground state

**Level 2** · `qm.oscillator.ground-state` · Griffiths (2005), section 2.3.1

The Gaussian ground state solves the time-independent Schrödinger equation, with energy
\(\hbar\omega/2\).

```latex
\psi_0(x) := \left(\frac{m\omega}{\pi\hbar}\right)^{1/4} e^{-\frac{m\omega}{2\hbar} x^2}
-\frac{\hbar^2}{2m} \frac{d^2\psi_0}{dx^2} + \frac{1}{2} m\omega^2 x^2 \psi_0 = \frac{1}{2}\hbar\omega\,\psi_0
```

```python
x = symbols("x", real=True)
m, omega, hbar = symbols("m omega hbar", positive=True)
psi0 = (m * omega / (pi * hbar)) ** Rational(1, 4) * exp(-m * omega * x**2 / (2 * hbar))
H_psi0 = -(hbar**2) / (2 * m) * diff(psi0, x, 2) + m * omega**2 * x**2 / 2 * psi0
assert simplify(H_psi0 - hbar * omega / 2 * psi0) == 0
assert integrate(psi0**2, (x, -oo, oo)) == 1
expected = True
```

## A stationary state with a Hermite polynomial

**Level 2** · `qm.oscillator.hermite-wavefunction` · Griffiths (2005), section 2.3.2

The stationary states \(\psi_n(x) = (m\omega/\pi\hbar)^{1/4} (2^n n!)^{-1/2} H_n(\xi)
e^{-\xi^2/2}\), with \(H_n\) the Hermite polynomials and \(\xi = \sqrt{m\omega/\hbar}\,x\):
the language must know \(H_n\) by its name. \(\psi_3\) is normalised.

```latex
\xi := \sqrt{\frac{m\omega}{\hbar}}\, x
\psi_n(x) := \left(\frac{m\omega}{\pi\hbar}\right)^{1/4} \frac{1}{\sqrt{2^n n!}} H_n(\xi) e^{-\xi^2/2}
\int_{-\infty}^{\infty} |\psi_3(x)|^2 \, dx
```

```python
x = symbols("x", real=True)
m, omega, hbar = symbols("m omega hbar", positive=True)
xi = sqrt(m * omega / hbar) * x


def psi(n):
    norm = (m * omega / (pi * hbar)) ** Rational(1, 4) / sqrt(2**n * factorial(n))
    return norm * hermite(n, xi) * exp(-(xi**2) / 2)


expected = simplify(integrate(psi(3) ** 2, (x, -oo, oo)))
assert expected == 1
assert simplify(integrate(psi(3) * psi(1), (x, -oo, oo))) == 0  # and orthogonal to psi_1
```

## The ladder operators

**Level 3** · `qm.oscillator.ladder-commutator` · Griffiths (2005), section 2.3.1

The raising and lowering operators, built from \(x\) and \(\hat p = -i\hbar\, d/dx\), and
written together with \(\pm\) as Griffiths writes them: \([a_-, a_+] = 1\).

```latex
\hat{p} := -i\hbar \frac{d}{dx}
a_\pm := \frac{1}{\sqrt{2\hbar m\omega}} \left(\mp i\hat{p} + m\omega x\right)
[a_-, a_+] = 1
```

```python
x = symbols("x", real=True)
m, omega, hbar = symbols("m omega hbar", positive=True)
f = Function("f")(x)  # a test function
p = lambda g: -I * hbar * diff(g, x)
a_plus = lambda g: (-I * p(g) + m * omega * x * g) / sqrt(2 * hbar * m * omega)
a_minus = lambda g: (I * p(g) + m * omega * x * g) / sqrt(2 * hbar * m * omega)
assert simplify(a_minus(a_plus(f)) - a_plus(a_minus(f)) - f) == 0
expected = True
```

## The potential energy of a stationary state

**Level 4** · `qm.oscillator.potential-energy` · Griffiths (2005), Example 2.5

Ladder operators acting on the number states, with no wave function:
\(\langle n|V|n\rangle = \tfrac{1}{2}\hbar\omega(n + \tfrac{1}{2})\), half the energy.

```latex
\langle n' | n \rangle = \delta_{n'n}
a_- |n\rangle = \sqrt{n}\, |n - 1\rangle
a_+ |n\rangle = \sqrt{n + 1}\, |n + 1\rangle
x := \sqrt{\frac{\hbar}{2m\omega}} \left(a_+ + a_-\right)
\langle n | \frac{1}{2} m\omega^2 x^2 | n \rangle
```

```python
m, omega, hbar, n = symbols("m omega hbar n")
size = 8
# Matrix elements in the number basis: <k|a_-|j> = sqrt(j) when k = j - 1.
a_minus = Matrix(size, size, lambda k, j: sqrt(j) if k == j - 1 else 0)
a_plus = a_minus.T
x = sqrt(hbar / (2 * m * omega)) * (a_plus + a_minus)
V = m * omega**2 / 2 * x**2
stated = hbar * omega * (n + Rational(1, 2)) / 2  # Griffiths, Example 2.5
for k in range(size - 1):  # the top of a truncated ladder is not a number state
    assert simplify(V[k, k] - stated.subs(n, k)) == 0
expected = stated
```
