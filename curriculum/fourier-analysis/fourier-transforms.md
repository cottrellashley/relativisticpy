# Fourier transforms

The Fourier transform from its definition, in Bracewell's convention
\(F(s) = \int_{-\infty}^{\infty} f(x)\, e^{-i 2\pi x s}\, dx\), with his rectangle
function \(\Pi\), triangle function \(\Lambda\) and normalised sinc, and his notation
\(f(x) \supset F(s)\) for a pair; Rayleigh's theorem and the convolution theorem; the
discrete transform of a sampled sequence; and the transform of a constant, which is a
distribution.

## The transform of a Gaussian

**Level 2** · `fa.transforms.gaussian` · standard

\(e^{-\pi x^2}\) is its own Fourier transform.

```latex
\int_{-\infty}^{\infty} e^{-\pi x^2} e^{-i 2\pi x s} \, dx
```

```python
x, s = symbols("x s")
expected = simplify(integrate(exp(-pi * x**2) * exp(-I * 2 * pi * x * s), (x, -oo, oo)))
assert expected == exp(-pi * s**2)
```

## The transform of the rectangle function

**Level 2** · `fa.transforms.rectangle` · Bracewell (2000), chapter 4

The unit rectangle \(\Pi(x)\), defined by cases, transforms into
\(\operatorname{sinc} s = \sin(\pi s)/\pi s\).

```latex
\Pi(x) := \begin{cases} 1 & |x| < \frac{1}{2} \\ 0 & \text{otherwise} \end{cases}
\int_{-\infty}^{\infty} \Pi(x)\, e^{-i 2\pi x s} \, dx
```

```python
x, s = symbols("x s")
transform = integrate(exp(-I * 2 * pi * x * s), (x, -Rational(1, 2), Rational(1, 2)))
expected = sin(pi * s) / (pi * s)
assert all(abs(N((transform - expected).subs(s, value))) < 1e-12 for value in (0.3, 1.7, -2.4))
```

## Rayleigh's theorem

**Level 2** · `fa.transforms.rayleigh` · Bracewell (2000), chapter 6

The energy of a function equals the energy of its transform, which Bracewell calls
Rayleigh's theorem and others Parseval's: for the rectangle and the sinc function, both
are 1.

```latex
\Pi(x) := \begin{cases} 1 & |x| < \frac{1}{2} \\ 0 & \text{otherwise} \end{cases}
\operatorname{sinc}(s) := \frac{\sin \pi s}{\pi s}
\int_{-\infty}^{\infty} |\Pi(x)|^2 \, dx = \int_{-\infty}^{\infty} |\operatorname{sinc}(s)|^2 \, ds
```

```python
x, s = symbols("x s")
energy = integrate(Integer(1), (x, -Rational(1, 2), Rational(1, 2)))
spectrum = integrate((sin(pi * s) / (pi * s)) ** 2, (s, -oo, oo))
assert energy == spectrum == 1
expected = True
```

## The convolution theorem

**Level 3** · `fa.transforms.convolution` · Bracewell (2000), chapter 6

The transform of a convolution is the product of the transforms. For the Gaussian \(f\),
\(f * f\) is the Gaussian \(e^{-\pi x^2/2}/\sqrt{2}\), and its transform is
\(F(s)^2 = e^{-2\pi s^2}\).

```latex
f(x) := e^{-\pi x^2}
F(s) := \int_{-\infty}^{\infty} f(x)\, e^{-i 2\pi x s} \, dx
h(x) := \int_{-\infty}^{\infty} f(u)\, f(x - u) \, du
\int_{-\infty}^{\infty} h(x)\, e^{-i 2\pi x s} \, dx = F(s)^2
```

```python
x, s, u = symbols("x s u")
f = lambda z: exp(-pi * z**2)
F = simplify(integrate(f(x) * exp(-I * 2 * pi * x * s), (x, -oo, oo)))
h = simplify(integrate(f(u) * f(x - u), (u, -oo, oo)))
assert simplify(h - exp(-pi * x**2 / 2) / sqrt(2)) == 0
H = simplify(integrate(h * exp(-I * 2 * pi * x * s), (x, -oo, oo)))
assert simplify(H - F**2) == 0
expected = True
```

## Bracewell's notation for a pair

**Level 4** · `fa.transforms.pair` · Bracewell (2000), chapter 4

\(f(x) \supset F(s)\) says that \(F\) is the transform of \(f\): a claim to check, here for
the Gaussian and for the triangle function, whose transform is \(\operatorname{sinc}^2 s\).

```latex
\Lambda(x) := \begin{cases} 1 - |x| & |x| < 1 \\ 0 & \text{otherwise} \end{cases}
e^{-\pi x^2} \supset e^{-\pi s^2}
\Lambda(x) \supset \frac{\sin^2 \pi s}{\pi^2 s^2}
```

```python
x, s = symbols("x s")
gaussian = simplify(integrate(exp(-pi * x**2) * exp(-I * 2 * pi * x * s), (x, -oo, oo)))
assert gaussian == exp(-pi * s**2)
triangle = 2 * integrate((1 - x) * cos(2 * pi * x * s), (x, 0, 1))  # Λ is even
assert all(
    abs(N((triangle - sin(pi * s) ** 2 / (pi * s) ** 2).subs(s, v))) < 1e-12
    for v in (0.3, 1.7, -2.4)
)
outputs = [True, True]
```

## The discrete Fourier transform

**Level 2** · `fa.transforms.dft` · standard

The transform of eight samples, \(X_k = \sum_n x_n e^{-2\pi i k n/N}\) for
\(k = 0, \ldots, N - 1\), with the samples indexed from zero: the sequence of its values.

```latex
x := (1, 2, 0, -1, 3, 0, 0, 2)
N := 8
X_k := \sum_{n=0}^{N-1} x_n e^{-2\pi i k n/N}, \quad k = 0, \ldots, N-1
X_k
```

```python
samples = np.array([1, 2, 0, -1, 3, 0, 0, 2])
transform = np.fft.fft(samples)
direct = [sum(samples[n] * np.exp(-2j * np.pi * k * n / 8) for n in range(8)) for k in range(8)]
assert np.allclose(transform, direct)
expected = [complex(value) for value in transform]
```

## The transform of a constant

**Level 5** · `fa.transforms.delta` · Arfken, Weber and Harris (2013), section 1.11

\(\int e^{-i\omega t}\,dt\) converges for no \(\omega\), but as a distribution it is
\(2\pi\delta(\omega)\): the limit of the transforms of \(e^{-\epsilon|t|}\), Lorentzians of
area \(2\pi\) that narrow onto \(\omega = 0\).

```latex
\int_{-\infty}^{\infty} e^{-i \omega t} \, dt
```

```python
omega, t = symbols("omega t", real=True)
epsilon = symbols("epsilon", positive=True)
lorentzian = 2 * integrate(exp(-epsilon * t) * cos(omega * t), (t, 0, oo), conds="none")
assert simplify(lorentzian - 2 * epsilon / (epsilon**2 + omega**2)) == 0
assert integrate(lorentzian, (omega, -oo, oo)) == 2 * pi  # its area is 2 pi at every epsilon
assert limit(lorentzian.subs(omega, 1), epsilon, 0, "+") == 0  # and it vanishes away from 0
expected = 2 * pi * DiracDelta(omega)
```
