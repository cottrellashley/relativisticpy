# Fourier series

The coefficients of a Fourier series, the series' convergence at a point, Parseval's
identity, and the overshoot of the partial sums at a jump.

## A Fourier coefficient

**Level 1** · `fa.series.coefficient` · Boyce and DiPrima (2012), section 10.2

The coefficient of \(\cos 3x\) in the Fourier series of \(x^2\) on \([-\pi, \pi]\), from
its defining integral.

```latex
\frac{1}{\pi} \int_{-\pi}^{\pi} x^2 \cos 3x \, dx
```

```python
x = symbols("x")
expected = integrate(x**2 * cos(3 * x), (x, -pi, pi)) / pi
assert expected == 4 * (-1) ** 3 / Integer(3) ** 2
```

## Convergence at a point

**Level 3** · `fa.series.convergence` · Boyce and DiPrima (2012), section 10.3

The square wave, \(-1\) on \((-\pi, 0)\) and \(1\) on \((0, \pi)\), has the sine
coefficients \(b_n\); where the wave is continuous its Fourier series converges to it, so
at \(x = \pi/2\) the series sums to 1.

```latex
b_n := \frac{2}{\pi} \int_0^{\pi} \sin nx \, dx
\sum_{n=1}^{\infty} b_n \sin\frac{n \pi}{2}
```

```python
x = symbols("x")
k = symbols("k", integer=True, nonnegative=True)
b = lambda n: 2 / pi * integrate(sin(n * x), (x, 0, pi))
# b_n is 4/(n pi) for odd n and zero for even n, and sin(n pi/2) = (-1)^k for n = 2k + 1
assert all(b(n) == (4 / (n * pi) if n % 2 else 0) for n in range(1, 9))
assert all(sin((2 * j + 1) * pi / 2) == (-1) ** j for j in range(6))
expected = summation(4 / ((2 * k + 1) * pi) * (-1) ** k, (k, 0, oo))
assert expected == 1
```

## Parseval's identity

**Level 3** · `fa.series.parseval` · standard

For \(f(x) = x\) on \((-\pi, \pi)\), the mean square of the function is the sum of the
squares of its coefficients \(b_n = 2(-1)^{n+1}/n\); both sides are \(2\pi^2/3\), which is
Euler's \(\sum 1/n^2 = \pi^2/6\).

```latex
b_n := \frac{1}{\pi} \int_{-\pi}^{\pi} x \sin nx \, dx
\frac{1}{\pi} \int_{-\pi}^{\pi} x^2 \, dx = \sum_{n=1}^{\infty} b_n^2
```

```python
x = symbols("x")
n = symbols("n", integer=True, positive=True)
b = lambda m: integrate(x * sin(m * x), (x, -pi, pi)) / pi
assert all(b(m) == 2 * (-1) ** (m + 1) / Integer(m) for m in range(1, 9))
left = integrate(x**2, (x, -pi, pi)) / pi
right = summation((2 * (-1) ** (n + 1) / n) ** 2, (n, 1, oo))
assert left == right == 2 * pi**2 / 3
expected = True
```

## The Gibbs phenomenon

**Level 4** · `fa.series.gibbs` · Arfken, Weber and Harris (2013), section 19.3

The partial sums of the square wave's series overshoot at the jump, and the overshoot does
not die away: at their first maximum, \(x = \pi/2N\), they tend to
\(\frac{2}{\pi}\operatorname{Si}(\pi) \approx 1.179\), not 1. A limit of a sum whose
number of terms and argument both depend on \(N\).

```latex
S_N(x) := \frac{4}{\pi} \sum_{k=1}^{N} \frac{\sin (2k - 1) x}{2k - 1}
\lim_{N \to \infty} S_N\left(\frac{\pi}{2N}\right)
```

```python
expected = 2 * Si(pi) / pi
partial = lambda N: 4 / np.pi * sum(np.sin((2 * k - 1) * np.pi / (2 * N)) / (2 * k - 1) for k in range(1, N + 1))
assert abs(partial(4000) - float(expected)) < 1e-6
assert abs(float(expected) - 1.17898) < 1e-5
```
