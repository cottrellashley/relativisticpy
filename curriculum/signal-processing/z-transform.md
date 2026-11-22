# The z-transform

The z-transform as its defining sum, with its region of convergence stated; the inverse
transform by its book name, \(\mathcal{Z}^{-1}\); and the transfer function of a system
read from its difference equation.

## The transform of a geometric sequence

**Level 2** · `sig.z.geometric` · Oppenheim and Schafer (2010), chapter 3

\(X(z) = \sum_n x[n] z^{-n}\) for \(x[n] = (\frac{1}{2})^n u[n]\): a geometric series that
converges for \(|z| > \frac{1}{2}\), the region the program states, to
\(\frac{z}{z - 1/2}\).

```latex
|z| > \frac{1}{2}
x[n] := \left(\frac{1}{2}\right)^n u[n]
X(z) := \sum_{n=-\infty}^{\infty} x[n] z^{-n}
X(z)
```

```python
z = symbols("z")
n = symbols("n", integer=True, nonnegative=True)
expected = z / (z - Rational(1, 2))
w = symbols("w")  # w = 1/(2z), with |w| < 1 in the region of convergence
assert summation(w**n, (n, 0, oo)).args[0] == (1 / (1 - w), Abs(w) < 1)
assert simplify(1 / (1 - 1 / (2 * z)) - expected) == 0
point = 0.9 + 0.4j
assert abs(sum((0.5 / point) ** m for m in range(200)) - point / (point - 0.5)) < 1e-12
```

## The inverse transform by partial fractions

**Level 3** · `sig.z.inverse` · Oppenheim and Schafer (2010), chapter 3

\(\mathcal{Z}^{-1}\), with the region \(|z| > \frac{1}{2}\) choosing the right-sided
sequence: partial fractions give \(x[n] = 2(\frac{1}{2})^n u[n] - (\frac{1}{4})^n u[n]\), so
\(x[3] = \frac{15}{64}\).

```latex
|z| > \frac{1}{2}
x[n] := \mathcal{Z}^{-1}\left\{\frac{1}{\left(1 - \frac{1}{4}z^{-1}\right)\left(1 - \frac{1}{2}z^{-1}\right)}\right\}
x[3]
```

```python
from scipy.signal import lfilter

z = symbols("z")
X = 1 / ((1 - 1 / (4 * z)) * (1 - 1 / (2 * z)))
w = symbols("w")  # w = 1/z
assert simplify(apart(X.subs(z, 1 / w), w) - (2 / (1 - w / 2) - 1 / (1 - w / 4))) == 0
x = lambda m: 2 * Rational(1, 2) ** m - Rational(1, 4) ** m
expected = x(3)
assert expected == Rational(15, 64)
denominator = np.polymul([1, -0.25], [1, -0.5])
impulse_response = lfilter([1], denominator, [1, 0, 0, 0, 0])
assert np.allclose(impulse_response, [float(x(m)) for m in range(5)])
```

## The transfer function of a difference equation

**Level 4** · `sig.z.transfer-function` · Oppenheim and Schafer (2010), chapter 5

A recursive system given by its difference equation. Transforming both sides, a delay
\(y[n - k]\) becomes \(z^{-k} Y(z)\), and the ratio of the transforms of output and input,
written with their capital letters, is
\(H(z) = \frac{1}{1 - \frac{3}{4}z^{-1} + \frac{1}{8}z^{-2}}\).

```latex
y[n] - \frac{3}{4} y[n-1] + \frac{1}{8} y[n-2] = x[n]
H(z) := \frac{Y(z)}{X(z)}
H(z)
```

```python
from scipy.signal import freqz

z = symbols("z")
expected = 1 / (1 - Rational(3, 4) / z + Rational(1, 8) / z**2)
assert sorted(solve(z**2 - Rational(3, 4) * z + Rational(1, 8), z)) == [
    Rational(1, 4),
    Rational(1, 2),
]  # its poles
_, response = freqz([1], [1, -0.75, 0.125], worN=[0.3])
assert abs(response[0] - complex(expected.subs(z, exp(I * Rational(3, 10))))) < 1e-12
```
