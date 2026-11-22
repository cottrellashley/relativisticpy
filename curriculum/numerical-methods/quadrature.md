# Quadrature

Integrals approximated from values of the integrand: Simpson's rule, the composite
trapezoidal rule, Gaussian quadrature and Romberg's table, with the errors the books
derive for them, as Burden and Faires write them.

## Simpson's rule

**Level 1** · `num.quadrature.simpson` · Burden and Faires (2011), section 4.3

Simpson's rule for \(\int_0^2 e^x\,dx\), with \(h = 1\), written with the square brackets
of the book: \(\frac{h}{3}[f(x_0) + 4f(x_1) + f(x_2)] = 6.421\), against the exact
\(e^2 - 1 = 6.389\).

```latex
\frac{1}{3}\left[e^{0} + 4e^{1} + e^{2}\right]
```

```python
expected = (1 + 4 * E + E**2) / 3
assert abs(float(expected) - 6.421) < 5e-4  # the book's table
assert abs(float(E**2 - 1) - 6.389) < 5e-4  # and the integral itself
```

## The composite trapezoidal rule

**Level 2** · `num.quadrature.composite-trapezoid` · Burden and Faires (2011), section 4.4

The trapezoidal rule on four subintervals of \([0, \pi]\), for \(\int_0^\pi \sin x\,dx = 2\),
with the nodes \(x_j = a + jh\) defined as a family and summed over:
\(\frac{\pi}{4}(1 + \sqrt{2}) = 1.896\).

```latex
f(x) := \sin x
a := 0
b := \pi
n := 4
h := \frac{b - a}{n}
x_j := a + j h
\frac{h}{2}\left[f(a) + 2\sum_{j=1}^{n-1} f(x_j) + f(b)\right]
```

```python
h = pi / 4
nodes = [j * h for j in range(5)]
expected = h / 2 * (sin(nodes[0]) + 2 * sum(sin(x) for x in nodes[1:4]) + sin(nodes[4]))
assert simplify(expected - pi * (1 + sqrt(2)) / 4) == 0
values = np.sin(np.linspace(0, np.pi, 5))
rule = np.pi / 4 * (values[0] / 2 + values[1:-1].sum() + values[-1] / 2)
assert abs(rule - float(expected)) < 1e-12
```

## Two-point Gaussian quadrature

**Level 2** · `num.quadrature.gauss-two-point` · Burden and Faires (2011), section 4.7

The Gauss–Legendre rule with the nodes \(\pm 1/\sqrt{3}\) and unit weights integrates every
cubic exactly over \([-1, 1]\): a claim with both sides known, which holds.

```latex
f(x) := x^3 + 2x^2 - x + 1
f\left(-\frac{1}{\sqrt{3}}\right) + f\left(\frac{1}{\sqrt{3}}\right) = \int_{-1}^{1} f(x)\,dx
```

```python
x = symbols("x")
f = x**3 + 2 * x**2 - x + 1
rule = f.subs(x, -1 / sqrt(3)) + f.subs(x, 1 / sqrt(3))
assert simplify(rule - integrate(f, (x, -1, 1))) == 0  # both are 10/3
nodes, weights = np.polynomial.legendre.leggauss(2)
assert np.allclose(nodes, [-1 / np.sqrt(3), 1 / np.sqrt(3)]) and np.allclose(weights, [1, 1])
expected = True
```

## The error of Simpson's rule

**Level 3** · `num.quadrature.simpson-error` · Burden and Faires (2011), section 4.3

Simpson's rule has degree of precision three: its error is
\(-\frac{h^5}{90} f^{(4)}(\xi)\) for some \(\xi\) between the nodes. For \(x^4\) the fourth
derivative is the constant 24, so the error term is exact whatever \(\xi\) is.

```latex
f(x) := x^4
h := 1
\int_0^2 f(x)\,dx - \frac{h}{3}\left[f(0) + 4f(1) + f(2)\right] = -\frac{h^5}{90} f^{(4)}(\xi)
```

```python
x = symbols("x")
f = x**4
simpson = Rational(1, 3) * (f.subs(x, 0) + 4 * f.subs(x, 1) + f.subs(x, 2))
difference = integrate(f, (x, 0, 2)) - simpson
assert difference == Rational(-4, 15)
assert diff(f, x, 4) == 24 and -Rational(1, 90) * 24 == difference  # for every xi
expected = True
```

## The trapezoidal rule's error, asymptotically

**Level 3** · `num.quadrature.trapezoid-asymptotics` · standard

The composite trapezoidal rule with \(n\) subintervals has an error of order \(h^2\): for
\(\int_0^1 e^x\,dx\), \(n^2 (T_n - \int_0^1 e^x\,dx)\) tends to
\(\frac{1}{12}[f'(1) - f'(0)] = \frac{e - 1}{12}\), the first term of the Euler–Maclaurin
formula. The sum must be found in closed form before the limit is taken.

```latex
f(x) := e^x
T_n := \frac{1}{n}\left[\frac{f(0) + f(1)}{2} + \sum_{j=1}^{n-1} f\left(\frac{j}{n}\right)\right]
\lim_{n \to \infty} n^2 \left(T_n - \int_0^1 f(x)\,dx\right)
```

```python
h = symbols("h", positive=True)
# The geometric sum in closed form: T_n = (e - 1) h (e^h + 1) / (2 (e^h - 1)), with h = 1/n.
trapezoid = (E - 1) * h * (exp(h) + 1) / (2 * (exp(h) - 1))
expected = limit((trapezoid - (E - 1)) / h**2, h, 0)
assert simplify(expected - (E - 1) / 12) == 0
assert series(trapezoid, h, 0, 4).removeO() == E - 1 + (E - 1) / 12 * h**2
n = 1000
values = np.exp(np.linspace(0, 1, n + 1))
rule = (values[0] / 2 + values[1:-1].sum() + values[-1] / 2) / n
assert abs(n**2 * (rule - (np.e - 1)) - float(expected)) < 1e-6
```

## Romberg integration

**Level 4** · `num.quadrature.romberg` · Burden and Faires (2011), section 4.5

Romberg's table for \(\int_0^\pi \sin x\,dx\): the trapezoidal rule on \(2^{k-1}\)
subintervals in the first column, and Richardson's extrapolation across it,
\(R_{k,j} = R_{k,j-1} + (R_{k,j-1} - R_{k-1,j-1})/(4^{j-1} - 1)\). A sequence with two
indices, defined by a recurrence: \(R_{3,3} = 1.99857073\).

```latex
f(x) := \sin x
a := 0
b := \pi
h_k := \frac{b - a}{2^{k-1}}
R_{k,1} := \frac{h_k}{2}\left[f(a) + f(b) + 2\sum_{i=1}^{2^{k-1}-1} f(a + i h_k)\right]
R_{k,j} := R_{k,j-1} + \frac{R_{k,j-1} - R_{k-1,j-1}}{4^{j-1} - 1}, \quad j = 2, \ldots, k
R_{3,3}
```

```python
R = {}
for k in (1, 2, 3):
    h = pi / 2 ** (k - 1)
    R[k, 1] = h / 2 * (sin(0) + sin(pi) + 2 * sum(sin(i * h) for i in range(1, 2 ** (k - 1))))
for k in (2, 3):
    for j in range(2, k + 1):
        R[k, j] = R[k, j - 1] + (R[k, j - 1] - R[k - 1, j - 1]) / (4 ** (j - 1) - 1)
expected = simplify(R[3, 3])
assert simplify(expected - pi * (6 + 16 * sqrt(2)) / 45) == 0
assert abs(float(expected) - 1.99857073) < 5e-9  # the book's table
```
