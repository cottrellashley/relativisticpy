# Differences, interpolation and rounding

Derivatives approximated by difference quotients, improved by Richardson's
extrapolation; the interpolating polynomial in Lagrange's form; and what rounding does
to a difference of nearly equal numbers.

## The midpoint formula for a derivative

**Level 1** · `num.differences.midpoint` · Burden and Faires (2011), section 4.1

The three-point midpoint formula \(\frac{1}{2h}[f(x_0 + h) - f(x_0 - h)]\) for
\(f(x) = x e^x\) at \(x_0 = 2\) with \(h = 0.1\), as the book prints it: 22.2288, against
\(f'(2) = 3e^2 = 22.1672\).

```latex
\frac{1}{0.2}\left[2.1 e^{2.1} - 1.9 e^{1.9}\right]
```

```python
x = symbols("x")
f = x * exp(x)
expected = (f.subs(x, Rational(21, 10)) - f.subs(x, Rational(19, 10))) / Rational(2, 10)
derivative = diff(f, x).subs(x, 2)
assert derivative == 3 * exp(2)
# the error is h^2/6 f'''(2) to leading order: 0.0616, as the book finds
assert abs(float(expected - derivative) - 0.01 / 6 * float(diff(f, x, 3).subs(x, 2))) < 1e-4
assert abs(float(expected - derivative) - 0.0616) < 5e-5
```

## Richardson's extrapolation

**Level 2** · `num.differences.richardson` · Burden and Faires (2011), section 4.2

The centred difference \(N_1(h)\) has an error of order \(h^2\); combining it at \(h\)
and \(h/2\) as \(N_2(h) = N_1(h/2) + \frac{N_1(h/2) - N_1(h)}{3}\) cancels that term. For
\(f(x) = e^x\) at 0, \(N_2(0.2) = 0.9999967\), against \(N_1(0.2) = 1.00668\). The book's
names \(N_1\) and \(N_2\) are functions of \(h\), not numerical evaluation.

```latex
f(x) := e^{x}
N_1(h) := \frac{f(h) - f(-h)}{2h}
N_2(h) := N_1\left(\frac{h}{2}\right) + \frac{N_1(h/2) - N_1(h)}{3}
N_2(0.2)
```

```python
h = symbols("h")
N1 = lambda step: (exp(step) - exp(-step)) / (2 * step)
expected = N1(Rational(1, 10)) + (N1(Rational(1, 10)) - N1(Rational(1, 5))) / 3
assert abs(float(expected) - 0.9999967) < 5e-8
assert abs(float(N1(Rational(1, 5))) - 1.00668) < 5e-6
# the extrapolated error is of order h^4: (N_2(h) - 1)/h^4 tends to -1/480
N2 = N1(h / 2) + (N1(h / 2) - N1(h)) / 3
assert limit((N2 - 1) / h**4, h, 0) == Rational(-1, 480)
```

## The Lagrange interpolating polynomial

**Level 3** · `num.interpolation.lagrange` · Burden and Faires (2011), section 3.1

The polynomial of degree two through \(f(x) = 1/x\) at the nodes 2, 2.75 and 4, built from
the Lagrange basis \(L_k(x) = \prod_{i \neq k} \frac{x - x_i}{x_k - x_i}\) and evaluated at
3: \(P(3) = 0.32955\), against \(f(3) = 0.33333\).

```latex
x_0 := 2
x_1 := 2.75
x_2 := 4
f(x) := \frac{1}{x}
L_k(x) := \prod_{\substack{i=0 \\ i \neq k}}^{2} \frac{x - x_i}{x_k - x_i}
P(x) := \sum_{k=0}^{2} f(x_k) L_k(x)
P(3)
```

```python
x = symbols("x")
nodes = [Integer(2), Rational(11, 4), Integer(4)]
P = sum(
    1 / nodes[k] * prod((x - nodes[i]) / (nodes[k] - nodes[i]) for i in range(3) if i != k)
    for k in range(3)
)
expected = P.subs(x, 3)
assert expected == Rational(29, 88)
assert abs(float(expected) - 0.32955) < 5e-6  # the book's value
coefficients = np.polyfit([2, 2.75, 4], [1 / 2, 1 / 2.75, 1 / 4], 2)
assert abs(np.polyval(coefficients, 3) - float(expected)) < 1e-12
```

## Cancellation in floating point

**Level 5** · `num.rounding.cancellation` · Higham (2002), chapter 1

\(\operatorname{fl}(\cdot)\), Higham's notation for a value computed in floating-point
arithmetic, here of unit roundoff \(u = 2^{-53}\), IEEE double precision. At \(x = 10^{-8}\),
\(\cos x\) rounds to exactly 1, so \(1 - \cos x\) cancels to nothing and the computed
\((1 - \cos x)/x^2\) is 0, though its exact value is \(\frac{1}{2}\) to sixteen digits. No
design of the language says how to ask for a computation in a given arithmetic.

```latex
u := 2^{-53}
x := 10^{-8}
\operatorname{fl}\left(\frac{1 - \cos x}{x^2}\right)
```

```python
import math

x = 1e-8
assert math.cos(x) == 1.0  # cos x rounds to 1 in double precision
expected = (1.0 - math.cos(x)) / x**2
assert expected == 0.0
t = symbols("t")
exact = ((1 - cos(t)) / t**2).subs(t, Rational(1, 10**8))
assert abs(N(exact, 30) - Rational(1, 2)) < 1e-16  # what exact arithmetic gives
```
