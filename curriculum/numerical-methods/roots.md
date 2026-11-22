# Roots of equations

Bisection, Newton's method and the secant method, written as the books write them: as
sequences defined by recurrences, \(p_n\) from \(p_{n-1}\), whose terms the language must
compute. Then the order of convergence, and Newton's method for a system.

## The number of bisection steps

**Level 1** · `num.roots.bisection-count` · Burden and Faires (2011), section 2.1

Bisection on \([1, 2]\) halves the bracket at each step, so an accuracy of \(10^{-3}\)
needs \(\lceil \log_2 10^3 \rceil = 10\) steps: the base of the logarithm as a subscript,
inside a ceiling.

```latex
\left\lceil \log_2 \frac{2 - 1}{10^{-3}} \right\rceil
```

```python
expected = ceiling(log(1000, 2))
assert expected == 10  # the book's ten iterations
assert 2.0**-10 < 1e-3 < 2.0**-9
```

## Newton's method

**Level 2** · `num.roots.newton` · Burden and Faires (2011), section 2.3

Newton's method for \(\cos x - x = 0\) from \(p_0 = \pi/4\), with the iterates defined by
the recurrence the book writes: \(p_2 = 0.7390851781\), already within \(5 \times 10^{-8}\)
of the root \(0.7390851332\).

```latex
f(x) := \cos x - x
p_0 := \frac{\pi}{4}
p_n := p_{n-1} - \frac{f(p_{n-1})}{f'(p_{n-1})}
p_2
```

```python
x = symbols("x")
f = cos(x) - x
p = pi / 4
for _ in range(2):
    p = p - f.subs(x, p) / diff(f, x).subs(x, p)
expected = N(p, 30)
assert abs(expected - 0.7390851781) < 5e-11  # the book's table
root = nsolve(f, x, 0.7, prec=30)
assert abs(expected - root) > 4e-8  # p_2 is not yet the root
```

## The secant method

**Level 2** · `num.roots.secant` · Burden and Faires (2011), section 2.3

The secant method for the same equation, a recurrence on the two previous iterates, from
\(p_0 = 0.5\) and \(p_1 = \pi/4\): \(p_4 = 0.7390851493\).

```latex
f(x) := \cos x - x
p_0 := 0.5
p_1 := \frac{\pi}{4}
p_n := p_{n-1} - \frac{f(p_{n-1})(p_{n-1} - p_{n-2})}{f(p_{n-1}) - f(p_{n-2})}
p_4
```

```python
import mpmath

mpmath.mp.dps = 30
f = lambda t: mpmath.cos(t) - t
p = [mpmath.mpf("0.5"), mpmath.pi / 4]
for _ in range(3):
    p.append(p[-1] - f(p[-1]) * (p[-1] - p[-2]) / (f(p[-1]) - f(p[-2])))
expected = Float(p[4], 30)
assert abs(expected - 0.7390851493) < 5e-11  # the book's table
assert abs(expected - mpmath.findroot(f, 0.7)) > 1e-8  # p_4 is not yet the root
```

## The order of Newton's method

**Level 3** · `num.roots.quadratic-convergence` · Burden and Faires (2011), section 2.4

Newton's method converges quadratically: near a simple root \(p\) the error is squared at
each step, \(p_{n+1} - p \approx \frac{f''(p)}{2f'(p)}(p_n - p)^2\). For \(x^2 - 2\) the
constant is \(\frac{\sqrt{2}}{4}\).

```latex
f(x) := x^2 - 2
g(x) := x - \frac{f(x)}{f'(x)}
p := \sqrt{2}
\lim_{x \to p} \frac{g(x) - p}{(x - p)^2}
```

```python
x = symbols("x")
f = x**2 - 2
g = x - f / diff(f, x)
expected = limit((g - sqrt(2)) / (x - sqrt(2)) ** 2, x, sqrt(2))
assert simplify(expected - sqrt(2) / 4) == 0
assert simplify(expected - (diff(f, x, 2) / (2 * diff(f, x))).subs(x, sqrt(2))) == 0
```

## Newton's method for a system

**Level 4** · `num.roots.newton-system` · Burden and Faires (2011), chapter 10

Two steps of Newton's method for \(x^2 + y^2 = 4\), \(xy = 1\), from \((2, 0)\), written
with the iterates labelled by superscripts in parentheses, \(\mathbf{x}^{(k)}\), and the
function and its Jacobian applied to a vector:
\(\mathbf{x}^{(k)} = \mathbf{x}^{(k-1)} - J(\mathbf{x}^{(k-1)})^{-1} F(\mathbf{x}^{(k-1)})\).

```latex
F(x, y) := \begin{pmatrix} x^2 + y^2 - 4 \\ x y - 1 \end{pmatrix}
J(x, y) := \begin{pmatrix} 2x & 2y \\ y & x \end{pmatrix}
\mathbf{x}^{(0)} := \begin{pmatrix} 2 \\ 0 \end{pmatrix}
\mathbf{x}^{(k)} := \mathbf{x}^{(k-1)} - J(\mathbf{x}^{(k-1)})^{-1} F(\mathbf{x}^{(k-1)})
\mathbf{x}^{(2)}
```

```python
x, y = symbols("x y")
F = Matrix([x**2 + y**2 - 4, x * y - 1])
J = F.jacobian([x, y])
assert J == Matrix([[2 * x, 2 * y], [y, x]])
point = Matrix([2, 0])
for _ in range(2):
    at = {x: point[0], y: point[1]}
    point = point - J.subs(at).inv() * F.subs(at)
expected = point
assert expected == Matrix([Rational(29, 15), Rational(31, 60)])
root = [np.sqrt(2 + np.sqrt(3)), np.sqrt(2 - np.sqrt(3))]  # where the iterates are going
assert np.allclose([float(v) for v in expected], root, atol=2e-3)
```
