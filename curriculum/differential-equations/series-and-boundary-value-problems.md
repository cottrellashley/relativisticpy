# Series solutions and boundary value problems

Equations with variable coefficients solved by power series, about an ordinary point and
a regular singular one, and problems on an interval with conditions at both ends: a
Green's function, the Rayleigh quotient, and the eigenvalues of a Sturm–Liouville problem.

## Airy's equation by power series

**Level 3** · `de.series.airy` · Boyce and DiPrima (2012), section 5.2

\(y = \sum a_n x^n\) put into \(y'' = xy\) gives the recurrence
\((n + 2)(n + 1) a_{n+2} = a_{n-1}\), with \(a_2 = 0\). Started from \(y(0) = 1\),
\(y'(0) = 0\), it gives the series of the first solution, here to \(x^9\).

```latex
a_0 := 1
a_1 := 0
a_2 := 0
a_{n+2} := \frac{a_{n-1}}{(n + 2)(n + 1)}
\sum_{n=0}^{9} a_n x^n
```

```python
x = symbols("x")
a = {0: Integer(1), 1: Integer(0), 2: Integer(0)}
for n in range(1, 8):
    a[n + 2] = a[n - 1] / ((n + 2) * (n + 1))
expected = sum(a[n] * x**n for n in range(10))
assert expected == 1 + x**3 / 6 + x**6 / 180 + x**9 / 12960
assert expand(diff(expected, x, 2) - x * expected) == -x**10 / 12960
```

## The exponents of an Euler equation

**Level 4** · `de.series.euler-exponents` · Boyce and DiPrima (2012), section 5.4

The book's operator notation: \(L[y]\) is the left side of the equation, and for an Euler
equation \(L[x^r] = x^r F(r)\), so the powers \(x^r\) that solve it are the roots of
\(F(r) = 2r^2 + r - 1\).

```latex
L[y] := 2x^2 y'' + 3x y' - y
solve(L[x^r] = 0, r)
```

```python
x, r = symbols("x r")
L = lambda y: 2 * x**2 * diff(y, x, 2) + 3 * x * diff(y, x) - y
assert expand(simplify(L(x**r) / x**r)) == 2 * r**2 + r - 1
expected = solve(Eq(L(x**r), 0), r)
assert expected == [-1, Rational(1, 2)]
```

## A Frobenius series

**Level 4** · `de.series.frobenius` · Boyce and DiPrima (2012), section 5.5

At the regular singular point \(x = 0\) of \(2x^2 y'' - x y' + (1 + x) y = 0\), the
solution for the exponent \(r = 1\), printed as the book prints it, the product of odd
numbers written with an ellipsis. It solves the equation.

```latex
y_1(x) := x \left[1 + \sum_{n=1}^{\infty} \frac{(-1)^n x^n}{n! \, [3 \cdot 5 \cdot 7 \cdots (2n + 1)]}\right]
2x^2 y_1'' - x y_1' + (1 + x) y_1 = 0
```

```python
x = symbols("x", positive=True)
n = symbols("n", integer=True, positive=True)
assert [factorial2(2 * m + 1) for m in range(1, 5)] == [3, 3 * 5, 3 * 5 * 7, 3 * 5 * 7 * 9]
y1 = x * (1 + summation((-1) ** n * x**n / (factorial(n) * factorial2(2 * n + 1)), (n, 1, oo)))
assert simplify(2 * x**2 * diff(y1, x, 2) - x * diff(y1, x) + (1 + x) * y1) == 0
expected = True
```

## A Green's function

**Level 3** · `de.bvp.green` · Arfken, Weber and Harris (2013), section 10.1

The solution of \(-u'' = f\) with \(u(0) = u(1) = 0\), as the integral of \(f\) against a
Green's function, a kernel in two variables defined piecewise. For \(f(\xi) = \xi\) the
solution is \((x - x^3)/6\), and its value at the midpoint is \(1/16\).

```latex
G(x, \xi) := \begin{cases} x (1 - \xi) & \text{if } x \le \xi \\ \xi (1 - x) & \text{otherwise} \end{cases}
u(x) := \int_0^1 G(x, \xi) \, \xi \, d\xi
u\left(\tfrac{1}{2}\right)
```

```python
x, xi = symbols("x xi")
u = Function("u")
solution = dsolve(Eq(-u(x).diff(x, 2), x), u(x), ics={u(0): 0, u(1): 0}).rhs
assert simplify(solution - (x - x**3) / 6) == 0
G = Piecewise((x * (1 - xi), x <= xi), (xi * (1 - x), True))
expected = integrate(G.subs(x, Rational(1, 2)) * xi, (xi, 0, 1))
assert expected == solution.subs(x, Rational(1, 2)) == Rational(1, 16)
```

## The Rayleigh quotient

**Level 4** · `de.bvp.rayleigh` · Strauss (2008), section 11.2

A functional, written with square brackets: the Rayleigh quotient of a trial function
that vanishes at both ends bounds the lowest eigenvalue of \(-y'' = \lambda y\),
\(y(0) = y(1) = 0\), from above. For \(x(1 - x)\) it is 10, against \(\pi^2 \approx 9.87\).

```latex
R[y] := \frac{\int_0^1 y'(x)^2 \, dx}{\int_0^1 y(x)^2 \, dx}
R[x(1 - x)]
```

```python
x = symbols("x")
trial = x * (1 - x)
expected = integrate(diff(trial, x) ** 2, (x, 0, 1)) / integrate(trial**2, (x, 0, 1))
assert expected == 10 and expected > pi**2
```

## An eigenvalue of a Sturm–Liouville problem

**Level 5** · `de.bvp.robin` · Boyce and DiPrima (2012), section 11.1

A boundary value problem with a Robin condition at one end has nontrivial solutions only
for its eigenvalues, the roots of \(\sin\sqrt\lambda + \sqrt\lambda \cos\sqrt\lambda = 0\),
which no formula gives. The smallest is \(\lambda_1 \approx 4.1159\).

```latex
y'' + \lambda y = 0, \quad y(0) = 0, \quad y'(1) + y(1) = 0
\lambda_1
```

```python
from scipy.optimize import brentq

root = brentq(lambda z: np.sin(z) + z * np.cos(z), np.pi / 2, np.pi)
expected = root**2
assert abs(expected - 4.1159) < 1e-4
tolerance = 1e-8
```
