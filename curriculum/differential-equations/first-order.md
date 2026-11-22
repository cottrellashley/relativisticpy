# First-order equations

Linear, separable and exact equations of the first order, with their initial values, as
an introductory course on differential equations writes them.

## Exponential growth

**Level 1** · `de.first-order.growth` · Boyce and DiPrima (2012), section 2.5

The simplest model of a population, written in Leibniz's notation, which says that
\(y\) is a function of \(t\), with its initial value.

```latex
dsolve(\frac{dy}{dt} = r y, y(0) = y_0)
```

```python
t, r, y0 = symbols("t r y_0")
y = Function("y")
expected = dsolve(Eq(y(t).diff(t), r * y(t)), y(t), ics={y(0): y0})
assert expected.rhs == y0 * exp(r * t)
```

## A linear equation

**Level 2** · `de.first-order.linear` · Boyce and DiPrima (2012), section 2.1

A linear equation with a variable coefficient, written with primes as the book writes it,
solved with an integrating factor: \(y = t^2 + 1/t^2\).

```latex
dsolve(t y' + 2y = 4t^2, y(1) = 2)
```

```python
t = symbols("t")
y = Function("y")
expected = dsolve(Eq(t * y(t).diff(t) + 2 * y(t), 4 * t**2), y(t), ics={y(1): 2})
assert simplify(expected.rhs - (t**2 + 1 / t**2)) == 0
```

## A separable equation

**Level 2** · `de.first-order.separable` · Boyce and DiPrima (2012), section 2.2

A separable, nonlinear equation: of the two branches of its implicit solution, the initial
value picks \(y = 1 - \sqrt{x^3 + 2x^2 + 2x + 4}\).

```latex
dsolve(\frac{dy}{dx} = \frac{3x^2 + 4x + 2}{2(y - 1)}, y(0) = -1)
```

```python
x = symbols("x")
y = Function("y")
equation = Eq(y(x).diff(x), (3 * x**2 + 4 * x + 2) / (2 * (y(x) - 1)))
expected = dsolve(equation, y(x), ics={y(0): -1})
assert expected.rhs == 1 - sqrt(x**3 + 2 * x**2 + 2 * x + 4)
```

## An exact equation

**Level 3** · `de.first-order.exact` · Boyce and DiPrima (2012), section 2.6

\(M + N y' = 0\) is exact when \(M_y = N_x\). Its solutions are then the level curves
\(\psi(x, y) = c\) of a potential with \(\psi_x = M\) and \(\psi_y = N\), found by
integrating \(M\) in \(x\) and adding what \(N\) still needs in \(y\):
\(\psi = y \sin x + x^2 e^y - y\).

```latex
M(x, y) := y \cos x + 2x e^y
N(x, y) := \sin x + x^2 e^y - 1
\frac{\partial M}{\partial y} = \frac{\partial N}{\partial x}
\psi(x, y) := \int M \, dx
\psi + \int \left(N - \frac{\partial \psi}{\partial y}\right) dy
```

```python
x, y = symbols("x y")
M = y * cos(x) + 2 * x * exp(y)
N = sin(x) + x**2 * exp(y) - 1
assert simplify(diff(M, y) - diff(N, x)) == 0
psi = integrate(M, x)
potential = psi + integrate(N - diff(psi, y), y)
assert simplify(potential - (y * sin(x) + x**2 * exp(y) - y)) == 0
outputs = [True, potential]
```
