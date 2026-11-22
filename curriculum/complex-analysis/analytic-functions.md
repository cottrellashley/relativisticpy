# Analytic functions

The Cauchy–Riemann equations, harmonic conjugates and the derivative with respect to
\(\bar z\). Ahlfors (1979) is *Complex Analysis*, 3rd edition.

## The Cauchy–Riemann equations

**Level 2** · `complex.analytic.cauchy-riemann` · Ahlfors (1979), ch. 2, §1.2

The real and imaginary parts of \(e^z\) satisfy both Cauchy–Riemann equations, two
equations on one line.

```latex
u := e^x \cos y
v := e^x \sin y
\frac{\partial u}{\partial x} = \frac{\partial v}{\partial y} \quad \text{and} \quad \frac{\partial u}{\partial y} = -\frac{\partial v}{\partial x}
```

```python
x, y = symbols("x y", real=True)
u, v = re(exp(x + I * y)), im(exp(x + I * y))
assert simplify(diff(u, x) - diff(v, y)) == 0
assert simplify(diff(u, y) + diff(v, x)) == 0
outputs = [True, True]
```

## The conjugate is not analytic

**Level 2** · `complex.analytic.conjugate` · Ahlfors (1979), ch. 2, §1.2

For \(\bar z = x - iy\) the first Cauchy–Riemann equation fails: \(\partial u/\partial x = 1\)
but \(\partial v/\partial y = -1\).

```latex
u := x
v := -y
\frac{\partial u}{\partial x} = \frac{\partial v}{\partial y}
```

```python
x, y = symbols("x y", real=True)
u, v = re(conjugate(x + I * y)), im(conjugate(x + I * y))
assert (diff(u, x), diff(v, y)) == (1, -1)
expected = False
```

## A harmonic conjugate

**Level 3** · `complex.analytic.harmonic-conjugate` · Ahlfors (1979), ch. 2, §1.2

From the harmonic function \(u = x^3 - 3xy^2\), its conjugate \(v = \int u_x\,dy\) by the
first Cauchy–Riemann equation, and the analytic function \(u + iv\) they make, which is
\(z^3\).

```latex
u := x^3 - 3 x y^2
v := \int \frac{\partial u}{\partial x} \, dy
u + i v = (x + i y)^3
```

```python
x, y = symbols("x y", real=True)
u = x**3 - 3 * x * y**2
v = integrate(diff(u, x), y)
assert v == 3 * x**2 * y - y**3
assert expand(u + I * v - (x + I * y)**3) == 0
expected = True
```

## A Wirtinger derivative

**Level 4** · `complex.analytic.wirtinger` · standard

The derivative with respect to \(\bar z\), \(\partial/\partial\bar z = \frac{1}{2}(\partial_x +
i\,\partial_y)\), of \(|z|^2 = z\bar z\). It is \(z\), not 0, which shows that \(|z|^2\) is not
analytic: \(|z|\) depends on \(\bar z\).

```latex
\frac{\partial}{\partial \bar{z}} |z|^2
```

```python
x, y = symbols("x y", real=True)
f = x**2 + y**2  # |z|^2, with z = x + iy
assert expand((diff(f, x) + I * diff(f, y)) / 2) == x + I * y
z = symbols("z")
expected = z
```
