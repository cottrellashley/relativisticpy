# Multivariable calculus

Iterated integrals, Hessians and Jacobians, Lagrange multipliers, and integrals over a
region, in the notations of a calculus course. Stewart (2015) is *Calculus: Early
Transcendentals*, 8th edition.

## An iterated integral

**Level 1** · `analysis.multivariable.iterated` · Stewart (2015), §15.2

A double integral over the region between two parabolas, as an iterated integral whose
inner limits depend on the outer variable.

```latex
\int_{-1}^{1} \int_{2x^2}^{1 + x^2} (x + 2y) \, dy \, dx
```

```python
x, y = symbols("x y")
expected = integrate(x + 2 * y, (y, 2 * x**2, 1 + x**2), (x, -1, 1))
assert expected == Rational(32, 15)
```

## The second-derivative test

**Level 2** · `analysis.multivariable.second-derivative-test` · Stewart (2015), §14.7

The Hessian determinant \(D = f_{xx} f_{yy} - (f_{xy})^2\) at the critical point
\((1, 1)\), with partial derivatives written as subscripts: \(D > 0\) and \(f_{xx} > 0\)
make it a local minimum.

```latex
f(x, y) := x^4 + y^4 - 4xy + 1
D(x, y) := f_{xx}(x, y) \, f_{yy}(x, y) - \left[ f_{xy}(x, y) \right]^2
D(1, 1)
```

```python
x, y = symbols("x y")
f = x**4 + y**4 - 4 * x * y + 1
D = diff(f, x, 2) * diff(f, y, 2) - diff(f, x, y) ** 2
assert expand(D) == 144 * x**2 * y**2 - 16
expected = D.subs({x: 1, y: 1})
assert expected == 128
```

## The Jacobian of polar coordinates

**Level 2** · `analysis.multivariable.jacobian` · Stewart (2015), §15.9

The Jacobian determinant in its notation \(\partial(x, y)/\partial(r, \theta)\), of
coordinates the program defines.

```latex
x := r \cos\theta
y := r \sin\theta
\frac{\partial(x, y)}{\partial(r, \theta)}
```

```python
r, theta = symbols("r theta")
expected = simplify(Matrix([r * cos(theta), r * sin(theta)]).jacobian([r, theta]).det())
assert expected == r
```

## Lagrange multipliers

**Level 3** · `analysis.multivariable.lagrange` · Stewart (2015), §14.8, Example 2

The extreme values of \(x^2 + 2y^2\) on the unit circle, where the gradient of \(f\) is a
multiple \(\lambda\) of the gradient of the constraint. The four solutions give the minimum
1, at \((\pm 1, 0)\), and the maximum 2, at \((0, \pm 1)\).

```latex
f(x, y) := x^2 + 2y^2
g(x, y) := x^2 + y^2
solve([\nabla f = \lambda \nabla g, g(x, y) = 1], [x, y, \lambda])
```

```python
x, y, lam = symbols("x y lambda")
f, g = x**2 + 2 * y**2, x**2 + y**2
equations = [Eq(diff(f, x), lam * diff(g, x)), Eq(diff(f, y), lam * diff(g, y)), Eq(g, 1)]
expected = solve(equations, [x, y, lam])  # in the order SymPy gives them
assert sorted(f.subs({x: p, y: q}) for p, q, _ in expected) == [1, 1, 2, 2]
```

## A double integral over a disk

**Level 4** · `analysis.multivariable.region` · Stewart (2015), §15.3

An integral over a region given by an inequality, \(\iint_R f\,dA\): the language must find
the region's bounds, or change to polar coordinates itself.

```latex
\iint_{x^2 + y^2 \le 1} e^{-(x^2 + y^2)} \, dA
```

```python
r, theta = symbols("r theta", positive=True)
expected = integrate(exp(-(r**2)) * r, (r, 0, 1), (theta, 0, 2 * pi))
assert simplify(expected - pi * (1 - exp(-1))) == 0
```
