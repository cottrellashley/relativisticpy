# Integrals

Definite and improper integrals, integration by substitution and by parts, and the Gaussian
integral, as a calculus course writes them. Stewart (2015) is *Calculus: Early
Transcendentals*, 8th edition.

## A substitution

**Level 1** · `analysis.integrals.substitution` · Stewart (2015), §5.5

A definite integral with its differential in the numerator, done in books by the
substitution \(u = 3 - 5x\).

```latex
\int_1^2 \frac{dx}{(3 - 5x)^2}
```

```python
x = symbols("x")
expected = integrate(1 / (3 - 5 * x) ** 2, (x, 1, 2))
assert expected == Rational(1, 14)
```

## Integration by parts

**Level 1** · `analysis.integrals.by-parts` · Stewart (2015), §7.1

The right-hand side of \(\int_0^1 x e^x\,dx = \left[x e^x\right]_0^1 - \int_0^1 e^x\,dx\):
the evaluation bracket, \(\left[F(x)\right]_a^b = F(b) - F(a)\).

```latex
\left[ x e^x \right]_0^1 - \int_0^1 e^x \, dx
```

```python
x = symbols("x")
F = x * exp(x)
expected = F.subs(x, 1) - F.subs(x, 0) - integrate(exp(x), (x, 0, 1))
assert expected == integrate(x * exp(x), (x, 0, 1)) == 1
```

## A Gaussian integral

**Level 2** · `analysis.integrals.gaussian` · standard

The Gaussian integral with a linear term, which completing the square reduces to
\(\int e^{-x^2} dx = \sqrt{\pi}\), for \(a > 0\) as the program states first.

```latex
a > 0
\int_{-\infty}^{\infty} e^{-a x^2 + b x} \, dx
```

```python
x, b = symbols("x b", real=True)
a = symbols("a", positive=True)
expected = sqrt(pi / a) * exp(b**2 / (4 * a))
assert simplify(integrate(exp(-a * x**2 + b * x), (x, -oo, oo)) - expected) == 0
```

## The Gaussian integral in polar coordinates

**Level 3** · `analysis.integrals.gaussian-polar` · standard

Poisson's derivation of \(\int e^{-x^2} dx = \sqrt{\pi}\): the square of the integral is an
integral over the plane, which polar coordinates make elementary.

```latex
I := \int_{-\infty}^{\infty} e^{-x^2} \, dx
I^2 = \int_0^{2\pi} \int_0^{\infty} e^{-r^2} r \, dr \, d\theta
```

```python
x = symbols("x")
r, theta = symbols("r theta", positive=True)
assert integrate(exp(-(x**2)), (x, -oo, oo)) ** 2 == pi
assert integrate(exp(-(r**2)) * r, (r, 0, oo), (theta, 0, 2 * pi)) == pi
expected = True
```
