# Expectations and generating functions

Expectations, variances and higher moments, from their definitions and from the
moment-generating and characteristic functions, \(\psi(t) = \mathbb{E}[e^{tX}]\) and
\(\phi(t) = \mathbb{E}[e^{itX}]\).

## The expectation of a die

**Level 1** · `prob.moments.die-mean` · standard

The mean of a fair die, as a sum over its faces.

```latex
\sum_{k=1}^{6} k \cdot \frac{1}{6}
```

```python
expected = Rational(7, 2)
assert sum(Rational(k, 6) for k in range(1, 7)) == expected
```

## The characteristic function of the normal distribution

**Level 2** · `prob.moments.normal-characteristic` · Feller (1971), section XV.2

The Fourier transform of the standard normal density, \(e^{-t^2/2}\).

```latex
\int_{-\infty}^{\infty} e^{i t x} \frac{1}{\sqrt{2\pi}} e^{-x^2/2} \, dx
```

```python
t, x = symbols("t x", real=True)
expected = exp(-(t**2) / 2)
phi = integrate(exp(I * t * x) * exp(-(x**2) / 2) / sqrt(2 * pi), (x, -oo, oo))
assert simplify(phi - expected) == 0
```

## The variance from the moment-generating function

**Level 2** · `prob.moments.variance-from-mgf` · DeGroot and Schervish (2012), section 5.4

The variance of the Poisson distribution, \(\psi''(0) - [\psi'(0)]^2 = \lambda\), from its
moment-generating function.

```latex
\psi(t) := e^{\lambda (e^{t} - 1)}
\psi''(0) - \left[\psi'(0)\right]^2
```

```python
t, lam = symbols("t lambda")
psi = exp(lam * (exp(t) - 1))
expected = simplify(diff(psi, t, 2).subs(t, 0) - diff(psi, t).subs(t, 0) ** 2)
assert expected == lam  # the variance of a Poisson variable is its mean
```

## The fourth moment of the normal distribution

**Level 3** · `prob.moments.normal-fourth-moment` · DeGroot and Schervish (2012), section 4.4

The moment-generating function of the standard normal from its definition, then its
fourth derivative at zero, written \(\psi^{(4)}(0)\): \(\mathbb{E}[X^4] = 3\).

```latex
\psi(t) := \int_{-\infty}^{\infty} e^{t x} \frac{1}{\sqrt{2\pi}} e^{-x^2/2} \, dx
\psi^{(4)}(0)
```

```python
from scipy.stats import norm

t, x = symbols("t x", real=True)
psi = integrate(exp(t * x) * exp(-(x**2) / 2) / sqrt(2 * pi), (x, -oo, oo))
expected = simplify(diff(psi, t, 4).subs(t, 0))
assert expected == 3
assert abs(norm.moment(4) - 3) < 1e-12
```

## The variance of an affine transformation

**Level 4** · `prob.moments.variance-affine` · DeGroot and Schervish (2012), section 4.3

\(\operatorname{Var}(aX + b) = a^2 \operatorname{Var}(X)\): the variance as an operator on
random variables, with the rules it obeys.

```latex
\operatorname{Var}(a X + b) = a^2 \operatorname{Var}(X)
```

```python
from sympy.stats import Exponential, Normal, variance

a, b, mu = symbols("a b mu", real=True)
sigma, rate = symbols("sigma lambda", positive=True)
for X in (Normal("X", mu, sigma), Exponential("X", rate)):
    assert simplify(variance(a * X + b) - a**2 * variance(X)) == 0
expected = True
```
