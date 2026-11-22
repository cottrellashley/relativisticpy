# Distributions

The standard distributions of probability, discrete and continuous: the binomial and
the Poisson, the normal and the gamma, with their probabilities and moments, as a course
in probability writes them.

## Three heads in ten tosses

**Level 1** · `prob.distributions.binomial-pmf` · Feller (1968), section VI.2

The binomial probability of three successes in ten fair trials, written out.

```latex
\binom{10}{3} \left(\frac{1}{2}\right)^{3} \left(1 - \frac{1}{2}\right)^{7}
```

```python
from scipy.stats import binom

expected = binomial(10, 3) * Rational(1, 2) ** 10
assert expected == Rational(15, 128)
assert abs(float(expected) - binom.pmf(3, 10, 0.5)) < 1e-15
```

## The Poisson limit of the binomial

**Level 1** · `prob.distributions.poisson-limit` · Feller (1968), section VI.5

Poisson's theorem as Feller states it: with \(p = \lambda / n\), the probability of
\(k\) successes in \(n\) trials tends to \(e^{-\lambda} \lambda^k / k!\). One limit, for
every \(k\) at once.

```latex
\lim_{n \to \infty} \binom{n}{k} \left(\frac{\lambda}{n}\right)^{k} \left(1 - \frac{\lambda}{n}\right)^{n - k}
```

```python
n, k, lam = symbols("n k lambda", positive=True)
expected = lam**k * exp(-lam) / factorial(k)
found = limit(binomial(n, k) * (lam / n) ** k * (1 - lam / n) ** (n - k), n, oo)
assert simplify(found - expected) == 0
```

## One standard deviation of the normal distribution

**Level 1** · `prob.distributions.normal-one-sigma` · DeGroot and Schervish (2012), section 5.6

The probability that a standard normal variable falls within one standard deviation of
its mean, as an integral of its density: about 0.6827.

```latex
\frac{1}{\sqrt{2\pi}} \int_{-1}^{1} e^{-x^2/2} \, dx
```

```python
from scipy.stats import norm

expected = erf(1 / sqrt(2))
assert abs(float(expected) - (norm.cdf(1) - norm.cdf(-1))) < 1e-12
assert abs(float(expected) - 0.6827) < 1e-4
```

## The variance of the gamma distribution

**Level 2** · `prob.distributions.gamma-variance` · DeGroot and Schervish (2012), section 5.7

The variance \(\alpha / \beta^2\) from the density, with Euler's gamma function written
\(\Gamma(\alpha)\), as every book on probability writes it.

```latex
f(x) := \frac{\beta^{\alpha}}{\Gamma(\alpha)} x^{\alpha - 1} e^{-\beta x}
\int_0^\infty x^2 f(x) \, dx - \left(\int_0^\infty x f(x) \, dx\right)^2
```

```python
alpha, beta, x = symbols("alpha beta x", positive=True)
f = beta**alpha / gamma(alpha) * x ** (alpha - 1) * exp(-beta * x)
mean = integrate(x * f, (x, 0, oo))
expected = simplify(integrate(x**2 * f, (x, 0, oo)) - mean**2)
assert expected == alpha / beta**2
```

## A random variable given by its distribution

**Level 4** · `prob.distributions.poisson-tilde` · DeGroot and Schervish (2012), section 5.4

A random variable declared by the distribution it follows, \(X \sim
\operatorname{Poisson}(\lambda)\), and a probability of an event about it.

```latex
X \sim \operatorname{Poisson}(\lambda)
P(X = 2)
```

```python
from scipy.stats import poisson
from sympy.stats import P, Poisson

lam = symbols("lambda", positive=True)
expected = lam**2 * exp(-lam) / factorial(2)
assert simplify(P(Eq(Poisson("X", lam), 2)) - expected) == 0
assert abs(float(expected.subs(lam, 1.3)) - poisson.pmf(2, 1.3)) < 1e-15
```
