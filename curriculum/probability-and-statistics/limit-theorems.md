# Sums of random variables and limit theorems

The distribution of a sum, the law of large numbers through Chebyshev's inequality, and
the central limit theorem, by moment-generating functions and checked with numbers.

## The sum of two exponential waiting times

**Level 2** · `prob.limits.exponential-convolution` · DeGroot and Schervish (2012), section 3.9

The density of the sum of two independent exponential variables is the convolution of
their densities: a gamma density, \(\lambda^2 z e^{-\lambda z}\).

```latex
\int_0^z \lambda e^{-\lambda x} \, \lambda e^{-\lambda (z - x)} \, dx
```

```python
from scipy.stats import gamma as gamma_distribution

lam, x, z = symbols("lambda x z", positive=True)
expected = integrate(lam * exp(-lam * x) * lam * exp(-lam * (z - x)), (x, 0, z))
assert simplify(expected - lam**2 * z * exp(-lam * z)) == 0
value = float(expected.subs({lam: 1.5, z: 0.8}))
assert abs(value - gamma_distribution(a=2, scale=1 / 1.5).pdf(0.8)) < 1e-12  # Gamma(2, lambda)
```

## How many tosses the law of large numbers needs

**Level 2** · `prob.limits.chebyshev` · DeGroot and Schervish (2012), section 6.2

Chebyshev's inequality bounds \(\Pr(|\bar X_n - \mu| \ge \varepsilon)\) by
\(\sigma^2 / (n \varepsilon^2)\). For a fair coin, of variance \(\sigma^2 = 1/4\), named
as statistics names it, this many tosses make the mean fall within 0.1 of one half with
probability at least 0.7.

```latex
\sigma^2 := \frac{1}{4}
\varepsilon := 0.1
\left\lceil \frac{\sigma^2}{0.3 \, \varepsilon^2} \right\rceil
```

```python
variance, epsilon = Rational(1, 4), Rational(1, 10)
expected = ceiling(variance / (Rational(3, 10) * epsilon**2))
assert expected == 84
assert 1 - variance / (84 * epsilon**2) >= Rational(7, 10) > 1 - variance / (83 * epsilon**2)
```

## The central limit theorem by moment-generating functions

**Level 2** · `prob.limits.clt-mgf` · DeGroot and Schervish (2012), section 6.3

A step of \(\pm 1\) has the moment-generating function \(\cosh t\). The sum of \(n\)
steps, divided by \(\sqrt{n}\), has \(\psi(t/\sqrt{n})^n\), which tends to \(e^{t^2/2}\):
that of the standard normal distribution.

```latex
\psi(t) := \frac{1}{2} e^{t} + \frac{1}{2} e^{-t}
\lim_{n \to \infty} \psi\left(\frac{t}{\sqrt{n}}\right)^{n}
```

```python
t, n = symbols("t n", positive=True)
expected = exp(t**2 / 2)
assert limit(cosh(t / sqrt(n)) ** n, n, oo) == expected
```

## The normal approximation to the binomial

**Level 2** · `prob.limits.normal-approximation` · DeGroot and Schervish (2012), section 6.4

The central limit theorem checked with numbers: the probability of at most 55 heads in
100 tosses, by the normal approximation with the correction for continuity, where
\(\Phi\) is the standard normal distribution function. It agrees with the exact binomial
sum, 0.86437, to within \(10^{-4}\).

```latex
n := 100
p := \frac{1}{2}
\Phi\left(\frac{55.5 - n p}{\sqrt{n p (1 - p)}}\right)
```

```python
from scipy.stats import norm

expected = (1 + erf(Rational(11, 10) / sqrt(2))) / 2  # Phi(1.1)
assert abs(float(expected) - norm.cdf(1.1)) < 1e-14
exact = sum(binomial(100, k) for k in range(56)) / Integer(2) ** 100
assert abs(float(exact) - 0.86437) < 1e-5
assert abs(float(expected) - float(exact)) < 1e-4  # the central limit theorem at work
```
