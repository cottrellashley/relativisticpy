# The error function

The error function, \(\operatorname{erf} z = \frac{2}{\sqrt\pi} \int_0^z e^{-t^2} dt\), and
its complement \(\operatorname{erfc} = 1 - \operatorname{erf}\): their derivative, their
series, their behaviour for large arguments, and the normal distribution. DLMF is the NIST
*Digital Library of Mathematical Functions*, chapter 7.

## The derivative of the error function

**Level 1** · `special.erf.derivative` · DLMF 7.2.1

\(\operatorname{erf}\) is a named function, written as an operator name, whose derivative is
the Gaussian.

```latex
\frac{d}{dx} \operatorname{erf} x
```

```python
x = symbols("x")
expected = diff(erf(x), x)
assert expected == 2 * exp(-(x**2)) / sqrt(pi)
```

## One standard deviation of a normal distribution

**Level 2** · `special.erf.normal` · DLMF 7.2.1

The probability that a normal variable lies within one standard deviation of its mean, the
68 per cent of the 68–95–99.7 rule, is \(\operatorname{erf}(1/\sqrt{2})\).

```latex
\frac{1}{\sqrt{2\pi}} \int_{-1}^{1} e^{-x^2/2} \, dx
```

```python
x = symbols("x")
expected = integrate(exp(-(x**2) / 2), (x, -1, 1)) / sqrt(2 * pi)
assert expected == erf(1 / sqrt(2))
assert abs(float(expected) - 0.6826894921) < 1e-10
```

## The complementary error function for large arguments

**Level 2** · `special.erf.asymptotic` · DLMF 7.12.1

The leading term of the asymptotic expansion,
\(\operatorname{erfc} x \sim e^{-x^2} / (x\sqrt\pi)\), as a limit. \(\operatorname{erfc}\)
is written without brackets, as DLMF writes it.

```latex
\lim_{x \to \infty} x e^{x^2} \operatorname{erfc} x
```

```python
x = symbols("x")
expected = limit(x * exp(x**2) * erfc(x), x, oo)
assert expected == 1 / sqrt(pi)
```

## The Maclaurin series of the error function

**Level 2** · `special.erf.series` · DLMF 7.6.1

The error function as its power series, an infinite sum that the language must recognise.

```latex
\operatorname{erf} x = \frac{2}{\sqrt{\pi}} \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n+1}}{n! \, (2n+1)}
```

```python
x = symbols("x")
n = symbols("n", integer=True, nonnegative=True)
total = Sum((-1) ** n * x ** (2 * n + 1) / (factorial(n) * (2 * n + 1)), (n, 0, oo)).doit()
assert simplify(2 / sqrt(pi) * total - erf(x)) == 0
expected = True
```
