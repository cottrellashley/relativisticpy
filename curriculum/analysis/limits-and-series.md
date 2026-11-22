# Limits and series

Limits, sums and Taylor series, written as an analysis course writes them.

## A standard limit

**Level 1** · `analysis.limits.sinc` · standard

```latex
\lim_{x \to 0} \frac{\sin x}{x}
```

```python
expected = 1
```

## The Basel problem

**Level 2** · `analysis.series.basel` · Euler (1734)

The sum of the reciprocal squares, carried out.

```latex
\sum_{n=1}^{\infty} \frac{1}{n^2}
```

```python
expected = pi**2 / 6
```

## A Taylor polynomial

**Level 2** · `analysis.series.taylor-exp` · standard

The Taylor series of the exponential to fourth order, as an analysis book writes it.

```latex
\sum_{k=0}^{4} \frac{x^k}{k!}
```

```python
x = symbols("x")
expected = 1 + x + x**2 / 2 + x**3 / 6 + x**4 / 24
```

## The radius of convergence

**Level 3** · `analysis.series.radius-ratio` · standard

The ratio test for \(\sum n^2 x^n\), as a derivation: the ratio of successive terms,
its limit, and the radius it gives.

```latex
a_n := n^2
\rho := \lim_{n \to \infty} \left| \frac{a_{n+1}}{a_n} \right|
\frac{1}{\rho}
```

```python
expected = 1
```

## A one-sided limit

**Level 1** · `analysis.limits.one-sided` · standard

The limit from the left, written \(x \to 0^-\). From the right the same expression tends
to 0, so the direction decides the answer.

```latex
\lim_{x \to 0^-} \frac{1}{1 + e^{1/x}}
```

```python
x = symbols("x")
assert limit(1 / (1 + exp(1 / x)), x, 0, "+") == 0
expected = limit(1 / (1 + exp(1 / x)), x, 0, "-")
assert expected == 1
```

## A limit that does not exist

**Level 2** · `analysis.limits.does-not-exist` · Stewart (2015), §2.3

The one-sided limits of \(|x|/x\) at 0 are \(-1\) and \(1\), so the limit does not exist.
The program must be refused, saying so, rather than give either value.

```latex
\lim_{x \to 0} \frac{|x|}{x}
```

```python
x = symbols("x")
assert limit(Abs(x) / x, x, 0, "-") == -1
assert limit(Abs(x) / x, x, 0, "+") == 1
error = "EvaluationError"
message = "does not exist"
```

## L'Hôpital's rule

**Level 2** · `analysis.limits.lhopital` · Stewart (2015), §4.4

The rule applied twice to an indeterminate form \(0/0\): the limit of a quotient equals the
limit of the quotient of second derivatives, written with primes on functions the program
defines.

```latex
f(x) := e^x - 1 - x
g(x) := x^2
\lim_{x \to 0} \frac{f(x)}{g(x)} = \lim_{x \to 0} \frac{f''(x)}{g''(x)}
```

```python
x = symbols("x")
f, g = exp(x) - 1 - x, x**2
assert limit(f / g, x, 0) == Rational(1, 2)
assert limit(diff(f, x, 2) / diff(g, x, 2), x, 0) == Rational(1, 2)
expected = True
```

## The harmonic series diverges

**Level 2** · `analysis.series.harmonic` · Stewart (2015), §11.2

A series whose sum is infinite, stated as an equation the language must decide.

```latex
\sum_{n=1}^{\infty} \frac{1}{n} = \infty
```

```python
n = symbols("n", integer=True, positive=True)
assert Sum(1 / n, (n, 1, oo)).doit() == oo
expected = True
```

## The radius of convergence by an upper limit

**Level 4** · `analysis.series.limsup` · Rudin (1976), Theorem 3.39

The Cauchy–Hadamard formula for \(\sum c_n z^n\), with \(\limsup\): here \(\sqrt[n]{|c_n|}\)
alternates between 3 and 1, so it has no limit, but its upper limit is 3 and the radius of
convergence is \(1/3\).

```latex
c_n := \left(2 + (-1)^n\right)^n
\alpha := \limsup_{n \to \infty} \sqrt[n]{|c_n|}
R := \frac{1}{\alpha}
R
```

```python
k = symbols("k", integer=True, positive=True)
c = lambda m: (2 + (-1) ** m) ** m
even = limit(c(2 * k) ** (1 / (2 * k)), k, oo)  # along the even n
odd = limit(c(2 * k + 1) ** (1 / (2 * k + 1)), k, oo)  # along the odd n
assert (even, odd) == (3, 1)
expected = 1 / Max(even, odd)
```
