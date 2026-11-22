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
