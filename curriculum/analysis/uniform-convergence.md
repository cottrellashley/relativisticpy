# Uniform convergence

When the limit of a sequence of functions can be taken inside an integral and when it
cannot, and statements of uniform convergence and uniform continuity, which an analysis
course proves rather than computes. Rudin (1976) is *Principles of Mathematical Analysis*,
3rd edition.

## A limit that cannot be taken inside an integral

**Level 3** · `analysis.uniform.interchange` · Rudin (1976), Example 7.6

\(f_n(x) = n x (1 - x^2)^n\) tends to 0 at every point of \([0, 1]\), but its integrals tend
to \(1/2\): the convergence is not uniform, and the limit and the integral do not commute.
Both sides can be computed.

```latex
f_n(x) := n x (1 - x^2)^n
\lim_{n \to \infty} \int_0^1 f_n(x) \, dx = \int_0^1 \lim_{n \to \infty} f_n(x) \, dx
```

```python
n = symbols("n", integer=True, positive=True)
x = symbols("x", positive=True)
integrals = integrate(n * x * (1 - x**2) ** n, (x, 0, 1))
assert simplify(integrals - n / (2 * n + 2)) == 0
assert limit(integrals, n, oo) == Rational(1, 2)  # the left side
# f_n(x) -> 0 at x = 0 and x = 1, and like n q^n with 0 < q < 1 in between
assert limit(n * Rational(1, 2) * Rational(3, 4) ** n, n, oo) == 0  # x = 1/2, for one
expected = False  # 1/2 on the left, 0 on the right
```

## Pointwise but not uniform convergence

**Level 5** · `analysis.uniform.not-uniform` · Rudin (1976), Theorem 7.9

\(x^n \to 0\) at every point of \([0, 1)\), but not uniformly, since
\(\sup_{0 \le x < 1} x^n = 1\) for every \(n\). The program states uniform convergence in
words, as Rudin does.

```latex
f_n(x) := x^n
f_n \to 0 \text{ uniformly on } [0, 1)
```

```python
n, x = symbols("n x", positive=True)
# the supremum of |x^n - 0| over [0, 1) is 1 for every n, so it does not tend to 0
assert limit(x**n, x, 1, "-") == 1
expected = False
```

## The Weierstrass M-test

**Level 5** · `analysis.uniform.m-test` · Rudin (1976), Theorem 7.10

\(|\sin(nx) / n^2| \le 1/n^2\), and \(\sum 1/n^2\) converges, so the series converges
uniformly on the whole line. The program states the conclusion in words.

```latex
\sum_{n=1}^{\infty} \frac{\sin nx}{n^2} \text{ converges uniformly on } \mathbb{R}
```

```python
n = symbols("n", integer=True, positive=True)
assert Sum(1 / n**2, (n, 1, oo)).is_convergent()  # the bounds M_n = 1/n^2
expected = True
```

## A function that is not uniformly continuous

**Level 5** · `analysis.uniform.uniform-continuity` · Rudin (1976), Definition 4.18

\(1/x\) is continuous on \((0, 1)\) but not uniformly: the points \(t\) and \(t/2\) are
within \(t/2\) of each other, but their images are \(1/t\) apart, which no \(\delta\) can
bound. A statement with nested quantifiers that must be refuted.

```latex
\forall \varepsilon > 0 \ \exists \delta > 0 \ \forall x, y \in (0, 1) : |x - y| < \delta \implies \left| \frac{1}{x} - \frac{1}{y} \right| < \varepsilon
```

```python
t = symbols("t", positive=True)
gap = Abs(1 / t - 1 / (t / 2))  # between x = t and y = t/2
assert simplify(gap - 1 / t) == 0 and limit(gap, t, 0, "+") == oo
expected = False
```
