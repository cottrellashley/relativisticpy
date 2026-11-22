# Binomial coefficients

Identities of the binomial coefficients, the binomial theorem, the sum of a column of
Pascal's triangle and Vandermonde's convolution, as Graham, Knuth and Patashnik (1994),
*Concrete Mathematics*, 2nd edition, write them: with the range of a sum given by an
inequality, or left to the coefficients themselves.

## The binomial theorem

**Level 2** · `nt.binomial.theorem` · Graham, Knuth and Patashnik (1994), chapter 5

A sum with a symbolic upper limit and a closed form, for every natural number \(n\):
\((x + y)^n\), for all \(x\) and \(y\), with no condition on their size.

```latex
n \in \mathbb{N}
\sum_{k=0}^{n} \binom{n}{k} x^k y^{n-k}
```

```python
x, y = symbols("x y")
n = symbols("n", integer=True, nonnegative=True)
expected = (x + y) ** n
for m in range(9):  # an identity of polynomials, checked for n = 0, ..., 8
    terms = sum(binomial(m, k) * x**k * y ** (m - k) for k in range(m + 1))
    assert expand(terms - (x + y) ** m) == 0
```

## Summing a column of Pascal's triangle

**Level 3** · `nt.binomial.upper-summation` · Graham, Knuth and Patashnik (1994), chapter 5

The range of summation written as an inequality under the \(\sum\), as Concrete
Mathematics writes it, in an identity in two natural numbers \(m\) and \(n\), true whether
or not \(\mathbb{N}\) includes 0. It holds for \(m > n\) too, where both sides are 0.

```latex
m, n \in \mathbb{N}
\sum_{0 \leq k \leq n} \binom{k}{m} = \binom{n+1}{m+1}
```

```python
n = symbols("n", integer=True, nonnegative=True)
k = symbols("k", integer=True)
for m in range(6):  # for each m = 0, ..., 5, an identity of polynomials in n
    found = summation(binomial(k, m), (k, 0, n))
    assert simplify(expand_func(found - binomial(n + 1, m + 1))) == 0
assert all(
    sum(binomial(j, a) for j in range(b + 1)) == binomial(b + 1, a + 1)
    for a in range(13)
    for b in range(13)
)
expected = True
```

## Vandermonde's convolution

**Level 4** · `nt.binomial.vandermonde` · Graham, Knuth and Patashnik (1994), chapter 5

A sum over every integer \(k\), written \(\sum_k\), its range left to the binomial
coefficients, which vanish when their lower index is negative: Vandermonde's convolution,
for any \(r\) and \(s\) and every integer \(n\).

```latex
n \in \mathbb{Z}
\sum_k \binom{r}{k} \binom{s}{n-k} = \binom{r+s}{n}
```

```python
from scipy.special import binom

r, s = symbols("r s")
for n in range(7):  # only the terms with 0 <= k <= n can be nonzero
    left = sum(binomial(r, k) * binomial(s, n - k) for k in range(n + 1))
    assert expand(expand_func(left - binomial(r + s, n))) == 0
assert binomial(r + s, -1) == 0  # for n < 0 no term survives, and the right side is 0
rng = np.random.default_rng(0)
for r0, s0 in rng.uniform(-3, 3, size=(5, 2)):  # real upper indices, numerically
    for n in range(6):
        left = sum(binom(r0, k) * binom(s0, n - k) for k in range(n + 1))
        assert abs(left - binom(r0 + s0, n)) < 1e-9 * max(1.0, abs(binom(r0 + s0, n)))
expected = True
```
