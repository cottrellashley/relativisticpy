# Inclusion and exclusion

Counting by including and excluding: the maps onto a set, and the permutations that fix
no point, with the limit of their share of all permutations. Stanley (2011) is
*Enumerative Combinatorics*, volume 1, 2nd edition.

## Maps onto a set

**Level 2** · `nt.counting.surjections` · Stanley (2011), chapter 2

The number of ways to put six distinct objects in three distinct boxes, leaving none
empty, by inclusion and exclusion over the boxes left empty: \(540 = 3!\,{6 \brace 3}\).

```latex
\sum_{j=0}^{3} (-1)^j \binom{3}{j} (3 - j)^6
```

```python
from itertools import product as words

from sympy.functions.combinatorial.numbers import stirling

expected = sum((-1) ** j * binomial(3, j) * (3 - j) ** 6 for j in range(4))
onto = sum(1 for f in words(range(3), repeat=6) if set(f) == {0, 1, 2})  # every map, counted
assert expected == onto == factorial(3) * stirling(6, 3) == 540
```

## Derangements

**Level 3** · `nt.counting.derangements` · Stanley (2011), chapter 2

The number of permutations of \(n\) objects that fix none, by inclusion and exclusion,
named \(D_n\); then the limit of their share of the \(n!\) permutations, which the
partial sums of the series for \(e^{-1}\) approach: \(1/e\).

```latex
D_n := n! \sum_{k=0}^{n} \frac{(-1)^k}{k!}
\lim_{n \to \infty} \frac{D_n}{n!}
```

```python
from itertools import permutations

k = symbols("k", integer=True, nonnegative=True)
expected = Sum((-1) ** k / factorial(k), (k, 0, oo)).doit()
assert expected == exp(-1)
for n in range(1, 8):  # D_n counts the permutations with no fixed point
    count = sum(1 for p in permutations(range(n)) if all(p[i] != i for i in range(n)))
    formula = factorial(n) * sum(Rational((-1) ** j, factorial(j)) for j in range(n + 1))
    assert count == formula == subfactorial(n)
assert abs(float(subfactorial(20) / factorial(20)) - np.exp(-1)) < 1e-15
```
