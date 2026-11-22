# Special numbers

The Fibonacci, Catalan, Stirling and Bell numbers: Binet's formula with the golden ratio,
a recurrence by convolution, Dobiński's series, and Karamata's notation for the Stirling
numbers, \({n \brace k}\) and \({n \brack k}\), which Knuth made standard. Graham, Knuth
and Patashnik (1994) is *Concrete Mathematics*, 2nd edition.

## Binet's formula

**Level 2** · `nt.special.binet` · Graham, Knuth and Patashnik (1994), section 6.6

The golden ratio \(\varphi\) and its conjugate \(\hat\varphi\), defined, then the family
\(F_n\) evaluated at \(n = 10\): 55. The user's \(\varphi\) is a number, and wins over
Euler's totient \(\varphi(n)\).

```latex
\varphi := \frac{1 + \sqrt{5}}{2}
\hat\varphi := \frac{1 - \sqrt{5}}{2}
F_n := \frac{\varphi^n - \hat\varphi^n}{\sqrt{5}}
F_{10}
```

```python
phi, phi_hat = (1 + sqrt(5)) / 2, (1 - sqrt(5)) / 2
expected = simplify((phi**10 - phi_hat**10) / sqrt(5))
F = [0, 1]
for _ in range(9):  # F_n = F_(n - 1) + F_(n - 2)
    F.append(F[-1] + F[-2])
assert expected == F[10] == fibonacci(10) == 55
```

## Dobiński's formula

**Level 2** · `nt.special.dobinski` · Dobiński (1877)

An infinite series with a closed form: the Bell number \(B_5 = 52\), the number of ways to
split five objects into blocks, is \(\frac{1}{e} \sum_k k^5 / k!\).

```latex
\frac{1}{e} \sum_{k=0}^{\infty} \frac{k^5}{k!}
```

```python
k = symbols("k", integer=True, nonnegative=True)
expected = Sum(k**5 / factorial(k), (k, 0, oo)).doit() / E
assert expected == bell(5) == 52
partial = sum(Rational(j**5, factorial(j)) for j in range(40))
assert abs(N(partial / E, 30) - 52) < 1e-25
```

## The Catalan numbers from their recurrence

**Level 3** · `nt.special.catalan-recurrence` · Graham, Knuth and Patashnik (1994), chapter 7

A sequence defined by a convolution, with an expression in the subscript of its left
side, then evaluated: \(C_{10} = 16796\), the number of ways to bracket a product of
eleven factors.

```latex
C_0 = 1
C_{n+1} = \sum_{k=0}^{n} C_k C_{n-k}, \quad n \geq 0
C_{10}
```

```python
C = [1]
for n in range(10):  # C_(n + 1) is the sum of C_k C_(n - k)
    C.append(sum(C[k] * C[n - k] for k in range(n + 1)))
expected = C[10]
assert expected == catalan(10) == binomial(20, 10) / 11 == 16796
```

## Stirling numbers of the second kind

**Level 4** · `nt.special.stirling-brace` · Graham, Knuth and Patashnik (1994), section 6.1

Karamata's braces, as Knuth writes them, \({n \brace k}\): the number of ways to split
\(n\) objects into \(k\) nonempty blocks. Summed over \(k\), they give the Bell number \(B_5 = 52\).

```latex
\sum_{k=0}^{5} {5 \brace k}
```

```python
from sympy.functions.combinatorial.numbers import stirling
from sympy.utilities.iterables import multiset_partitions

expected = sum(stirling(5, k) for k in range(6))
assert [stirling(5, k) for k in range(6)] == [0, 1, 15, 25, 10, 1]
assert expected == bell(5) == len(list(multiset_partitions(list(range(5))))) == 52
```

## Stirling numbers of the first kind

**Level 4** · `nt.special.stirling-brack` · Graham, Knuth and Patashnik (1994), section 6.1

Karamata's brackets, as Knuth writes them, \({n \brack k}\): the number of permutations
of \(n\) objects with \(k\) cycles. Summed over \(k\), they count every permutation: \(5! = 120\).

```latex
\sum_{k=0}^{5} {5 \brack k}
```

```python
from itertools import permutations

from sympy.combinatorics import Permutation
from sympy.functions.combinatorial.numbers import stirling

expected = sum(stirling(5, k, kind=1) for k in range(6))
cycles = [Permutation(list(p)).cycles for p in permutations(range(5))]
assert [cycles.count(k) for k in range(6)] == [stirling(5, k, kind=1) for k in range(6)]
assert expected == factorial(5) == 120
```
