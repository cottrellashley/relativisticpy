# Generating functions and partitions

Power series as a store of numbers: the coefficient of \(x^n\), written \([x^n]\), in the
generating function of the Catalan numbers, and Euler's generating function for the
partitions of an integer, with one of Ramanujan's congruences. Graham, Knuth and Patashnik
(1994) is *Concrete Mathematics*, 2nd edition; Hardy and Wright (2008) is *An Introduction
to the Theory of Numbers*, 6th edition.

## A coefficient of a generating function

**Level 4** · `nt.generating.catalan` · Graham, Knuth and Patashnik (1994), chapter 7

The notation \([x^n] F(x)\) for the coefficient of \(x^n\) in the power series of \(F\);
the brackets are not a list. From the generating function of the Catalan numbers,
\(C_{10} = 16796\).

```latex
[x^{10}] \frac{1 - \sqrt{1 - 4x}}{2x}
```

```python
x = symbols("x")
expected = series((1 - sqrt(1 - 4 * x)) / (2 * x), x, 0, 11).removeO().coeff(x, 10)
assert expected == catalan(10) == binomial(20, 10) / 11 == 16796
```

## Euler's generating function for partitions

**Level 3** · `nt.partitions.euler-product` · Hardy and Wright (2008), chapter 19

An infinite product, named, then claimed equal to its expansion to a given order: its
coefficients are the numbers of partitions, \(p(0), \ldots, p(5) = 1, 1, 2, 3, 5, 7\). Only
the factors with \(k \leq 5\) reach \(x^5\).

```latex
P(x) := \prod_{k=1}^{\infty} \frac{1}{1 - x^k}
P(x) = 1 + x + 2x^2 + 3x^3 + 5x^4 + 7x^5 + O(x^6)
```

```python
x = symbols("x")
truncated = prod(1 / (1 - x**k) for k in range(1, 6))  # each factor k >= 6 is 1 + O(x^6)
assert series(truncated, x, 0, 6) == 1 + x + 2 * x**2 + 3 * x**3 + 5 * x**4 + 7 * x**5 + O(x**6)
assert [partition(m) for m in range(6)] == [1, 1, 2, 3, 5, 7]
expected = True
```

## Ramanujan's congruence

**Level 5** · `nt.partitions.ramanujan-congruence` · Ramanujan (1919), Proc. Cambridge Philos. Soc. 19, 207

A congruence claimed for every \(k\), about the partition function defined by its
generating function: \(p(4) = 5\), \(p(9) = 30\), \(p(14) = 135\), and every \(p(5k + 4)\)
is a multiple of 5. Checking finitely many \(k\) does not decide it; it takes a proof,
Ramanujan's.

```latex
p(n) := [x^n] \prod_{j=1}^{\infty} \frac{1}{1 - x^j}
p(5k + 4) \equiv 0 \pmod{5} \quad \text{for } k = 0, 1, 2, \ldots
```

```python
top = 1004
p = [1] + [0] * top  # p(n) by Euler's recurrence over the pentagonal numbers
for n in range(1, top + 1):
    total, j = 0, 1
    while j * (3 * j - 1) // 2 <= n:
        sign = 1 if j % 2 else -1
        total += sign * p[n - j * (3 * j - 1) // 2]
        if j * (3 * j + 1) // 2 <= n:
            total += sign * p[n - j * (3 * j + 1) // 2]
        j += 1
    p[n] = total
assert all(p[n] == partition(n) for n in range(0, top + 1, 17))
assert p[4] == 5 and p[9] == 30 and p[14] == 135
assert all(p[5 * k + 4] % 5 == 0 for k in range(201))  # k = 0, 1, ..., 200
expected = True  # for every k, as Ramanujan proved
```
