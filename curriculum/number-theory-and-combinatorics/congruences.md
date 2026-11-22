# Divisibility and congruences

The greatest common divisor and Euclid's algorithm, the remainder of a division, Gauss's
notation for congruences, \(a \equiv b \pmod{m}\), inverses modulo a prime, and the
theorems of Fermat and Wilson and the Chinese remainder theorem. Hardy and Wright (2008) is
*An Introduction to the Theory of Numbers*, 6th edition; Apostol (1976) is *Introduction
to Analytic Number Theory*; Graham, Knuth and Patashnik (1994) is *Concrete Mathematics*,
2nd edition.

## Euclid's algorithm

**Level 1** · `nt.congruences.gcd` · Euclid, *Elements*, Book VII, Proposition 2

The greatest common divisor of 1071 and 462, which Euclid's algorithm of repeated
remainders finds in three divisions: 21.

```latex
\gcd(1071, 462)
```

```python
a, b = 1071, 462
while b:  # Euclid's algorithm: replace (a, b) by (b, a mod b)
    a, b = b, a % b
expected = gcd(1071, 462)
assert expected == a == 21
```

## Wilson's theorem

**Level 1** · `nt.congruences.wilson` · Hardy and Wright (2008), chapter 6

The remainder of a division, written \(\bmod\) as Graham, Knuth and Patashnik write it. By
Wilson's theorem \((p - 1)!\) leaves the remainder \(p - 1\) on division by a prime \(p\),
so \(12! \bmod 13 = 12\).

```latex
12! \bmod 13
```

```python
expected = factorial(12) % 13
assert expected == 13 - 1  # (p - 1)! = -1 (mod p) for the prime p = 13
assert int(np.prod(np.arange(1, 13, dtype=np.int64))) % 13 == 12
```

## Fermat's little theorem

**Level 2** · `nt.congruences.fermat` · Hardy and Wright (2008), chapter 6

Gauss's notation for a congruence, a claim to decide: 101 is prime, so Fermat's theorem
gives \(3^{100} \equiv 1 \pmod{101}\). Here \(\equiv\) with \(\pmod{m}\) is a congruence,
not a definition.

```latex
3^{100} \equiv 1 \pmod{101}
```

```python
assert isprime(101)  # Fermat: a^(p - 1) = 1 (mod p) for a prime p that does not divide a
expected = pow(3, 100, 101) == 1
assert expected and (Integer(3) ** 100 - 1) % 101 == 0
```

## An inverse modulo a prime

**Level 2** · `nt.congruences.inverse` · standard

\(a^{-1} \bmod m\) is the inverse of \(a\) modulo \(m\), not the remainder of the fraction
\(1/a\): Euclid's algorithm, extended, gives \(17 \cdot 38 - 43 \cdot 15 = 1\), so
\(17^{-1} \bmod 43 = 38\).

```latex
17^{-1} \bmod 43
```

```python
expected = mod_inverse(17, 43)
s, t, g = gcdex(17, 43)  # Bezout: 17 s + 43 t = gcd(17, 43) = 1
assert g == 1 and s % 43 == expected == 38
assert (17 * expected) % 43 == 1
```

## The Chinese remainder theorem

**Level 3** · `nt.congruences.chinese-remainder` · Apostol (1976), chapter 5

Sunzi's problem: a number leaves the remainders 2, 3 and 2 on division by 3, 5 and 7.
Three congruences, then a range that makes the solution unique, since by the Chinese
remainder theorem there is exactly one modulo \(3 \cdot 5 \cdot 7 = 105\): 23.

```latex
x \equiv 2 \pmod{3}
x \equiv 3 \pmod{5}
x \equiv 2 \pmod{7}
0 \leq x < 105
x
```

```python
from sympy.ntheory.modular import crt

solution, modulus = crt([3, 5, 7], [2, 3, 2])
assert modulus == 105
expected = Integer(solution)
assert expected == 23
assert [n for n in range(105) if n % 3 == 2 and n % 5 == 3 and n % 7 == 2] == [23]
```
