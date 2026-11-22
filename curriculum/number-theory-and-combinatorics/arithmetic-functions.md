# Arithmetical functions

Functions that number theory names by a letter, Euler's totient \(\varphi(n)\) and the
Möbius function \(\mu(n)\) of a positive integer and the prime-counting function
\(\pi(x)\) of a real number, with sums over the divisors of a number, \(\sum_{d \mid n}\),
and products over its primes, \(\prod_{p \mid n}\). Apostol (1976) is *Introduction to
Analytic Number Theory*.

## Euler's totient

**Level 1** · `nt.arithmetic.totient` · Apostol (1976), chapter 2

\(\varphi(n)\) counts the integers from 1 to \(n\) that are prime to \(n\). Applied to an
integer, \(\varphi\) is Euler's function, as every book on number theory means it.

```latex
\varphi(36)
```

```python
expected = totient(36)
assert expected == sum(1 for k in range(1, 37) if gcd(k, 36) == 1) == 12
```

## A perfect number

**Level 1** · `nt.arithmetic.perfect-number` · Euclid, *Elements*, Book IX, Proposition 36

A sum over the divisors of a number, \(\sum_{d \mid n}\): the divisors of 28 add up to 56,
twice 28, so 28 is perfect.

```latex
\sum_{d \mid 28} d
```

```python
expected = sum(d for d in range(1, 29) if 28 % d == 0)
assert expected == divisor_sigma(28) == 2 * 28 == 56
```

## Möbius inversion

**Level 2** · `nt.arithmetic.mobius-inversion` · Apostol (1976), chapter 2

The Möbius function by its name, \(\mu\), in a sum over divisors. Inverting Gauss's
\(\sum_{d \mid n} \varphi(d) = n\) gives \(\varphi(n) = \sum_{d \mid n} \mu(d)\, n/d\),
which for \(n = 36\) is \(\varphi(36) = 12\).

```latex
\sum_{d \mid 36} \mu(d) \frac{36}{d}
```

```python
expected = sum(mobius(d) * Rational(36, d) for d in divisors(36))
assert expected == totient(36) == 12
assert sum(totient(d) for d in divisors(36)) == 36  # Gauss's identity, which this inverts
```

## Euler's product for the totient

**Level 4** · `nt.arithmetic.totient-product` · Apostol (1976), chapter 2

A product over the primes that divide \(n\): number theorists let the letter \(p\) range
over primes, so \(\prod_{p \mid 36}\) has the two factors \(p = 2\) and \(p = 3\), and
\(36 (1 - \frac{1}{2})(1 - \frac{1}{3}) = \varphi(36) = 12\).

```latex
36 \prod_{p \mid 36} \left(1 - \frac{1}{p}\right)
```

```python
assert primefactors(36) == [2, 3]
expected = 36 * prod(1 - Rational(1, p) for p in primefactors(36))
assert expected == totient(36) == 12
# over every divisor of 36 instead, the factor for d = 1 would make the product 0
assert 36 * prod(1 - Rational(1, d) for d in divisors(36)) == 0
```

## The prime number theorem

**Level 5** · `nt.arithmetic.prime-number-theorem` · Hadamard (1896); de la Vallée Poussin (1896)

The number of primes up to \(x\), defined in set-builder notation, is asymptotic to
\(x / \ln x\). No limit that can be computed decides it: it takes the theorem.

```latex
\pi(x) := \#\{p \leq x : p \text{ prime}\}
\pi(x) \sim \frac{x}{\ln x}
```

```python
ratios = [primepi(10**e) * log(10**e) / 10**e for e in range(3, 10)]
assert all(ratios[i] > ratios[i + 1] > 1 for i in range(len(ratios) - 1))  # 1.16, ..., 1.054
is_prime = np.ones(10**6 + 1, dtype=bool)  # the sieve of Eratosthenes, for pi(10^6)
is_prime[:2] = False
for i in range(2, 1001):
    if is_prime[i]:
        is_prime[i * i :: i] = False
assert is_prime.sum() == primepi(10**6) == 78498
x = symbols("x", positive=True)
assert limit(li(x) / (x / log(x)), x, oo) == 1  # pi(x) ~ li(x), the form the theorem takes
expected = True
```
