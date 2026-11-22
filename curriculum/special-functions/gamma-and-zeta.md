# The Gamma, Beta and zeta functions

Euler's Gamma and Beta functions and Riemann's zeta function, in the notation of the NIST
*Digital Library of Mathematical Functions* (DLMF), chapters 5 and 25. A Greek letter
applied to an argument is the function that every book means by it.

## The Gamma function at one half

**Level 1** · `special.gamma.half` · DLMF 5.4.6

\(\Gamma\) applied to a number is Euler's Gamma function.

```latex
\Gamma\left(\tfrac{1}{2}\right)
```

```python
expected = gamma(Rational(1, 2))
assert expected == sqrt(pi)
```

## The Beta function

**Level 1** · `special.gamma.beta` · DLMF 5.12.1

\(\mathrm{B}(a, b) = \int_0^1 t^{a-1} (1 - t)^{b-1}\,dt = \Gamma(a)\Gamma(b)/\Gamma(a + b)\),
at \(a = b = 1/2\).

```latex
\mathrm{B}\left(\tfrac{1}{2}, \tfrac{1}{2}\right)
```

```python
t = symbols("t")
expected = integrate(t ** Rational(-1, 2) * (1 - t) ** Rational(-1, 2), (t, 0, 1))
assert expected == gamma(Rational(1, 2)) ** 2 / gamma(1) == pi
```

## The reflection formula

**Level 2** · `special.gamma.reflection` · DLMF 5.5.3

An identity between the Gamma function at \(z\) and at \(1 - z\), to be decided for every
\(z\).

```latex
\Gamma(z) \Gamma(1 - z) = \frac{\pi}{\sin \pi z}
```

```python
z = symbols("z")
assert simplify(gammasimp(gamma(z) * gamma(1 - z)) - pi / sin(pi * z)) == 0
expected = True
```

## The zeta function at 2

**Level 1** · `special.zeta.two` · DLMF 25.6.1

\(\zeta\) applied to a number is Riemann's zeta function.

```latex
\zeta(2)
```

```python
n = symbols("n", integer=True, positive=True)
expected = zeta(2)
assert expected == pi**2 / 6 == Sum(1 / n**2, (n, 1, oo)).doit()
```

## The functional equation

**Level 3** · `special.zeta.functional-equation` · DLMF §25.4

Riemann's functional equation, which continues \(\zeta\) from \(\operatorname{Re} s > 1\) to
the whole plane: an identity between transcendental functions that the language must decide
for every \(s\).

```latex
\zeta(s) = 2^s \pi^{s - 1} \sin\left(\frac{\pi s}{2}\right) \Gamma(1 - s) \zeta(1 - s)
```

```python
s = symbols("s")
right = 2**s * pi ** (s - 1) * sin(pi * s / 2) * gamma(1 - s) * zeta(1 - s)
for value in (Rational(1, 3), Rational(5, 2), Rational(1, 2) + 3 * I):
    assert abs(N(zeta(value) - right.subs(s, value), 30)) < 1e-25
expected = True
```

## Euler's product

**Level 5** · `special.zeta.euler-product` · DLMF 25.2.11

A product over the primes, \(\prod_p\), here Euler's product for \(\zeta(2)\): an index set
defined by a property, and a theorem to decide the equation.

```latex
\prod_{p \text{ prime}} \frac{1}{1 - p^{-2}} = \frac{\pi^2}{6}
```

```python
primes = np.array(list(sieve.primerange(2, 10**5)), dtype=float)
partial = np.prod(1 / (1 - primes**-2))  # the primes below 100 000
assert abs(partial - float(pi**2 / 6)) < 1e-5
expected = True
```
