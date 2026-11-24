# Number theory and sequences

> **For:** anyone who computes with integers: number theory, combinatorics and signals.
> **You will learn:** the arithmetic functions, sums over divisors and primes,
> congruences, and sequences, as signal processing writes them.
> **Before this:** [The language](language.md).

```python
>>> from relativisticpy import Workbook

```

## Arithmetic functions

The arithmetic functions are written by the letters number theory gives them: Euler's
totient `\varphi(n)`, the Möbius function `\mu(n)`, the divisor functions `\sigma_k(n)` and
`d(n)`, `\omega(n)` and `\Omega(n)`, von Mangoldt's `\Lambda(n)` and the partitions `p(n)`.
Physics writes the same letters for an angle, a mean, a frequency or a momentum, so a letter
is the arithmetic function only where its argument is an integer: a whole number above 0, a
name the program assumes an integer, as `n \in \mathbb{N}` does, or the index of a sum.
Anywhere else it is what it was, and a note says which reading was taken. π is a constant,
so `\pi(10)` is 10π; `\pi(n)` counts primes only of a name assumed an integer. `H_n` and
`B_n` are the harmonic and the Bernoulli numbers only where n is such a name, so that `H_0`
stays Hubble's constant; `H_n(x)` is Hermite's polynomial. The Stirling numbers are written
in Karamata's brackets, `{n \brace k}` and `{n \brack k}`, and the Catalan, Fibonacci and Bell
numbers by their names (see the [reference](reference.md#functions)). `C(n, k)`, as Rosen
writes it, is the binomial coefficient where nothing defines C and both arguments are
counts: numbers, or names books write whole numbers with, as n and k, with a note; `C(x, t)`
stays a function, as a concentration is:

```python
>>> nt = Workbook()
>>> nt.exe(r"""
...     \varphi(36) + \mu(30) + d(28)
...     \varphi(x)
...     \pi(10)
...     n \in \mathbb{N}
...     H_{n+1} - H_n
...     \sum_{k=0}^{5} {5 \brace k}
...     C(5, 2)
... """)
[17, phi(x), 10*pi, 1/(n + 1), 52, 10]

```

## Sums over divisors and primes

A condition under a sum or a product gives its index its values: `d \mid n` the divisors of
n, `p \le x` the primes up to x, as number theorists let p range over the primes, and
`p \text{ prime}` every prime, whose product of Euler's factors is a value of ζ. The rows of
`\substack` are conditions that each value meets. A sum over an index alone, `\sum_k`, of
binomial coefficients is over every integer, as Concrete Mathematics writes it, and the
coefficients leave the terms that are not 0:

```python
>>> nt.exe(r"""
...     \sum_{d \mid 28} d
...     36 \prod_{p \mid 36} \left(1 - \frac{1}{p}\right)
...     \sum_{\substack{1 \le k \le 10 \\ \gcd(k, 10) = 1}} k
...     \prod_{p \text{ prime}} \frac{1}{1 - p^{-2}}
...     \sum_k \binom{n}{k} x^k y^{n-k}
... """)
[56, 12, 20, pi**2/6, (x + y)**n]

```

## Congruences

`a \bmod m` is the remainder, and `a^{-1} \bmod m` the inverse of a modulo m. `\equiv` with
a modulus on its right is a congruence, a claim decided by the remainders; standing alone,
of a name nothing defines, it is an assumption, as `x > 0` is, and congruences and bounds
that leave the name one value give it that value, by the Chinese remainder theorem.
`\equiv` without a modulus still defines. `solve` gives the remainders that the solutions
of congruences leave, and `[x^n] F(x)` is the coefficient of x^n in the power series of F:

```python
>>> crt = Workbook()
>>> crt.exe(r"""
...     12! \bmod 13
...     17^{-1} \bmod 43
...     3^{100} \equiv 1 \pmod{101}
...     x \equiv 2 \pmod{3}
...     x \equiv 3 \pmod{5}
...     x \equiv 2 \pmod{7}
...     0 \leq x < 105
...     x
...     solve(y^2 \equiv 1 \pmod{8}, y)
...     [z^{10}] \frac{1 - \sqrt{1 - 4z}}{2z}
... """)
[12, 38, True, 23, [1, 3, 5, 7], 16796]

```

## Sequences

Square brackets right after a name are read by what the name is. `x[n] := \ldots` defines a
sequence, a function of the integer n, as Oppenheim and Schafer write a discrete-time
signal, and `x[3]` and `x[n - 1]` are its values. Where nothing defines δ or u, `\delta[n]`
is the unit impulse, 1 at n = 0 and 0 elsewhere, and `u[n]` the unit step, 1 from n = 0 on,
each with a note. That reading needs the brackets to hold an integer as books write a
point, whole numbers and names in sums, differences and multiples, so that
`\delta[(-g)^{-1/2}]` stays a variation. Where δ has a value, as after `\delta := 0.1`,
`\delta[n]` is δ times n, and a note says so. A list or a tensor is read as before, and in
plain text, brackets after a name that is no sequence are refused, as Python's indexing.

A sum over a sequence's index is taken a term at a time: an impulse leaves its term at the
one point where it is 1, and a step cuts its term's range to where it is 1. What is left is
summed in closed form where SymPy finds one, under what the program assumes, as `|z| > 1/2`.
A sum without limits whose index is a sequence's point, as in `\sum_k x[k] h[n - k]`, runs
over every integer, with a note:

```python
>>> signals = Workbook()
>>> signals.exe(r"""
...     x[n] := \delta[n] + 2\delta[n-1] + 3\delta[n-2]
...     h[n] := \delta[n] + \delta[n-1]
...     y[n] := \sum_k x[k] h[n-k]
...     [y[0], y[1], y[2], y[3]]
...     g[n] := \left(\frac{1}{2}\right)^n u[n]
...     g[3] + g[-1]
...     \sum_{n=-\infty}^{\infty} g[n]
...     \mathcal{Z}\{g[n]\}(z)
... """)
[[1, 3, 5, 3], 1/8, 2, 2*z/(2*z - 1)]
>>> print(signals.explain(r"u[n - 1]"))
line 1: u[n - 1]
  sequence or product: u[n - 1] is step, the unit step, since u[n] is how signal processing writes it, and nothing defines u
  n: a symbol
  note: u[n - 1] is read as the unit step, step; for a sequence of your own, define u[n] := ... first

```

## Difference equations

A sequence defined through its own members, as `s[n] := \frac{1}{2} s[n-1] + u[n]`, is a
difference equation, at initial rest, as books take one: the sequence is 0 before its input
starts. Its values at whole numbers are computed one after another, and at a symbol in
closed form, where SymPy's `rsolve` finds one. `solve` solves a difference equation for a
sequence nothing defines, at rest too. An equation between sequences that nothing defines,
as `y[n] - \frac{3}{4} y[n-1] = x[n]`, is a difference equation of its own, and their capital
letters are their z-transforms, a delay of k samples \(z^{-k}\): so `Y(z)/X(z)` is the
transfer function, whose `poles` and `zeros` are the roots of its denominator and
numerator. The z-transform of a sequence written with impulses and steps is taken over every
n, and the inverse gives a sequence back, with its step:

```python
>>> systems = Workbook()
>>> systems.exe(r"""
...     s[n] := \frac{1}{2} s[n-1] + u[n]
...     s[2]
...     s[n]
...     solve(v[n] - \frac{1}{2} v[n-1] = \delta[n], v[n])
...     y[n] - \frac{3}{4} y[n-1] + \frac{1}{8} y[n-2] = x[n]
...     H(z) := \frac{Y(z)}{X(z)}
...     H(z)
...     poles(H(z))
...     h[n] := \mathcal{Z}^{-1}\{H(z)\}
...     h[n]
... """)
[7/4, (2 - 1/2**n)*step(n), [step(n)/2**n], Eq(y(n) + y(n - 2)/8 - 3*y(n - 1)/4, x(n)), 8*z**2/(8*z**2 - 6*z + 1), [1/4, 1/2], (2**(n + 1) - 1)*step(n)/4**n]

```

## The frequency response

The frequency response is written at \(e^{j\omega}\): `H(e^{j\omega}) := \sum_n h[n]
e^{-j\omega n}` defines H as a function of ω, and H at \(e^{j\theta}\) is its value at θ,
whose modulus and phase are `|H(e^{j\omega})|` and `\arg H(e^{j\omega})`. j is the imaginary
unit where a program says so, `j := \sqrt{-1}`, as engineering books do.

```python
>>> dtft = Workbook()
>>> dtft.exe(r"""
...     j := \sqrt{-1}
...     \omega \in \mathbb{R}
...     |a| < 1
...     h[n] := a^n u[n]
...     H(e^{j\omega}) := \sum_n h[n] e^{-j\omega n}
...     H(e^{j\omega})
...     H(e^{j\pi})
... """)
[1/(-a*exp(-I*omega) + 1), 1/(a + 1)]

```

## Convolution

An asterisk, `*` or `\ast`, between two sequences at points of one index is their
convolution, \(\sum_k x[k]\,h[n - k]\) over every integer k, and between two functions
applied at one variable it is \(\int f(\tau)\,g(t - \tau)\,d\tau\) over every τ, with u and δ
in them the unit step and the Dirac delta where nothing defines them; each is said in a
note. Between anything else, as `2*x`, it multiplies, and written side by side or with
`\cdot`, the two multiply. `\circledast` is the circular convolution, of the length its
subscript gives, `\circledast_N`, or else of the longer sequence (`design/notation.md`, 27):

```python
>>> convolved = Workbook()
>>> convolved.exe(r"""
...     x[n] := \delta[n] + 2\delta[n-1] + 3\delta[n-2] + 4\delta[n-3]
...     h[n] := \delta[n] + \delta[n-1]
...     y[n] := x[n] * h[n]
...     [y[0], y[1], y[2], y[3], y[4]]
...     (x \circledast h)[0]
...     t > 0
...     \left(e^{-t} u(t)\right) * \left(e^{-2t} u(t)\right)
... """)
[[1, 3, 5, 7, 4], 5, (exp(t) - 1)*exp(-2*t)]

```

**Next:** [Quantum notation](quantum.md), or the
[reference](reference.md#functions), for every arithmetic function by name.
