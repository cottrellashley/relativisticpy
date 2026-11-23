# Sums and series

The sum of the terms of a sequence, finite, \(\sum_{k=1}^{n} a_k\), or infinite, a series
\(\sum_{n=1}^{\infty} a_n\). Stewart (2015) is *Calculus: Early Transcendentals*, 8th
edition; Graham, Knuth and Patashnik (1994) is *Concrete Mathematics*, 2nd edition; Knuth
(1984) is *The TeXbook*; AsciiMath is the plain text notation of asciimath.org; UnicodeMath
is the linear format of Unicode Technical Note #28; the Unicode Standard is version 15.0.

`«f»` is the term, `«var»` the index, and `«lower»` and `«upper»` its limits; `«lowered»`
and `«raised»` are the limits in Unicode's subscript and superscript characters.

A series that converges only for some values of a parameter, as the geometric series
\(\sum r^n = \frac{1}{1 - r}\) for \(|r| < 1\), is given as books give it, and a note says
where it holds (`design/notation.md`, decisions 1 and 4). A series of positive terms that
does not converge has the sum ∞; one whose partial sums keep oscillating has no sum, and the
program must say so rather than give one.

## Names

```python
k, n, x, r = symbols("k n x r")
big_n = symbols("N")
```

## Operands

| operand | written in | `f` | `var` | `lower` | `upper` | `lowered` | `raised` | separates |
|---|---|---|---|---|---|---|---|---|
| `squares` | latex, text | `k^2` | `k` | `1` | `10` | | | a finite sum of numbers, to a limit of two digits |
| | unicode | `k²` | `k` | `1` | `10` | `ₖ₌₁` | `¹⁰` | |
| | python | `k**2` | `k` | `1` | `10` | | | |
| `first-n` | latex, text, unicode | `k` | `k` | `1` | `N` | | | a limit that is a name: a sum in closed form |
| | python | `k` | `k` | `1` | `big_n` | | | |
| `geometric` | latex | `\frac{1}{2^n}` | `n` | `0` | `\infty` | | | a series from 0 |
| | text | `1/2^n` | `n` | `0` | `oo` | | | |
| | unicode | `1/2ⁿ` | `n` | `0` | `∞` | | | |
| | python | `1/2**n` | `n` | `0` | `oo` | | | |
| `basel` | latex | `\frac{1}{n^2}` | `n` | `1` | `\infty` | | | a series whose sum is not rational, π²/6 |
| | text | `1/n^2` | `n` | `1` | `oo` | | | |
| | unicode | `1/n²` | `n` | `1` | `∞` | | | |
| | python | `1/n**2` | `n` | `1` | `oo` | | | |
| `alternating-harmonic` | latex | `\frac{(-1)^{n+1}}{n}` | `n` | `1` | `\infty` | | | a sign that alternates, a series that converges but not absolutely, to ln 2 |
| | text | `(-1)^(n+1)/n` | `n` | `1` | `oo` | | | |
| | unicode | `(−1)ⁿ⁺¹/n` | `n` | `1` | `∞` | | | |
| | python | `(-1)**(n + 1)/n` | `n` | `1` | `oo` | | | |
| `exponential` | latex | `\frac{x^n}{n!}` | `n` | `0` | `\infty` | | | a parameter beside the index, and a factorial: a sum over n alone |
| | text | `x^n/n!` | `n` | `0` | `oo` | | | |
| | unicode | `xⁿ/n!` | `n` | `0` | `∞` | | | |
| | python | `x**n/factorial(n)` | `n` | `0` | `oo` | | | |
| `geometric-ratio` | latex | `r^n` | `n` | `0` | `\infty` | | | a series that converges only for some values of its parameter |
| | text | `r^n` | `n` | `0` | `oo` | | | |
| | unicode | `rⁿ` | `n` | `0` | `∞` | | | |
| | python | `r**n` | `n` | `0` | `oo` | | | |
| `harmonic` | latex | `\frac{1}{n}` | `n` | `1` | `\infty` | | | a series of positive terms that diverges: its sum is ∞ |
| | text | `1/n` | `n` | `1` | `oo` | | | |
| | unicode | `1/n` | `n` | `1` | `∞` | | | |
| | python | `1/n` | `n` | `1` | `oo` | | | |
| `grandi` | latex | `(-1)^n` | `n` | `0` | `\infty` | | | a series whose partial sums keep oscillating: no sum |
| | text | `(-1)^n` | `n` | `0` | `oo` | | | |
| | unicode | `(−1)ⁿ` | `n` | `0` | `∞` | | | |
| | python | `(-1)**n` | `n` | `0` | `oo` | | | |

## Oracle

```python
found = summation(f, (var, lower, upper))
if isinstance(found, Piecewise):  # where it converges, as books give it
    found = found.args[0][0]
if found.has(Sum):
    found = None
    error = "RelativisticPyError"
else:
    expected = found
```

## Second route

Term by term: a finite sum added up, at a value of its limit where that is a name; a series
summed numerically, by SymPy's numerical summation rather than its closed form; and a series
with no value shown not to converge, its partial sums growing or oscillating.

```python
at = {x: Rational(7, 10), r: Rational(1, 3), big_n: 7}
term, top = f.subs(at), S(upper).subs(at)
if top.is_finite:
    added = sum(term.subs(var, j) for j in range(int(lower), int(top) + 1))
    assert found.subs(at) == added
elif found is None:
    assert Sum(f, (var, lower, upper)).is_convergent() is S.false
    partial = [sum(term.subs(var, j) for j in range(int(lower), end)) for end in (100, 101)]
    assert abs(partial[0] - partial[1]) >= 1
elif found.is_infinite:
    assert Sum(f, (var, lower, upper)).is_convergent() is S.false
    assert all(term.subs(var, j) > 0 for j in range(int(lower), int(lower) + 50))
else:
    numerically = N(Sum(term, (var, lower, upper)), 30)
    assert abs(numerically - N(found.subs(at), 30)) < 10**-15
```

## Forms

### The sum

**Form** · `sum` · Stewart (2015), §11.2, Series: `\sum_{n=1}^{\infty} a_n`

```latex
\sum_{«var»=«lower»}^{«upper»} «f»
```

### An upper limit without braces

**Form** · `one-token-limits` · Knuth (1984), chapter 16: a superscript without braces is one character or command, as in `\sum_{n=1}^\infty`

```latex
\sum_{«var»=«lower»}^«upper» «f»
```

**Not of** · `squares` · an upper limit of two digits is written in braces

### The limits below and above in a line of text

**Form** · `limits` · Knuth (1984), chapter 17: `\limits` sets an operator's limits below and above it

```latex
\sum\limits_{«var»=«lower»}^{«upper»} «f»
```

### In display style

**Form** · `displaystyle` · Knuth (1984), chapter 17: `\displaystyle`, as a line of text sets a sum as a display does

```latex
\displaystyle\sum_{«var»=«lower»}^{«upper»} «f»
```

### The index between its limits

**Form** · `between` · Graham, Knuth and Patashnik (1994), §2.1, Notation: `\sum_{1 \le k \le n} a_k`

```latex
\sum_{«lower» \le «var» \le «upper»} «f»
```

**Only of** · `squares`, `first-n` · a sum with a last term; a series is written with its first alone

### A condition on the index

**Form** · `at-least` · Graham, Knuth and Patashnik (1994), §2.1, Notation: the general form `\sum_{P(k)} a_k`, a sum over the integers k for which P(k) holds

```latex
\sum_{«var» \ge «lower»} «f»
```

**Not of** · `squares`, `first-n` · a condition with no upper bound is a series

### Of terms the program defines

**Form** · `terms` · Stewart (2015), §11.2: `\sum_{n=1}^{\infty} a_n`, of terms defined first

```latex
a_{«var»} := «f»
\sum_{«var»=«lower»}^{«upper»} a_{«var»}
```

### A call of summation

**Form** · `summation` · SymPy's `summation(1/n**2, (n, 1, oo))`

```text
summation(«f», («var», «lower», «upper»))
```

### The sum in plain text

**Form** · `asciimath` · AsciiMath: `sum_(i=1)^n i^3`

```text
sum_(«var»=«lower»)^«upper» «f»
```

### The sign in Unicode

**Form** · `unicode-sign` · docs/reference.md, Unicode: ∑, U+2211 N-ARY SUMMATION, is `\sum`

```unicode
∑_{«var»=«lower»}^{«upper»} «f»
```

### The sum in UnicodeMath

**Form** · `unicodemath` · UnicodeMath, Unicode Technical Note #28: `∑_(n=1)^∞`, the limits in round brackets

```unicode
∑_(«var»=«lower»)^«upper» «f»
```

### Limits as subscript and superscript characters

**Form** · `unicode-scripts` · the Unicode Standard, Superscripts and Subscripts, U+2070 to U+209F: `∑ₖ₌₁¹⁰`

```unicode
∑«lowered»«raised» «f»
```
