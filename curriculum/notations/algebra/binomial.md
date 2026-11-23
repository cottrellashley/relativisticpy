# Binomial coefficients

The number of ways to choose k things of n, \(\binom{n}{k}\), stacked in brackets, as a
function, and by factorials; and, as Newton's binomial series takes it, of a top that is
not a whole number. Graham, Knuth and Patashnik (1994) is *Concrete Mathematics*, 2nd
edition; Knuth (1984) is *The TeXbook*; the amsmath package is described in its *User's
Guide* (American Mathematical Society, 2020); DLMF is the NIST *Digital Library of
Mathematical Functions*; Rosen (2019) is *Discrete Mathematics and Its Applications*, 8th
edition; the Unicode Standard is version 15.0.

`«top»` is the number chosen from and `«bottom»` the number chosen; `«(top)»` and
`«(bottom)»` are the same before `!`, in brackets where they are more than one letter or
digit.

## Names

```python
n, k = symbols("n k")
```

## Operands

| operand | written in | `top` | `bottom` | `(top)` | `(bottom)` | separates |
|---|---|---|---|---|---|---|
| `five-two` | latex, text, unicode | `5` | `2` | `5` | `2` | a value |
| | python | `5` | `2` | | | |
| `ten-three` | latex, text, unicode | `10` | `3` | `10` | `3` | numbers of two digits |
| | python | `10` | `3` | | | |
| `n-two` | latex, text, unicode | `n` | `2` | `n` | `2` | a top that is a letter: n(n − 1)/2 |
| | python | `n` | `2` | | | |
| `symbols` | latex, text, unicode | `n` | `k` | `n` | `k` | |
| | python | `n` | `k` | | | |
| `none` | latex, text, unicode | `n` | `0` | `n` | `0` | choosing nothing, which is done one way |
| | python | `n` | `0` | | | |
| `more-than-there-are` | latex, text, unicode | `3` | `5` | `3` | `5` | a bottom above the top, which is chosen no way |
| | python | `3` | `5` | | | |
| `half` | latex | `\frac{1}{2}` | `2` | | | a top that is not a whole number, as in Newton's binomial series |
| | text, unicode | `1/2` | `2` | | | |
| | python | `Rational(1, 2)` | `2` | | | |
| `negative-top` | latex, text | `-1` | `3` | | | a negative top: (−1)ᵏ |
| | unicode | `−1` | `3` | | | |
| | python | `-1` | `3` | | | |

## Oracle

```python
expected = binomial(S(top), S(bottom))
```

## Second route

The falling factorial of the top, as many factors as the bottom says, over the bottom's
factorial (Graham, Knuth and Patashnik (1994), eq. (5.1)): compared at a point where the
bottom is a letter.

```python
by_falling = ff(S(top), S(bottom)) / factorial(S(bottom))
at = {n: Rational(17, 2), k: 3}
assert simplify(expected.subs(at) - by_falling.subs(at)) == 0
assert expected.is_number or simplify((expected - by_falling).subs(k, 4)) == 0
```

## Forms

### amsmath's binomial

**Form** · `binom` · the amsmath package, Fractions and related constructions: `\binom{n}{k}`

```latex
\binom{«top»}{«bottom»}
```

### amsmath's binomial in display style

**Form** · `dbinom` · the amsmath package, Fractions and related constructions: `\dbinom`

```latex
\dbinom{«top»}{«bottom»}
```

### amsmath's binomial in text style

**Form** · `tbinom` · the amsmath package, Fractions and related constructions: `\tbinom`

```latex
\tbinom{«top»}{«bottom»}
```

### Plain TeX's binomial

**Form** · `choose` · Knuth (1984), chapter 17: `{n \choose k}`

```latex
{«top» \choose «bottom»}
```

### By factorials

**Form** · `factorials` · DLMF, §26.3: `\binom{m}{n} = \frac{m!}{(m - n)!\, n!}`, as Rosen (2019), §6.3, writes it, `\frac{n!}{r!(n - r)!}`

```latex
\frac{«(top)»!}{«(bottom)»!(«top» - «bottom»)!}
```

**Only of** · `five-two`, `ten-three`, `n-two`, `symbols`, `none` · factorials of whole numbers that are not negative

### The letter C

**Form** · `C` · Rosen (2019), §6.3: `C(n, r)`, the number of r-combinations of a set of n elements

```latex
C(«top», «bottom»)
```

### A call of binomial

**Form** · `binomial-call` · docs/language.md, Functions you can call: `binomial`; SymPy's name

```text
binomial(«top», «bottom»)
```

### The letter C, in plain text

**Form** · `text-C` · Rosen (2019), §6.3: `C(n, r)`

```text
C(«top», «bottom»)
```

### By factorials, in plain text

**Form** · `text-factorials` · DLMF, §26.3, in a line of plain text: `n!/(k!(n-k)!)`

```text
«(top)»!/(«(bottom)»!(«top» - «bottom»)!)
```

**Only of** · `five-two`, `ten-three`, `n-two`, `symbols`, `none` · factorials of whole numbers that are not negative

### By factorials, in Unicode

**Form** · `unicode-factorials` · as typed, with Unicode's minus sign, U+2212

```unicode
«(top)»!/(«(bottom)»!(«top» − «bottom»)!)
```

**Only of** · `five-two`, `ten-three`, `n-two`, `symbols`, `none` · factorials of whole numbers that are not negative
