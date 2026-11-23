# Powers

A base raised to an exponent, \(a^n\), and above all a power of a product or of a bracket,
\((xy)^2\), which is the power of the whole. Knuth (1984) is *The TeXbook*; Lamport (1994) is
*LaTeX: A Document Preparation System*, 2nd edition; the Python language reference is that
of Python 3.10; AsciiMath is the plain text notation of asciimath.org; the Unicode Standard
is version 15.0.

`«(base)»` is the base as it is written under an exponent, in brackets where it is more than
one letter or number; `«(exponent)»` is the exponent as a line of plain text writes it after
`^`, in brackets where it is more than one number; `«raised»` is the exponent in Unicode's
superscript characters.

## Names

```python
x, y, a, n, alpha, beta = symbols("x y a n alpha beta")
```

## Operands

| operand | written in | `base` | `exponent` | `(base)` | `(exponent)` | `raised` | separates |
|---|---|---|---|---|---|---|---|
| `square` | latex, text | `x` | `2` | `x` | `2` | | |
| | unicode | `x` | `2` | `x` | `2` | `²` | |
| | python | `x` | `2` | | | | |
| `two-digits` | latex, text | `x` | `10` | `x` | `10` | | an exponent of two digits, which is one number |
| | unicode | `x` | `10` | `x` | `10` | `¹⁰` | |
| | python | `x` | `10` | | | | |
| `negative-exponent` | latex, text | `x` | `-1` | `x` | `(-1)` | | a sign in the exponent |
| | unicode | `x` | `−1` | `x` | `(−1)` | `⁻¹` | |
| | python | `x` | `-1` | | | | |
| `fraction-exponent` | latex, text | `x` | `1/2` | `x` | `(1/2)` | | an exponent that is a quotient |
| | unicode | `x` | `1/2` | `x` | `(1/2)` | | |
| | python | `x` | `Rational(1, 2)` | | | | |
| `stacked-exponent` | latex | `x` | `\frac{1}{2}` | `x` | | | a built-up fraction in the exponent |
| | python | `x` | `Rational(1, 2)` | | | | |
| `sum-exponent` | latex, text, unicode | `a` | `n + 1` | `a` | `(n + 1)` | | an exponent that is a sum |
| | python | `a` | `n + 1` | | | | |
| `sum-base` | latex, text, unicode | `x + 1` | `2` | `(x + 1)` | `2` | | a power of a whole sum |
| | python | `x + 1` | `2` | | | | |
| `product-base` | latex | `xy` | `2` | `(xy)` | `2` | | a power of a product from a power of its last factor |
| | text | `x*y` | `2` | `(x*y)` | `2` | | |
| | unicode | `x y` | `2` | `(x y)` | `2` | `²` | |
| | python | `x*y` | `2` | | | | |
| `number-times-letter` | latex, unicode | `2x` | `3` | `(2x)` | `3` | `³` | a power of a number times a letter, which is eight times the cube |
| | text | `2*x` | `3` | `(2*x)` | `3` | | |
| | python | `2*x` | `3` | | | | |
| `greek-product` | latex | `\alpha\beta` | `2` | `(\alpha\beta)` | `2` | | a power of a product of Greek letters, written as commands |
| | text | `alpha*beta` | `2` | `(alpha*beta)` | `2` | | |
| | unicode | `αβ` | `2` | `(αβ)` | `2` | `²` | |
| | python | `alpha*beta` | `2` | | | | |
| `power-of-power` | latex | `x^2` | `3` | `(x^2)` | `3` | | a power of a power, which multiplies the exponents |
| | text | `x^2` | `3` | `(x^2)` | `3` | | |
| | unicode | `x²` | `3` | `(x²)` | `3` | `³` | |
| | python | `x**2` | `3` | | | | |
| `negative-base` | latex, text | `-3` | `2` | `(-3)` | `2` | | the square of a negative number from the negative of a square |
| | unicode | `−3` | `2` | `(−3)` | `2` | `²` | |
| | python | `-3` | `2` | | | | |
| `tower` | latex | `2` | `3^2` | `2` | `(3^2)` | | a power of a power, taken from the top |
| | text, unicode | `2` | `3^2` | `2` | `(3^2)` | | |
| | python | `2` | `3**2` | | | | |

## Oracle

```python
expected = Pow(S(base), S(exponent))
```

## Second route

A power is the exponential of its exponent times the logarithm of its base: compared at a
point, as numbers.

```python
at = {x: Rational(17, 10), y: Rational(13, 10), a: Rational(7, 5), n: Rational(5, 2)}
at.update({alpha: Rational(3, 7), beta: Rational(11, 5)})
by_logarithm = exp(S(exponent) * log(S(base))).subs(at)
assert abs(N(expected.subs(at) - by_logarithm, 30)) < 1e-20
```

## Forms

### A superscript in braces

**Form** · `braces` · Lamport (1994), §3.3.1, Subscripts and Superscripts: `x^{2y}`

```latex
«(base)»^{«exponent»}
```

### A superscript without braces

**Form** · `one-token` · Knuth (1984), chapter 16: a superscript without braces is the one character after `^`, as in `x^2`

```latex
«(base)»^«exponent»
```

**Only of** · `square`, `sum-base`, `product-base`, `number-times-letter`, `greek-product`, `power-of-power`, `negative-base` · the exponent is one digit

### A superscript on brackets that grow

**Form** · `left-right` · Lamport (1994), §3.3.4, Delimiters: `\left(` and `\right)`

```latex
\left(«base»\right)^{«exponent»}
```

### A caret in plain text

**Form** · `caret` · AsciiMath: `x^2`, and `x^(n+1)`, the brackets grouping the exponent

```text
«(base)»^«(exponent)»
```

### Two stars

**Form** · `double-star` · docs/language.md, Numbers and arithmetic: powers are written `^` or `**`, as Python and Fortran write them

```text
«(base)»**«(exponent)»
```

### A tower in plain text, without brackets

**Form** · `caret-tower` · the Python language reference, §6.5, The power operator: an unbracketed run of powers is evaluated from right to left, so `2**3**2` is `2**(3**2)`; Mathematica's `^` likewise

```text
2^3^2
```

**Only of** · `tower` · the operand is this tower

### A superscript character

**Form** · `unicode-superscript` · the Unicode Standard, Superscripts and Subscripts, U+2070 to U+209F, and U+00B2 SUPERSCRIPT TWO: `x²`

```unicode
«(base)»«raised»
```

### A caret in Unicode

**Form** · `unicode-caret` · as typed, with Unicode's minus sign, U+2212, and Greek letters

```unicode
«(base)»^«(exponent)»
```
