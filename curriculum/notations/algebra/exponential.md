# The exponential

Euler's number raised to a power, \(e^x\), as a power of e and as the function exp.
Stewart (2015) is *Calculus: Early Transcendentals*, 8th edition; ISO 80000-2:2019 is the
standard for mathematical signs and symbols; DLMF is the NIST *Digital Library of
Mathematical Functions*; Bishop (2006) is *Pattern Recognition and Machine Learning*;
Lamport (1994) is *LaTeX: A Document Preparation System*, 2nd edition.

`«(power)»` is the power as a line of plain text writes it after `^`: in brackets where it
is more than one letter or number.

## Names

```python
x, a, t = symbols("x a t")
```

## Operands

| operand | written in | `power` | `(power)` | separates |
|---|---|---|---|---|
| `x` | latex, text, unicode | `x` | `x` | |
| | python | `x` | | |
| `2x` | latex, unicode | `2x` | `(2x)` | a power that is a product |
| | text | `2*x` | `(2*x)` | |
| | python | `2*x` | | |
| `negative-square` | latex, text | `-x^2` | `(-x^2)` | a sign and a power in the power |
| | unicode | `−x²` | `(−x²)` | |
| | python | `-x**2` | | |
| `two-letters` | latex | `at` | `(at)` | a power of two letters, which is their product |
| | text | `a*t` | `(a*t)` | |
| | unicode | `a t` | `(a t)` | |
| | python | `a*t` | | |
| `i-pi` | latex | `i\pi` | `(i\pi)` | Euler's identity: a value, with i and π in the power |
| | text | `i*pi` | `(i*pi)` | |
| | unicode | `iπ` | `(iπ)` | |
| | python | `I*pi` | | |
| `sum` | latex, text, unicode | `x + 1` | `(x + 1)` | a power that is a sum |
| | python | `x + 1` | | |
| `zero` | latex, text, unicode | `0` | `0` | a value |
| | python | `0` | | |
| `logarithm` | latex | `\ln 2` | `(\ln 2)` | the exponential of a logarithm, which is its argument |
| | text | `ln(2)` | `(ln(2))` | |
| | unicode | `ln 2` | `(ln 2)` | |
| | python | `log(2)` | | |

## Oracle

```python
expected = exp(S(power))
```

## Second route

The exponential as the sum of its series, at a point.

```python
at = {x: Rational(17, 10), a: Rational(7, 5), t: Rational(3, 5)}
value = S(power).subs(at)
by_series = sum(value**k / factorial(k) for k in range(80))
assert abs(N(expected.subs(at) - by_series, 30)) < 1e-20
```

## Forms

### A power of e

**Form** · `e-power` · Stewart (2015), §1.4, The Number e: `e^x`

```latex
e^{«power»}
```

### A power of an upright e

**Form** · `upright-e` · ISO 80000-2:2019 and DLMF, §4.2: `\mathrm{e}^z`, the constant set upright

```latex
\mathrm{e}^{«power»}
```

### exp with round brackets

**Form** · `exp-round` · DLMF, §4.2: `\exp z`, its argument in brackets, as the DLMF writes the arguments of other functions

```latex
\exp(«power»)
```

### exp with brackets that grow

**Form** · `exp-left-right` · Lamport (1994), §3.3.4: `\left(` and `\right)`

```latex
\exp\left(«power»\right)
```

### exp with square brackets

**Form** · `exp-square` · docs/latex.md, Functions take what follows them: `\exp[-x^2]`

```latex
\exp[«power»]
```

### exp with braces

**Form** · `exp-braces` · Bishop (2006), eq. (1.46): the Gaussian written `\exp\left\{ ... \right\}`

```latex
\exp\left\{«power»\right\}
```

### exp without brackets

**Form** · `exp-bare` · DLMF, §4.2: `\exp z`

```latex
\exp «power»
```

**Only of** · `x`, `2x`, `two-letters`, `i-pi`, `zero` · a sum, a sign or a function is an argument only in brackets

### A call of exp

**Form** · `exp-call` · docs/language.md, Functions you can call: `exp`

```text
exp(«power»)
```

### A power of e, in plain text

**Form** · `caret` · docs/language.md, Numbers and arithmetic: `e` is the constant, and `^` a power

```text
e^«(power)»
```

### A power of e, with two stars

**Form** · `double-star` · docs/language.md, Numbers and arithmetic: powers are written `^` or `**`

```text
e**«(power)»
```

### A power of e, in Unicode

**Form** · `unicode-caret` · as typed, with Unicode's letters, superscripts and minus sign

```unicode
e^«(power)»
```

### A call of exp, in Unicode

**Form** · `unicode-call` · as typed, with Unicode's letters, superscripts and minus sign

```unicode
exp(«power»)
```
