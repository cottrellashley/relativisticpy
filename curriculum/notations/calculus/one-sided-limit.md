# One-sided limits

The limit of a function of one variable as the variable comes to a point from one side,
\(\lim_{x \to a^+} f(x)\) from above and \(\lim_{x \to a^-} f(x)\) from below. Stewart (2015)
is *Calculus: Early Transcendentals*, 8th edition; Lamport (1994) is *LaTeX: A Document
Preparation System*, 2nd edition; Apostol (1974) is *Mathematical Analysis*, 2nd edition;
Durrett (2019) is *Probability: Theory and Examples*, 5th edition; Forster (2016) is
*Analysis 1*, 12th edition; the unicode-math package is the LaTeX package of that name, which
takes Unicode's symbols in maths; AsciiMath is the plain text notation of asciimath.org;
UnicodeMath is the linear format of Unicode Technical Note #28; the Unicode Standard is
version 15.0.

`«f»` is the function, `«var»` its variable and `«point»` where it goes. The side is written
`«side»`, `+` from above and `-` from below, as a sign after the point; `«arrow»`, `\downarrow`
or `\uparrow`, as an arrow that says which way the variable goes; `«diagonal»`, `\searrow` or
`\nearrow`, as a slanted one; and `«raised»`, `⁺` or `⁻`, in Unicode's superscript
characters. The operands go to the same points from both sides, so a side that is not read
gives another value.

## Names

```python
x = symbols("x")
```

## Operands

| operand | written in | `f` | `var` | `point` | `side` | `arrow` | `diagonal` | `raised` | separates |
|---|---|---|---|---|---|---|---|---|---|
| `reciprocal-above` | latex | `\frac{1}{x}` | `x` | `0` | `+` | `\downarrow` | `\searrow` | | an infinite limit from above, of a function with no limit from both sides |
| | text | `1/x` | `x` | `0` | `+` | | | | |
| | unicode | `1/x` | `x` | `0` | | | | `⁺` | |
| | python | `1/x` | `x` | `0` | `"+"` | | | | |
| `reciprocal-below` | latex | `\frac{1}{x}` | `x` | `0` | `-` | `\uparrow` | `\nearrow` | | the same function from below, minus infinity |
| | text | `1/x` | `x` | `0` | `-` | | | | |
| | unicode | `1/x` | `x` | `0` | | | | `⁻` | |
| | python | `1/x` | `x` | `0` | `"-"` | | | | |
| `sign-above` | latex | `\frac{\|x\|}{x}` | `x` | `0` | `+` | `\downarrow` | `\searrow` | | a jump, from above |
| | text | `abs(x)/x` | `x` | `0` | `+` | | | | |
| | unicode | `\|x\|/x` | `x` | `0` | | | | `⁺` | |
| | python | `Abs(x)/x` | `x` | `0` | `"+"` | | | | |
| `sign-below` | latex | `\frac{\|x\|}{x}` | `x` | `0` | `-` | `\uparrow` | `\nearrow` | | the jump, from below |
| | text | `abs(x)/x` | `x` | `0` | `-` | | | | |
| | unicode | `\|x\|/x` | `x` | `0` | | | | `⁻` | |
| | python | `Abs(x)/x` | `x` | `0` | `"-"` | | | | |
| `exponential-below` | latex | `e^{1/x}` | `x` | `0` | `-` | `\uparrow` | `\nearrow` | | a function that is 0 from below and infinite from above |
| | text | `exp(1/x)` | `x` | `0` | `-` | | | | |
| | unicode | `e^(1/x)` | `x` | `0` | | | | `⁻` | |
| | python | `exp(1/x)` | `x` | `0` | `"-"` | | | | |
| `x-log-x` | latex | `x \ln x` | `x` | `0` | `+` | `\downarrow` | `\searrow` | | a function defined only above the point |
| | text | `x*log(x)` | `x` | `0` | `+` | | | | |
| | unicode | `x ln x` | `x` | `0` | | | | `⁺` | |
| | python | `x*log(x)` | `x` | `0` | `"+"` | | | | |
| `floor-below` | latex | `\lfloor x \rfloor` | `x` | `2` | `-` | `\uparrow` | `\nearrow` | | a step at a point that is not 0, from below |
| | text | `floor(x)` | `x` | `2` | `-` | | | | |
| | unicode | `⌊x⌋` | `x` | `2` | | | | `⁻` | |
| | python | `floor(x)` | `x` | `2` | `"-"` | | | | |

## Oracle

```python
expected = limit(f, var, point, side)
```

## Second route

The function a small step from the point, on its side, as numbers: near the limit where it
is a number, and large, with its sign, where it is infinite.

```python
toward = 1 if side == "+" else -1
steps = [1 / (sqrt(2) * 10**k) for k in (8, 11, 14)]
values = [N(f.subs(var, point + toward * step), 50) for step in steps]
if expected.is_infinite:
    assert all(value * sign(expected) > 10**5 for value in values)
else:
    assert all(abs(value - expected) < 10**-5 for value in values)
```

## Forms

### The side as a superscript

**Form** · `superscript` · Stewart (2015), §2.2, One-Sided Limits: `\lim_{x \to a^-} f(x)` and `\lim_{x \to a^+} f(x)`

```latex
\lim_{«var» \to «point»^«side»} «f»
```

### The side as a superscript in braces

**Form** · `superscript-braces` · Lamport (1994), §3.3.1: a superscript in braces, as LaTeX's manual writes every script

```latex
\lim_{«var» \to «point»^{«side»}} «f»
```

### The side as a sign after the point

**Form** · `sign-after` · Apostol (1974), one-sided limits: `f(c+) = \lim_{x \to c+} f(x)`

```latex
\lim_{«var» \to «point»«side»} «f»
```

### An arrow that says the way

**Form** · `vertical-arrow` · Durrett (2019), §1.2, distribution functions: `\lim_{y \downarrow x} F(y) = F(x)` and `\lim_{y \uparrow x} F(y)`

```latex
\lim_{«var» «arrow» «point»} «f»
```

### A slanted arrow

**Form** · `slanted-arrow` · Forster (2016), limits of functions: `\lim_{x \searrow a}` and `\lim_{x \nearrow a}`

```latex
\lim_{«var» «diagonal» «point»} «f»
```

### Unicode's arrow in LaTeX

**Form** · `unicode-math` · the unicode-math package: Unicode's arrow, U+2192, typed in LaTeX's maths for `\to`

```latex
\lim_{«var» → «point»^«side»} «f»
```

### A call of limit, with its side

**Form** · `limit` · docs/reference.md, Functions: `limit`; SymPy's `limit(1/x, x, 0, '+')`, the side its fourth argument

```text
limit(«f», «var», «point», '«side»')
```

### The limit in plain text

**Form** · `asciimath` · AsciiMath: `lim_(N->oo)`, with the side as a superscript on the point

```text
lim_(«var»->«point»^«side») «f»
```

### The limit in UnicodeMath

**Form** · `unicodemath` · UnicodeMath, Unicode Technical Note #28: `lim_(n→∞)`, the side in the superscript characters U+207A and U+207B

```unicode
lim_(«var»→«point»«raised») «f»
```
