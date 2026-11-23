# Brackets that group

A factor times a sum in brackets, \(a(b + c)\), with each kind of bracket and each way of
sizing it. Lamport (1994) is *LaTeX: A Document Preparation System*, 2nd edition; Knuth
(1984) is *The TeXbook*; the amsmath package is described in its *User's Guide* (American
Mathematical Society, 2020); Stewart (2015) is *Calculus: Early Transcendentals*, 8th
edition; AsciiMath is the plain text notation of asciimath.org.

## Names

```python
x, a, b, c = symbols("x a b c")
```

## Operands

| operand | written in | `factor` | `inside` | separates |
|---|---|---|---|---|
| `number` | latex, text, unicode | `2` | `x + 1` | |
| | python | `2` | `x + 1` | |
| `letter` | latex, text | `x` | `x - 1` | a name before a bracket, which is a product where the name is a variable in it (design/notation.md, decision 28) |
| | unicode | `x` | `x − 1` | |
| | python | `x` | `x - 1` | |
| `fraction` | latex | `\frac{1}{2}` | `a + b` | |
| | text, unicode | `(1/2)` | `a + b` | |
| | python | `Rational(1, 2)` | `a + b` | |
| `nested` | latex, text | `2` | `3 + 4(5 - 1)` | brackets within brackets |
| | unicode | `2` | `3 + 4(5 − 1)` | |
| | python | `2` | `3 + 4*(5 - 1)` | |
| `three-letters` | latex, text, unicode | `a` | `b + c` | |
| | python | `a` | `b + c` | |

## Oracle

```python
expected = expand(S(factor) * inside)
```

## Second route

The product, factor by factor, at a point.

```python
at = {x: Rational(17, 10), a: Rational(13, 10), b: Rational(7, 5), c: Rational(11, 5)}
assert (expected - S(factor) * inside).subs(at) == 0
```

## Forms

### Round brackets

**Form** · `round` · Lamport (1994), §3.3.4, Delimiters

```latex
«factor»(«inside»)
```

### Round brackets that grow

**Form** · `left-right` · Lamport (1994), §3.3.4: `\left(` and `\right)`

```latex
«factor»\left(«inside»\right)
```

### Round brackets of a size

**Form** · `bigl` · the amsmath package, Delimiter sizes: `\bigl(` and `\bigr)`

```latex
«factor»\bigl(«inside»\bigr)
```

### Round brackets of plain TeX's size

**Form** · `big` · Knuth (1984), plain TeX's sizes of delimiters: `\big(` and `\big)`

```latex
«factor»\big(«inside»\big)
```

### Larger round brackets

**Form** · `Bigl` · the amsmath package, Delimiter sizes: `\Bigl(` and `\Bigr)`

```latex
«factor»\Bigl(«inside»\Bigr)
```

### Square brackets

**Form** · `square` · Stewart (2015), §3.2: `\frac{d}{dx}[f(x)g(x)]`, square brackets that group

```latex
«factor»[«inside»]
```

### Square brackets that grow

**Form** · `left-right-square` · docs/latex.md, Brackets group: `\left[ ... \right]` groups like parentheses

```latex
«factor»\left[«inside»\right]
```

### Braces

**Form** · `braces` · docs/latex.md, Brackets group: `\{ ... \}` groups like parentheses

```latex
«factor»\{«inside»\}
```

### Braces that grow

**Form** · `left-right-braces` · Lamport (1994), §3.3.4: `\left\{` and `\right\}`

```latex
«factor»\left\{«inside»\right\}
```

### Round brackets, in plain text

**Form** · `text-round` · the brackets of plain text, as AsciiMath writes `2(x+1)`

```text
«factor»(«inside»)
```

### Round brackets and a star, in plain text

**Form** · `text-star` · the brackets of plain text, the product written with a star

```text
«factor»*(«inside»)
```

### Round brackets, in Unicode

**Form** · `unicode-round` · as typed, with Unicode's minus sign

```unicode
«factor»(«inside»)
```
