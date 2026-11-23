# Absolute values

The absolute value of a real quantity, \(|x|\), between bars and as a function. Lamport
(1994) is *LaTeX: A Document Preparation System*, 2nd edition; the amsmath package is
described in its *User's Guide* (American Mathematical Society, 2020); Knuth (1984) is
*The TeXbook*; Stewart (2015) is *Calculus: Early Transcendentals*, 8th edition; the
physics package is the LaTeX package of that name (de la Barrera, 2012).

## Names

```python
x = symbols("x", real=True)
```

## Operands

| operand | written in | `inside` | separates |
|---|---|---|---|
| `negative-number` | latex, text | `-3` | a value |
| | unicode | `−3` | |
| | python | `-3` | |
| `symbol` | latex, text, unicode | `x` | |
| | python | `x` | |
| `difference` | latex, text | `x - 1` | bars that hold a whole sum |
| | unicode | `x − 1` | |
| | python | `x - 1` | |
| `difference-of-numbers` | latex, text | `3 - 5` | |
| | unicode | `3 − 5` | |
| | python | `3 - 5` | |
| `fraction` | latex | `-\frac{1}{2}` | |
| | text | `-1/2` | |
| | unicode | `−1/2` | |
| | python | `-Rational(1, 2)` | |
| `product` | latex | `-2x` | bars on a negative multiple of a letter, which drop its sign |
| | text | `-2*x` | |
| | unicode | `−2x` | |
| | python | `-2*x` | |
| `negative-inside` | latex, text | `x - 5` | a quantity that is negative where it is compared, whose bars cannot be dropped |
| | unicode | `x − 5` | |
| | python | `x - 5` | |
| `nested` | latex, text | `\|x\| - 3` | bars within bars, each closing the one it opened |
| | unicode | `\|x\| − 3` | |
| | python | `Abs(x) - 3` | |

## Oracle

```python
expected = Abs(S(inside))
```

## Second route

The absolute value of a real quantity is the root of its square, at a point.

```python
at = {x: -Rational(17, 10)}
assert abs(N((expected - sqrt(S(inside) ** 2)).subs(at), 30)) < 1e-20
```

## Forms

### Bars

**Form** · `bars` · Stewart (2015), Appendix A: `|a|`

```latex
|«inside»|
```

### Bars that grow

**Form** · `left-right` · Lamport (1994), §3.3.4: `\left|` and `\right|`

```latex
\left| «inside» \right|
```

### The amsmath package's bars

**Form** · `lvert` · the amsmath package, Vertical bar notations: `\lvert` and `\rvert`

```latex
\lvert «inside» \rvert
```

### The amsmath package's bars, growing

**Form** · `left-lvert` · the amsmath package, Vertical bar notations: `\left\lvert` and `\right\rvert`

```latex
\left\lvert «inside» \right\rvert
```

### Bars by their name

**Form** · `vert` · Knuth (1984), plain TeX's delimiters: `\vert` is `|`

```latex
\vert «inside» \vert
```

### Bars of a size

**Form** · `bigl` · the amsmath package, Delimiter sizes: `\bigl|` and `\bigr|`

```latex
\bigl| «inside» \bigr|
```

### The physics package's absolute value

**Form** · `physics-abs` · the physics package, Automatic bracing: `\abs{x}`

```latex
\abs{«inside»}
```

**Refused** · `ParseError` · docs/reference.md, Not supported: `\abs` is an error that says to write `|x|`

### A call of abs

**Form** · `abs-call` · docs/language.md, Functions you can call: `Abs`; docs/reference.md: `abs`

```text
abs(«inside»)
```

### A call of Abs

**Form** · `Abs-call` · docs/language.md, Functions you can call: `Abs`, SymPy's name

```text
Abs(«inside»)
```

### Bars, in plain text

**Form** · `text-bars` · docs/language.md, Piecewise functions and other notation: `|-3|`

```text
|«inside»|
```

### Bars, in Unicode

**Form** · `unicode-bars` · as typed, with Unicode's minus sign

```unicode
|«inside»|
```
