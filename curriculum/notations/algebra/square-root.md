# Square roots

The principal square root of a quantity, \(\sqrt{a}\), with a radical sign, as a power of
one half, and as a function. Lamport (1994) is *LaTeX: A Document Preparation System*, 2nd
edition; Knuth (1984) is *The TeXbook*; Stewart (2015) is *Calculus: Early
Transcendentals*, 8th edition; AsciiMath is the plain text notation of asciimath.org; the
Unicode Standard is version 15.0.

`«(radicand)»` is the radicand as it is written under an exponent or after a radical sign that
has no bar: in brackets where it is more than one letter or number.

## Names

```python
x = symbols("x")
```

## Operands

| operand | written in | `radicand` | `(radicand)` | separates |
|---|---|---|---|---|
| `symbol` | latex, text, unicode | `x` | `x` | |
| | python | `x` | | |
| `perfect-square` | latex, text, unicode | `16` | `16` | a root that is a whole number |
| | python | `16` | | |
| `two` | latex, text, unicode | `2` | `2` | a root that stays exact, and is no decimal |
| | python | `2` | | |
| `sum` | latex, text | `x^2 + 1` | `(x^2 + 1)` | a radical that covers a whole sum |
| | unicode | `x² + 1` | `(x² + 1)` | |
| | python | `x**2 + 1` | | |
| `fraction` | latex | `\frac{1}{4}` | `\left(\frac{1}{4}\right)` | |
| | text, unicode | `1/4` | `(1/4)` | |
| | python | `Rational(1, 4)` | | |
| `negative` | latex, text | `-4` | `(-4)` | the root of a negative number, which is imaginary |
| | unicode | `−4` | `(−4)` | |
| | python | `-4` | | |
| `product` | latex, unicode | `4x` | `(4x)` | a radical that covers a product |
| | text | `4*x` | `(4*x)` | |
| | python | `4*x` | | |

## Oracle

```python
expected = sqrt(S(radicand))
```

## Second route

The principal root: its square is the radicand, and its real part is not negative.

```python
assert simplify(expected**2 - S(radicand)) == 0
assert re(N(expected.subs(x, Rational(17, 10)))) >= 0
```

## Forms

### The radical sign

**Form** · `sqrt` · Lamport (1994), §3.3.1, Roots: `\sqrt{x + y}`

```latex
\sqrt{«radicand»}
```

### The radical sign without braces

**Form** · `sqrt-one-token` · Knuth (1984): an argument without braces is one token, as in `\sqrt2`

```latex
\sqrt «radicand»
```

**Only of** · `symbol`, `two` · the radicand is one character

### A root of index two

**Form** · `sqrt-index` · Lamport (1994), §3.3.1, Roots: `\sqrt[n]{2}`, with the index written

```latex
\sqrt[2]{«radicand»}
```

### Plain TeX's root

**Form** · `root-of` · Knuth (1984), plain TeX's macros: `\root n \of {x}`

```latex
\root 2 \of {«radicand»}
```

### The power one half

**Form** · `power-half` · Stewart (2015), §1.2, power functions: `x^{1/n} = \sqrt[n]{x}`, of n = 2

```latex
«(radicand)»^{1/2}
```

### The power one half, built up

**Form** · `power-frac` · Stewart (2015), §1.2, power functions, with the exponent set as `\frac{1}{2}`

```latex
«(radicand)»^{\frac{1}{2}}
```

### A call of sqrt

**Form** · `sqrt-call` · docs/language.md, Functions you can call: `sqrt`; AsciiMath's `sqrt(x)`, and every programming language's name

```text
sqrt(«radicand»)
```

### The power one half in plain text

**Form** · `caret-half` · AsciiMath: `x^(1/2)`, the brackets grouping the exponent

```text
«(radicand)»^(1/2)
```

### A call of root

**Form** · `root-call` · docs/reference.md, Functions, Numbers: `root`

```text
root(«radicand», 2)
```

### The radical sign in Unicode

**Form** · `unicode-radical` · the Unicode Standard, U+221A SQUARE ROOT: `√2`

```unicode
√«(radicand)»
```
