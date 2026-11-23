# Roots of any index

The principal nth root of a quantity, \(\sqrt[n]{a}\), with a radical sign and its index,
and as a power of one over n. Lamport (1994) is *LaTeX: A Document Preparation System*,
2nd edition; Knuth (1984) is *The TeXbook*; Stewart (2015) is *Calculus: Early
Transcendentals*, 8th edition; AsciiMath is the plain text notation of asciimath.org; the
Unicode Standard is version 15.0.

`«(radicand)»` is the radicand as it is written under an exponent: in brackets where it is more
than one letter or number.

## Names

```python
x, n = symbols("x n")
```

## Operands

| operand | written in | `radicand` | `index` | `(radicand)` | separates |
|---|---|---|---|---|---|
| `cube-root-of-8` | latex, text, unicode | `8` | `3` | `8` | a root that is a whole number |
| | python | `8` | `3` | | |
| `cube-root` | latex, text, unicode | `x` | `3` | `x` | |
| | python | `x` | `3` | | |
| `fourth-root` | latex, text, unicode | `x` | `4` | `x` | |
| | python | `x` | `4` | | |
| `fifth-root-of-32` | latex, text, unicode | `32` | `5` | `32` | |
| | python | `32` | `5` | | |
| `of-a-sum` | latex, text | `x^2 + 1` | `3` | `(x^2 + 1)` | a radical that covers a whole sum |
| | unicode | `x² + 1` | `3` | `(x² + 1)` | |
| | python | `x**2 + 1` | `3` | | |
| `of-a-fraction` | latex | `\frac{27}{8}` | `3` | `\left(\frac{27}{8}\right)` | |
| | text, unicode | `27/8` | `3` | `(27/8)` | |
| | python | `Rational(27, 8)` | `3` | | |
| `symbolic-index` | latex, text, unicode | `x` | `n` | `x` | an index that is a letter |
| | python | `x` | `n` | | |

## Oracle

```python
expected = root(S(radicand), S(index))
```

## Second route

The root, raised to its index, is the radicand: compared at a point, as numbers.

```python
at = {x: Rational(17, 10), n: 7}
assert abs(N((expected ** S(index) - S(radicand)).subs(at), 30)) < 1e-20
```

## Forms

### The radical sign with its index

**Form** · `sqrt-index` · Lamport (1994), §3.3.1, Roots: `\sqrt[n]{2}`

```latex
\sqrt[«index»]{«radicand»}
```

### Plain TeX's root

**Form** · `root-of` · Knuth (1984), plain TeX's macros: `\root n \of {x}`

```latex
\root «index» \of {«radicand»}
```

### The power one over n

**Form** · `power-solidus` · Stewart (2015), §1.2, power functions: `x^{1/n} = \sqrt[n]{x}`

```latex
«(radicand)»^{1/«index»}
```

### The power one over n, built up

**Form** · `power-frac` · Stewart (2015), §1.2, power functions, with the exponent set as `\frac{1}{n}`

```latex
«(radicand)»^{\frac{1}{«index»}}
```

### A call of root

**Form** · `root-call` · docs/reference.md, Functions, Numbers: `root`

```text
root(«radicand», «index»)
```

### The power one over n in plain text

**Form** · `caret` · AsciiMath: `x^(1/3)`, the brackets grouping the exponent

```text
«(radicand)»^(1/«index»)
```

### AsciiMath's root

**Form** · `asciimath-root` · AsciiMath: `root(3)(x)`

```text
root(«index»)(«radicand»)
```

### The cube root sign

**Form** · `unicode-cube-root` · the Unicode Standard, U+221B CUBE ROOT

```unicode
∛«(radicand)»
```

**Only of** · `cube-root-of-8`, `cube-root`, `of-a-sum`, `of-a-fraction` · the index is three

### The fourth root sign

**Form** · `unicode-fourth-root` · the Unicode Standard, U+221C FOURTH ROOT

```unicode
∜«(radicand)»
```

**Only of** · `fourth-root` · the index is four
