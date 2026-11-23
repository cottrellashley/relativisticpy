# Logarithms to a base

The logarithm of a quantity to a given base, \(\log_b x\). Stewart (2015) is *Calculus:
Early Transcendentals*, 8th edition; ISO 80000-2:2019 is the standard for mathematical
signs and symbols; Lamport (1994) is *LaTeX: A Document Preparation System*, 2nd edition;
AsciiMath is the plain text notation of asciimath.org; the Unicode Standard is version
15.0.

`«lowered»` is the base in Unicode's subscript characters.

## Names

```python
x, b = symbols("x b")
```

## Operands

| operand | written in | `base` | `argument` | `lowered` | separates |
|---|---|---|---|---|---|
| `binary` | latex, text | `2` | `8` | | a value |
| | unicode | `2` | `8` | `₂` | |
| | python | `2` | `8` | | |
| `decimal` | latex, text | `10` | `1000` | | a base of two digits |
| | unicode | `10` | `1000` | `₁₀` | |
| | python | `10` | `1000` | | |
| `symbols` | latex, text, unicode | `b` | `x` | | |
| | python | `b` | `x` | | |
| `power` | latex | `2` | `2^{10}` | | the logarithm of a power from the power of a logarithm |
| | text | `2` | `2^10` | | |
| | unicode | `2` | `2¹⁰` | `₂` | |
| | python | `2` | `2**10` | | |
| `of-a-symbol` | latex, text | `2` | `x` | | |
| | unicode | `2` | `x` | `₂` | |
| | python | `2` | `x` | | |
| `ternary` | latex, text | `3` | `81` | | |
| | unicode | `3` | `81` | `₃` | |
| | python | `3` | `81` | | |

## Oracle

```python
expected = log(S(argument), S(base))
```

## Second route

The base raised to the logarithm is the argument, at a point.

```python
at = {x: Rational(17, 10), b: Rational(7, 2)}
assert abs(N((S(base) ** expected - S(argument)).subs(at), 30)) < 1e-20
```

## Forms

### The base as a subscript in braces

**Form** · `braces` · Stewart (2015), §1.5: `\log_b x`

```latex
\log_{«base»} «argument»
```

### The base as a subscript without braces

**Form** · `one-token` · Stewart (2015), §1.5: `\log_2 8`, the base one character

```latex
\log_«base» «argument»
```

**Not of** · `decimal` · a base of two digits is written in braces

### The argument in round brackets

**Form** · `round` · Stewart (2015), §1.5: `\log_b(b^x) = x`

```latex
\log_{«base»}(«argument»)
```

### The argument in brackets that grow

**Form** · `left-right` · Lamport (1994), §3.3.4: `\left(` and `\right)`

```latex
\log_{«base»}\left(«argument»\right)
```

### The change of base, with ln

**Form** · `change-of-base` · Stewart (2015), §1.5: `\log_b x = \frac{\ln x}{\ln b}`

```latex
\frac{\ln «argument»}{\ln «base»}
```

### The common logarithm

**Form** · `lg` · ISO 80000-2:2019: `\lg x` is the logarithm to the base 10

```latex
\lg «argument»
```

**Only of** · `decimal` · the base is ten

### The binary logarithm

**Form** · `lb` · ISO 80000-2:2019: `\operatorname{lb} x` is the logarithm to the base 2

```latex
\operatorname{lb} «argument»
```

**Only of** · `binary`, `of-a-symbol` · the base is two, and the argument one term

### A call of log with its base

**Form** · `log-call` · docs/reference.md, Functions, Numbers: `log`; SymPy's `log(x, b)`

```text
log(«argument», «base»)
```

### The base named in the function

**Form** · `log2-log10` · the names of C, Python and NumPy: `log2(8)`, `log10(1000)`

```text
log«base»(«argument»)
```

**Only of** · `binary`, `decimal`, `power`, `of-a-symbol` · the base is two or ten

### The base after an underscore

**Form** · `underscore` · AsciiMath: `log_2(8)`, the base typed as a subscript is in plain text

```text
log_«base»(«argument»)
```

### The change of base, in plain text

**Form** · `text-change-of-base` · Stewart (2015), §1.5, in plain text: `ln(x)/ln(b)`

```text
ln(«argument»)/ln(«base»)
```

### The base as a subscript character

**Form** · `unicode-subscript` · the Unicode Standard, Superscripts and Subscripts, U+2080 to U+2089: `log₂ 8`

```unicode
log«lowered» «argument»
```

### The base as a subscript character, the argument in brackets

**Form** · `unicode-subscript-round` · the Unicode Standard, Superscripts and Subscripts: `log₂(8)`

```unicode
log«lowered»(«argument»)
```
