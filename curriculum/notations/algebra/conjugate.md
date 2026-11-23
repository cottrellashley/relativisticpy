# The complex conjugate

The conjugate of a complex quantity, written with a bar, with an overline and with a star.
A bar over a defined complex quantity is its conjugate (`design/notation.md`, decision 8),
and so is a star raised on one (decision 6). Churchill and Brown (2014) is *Complex
Variables and Applications*, 9th edition; Ahlfors (1979) is *Complex Analysis*, 3rd
edition; Griffiths (2018) is *Introduction to Quantum Mechanics*, 3rd edition; ISO
80000-2:2019 is the standard for mathematical signs and symbols; Lamport (1994) is *LaTeX: A
Document Preparation System*, 2nd edition; the Unicode Standard is version 15.0.

The quantity is defined first, as z, with x and y real.

## Setup

```latex
x, y \in \mathbb{R}
z := «value»
```

```text
Reals := [x, y]
z := «value»
```

```unicode
x, y ∈ ℝ
z := «value»
```

## Names

```python
x, y = symbols("x y", real=True)
```

## Operands

| operand | written in | `value` | separates |
|---|---|---|---|
| `3+4i` | latex, text, unicode | `3 + 4i` | a value |
| | python | `3 + 4 * I` | |
| `imaginary` | latex, text, unicode | `2i` | |
| | python | `2 * I` | |
| `real` | latex, text, unicode | `5` | a real number, which is its own conjugate |
| | python | `5` | |
| `x+iy` | latex, unicode | `x + iy` | letters declared real, beside the unit |
| | text | `x + i*y` | |
| | python | `x + I * y` | |
| `quotient` | latex | `\frac{1}{1 + i}` | |
| | text, unicode | `1/(1 + i)` | |
| | python | `1 / (1 + I)` | |
| `exponential` | latex | `e^{ix}` | the conjugate of an exponential of a real letter |
| | text | `exp(i*x)` | |
| | unicode | `e^(ix)` | |
| | python | `exp(I * x)` | |

## Oracle

```python
expected = conjugate(value)
```

## Second route

The conjugate by its real and imaginary parts.

```python
assert simplify(expected - (re(value) - I * im(value))) == 0
```

## Forms

### A bar over the name

**Form** · `bar` · Ahlfors (1979), §1.4, Conjugation, Absolute Value: `\bar{z}`; design/notation.md, decision 8

```latex
\bar{z}
```

### A bar over the name, without braces

**Form** · `bar-one-token` · Ahlfors (1979), §1.4, with the accent's argument one character: `\bar z`

```latex
\bar z
```

### An overline over the name

**Form** · `overline` · Churchill and Brown (2014), §6: `\overline{z}`; ISO 80000-2:2019

```latex
\overline{z}
```

### A star on the name

**Form** · `star` · Griffiths (2018), §1.4: `\Psi^*`; ISO 80000-2:2019: `z^*`; design/notation.md, decision 6

```latex
z^*
```

### A star on the name, in braces

**Form** · `star-braces` · Griffiths (2018), §1.4, with the star in braces

```latex
z^{*}
```

### A star by its command

**Form** · `ast` · Lamport (1994), §3.3.2, Binary Operation Symbols: `\ast`, the star as a command, `z^\ast`

```latex
z^\ast
```

### An overline over the quantity

**Form** · `overline-expression` · Churchill and Brown (2014), §6: `\overline{z_1 + z_2}`, an overline over an expression

```latex
\overline{«value»}
```

### A star on the quantity in brackets

**Form** · `star-expression` · Griffiths (2018), §1.4, of an expression in brackets

```latex
(«value»)^*
```

### A call of conjugate

**Form** · `conjugate-call` · docs/language.md, Functions you can call: `conjugate`

```text
conjugate(z)
```

### A call of conj

**Form** · `conj-call` · docs/reference.md, Functions, Elementary functions: `conj`; NumPy's and MATLAB's name

```text
conj(z)
```

### A call of conjugate on the quantity

**Form** · `conjugate-call-expression` · docs/language.md, Functions you can call: `conjugate`

```text
conjugate(«value»)
```

### A combining macron

**Form** · `unicode-macron` · the Unicode Standard, U+0304 COMBINING MACRON: `z̄`

```unicode
z̄
```

**Refused** · `ParseError` · docs/reference.md, Unicode: a combining accent is an error that names it

### A star after the name

**Form** · `unicode-star` · as typed: `z*`, the star of a conjugate on the line

```unicode
z*
```
