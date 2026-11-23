# Factorials

The factorial of a number, \(n! = 1 \cdot 2 \cdots n\), with an exclamation mark, as a value of
the gamma function and as a function. Graham, Knuth and Patashnik (1994) is *Concrete
Mathematics*, 2nd edition; DLMF is the NIST *Digital Library of Mathematical Functions*;
Lamport (1994) is *LaTeX: A Document Preparation System*, 2nd edition; the Unicode Standard
is version 15.0.

`«(argument)»` is the number as it is written before `!`: in brackets where it is more than
one letter or digit; `«next»` is the number one more, as the gamma function takes it.

## Names

```python
n = symbols("n")
```

## Operands

| operand | written in | `argument` | `(argument)` | `next` | separates |
|---|---|---|---|---|---|
| `five` | latex, text, unicode | `5` | `5` | `6` | a value |
| | python | `5` | | | |
| `zero` | latex, text, unicode | `0` | `0` | `1` | the empty product, which is 1 |
| | python | `0` | | | |
| `ten` | latex, text, unicode | `10` | `10` | `11` | a number of two digits, whose factorial is of both |
| | python | `10` | | | |
| `symbol` | latex, text, unicode | `n` | `n` | `n + 1` | |
| | python | `n` | | | |
| `sum` | latex, text, unicode | `n + 1` | `(n + 1)` | `n + 2` | a factorial of a whole sum, in brackets, from a sum with a factorial in it |
| | python | `n + 1` | | | |
| `double` | latex, unicode | `2n` | `(2n)` | `2n + 1` | the factorial of 2n from twice the factorial of n |
| | text | `2*n` | `(2*n)` | `2*n + 1` | |
| | python | `2*n` | | | |
| `half` | latex | `\frac{1}{2}` | `\left(\frac{1}{2}\right)` | `\frac{3}{2}` | a number that is not whole, whose factorial is the gamma function's |
| | text, unicode | `1/2` | `(1/2)` | `3/2` | |
| | python | `Rational(1, 2)` | | | |
| `negative` | latex, text | `-1` | `(-1)` | `0` | a negative whole number, which has no factorial: it must be refused |
| | unicode | `−1` | `(−1)` | `0` | |
| | python | `-1` | | | |

## Oracle

```python
value = S(argument)
if value.is_integer and value.is_negative:
    error = "RelativisticPyError"
else:
    expected = factorial(value)
```

## Second route

The factorial is the gamma function one place on, and a negative whole number is a pole of it.

```python
if value.is_integer and value.is_negative:
    assert gamma(value + 1) is zoo
else:
    assert simplify(gamma(value + 1) - expected) == 0
```

## Forms

### An exclamation mark

**Form** · `bang` · Graham, Knuth and Patashnik (1994), §4.4, Factorial factors: `n! = 1 \cdot 2 \cdot \ldots \cdot n`

```latex
«(argument)»!
```

### The gamma function

**Form** · `gamma` · DLMF, §5.4(i): `\Gamma(n + 1) = n!`; design/notation.md, decision 15: `\Gamma(x)` with one argument in brackets is Euler's gamma

```latex
\Gamma(«next»)
```

### The gamma function, with brackets that grow

**Form** · `gamma-left-right` · DLMF, §5.4(i), with the argument in brackets that grow: Lamport (1994), §3.3.4, Delimiters

```latex
\Gamma\left(«next»\right)
```

### An exclamation mark, in plain text

**Form** · `text-bang` · docs/language.md, Piecewise functions and other notation: `4!`

```text
«(argument)»!
```

### A call of factorial

**Form** · `factorial-call` · docs/language.md, Functions you can call: `factorial`; SymPy's name

```text
factorial(«argument»)
```

### A call of gamma

**Form** · `gamma-call` · docs/reference.md, Functions, Elementary functions: `gamma`; docs/language.md: Euler's gamma function is `gamma(x)`, spelled out

```text
gamma(«next»)
```

### An exclamation mark, in Unicode

**Form** · `unicode-bang` · as typed, with Unicode's minus sign, U+2212

```unicode
«(argument)»!
```

### The gamma function, in Unicode

**Form** · `unicode-gamma` · the Unicode Standard, U+0393 GREEK CAPITAL LETTER GAMMA: `Γ(n + 1)`

```unicode
Γ(«next»)
```
