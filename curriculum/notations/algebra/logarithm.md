# The natural logarithm

The logarithm to the base e, written ln and written log. Stewart (2015) is *Calculus:
Early Transcendentals*, 8th edition; Rudin (1976) is *Principles of Mathematical Analysis*,
3rd edition; Lamport (1994) is *LaTeX: A Document Preparation System*, 2nd edition.

A bare `\log` is the natural logarithm (`design/notation.md`, decision 29).

## Names

```python
x, y, t = symbols("x y t")
```

## Operands

| operand | written in | `argument` | separates |
|---|---|---|---|
| `x` | latex, text, unicode | `x` | |
| | python | `x` | |
| `e` | latex, text, unicode | `e` | a value |
| | python | `E` | |
| `power-of-e` | latex, text | `e^2` | the logarithm of a power from the power of a logarithm |
| | unicode | `e²` | |
| | python | `E**2` | |
| `one` | latex, text, unicode | `1` | |
| | python | `1` | |
| `product` | latex, unicode | `2x` | an argument that is a product, taken whole |
| | text | `2*x` | |
| | python | `2*x` | |
| `two-letters` | latex | `xy` | an argument of two letters, which is their product |
| | text | `x*y` | |
| | unicode | `x y` | |
| | python | `x*y` | |
| `sum` | latex, text, unicode | `x + 1` | a sum, which is an argument only in brackets |
| | python | `x + 1` | |
| `reciprocal` | latex | `\frac{1}{x}` | a fraction straight after the function, which is its argument |
| | text, unicode | `1/x` | |
| | python | `1/x` | |
| `root` | latex | `\sqrt{x}` | |
| | text | `sqrt(x)` | |
| | unicode | `√x` | |
| | python | `sqrt(x)` | |
| `power-of-x` | latex, text | `x^3` | the logarithm of a power from the cube of a logarithm |
| | unicode | `x³` | |
| | python | `x**3` | |

## Oracle

```python
expected = log(S(argument))
```

## Second route

The logarithm as the integral of 1/t from 1, at a point.

```python
at = {x: Rational(17, 10), y: Rational(13, 10)}
by_integral = Integral(1 / t, (t, 1, S(argument).subs(at))).evalf(30)
assert abs(N(expected.subs(at), 30) - by_integral) < 1e-20
```

## Forms

### ln without brackets

**Form** · `ln-bare` · Stewart (2015), §1.5: `\ln x`

```latex
\ln «argument»
```

**Not of** · `sum`, `reciprocal` · a sum is an argument only in brackets, and a quotient after ln is written built up

### ln of a fraction, built up

**Form** · `ln-fraction` · Stewart (2015), §1.5, the laws of logarithms: `\ln \frac{x}{y}`

```latex
\ln «argument»
```

**Only of** · `reciprocal` · the argument is a built-up fraction

### ln with round brackets

**Form** · `ln-round` · Stewart (2015), §1.5: `\ln(e^x) = x`

```latex
\ln(«argument»)
```

### ln with brackets that grow

**Form** · `ln-left-right` · Lamport (1994), §3.3.4: `\left(` and `\right)`

```latex
\ln\left(«argument»\right)
```

### log without brackets

**Form** · `log-bare` · Rudin (1976), chapter 8: `\log x` is the natural logarithm; design/notation.md, decision 29

```latex
\log «argument»
```

**Not of** · `sum`, `reciprocal` · a sum is an argument only in brackets, and a quotient after log is written built up

### log with round brackets

**Form** · `log-round` · Rudin (1976), chapter 8; design/notation.md, decision 29

```latex
\log(«argument»)
```

### log to the base e

**Form** · `log-base-e` · Stewart (2015), §1.5: `\log_e x = \ln x`

```latex
\log_e(«argument»)
```

### A call of ln

**Form** · `ln-call` · docs/language.md, Functions you can call: `ln`

```text
ln(«argument»)
```

### A call of log

**Form** · `log-call` · docs/language.md, Functions you can call: `log`; SymPy's and Python's name for the natural logarithm

```text
log(«argument»)
```

### ln without brackets, in plain text

**Form** · `text-bare` · docs/language.md, Functions you can call: a function without parentheses applies to what follows it

```text
ln «argument»
```

**Only of** · `x`, `e`, `power-of-e`, `one` · a product, a sum or a quotient is an argument only in brackets

### ln with brackets, in Unicode

**Form** · `unicode-round` · as typed, with Unicode's superscripts and its radical sign

```unicode
ln(«argument»)
```

### ln without brackets, in Unicode

**Form** · `unicode-bare` · as typed: `ln x`, `ln 2x`

```unicode
ln «argument»
```

**Not of** · `sum`, `reciprocal` · a sum or a quotient is an argument only in brackets
