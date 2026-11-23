# Indefinite integrals

An antiderivative of a function of one variable, \(\int f(x)\,dx\), with its differential
written last, first and in a numerator. Stewart (2015) is *Calculus: Early Transcendentals*,
8th edition; DLMF is the NIST *Digital Library of Mathematical Functions*; ISO 80000-2:2019
is the standard for mathematical signs and symbols; Sakurai (1994) is *Modern Quantum
Mechanics*, revised edition; the physics package is the LaTeX package of that name (de la
Barrera, 2012); AsciiMath is the plain text notation of asciimath.org; the Unicode Standard
is version 15.0.

`«f»` is the integrand, written in brackets where it is a sum; `«var»` the variable; and
`«below»` the denominator of an integrand that is a reciprocal.

An antiderivative is one up to a constant (Stewart (2015), §5.4): an answer that differs
from the oracle's by a constant, as \(-\frac{1}{2}\cos^2 x\) does from \(\frac{1}{2}\sin^2 x\),
is as right. A parameter that nothing declares is taken as books take it, a scale that is
positive (`design/notation.md`, decision 1).

## Names

```python
x, t = symbols("x t")
a = symbols("a", positive=True)
```

## Operands

| operand | written in | `f` | `var` | `below` | separates |
|---|---|---|---|---|---|
| `power` | latex, text | `x^2` | `x` | | a power |
| | unicode | `x²` | `x` | | |
| | python | `x**2` | `x` | | |
| `reciprocal` | latex | `\frac{1}{x}` | `x` | `x` | a logarithm, and a differential in the numerator |
| | text, unicode | `1/x` | `x` | | |
| | python | `1/x` | `x` | | |
| `cosine` | latex | `\cos x` | `x` | | a function without brackets, ended by the differential |
| | text | `cos(x)` | `x` | | |
| | unicode | `cos x` | `x` | | |
| | python | `cos(x)` | `x` | | |
| `exponential` | latex | `e^{2x}` | `x` | | e as the exponential, of a multiple of the variable |
| | text | `exp(2*x)` | `x` | | |
| | unicode | `e^(2x)` | `x` | | |
| | python | `exp(2*x)` | `x` | | |
| `arctangent` | latex | `\frac{1}{1 + x^2}` | `x` | `1 + x^2` | a reciprocal of a sum: the arctangent |
| | text | `1/(1 + x^2)` | `x` | | |
| | unicode | `1/(1 + x²)` | `x` | | |
| | python | `1/(1 + x**2)` | `x` | | |
| `by-parts` | latex | `x e^x` | `x` | | a product, taken whole up to the differential |
| | text | `x*exp(x)` | `x` | | |
| | unicode | `x e^x` | `x` | | |
| | python | `x*exp(x)` | `x` | | |
| `scale` | latex | `e^{-a t}` | `t` | | a parameter nothing declares, which books take positive, and a variable that is not x |
| | text | `exp(-a*t)` | `t` | | |
| | unicode | `e^(−a t)` | `t` | | |
| | python | `exp(-a*t)` | `t` | | |
| `two-functions` | latex | `\sin x \cos x` | `x` | | two functions side by side, each taking its own argument |
| | text | `sin(x)*cos(x)` | `x` | | |
| | unicode | `sin x cos x` | `x` | | |
| | python | `sin(x)*cos(x)` | `x` | | |
| `sum` | latex | `(3x^2 + 2x + 1)` | `x` | | a sum, which an integral takes in brackets |
| | text | `(3*x^2 + 2*x + 1)` | `x` | | |
| | unicode | `(3x² + 2x + 1)` | `x` | | |
| | python | `3*x**2 + 2*x + 1` | `x` | | |
| `root` | latex | `\sqrt{x}` | `x` | | a root, a power of one half |
| | text | `sqrt(x)` | `x` | | |
| | unicode | `√x` | `x` | | |
| | python | `sqrt(x)` | `x` | | |

## Oracle

```python
expected = integrate(f, var)
up_to_a_constant = var
```

## Second route

The derivative of the antiderivative is the integrand.

```python
assert simplify(diff(expected, var) - f) == 0
```

## Forms

### The differential last, after a thin space

**Form** · `thin-space` · Stewart (2015), §5.4, Indefinite Integrals: `\int f(x)\,dx = F(x)` means `F'(x) = f(x)`

```latex
\int «f» \, d«var»
```

### The differential last, with no space

**Form** · `no-space` · Stewart (2015), §5.4, typed without the thin space: `\int f(x) dx`

```latex
\int «f» d«var»
```

### An upright d

**Form** · `upright-d` · ISO 80000-2:2019 and DLMF, §1.4: `\int f(x)\,\mathrm{d}x`, the d set upright

```latex
\int «f» \, \mathrm{d}«var»
```

### The differential first

**Form** · `differential-first` · Sakurai (1994), §1.6: `\int d\xi'\,|\xi'\rangle\langle\xi'|`, the differential written straight after the integral sign

```latex
\int d«var» \, «f»
```

### The differential in the numerator

**Form** · `differential-in-numerator` · Stewart (2015), §7.5, Table of Integration Formulas: `\int \frac{dx}{x^2 + a^2}`

```latex
\int \frac{d«var»}{«below»}
```

### Of a function the program defines

**Form** · `function` · Stewart (2015), §5.4: `\int f(x)\,dx`, of a function defined first

```latex
f(«var») := «f»
\int f(«var») \, d«var»
```

### The physics package's differential

**Form** · `physics-dd` · the physics package, Derivatives: `\dd{x}`

```latex
\int «f» \dd{«var»}
```

**Refused** · `ParseError` · docs/reference.md, Not supported: `\dd` is an error that says to write `d x`

### A call of integrate

**Form** · `integrate` · docs/language.md, Calculus: `integrate`; SymPy's `integrate(x**2, x)`

```text
integrate(«f», «var»)
```

### The integral in plain text

**Form** · `asciimath` · AsciiMath: `int f(x)dx`

```text
int «f» d«var»
```

### The integral sign in Unicode

**Form** · `unicode-sign` · docs/latex.md, Unicode: ∫, U+222B INTEGRAL, is `\int`

```unicode
∫ «f» d«var»
```

### Unicode's differential d

**Form** · `unicode-differential` · the Unicode Standard, U+2146 DOUBLE-STRUCK ITALIC SMALL D, the differential d: `∫ f(x) ⅆx`

```unicode
∫ «f» ⅆ«var»
```
