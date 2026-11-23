# Definite integrals

The integral of a function of one variable between two limits, \(\int_a^b f(x)\,dx\), with
its differential written last, first and in a numerator. Stewart (2015) is *Calculus: Early
Transcendentals*, 8th edition; Knuth (1984) is *The TeXbook*; DLMF is the NIST *Digital
Library of Mathematical Functions*; ISO 80000-2:2019 is the standard for mathematical signs
and symbols; Sakurai (1994) is *Modern Quantum Mechanics*, revised edition; Riley, Hobson
and Bence (2006) is *Mathematical Methods for Physics and Engineering*, 3rd edition; the
physics package is the LaTeX package of that name (de la Barrera, 2012); AsciiMath is the
plain text notation of asciimath.org; the Unicode Standard is version 15.0.

`«f»` is the integrand, written in brackets where it is a sum; `«var»` the variable;
`«lower»` and `«upper»` the limits; `«below»` the denominator of an integrand that is a
reciprocal; and `«lowered»` and `«raised»` the limits in Unicode's subscript and
superscript characters.

A parameter that nothing declares is taken as books take it, a scale that is positive
(`design/notation.md`, decision 1).

## Names

```python
x, t, y, a, b, x_0, x_1 = symbols("x t y a b x_0 x_1")
s = symbols("s", positive=True)
```

## Operands

| operand | written in | `f` | `var` | `lower` | `upper` | `below` | `lowered` | `raised` | separates |
|---|---|---|---|---|---|---|---|---|---|
| `x^2` | latex, text | `x^2` | `x` | `0` | `1` | | | | a value: one third |
| | unicode | `x²` | `x` | `0` | `1` | | `₀` | `¹` | |
| | python | `x**2` | `x` | `0` | `1` | | | | |
| `sine` | latex | `\sin x` | `x` | `0` | `\pi` | | | | a limit that is a command |
| | text | `sin(x)` | `x` | `0` | `pi` | | | | |
| | unicode | `sin x` | `x` | `0` | `π` | | | | |
| | python | `sin(x)` | `x` | `0` | `pi` | | | | |
| `product` | latex | `x e^x` | `x` | `0` | `1` | | | | an integrand that is a product, taken whole up to the differential |
| | text | `x*exp(x)` | `x` | `0` | `1` | | | | |
| | unicode | `x e^x` | `x` | `0` | `1` | | `₀` | `¹` | |
| | python | `x*exp(x)` | `x` | `0` | `1` | | | | |
| `two-letters` | latex | `xy` | `x` | `0` | `1` | | | | two letters before the differential: a product, and no part of it |
| | text | `x*y` | `x` | `0` | `1` | | | | |
| | python | `x*y` | `x` | `0` | `1` | | | | |
| `sum` | latex, text | `(x^2 + 1)` | `x` | `0` | `2` | | | | a sum, which an integral takes in brackets |
| | unicode | `(x² + 1)` | `x` | `0` | `2` | | `₀` | `²` | |
| | python | `x**2 + 1` | `x` | `0` | `2` | | | | |
| `reciprocal` | latex | `\frac{1}{1 + x^2}` | `x` | `0` | `1` | `1 + x^2` | | | a differential in the numerator of the integrand |
| | text | `1/(1 + x^2)` | `x` | `0` | `1` | | | | |
| | unicode | `1/(1 + x²)` | `x` | `0` | `1` | | `₀` | `¹` | |
| | python | `1/(1 + x**2)` | `x` | `0` | `1` | | | | |
| `in-t` | latex | `\frac{1}{t}` | `t` | `1` | `2` | `t` | | | a variable that is not x |
| | text | `1/t` | `t` | `1` | `2` | | | | |
| | unicode | `1/t` | `t` | `1` | `2` | | `₁` | `²` | |
| | python | `1/t` | `t` | `1` | `2` | | | | |
| `to-infinity` | latex | `e^{-x}` | `x` | `0` | `\infty` | | | | an infinite limit |
| | text | `exp(-x)` | `x` | `0` | `oo` | | | | |
| | unicode | `e^(−x)` | `x` | `0` | `∞` | | | | |
| | python | `exp(-x)` | `x` | `0` | `oo` | | | | |
| `gaussian` | latex | `e^{-x^2}` | `x` | `-\infty` | `\infty` | | | | a lower limit with a sign |
| | text | `exp(-x^2)` | `x` | `-oo` | `oo` | | | | |
| | unicode | `e^(−x²)` | `x` | `−∞` | `∞` | | | | |
| | python | `exp(-x**2)` | `x` | `-oo` | `oo` | | | | |
| `negative-limit` | latex, text | `x^2` | `x` | `-1` | `1` | | | | a negative number as a limit |
| | unicode | `x²` | `x` | `−1` | `1` | | | | |
| | python | `x**2` | `x` | `-1` | `1` | | | | |
| `scale` | latex | `e^{-st}` | `t` | `0` | `\infty` | | | | a parameter nothing declares, which books take positive (decision 1) |
| | text | `exp(-s*t)` | `t` | `0` | `oo` | | | | |
| | python | `exp(-s*t)` | `t` | `0` | `oo` | | | | |
| `letters-as-limits` | latex, text, unicode | `x` | `x` | `a` | `b` | | | | limits that are letters |
| | python | `x` | `x` | `a` | `b` | | | | |
| `subscripts-as-limits` | latex | `2x` | `x` | `x_0` | `x_1` | | | | limits that are names with subscripts |
| | text | `2*x` | `x` | `x_0` | `x_1` | | | | |
| | python | `2*x` | `x` | `x_0` | `x_1` | | | | |
| `oscillation` | latex | `\sin x` | `x` | `0` | `\infty` | | | | an integral whose value never settles, which has none: it must be refused |
| | text | `sin(x)` | `x` | `0` | `oo` | | | | |
| | unicode | `sin x` | `x` | `0` | `∞` | | | | |
| | python | `sin(x)` | `x` | `0` | `oo` | | | | |
| `pole-inside` | latex | `\frac{1}{x}` | `x` | `-1` | `1` | `x` | | | a pole between the limits, which leaves the integral no value: it must be refused |
| | text | `1/x` | `x` | `-1` | `1` | | | | |
| | unicode | `1/x` | `x` | `−1` | `1` | | | | |
| | python | `1/x` | `x` | `-1` | `1` | | | | |

## Oracle

An integral that SymPy finds undefined, or bounded only between two values, has no value.

```python
value = integrate(f, (var, lower, upper))
if value is nan or isinstance(value, AccumBounds):
    error = "RelativisticPyError"
else:
    expected = value
```

## Second route

Numerical quadrature, with a number for each parameter. Of an integral that has no value,
the integral up to t has no limit as t runs to the point where the integrand misbehaves:
a pole between the limits, or else the upper limit.

```python
if "error" in dir():
    t = symbols("t")
    poles = [p for p in singularities(f, var) if S(lower) < p < S(upper)]
    up_to_t = integrate(f, (var, lower, t))
    reached = limit(up_to_t, t, poles[0] if poles else S(upper), "-")
    assert isinstance(reached, AccumBounds) or reached.is_infinite
else:
    at = {y: Rational(13, 10), s: Rational(7, 5), a: Rational(2, 5), b: Rational(9, 5)}
    at.update({x_0: Rational(1, 2), x_1: Rational(3, 2)})
    limits = (var, S(lower).subs(at), S(upper).subs(at))
    by_quadrature = Integral(f.subs(at), limits).evalf(25)
    assert abs(N(expected.subs(at), 25) - by_quadrature) < 1e-15
```

## Forms

### The differential last, after a thin space

**Form** · `thin-space` · Stewart (2015), §5.2: `\int_a^b f(x)\,dx`; Knuth (1984), chapter 18: a thin space before `dx`

```latex
\int_{«lower»}^{«upper»} «f» \, d«var»
```

### The differential last, with no space

**Form** · `no-space` · Stewart (2015), §5.2, typed without the thin space: `\int_a^b f(x) dx`

```latex
\int_{«lower»}^{«upper»} «f» d«var»
```

### Limits without braces

**Form** · `one-token-limits` · Knuth (1984), chapter 16: a subscript or superscript without braces is one character or command, as in `\int_0^\infty`

```latex
\int_«lower»^«upper» «f» \, d«var»
```

**Not of** · `gaussian`, `negative-limit`, `subscripts-as-limits`, `pole-inside` · a limit of more than one token is written in braces

### An upright d

**Form** · `upright-d` · ISO 80000-2:2019 and DLMF, §1.4: `\int_a^b f(x)\,\mathrm{d}x`, the d set upright

```latex
\int_{«lower»}^{«upper»} «f» \, \mathrm{d}«var»
```

### The differential first

**Form** · `differential-first` · Sakurai (1994), §1.6: `\int d\xi'\,|\xi'\rangle\langle\xi'|`, the differential written straight after the integral sign

```latex
\int_{«lower»}^{«upper»} d«var» \, «f»
```

### The differential in the numerator

**Form** · `differential-in-numerator` · Stewart (2015), §7.5, Table of Integration Formulas: `\int \frac{dx}{x^2 + a^2}`

```latex
\int_{«lower»}^{«upper»} \frac{d«var»}{«below»}
```

### Limits set above and below

**Form** · `limits` · Knuth (1984), chapter 17: `\limits` sets the limits above and below the sign

```latex
\int\limits_{«lower»}^{«upper»} «f» \, d«var»
```

### Limits that name the variable

**Form** · `named-limits` · Riley, Hobson and Bence (2006), §6.1: limits written `x = a` and `x = b`

```latex
\int_{«var»=«lower»}^{«var»=«upper»} «f» \, d«var»
```

### The physics package's differential

**Form** · `physics-dd` · the physics package, Derivatives: `\dd{x}`

```latex
\int_{«lower»}^{«upper»} «f» \dd{«var»}
```

**Refused** · `ParseError` · docs/reference.md, Not supported: `\dd` is an error that says to write `d x`

### A call of integrate

**Form** · `integrate` · docs/language.md, Calculus: `integrate(x y, (x, 0, 1), (y, 0, 2))`; SymPy's name

```text
integrate(«f», («var», «lower», «upper»))
```

### The integral in plain text

**Form** · `asciimath` · AsciiMath: `int_0^1 f(x)dx`

```text
int_«lower»^«upper» «f» d«var»
```

**Not of** · `gaussian`, `negative-limit`, `subscripts-as-limits`, `pole-inside` · a limit of more than one token needs brackets, which AsciiMath writes round

### The integral sign in Unicode

**Form** · `unicode-sign` · docs/latex.md, Unicode: `∫_0^1 θ² dθ`

```unicode
∫_«lower»^«upper» «f» d«var»
```

**Not of** · `gaussian`, `negative-limit`, `pole-inside` · a limit of more than one character is written in braces

### The integral sign in Unicode, limits in braces

**Form** · `unicode-sign-braces` · docs/latex.md, Unicode, with LaTeX's braces round each limit

```unicode
∫_{«lower»}^{«upper»} «f» d«var»
```

### Limits as subscript and superscript characters

**Form** · `unicode-scripts` · the Unicode Standard, Superscripts and Subscripts, U+2070 to U+209F: `∫₀¹ x² dx`

```unicode
∫«lowered»«raised» «f» d«var»
```
