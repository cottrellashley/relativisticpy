# Ordinary derivatives

The derivative of a function of one variable, written on the expression in place,
\(\frac{d}{dx} x^3\), and on the name of a function the program defines, \(f'(x)\),
\(\frac{df}{dx}\) and \(\dot{x}\). Stewart (2015) is *Calculus: Early Transcendentals*, 8th
edition; Abramowitz and Stegun (1964) is the *Handbook of Mathematical Functions*; DLMF is
the NIST *Digital Library of Mathematical Functions*; ISO 80000-2:2019 is the standard for
mathematical signs and symbols; Zorich (2015) is *Mathematical Analysis I*, 2nd edition;
Feynman (1963) is *The Feynman Lectures on Physics*, volume I; Landau and Lifshitz (1976)
is *Mechanics*, 3rd edition; the physics package is the LaTeX package of that name (de la
Barrera, 2012); AsciiMath is the plain text notation of asciimath.org.

`«f»` is the expression, `«var»` its variable, and `«name»` the name a form gives the
function before it differentiates it.

## Names

```python
x, t, theta, a = symbols("x t theta a")
x_0, rate = symbols("x_0 rate")
```

## Operands

| operand | written in | `f` | `var` | `name` | separates |
|---|---|---|---|---|---|
| `x^3` | latex, text | `x^3` | `x` | `f` | a power |
| | unicode | `x³` | `x` | `f` | |
| | python | `x**3` | `x` | | |
| `product` | latex | `x^2 \sin x` | `x` | `y` | an operator that takes the whole product from one that takes its first factor |
| | text | `x^2*sin(x)` | `x` | `y` | |
| | unicode | `x² sin x` | `x` | `y` | |
| | python | `x**2*sin(x)` | `x` | | |
| `quotient` | latex | `\frac{x + 1}{x - 1}` | `x` | `g` | a quotient, taken whole |
| | text | `(x + 1)/(x - 1)` | `x` | `g` | |
| | unicode | `(x + 1)/(x − 1)` | `x` | `g` | |
| | python | `(x + 1)/(x - 1)` | `x` | | |
| `composite` | latex | `e^{-x^2}` | `x` | `\psi` | the chain rule, through a power in an exponent; a function with a Greek name |
| | text | `exp(-x^2)` | `x` | `psi` | |
| | unicode | `e^(−x²)` | `x` | `ψ` | |
| | python | `exp(-x**2)` | `x` | | |
| `sum` | latex, text | `x^3 - 2x + 1` | `x` | `f` | a sum, which needs brackets after an operator |
| | unicode | `x³ − 2x + 1` | `x` | `f` | |
| | python | `x**3 - 2*x + 1` | `x` | | |
| `negative-power` | latex | `x^{-2}` | `x` | `f_1` | a negative number in a script; a function whose name has a subscript |
| | text | `x^(-2)` | `x` | `f_1` | |
| | unicode | `x⁻²` | `x` | `f₁` | |
| | python | `x**-2` | `x` | | |
| `in-t` | latex | `t^2 e^{-t}` | `t` | `x` | a variable that is not x, and a function that is: Newton's dot is written of it |
| | text | `t^2*exp(-t)` | `t` | `x` | |
| | unicode | `t² e^(−t)` | `t` | `x` | |
| | python | `t**2*exp(-t)` | `t` | | |
| `in-theta` | latex | `\sin^2\theta` | `\theta` | `r` | a variable written as a Greek letter, after the d |
| | text | `sin(theta)^2` | `theta` | `r` | |
| | unicode | `sin²θ` | `θ` | `r` | |
| | python | `sin(theta)**2` | `theta` | | |
| `subscript` | latex | `a x_0^2` | `x_0` | `f` | a variable with a subscript, beside a parameter |
| | text | `a*x_0^2` | `x_0` | `f` | |
| | unicode | `a x₀²` | `x₀` | `f` | |
| | python | `a*x_0**2` | `x_0` | | |
| `long-name` | latex | `\mathrm{rate}\, x^2` | `x` | `f` | a name of several letters, which is one name |
| | text | `rate*x^2` | `x` | `f` | |
| | python | `rate*x**2` | `x` | | |

## Oracle

```python
expected = diff(f, var)
```

## Second route

The limit of the difference quotient.

```python
h = symbols("h")
assert simplify(limit((f.subs(var, var + h) - f) / h, h, 0) - expected) == 0
```

## Forms

### The operator before its operand

**Form** · `operator` · Abramowitz and Stegun (1964), §4.3, differentiation formulas: `\frac{d}{dz} \sin z = \cos z`

The operand is the term that follows the operator: a product or a quotient is taken whole.

```latex
\frac{d}{d«var»} «f»
```

**Not of** · `sum` · after the operator, a sum is written in brackets

### The operator with an upright d

**Form** · `operator-upright` · DLMF, §4.7 and §4.20: `\frac{\mathrm{d}}{\mathrm{d}z} \ln z = \frac{1}{z}`; ISO 80000-2:2019 sets d upright

```latex
\frac{\mathrm{d}}{\mathrm{d}«var»} «f»
```

**Not of** · `sum` · after the operator, a sum is written in brackets

### The operator as a display fraction

**Form** · `operator-dfrac` · the amsmath package's `\dfrac`, as a text sets the operator in a line of prose

```latex
\dfrac{d}{d«var»} «f»
```

**Not of** · `sum` · after the operator, a sum is written in brackets

### The operator before round brackets

**Form** · `operator-round` · Stewart (2015), §3.1: `\frac{d}{dx}(x^n) = nx^{n-1}`

```latex
\frac{d}{d«var»}(«f»)
```

### The operator before square brackets

**Form** · `operator-square` · Stewart (2015), §3.2, the product rule: `\frac{d}{dx}[f(x)g(x)]`

```latex
\frac{d}{d«var»}[«f»]
```

### The operator before brackets that grow

**Form** · `operator-left-right` · Stewart (2015), §3.1, set with LaTeX's `\left(` and `\right)`, which size the brackets to what they hold

```latex
\frac{d}{d«var»}\left(«f»\right)
```

### The operator before brackets of a given size

**Form** · `operator-bigl` · Stewart (2015), §3.1, set with the amsmath package's `\bigl(` and `\bigr)`

```latex
\frac{d}{d«var»}\bigl(«f»\bigr)
```

### The operand in the numerator

**Form** · `numerator` · Feynman (1963), §15-1: `F = d(mv)/dt`, set as a fraction

```latex
\frac{d(«f»)}{d«var»}
```

### A prime after brackets

**Form** · `prime-of-brackets` · Zorich (2015), §5.2, the table of derivatives: `(\sin x)' = \cos x`

The variable is not written: it is the one the expression has.

```latex
(«f»)'
```

**Not of** · `subscript`, `long-name` · with a parameter beside it, the variable must be named

### The physics package's operator

**Form** · `physics-dv-operator` · the physics package, Derivatives: `\dv{x}` is `\frac{d}{dx}`

```latex
\dv{«var»} «f»
```

**Not of** · `sum` · after the operator, a sum is written in brackets

### The physics package's fraction

**Form** · `physics-dv` · the physics package, Derivatives: `\dv{f}{x}` is `\frac{df}{dx}`

```latex
\dv{«f»}{«var»}
```

**Not of** · `sum` · a sum beside the d of a numerator is written in brackets

### Leibniz's fraction, of a function's name

**Form** · `leibniz-fraction` · Stewart (2015), §2.8, Other Notations: `\frac{dy}{dx}`, `\frac{df}{dx}`

```latex
«name»(«var») := «f»
\frac{d«name»}{d«var»}
```

### Leibniz's fraction with an upright d

**Form** · `leibniz-fraction-upright` · ISO 80000-2:2019: `\frac{\mathrm{d}f}{\mathrm{d}x}`

```latex
«name»(«var») := «f»
\frac{\mathrm{d}«name»}{\mathrm{d}«var»}
```

### Leibniz's fraction with a solidus

**Form** · `leibniz-solidus` · Stewart (2015), §2.8: `dy/dx`, as the fraction is set in a line of prose

```latex
«name»(«var») := «f»
d«name»/d«var»
```

### The operator before the function at its variable

**Form** · `operator-of-name` · Stewart (2015), §2.8, Other Notations: `\frac{d}{dx} f(x)`

```latex
«name»(«var») := «f»
\frac{d}{d«var»} «name»(«var»)
```

### A prime on the name, at its variable

**Form** · `prime` · Stewart (2015), §2.8: `f'(x)`

```latex
«name»(«var») := «f»
«name»'(«var»)
```

### A prime on the name alone

**Form** · `prime-alone` · Stewart (2015), §2.8, Other Notations: `y'`; docs/language.md: a function named without its argument is the function at its parameters

```latex
«name»(«var») := «f»
«name»'
```

### The operator D

**Form** · `D` · Stewart (2015), §2.8, Other Notations: `Df(x)`

```latex
«name»(«var») := «f»
D«name»(«var»)
```

### The operator D with its variable

**Form** · `D-subscript` · Stewart (2015), §2.8, Other Notations: `D_x f(x)`

```latex
«name»(«var») := «f»
D_«var» «name»(«var»)
```

**Not of** · `subscript` · a variable with a subscript of its own is not written as a subscript

### Newton's dot

**Form** · `dot` · Landau and Lifshitz (1976), §1: `\dot{q}`, the derivative with respect to time

```latex
«name»(«var») := «f»
\dot{«name»}
```

**Only of** · `in-t` · a dot is a derivative with respect to time

### Newton's dot, at the time

**Form** · `dot-at-time` · Landau and Lifshitz (1976), §1: `\dot{q}`, written of a function at the time, `\dot{x}(t)`

```latex
«name»(«var») := «f»
\dot{«name»}(«var»)
```

**Only of** · `in-t` · a dot is a derivative with respect to time

### The physics package's fraction, of a name

**Form** · `physics-dv-of-name` · the physics package, Derivatives: `\dv{f}{x}`

```latex
«name»(«var») := «f»
\dv{«name»}{«var»}
```

### A call of diff

**Form** · `diff` · docs/language.md, Calculus: `diff(sin(x), x)`; SymPy's name

```text
diff(«f», «var»)
```

### The language's plain fraction

**Form** · `braces` · docs/language.md, Calculus: `d{x^3}/d{x}`

```text
d{«f»}/d{«var»}
```

### The operator in plain text

**Form** · `slash-operator` · AsciiMath: `d/dx f(x)`; as a derivative is typed where no fraction can be set

```text
d/d«var» («f»)
```

### The operand in the numerator, in plain text

**Form** · `slash-numerator` · Feynman (1963), §15-1: `d(mv)/dt`, typed in a line of plain text

```text
d(«f»)/d«var»
```

### A prime on the name, in plain text

**Form** · `text-prime` · docs/language.md, Calculus: Lagrange's primes differentiate a function you have defined, as in `q''(t)`

```text
«name»(«var») := «f»
«name»'(«var»)
```

### A call of diff on the name

**Form** · `diff-of-name` · docs/language.md, Calculus: `diff`; of a function at its variable, as in `dsolve(f'(x) = f(x), f(x))`

```text
«name»(«var») := «f»
diff(«name»(«var»), «var»)
```

### Leibniz's fraction in plain text

**Form** · `text-leibniz` · Stewart (2015), §2.8: `dy/dx`, typed in a line of plain text; AsciiMath: `(dy)/(dx)`

```text
«name»(«var») := «f»
d«name»/d«var»
```

### The operator in Unicode

**Form** · `unicode-operator` · as typed with Unicode's superscripts, its Greek letters and its minus sign, U+2212

```unicode
d/d«var» («f»)
```

### A prime on the name, in Unicode

**Form** · `unicode-prime` · as typed with Unicode's superscripts, its Greek letters and its minus sign, U+2212

```unicode
«name»(«var») := «f»
«name»'(«var»)
```
