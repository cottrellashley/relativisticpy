# Partial derivatives

The derivative of a function of several variables with respect to one of them, the others
held fixed, \(\frac{\partial f}{\partial x}\), written on the expression in place and on the
name of a function the program defines, as \(f_x\), \(\partial_x f\) and \(D_1 f\). Stewart
(2015) is *Calculus: Early Transcendentals*, 8th edition; ISO 80000-2:2019 is the standard
for mathematical signs and symbols; the physics package is the LaTeX package of that name
(de la Barrera, 2012); AsciiMath is the plain text notation of asciimath.org; the Unicode
Standard is version 15.0.

`«f»` is the expression, `«var»` the variable it is differentiated by, `«name»` the name a
form gives the function, `«params»` its parameters, `«number»` the place of the variable
among them, counted from 1, and `«dependent»` the name of its value, as z is in
\(z = f(x, y)\). `«lowered»` is the variable as a Unicode subscript character, where
Unicode has one.

## Names

```python
x, y, z, r, theta, x_1, x_2 = symbols("x y z r theta x_1 x_2")
```

## Operands

| operand | written in | `f` | `var` | `name` | `params` | `number` | `dependent` | `lowered` | separates |
|---|---|---|---|---|---|---|---|---|---|
| `x^2y^3` | latex | `x^2 y^3` | `x` | `f` | `x, y` | `1` | `z` | | the first of two variables: by y, it is 3x²y² |
| | text | `x^2*y^3` | `x` | `f` | `x, y` | `1` | `z` | | |
| | unicode | `x²y³` | `x` | `f` | `x, y` | | | `ₓ` | |
| | python | `x**2*y**3` | `x` | | | | | | |
| `in-y` | latex | `x^2 y^3` | `y` | `f` | `x, y` | `2` | `z` | | the second of two variables: by x, it is 2xy³ |
| | text | `x^2*y^3` | `y` | `f` | `x, y` | `2` | `z` | | |
| | unicode | `x²y³` | `y` | `f` | `x, y` | | | | |
| | python | `x**2*y**3` | `y` | | | | | | |
| `exponential` | latex | `x e^{xy}` | `x` | `g` | `x, y` | `1` | `z` | | the variable in a factor and in an exponent: the product rule and the chain rule |
| | text | `x*exp(x*y)` | `x` | `g` | `x, y` | `1` | `z` | | |
| | unicode | `x e^(xy)` | `x` | `g` | `x, y` | | | `ₓ` | |
| | python | `x*exp(x*y)` | `x` | | | | | | |
| `quotient` | latex | `\frac{x}{x + y}` | `x` | `h` | `x, y` | `1` | `z` | | a quotient, taken whole, with the variable above and below |
| | text | `x/(x + y)` | `x` | `h` | `x, y` | `1` | `z` | | |
| | unicode | `x/(x + y)` | `x` | `h` | `x, y` | | | `ₓ` | |
| | python | `x/(x + y)` | `x` | | | | | | |
| `sine` | latex | `\sin(xy)` | `y` | `u` | `x, y` | `2` | `z` | | a function of a product, by its second variable: by x, it is y cos(xy) |
| | text | `sin(x*y)` | `y` | `u` | `x, y` | `2` | `z` | | |
| | unicode | `sin(xy)` | `y` | `u` | `x, y` | | | | |
| | python | `sin(x*y)` | `y` | | | | | | |
| `three-variables` | latex | `x y z^2` | `z` | `f` | `x, y, z` | `3` | `w` | | the third of three variables |
| | text | `x*y*z^2` | `z` | `f` | `x, y, z` | `3` | `w` | | |
| | unicode | `xyz²` | `z` | `f` | `x, y, z` | | | | |
| | python | `x*y*z**2` | `z` | | | | | | |
| `polar` | latex | `r^2 \sin\theta` | `\theta` | `f` | `r, \theta` | `2` | `z` | | a variable written as a Greek letter |
| | text | `r^2*sin(theta)` | `theta` | `f` | `r, theta` | `2` | `z` | | |
| | unicode | `r² sin θ` | `θ` | `f` | `r, θ` | | | | |
| | python | `r**2*sin(theta)` | `theta` | | | | | | |
| `subscripted` | latex | `x_1^2 x_2` | `x_1` | `f` | `x_1, x_2` | `1` | `z` | | variables with subscripts of their own, side by side |
| | text | `x_1^2*x_2` | `x_1` | `f` | `x_1, x_2` | `1` | `z` | | |
| | unicode | `x₁²x₂` | `x₁` | `f` | `x₁, x₂` | | | | |
| | python | `x_1**2*x_2` | `x_1` | | | | | | |

## Oracle

```python
expected = diff(f, var)
```

## Second route

The limit of the difference quotient in the variable, the others held fixed.

```python
h = symbols("h")
assert simplify(limit((f.subs(var, var + h) - f) / h, h, 0) - expected) == 0
```

## Forms

### The operator before the expression

**Form** · `operator` · docs/language.md, Calculus: `\frac{\partial^2}{\partial x \partial y} x^2 y^3`, the operator on the expression in place

```latex
\frac{\partial}{\partial «var»} «f»
```

### Leibniz's fraction, of a function's name

**Form** · `fraction` · Stewart (2015), §14.3, Notations for Partial Derivatives: `\frac{\partial f}{\partial x}`

```latex
«name»(«params») := «f»
\frac{\partial «name»}{\partial «var»}
```

### The operator before the function at its variables

**Form** · `operator-of-name` · Stewart (2015), §14.3, Notations for Partial Derivatives: `\frac{\partial}{\partial x} f(x, y)`

```latex
«name»(«params») := «f»
\frac{\partial}{\partial «var»} «name»(«params»)
```

### Leibniz's fraction, of the function's value

**Form** · `dependent` · Stewart (2015), §14.3, Notations for Partial Derivatives: if \(z = f(x, y)\), `\frac{\partial z}{\partial x}`

```latex
«dependent» = «f»
\frac{\partial «dependent»}{\partial «var»}
```

### The variable as a subscript

**Form** · `subscript` · Stewart (2015), §14.3, Notations for Partial Derivatives: `f_x`; design/notation.md, decision 5.6

```latex
«name»(«params») := «f»
«name»_«var»
```

**Not of** · `subscripted` · a variable with a subscript of its own is not written as a subscript

### The variable as a subscript, at the point

**Form** · `subscript-at-point` · Stewart (2015), §14.3, Notations for Partial Derivatives: `f_x(x, y)`

```latex
«name»(«params») := «f»
«name»_«var»(«params»)
```

**Not of** · `subscripted` · a variable with a subscript of its own is not written as a subscript

### The place of the variable as a subscript

**Form** · `subscript-number` · Stewart (2015), §14.3, Notations for Partial Derivatives: `f_1`, the derivative by the first variable

```latex
«name»(«params») := «f»
«name»_«number»
```

### The operator D, with the place of the variable

**Form** · `D-number` · Stewart (2015), §14.3, Notations for Partial Derivatives: `D_1 f`

```latex
«name»(«params») := «f»
D_«number» «name»
```

### The operator D, with the variable

**Form** · `D-subscript` · Stewart (2015), §14.3, Notations for Partial Derivatives: `D_x f`

```latex
«name»(«params») := «f»
D_«var» «name»
```

**Not of** · `subscripted` · a variable with a subscript of its own is not written as a subscript

### The partial sign with the variable as a subscript

**Form** · `partial-subscript` · ISO 80000-2:2019, partial derivative: `\partial_x f`

```latex
«name»(«params») := «f»
\partial_{«var»} «name»
```

### Leibniz's fraction with a solidus

**Form** · `solidus` · ISO 80000-2:2019, partial derivative: `\partial f/\partial x`, as the fraction is set in a line of text

```latex
«name»(«params») := «f»
\partial «name»/\partial «var»
```

### The physics package's fraction

**Form** · `physics-pdv` · the physics package, Derivatives: `\pdv{f}{x}` is `\frac{\partial f}{\partial x}`; docs/language.md, Calculus: `\pdv{x^2 y}{y}`

```latex
\pdv{«f»}{«var»}
```

### The physics package's fraction, of a name

**Form** · `physics-pdv-of-name` · the physics package, Derivatives: `\pdv{f}{x}`

```latex
«name»(«params») := «f»
\pdv{«name»}{«var»}
```

### A call of diff

**Form** · `diff` · docs/language.md, Calculus: `diff(sin(x), x)`; SymPy's name, which differentiates partially by the variable it is given

```text
diff(«f», «var»)
```

### A call of diff on the name

**Form** · `diff-of-name` · SymPy's `diff`, of a function at its variables

```text
«name»(«params») := «f»
diff(«name»(«params»), «var»)
```

### The variable as a subscript, in plain text

**Form** · `text-subscript` · Stewart (2015), §14.3: `f_x`, typed in a line of plain text; design/notation.md, decision 5.6

```text
«name»(«params») := «f»
«name»_«var»
```

**Not of** · `subscripted` · a variable with a subscript of its own is not written as a subscript

### Leibniz's fraction in AsciiMath

**Form** · `asciimath` · AsciiMath: `del` is ∂, and `(del f)/(del x)` is the fraction

```text
«name»(«params») := «f»
(del «name»)/(del «var»)
```

### Leibniz's fraction in Unicode

**Form** · `unicode-fraction` · the Unicode Standard, U+2202 PARTIAL DIFFERENTIAL: `∂f/∂x`, as typed

```unicode
«name»(«params») := «f»
∂«name»/∂«var»
```

### The operator in Unicode

**Form** · `unicode-operator` · the Unicode Standard, U+2202 PARTIAL DIFFERENTIAL: `∂/∂x`, as typed before the expression in brackets

```unicode
∂/∂«var» («f»)
```

### The partial sign with an underscore, in Unicode

**Form** · `unicode-partial-subscript` · ISO 80000-2:2019: `\partial_x f`, typed with U+2202

```unicode
«name»(«params») := «f»
∂_«var» «name»
```

### The partial sign with a subscript character

**Form** · `unicode-subscript` · the Unicode Standard, U+2093 LATIN SUBSCRIPT SMALL LETTER X: `∂ₓf`

```unicode
«name»(«params») := «f»
∂«lowered»«name»
```
