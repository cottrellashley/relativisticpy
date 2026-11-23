# Mixed partial derivatives

The second derivative of a function of several variables by two of them,
\(\frac{\partial^2 f}{\partial y\,\partial x}\), written as a fraction, as the derivative of a
derivative, and with subscripts, as \(f_{xy}\), \((f_x)_y\) and \(f_{12}\). The functions are
smooth, so the order of the two derivatives does not change the answer (Clairaut's
theorem); a form read as the second derivative by one variable twice does. Stewart (2015)
is *Calculus: Early Transcendentals*, 8th edition; the physics package is the LaTeX package
of that name (de la Barrera, 2012); the Unicode Standard is version 15.0.

`«f»` is the expression, `«first»` and `«second»` the variables it is differentiated by,
first and second, `«name»` the name a form gives the function and `«params»` its
parameters; `«numbers»` are the places of the two variables among them, counted from 1.

## Names

```python
x, y, z, r, theta = symbols("x y z r theta")
```

## Operands

| operand | written in | `f` | `first` | `second` | `name` | `params` | `numbers` | separates |
|---|---|---|---|---|---|---|---|---|
| `x^3y^2` | latex | `x^3 y^2` | `x` | `y` | `f` | `x, y` | `12` | a mixed derivative, 6x²y, from the second by x twice, 6xy², and by y twice, 2x³ |
| | text | `x^3*y^2` | `x` | `y` | `f` | `x, y` | `12` | |
| | unicode | `x³y²` | `x` | `y` | `f` | `x, y` | | |
| | python | `x**3*y**2` | `x` | `y` | | | | |
| `exponential` | latex | `e^{xy}` | `x` | `y` | `g` | `x, y` | `12` | both variables in an exponent: the product rule after the chain rule |
| | text | `exp(x*y)` | `x` | `y` | `g` | `x, y` | `12` | |
| | unicode | `e^(xy)` | `x` | `y` | `g` | `x, y` | | |
| | python | `exp(x*y)` | `x` | `y` | | | | |
| `trigonometric` | latex | `x \sin y + y^2 \cos x` | `x` | `y` | `h` | `x, y` | `12` | a sum of products of functions of each variable |
| | text | `x*sin(y) + y^2*cos(x)` | `x` | `y` | `h` | `x, y` | `12` | |
| | unicode | `x sin y + y² cos x` | `x` | `y` | `h` | `x, y` | | |
| | python | `x*sin(y) + y**2*cos(x)` | `x` | `y` | | | | |
| `quotient` | latex | `\frac{x}{x + y}` | `x` | `y` | `u` | `x, y` | `12` | a quotient, taken whole: (x − y)/(x + y)³ |
| | text | `x/(x + y)` | `x` | `y` | `u` | `x, y` | `12` | |
| | unicode | `x/(x + y)` | `x` | `y` | `u` | `x, y` | | |
| | python | `x/(x + y)` | `x` | `y` | | | | |
| `three-variables` | latex | `x^2 y z^3` | `x` | `z` | `f` | `x, y, z` | `13` | the first and the third of three variables |
| | text | `x^2*y*z^3` | `x` | `z` | `f` | `x, y, z` | `13` | |
| | unicode | `x²yz³` | `x` | `z` | `f` | `x, y, z` | | |
| | python | `x**2*y*z**3` | `x` | `z` | | | | |
| `polar` | latex | `r^3 \cos\theta` | `r` | `\theta` | `f` | `r, \theta` | `12` | a variable written as a Greek letter |
| | text | `r^3*cos(theta)` | `r` | `theta` | `f` | `r, theta` | `12` | |
| | unicode | `r³ cos θ` | `r` | `θ` | `f` | `r, θ` | | |
| | python | `r**3*cos(theta)` | `r` | `theta` | | | | |

## Oracle

```python
expected = diff(f, first, second)
```

## Second route

The coefficient of the product of the two steps in the function's Taylor series about the
point.

```python
h, k = symbols("h k")
shifted = f.subs({first: first + h, second: second + k}, simultaneous=True)
taylor = series(series(shifted, h, 0, 2).removeO(), k, 0, 2).removeO()
assert simplify(expand(taylor).coeff(h, 1).coeff(k, 1) - expected) == 0
```

## Forms

### The operator before the expression

**Form** · `operator` · docs/language.md, Calculus: `\frac{\partial^2}{\partial x \partial y} x^2 y^3`

```latex
\frac{\partial^2}{\partial «first» \partial «second»} «f»
```

**Not of** · `trigonometric` · after the operator, a sum is written in brackets

### Leibniz's fraction, of a function's name

**Form** · `fraction` · Stewart (2015), §14.3, Higher Derivatives: `\frac{\partial^2 f}{\partial y \, \partial x}`, the variable of the first derivative written last

```latex
«name»(«params») := «f»
\frac{\partial^2 «name»}{\partial «second» \, \partial «first»}
```

### The derivative of the derivative

**Form** · `nested` · Stewart (2015), §14.3, Higher Derivatives: `\frac{\partial}{\partial y}\left(\frac{\partial f}{\partial x}\right)`

```latex
«name»(«params») := «f»
\frac{\partial}{\partial «second»}\left(\frac{\partial «name»}{\partial «first»}\right)
```

### Both variables as a subscript

**Form** · `subscripts` · Stewart (2015), §14.3, Higher Derivatives: `f_{xy}`, first by x, then by y; design/notation.md, decision 5.6

```latex
«name»(«params») := «f»
«name»_{«first»«second»}
```

### A subscript on a subscript

**Form** · `subscript-of-subscript` · Stewart (2015), §14.3, Higher Derivatives: `(f_x)_y`

```latex
«name»(«params») := «f»
(«name»_«first»)_«second»
```

### The places of the variables as a subscript

**Form** · `subscript-numbers` · Stewart (2015), §14.3, Higher Derivatives: `f_{12}`

```latex
«name»(«params») := «f»
«name»_{«numbers»}
```

### Two partial signs with subscripts

**Form** · `partial-subscripts` · docs/tensors.md, Derivatives: `\partial_t \partial_t`, one derivative after another, each acting on the factor after it

```latex
«name»(«params») := «f»
\partial_{«first»} \partial_{«second»} «name»
```

### The physics package's fraction

**Form** · `physics-pdv` · the physics package, Derivatives: `\pdv{f}{x}{y}` is the mixed derivative \(\frac{\partial^2 f}{\partial x \partial y}\)

```latex
«name»(«params») := «f»
\pdv{«name»}{«first»}{«second»}
```

### A call of diff

**Form** · `diff` · SymPy's `diff(f, x, y)`, by each variable in turn; docs/language.md, Calculus

```text
diff(«f», «first», «second»)
```

### Both variables as a subscript, in plain text

**Form** · `text-subscripts` · Stewart (2015), §14.3: `f_{xy}`, typed in a line of plain text

```text
«name»(«params») := «f»
«name»_{«first»«second»}
```

**Not of** · `polar` · a Greek letter spelled out would run into the letter before it

### Leibniz's fraction in Unicode

**Form** · `unicode-fraction` · the Unicode Standard, U+2202 PARTIAL DIFFERENTIAL and U+00B2 SUPERSCRIPT TWO: `∂²f/∂y∂x`, as typed

```unicode
«name»(«params») := «f»
∂²«name»/∂«second»∂«first»
```

### The derivative of the derivative, in Unicode

**Form** · `unicode-nested` · the Unicode Standard, U+2202 PARTIAL DIFFERENTIAL: `∂/∂y (∂f/∂x)`, as typed

```unicode
«name»(«params») := «f»
∂/∂«second» (∂«name»/∂«first»)
```
