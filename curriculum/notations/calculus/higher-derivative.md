# Higher derivatives

The second and third derivatives of a function of one variable, written on the expression
in place, \(\frac{d^2}{dx^2} x^4\), and on the name of a function the program defines,
\(f''(x)\), \(\frac{d^2 y}{dx^2}\), \(f^{(3)}(x)\) and \(\ddot{x}\). Stewart (2015) is
*Calculus: Early Transcendentals*, 8th edition; ISO 80000-2:2019 is the standard for
mathematical signs and symbols; Lamport (1994) is *LaTeX: A Document Preparation
System*, 2nd edition; Landau and Lifshitz (1976) is *Mechanics*, 3rd edition;
the amsmath package is described in its *User's Guide* (American Mathematical Society,
2020); the physics package is the LaTeX package of that name (de la Barrera, 2012); the
Unicode Standard is version 15.0.

`«f»` is the expression, `«var»` its variable, `«name»` the name a form gives the function,
and `«order»` the order of the derivative. `«primes»` is the order as primes, `«dots»` as
Newton's dots, `«variables»` as the variable repeated, and `«primed»` and `«raised»` as
Unicode's prime and superscript characters.

## Names

```python
x, t, omega = symbols("x t omega")
```

## Operands

| operand | written in | `f` | `var` | `name` | `order` | `primes` | `dots` | `variables` | `primed` | `raised` | separates |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `x^4` | latex, text | `x^4` | `x` | `f` | `2` | `''` | | `x, x` | | | a second derivative from the first, 4x³ |
| | unicode | `x⁴` | `x` | `f` | `2` | `''` | | | `″` | `²` | |
| | python | `x**4` | `x` | | `2` | | | | | | |
| `x^5` | latex, text | `x^5` | `x` | `f` | `3` | `'''` | | `x, x, x` | | | a third derivative from the second, 20x³ |
| | unicode | `x⁵` | `x` | `f` | `3` | `'''` | | | `‴` | `³` | |
| | python | `x**5` | `x` | | `3` | | | | | | |
| `product` | latex | `x^2 e^x` | `x` | `y` | `2` | `''` | | | | | a product, differentiated twice by the product rule |
| | text | `x^2*exp(x)` | `x` | `y` | `2` | `''` | | `x, x` | | | |
| | unicode | `x² e^x` | `x` | `y` | `2` | `''` | | | `″` | `²` | |
| | python | `x**2*exp(x)` | `x` | | `2` | | | | | | |
| `sine` | latex | `\sin 3x` | `x` | `g` | `2` | `''` | | | | | the chain rule twice, with a factor that squares: the first derivative is 3 cos 3x |
| | text | `sin(3*x)` | `x` | `g` | `2` | `''` | | `x, x` | | | |
| | unicode | `sin 3x` | `x` | `g` | `2` | `''` | | | `″` | `²` | |
| | python | `sin(3*x)` | `x` | | `2` | | | | | | |
| `quotient` | latex | `\frac{1}{1 + x}` | `x` | `f` | `2` | `''` | | | | | a quotient, taken whole: 2/(1 + x)³ |
| | text | `1/(1 + x)` | `x` | `f` | `2` | `''` | | `x, x` | | | |
| | unicode | `1/(1 + x)` | `x` | `f` | `2` | `''` | | | `″` | `²` | |
| | python | `1/(1 + x)` | `x` | | `2` | | | | | | |
| `logarithm` | latex | `\ln x` | `x` | `f` | `3` | `'''` | | | | | a third derivative of a logarithm, 2/x³ |
| | text | `ln(x)` | `x` | `f` | `3` | `'''` | | `x, x, x` | | | |
| | unicode | `ln x` | `x` | `f` | `3` | `'''` | | | `‴` | `³` | |
| | python | `log(x)` | `x` | | `3` | | | | | | |
| `in-t` | latex | `t^3 e^{-t}` | `t` | `x` | `2` | `''` | `\ddot` | | | | a variable that is not x, and a function that is: Newton's dots are written of it |
| | text | `t^3*exp(-t)` | `t` | `x` | `2` | `''` | | `t, t` | | | |
| | unicode | `t³ e^(−t)` | `t` | `x` | `2` | `''` | | | `″` | `²` | |
| | python | `t**3*exp(-t)` | `t` | | `2` | | | | | | |
| `oscillation` | latex | `\cos\omega t` | `t` | `x` | `3` | `'''` | `\dddot` | | | | a third derivative in time, through a constant factor: ω³ sin ωt |
| | text | `cos(omega*t)` | `t` | `x` | `3` | `'''` | | `t, t, t` | | | |
| | unicode | `cos ωt` | `t` | `x` | `3` | `'''` | | | `‴` | `³` | |
| | python | `cos(omega*t)` | `t` | | `3` | | | | | | |

## Oracle

```python
expected = diff(f, var, order)
```

## Second route

The coefficient of the power of the step in the Taylor series about the variable, times
the order's factorial.

```python
h = symbols("h")
taylor = series(f.subs(var, var + h), h, 0, order + 1).removeO()
assert simplify(taylor.coeff(h, order) * factorial(order) - expected) == 0
```

## Forms

### The operator before its operand

**Form** · `operator` · Stewart (2015), §2.8, Higher Derivatives: `\frac{d^2}{dx^2} f(x)`, written of the expression in place

```latex
\frac{d^«order»}{d«var»^«order»} «f»
```

### The operator with an upright d

**Form** · `operator-upright` · ISO 80000-2:2019 sets d upright: `\frac{\mathrm{d}^n f}{\mathrm{d}x^n}`

```latex
\frac{\mathrm{d}^«order»}{\mathrm{d}«var»^«order»} «f»
```

### Leibniz's fraction, of a function's name

**Form** · `leibniz-fraction` · Stewart (2015), §2.8, Higher Derivatives: `\frac{d^2 y}{dx^2}`

```latex
«name»(«var») := «f»
\frac{d^«order» «name»}{d«var»^«order»}
```

### Leibniz's fraction, its orders in braces

**Form** · `leibniz-braces` · Lamport (1994), §3.3.1: a superscript in braces, `\frac{d^{2}y}{dx^{2}}`

```latex
«name»(«var») := «f»
\frac{d^{«order»}«name»}{d«var»^{«order»}}
```

### The derivative of the derivative

**Form** · `nested` · Stewart (2015), §2.8, Higher Derivatives: `\frac{d}{dx}\left(\frac{dy}{dx}\right) = \frac{d^2 y}{dx^2}`

```latex
«name»(«var») := «f»
\frac{d}{d«var»}\left(\frac{d«name»}{d«var»}\right)
```

**Only of** · `x^4`, `product`, `sine`, `quotient`, `in-t` · the second derivative

### Primes on the name, at its variable

**Form** · `prime` · Stewart (2015), §2.8, Higher Derivatives: `f''(x)`, `f'''(x)`

```latex
«name»(«var») := «f»
«name»«primes»(«var»)
```

### Primes on the name alone

**Form** · `prime-alone` · Stewart (2015), §2.8, Higher Derivatives: `y''`, `y'''`; docs/language.md: a function named without its argument is the function at its parameters

```latex
«name»(«var») := «f»
«name»«primes»
```

### The order in brackets

**Form** · `order-in-brackets` · Stewart (2015), §2.8, Higher Derivatives: `f^{(n)}(x)`; docs/language.md, Names and symbols: `f^{(4)}` is the fourth derivative

```latex
«name»(«var») := «f»
«name»^{(«order»)}(«var»)
```

### Newton's dots

**Form** · `dots` · Landau and Lifshitz (1976), §1: `\ddot{q}`, the accelerations; the amsmath package, Accents in math: `\dddot`, three dots

```latex
«name»(«var») := «f»
«dots»{«name»}
```

### The physics package's fraction, of a name

**Form** · `physics-dv` · the physics package, Derivatives: `\dv[n]{f}{x}` is the nth derivative

```latex
«name»(«var») := «f»
\dv[«order»]{«name»}{«var»}
```

### A call of diff, with the order

**Form** · `diff` · SymPy's `diff(f, x, n)`; docs/language.md, Calculus: `diff(sin(x), x)`

```text
diff(«f», «var», «order»)
```

### A call of diff, the variable repeated

**Form** · `diff-repeated` · SymPy's `diff(f, x, x)`, the variable written once for each derivative

```text
diff(«f», «variables»)
```

### Primes on the name, in plain text

**Form** · `text-prime` · docs/language.md, Calculus: Lagrange's primes differentiate a function you have defined, as in `q''(t)`

```text
«name»(«var») := «f»
«name»«primes»(«var»)
```

### The operator in plain text

**Form** · `slash-operator` · AsciiMath's `d/dx f(x)`, with the order written as a power, as a second derivative is typed where no fraction can be set: `d^2/dx^2`

```text
d^«order»/d«var»^«order» («f»)
```

### Primes on the name, in Unicode

**Form** · `unicode-prime` · as typed, with Unicode's superscripts and minus sign, and the apostrophe for the prime

```unicode
«name»(«var») := «f»
«name»«primes»(«var»)
```

### A double or triple prime character

**Form** · `unicode-primed` · the Unicode Standard, U+2033 DOUBLE PRIME and U+2034 TRIPLE PRIME: `f″(x)`

```unicode
«name»(«var») := «f»
«name»«primed»(«var»)
```

### The order in superscript brackets

**Form** · `unicode-order` · the Unicode Standard, U+207D and U+207E SUPERSCRIPT LEFT and RIGHT PARENTHESIS: `f⁽²⁾(x)`

```unicode
«name»(«var») := «f»
«name»⁽«raised»⁾(«var»)
```

### The operator in Unicode

**Form** · `unicode-operator` · as typed with Unicode's superscripts: `d²/dx²`

```unicode
d«raised»/d«var»«raised» («f»)
```
