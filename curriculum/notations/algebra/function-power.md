# A power of a function

A function's value raised to a power, written on the function's name: \(\sin^2 x\) is
\((\sin x)^2\). Stewart (2015) is *Calculus: Early Transcendentals*, 8th edition; Lamport
(1994) is *LaTeX: A Document Preparation System*, 2nd edition; Griffiths (2013) is
*Introduction to Electrodynamics*, 4th edition; AsciiMath is the plain text
notation of asciimath.org; Maple and Maxima are computer algebra systems; the Unicode
Standard is version 15.0.

The power −1 on a function's name is its inverse, and is a construct of its own
(`inverse-function`).

`«raised»` is the power in Unicode's superscript characters.

## Names

```python
x, theta, n = symbols("x theta n")
```

## Operands

| operand | written in | `F` | `argument` | `power` | `raised` | separates |
|---|---|---|---|---|---|---|
| `sin-squared` | latex | `\sin` | `x` | `2` | | |
| | text | `sin` | `x` | `2` | | |
| | unicode | `sin` | `x` | `2` | `²` | |
| | python | `sin` | `x` | `2` | | |
| `cos-squared-theta` | latex | `\cos` | `\theta` | `2` | | an argument that is a command |
| | text | `cos` | `theta` | `2` | | |
| | unicode | `cos` | `θ` | `2` | `²` | |
| | python | `cos` | `theta` | `2` | | |
| `tan-cubed` | latex | `\tan` | `x` | `3` | | |
| | text | `tan` | `x` | `3` | | |
| | unicode | `tan` | `x` | `3` | `³` | |
| | python | `tan` | `x` | `3` | | |
| `sin-squared-2x` | latex | `\sin` | `2x` | `2` | | a power on the name and a product as the argument |
| | text | `sin` | `2*x` | `2` | | |
| | unicode | `sin` | `2x` | `2` | `²` | |
| | python | `sin` | `2*x` | `2` | | |
| `ln-squared` | latex | `\ln` | `x` | `2` | | a power on a logarithm |
| | text | `ln` | `x` | `2` | | |
| | unicode | `ln` | `x` | `2` | `²` | |
| | python | `log` | `x` | `2` | | |
| `sin-to-the-n` | latex | `\sin` | `x` | `n` | | a power that is a letter |
| | text | `sin` | `x` | `n` | | |
| | unicode | `sin` | `x` | `n` | `ⁿ` | |
| | python | `sin` | `x` | `n` | | |

## Oracle

```python
expected = F(argument) ** power
```

## Second route

The power as the exponential of its logarithm, at a point.

```python
at = {x: Rational(7, 10), theta: Rational(3, 5), n: Rational(5, 2)}
by_logarithm = exp(S(power) * log(F(argument))).subs(at)
assert abs(N(expected.subs(at) - by_logarithm, 30)) < 1e-20
```

## Forms

### The power on the name

**Form** · `on-the-name` · Stewart (2015), Appendix D: `\sin^2\theta + \cos^2\theta = 1`

```latex
«F»^«power» «argument»
```

### The power on the name, in braces

**Form** · `on-the-name-braces` · Stewart (2015), Appendix D, with the power in braces, as Lamport (1994), §3.3.1, Subscripts and Superscripts, sets a superscript

```latex
«F»^{«power»} «argument»
```

### The power on the name, the argument in brackets

**Form** · `on-the-name-round` · Griffiths (2013), §9.2.3: `\cos^2(kz - \omega t + \delta)`, the argument in brackets

```latex
«F»^«power»(«argument»)
```

### The whole in round brackets

**Form** · `round` · Stewart (2015), Appendix D: `\sin^2\theta` is `(\sin\theta)^2`

```latex
(«F» «argument»)^«power»
```

### The whole in square brackets

**Form** · `square` · Stewart (2015), §3.4: `[f(x)]^n`, a power of a function's value

```latex
[«F»(«argument»)]^«power»
```

### The whole in brackets that grow

**Form** · `left-right` · Lamport (1994), §3.3.4: `\left(` and `\right)`

```latex
\left(«F» «argument»\right)^{«power»}
```

### A power after a call

**Form** · `call-power` · the plain text of Maple and Maxima: `sin(x)^2`

```text
«F»(«argument»)^«power»
```

### Two stars after a call

**Form** · `call-double-star` · SymPy's and Python's: `sin(x)**2`

```text
«F»(«argument»)**«power»
```

### The call in brackets

**Form** · `text-round` · the plain text of Maple and Maxima, the whole in brackets: `(sin(x))^2`

```text
(«F»(«argument»))^«power»
```

### The power on the name, in plain text

**Form** · `text-on-the-name` · AsciiMath: `sin^2 x`, as the textbook's form is typed

```text
«F»^«power»(«argument»)
```

### A superscript character on the name

**Form** · `unicode-on-the-name` · the Unicode Standard, U+00B2 SUPERSCRIPT TWO: `sin²x`

```unicode
«F»«raised»«argument»
```

### A superscript character on the name, with a space

**Form** · `unicode-on-the-name-spaced` · the Unicode Standard, U+00B2: `sin² x`

```unicode
«F»«raised» «argument»
```

### A superscript character on the name, the argument in brackets

**Form** · `unicode-on-the-name-round` · the Unicode Standard, U+00B2: `sin²(x)`

```unicode
«F»«raised»(«argument»)
```

### A superscript character after brackets

**Form** · `unicode-round` · the Unicode Standard, U+00B2: `(sin x)²`

```unicode
(«F» «argument»)«raised»
```
