# Where a function's argument ends

A function of an argument, times another factor: \(\sin\theta \cdot \frac{a}{b}\). Without
brackets, the argument is the product of factors that follows the function, up to an
operator, explicit spacing, a fraction, a derivative or another function
(`design/notation.md`, decision 11; docs/latex.md, Functions take what follows them).
Stewart (2015) is *Calculus: Early Transcendentals*, 8th edition; Knuth (1984) is *The
TeXbook*.

`«F»` is the function, `«argument»` its argument, and `«rest»` the factor beside it.

## Names

```python
x, y, theta, a, b = symbols("x y theta a b")
```

## Operands

| operand | written in | `F` | `argument` | `rest` | separates |
|---|---|---|---|---|---|
| `fraction` | latex | `\sin` | `\theta` | `\frac{a}{b}` | a fraction after the argument, which is a factor (decision 11) |
| | text | `sin` | `theta` | `(a/b)` | |
| | python | `sin` | `theta` | `a/b` | |
| `function` | latex | `\sin` | `x` | `\cos x` | another function, which ends the argument |
| | text | `sin` | `x` | `cos(x)` | |
| | python | `sin` | `x` | `cos(x)` | |
| `multiple-angle` | latex | `\cos` | `2x` | `\sin x` | an argument of two factors, before another function |
| | text | `cos` | `2*x` | `sin(x)` | |
| | python | `cos` | `2*x` | `sin(x)` | |
| `letter` | latex | `\sin` | `x` | `y` | a letter, which is a factor only where something ends the argument |
| | text | `sin` | `x` | `y` | |
| | python | `sin` | `x` | `y` | |
| `logarithms` | latex | `\ln` | `x` | `\ln y` | |
| | text | `ln` | `x` | `ln(y)` | |
| | python | `log` | `x` | `log(y)` | |
| `root` | latex | `\cos` | `x` | `\sqrt{y}` | |
| | text | `cos` | `x` | `sqrt(y)` | |
| | python | `cos` | `x` | `sqrt(y)` | |

## Oracle

```python
expected = F(argument) * rest
```

## Second route

The product divided by the factor is the function's value, at a point.

```python
at = {x: Rational(7, 10), y: Rational(13, 10), theta: Rational(3, 5)}
at.update({a: Rational(2, 5), b: Rational(9, 10)})
assert abs(N((expected / rest - F(argument)).subs(at), 30)) < 1e-20
```

## Forms

### The factor straight after the argument

**Form** · `after` · Stewart (2015), Appendix D, the double-angle formula `\sin 2x = 2 \sin x \cos x`; design/notation.md, decision 11: a fraction or another function ends the argument

```latex
«F» «argument» «rest»
```

**Not of** · `letter`, `root` · a letter after the argument would be part of it, and a root is written first

### The factor after a thin space

**Form** · `after-thin-space` · design/notation.md, decision 11: `\sin\theta\,\frac{a}{b}` is `\sin\theta \cdot a/b`; docs/latex.md: the argument stops at explicit spacing

```latex
«F» «argument» \, «rest»
```

### The factor after a centred dot

**Form** · `after-cdot` · docs/latex.md, Functions take what follows them: the argument stops at an operator

```latex
«F» «argument» \cdot «rest»
```

### The factor first

**Form** · `before` · Stewart (2015), §3.3: `y = x^2 \sin x`, a factor written before the function

```latex
«rest» «F» «argument»
```

### The argument in brackets

**Form** · `round` · Stewart (2015), §1.1: `f(x)`, the argument in brackets

```latex
«F»(«argument») «rest»
```

### The function in brackets

**Form** · `function-in-brackets` · Stewart (2015), Appendix D: `(\sin\theta)`, the function with its argument in brackets

```latex
(«F» «argument») «rest»
```

### Calls with a star

**Form** · `call-star` · the plain text of Maple, Maxima and SymPy: `sin(x)*cos(x)`

```text
«F»(«argument»)*«rest»
```

### Calls side by side

**Form** · `call-spaced` · docs/reference.md, Names and numbers: a space between factors multiplies

```text
«F»(«argument») «rest»
```
