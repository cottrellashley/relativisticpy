# A function and its argument

A named function applied to an argument, with brackets and without: \(\sin x\) and
\(\sin(x)\). Lamport (1994) is *LaTeX: A Document Preparation System*, 2nd edition;
Stewart (2015) is *Calculus: Early Transcendentals*, 8th edition; the amsmath package is
described in its *User's Guide* (American Mathematical Society, 2020); Knuth (1984) is *The
TeXbook*.

`«F»` is the function as each notation names it, `«argument»` its argument, and `«name»` the
function's name in letters, as `\operatorname` takes it.

## Names

```python
x, t, omega, theta, a, b = symbols("x t omega theta a b")
```

## Operands

| operand | written in | `F` | `argument` | `name` | separates |
|---|---|---|---|---|---|
| `sin-x` | latex | `\sin` | `x` | `sin` | |
| | text, unicode | `sin` | `x` | | |
| | python | `sin` | `x` | | |
| `cos-2x` | latex | `\cos` | `2x` | `cos` | an argument that is a number times a letter, taken whole (design/notation.md, decision 11) |
| | text | `cos` | `2*x` | | |
| | unicode | `cos` | `2x` | | |
| | python | `cos` | `2*x` | | |
| `sin-omega-t` | latex | `\sin` | `\omega t` | `sin` | an argument of two letters, taken whole |
| | text | `sin` | `omega*t` | | |
| | unicode | `sin` | `ωt` | | |
| | python | `sin` | `omega*t` | | |
| `ln-of-a-power` | latex | `\ln` | `x^2` | `ln` | an argument with a power, which is the argument's |
| | text | `ln` | `x^2` | | |
| | unicode | `ln` | `x²` | | |
| | python | `log` | `x**2` | | |
| `tan-of-a-fraction` | latex | `\tan` | `\frac{x}{2}` | `tan` | a fraction straight after the function, which is its argument |
| | python | `tan` | `x/2` | | |
| `cosh-of-a-sum` | latex | `\cosh` | `a + b` | `cosh` | a sum, which is an argument only in brackets |
| | text, unicode | `cosh` | `a + b` | | |
| | python | `cosh` | `a + b` | | |
| `arctan-of-1` | latex | `\arctan` | `1` | `arctan` | a value |
| | text | `atan` | `1` | | |
| | unicode | `arctan` | `1` | | |
| | python | `atan` | `1` | | |
| `cos-theta` | latex | `\cos` | `\theta` | `cos` | an argument that is a command |
| | text | `cos` | `theta` | | |
| | unicode | `cos` | `θ` | | |
| | python | `cos` | `theta` | | |

## Oracle

```python
expected = F(argument)
```

## Second route

The function by its definition through exponentials and logarithms, at a point.

```python
at = {x: Rational(17, 10), t: Rational(13, 10), omega: Rational(7, 5), theta: Rational(3, 5)}
at.update({a: Rational(2, 5), b: Rational(9, 10)})
by_definition = {
    sin: lambda z: (exp(I * z) - exp(-I * z)) / (2 * I),
    cos: lambda z: (exp(I * z) + exp(-I * z)) / 2,
    tan: lambda z: (exp(I * z) - exp(-I * z)) / (I * (exp(I * z) + exp(-I * z))),
    cosh: lambda z: (exp(z) + exp(-z)) / 2,
    log: lambda z: integrate(1 / t, (t, 1, z)),
    atan: lambda z: integrate(1 / (1 + t**2), (t, 0, z)),
}
assert abs(N((expected - by_definition[F](S(argument))).subs(at), 30)) < 1e-20
```

## Forms

### Without brackets

**Form** · `bare` · Lamport (1994), §3.3.2, Log-like functions: `\sin x`; Stewart (2015), Appendix D

The argument is what follows the function, up to an operator.

```latex
«F» «argument»
```

**Not of** · `cosh-of-a-sum` · a sum is an argument only in brackets

### With round brackets

**Form** · `round` · Stewart (2015), §1.1: `f(x)`, the argument in brackets

```latex
«F»(«argument»)
```

### With brackets that grow

**Form** · `left-right` · Lamport (1994), §3.3.4: `\left(` and `\right)`

```latex
«F»\left(«argument»\right)
```

### With a space before the brackets

**Form** · `spaced-round` · Lamport (1994), §3.3.2: `\sin (x)`, where the space after a command means nothing

```latex
«F» («argument»)
```

### With braces

**Form** · `braces` · Knuth (1984): braces group and print nothing, so `\sin{x}` is `\sin x`

```latex
«F»{«argument»}
```

**Not of** · `cosh-of-a-sum` · a sum is an argument only in brackets

### With square brackets

**Form** · `square` · docs/latex.md, Functions take what follows them: `\exp[-x^2]`, square brackets holding an argument

```latex
«F»[«argument»]
```

### Named with operatorname

**Form** · `operatorname` · the amsmath package, Operator names: `\operatorname{sin}`

```latex
\operatorname{«name»}(«argument»)
```

### A call, in plain text

**Form** · `call` · docs/language.md, Functions you can call: `sin(x)`

```text
«F»(«argument»)
```

### Without brackets, in plain text

**Form** · `text-bare` · docs/language.md, Functions you can call: `sin x` is `sin(x)`

```text
«F» «argument»
```

**Not of** · `cosh-of-a-sum`, `cos-2x`, `sin-omega-t` · a sum or a product is an argument only in brackets

### A call, in Unicode

**Form** · `unicode-call` · as typed, with Greek letters and superscripts as characters

```unicode
«F»(«argument»)
```

### Without brackets, in Unicode

**Form** · `unicode-bare` · as typed: `sin θ`, `cos 2x`

```unicode
«F» «argument»
```

**Not of** · `cosh-of-a-sum` · a sum is an argument only in brackets
