# Ordinary differential equations

The general solution of an ordinary differential equation in an unknown function y of t,
of the first order, \(\frac{dy}{dt} = f(t, y)\), and of the second, \(\frac{d^2y}{dt^2} =
f(t, y)\), with the derivative written as books write it. Each form asks for the solution
with the language's call, `dsolve(equation, y(t))` (docs/language.md, Equations; written in
a LaTeX statement in docs/tensors.md, Tensor equations), of the equation written in its
book's notation. Stewart (2015) is *Calculus: Early Transcendentals*, 8th edition; ISO
80000-2:2019 is the standard for mathematical signs and symbols; Strogatz (2015) is
*Nonlinear Dynamics and Chaos*, 2nd edition; Taylor (2005) is *Classical Mechanics*; the
Unicode Standard is version 15.0.

`«rhs»` is the right side of the equation as it is written beside a derivative of y alone,
with y; `«at»`, as it is written beside one of y at its variable, with y(t). In Python the
right side is written with y(t), and `order` is the order of the equation. The solution
holds as many constants as its order, which SymPy names C1 and C2.

## Names

```python
t, r, g = symbols("t r g")
y = Function("y")
```

## Operands

| operand | written in | `rhs` | `at` | `order` | separates |
|---|---|---|---|---|---|
| `growth` | latex, unicode | `r y` | `r y(t)` | | a parameter beside the unknown function: growth at a rate |
| | text | `r*y` | `r*y(t)` | | |
| | python | `r*y(t)` | | `1` | |
| `forced` | latex | `t - y` | `t - y(t)` | | the variable on the right: a linear equation with a forcing |
| | text | `t - y` | `t - y(t)` | | |
| | unicode | `t − y` | `t − y(t)` | | |
| | python | `t - y(t)` | | `1` | |
| `exponential-forcing` | latex | `2y + e^{t}` | `2y(t) + e^{t}` | | e as the exponential, beside the unknown function |
| | text | `2*y + exp(t)` | `2*y(t) + exp(t)` | | |
| | unicode | `2y + e^t` | `2y(t) + e^t` | | |
| | python | `2*y(t) + exp(t)` | | `1` | |
| `separable` | latex | `t y^2` | `t y(t)^2` | | a power of the unknown function: an equation that is not linear |
| | text | `t*y^2` | `t*y(t)^2` | | |
| | unicode | `t y²` | `t y(t)²` | | |
| | python | `t*y(t)**2` | | `1` | |
| `oscillator` | latex, text | `-y` | `-y(t)` | | a second derivative, and two constants |
| | unicode | `−y` | `−y(t)` | | |
| | python | `-y(t)` | | `2` | |
| `angular-frequency` | latex | `-4y` | `-4y(t)` | | a multiple of the unknown function: the frequency 2 |
| | text | `-4*y` | `-4*y(t)` | | |
| | unicode | `−4y` | `−4y(t)` | | |
| | python | `-4*y(t)` | | `2` | |
| `free-fall` | latex, text | `-g` | `-g` | | a right side without the unknown function, integrated twice |
| | unicode | `−g` | `−g` | | |
| | python | `-g` | | `2` | |

## Oracle

```python
expected = dsolve(Eq(y(t).diff(t, order), rhs), y(t))
```

## Second route

The solution put back into the equation satisfies it, and holds as many constants as the
equation's order.

```python
equation = Eq(y(t).diff(t, order), rhs)
assert checkodesol(equation, expected) == (True, 0)
assert len(expected.free_symbols - {t, r, g}) == order
```

## Forms

### Leibniz's fraction

**Form** · `leibniz` · Stewart (2015), §9.1, Modeling with Differential Equations: `\frac{dP}{dt} = kP`

```latex
dsolve(\frac{dy}{dt} = «rhs», y(t))
```

**Not of** · `oscillator`, `angular-frequency`, `free-fall` · an equation of the second order

### Leibniz's fraction with an upright d

**Form** · `leibniz-upright` · ISO 80000-2:2019: `\frac{\mathrm{d}f}{\mathrm{d}x}`, the d set upright

```latex
dsolve(\frac{\mathrm{d}y}{\mathrm{d}t} = «rhs», y(t))
```

**Not of** · `oscillator`, `angular-frequency`, `free-fall` · an equation of the second order

### A prime on the name

**Form** · `prime` · Stewart (2015), §9.2, Direction Fields and Euler's Method: `y' = x + y`

```latex
dsolve(y' = «rhs», y(t))
```

**Not of** · `oscillator`, `angular-frequency`, `free-fall` · an equation of the second order

### A prime on the name, at its variable

**Form** · `prime-at` · docs/coming-from.md, Mathematica: `dsolve(y'(x) = y(x), y(x))`

```latex
dsolve(y'(t) = «at», y(t))
```

**Not of** · `oscillator`, `angular-frequency`, `free-fall` · an equation of the second order

### Newton's dot

**Form** · `dot` · Strogatz (2015), §2.1, A Geometric Way of Thinking: `\dot{x} = \sin x`

```latex
dsolve(\dot{y} = «rhs», y(t))
```

**Not of** · `oscillator`, `angular-frequency`, `free-fall` · an equation of the second order

### The operator before the function at its variable

**Form** · `operator-of-name` · Stewart (2015), §2.8, Other Notations: `\frac{d}{dx} f(x)`

```latex
dsolve(\frac{d}{dt} y(t) = «at», y(t))
```

**Not of** · `oscillator`, `angular-frequency`, `free-fall` · an equation of the second order

### A prime at the variable, in plain text

**Form** · `text-prime-at` · docs/coming-from.md, Mathematica: `dsolve(y'(x) = y(x), y(x))`

```text
dsolve(y'(t) = «at», y(t))
```

**Not of** · `oscillator`, `angular-frequency`, `free-fall` · an equation of the second order

### A call of diff, in plain text

**Form** · `text-diff` · docs/language.md, Calculus: `diff(sin(x), x)`, of the unknown function, as SymPy writes `Eq(diff(y(t), t), ...)`

```text
dsolve(diff(y(t), t) = «at», y(t))
```

**Not of** · `oscillator`, `angular-frequency`, `free-fall` · an equation of the second order

### Unicode's prime

**Form** · `unicode-prime` · the Unicode Standard, U+2032 PRIME: `y′(t)`

```unicode
dsolve(y′(t) = «at», y(t))
```

**Not of** · `oscillator`, `angular-frequency`, `free-fall` · an equation of the second order

### A letter with a dot above, in Unicode

**Form** · `unicode-dot` · the Unicode Standard, U+1E8F LATIN SMALL LETTER Y WITH DOT ABOVE: `ẏ`

```unicode
dsolve(ẏ = «rhs», y(t))
```

**Not of** · `oscillator`, `angular-frequency`, `free-fall` · an equation of the second order

**Refused** · `ParseError` · docs/reference.md, Unicode: a letter with an accent, as ȧ, is an error that names it, and says to write the command, `\dot{a}`

### Leibniz's fraction of the second order

**Form** · `leibniz-second` · Stewart (2015), §17.1, Second-Order Linear Equations: `P(x)\frac{d^2y}{dx^2} + Q(x)\frac{dy}{dx} + R(x)y = G(x)`

```latex
dsolve(\frac{d^2 y}{dt^2} = «rhs», y(t))
```

**Only of** · `oscillator`, `angular-frequency`, `free-fall` · an equation of the second order

### Two primes on the name

**Form** · `prime-second` · Stewart (2015), §17.1, Second-Order Linear Equations: `y'' + y' - 6y = 0`

```latex
dsolve(y'' = «rhs», y(t))
```

**Only of** · `oscillator`, `angular-frequency`, `free-fall` · an equation of the second order

### Two primes on the name, at its variable

**Form** · `prime-at-second` · docs/language.md, Calculus: Lagrange's primes, as in `q''(t)`

```latex
dsolve(y''(t) = «at», y(t))
```

**Only of** · `oscillator`, `angular-frequency`, `free-fall` · an equation of the second order

### Newton's two dots

**Form** · `ddot` · Taylor (2005), §5.2, Simple Harmonic Motion: `\ddot{x} = -\omega^2 x`

```latex
dsolve(\ddot{y} = «rhs», y(t))
```

**Only of** · `oscillator`, `angular-frequency`, `free-fall` · an equation of the second order

### Two primes at the variable, in plain text

**Form** · `text-prime-second` · docs/coming-from.md, SymPy: `f''(x)` for `sympy.diff(f(x), x, 2)`

```text
dsolve(y''(t) = «at», y(t))
```

**Only of** · `oscillator`, `angular-frequency`, `free-fall` · an equation of the second order

### Unicode's double prime

**Form** · `unicode-double-prime` · the Unicode Standard, U+2033 DOUBLE PRIME: `y″(t)`

```unicode
dsolve(y″(t) = «at», y(t))
```

**Only of** · `oscillator`, `angular-frequency`, `free-fall` · an equation of the second order
