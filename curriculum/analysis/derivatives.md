# Continuity and derivatives

Continuity at a point, the chain rule, implicit differentiation, and a derivative that does
not exist, as a first course in analysis writes them. Stewart (2015) is *Calculus: Early
Transcendentals*, 8th edition; Rudin (1976) is *Principles of Mathematical Analysis*, 3rd
edition.

## The chain rule

**Level 1** · `analysis.derivatives.chain-rule` · Stewart (2015), §3.4

A derivative through three nested functions, each written as a command.

```latex
\frac{d}{dx} \sin(\cos(\tan x))
```

```python
x = symbols("x")
expected = -cos(cos(tan(x))) * sin(tan(x)) * sec(x) ** 2
assert simplify(expected - diff(sin(cos(tan(x))), x)) == 0
```

## Continuity at a point

**Level 2** · `analysis.derivatives.continuity` · Rudin (1976), Example 5.6

\(x \sin(1/x)\), completed by \(f(0) = 0\), is continuous at 0: a function defined in
pieces, and continuity stated as an equation between a limit and a value.

```latex
f(x) := \begin{cases} x \sin\frac{1}{x} & \text{if } x \neq 0 \\ 0 & \text{if } x = 0 \end{cases}
\lim_{x \to 0} f(x) = f(0)
```

```python
x = symbols("x", real=True)
assert limit(x * sin(1 / x), x, 0) == 0
expected = True
```

## Implicit differentiation

**Level 3** · `analysis.derivatives.implicit` · Stewart (2015), §3.5, Example 2

The folium of Descartes defines \(y\) as a function of \(x\) implicitly. The language must
read the curve and the derivative together: \(y\) depends on \(x\) through the equation
before it.

```latex
x^3 + y^3 = 6xy
\frac{dy}{dx}
```

```python
x, y = symbols("x y")
expected = (2 * y - x**2) / (y**2 - 2 * x)
assert simplify(expected - idiff(x**3 + y**3 - 6 * x * y, y, x)) == 0
```

## A derivative that does not exist

**Level 3** · `analysis.derivatives.not-differentiable` · Rudin (1976), Example 5.6

\(x \sin(1/x)\), completed by \(f(0) = 0\), is not differentiable at 0: its difference
quotient there is \(\sin(1/h)\), which oscillates. Differentiating the two pieces separately
would give 0, from the constant piece; the program must be refused instead.

```latex
f(x) := \begin{cases} x \sin\frac{1}{x} & \text{if } x \neq 0 \\ 0 & \text{if } x = 0 \end{cases}
f'(0)
```

```python
h = symbols("h", real=True)
# the difference quotient (f(h) - f(0)) / h is sin(1/h), which has no limit at 0
assert limit(sin(1 / h), h, 0) == AccumBounds(-1, 1)
error = "EvaluationError"
message = "does not exist"
```
