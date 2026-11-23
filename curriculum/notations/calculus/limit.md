# Limits

The limit of a function of one variable at a point or at infinity, \(\lim_{x \to a} f(x)\),
from both sides. Stewart (2015) is *Calculus: Early Transcendentals*, 8th edition; Knuth
(1984) is *The TeXbook*; the unicode-math package is the LaTeX package of that name, which
takes Unicode's symbols in maths; AsciiMath is the plain text notation of asciimath.org;
UnicodeMath is the linear format of Unicode Technical Note #28; the Unicode Standard is
version 15.0.

`«f»` is the function, `«var»` its variable and `«point»` where the variable goes. A limit
from both sides is a value only where the limits from below and from above are the same
(Stewart (2015), §2.2): where they differ, or where the function keeps oscillating, the
limit does not exist, and the program must say so rather than give a value.

## Names

```python
x, n, a = symbols("x n a")
```

## Operands

| operand | written in | `f` | `var` | `point` | separates |
|---|---|---|---|---|---|
| `sine-over-x` | latex | `\frac{\sin x}{x}` | `x` | `0` | a quotient taken whole, at a point where it is 0/0 |
| | text, unicode | `sin(x)/x` | `x` | `0` | |
| | python | `sin(x)/x` | `x` | `0` | |
| `e` | latex | `\left(1 + \frac{1}{n}\right)^n` | `n` | `\infty` | a limit at infinity, of a variable that is n |
| | text | `(1 + 1/n)^n` | `n` | `oo` | |
| | unicode | `(1 + 1/n)ⁿ` | `n` | `∞` | |
| | python | `(1 + 1/n)**n` | `n` | `oo` | |
| `rational-at-infinity` | latex | `\frac{3x^2 + x}{2x^2 - 1}` | `x` | `\infty` | a quotient of polynomials at infinity, the ratio of their leading coefficients |
| | text | `(3*x^2 + x)/(2*x^2 - 1)` | `x` | `oo` | |
| | unicode | `(3x² + x)/(2x² − 1)` | `x` | `∞` | |
| | python | `(3*x**2 + x)/(2*x**2 - 1)` | `x` | `oo` | |
| `removable` | latex | `\frac{x^2 - 4}{x - 2}` | `x` | `2` | a point that is not 0 |
| | text | `(x^2 - 4)/(x - 2)` | `x` | `2` | |
| | unicode | `(x² − 4)/(x − 2)` | `x` | `2` | |
| | python | `(x**2 - 4)/(x - 2)` | `x` | `2` | |
| `minus-infinity` | latex | `x e^{x}` | `x` | `-\infty` | a limit at minus infinity, a point with a sign |
| | text | `x*exp(x)` | `x` | `-oo` | |
| | unicode | `x e^x` | `x` | `−∞` | |
| | python | `x*exp(x)` | `x` | `-oo` | |
| `parameter` | latex | `\frac{\sin ax}{x}` | `x` | `0` | a parameter beside the variable: the limit is in x alone |
| | text | `sin(a*x)/x` | `x` | `0` | |
| | unicode | `sin(a x)/x` | `x` | `0` | |
| | python | `sin(a*x)/x` | `x` | `0` | |
| `infinite` | latex | `\frac{1}{x^2}` | `x` | `0` | an infinite limit, the same from both sides |
| | text | `1/x^2` | `x` | `0` | |
| | unicode | `1/x²` | `x` | `0` | |
| | python | `1/x**2` | `x` | `0` | |
| `reciprocal` | latex | `\frac{1}{x}` | `x` | `0` | limits from the two sides that differ, −∞ and ∞: no limit |
| | text, unicode | `1/x` | `x` | `0` | |
| | python | `1/x` | `x` | `0` | |
| `sign` | latex | `\frac{\|x\|}{x}` | `x` | `0` | a jump, from −1 to 1: no limit |
| | text | `abs(x)/x` | `x` | `0` | |
| | unicode | `\|x\|/x` | `x` | `0` | |
| | python | `Abs(x)/x` | `x` | `0` | |
| `oscillating` | latex | `\sin\frac{\pi}{x}` | `x` | `0` | a function that keeps oscillating: no limit |
| | text | `sin(pi/x)` | `x` | `0` | |
| | unicode | `sin(π/x)` | `x` | `0` | |
| | python | `sin(pi/x)` | `x` | `0` | |

## Oracle

The limit from both sides: the limits from below and from above, where they are the same.

```python
f, var, point = S(f), S(var), S(point)
if point.is_infinite:
    found = limit(f, var, point)
else:
    below, above = limit(f, var, point, "-"), limit(f, var, point, "+")
    found = below if below == above else None
if found is None or found.has(AccumBounds, zoo, nan):
    found = None
    error = "RelativisticPyError"
else:
    expected = found
```

## Second route

The function a small step from the point, on each side, as numbers: where the limit is a
number, every value is near it; where it is infinite, every value is large, with its sign;
where there is none, the two sides, or the values on one side, disagree.

```python
at = {a: Rational(7, 5)}
steps = [1 / (sqrt(2) * 10**k) for k in (6, 9, 12)]
if point.is_infinite:
    sides = [[sign(point) / step for step in steps]]
else:
    sides = [[point - step for step in steps], [point + step for step in steps]]
values = [[N(f.subs(at).subs(var, there), 50) for there in side] for side in sides]
if found is None:
    ends = [side[-1] for side in values]
    spread = max(max(side) - min(side) for side in values)
    assert spread > Rational(1, 10) or abs(ends[0] - ends[-1]) > Rational(1, 10)
elif found.is_infinite:
    assert all(value * sign(found) > 10**5 for side in values for value in side)
else:
    wanted = N(found.subs(at), 50)
    assert all(abs(value - wanted) < 10**-5 for side in values for value in side)
```

## Forms

### The limit

**Form** · `lim` · Stewart (2015), §2.2, The Limit of a Function: `\lim_{x \to a} f(x) = L`

```latex
\lim_{«var» \to «point»} «f»
```

### The limit with a long arrow's name

**Form** · `rightarrow` · Knuth (1984), Appendix B: plain TeX's `\to` is `\rightarrow`, which books also write

```latex
\lim_{«var» \rightarrow «point»} «f»
```

### Below the operator in a line of text

**Form** · `limits` · Knuth (1984), chapter 17: `\limits` sets the subscript of an operator below it

```latex
\lim\limits_{«var» \to «point»} «f»
```

### In display style

**Form** · `displaystyle` · Knuth (1984), chapter 17: `\displaystyle`, as a line of text sets a limit as a display does

```latex
\displaystyle\lim_{«var» \to «point»} «f»
```

### The function in brackets

**Form** · `brackets` · Stewart (2015), §2.3, Calculating Limits Using the Limit Laws: the function after the limit in brackets, sized by `\left(` and `\right)`

```latex
\lim_{«var» \to «point»} \left(«f»\right)
```

### Of a function the program defines

**Form** · `function` · Stewart (2015), §2.2: `\lim_{x \to a} f(x)`, of a function defined first

```latex
f(«var») := «f»
\lim_{«var» \to «point»} f(«var»)
```

### Unicode's arrow in LaTeX

**Form** · `unicode-math` · the unicode-math package: Unicode's arrow, U+2192, typed in LaTeX's maths for `\to`

```latex
\lim_{«var» → «point»} «f»
```

### A call of limit

**Form** · `limit` · docs/reference.md, Functions: `limit` `lim`; SymPy's `limit(sin(x)/x, x, 0)`

```text
limit(«f», «var», «point»)
```

### A call of lim

**Form** · `lim-call` · docs/reference.md, Functions: `lim`, the other name of `limit`

```text
lim(«f», «var», «point»)
```

### The limit in plain text

**Form** · `asciimath` · AsciiMath: `lim_(N->oo)`, the variable and its point in round brackets

```text
lim_(«var»->«point») «f»
```

### The limit in UnicodeMath

**Form** · `unicodemath` · UnicodeMath, Unicode Technical Note #28: `lim_(n→∞) a_n`, with the arrow U+2192

```unicode
lim_(«var»→«point») «f»
```

### The limit typed with LaTeX's braces

**Form** · `unicode-braces` · docs/latex.md, Unicode: Unicode reads as the LaTeX it stands for; as typed with LaTeX's braces and the arrow U+2192

```unicode
lim_{«var»→«point»} «f»
```
