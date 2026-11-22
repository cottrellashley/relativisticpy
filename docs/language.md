# The language

> **For:** everyone who writes RelativisticPy, whatever the field. **You will learn:**
> statements and comments, numbers and names, what `:=` and `=` mean, functions,
> equations, calculus, piecewise functions and matrices, and what the language makes of
> Python's habits. **Before this:** [Getting started](getting-started.md).

A RelativisticPy program is a sequence of statements: definitions, which
give names to things, and expressions, whose values are the program's results.
Everything below can be written in plain text or in LaTeX; the
[LaTeX page](latex.md) explains how the two relate. The
[language reference](reference.md) lists everything the language reads, on one page.

```python
>>> from relativisticpy import Workbook
>>> wb = Workbook()

```

## Statements

One statement per line. Semicolons, LaTeX's `\\` and `\newline` also separate
statements. Comments start with `#` (or LaTeX's `%`).

```python
>>> wb.exe("n := 2; n^2   # a comment")
[4]

```

Since `%` starts a comment, `x % 2` would be `x`: in plain text, a `%` between
an operand and a number, as Python writes a remainder (`x % 2`, or
`x % 2 == 0`), is an error. Write `#` to comment there. A word after it, as in
`M := 1.989e30 % kg`, is a comment.

A statement continues onto the next line when a bracket is open, when the line
ends with an operator, or when the next line starts with an operator that
cannot start a statement, such as `+`, `*` or `=`. A line that starts with `-`
is a statement of its own, as `-x` is, unless it is indented further than the
statement before it, as long equations usually are:

```python
>>> wb.exe("""
...     total := 1 + 2 +
...              3
...     + 4
...     total
...        - 5
...     -total
... """)
[5, -10]

```

A line that ends with an operator by mistake is reported as the mistake, rather
than the statement on the next line, which was read as its continuation.

## Numbers and arithmetic

Arithmetic is exact: `1/2` is one half, not `0.5`. Decimals stay decimals.
Powers are written `^` or `**`, and juxtaposition is multiplication, though
not of two numbers: `10 000` is an error. Digits grouped in threes by thin
spaces, as SI writes them, are one number: `299\,792\,458`.

```python
>>> wb.exe("1/2 + 1/3\n2.5 * 2\n2^10\n3 p q\n299\\,792\\,458")
[5/6, 5.00000000000000, 1024, 3*p*q, 299792458]

```

`pi`, `e` and `i` are the constants, and `oo` (or `\infty`) is infinity.

```python
>>> wb.exe("e^{i pi}\n1/oo")
[-1, 0]

```

## Names and symbols

A name that has not been defined is a symbol. Greek letters can be spelled
`theta`, `\theta` or `θ`; all three are the same symbol.

```python
>>> wb.expr(r"theta + \theta + θ")
3*theta

```

A subscript on a symbol makes a new symbol: `r_{s}` is the symbol `r_s`, and so
is `r_s`.

```python
>>> wb.expr("r_{s} := 2 G M\nr_s / M")
2*G

```

Declare assumptions about symbols *before* you use them, so that SymPy can
simplify with them:

```python
>>> wb.exe("Positive := [a]\nsqrt(a^2)\nsqrt(b^2)")
[a, sqrt(b**2)]

```

The declarations are `Coordinates`, `Constants`, `Reals`, `Integers`, `Complex` and
`Positive`.

## Definitions, and what `=` means

`:=` defines a name, or gives it a new value; LaTeX's `\equiv` and `\coloneqq` mean the
same.

```python
>>> wb.expr(r"""
...     E \equiv m c^2
...     E
... """)
c**2*m

```

Papers write `=` both to define and to claim, so `=` defines only what has no meaning
yet, and otherwise states something to check or to solve. What it does depends on what
its left side already means:

| You write | When | It means |
|---|---|---|
| `x := 2`, `x \equiv 2`, `x \coloneqq 2` | always | define x, or give it a new value |
| `x = 2` | x has no meaning yet | define x; so do `f(x) = x^2`, a new tensor `T_{ab} = ...` and a line element `ds^2 = ...` |
| `x = 3` | x already has a value | a claim, checked: is x equal to 3? |
| `R = 0`, `g_{tt} = -1` | a metric is in force | a claim about a built-in, or about a component |
| `t = 0` | t is a coordinate | a condition: read, but not computed |
| `x^2 + y^2 = 1` | otherwise | an equation, to solve |
| `a = b = c` | in a derivation | one claim for each link |

To give a name that has a value a new one, or a built-in's name a meaning of your own,
write `:=`.

Functions are defined by writing their parameters:

```python
>>> wb.expr("h(x, y) := x^2 - y\nh(3, 1)")
8

```

Longer functions are blocks. Names defined inside a function stay inside it,
and a function sees the names of the scope it was defined in:

```python
>>> wb.expr("""
...     k := 10
...     energy : (v) -> {
...         half := 1/2
...         half k v^2
...     }
...     energy(2) + half
... """)
half + 20

```

## Equations

`=` between expressions that are not a name being defined is an equation.

```python
>>> wb.expr("x^2 + y^2 = 1")
Eq(x**2 + y**2, 1)

```

Equations are what `solve` and `dsolve` solve, and `RHS` and `LHS` take them
apart:

```python
>>> wb.expr("solve(x^2 = 4, x)")
[-2, 2]
>>> wb.expr("RHS(dsolve(f'(x) = f(x), f(x)))")
C1*exp(x)

```

## Evaluation

An evaluation bar substitutes values, as on paper:

```python
>>> wb.expr(r"x^2 + y |_{x=3, y=1}")
10
>>> wb.expr(r"\left. \frac{d}{dx} x^3 \right|_{x=2}")
12

```

## Calculus

Derivatives can be written in every common notation:

```python
>>> wb.exe(r"""
...     diff(sin(x), x)
...     \frac{d}{dx} \sin x
...     \frac{\partial^2}{\partial x \partial y} x^2 y^3
...     \pdv{x^2 y}{y}
...     d{x^3}/d{x}
... """)
[cos(x), cos(x), 6*x*y**2, x**2, 3*x**2]

```

Lagrange's primes differentiate a function you have defined, or an unknown one:

```python
>>> wb.exe("q(t) := t^3\nq''(t)\nu'(t)")
[6*t, Derivative(u(t), t)]

```

Integrals end at their differential, which may also come first or sit in the
numerator of a fraction. A differential alone integrates 1:

```python
>>> wb.exe(r"""
...     \int_0^1 x^2 \, dx
...     \int_0^\infty e^{-x} dx
...     \int \frac{dx}{x}
...     \int_0^L dx
...     integrate(x y, (x, 0, 1), (y, 0, 2))
... """)
[1/3, 1, log(x), L, 1]

```

Limits, sums and products follow LaTeX. Sums and products are kept as written
until you ask for their value with `doit`:

```python
>>> wb.exe(r"""
...     \lim_{x \to 0} \frac{\sin x}{x}
...     \lim_{x \to 0^+} \frac{1}{x}
...     \sum_{n=1}^{\infty} \frac{1}{n^2}
...     doit(\sum_{n=1}^{\infty} \frac{1}{n^2})
... """)
[1, oo, Sum(n**(-2), (n, 1, oo)), pi**2/6]

```

## Functions you can call

Elementary functions (`sin`, `cos`, `tan`, `exp`, `ln`, `log`, `sqrt`, the
inverse and hyperbolic functions, `Abs`, `sign`, `floor`, `ceiling`,
`factorial`, `factorial2` (`n!!`), `binomial`, `gamma`, `re`, `im`, `conjugate`,
...), algebra (`simplify`, `expand`, `factor`, `collect`, `cancel`, `apart`,
`together`, `trigsimp`, `subs`, `N`), calculus (`diff`, `integrate`, `limit`,
`series`, `doit`), equations (`solve`, `dsolve`, `RHS`, `LHS`, `Eq`), matrices
(`diag`, `det`, `inverse`, `transpose`, `trace`), `latex`, `symbols`, and the
[spacetimes](tensors.md#spacetimes). A function without parentheses applies to
what follows it: `sin x` is `sin(x)`. In LaTeX it does so only when it is
written as a command, `\sin x` or `\operatorname{sign} x`, and a function's
name with nothing to act on is a symbol, as the lapse \(N\) is in
`ds^2 = -N^2 dt^2 + \ldots`. A Greek letter written as one, `\gamma` or `γ`,
is always a letter, so `\gamma(r)` in a metric is an unknown function; Euler's
gamma function is `gamma(x)`, spelled out.

A function you call without defining it stays unknown, which is how to write
an unknown function such as \(A(r)\) in a metric.

## Piecewise functions and other notation

```python
>>> wb.exe(r"""
...     H(x) := \begin{cases} 1 & \text{if } x \geq 0 \\ 0 & \text{otherwise} \end{cases}
...     H(2) + H(-2)
...     step(x) := { 1, x >= 0; 0, otherwise }
...     step(-1)
...     |-3| + 4! + \binom{5}{2} + \lfloor 2.7 \rfloor
... """)
[1, 0, 39]

```

Matrices are written as LaTeX environments, and `vmatrix` is a determinant:

```python
>>> wb.exe(r"""
...     \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}
...     \begin{vmatrix} a & b \\ c & d \end{vmatrix}
... """)
[[[1, 2], [3, 4]], a*d - b*c]

```

In arithmetic, lists and matrices are vectors and matrices: a number scales
them, arrays of one shape add, matrices multiply as matrices, and a square
matrix has integer powers, so \(A^{-1}\) is its inverse. Anything else, such as
adding a number to a list, is an error.

```python
>>> wb.exe(r"""
...     2 [1, 2] + [0, 1]
...     \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} u \\ v \end{pmatrix}
...     \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}^{-1}
... """)
[[2, 5], [[u + 2*v], [3*u + 4*v]], [[-2, 1], [3/2, -1/2]]]

```

## Coming from Python

The language looks like Python in places, `:=`, `**` and calls such as
`diff(f, x)`, but it is not Python. What Python would mean by something the
language reads otherwise is an error rather than a different computation, with
what to write instead:

| Python | What the language says |
|---|---|
| `x[1]` | it indexes as Python does, which the language does not: a component is written with indices, `V^{0}` |
| `[i^2 for i in range(3)]` | a list comprehension: write the list out, or a sum, `\sum_{i=0}^{2} i^2` |
| `import sympy` | the functions are built in: there is nothing to import |
| `series(sin(x), x, 0, n=5)` | a keyword argument: give the value in its place, `series(sin(x), x, 0, 5)` |
| `x % 2` | `%` starts a comment, as in LaTeX, so this would be `x` |
| `"\frac{1}{2}"` | Python turned `\f` into a form feed: write LaTeX in a raw string, `r"\frac{1}{2}"` |

```python
>>> wb.expr("x := [1, 2, 3]\nx[1]")
Traceback (most recent call last):
  ...
relativisticpy.errors.ParseError: x[1] indexes as Python does, which the language does not
 --> <input>:2:1
  |
2 | x[1]
  | ^^^^
hint: a component is written with indices, as in V^{0}; for a product, put a space before the bracket

```

An equation in a call is an equation, as in `subs(x^2, x = 2)` or
`solve(x^2 = 4, x)`: only after the first argument, a name set to a value with
no space around `=`, and used by no other argument, is taken for a keyword
argument.

**Next:** [LaTeX](latex.md), for how LaTeX is read, then
[Tensors and geometry](tensors.md) for General Relativity or
[Beyond spacetime](spaces.md) for data. When a result surprises you, see
[Troubleshooting](troubleshooting.md).
