# The language

> **For:** everyone who writes RelativisticPy, whatever the field. **You will learn:**
> statements, names, what `:=` and `=` mean, functions, equations and calculus, where
> each field's notation is, and what the language makes of Python's habits.
> **Before this:** [Getting started](getting-started.md).

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

A loop written in words, as an algorithm is printed, `\textbf{for } k = 1, \ldots, 5`
with its steps on the lines after it, indented further, is refused, steps and all: the
language has no loops. Write each step out, or what the loop computes as a sequence, from
its first member, as `x_0 := 1` then `x_{k+1} := x_k / 2`.

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

A subscript that labels a quantity is part of its name, whatever it is written with:
`\lambda_1`, `\delta_0`, `V_{10}`, `M_\odot`, `r_+` and `\sigma_x` are names of their own,
and a name the program defines wins over a built-in tensor, as `R_\odot` does over the
Ricci tensor. An index of a tensor in force, a label declared over a space, and a
built-in tensor with its indices stay indices, as in `T_{\mu\nu}` and `g_{tt}`. On the
left of `:=` a number or an order raised on a name is part of it, as in `\sigma^2 := ...`,
and `\Delta x` is one name, the change in x:

```python
>>> names = Workbook()
>>> names.exe(r"""
...     \lambda_1 := 2
...     \sigma^2 := \frac{1}{4}
...     \Delta x := \frac{1}{2}
...     \lambda_1^2 + \sigma^2 + \Delta x
... """)
[19/4]

```

A one-letter subscript that the value of a name being defined is written with makes a
family, and a member is the family at its subscript. A family written with its own
members is a recurrence: given its first members, a member is computed one after
another, or in closed form where SymPy finds one:

```python
>>> names.exe(r"""
...     c_n := \frac{1}{n^2}
...     F_0 := 0
...     F_1 := 1
...     F_{n+2} := F_{n+1} + F_n
...     c_1 + c_2
...     F_{10}
... """)
[5/4, 55]

```

A name with scripts, then names in brackets, defines a function, and so does a primed
name on the left of `:=`. Of a function of one variable, `f^{(4)}` is the fourth
derivative, and a function named without its argument is the function at its
parameters, as y is in `\frac{dy}{dx}`:

```python
>>> names.exe(r"""
...     R_{nl}(r) := r^2 e^{-r}
...     H'(x) := 2 x
...     y(x) := x^3
...     R_{nl}(1) + H'(3)
...     y^{(2)}(1) + \frac{dy}{dx}
... """)
[exp(-1) + 6, 3*x**2 + 6]

```

`Workbook.explain` says how each was read, and a note says so where another reading was
possible.

A bar over a quantity the program defines or declares complex, `\bar{z}` or
`\overline{z}`, is its complex conjugate, and so is a bar over an expression,
`\overline{3 + 4i}`; a star is too, on such a name or on brackets, `z^*` and
`(3 + 4i)^*`. Over a name that means nothing, `\bar{x}` is a name of its own, as it is
where the program defines it, `\bar{x} := \ldots`.

State what a symbol is as books do, with a relation on its name alone: `a > 0`,
`0 < x < 1`, `a, b > 0`, `x \in \mathbb{R}`, `n \in \mathbb{N}` or `n \in \mathbb{Z}_{>0}`.
It is an assumption for the rest of the program, shows nothing, and SymPy simplifies
with it:

```python
>>> wb.exe(r"""
...     a, b > 0
...     \sqrt{a^2 b^2}
...     \sqrt{c^2}
... """)
[a*b, sqrt(c**2)]

```

A name may be used before it is assumed: what was defined with it takes the assumption
from then on, and what was shown stays as it was. What SymPy cannot keep on a symbol,
as x < 1 in `0 < x < 1` or `a > r` in `a > r > 0`, is kept beside it and used where it
decides something, as by `solve`; so is what is assumed of a coordinate, whose symbol
stays as it is. A relation between names alone, as `T_h > T_c`, stays a condition, and
one on a name that has a value is a claim, checked. `Workbook.explain` says which.

The declarations `Constants`, `Reals`, `Integers`, `Complex` and `Positive`, as in
`Positive := [a]`, say the same of a list of names, and `Coordinates` declares the
coordinates.

Where a program says nothing of a name, its symbol stays general. An integral or a limit
that would then hold conditions on a parameter, a modulus of one, or be left undone, is
given under the assumption books make, and a note says so: a parameter is a scale, a
width, a mass, a frequency, positive, and the variable of an integral over real limits is
real. The fewest parameters are assumed, and none the program said something of:

```python
>>> step = list(wb.each(r"\int_0^\infty e^{-s t} \, dt"))[0]
>>> step.values, step.assuming
([1/s], ['s > 0'])
>>> step.notes
['assuming s > 0, as books take a scale to be; the program does not say what s is']

```

`solve` keeps the solutions the program's assumptions allow, and says which it dropped.

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

A claim is decided, or refused as undecided. An equation SymPy cannot show to hold,
though numbers do not show its sides to differ, as an identity it cannot prove, and an
inequality SymPy cannot compare, as `e^x \geq 1 + x`, raise an `UndecidedError` rather
than show the claim as if it were an answer. An equation that numbers show not to hold,
as `x^2 + y^2 = 1`, stays an equation, and an inequality between rational functions of
names, as `a > 0` or `x \geq y + 1`, is a condition, shown as written.

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

An equation is shown, not solved, unless it is given to `solve`: a name it holds stays a
symbol. A value after it that holds that name says so in a note, with how to solve it:

```python
>>> step = list(Workbook().each("x^2 = 4\nx + 1"))[1]
>>> step.values, step.notes
([x + 1], ['x is a symbol here: the equation on line 1 holds x, and is shown, not solved; for the values of x, solve it, as in solve(x^2 = 4, x)'])

```

## Evaluation

An evaluation bar substitutes values, as on paper:

```python
>>> wb.expr(r"x^2 + y |_{x=3, y=1}")
10
>>> wb.expr(r"\left. \frac{d}{dx} x^3 \right|_{x=2}")
12

```

A bar written with `\left.` and `\right|` evaluates what they enclose, and only that. With
limits, a bar or a bracket is the difference of two values, F(b) − F(a), in the variable
written with them; where none is written, in that of an integral beside it, or else in the
one symbol it holds:

```python
>>> wb.expr(r"x + \left. x^2 \right|_{x=2}")
x + 4
>>> wb.expr(r"\left[ x e^x \right]_0^1 - \int_0^1 e^x \, dx")
1

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

A subscript on a function the program defines is a partial derivative, as Stewart writes
it: `f_x` and `f_{xy}` by the variables they name, `f_1` and `f_{12}` by those they number,
and `D_1 f` too, each with a note:

```python
>>> wb.exe(r"""
...     f(x, y) := x^2 y^3
...     f_x
...     f_{12}(1, 2)
...     D_1 f
... """)
[2*x*y**3, 24, 2*x*y**3]

```

Integrals end at their differential, which may also come first or sit in the
numerator of a fraction, where it ends the integral with its term. An evaluation
bar before the differential is inside the integral. A differential alone
integrates 1:

```python
>>> wb.exe(r"""
...     \int_0^1 x^2 \, dx
...     \int_0^\infty e^{-x} dx
...     \int \frac{dx}{x}
...     \int_1^\infty \frac{dr}{r^2} - \pi
...     \int_0^\pi \left. \sin(x y) \right|_{y=1} dx
...     \int_0^L dx
...     integrate(x y, (x, 0, 1), (y, 0, 2))
... """)
[1/3, 1, log(x), 1 - pi, 2, L, 1]

```

Where nothing defines δ, `\delta(x)` of a variable, or of a point as in `\delta(x - a)`, is
Dirac's delta, with a note; δ of what is no point, as `\delta(g^{-1})`, is not:

```python
>>> wb.expr(r"\int_{-\infty}^{\infty} x^2 \delta(x - 2) \, dx")
4

```

Limits, sums and products follow LaTeX. A sum or a product is its value where SymPy
finds a closed form within a few seconds, with its index an integer; one that converges
only under conditions on a parameter is given as books give it, with a note, as an
integral is. Otherwise it is kept as written, with a note:

```python
>>> wb.exe(r"""
...     \lim_{x \to 0} \frac{\sin x}{x}
...     \lim_{x \to 0^+} \frac{1}{x}
...     \sum_{n=1}^{\infty} \frac{1}{n^2}
...     \sum_{k=1}^{N} k
...     \sum_{n=1}^{\infty} \frac{1}{n^n}
... """)
[1, oo, pi**2/6, N**2/2 + N/2, Sum(n**(-n), (n, 1, oo))]

```

In plain text and Unicode, calculus is typed as AsciiMath types it, and reads as the LaTeX
it spells: a solidus between differentials is Leibniz's fraction, `int`, `lim` and `sum`
with their scripts are `\int`, `\lim` and `\sum`, and `sum(f, (n, a, b))` is `\sum`, as
`integrate(f, (x, a, b))` is `\int`:

```python
>>> wb.exe("""
...     d/dx sin(x)
...     d(x^3)/dx
...     int_0^1 x^2 dx
...     lim_(x->0) sin(x)/x
...     sum(1/n^2, (n, 1, oo))
...     ∫_0^π sin²x dx
... """)
[cos(x), 3*x**2, 1/3, 1, pi**2/6, pi/2]

```

The Laplace, Fourier and z-transforms and their inverses are written as books write them,
a calligraphic letter with braces, `\mathcal{L}\{f(t)\}(s)`, which is the transform
whatever the plain letter means. Their variables are named by convention, by the letters
used: ℒ takes t to s, 𝒵 takes n to z, from n = 0, and ℱ takes x to s, with
\(e^{-2\pi i x s}\) as Bracewell writes it, or t to ω, with \(e^{-i\omega t}\). A program
chooses another convention by defining ℱ by its integral,
`\mathcal{F}\{f\}(\omega) := \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty} f(t)\, e^{-i\omega t}\, dt`,
and a note says which convention is used:

```python
>>> transforms = Workbook()
>>> transforms.exe(r"""
...     \mathcal{L}\{t^2\}
...     \mathcal{L}^{-1}\left\{\frac{1}{s^2 + 1}\right\}
...     \mathcal{F}\{e^{-\pi x^2}\}
...     \mathcal{Z}\{(1/2)^n\}(z)
... """)
[2/s**3, sin(t), exp(-pi*s**2), 2*z/(2*z - 1)]

```

The greatest and the least value of what follows, and where it is, are written as books
write them, `\max_{x}`, `\min_{x \in S}` or `\arg\max_{x}`, with constraints after
"subject to": [Optima](optima.md) has them.

## Functions you can call

Elementary functions (`sin`, `cos`, `tan`, `exp`, `ln`, `log`, `sqrt`, the
inverse and hyperbolic functions, `Abs`, `sign`, `floor`, `ceiling`,
`factorial`, `factorial2` (`n!!`), `binomial`, `gamma`, `re`, `im`, `conjugate`,
...), algebra (`simplify`, `expand`, `factor`, `collect`, `cancel`, `apart`,
`together`, `trigsimp`, `subs`, `N`), calculus (`diff`, `integrate`, `limit`,
`series`, `doit`), equations (`solve`, `dsolve`, `RHS`, `LHS`, `Eq`), matrices
(`det`, `trace`, `rank`, `inverse`, `expm`, `norm` and the rest of
[Matrices](matrices.md)), `latex`, `symbols`, and the
[spacetimes](tensors.md#spacetimes). A function without parentheses applies to
what follows it: `sin x` is `sin(x)`. In LaTeX it does so only when it is
written as a command, `\sin x` or `\operatorname{sign} x`, and a function's
name with nothing to act on is a symbol, as the lapse \(N\) is in
`ds^2 = -N^2 dt^2 + \ldots`. A Greek letter written as one, `\gamma` or `γ`,
is always a letter, so `\gamma(r)` in a metric is an unknown function; Euler's
gamma function is `gamma(x)`, spelled out.

A function you call without defining it stays unknown, which is how to write
an unknown function such as \(A(r)\) in a metric. So does a name with a label that nothing
defines, before names in brackets, as `\Phi_N(r)` and `V_{\rm eff}(r)`, and a name
written with `\operatorname` that nothing defines, which applies to what follows it:
`\operatorname{lb} 8` is lb(8). A note says each is unknown.

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

## Fields

Each field is read in the notation of its books, and has a page of its own:

- [Matrices](matrices.md): matrices written as environments or as lists of rows, and the
  notation of linear algebra, as `A^T`, `\operatorname{tr} A`, `\|A\|_2` and `A \otimes B`.
- [Vectors](vectors.md): the dot and cross products, unit vectors, `\nabla` in the
  coordinates in force, and integrals along curves and over regions.
- [Random variables](probability.md): `X \sim D`, probabilities, expectations, variances,
  and the laws they follow.
- [Optima](optima.md): `\max`, `\min`, `\sup`, `\inf` and `\arg\max`, with constraints
  after "subject to".
- [Number theory and sequences](discrete.md): the arithmetic functions, sums over divisors
  and primes, congruences, and sequences, `x[n]`, as signal processing writes them.
- [Quantum notation](quantum.md): Dirac's kets and bras, commutators, and operators as
  values.

## Coming from Python

The language looks like Python in places, `:=`, `**` and calls such as
`diff(f, x)`, but it is not Python. What Python would mean by something the
language reads otherwise is an error rather than a different computation, with
what to write instead:

| Python | What the language says |
|---|---|
| `x[1]` | of a list, it indexes as Python does, which the language does not: a component is written with indices, `V^{0}`; of a sequence, defined as `x[n] := ...`, it is its value at 1 |
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

Three habits of NumPy and MATLAB are read as they mean, each with a note, where what they
are written on is a matrix, and refused elsewhere: `A @ B` is the product of matrices,
`A.T` the transpose, and `eig(A)` the eigenvalues, as `spec(A)` writes them.

**Next:** [LaTeX](latex.md), for how LaTeX is read, then the page of your
[field](#fields), [Tensors and geometry](tensors.md) for General Relativity, or
[Beyond spacetime](spaces.md) for data. When a result surprises you, see
[Troubleshooting](troubleshooting.md).
