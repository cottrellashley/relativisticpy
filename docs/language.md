# The language

> **For:** everyone who writes RelativisticPy, whatever the field. **You will learn:**
> statements, names, what `:=` and `=` mean, functions, equations, calculus,
> matrices and quantum notation as books write them, and what the language makes of
> Python's habits. **Before this:**
> [Getting started](getting-started.md).

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

## Optima

The greatest and the least value of what follows, and where it is, are written as books
write them, with the variables beneath the operator: `\max_{x}`, `\min_{x \in S}`,
`\sup_{0 \le x < 1}` and `\inf_{x > 0}`, and `\arg\max_{x}`, `\operatorname*{arg\,min}_{w}`
or `\underset{x}{\min}`, which give the point, not the value. In plain text, `min_{x}` and
`argmax_{x >= 0}`. A set or a condition beneath the operator says what the variables range
over; constraints follow the words "subject to" or "s.t.", on the same line or on the
lines after it, and names before a comma share the relation after them, as in
`x, y \geq 0`. An optimum is of the whole sum it begins, up to a relation or "subject to",
as `\min_x x^2 - 4x + 7` is, and of its factor alone where it is a factor, as in
`2 \max_x f(x)`. `\max(a, b)` is still the greater of two values:

```python
>>> optima = Workbook()
>>> optima.exe(r"""
...     \min_{x \in \mathbb{R}} \left(x^2 - 4x + 7\right)
...     \arg\max_{x \geq 0} \; x e^{-x}
...     \max_{0 \leq x \leq 3} \left(x^3 - 3x\right)
...     \sup_{x \geq 0} \frac{x}{1 + x}
...     \min_{x, y} \; x + y \quad \text{subject to} \quad x^2 + y^2 = 2
...     \max_{x, y} \; 3x + 5y \quad \text{subject to} \quad x \leq 4, \quad 2y \leq 12, \quad 3x + 2y \leq 18, \quad x, y \geq 0
... """)
[3, 1, 18, 1, -2, 36]

```

An optimum over variables is found as the program runs, by the
[optimizers](reference.md#optimizers) the integrations declare, in turn. SymPy's come
first, and find closed forms: from the stationary points and the ends of the domain of a
function of one variable, the Lagrange and KKT conditions of several, the simplex method of
a linear program, or by trying each point of a finite domain, as of `x \in \{0, 1\}^3`.
SciPy's come next, where the problem has no parameters, and find numbers; a problem written
with decimals, as data is, goes to them first. A note says which optimizer found the
optimum, how, and whether it is proven global:

```python
>>> [step.notes for step in optima.each(r"\min_{x, y} \; x + y \quad \text{subject to} \quad x^2 + y^2 = 2")]
[['the global minimum, found exactly by SymPy, from the Lagrange conditions; the constraints bound the region (kkt)']]

```

The variables are real, or whole numbers, as `n \in \mathbb{N}` says. A variable is a
vector where its set is `\mathbb{R}^n` or `\{0, 1\}^n`, or a matrix it is multiplied by
gives its length, as `A \mathbf{x}` does, and its components may be written `x_1, x_2`; a
tensor, where it is written with an index of a space. What the program assumes of its
parameters is used, as `a > 0`; where the answer depends on signs it does not say, it is
found as books find it, with the parameters real and the fewest of them positive, which a
note says (decision 1 of [notation](notation.md)). A maximum that no point reaches, or that
is infinite, is refused, and `\sup` gives the bound:

```python
>>> optima.exe(r"""
...     A := \begin{pmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \end{pmatrix}
...     \mathbf{b} := \begin{pmatrix} 1 \\ 2 \end{pmatrix}
...     \arg\min_{\mathbf{x}} \|\mathbf{x}\|_2 \quad \text{subject to} \quad A\mathbf{x} = \mathbf{b}
...     \max_{x \in \{0, 1\}^3} \; 60x_1 + 100x_2 + 120x_3 \quad \text{subject to} \quad 10x_1 + 20x_2 + 30x_3 \leq 50
...     L(v) = -m c^2 \sqrt{1 - \frac{v^2}{c^2}}
...     H(p) = \max_v \left(p v - L(v)\right)
...     H(p)
... """)
[[[0], [1], [1]], 220, c*sqrt(c**2*m**2 + p**2)]
>>> optima.expr(r"\max_x x^2")
Traceback (most recent call last):
  ...
relativisticpy.errors.EvaluationError: \max_x has no maximum: what it is of grows without bound, to ∞
 --> <input>:1:1
  |
1 | \max_x x^2
  | ^^^^^^^^^^
hint: the bound it never reaches is \sup, as in \sup_{x} f(x)

```

Over an index of data, an optimum binds the index as a sum does, and is the greatest or
the least value along it, and `\arg\max` its label: see
[Beyond spacetime](spaces.md#optima-over-an-index). `A \succ 0` and `A \succeq 0` say that
a symmetric matrix is positive definite or semidefinite, and between vectors, as in the
constraint `w \succeq 0`, hold of each component.

## Functions you can call

Elementary functions (`sin`, `cos`, `tan`, `exp`, `ln`, `log`, `sqrt`, the
inverse and hyperbolic functions, `Abs`, `sign`, `floor`, `ceiling`,
`factorial`, `factorial2` (`n!!`), `binomial`, `gamma`, `re`, `im`, `conjugate`,
...), algebra (`simplify`, `expand`, `factor`, `collect`, `cancel`, `apart`,
`together`, `trigsimp`, `subs`, `N`), calculus (`diff`, `integrate`, `limit`,
`series`, `doit`), equations (`solve`, `dsolve`, `RHS`, `LHS`, `Eq`), matrices
(`det`, `trace`, `rank`, `inverse`, `expm`, `norm` and the rest of
[Matrices](#matrices)), `latex`, `symbols`, and the
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

## Number theory and combinatorics

The arithmetic functions are written by the letters number theory gives them: Euler's
totient `\varphi(n)`, the Möbius function `\mu(n)`, the divisor functions `\sigma_k(n)` and
`d(n)`, `\omega(n)` and `\Omega(n)`, von Mangoldt's `\Lambda(n)` and the partitions `p(n)`.
Physics writes the same letters for an angle, a mean, a frequency or a momentum, so a letter
is the arithmetic function only where its argument is an integer: a whole number above 0, a
name the program assumes an integer, as `n \in \mathbb{N}` does, or the index of a sum.
Anywhere else it is what it was, and a note says which reading was taken. π is a constant,
so `\pi(10)` is 10π; `\pi(n)` counts primes only of a name assumed an integer. `H_n` and
`B_n` are the harmonic and the Bernoulli numbers only where n is such a name, so that `H_0`
stays Hubble's constant; `H_n(x)` is Hermite's polynomial. The Stirling numbers are written
in Karamata's brackets, `{n \brace k}` and `{n \brack k}`, and the Catalan, Fibonacci and Bell
numbers by their names (see the [reference](reference.md#functions)):

```python
>>> nt = Workbook()
>>> nt.exe(r"""
...     \varphi(36) + \mu(30) + d(28)
...     \varphi(x)
...     \pi(10)
...     n \in \mathbb{N}
...     H_{n+1} - H_n
...     \sum_{k=0}^{5} {5 \brace k}
... """)
[17, phi(x), 10*pi, 1/(n + 1), 52]

```

A condition under a sum or a product gives its index its values: `d \mid n` the divisors of
n, `p \le x` the primes up to x, as number theorists let p range over the primes, and
`p \text{ prime}` every prime, whose product of Euler's factors is a value of ζ. The rows of
`\substack` are conditions that each value meets. A sum over an index alone, `\sum_k`, of
binomial coefficients is over every integer, as Concrete Mathematics writes it, and the
coefficients leave the terms that are not 0:

```python
>>> nt.exe(r"""
...     \sum_{d \mid 28} d
...     36 \prod_{p \mid 36} \left(1 - \frac{1}{p}\right)
...     \sum_{\substack{1 \le k \le 10 \\ \gcd(k, 10) = 1}} k
...     \prod_{p \text{ prime}} \frac{1}{1 - p^{-2}}
...     \sum_k \binom{n}{k} x^k y^{n-k}
... """)
[56, 12, 20, pi**2/6, (x + y)**n]

```

`a \bmod m` is the remainder, and `a^{-1} \bmod m` the inverse of a modulo m. `\equiv` with
a modulus on its right is a congruence, a claim decided by the remainders; standing alone,
of a name nothing defines, it is an assumption, as `x > 0` is, and congruences and bounds
that leave the name one value give it that value, by the Chinese remainder theorem.
`\equiv` without a modulus still defines. `solve` gives the remainders that the solutions
of congruences leave, and `[x^n] F(x)` is the coefficient of x^n in the power series of F:

```python
>>> crt = Workbook()
>>> crt.exe(r"""
...     12! \bmod 13
...     17^{-1} \bmod 43
...     3^{100} \equiv 1 \pmod{101}
...     x \equiv 2 \pmod{3}
...     x \equiv 3 \pmod{5}
...     x \equiv 2 \pmod{7}
...     0 \leq x < 105
...     x
...     solve(y^2 \equiv 1 \pmod{8}, y)
...     [z^{10}] \frac{1 - \sqrt{1 - 4z}}{2z}
... """)
[12, 38, True, 23, [1, 3, 5, 7], 16796]

```

## Sequences

Square brackets right after a name are read by what the name is. `x[n] := \ldots` defines a
sequence, a function of the integer n, as Oppenheim and Schafer write a discrete-time
signal, and `x[3]` and `x[n - 1]` are its values. Where nothing defines δ or u, `\delta[n]`
is the unit impulse, 1 at n = 0 and 0 elsewhere, and `u[n]` the unit step, 1 from n = 0 on,
each with a note. That reading needs the brackets to hold an integer as books write a
point, whole numbers and names in sums, differences and multiples, so that
`\delta[(-g)^{-1/2}]` stays a variation. A list or a tensor is read as before, and in plain
text, brackets after a name that is no sequence are refused, as Python's indexing.

A sum over a sequence's index is taken a term at a time: an impulse leaves its term at the
one point where it is 1, and a step cuts its term's range to where it is 1. What is left is
summed in closed form where SymPy finds one, under what the program assumes, as `|z| > 1/2`.
A sum without limits whose index is a sequence's point, as in `\sum_k x[k] h[n - k]`, runs
over every integer, with a note:

```python
>>> signals = Workbook()
>>> signals.exe(r"""
...     x[n] := \delta[n] + 2\delta[n-1] + 3\delta[n-2]
...     h[n] := \delta[n] + \delta[n-1]
...     y[n] := \sum_k x[k] h[n-k]
...     [y[0], y[1], y[2], y[3]]
...     g[n] := \left(\frac{1}{2}\right)^n u[n]
...     g[3] + g[-1]
...     \sum_{n=-\infty}^{\infty} g[n]
...     \mathcal{Z}\{g[n]\}(z)
... """)
[[1, 3, 5, 3], 1/8, 2, 2*z/(2*z - 1)]
>>> print(signals.explain(r"u[n - 1]"))
line 1: u[n - 1]
  sequence or product: u[n - 1] is step, the unit step, since u[n] is how signal processing writes it, and nothing defines u
  n: a symbol
  note: u[n - 1] is read as the unit step, step; for a sequence of your own, define u[n] := ... first

```

A sequence defined through its own members, as `s[n] := \frac{1}{2} s[n-1] + u[n]`, is a
difference equation, at initial rest, as books take one: the sequence is 0 before its input
starts. Its values at whole numbers are computed one after another, and at a symbol in
closed form, where SymPy's `rsolve` finds one. `solve` solves a difference equation for a
sequence nothing defines, at rest too. An equation between sequences that nothing defines,
as `y[n] - \frac{3}{4} y[n-1] = x[n]`, is a difference equation of its own, and their capital
letters are their z-transforms, a delay of k samples \(z^{-k}\): so `Y(z)/X(z)` is the
transfer function, whose `poles` and `zeros` are the roots of its denominator and
numerator. The z-transform of a sequence written with impulses and steps is taken over every
n, and the inverse gives a sequence back, with its step:

```python
>>> systems = Workbook()
>>> systems.exe(r"""
...     s[n] := \frac{1}{2} s[n-1] + u[n]
...     s[2]
...     s[n]
...     solve(v[n] - \frac{1}{2} v[n-1] = \delta[n], v[n])
...     y[n] - \frac{3}{4} y[n-1] + \frac{1}{8} y[n-2] = x[n]
...     H(z) := \frac{Y(z)}{X(z)}
...     H(z)
...     poles(H(z))
...     h[n] := \mathcal{Z}^{-1}\{H(z)\}
...     h[n]
... """)
[7/4, (2 - 1/2**n)*step(n), [step(n)/2**n], Eq(y(n) + y(n - 2)/8 - 3*y(n - 1)/4, x(n)), 8*z**2/(8*z**2 - 6*z + 1), [1/4, 1/2], (2**(n + 1) - 1)*step(n)/4**n]

```

The frequency response is written at \(e^{j\omega}\): `H(e^{j\omega}) := \sum_n h[n]
e^{-j\omega n}` defines H as a function of ω, and H at \(e^{j\theta}\) is its value at θ,
whose modulus and phase are `|H(e^{j\omega})|` and `\arg H(e^{j\omega})`. j is the imaginary
unit where a program says so, `j := \sqrt{-1}`, as engineering books do.

```python
>>> dtft = Workbook()
>>> dtft.exe(r"""
...     j := \sqrt{-1}
...     \omega \in \mathbb{R}
...     |a| < 1
...     h[n] := a^n u[n]
...     H(e^{j\omega}) := \sum_n h[n] e^{-j\omega n}
...     H(e^{j\omega})
...     H(e^{j\pi})
... """)
[1/(-a*exp(-I*omega) + 1), 1/(a + 1)]

```

An asterisk, `*` or `\ast`, between two sequences at points of one index is their
convolution, \(\sum_k x[k]\,h[n - k]\) over every integer k, and between two functions
applied at one variable it is \(\int f(\tau)\,g(t - \tau)\,d\tau\) over every τ, with u and δ
in them the unit step and the Dirac delta where nothing defines them; each is said in a
note. Between anything else, as `2*x`, it multiplies, and written side by side or with
`\cdot`, the two multiply. `\circledast` is the circular convolution, of the length its
subscript gives, `\circledast_N`, or else of the longer sequence (`design/notation.md`, 27):

```python
>>> convolved = Workbook()
>>> convolved.exe(r"""
...     x[n] := \delta[n] + 2\delta[n-1] + 3\delta[n-2] + 4\delta[n-3]
...     h[n] := \delta[n] + \delta[n-1]
...     y[n] := x[n] * h[n]
...     [y[0], y[1], y[2], y[3], y[4]]
...     (x \circledast h)[0]
...     t > 0
...     \left(e^{-t} u(t)\right) * \left(e^{-2t} u(t)\right)
... """)
[[1, 3, 5, 7, 4], 5, (exp(t) - 1)*exp(-2*t)]

```

## Random variables

`X \sim D` declares a random variable, as books write it: `\sim` between a name nothing
defines and a distribution, for the rest of the program. It shows nothing, and a note says
what it declared. Each distribution is written in the forms books give it, which say what its
arguments are: `\mathcal{N}(\mu, \sigma^2)` and `N(\mu, \sigma^2)` take the variance, as
most books write them, `\Gamma(k, \theta)` the shape and the scale, and
`\operatorname{Exp}(\lambda)` the rate. Called by its name in plain text, a distribution takes
its parameters as SymPy and SciPy do, `Normal(mu, sigma)` the standard deviation. The
[reference](reference.md#random-variables) lists them, with the binomial, Poisson, Bernoulli,
uniform, gamma, χ², Student's t, beta and geometric distributions.

`\Pr(A)`, `\mathbb{P}(A)` and `P(X = 2)` are probabilities, `\mathbb{E}[X]` an expectation,
and `\operatorname{Var}(X)` and `\operatorname{Cov}(X, Y)` the variance and the covariance;
inside them `\mid` is the conditional bar. P alone, a pressure elsewhere, is a probability
only of an event of random variables, and E an expectation only of them. SymPy computes each
exactly; where it finds no closed form in 5 seconds and the problem has no parameters, SciPy
computes it numerically, and a note says so:

```python
>>> rv = Workbook()
>>> rv.exe(r"""
...     N \sim \operatorname{Poisson}(\lambda)
...     P(N = 2)
...     Z \sim \mathcal{N}(0, 1)
...     \Pr(Z > 1)
...     \operatorname{Var}(3Z + 1)
...     \mathbb{E}[Z^4]
... """)
[lambda**2*exp(-lambda)/2, 1/2 - erf(sqrt(2)/2)/2, 9, 3]

```

A law may be given by another random variable, as a Poisson number of trials gives a
binomial number of successes, or for each value another takes. Given a random variable, an
expectation is a function of it, and the laws of total expectation and probability give the
rest:

```python
>>> rv.exe(r"""
...     X \mid N \sim \operatorname{Bin}(N, p)
...     \mathbb{E}[X \mid N]
...     \mathbb{E}[X]
...     \operatorname{Var}(X)
...     Y \sim \operatorname{Bernoulli}(1/2)
...     W \mid Y = 0 \sim \mathcal{N}(-1, 1)
...     W \mid Y = 1 \sim \mathcal{N}(1, 1)
...     \Pr(W > 0)
... """)
[p*N, lambda*p, lambda*p, 1/2]

```

Random variables declared in one statement with `\overset{\text{iid}}{\sim}`, or with
`\text{independent}` after the distribution, are independent; declared apart, they are taken
independent, and a note says so. `X_1, \ldots, X_n \overset{\text{iid}}{\sim} D` declares a
sample of n, whose members `X_i` a sum takes in turn. A capital letter nothing defines,
inside a probability, an expectation or a variance, is a random variable of a law not given,
as books write one, so that a rule that holds for every law is decided. The probabilities a
program gives events, as `\Pr(A \mid B_1) := 0.9`, are what it writes of them:

```python
>>> sample = Workbook()
>>> sample.exe(r"""
...     X_1, \ldots, X_n \overset{\text{iid}}{\sim} \mathcal{N}(\mu, \sigma^2)
...     \bar{X} := \frac{1}{n} \sum_{i=1}^{n} X_i
...     \mathbb{E}[\bar{X}]
...     \operatorname{Var}(\bar{X})
...     \operatorname{Var}(a Y + b) = a^2 \operatorname{Var}(Y)
...     \Pr(B) := 0.01
...     \Pr(A \mid B) := 0.9
...     \Pr(A \mid B) \Pr(B)
... """)
[mu, sigma**2/n, True, 0.00900000000000000]

```

`f_X(x)`, `F_X(x)` and `F_X^{-1}(q)` are X's density, distribution function and quantile, and
`M_X(t)` and `\varphi_X(t)` its moment-generating and characteristic functions. Where a
probability holds only for some values of a parameter the program says nothing of, it is
given where the parameter is among the values the variable takes, as a book gives it, and a
note says so:

```python
>>> rv.exe(r"""
...     T \sim \operatorname{Exp}(\lambda)
...     \Pr(T > t)
...     f_Z(x)
...     F_Z^{-1}(0.975)
...     M_N(s)
... """)
[exp(-lambda*t), sqrt(2)*exp(-x**2/2)/(2*sqrt(pi)), 1.95996398454005, exp(lambda*(exp(s) - 1))]

```

Over data, `\mathbb{E}_t`, `\operatorname{Var}_t` and `\operatorname{Cov}_t` are statistics
along an index, as [Beyond spacetime](spaces.md#statistics) says; without an index, of data,
they are refused, saying so.

## Matrices

Matrices are written as LaTeX environments, or as lists of rows, and `vmatrix` is a
determinant:

```python
>>> wb.exe(r"""
...     \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}
...     \begin{vmatrix} a & b \\ c & d \end{vmatrix}
... """)
[[[1, 2], [3, 4]], a*d - b*c]

```

In arithmetic, lists and matrices are vectors and matrices: a number scales
them, arrays of one shape add, matrices multiply as matrices, and a square
matrix has integer powers, so \(A^{-1}\) is its inverse. A matrix written in blocks,
as `\begin{pmatrix} B & AB \end{pmatrix}`, is one matrix. Anything else, such as
adding a number to a list, is an error.

```python
>>> wb.exe(r"""
...     2 [1, 2] + [0, 1]
...     \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} u \\ v \end{pmatrix}
...     \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}^{-1}
... """)
[[2, 5], [[u + 2*v], [3*u + 4*v]], [[-2, 1], [3/2, -1/2]]]

```

The rest of the notation of matrices is read as books write it, by what it is written
on: a superscript T is the transpose of a matrix and a power of a number, and
\(\rho(A)\) is the spectral radius of a matrix, while \(\rho(1 + 3w)\) is not. Each
reading is recorded, as `Workbook.explain` shows, with a note where another was
possible, and a definition of the program's own always wins. The functions of a matrix
are in the [reference](reference.md#matrices).

| Written | Read as |
|---|---|
| `A^T`, `A^\top`, `A^\intercal`, `Aᵀ` | the transpose, where T has no value |
| `A^\dagger`, `A^*`, `A^{-1}`, `A^+` | the conjugate transpose, the inverse and the pseudoinverse |
| `\operatorname{tr} A`, `\det A`, `\operatorname{rank} A` | the trace, the determinant and the rank; without brackets, of the factors that follow, so `\operatorname{tr} A B` is tr(AB) |
| `e^{A t}`, `\exp(A)`, `\log A`, `\sqrt{A}`, `A^{1/2}`, `A^k` | functions of the matrix, not of each entry |
| `\rho(A)`, `\kappa(A)`, `\lambda_{\max}(A)`, `\lambda_k(A)` | the spectral radius, the condition number and the eigenvalues, counted from the smallest |
| `\|A\|_2`, `\|A\|_F`, `\|A\|_\infty`, `\|x\|_1` | norms, of orders 2, Frobenius's, \(\infty\) and 1 |
| `\operatorname{adj} A`, `\ker A`, `\dim \ker A`, `\operatorname{vec} A`, `(A^3)_{12}` | the adjugate, the null space and its dimension, the columns stacked, and an entry, counted from 1 |
| `A \otimes B`, `A \oplus B` | the Kronecker product and the direct sum; between tensors, `\otimes` is their product |
| `I`, `\mathbb{1}`, `𝟙`, `\mathbf{1}` | added to a matrix, the identity of its size; `\mathbf{1}` in a product, the vector of ones of the size it meets |

A number added to a square matrix is that multiple of the identity, and a row times a
matrix times a column, a 1 × 1 matrix, is the number it holds. A function of a matrix of
a name nothing defines, as `\det M` or `\operatorname{tr}(\ln M)`, is an unknown value,
det(M) or tr(log M), since M may be a matrix of any size: a claim about it relates, and is
not proven by its 1 × 1 case. Each says so in a note.

```python
>>> wb.exe(r"""
...     A := \begin{pmatrix} 0 & 1 \\ -2 & -3 \end{pmatrix}
...     B := \begin{pmatrix} 0 \\ 1 \end{pmatrix}
...     C := \begin{pmatrix} 1 & 0 \end{pmatrix}
...     \left(\operatorname{tr} A\right)^2 - 4 \det A
...     \operatorname{rank} \begin{pmatrix} B & AB \end{pmatrix}
...     C (s I - A)^{-1} B
...     \lambda_{\max}(A) + \|A\|_\infty
... """)
[1, 2, 1/(s**2 + 3*s + 2), 4]
>>> wb.exe(r"""
...     R := \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}
...     e^{R t}
...     R^T R + \rho(R)
... """)
[[[cos(t), sin(t)], [-sin(t), cos(t)]], [[2, 0], [0, 2]]]
>>> print(wb.explain(r"R^T - \lambda I"))
line 1: R^T - \lambda I
  matrix or number: R^T is the transpose of R, since R is a matrix, and T has no value
  matrix or number: I is the identity of the size of R^T, since nothing defines I, and it is added to the matrix R^T
  R: a value, defined on line 2
  \lambda: a symbol
  I: the identity
  note: I is read as the identity of the size of R^T, as it is added to it; to name it, define I

```

Given a map written with its indices, as `A^{a}{}_{b}`, a function of a matrix acts on
its matrix of components, and its value is a number or a map of the same space (see
[Beyond spacetime](spaces.md#linear-algebra)).

## Quantum notation

Dirac's notation, commutators and operators are read as books write them, by what they
are written with. As for matrices, each reading is recorded, as `Workbook.explain` shows,
with a note where another was possible.

A ket a program defines as a column is that vector, and its bra is the conjugate
transpose: a bracket is their inner product, `\langle\phi|A|\psi\rangle` a row times a
matrix times a column, and `|a\rangle\langle b|` a matrix. A label may be a sign, an
expression or several, as in `|j, m\rangle`, and `|00\rangle`, of kets the program
defines, is `|0\rangle \otimes |0\rangle`. `[A, B]` of matrices or operators is their
commutator, AB - BA, and `\{A, B\}` their anticommutator; of anything else they are a
list and a set, as before.

```python
>>> qm = Workbook()
>>> qm.exe(r"""
...     |+\rangle := \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 1 \end{pmatrix}
...     |-\rangle := \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ -1 \end{pmatrix}
...     X := \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
...     Z := \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
...     \langle + | X | + \rangle
...     |+\rangle\langle+| + |-\rangle\langle-|
...     [X, Z]
... """)
[1, [[1, 0], [0, 1]], [[0, -2], [2, 0]]]

```

A function of one variable is a wavefunction, and so is a bracket defined as
`\langle x|\alpha\rangle := ...`. A bracket of wavefunctions is an integral, of
\(\phi^* A \psi\), over the domain the program states for the variable, or over the whole
line, and angle brackets around an operator are its expectation value, in the one state
the program defines or the state a subscript names, as in `\langle x \rangle_\psi`.
Written the other way round, a bracket a program defines is its conjugate, its labels
real; and `\psi^*(x)` is the conjugate of \(\psi(x)\).

An expression that ends in an operator with nothing to act on, as `-i\hbar\frac{d}{dx}`
does, is an operator. Defined, it is a value: it acts on what follows it, as in
`\hat{p}\psi(x)`, and products and powers compose it, as in `\hat{p}^2`. Defined with
`\pm` in its name, as in `a_\pm := ...`, it is two, one for each sign. A claim between
operators is decided by applying both sides to a test function, as a note says.

```python
>>> qm.exe(r"""
...     0 \leq x \leq a
...     \psi(x) := \sqrt{\frac{2}{a}} \sin\left(\frac{\pi x}{a}\right)
...     \hat{p} := -i\hbar \frac{d}{dx}
...     \langle\psi|\psi\rangle
...     \langle x \rangle_\psi
...     \langle \hat{p}^2 \rangle_\psi
...     a_\pm := \frac{1}{\sqrt{2\hbar m\omega}} \left(\mp i\hat{p} + m\omega x\right)
...     [a_-, a_+] = 1
... """)
[1, a/2, pi**2*hbar**2/a**2, True]

```

Angle brackets hold what a program defines, as `\langle x^2 \rangle := ...` does, or the
bracket of two, as `\langle p, q \rangle := \int_{-1}^{1} p(x) q(x) \, dx` defines; around
two vectors they are their inner product. An `=` whose left side is written with
operators and kets that nothing defines states a rule for what follows, not a claim: a
commutator, as `[\hat{x}, \hat{p}] = i\hbar`, operators acting on a ket, as
`a|n\rangle = \sqrt{n}|n - 1\rangle`, or a bracket, as `\langle m|n\rangle =
\delta_{mn}`, whose labels stand for any label. The algebra of those operators and kets
reduces by the rules: operators act on kets, bras meet kets, and products are put in the
order the commutators give. The modulus of a bracket of them is the root of the bracket
times its conjugate, so that a sum over a label, as of `|\langle k|H'|0\rangle|^2`, is
taken.

```python
>>> rules = Workbook()
>>> rules.exe(r"""
...     \langle (1, 2), (3, 4) \rangle
...     \langle m | n \rangle = \delta_{mn}
...     a |n\rangle = \sqrt{n}\, |n - 1\rangle
...     a^\dagger |n\rangle = \sqrt{n + 1}\, |n + 1\rangle
...     \langle n | (a + a^\dagger)^2 | n \rangle
...     [\hat{x}, \hat{p}] = i\hbar
...     [\hat{x}^3, \hat{p}] = 3i\hbar \hat{x}^2
... """)
[11, 2*n + 1, True]

```

`\langle j_1, j_2; m_1, m_2|j_1, j_2; j, m\rangle` is a Clebsch–Gordan coefficient, as
Sakurai writes it, and the physics and braket packages' `\ket`, `\bra`, `\braket`,
`\ketbra`, `\expval` and `\mel` read as what they print.

## Vectors

A list, a tuple or a column of numbers is a vector, and is read as one by what it meets:
`\cdot` between vectors is their dot product and `\times` their cross product, while
between numbers both multiply; `|\mathbf{v}|` and `\|\mathbf{v}\|` are its length, and
`\mathbf{p}^2` is \(\mathbf{p} \cdot \mathbf{p}\). Derivatives, integrals, evaluation
bars and limits of a vector are taken component by component. Where nothing defines
them, `\hat{\mathbf{x}}`, `\hat{\imath}` and `\mathbf{i}`, `\mathbf{j}`, `\mathbf{k}` are
the unit vectors of Cartesian coordinates, and so is `\hat{x}`, unless an operator is
near, as `\hat{p}` is; `\hat{\mathbf{r}}` and `\mathbf{e}_\theta` are those of the
coordinates in force. A typeface does not make a name of its own yet: `\mathbf{r}` is
`r` (decision 9 of the [notation decisions](notation.md)).

`\nabla` with no index is vector calculus's: `\nabla f`, `\nabla \cdot \mathbf{v}`,
`\nabla \times \mathbf{v}` and `\nabla^2 f`, or `\Delta f`, act on the factor after them,
and `(\mathbf{u} \cdot \nabla)` on what follows it. An equation between vectors is
decided component by component, and `\mathbf{0}` is the zero vector.

```python
>>> vc = Workbook()
>>> vc.exe(r"""
...     \mathbf{v} := xy\,\hat{\mathbf{x}} + 2yz\,\hat{\mathbf{y}} + 3zx\,\hat{\mathbf{z}}
...     \mathbf{v} \cdot \hat{\mathbf{y}}
...     \nabla \cdot \mathbf{v}
...     \nabla \times \mathbf{v}
...     \nabla \times (\nabla f(x, y, z)) = \mathbf{0}
...     \mathbf{r}(t) := (\cos t, \sin t, t)
...     |\mathbf{r}'(t)|
...     \int_0^{2\pi} \mathbf{r}(t) \, dt
... """)
[2*y*z, 3*x + y + 2*z, [-2*y, -3*z, -x], equation: holds in all 3 components, sqrt(2), [0, 0, 2*pi**2]]

```

The operators are those of the coordinates in force, computed from their metric as index
notation gives them, with each vector's components along their unit vectors: the
gradient is \(g^{ij}\partial_j f\) and the divergence
\(\frac{1}{\sqrt{g}}\partial_i(\sqrt{g}\, v^i)\). The scale factors come from a line
element, from the Cartesian coordinates the program writes in the coordinates it
declares, as `x := r \sin\theta \cos\phi`, or, for coordinates it only declares, from
their names: spherical, cylindrical or polar, as a note says. Without coordinates, they
are x, y and z, or the variables of the function, as of `f(x, y)`.

```python
>>> cylinder = Workbook()
>>> cylinder.exe(r"""
...     ds^2 = d\rho^2 + \rho^2\, d\varphi^2 + dz^2
...     \mathbf{u} := \omega \rho\, \hat{\boldsymbol{\varphi}}
...     \nabla \times \mathbf{u}
...     \nabla^2 \rho^2
... """)
[[0, 0, 2*omega], 4]

```

Where the metric in force is written as the identity, as Cartesian coordinates' line
element is, an index may be repeated in any position and is summed, as in Cartesian
tensors; so too in a space declared as \(\mathbb{R}^n\). There, `\epsilon_{ijk}`, which
nothing defines, is the Levi-Civita symbol, +1 of 1, 2, 3; in spacetime, whose sign is a
convention, it is not read yet.

```python
>>> cartesian = Workbook()
>>> cartesian.exe(r"""
...     ds^2 = dx^2 + dy^2 + dz^2
...     A_{i} := [u(x, y, z), v(x, y, z), w(x, y, z)]
...     \epsilon_{ijk} \partial_j (\epsilon_{klm} \partial_l A_m) = \partial_i \partial_j A_j - \partial_j \partial_j A_i
... """)
[ε_{ijk} ∂_j (ε_{klm} ∂_l A_m) = ∂_i ∂_j A_j − ∂_j ∂_j A_i: holds in all 3 components]

```

Of a function the program defines, written at a point rather than at its variables,
`\nabla f(\mathbf{a})` is its gradient there, a column, and `\nabla^2 f(\mathbf{a})` its
Hessian, as optimisation books write them, said in a note; `\Delta f` is always the
Laplacian. Integrals over a disk or a ball about the origin, written
`\iint_{x^2 + y^2 \le 1}`, and over a box, `\iiint_{[0, 1]^3}`, are taken in turn; any
other region is refused. An integral along a curve the program defines as a function of
one parameter whose value is a vector, with the parameter's range stated, as
`C(t) := (\cos t, \sin t, 0), \quad 0 \le t \le 2\pi`, is written `\oint_C` or `\int_C`:
in it, `d\mathbf{r}` is the curve's tangent times the differential of its parameter, the
curve's name is its point, and x, y and z are its components, so that
`\oint_C \mathbf{F} \cdot d\mathbf{r}` is the circulation of a field. Where SymPy finds
no closed form in five seconds, and nothing but the parameters is left, it is computed
numerically, as a note says.

```python
>>> newton = Workbook()
>>> newton.exe(r"""
...     f(x, y) := x^2 + x y + y^2 - 3x
...     \nabla^2 f(0, 0)
...     \begin{pmatrix} 0 \\ 0 \end{pmatrix} - \left[\nabla^2 f(0, 0)\right]^{-1} \nabla f(0, 0)
...     \Delta f
...     \iint_{x^2 + y^2 \le 1} e^{-(x^2 + y^2)} \, dA
...     C(t) := (\cos t, \sin t, 0), \quad 0 \le t \le 2\pi
...     \oint_C (-y\,\hat{\mathbf{x}} + x\,\hat{\mathbf{y}}) \cdot d\mathbf{r}
... """)
[[[2, 1], [1, 2]], [[2], [-1]], 4, -pi*exp(-1) + pi, 2*pi]

```

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

**Next:** [LaTeX](latex.md), for how LaTeX is read, then
[Tensors and geometry](tensors.md) for General Relativity or
[Beyond spacetime](spaces.md) for data. When a result surprises you, see
[Troubleshooting](troubleshooting.md).
