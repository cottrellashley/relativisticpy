# LaTeX

RelativisticPy reads LaTeX as mathematics. Paste an equation from a paper or
your notes and it runs; mix it with plain text as you like.

```python
>>> from relativisticpy import Workbook
>>> wb = Workbook()
>>> wb.expr(r"\frac{1}{2} \sqrt{\pi} \cdot \left. \frac{d}{dx} x^3 \right|_{x=1}")
3*sqrt(pi)/2

```

**From Python, write LaTeX in a raw string,** `r"..."`. In an ordinary string,
Python turns `\a`, `\b`, `\f`, `\n`, `\r`, `\t` and `\v` into control
characters before RelativisticPy sees them, so `"\frac{1}{2}"` arrives as a
form feed and `rac{1}{2}`, and `"\theta"` as a tab and `heta`. RelativisticPy
says so, rather than reading them:

```python
>>> wb.expr("\frac{1}{2}")
Traceback (most recent call last):
  ...
relativisticpy.errors.ParseError: Python turned \f in \frac into a form feed
 --> <input>:1:1
  |
1 | ␌rac{1}{2}
  | ^ a form feed, not \f
note: in a Python string that is not raw, \a, \b, \f, \n, \r, \t and \v are control characters
hint: write the string as a raw string, with r before its quote: r"\frac{1}{2}"

```

## How LaTeX is read

**Typesetting is ignored.** Spacing (`\,`, `\;`, `\quad`), sizing (`\left`,
`\big`), `\displaystyle`, `\label{...}`, `\tag{...}`, math delimiters (`$`, `$$`,
`\[ \]`) and the `&` of alignment environments only change how an equation
looks, so they are skipped. `\mathrm{d}x` is `d x`.

**Letters are symbols.** As in typeset maths, a run of letters in LaTeX is a
product of single-letter symbols: `\frac{2GM}{c^2 r}` is
\(2 G M / (c^2 r)\). Greek names stay whole (`\alpha`, and `alpha` too), as do
names you have defined and builtins such as `diff`. Wrap a longer name in
`\mathrm{...}` or `\text{...}` to keep it one symbol; an escaped underscore
joins words, as in `\mathrm{geodesic\_equations}`. Every statement in a display
(`\begin{equation}`, `align` and the other environments, `\[ \]`, `$$`, and the
displays of a Markdown notebook) is LaTeX, however it is written. In plain
text (a statement outside a display, without LaTeX commands), `GM` is a single
name, as in a programming language.

A Markdown notebook runs its displays in order: `$$ ... $$`, `\[ ... \]`,
display environments such as `\begin{equation}`, and fenced blocks of maths,
```` ```math ````, as GitHub writes them. A fenced block marked
`relativisticpy` (or `relpy`) is a program, run as it is written; other code
blocks, code spans and inline maths are not run.

```python
>>> wb.exe(r"""
...     \frac{2GM}{c^2 r}
...     \mathrm{GM} + 1
... """)
[2*G*M/(c**2*r), GM + 1]

```

**A script without braces is one letter, as in TeX.** `^` and `_` take one
letter, one command or a braced group, so `R^a_b` is \(R^{a}{}_{b}\), and
`\delta^a_b` and `G^t_t` mean what they look like when typeset. TeX sets
`\Gamma^r_tt` as \(\Gamma^r{}_t\, t\), and so does RelativisticPy: write
`\Gamma^{r}_{tt}`. A Greek letter spelled out is one letter, as its command
is (`x^mu` is \(x^\mu\)), and a number is whole (`x^10` is \(x^{10}\)). A
name written with an underscore in plain text, such as `r_s` or `x_0`, is one
name, as in a programming language.

```python
>>> wb.exe(r"""
...     x^ab + y^{ab}
...     r_s := 2 G M
...     1 - r_s/r
... """)
[b*x**a + y**ab, -2*G*M/r + 1]

```

**Functions take what follows them.** `\sin x` is \(\sin(x)\), `\sin^2\theta`
is \(\sin^2(\theta)\) and `\sin^{-1} x` is \(\arcsin(x)\). Square brackets hold
an argument as parentheses do: `\exp[-x^2]`. Without brackets a function's
argument runs over the following factors (`\sin\omega t` is
\(\sin(\omega t)\), and `\ln A(r)` is \(\ln A(r)\)) and stops at explicit
spacing, an operator, another function or a differential: in
`\sin^2\theta\, d\phi^2` the argument is \(\theta\). Only `\log` takes a
subscript, its base, as in `\log_2 x`; `\max_x` and `\det_g` are errors.

**What RelativisticPy prints reads back.** Results are typeset as SymPy
typesets them, with a function's arguments in braces, `a{\left(t \right)}`,
and a derivative of the function after it alone, `\frac{d}{d t} a{\left(t
\right)}`. Both read back as they were meant, so a result written into a paper
holds when the paper is [checked](checking.md):

```python
>>> wb.expr(r"\frac{d}{d t} a{\left(t \right)} \frac{d}{d t} b{\left(t \right)} + a^{2}{\left(t \right)}")
a(t)**2 + Derivative(a(t), t)*Derivative(b(t), t)

```

**Brackets group.** `\left[ ... \right]` and `\{ ... \}` group like parentheses.
`\left| ... \right|` is an absolute value and `\left. ... \right|_{x=a}` an
evaluation bar.

**Definitions.** `\equiv`, `\coloneqq` and `≔` all mean `:=`.

## Environments

| Environment | Meaning |
|---|---|
| `matrix`, `pmatrix`, `bmatrix`, `Bmatrix`, `smallmatrix`, `array` | a matrix |
| `vmatrix` | a determinant |
| `cases`, `dcases` | a piecewise function, with `\text{if}` and `\text{otherwise}`; so is an `array` in `\left\{ ... \right.` |
| `equation`, `align`, `gather`, `split`, `eqnarray`, `flalign`, `alignat`, `aligned`, `gathered`, `math`, breqn's `dmath` and `dgroup`, IEEEtrantools' `IEEEeqnarray`, and their starred forms | statements separated by `\\`; in `align`, each pair of columns is an equation |
| `multline` | one equation, broken over rows |

```python
>>> wb.exe(r"""
...     \begin{align}
...         a &:= 2 \\
...         b &:= a^2 + 1 \\
...         a + b
...     \end{align}
... """)
[7]

```

## Commands

| | Commands |
|---|---|
| Fractions and roots | `\frac` `\dfrac` `\tfrac` `\cfrac` `\sqrt` `\sqrt[n]` `\binom` |
| Big operators | `\sum` `\prod` `\int` `\iint` `\iiint` `\oint` `\lim` |
| Derivatives | `\frac{d}{dx}` `\frac{\partial^2 f}{\partial x \partial y}` `\pdv` `\dv` `\partial_\mu` `\nabla_\mu` |
| Functions | `\sin` `\cos` `\tan` `\cot` `\sec` `\csc` `\arcsin` `\arccos` `\arctan` `\sinh` `\cosh` `\tanh` `\coth` `\exp` `\ln` `\log` `\log_b` `\det` `\min` `\max` `\gcd` `\arg` `\Re` `\Im` `\operatorname{...}` |
| Accents | `\dot` `\ddot` (time derivatives of functions of one variable) and `\bar` `\hat` `\tilde` `\vec` `\check` `\breve` |
| Delimiters | `\left` `\right` `\big`... `\lvert` `\rvert` `\lfloor` `\rfloor` `\lceil` `\rceil` |
| Relations | `=` `\equiv` `\coloneqq` `<` `>` `\le` `\leq` `\ge` `\geq` `\ne` `\neq` `\to` |
| Operators | `\cdot` `\times` `\div` |
| Symbols | all Greek letters and their `\var` forms, `\infty`, `\hbar`, `\ell` |
| Text and style | `\text` `\mathrm` `\mathbf` `\mathcal` `\boldsymbol` and friends |

An unknown command is an error, with a suggestion:

```python
>>> wb.expr(r"\farc{1}{2}")
Traceback (most recent call last):
  ...
relativisticpy.errors.ParseError: unknown LaTeX command \farc
 --> <input>:1:1
  |
1 | \farc{1}{2}
  | ^^^^^
hint: did you mean \frac?

```

A command of LaTeX or of a package that RelativisticPy does not read, such as
the physics package's `\abs` or `\dd`, says what to write instead, and an
unknown environment lists the environments there are:

```python
>>> wb.expr(r"\abs{x}")
Traceback (most recent call last):
  ...
relativisticpy.errors.ParseError: \abs, the physics package's absolute value, is not supported
 --> <input>:1:1
  |
1 | \abs{x}
  | ^^^^
hint: write |x|, or \left| x \right|

```

## Unicode

Unicode maths reads like the LaTeX it stands for: `θ` is `\theta`, `∂` is
`\partial`, `∫` is `\int`, `·` is `\cdot`, `≤` is `\le`, and superscripts are
spelled out, so `r²` is `r^{2}`. A symbol is one symbol, as its command is:
`ℏ` is `\hbar` and `⊙` is `\odot`. Unicode's spaces (the no-break space, the
thin space) are spacing, like `~` and `\,`, and a byte order mark is ignored.

```python
>>> wb.expr("∫_0^1 θ² dθ")
1/3

```

## Notation from papers

Papers are written in LaTeX that is older, looser and more personal than a
notebook's, and RelativisticPy reads it too. That is what lets it
[check a paper](checking.md) as it stands.

**Derivations.** Relations chain, and each link is a claim. In an alignment, a
row that starts with a relation (`&=&`), or with an operator after an
alignment marker (`& & + c`), continues the row before it, as does a bracket
broken over two rows, `\left( a \right. \\ & \left. + b \right)`:

```python
>>> wb.exe(r"""
...     \begin{eqnarray}
...       E &=& {1 \over 2} m v^2 \nonumber \\
...         &=& {m v^2 \over 2} \ ,
...     \end{eqnarray}
... """)
[Eq(E, m*v**2/2), True]

```

**Sentences.** The punctuation that ends a sentence is ignored, as in
`E = mc^2\ .`. Commas and wide spaces (`\quad`, `\qquad`) between equations
separate them, as do `\Rightarrow`, `\implies` and `\iff`, and words such as
`\text{and}`, `\text{or}` and `\text{where}`:

```python
>>> wb.exe(r"{\rm Tr}\,M + {n \choose 2} \qquad r_+ + r^* \qquad \hbar\omega \ .")
[M*Tr + binomial(n, 2), r^* + r_+, hbar*omega]
>>> wb.exe(r"x \equiv 2, \quad y \equiv 3 \quad \text{and} \quad x + y")
[5]

```

A definition shares its line with other equations, as above, and names listed
before `\in` are one declaration: `a, b \in \mathcal{A}`. Names set equal to a
list of values, `\mu, \nu = 0, 1, 2, 3`, give the range of indices; that is read
but not computed, and defines nothing.

| | Notation |
|---|---|
| Plain TeX | `{a \over b}`, `{n \choose k}`, the font switches `{\rm d}`, `{\bf x}`, `{\cal L}`, `\matrix{a & b \cr c & d}`, `\pmatrix`, `\cases`, `\bgroup` |
| Relations read but not computed | `\sim` `\simeq` `\approx` `\cong` `\propto` `\ll` `\gg` `\in` `\subset` `\to` `\leftrightarrow`, `:` as in `\phi : M \to N`, and ranges such as `\mu, \nu = 0, 1, 2, 3` |
| Identities | `A \equiv B`, when A cannot be defined, states that the two are equal |
| Labels on symbols | `r_+`, `i^-`, `r_\pm`, `r^*` and prescripts such as `{}^{(3)}R` name symbols rather than raising to powers |
| Primes | after a numeric subscript a prime marks the whole name, so `\phi_0'(r)` is the derivative of \(\phi_0\); after a braced script it marks the symbol, so `R_{\mu\nu}'` reads as `R'_{\mu\nu}`; after an unbraced index, the index: `x^\mu'` is `x^{\mu'}` |
| Index labels | `\mu'`, `\mu_1`, `\mu_{p+1}`, `\phi_{t*}`, nested groups such as `R_{\rho\sigma{\mu\nu}}`, `\phi,{}_\mu` for \(\phi_{,\mu}\), and MTW's staggered `{\Gamma^r}_{tt}` |
| Operators | `\Box` (\(\nabla^\mu\nabla_\mu\)), the Lie derivative `\pounds_V`, the wedge product `A_{\mu} \wedge B_{\nu}`, `\otimes`, `\pm` and `\mp` (both values) |
| Not computed | `\cdots` and `\ldots`, the Hodge dual `*F`, composition `\circ`, integrals over regions `\int d^4x`, and sums over indices without limits, `\sum_b`, are read, and say why they cannot be computed |

A relation that is read but not computed says so:

```python
>>> wb.expr(r"\rho \propto a^{-3}")
Traceback (most recent call last):
  ...
relativisticpy.errors.CompileError: '∝' is read but not computed: RelativisticPy decides equations and inequalities
 --> <input>:1:1
  |
1 | \rho \propto a^{-3}
  | ^^^^^^^^^^^^^^^^^^^

```
