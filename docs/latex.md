# LaTeX

RelativisticPy reads LaTeX as mathematics. Paste an equation from a paper or
your notes and it runs; mix it with plain text as you like.

```python
>>> from relativisticpy import Workbook
>>> wb = Workbook()
>>> wb.expr(r"\frac{1}{2} \sqrt{\pi} \cdot \left. \frac{d}{dx} x^3 \right|_{x=1}")
3*sqrt(pi)/2

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
joins words, as in `\mathrm{geodesic\_equations}`. In plain text (a
statement without LaTeX commands), `GM` is a single name, as in a programming
language.

```python
>>> wb.exe(r"""
...     \frac{2GM}{c^2 r}
...     \mathrm{GM} + 1
... """)
[2*G*M/(c**2*r), GM + 1]

```

**Functions take what follows them.** `\sin x` is \(\sin(x)\), `\sin^2\theta`
is \(\sin^2(\theta)\) and `\sin^{-1} x` is \(\arcsin(x)\). Without parentheses a
function's argument runs over the following factors (`\sin\omega t` is
\(\sin(\omega t)\)) and stops at explicit spacing, an operator, another
function or a differential: in `\sin^2\theta\, d\phi^2` the argument is
\(\theta\).

**Brackets group.** `\left[ ... \right]` and `\{ ... \}` group like parentheses.
`\left| ... \right|` is an absolute value and `\left. ... \right|_{x=a}` an
evaluation bar.

**Definitions.** `\equiv`, `\coloneqq` and `≔` all mean `:=`.

## Environments

| Environment | Meaning |
|---|---|
| `matrix`, `pmatrix`, `bmatrix`, `Bmatrix`, `smallmatrix`, `array` | a matrix |
| `vmatrix` | a determinant |
| `cases` | a piecewise function, with `\text{if}` and `\text{otherwise}` |
| `align`, `equation`, `gather`, `multline`, `split` and starred forms | statements separated by `\\` |

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

## Unicode

Unicode maths reads like the LaTeX it stands for: `θ` is `\theta`, `∂` is
`\partial`, `∫` is `\int`, `·` is `\cdot`, `≤` is `\le`, and superscripts are
spelled out, so `r²` is `r^{2}`.

```python
>>> wb.expr("∫_0^1 θ² dθ")
1/3

```
