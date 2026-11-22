# Troubleshooting

> **For:** anyone whose result is not what they expected. **You will learn:** the
> mistakes that are easiest to make, why each happens, and what to write instead.

```python
>>> from relativisticpy import Workbook, holds
>>> wb = Workbook()

```

## LaTeX from Python fails, or names a tab or a form feed

In an ordinary Python string, `\a`, `\b`, `\f`, `\n`, `\r`, `\t` and `\v` are control
characters, and Python turns them into those before RelativisticPy sees the text:
`"\theta"` arrives as a tab and `heta`. RelativisticPy refuses such text and says so:

```python
>>> wb.expr("\theta + 1")
Traceback (most recent call last):
  ...
relativisticpy.errors.ParseError: Python turned \t in \theta into a tab
...

```

Write LaTeX in a raw string, with `r` before its quote: `r"\theta + 1"`. A triple-quoted
string takes several lines: `r"""..."""`.

## `=` checked a claim where you meant to define

`:=` always defines, or redefines. `=` defines only what has no meaning yet: a new name,
a new function, a new tensor, a line element. Otherwise it is a claim, which is checked.
So with a metric in force, `R = 0` asks whether the Ricci scalar vanishes; it does not
make `R` zero:

```python
>>> sphere = Workbook()
>>> sphere.expr(r"ds^2 = a^2 (d\theta^2 + \sin^2\theta\, d\phi^2)")
>>> holds(sphere.expr("R = 0"))
False
>>> sphere.expr("R")
2/a**2

```

To give a built-in's name a meaning of your own, write `R := 0`; to give a name that has
a value a new one, write `x := 3`.

## A derivative came out as 0

In `\frac{d f}{dx}`, `f` is a symbol unless something says it depends on x, and the
derivative of a symbol is 0. Write the function with its argument:

```python
>>> wb.exe(r"""
...     \frac{d f}{dx}
...     \frac{d f(x)}{dx}
... """)
[0, Derivative(f(x), x)]

```

## A script took one letter

As in TeX, `^` and `_` without braces take one letter, one command or one number:
`x^ab` is x to the power a, times b, and `\Gamma^r_tt` is `\Gamma^r_t` times t. Put
indices of more than one letter in braces:

```python
>>> wb.exe(r"x^ab + y^{ab}")
[b*x**a + y**ab]

```

## Letters became a product

In LaTeX, a run of letters is a product of single letters, as it is typeset:
`\frac{2GM}{c^2 r}` is 2·G·M/(c²r). Greek names, names you have defined and builtins
stay whole. To keep a longer name whole, write it in `\mathrm{...}` or `\text{...}`. In
plain text, outside a display and without LaTeX commands, a run of letters is one name:

```python
>>> wb.exe(r"""
...     \frac{2GM}{c^2 r}
...     \mathrm{GM} + 1
... """)
[2*G*M/(c**2*r), GM + 1]

```

## A name followed by a bracket became a function

`A(r)` with no space is a call: if nothing defines `A`, it is an unknown function, which
is how a metric writes one. To multiply by a bracket, leave a space or write `\,`:

```python
>>> wb.exe(r"""
...     A(r)
...     G\,(p^2 + q^2)
... """)
[A(r), G*(p**2 + q**2)]

```

## A line did not continue

A statement goes on to the next line while a bracket is open, after a line that ends
with an operator, and before a line that starts with one such as `+` or `=`. A line
that starts with `-` is a statement of its own, as `-x` is, unless it is indented further
than the line before. Indent the continuation of a long equation, or end the line before
it with the operator.

## `%` cut a line short

`%` starts a comment, as in LaTeX, so `x % 2` is not a remainder: RelativisticPy says so.
Write `#` for a comment in plain text.

## `R` or `g` is a symbol, not the curvature

`R` is the Ricci scalar, and `g` the determinant of the metric, unless the metric is
written with them, as the radius `R` of a sphere is; then they are its symbols. `Ric` is
always the Ricci scalar, and `\det(g_{\mu\nu})` always the determinant.

## The metric uses a name defined after it

Papers write "where Δ = 1 − 2M/r" after the line element. Define the names a metric is
written with before its line element, as a reader is told them, so that the metric is
the one meant from the start.

## Getting what you want out of a result

| To get | Write |
|---|---|
| a decimal number | `N(x)`, or `N(x, 5)` for five digits |
| the non-zero components of a tensor | `T.nonzero()` in Python |
| one component | `\Gamma^{r}_{\theta\theta}`, or `T["r", "theta", "theta"]` in Python |
| the equations of a tensor equation, to solve | `E.equations()` in Python |
| whether a claim holds | `holds(value)` in Python, or `E.holds()` |
| LaTeX | `latex(value)`, in the language or in Python |
| a NumPy function | `relativisticpy.numeric.lambdify`; see [Numerics](numerics.md) |

```python
>>> wb.exe("N(pi)\nN(pi, 5)")
[3.14159265358979, 3.1416]

```

## Still stuck

[The language reference](reference.md) lists everything the language reads, and
`relativisticpy help reference` prints it offline. An equation copied from a paper that
fails to run is a bug: please
[report it](https://github.com/cottrellashley/relativisticpy/issues).

**Next:** [The language](language.md), for the rules behind these answers, or the
[glossary](glossary.md).
