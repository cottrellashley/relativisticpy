# Glossary

> **For:** everyone, whenever a word in these pages needs pinning down. **You will
> learn:** what each term means here; the pages use each in this one sense.

```python
>>> from relativisticpy import Workbook, holds
>>> wb = Workbook()
>>> wb.exe(r"""
...     ds^2 = a^2 (d\theta^2 + \sin^2\theta\, d\phi^2)
...     R
...     x^2 + y^2 = 1
... """)
[2/a**2, Eq(x**2 + y**2, 1)]
>>> holds(wb.expr("R = 0")), holds(wb.expr(r"R = \frac{2}{a^2}"))
(False, True)

```

The line element is a *definition*, `R` an *expression*, `x^2 + y^2 = 1` an *equation*
to solve, and `R = 0` a *claim*, which does not hold on a sphere.

## Programs

**Workbook.** A session in which programs run, from Python: `Workbook()`. Definitions
persist from one call to the next, as in the cells of a notebook.

**Program.** Text in the language: a sequence of statements, given as a string, read
from a file (by convention ending in `.relpy`), or taken from the displays of a
Markdown notebook or a LaTeX document.

**Statement.** One step of a program, on a line of its own or separated by `;`: a
definition, a declaration, an expression or an equation.

**Expression.** A statement that computes a value, such as `R_{abcd} R^{abcd}`. Its value
is a *result*.

**Result.** What an expression gives: a SymPy value, a `Tensor`, a `TensorEquation`, an
`IndexEquation` and so on. [The Python API](python.md#results) lists them.

**Step.** What one statement gave, or the error it raised, when `wb.each` runs a program
one statement at a time.

**Raw string.** A Python string with `r` before its quote, `r"\frac{1}{2}"`, in which a
backslash stays a backslash. LaTeX written in Python belongs in raw strings.

## Meanings of `=`

**Definition.** A statement that gives a name a meaning: `x := 2`, `f(x) := x^2`,
`T_{\mu\nu} := \ldots`, or a line element. `:=` always defines; `=` defines only what has
no meaning yet.

**Declaration.** A statement that says what names are, rather than what they equal:
`Coordinates := [t, r, \theta, \phi]`, `Positive := [a]`, `a, b \in \mathcal{A}`,
`MetricSymbol := G`.

**Equation.** Two sides joined by `=`. What it does depends on its left side: it is a
definition, a claim, a condition, or an equation to solve, as `x^2 + y^2 = 1` is.

**Claim.** An equation whose sides both have a meaning already, such as `R = 0` with a
metric in force. A claim is true or false, and RelativisticPy checks it: `holds(claim)`.

**Identity.** A claim that holds whatever its symbols stand for:
`\sin^2 x + \cos^2 x = 1`, or, between tensors without components,
`G_{\mu\nu} = R_{\mu\nu} - \frac{1}{2} R g_{\mu\nu}`.

**Condition.** An equation that fixes a coordinate, such as `t = 0` or
`\theta = \pi/2`. It is read, but not computed.

**Relation.** A statement joined by a relation other than `=` or an inequality, such as
`\approx`, `\propto` or `\sim`. It is read, but not computed. (The verdict *relates* of a
check is something else: an equation between quantities nothing defines.)

## Names

**Symbol.** A name nothing defines, which stands for itself: in `x^2 + 1`, x.

**Unknown function.** A function nothing defines, written with its arguments, as `A(r)`
in a metric. It stays unknown, and can be solved for with `dsolve`.

**Built-in.** What the language names without your defining it: the built-in tensors
(the metric, the Christoffel symbols, the Riemann, Ricci, Einstein and Weyl tensors, the
Kronecker delta), the built-in scalars (`R`, `Kretschmann`), the functions, which are
called *builtins* (`sin`, `simplify`, `Schwarzschild`), and the constants (`pi`, `e`,
`i`). [The language reference](reference.md) lists them all.

## Tensors

**Index notation.** Writing tensors with indices, up with `^` and down with `_`. An index
that appears once up and once down in a term is *summed*, by Einstein's convention; one
that appears once is *free*, and every term has the same free indices.

**Components.** The values of a tensor for each choice of its indices, such as `g_{tt}`.
With a metric in force, the built-in tensors and those defined from them have
components, and an equation between them stands for one equation for each component: a
`TensorEquation`.

**Without components.** A tensor nothing gives components to, such as `T_{\mu\nu}` in
Einstein's equation, is kept as notation, and an equation between such tensors is an
`IndexEquation`, which `solve`, `trace`, `simplify` and `subs` rearrange as on paper.
[Solving tensor equations](notation.md) is about these.

**Metric in force.** The metric defined last, by `g_{\mu\nu} := \ldots` or a line
element. The built-in tensors are those of its geometry.

**Line element.** The metric written as `ds^2 = \ldots`, in the differentials of the
coordinates.

**Coordinates.** The names the components of the metric depend on: declared with
`Coordinates := [\ldots]`, or taken from a line element's differentials, in the order
they first appear.

**Space.** What an index ranges over: the coordinates of spacetime, the assets of a
portfolio, the days of a series, the states of a qubit. A space is declared, as
`\mathcal{H} := \mathbb{C}^{2}`, or comes with data.

**Data.** Numbers bound from Python to a tensor, as in `wb["R_{ta}"] = returns`.

## Documents

**Notebook.** A Markdown file whose displays are a program: `relativisticpy notes.md`
runs it, and `--render` typesets its results under each display. A Jupyter notebook is
something else, from which RelativisticPy is used as a library; see
[Jupyter and notebooks](jupyter.md).

**Display.** An equation set apart from the text: an environment such as `equation` or
`align`, `$$ ... $$` or `\[ ... \]`. A display may hold several statements. The summary
of a check counts displays as its equations.

**Check.** Running the displays of a document in order and giving each statement a
verdict: `relativisticpy check paper.tex`, or `wb.check(document)`.

**Verdict.** What became of a statement in a check: *holds*, *differs*, *relates*,
*ill-formed*, *defines*, *computes*, *not computed*, *not understood* or *not parsed*.
See [Checking papers](checking.md#verdicts).

**Finding.** One statement's verdict, with where the statement is and why. A **report**
is the findings of a check, with their summary.

**Next:** [Troubleshooting](troubleshooting.md), for the mistakes these words help to
explain, or [the language reference](reference.md).
