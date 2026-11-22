# Jupyter and notebooks

> **For:** anyone who works in Jupyter, or writes derivations in Markdown. **You will
> learn:** how results display in Jupyter, a first cell to start from, and how a Markdown
> file becomes a notebook that RelativisticPy runs and typesets.

To see a whole program typeset with its results, or to compute mathematics as you type
it, in Jupyter or marimo, see [Live mathematics in notebooks](live.md).

## Jupyter

Install RelativisticPy where Jupyter runs, with the extras your work needs (see
[Getting started](getting-started.md#install)), and start from this cell:

```python
>>> from relativisticpy import Workbook, display, holds
>>> wb = Workbook()
>>> wb.expr(r"g_{\mu\nu} := Schwarzschild(M)")

```

A workbook is a session, like the notebook it is in: what one cell defines, the next
uses. Each result displays as mathematics. A scalar is a SymPy expression, which
Jupyter typesets; a tensor shows its non-zero components, labelled by coordinate; an
equation between tensors says whether it holds; and a metric shows its line element.
What Jupyter shows is the LaTeX of `display.latex`:

```python
>>> print(wb.expr(r"\Gamma^{r}_{ab}")._repr_latex_())
$\displaystyle \begin{aligned}
\Gamma^{r}_{tt} &= \frac{M \left(r - 2 M\right)}{r^{3}} \\
\Gamma^{r}_{rr} &= \frac{M}{r \left(2 M - r\right)} \\
\Gamma^{r}_{\theta\theta} &= 2 M - r \\
\Gamma^{r}_{\phi\phi} &= \left(2 M - r\right) \sin^{2}{\left(\theta \right)}
\end{aligned}$
>>> print(wb.expr(r"R_{\mu\nu} = 0")._repr_latex_())
$$ R_{\mu\nu} = 0 \tag*{holds in all 10 components} $$

```

Numerical data shows as a table, labelled by its spaces, and large data is summarised,
never printed in full. An error shows as RelativisticPy's diagnostic, the place in the
source and the hint, without the frames of Python's traceback; set the environment
variable `RELATIVISTICPY_DEBUG=1` to see them.

A few habits help in a notebook:

- Write LaTeX in raw strings, `r"..."`, and longer programs in triple-quoted ones,
  `r"""..."""`.
- `wb.exe(...)` returns every value of a cell's program, `wb.expr(...)` the last.
- `display.unicode(value)` gives a result as text, for `print`, and
  `display.latex(value)` as LaTeX, for a paper.
- `Workbook(workers=4)` shares component-wise work, such as simplifying the Riemann
  tensor of Kerr, among four processes.

## Markdown notebooks

A Markdown file is a notebook too. Write the derivation in Markdown, with the equations
in displays, `$$ ... $$`, as usual: GitHub and most editors typeset them, and
RelativisticPy runs them, in order, as one program.

```bash
relativisticpy notes.md            # run the displays and print the results
relativisticpy notes.md --render   # print the document, with the results typeset under each display
```

From Python, `Workbook.markdown(text)` does the same as `--render`, and
`Workbook.run_file(path)` the same as running the file:

```python
>>> print(Workbook().markdown("""A number, squared:
...
... $$
... x := 2
... x^2
... $$
... """))
A number, squared:
<BLANKLINE>
$$
x := 2
x^2
$$
<BLANKLINE>
<!-- relativisticpy:result -->
$$
4
$$
<!-- /relativisticpy:result -->
<BLANKLINE>

```

The displays a notebook runs are `$$ ... $$`, `\[ ... \]`, display environments such as
`\begin{equation}`, and fenced blocks of maths, as GitHub writes them. A fenced block
marked `relativisticpy` (or `relpy`) is a program, run as it is written; other code
blocks, code spans and inline maths are text. The [examples](examples.md) are notebooks
of this kind, shown with their results.

**Next:** [Live mathematics in notebooks](live.md), for `show` and the editor,
[Examples](examples.md), or [the Python API](python.md) for every method of a workbook.
