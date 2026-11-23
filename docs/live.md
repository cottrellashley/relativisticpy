# Live mathematics in notebooks

> **For:** anyone who works in marimo or Jupyter. **You will learn:** how results show
> themselves typeset, what `show` shows, and how the editor computes as you type.
> **Before this:** [Jupyter and notebooks](jupyter.md).

Write mathematics in a notebook and see it computed as it is written. Results show
themselves typeset. `show` runs a program and shows each statement with its results,
its notes and its errors, in place. The editor computes a program as it is typed, and
computes any part of a statement you select.

## Install

```bash
pip install "relativisticpy[live] @ git+https://github.com/cottrellashley/relativisticpy"
```

The `live` extra adds anywidget, which the editor needs. Typeset results and `show` need
nothing more. marimo and Jupyter are your own: RelativisticPy installs neither.

## Results show themselves

In marimo and in Jupyter, a tensor, a metric, an equation, index notation and a geodesic
show themselves typeset, with nothing to call. marimo calls `_display_`, and Jupyter
`_repr_latex_`, or `_repr_html_` for numerical data, which is a table. Each goes through
`relativisticpy.display`, so a result looks the same in a notebook as at the command line:

```python
>>> from relativisticpy import Workbook, display
>>> wb = Workbook()
>>> wb.expr(r"g_{\mu\nu} := Schwarzschild(M)")
>>> vacuum = wb.expr(r"R_{\mu\nu} = 0")
>>> vacuum._repr_latex_() == display.jupyter_latex(vacuum)
True
>>> print(display.jupyter_latex(vacuum))
$$ R_{\mu\nu} = 0 \tag*{holds in all 10 components} $$

```

A scalar is a SymPy expression, which marimo and Jupyter typeset themselves.

## show

`show(source)` runs a program, one statement at a time, and gives back what the cell
shows: each statement typeset, then its results, what it defines and its notes. A
statement that fails shows its error in its place, as `error.render()` gives it, and the
statements after it still run, as `relativisticpy --render` typesets a Markdown notebook:

```python
>>> from relativisticpy.live import show
>>> shown = show(r"""
... g_{\mu\nu} := Schwarzschild(M)
... R_{abcd} R^{abcd}
... \frca{1}{2}
... \Gamma^{r}_{\theta\theta}
... """)
>>> print(shown)
g_{\mu\nu} := Schwarzschild(M)
  defines the metric g_{μν} and the coordinates t, r, θ, φ
R_{abcd} R^{abcd}
  48M²/r⁶
ParseError: unknown LaTeX command \frca
 --> <input>:4:1
  |
4 | \frca{1}{2}
  | ^^^^^
hint: did you mean \frac?
\Gamma^{r}_{\theta\theta}
  2M − r

```

That is the text of it, for a console. In a notebook it is HTML, typeset by the
notebook: by KaTeX in marimo and by MathJax in Jupyter. From Python, it has the `steps`
of the program, as `Workbook.each` gives them, their `values` and `errors`, whether it
is `ok`, the `workbook` it ran in, and its `html()`:

```python
>>> shown.values
[48*M**2/r**6, 2*M - r]
>>> [error.kind for error in shown.errors], shown.ok
(['ParseError'], False)

```

Without a workbook, each call runs in a new one. With `workbook=`, the definitions of one
call are there in the next, as from one cell to the next:

```python
>>> show(r"\Gamma^{t}_{tr}", workbook=wb).values
[M/(r*(-2*M + r))]

```

## Python objects

Python reads the files, in its own cells, and the mathematics reads the objects by name:
`\texttt{signal}` is the object passed to `show` as `signal=signal`. The objects passed
come before the scope of the workbook, if it has one, and are read for that call alone
(see [Beyond spacetime](spaces.md#python-objects-by-name) for what each becomes):

```python
>>> import numpy as np
>>> signal = np.array([1.0, 2.0, 6.0])
>>> show(r"x_t := \texttt{signal}" "\n" r"\mathbb{E}_t[x_t]", signal=signal).values
[3.0]

```

In marimo, pass each object a cell's mathematics reads by name. marimo runs a cell again
whenever a name it uses changes, and it sees the names the cell passes, so the mathematics
follows the data. One cell reads the table:

```python
prices = pd.read_csv("prices.csv", index_col="date")
```

and another shows the mathematics, and runs again whenever `prices` changes:

```python
show(
    r"""
P_{ta} := \texttt{prices}
\mathbb{E}_t[P_{ta}]
""",
    prices=prices,
)
```

`Workbook(scope=globals())` reads any name of the notebook too, but marimo cannot see
which, so a cell that uses it does not run again when one of them changes.

## The editor

The editor is a widget: the source on the left, in LaTeX, plain text or Unicode, and on
the right each statement typeset with its results. In marimo:

```python
import marimo as mo
from relativisticpy import Workbook
from relativisticpy.live import Editor

editor = mo.ui.anywidget(Editor(Workbook()))
editor
```

In Jupyter and VS Code, `Editor()` as the last line of a cell shows it. What it does:

- **It computes as you type.** About 300 ms after you stop typing, the program runs, from
  its first changed statement: the statements before it keep what came of them.
  Ctrl+Enter runs it at once. A run that more typing makes stale is dropped.
- **It never changes your workbook.** `Editor(wb)` starts each run from a copy of `wb` as
  it is then, so a half-typed definition never reaches it, and what other cells define
  in `wb` is there at the next run.
- **Errors are where they are.** The span of an error is underlined in the source, and its
  message, notes and hint show under the pointer and beside the statement.
- **Point and compute.** Select part of a statement and right-click it, or press
  *Compute selection*, for a menu: compute, simplify, expand, does it hold, and copy
  LaTeX. The part computed is the smallest expression that holds the selection, in the
  workbook as it stands at that statement, and its answer shows under the statement.
  Right-click a result for the same menu, on its statement, and to copy the result as
  LaTeX or text. Click a statement on the right to select it in the source.

### Its value is data

What the editor shows is data: `source`, and `statements`, one for each statement, each
as a line of `relativisticpy --format json` gives it, with its values as
`display.data` describes them, each saying whether it is `complete` and whether it is
`approximate` (see [the command line](cli.md#forms-of-output)). In marimo, another cell
that reads `editor.value["statements"]` runs again whenever they change. From Python,
`run` computes the source at once, and waits for it:

```python
>>> from relativisticpy.live import Editor
>>> editor = Editor(wb, source="K := R_{abcd} R^{abcd}\nK r^6")
>>> statements = editor.run()
>>> [entry["statement"] for entry in statements]
['K := R_{abcd} R^{abcd}', 'K r^6']
>>> statements[0]["definition"], statements[1]["values"][0]["text"]
('defines K', '48*M^2')
>>> "K" in wb
False

```

Beside the fields of `--format json`, an entry has `start` and `end`, where its statement
is in the source, `typeset`, the LaTeX it is typeset from, and `definition`, what it
defines in words. Its `error` has `start` and `end` too, and is `rendered` as the command
line prints it. `compute` computes a part of the source, as the menu does, with `compute`,
`simplify`, `expand`, `holds` or `latex`:

```python
>>> editor.source = r"\frac{d}{dr}\left(1 - \frac{2M}{r}\right) = \frac{2M}{r^2}"
>>> answer = editor.compute(0, 41, "simplify")
>>> answer["part"], answer["values"][0]["text"]
('\\frac{d}{dr}\\left(1 - \\frac{2M}{r}\\right)', '2*M/r^2')
>>> editor.compute(0, len(editor.source), "holds")["holds"]
True
>>> editor.close()

```

### Where it computes

The editor computes in a worker process of its own, so that the notebook's kernel never
waits on it, in marimo and in Jupyter alike:

- **A time limit.** A run, or a part computed, may take 30 seconds, or what
  `Editor(time_limit=...)` gives it. The statement over it is stopped and says so, the
  statements before it keep their results, and those after it did not run.
- **Stale runs stop.** A run that more typing makes stale is interrupted, as Ctrl-C would
  interrupt it.
- **A computation that does not stop** when asked, as one in a library's compiled code
  may not, has its worker stopped five seconds later, and the next run starts another.
- **Memory.** The worker may use 4096 MiB, or `Editor(memory_limit=...)`, where the system
  enforces it, as Linux does. A computation that needs more fails, and the editor goes on.
- **Your workbook goes with it.** The worker is given the workbook, pickled, with what it
  has computed, whenever it has changed.
- **Python objects go as data.** `Editor(wb, scope=globals())`, or a dict of objects,
  gives its programs a scope; without one, the workbook's. The objects a program names
  are read in the notebook's process as each run starts, and go to the worker as what
  they become, arrays, numbers and expressions, whenever they have changed: never the
  scope itself. A Python function goes by name, where the worker can import it, as
  `numpy.hanning` or a function of an installed module.

Where no process can be started, as in a notebook running in the browser, the editor
computes in the notebook's own process, without a time limit; so it does for a workbook
holding data no other process can take, or a program that calls a function defined in
the notebook, which no other process can import, and says why under each run. On Windows,
a stale run or a run over its time stops its worker, as there is no Ctrl-C to send it.

### Typesetting

The editor typesets with Temml 0.14.0, which RelativisticPy carries, with its MIT licence,
so that it works offline and loads nothing from the network. Temml writes MathML, which
the browser draws in the system's font for mathematics: Cambria Math on Windows and STIX
Two Math on macOS. On Linux, install one, such as Noto Sans Math or Latin Modern Math;
without one, brackets around a matrix do not stretch.

## Later

- Right-clicking a part of a typeset result, rather than all of it.
- Notebooks exported as pages that compute in the reader's browser.

**Next:** [Jupyter and notebooks](jupyter.md), for Markdown notebooks and the command
line's `--render`, or [the Python API](python.md), for every method of a workbook.
