# Live mathematics: notebooks and the editor

Write mathematics in a notebook and see it computed as it is written. Point at any part
of it to compute that part. Bring in Python's objects by name. This is how RelativisticPy
works in marimo and Jupyter.

## What a notebook shows

- **A value shows itself typeset,** with nothing to call. In marimo the hook is `_mime_`
  or `_display_`. In Jupyter it is `_repr_latex_`, or `_repr_html_` for numerical
  tables. Every hook goes through `display`, so a value looks the same in a notebook, at
  the command line and in the MCP server.
- **`show(source, **objects)`** runs a program and shows each statement typeset with its
  results, its notes and its errors, as `relativisticpy --render` does, but as HTML.
  Objects are passed by keyword, as in `show(r"\hat{x} := \mathcal{F}[\texttt{signal}]",
  signal=signal)`. marimo tracks the Python names a cell uses, so the cell runs again
  whenever `signal` changes. A workbook given as `workbook=` keeps its definitions from
  one cell to the next.

## The editor

`Editor(workbook)` is a widget built with anywidget. It works in marimo
(`mo.ui.anywidget(editor)`), in Jupyter and in VS Code.

- **Source on the left, results on the right.** The source is LaTeX or plain text. Each
  statement is shown typeset beside its results.
- **Computed as it is typed.** After a pause of about 300 ms, the program runs in a
  worker, with a time limit. A run made stale by more typing is dropped. Only the
  statements from the first changed one onwards run again. Each run uses a copy of the
  session, so a half-typed definition never reaches it.
- **Errors in place.** The span of an error is underlined, and its message, notes and
  hint show on hover.
- **Point and compute.** Select part of a statement, or right-click a part of its
  typeset form, for a menu: compute, simplify, expand, series, does it hold, copy as
  LaTeX. The answer appears under the statement. Every node of the syntax tree has a
  span, so a selection is a span of the source: the part runs in the session as it is at
  that statement.
- **Its value is data.** The source, and each statement's results as `display.data`
  gives them, so that another cell can use them, and react when they change.

## Python objects

Objects come in by name, `\texttt{signal}`, from the scope a workbook is given, as
`design/integrations.md` describes. `show(..., **objects)` and `Editor(workbook,
scope=...)` give that scope. A checked document has no scope.

## Typesetting

The host typesets where it can: marimo with KaTeX, Jupyter with MathJax. The editor
typesets its own output, and it must work offline. So the widget ships what it needs
with the package; it never loads it from a CDN.

## Later

- **Pages with no server.** RelativisticPy is pure Python and SymPy runs in Pyodide. A
  marimo notebook exported to WebAssembly can therefore run the mathematics in the
  reader's browser, so a derivation can be shared as a page that computes.
- **A language server.** Errors, hovers and "compute this" in any editor, from the
  same functions the widget uses.

## Verification

- `show` and the display hooks are tested without a browser: the HTML they give, and
  the mime bundles.
- The editor's Python side is tested without a browser: from source to results, stale
  runs, errors with their spans, computing a selection.
- A browser test with Playwright runs a marimo notebook with the editor, types a
  program, and checks the typeset results and the right-click menu.
- `pip install relativisticpy` installs none of this. The widget needs the `live`
  extra, with anywidget. marimo and Jupyter are the user's own.
