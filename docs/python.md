# Python API

Everything the language does is available from Python, and the language's
results are ordinary Python values.

## Workbook

```python
>>> from relativisticpy import Workbook
>>> wb = Workbook(workers=1)

```

| | |
|---|---|
| `wb.expr(source)` | run a program; the value of its last expression, or `None` |
| `wb.exe(source)` (also `wb.run`) | run a program; the value of every expression, as a list |
| `wb.each(source)` | run a program one statement at a time, carrying on after errors; one `Step` per statement |
| `wb.run_file(path)` | run a file; in a Markdown file the `$$...$$` blocks are the program |
| `wb.check(document)` | run the equations of a LaTeX document or notebook and give a verdict on each: see [Checking papers](checking.md) |
| `wb.markdown(text)` | run a Markdown document's `$$...$$` blocks and typeset the results after each |
| `wb[name]` | the value bound to a name |
| `wb.metric`, `wb.coordinates` | the metric in force (a `Metric`) and the coordinates |
| `wb.ir(source)` | the IR a program compiles to, as text |
| `wb.tokens(source)`, `wb.parse(source)` | the tokens and syntax tree |
| `wb.reset()` | forget every definition |

`workers` sets how many processes share component-wise work such as
simplification (`None` for one per CPU). A workbook is a context manager, which
stops its workers on exit.

```python
>>> wb.exe(r"""
...     Coordinates := [x, y]
...     V^{a} := [y, -x]
...     V^{a}
... """)
[Tensor('^{a}', [y, -x])]
>>> wb["V"].components
[y, -x]

```

`each` is how a document can be run as a whole even where some of it fails:

```python
>>> for step in Workbook().each("x := 2; R_{ab}; x + 1"):
...     print(step.error.message if step.error else step.values)
[]
the Ricci tensor needs coordinates
[3]

```

Errors in syntax are raised before anything runs. Each `Step` has the
statement's `span`, its `values` (an empty list for a definition) or its
`error`, and for an equation the values of its two `sides`.

## Documents

`relativisticpy.documents.Document` reads a LaTeX document for its equations:
`Document.read(path)` or `Document(text, name)`. Its `equations()` are the
display equations in order, each with its `source` (the macros expanded, the
position that of the file), its `environment` and its `section`. `macros` are
the macros it defines.

## Tensor

A `Tensor` has `components` (a SymPy array whose axes follow `indices`),
`indices`, `coordinates` and an optional `name`. Arithmetic follows index
notation: `+` aligns operands by index name, and `*` sums over every index that
appears once up and once down.

```python
>>> from sympy import symbols
>>> from relativisticpy import Tensor
>>> t, x = symbols("t x")
>>> eta = Tensor([[-1, 0], [0, 1]], "_{a b}", (t, x), name="eta")
>>> v = Tensor([2, 3], "^{b}", (t, x))
>>> eta * v
Tensor('_{a}', [-2, 3])
>>> eta * v * v.relabel("^{a}")
5

```

`tensor.nonzero()` maps coordinate names to the nonzero components,
`tensor.latex()` typesets it (as does Jupyter), and `simplify`, `subs` and
`applyfunc` act on every component.

## Metric

A `Metric` is the tensor \(g_{\mu\nu}\) that knows its geometry. Each quantity is
computed when first asked for, then cached:

```python
>>> from sympy import diag, sin
>>> from relativisticpy import Metric
>>> theta, phi, a = symbols("theta phi a", positive=True)
>>> sphere = Metric(diag(a**2, a**2 * sin(theta)**2), (theta, phi))
>>> sphere.ricci_scalar
2/a**2
>>> sphere.christoffel.nonzero()
{('theta', 'phi', 'phi'): -sin(theta)*cos(theta), ('phi', 'theta', 'phi'): cos(theta)/sin(theta), ('phi', 'phi', 'theta'): cos(theta)/sin(theta)}

```

| | |
|---|---|
| `inverse`, `determinant` | \(g^{\mu\nu}\) and \(\det g_{\mu\nu}\) |
| `christoffel`, `riemann`, `ricci`, `einstein`, `weyl` | the tensors, with indices `^{a}_{b c}`, `^{a}_{b c d}`, `_{a b}`, `_{a b}`, `_{a b c d}` |
| `ricci_scalar`, `kretschmann` | the scalars |
| `line_element` | \(ds^2\) in the differentials `dt`, `dr`, ... |
| `geodesic_equations(parameter)` | the geodesic equations |
| `Metric.from_line_element(expr, coordinates)` | a metric from \(ds^2\) |

A metric is checked on construction: it must be square, symmetric and
non-degenerate.

## Spacetimes

`relativisticpy.spacetimes` has `minkowski`, `schwarzschild`,
`reissner_nordstrom`, `kerr`, `kerr_newman`, `flrw`, `de_sitter` and
`anti_de_sitter`, each returning a `Metric`. Parameters are names, numbers or
SymPy expressions, and `G` and `c` default to one:

```python
>>> from relativisticpy import spacetimes
>>> spacetimes.schwarzschild("M", G="G", c="c").kretschmann
48*G**2*M**2/(c**4*r**6)

```

## Errors

Every error derives from `RelativisticPyError` and, when it comes from a
program, carries the `span` of source text responsible:

| | |
|---|---|
| `ParseError` | the text is not valid RelativisticPy |
| `CompileError` | a statement makes no sense: an unknown tensor, a metric without coordinates |
| `IndexNotationError` | a `CompileError` breaking the rules of index notation |
| `EvaluationError` | a computation failed, such as inverting a degenerate metric |

## LaTeX output

`relativisticpy.latex(value)` typesets any result; tensors, metrics and tensor
equations also display themselves in Jupyter.

```python
>>> from relativisticpy import latex
>>> latex(wb.expr(r"\frac{x^2}{2}"))
'\\frac{x^{2}}{2}'

```
