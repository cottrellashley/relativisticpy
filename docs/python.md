# Python API

> **For:** programmers and agents who use RelativisticPy from Python. **You will learn:**
> the workbook, every kind of result, whether a claim holds, how results are shown, and
> errors. **Before this:** [Getting started](getting-started.md).

Everything the language does is available from Python, and the language's
results are ordinary Python values. This page is the reference: the
[Workbook](#workbook), then what each result is, how to ask whether a claim
holds, and how results are shown. Write LaTeX in raw strings, `r"..."`, and see
[the language reference](reference.md) for what a program may say.

## Workbook

```python
>>> from relativisticpy import Workbook
>>> wb = Workbook(workers=1)

```

| Method | What it does |
|---|---|
| `wb.expr(source)` | run a program; the value of its last expression, or `None` |
| `wb.exe(source)` (also `wb.run`) | run a program; the value of every expression, as a list |
| `wb.each(source)` | run a program one statement at a time, carrying on after errors; one `Step` per statement |
| `wb.run_file(path)` | run a file; in a Markdown file the `$$...$$` blocks are the program |
| `wb.check(document)` | run the equations of a LaTeX document or notebook and give a verdict on each, a `Report`: see [Checking papers](checking.md). A string is a path; pass LaTeX text as `Document(text, name)` |
| `wb.markdown(text)` | run a Markdown document's `$$...$$` blocks and typeset the results after each |
| `wb[name]` | the value bound to a name |
| `wb["R_{ta}"] = data` | bind data: a NumPy array, pandas DataFrame or Series, SciPy sparse matrix, SymPy array or list (see [Beyond spacetime](spaces.md)) |
| `wb.space(name, dim, basis=, labels=, field=, measure=)` | declare a space for indices to range over, with its basis, labels, field (`"real"` or `"complex"`) and the weights of its expectations; returns the `Space` |
| `wb.grad(source, wrt)` | the gradient of a scalar with respect to bound data, by automatic differentiation with JAX |
| `wb.metric`, `wb.coordinates` | the metric in force (a `Metric`) and the coordinates |
| `wb.copy()` | a workbook in the same state, whose definitions are its own from then on; it shares what has been computed |
| `wb.ir(source)` | the IR a program compiles to, as text |
| `wb.explain(source)` | how each statement is read, and why: see [How a statement is read](#how-a-statement-is-read) |
| `wb.tokens(source)`, `wb.parse(source)` | the tokens and syntax tree |
| `wb.reset()` | forget every definition |

`workers` sets how many processes share component-wise work such as
simplification (`None` for one per CPU). `backend="jax"` keeps bound data as JAX
arrays, so that it is computed with JAX. A workbook is a context manager, which
stops its workers on exit.

```python
>>> wb.exe(r"""
...     Coordinates := [x, y]
...     V^{a} := [y, -x]
...     V^{a}
... """)
[V^a in (x, y): 2 components non-zero
  V^x = y
  V^y = −x]
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
statement's `span` and `statement` (its syntax tree), its `values` (an empty list
for a definition) or its `error`, for an equation the values of its two `sides`,
and `notes` on how the statement was read where that could surprise. `zero`
says whether an equation's two sides are both zero.

## How a statement is read

`wb.explain(source)` says how each statement of a program is read, and why: how each
name was resolved, against what the workbook defines, its coordinates and the built-in
names; each ambiguous form decided, with why: a definition or a claim, notation or
components, a product or a call, an index or a power, a run of letters, a constant or a
symbol, a function named without its argument; and the notes given where the less
likely reading was taken. Nothing is computed, and nothing is defined: as with `wb.ir`,
the workbook is left as it was. Printed, the readings are text, with the values known
without computing shown as results are:

```python
>>> wb = Workbook()
>>> wb.exe(r"Coordinates := [t, r]" "\n" r"g_{\mu\nu} := diag(-(1 - 2M/r), 1)" "\n" "K := 2")
[]
>>> explanation = wb.explain(r"""
...     K = \frac{2GM}{r}
...     e^{i\pi} + \Gamma^t_{tr}
... """)
>>> print(explanation)
line 2: K = \frac{2GM}{r}
  definition or claim: K = ... is a claim, since K already has a value
  letters: GM is G M, since in LaTeX a run of letters that names nothing is a product
  K: a value, defined on line 3, 2
  G: a symbol
  M: a symbol the metric is written with
  r: a coordinate
  note: K already has a value, so = compares it; to give K a new value, write K := ...
line 3: e^{i\pi} + \Gamma^t_{tr}
  constant or symbol: e is a constant, since nothing defines e
  index or power: e^{i\pi} is a power, since e is not a tensor
  constant or symbol: i is a constant, since nothing defines i
  constant or symbol: \pi is a constant, since nothing defines \pi
  index or power: \Gamma^t_{tr} is a tensor with its indices, since \Gamma is the Christoffel connection
  e: a constant, e
  i: a constant, i
  \pi: a constant, π

```

Each reading is a `Reading`, with the statement's `span`; its `names`, each a
`Resolution` with what is `written`, its `meaning` and, where it is known without
computing, its `value`; its `decisions`, each a `Decision` with its `kind`, what is
`written`, the `reading` taken, `why`, and the `note` given, if any; its `notes`; and
the `error` the statement gives, if it gives one, without stopping the readings after
it:

```python
>>> reading = explanation[0]
>>> [(decision.kind, decision.reading) for decision in reading.decisions]
[('definition or claim', 'a claim'), ('letters', 'G M')]
>>> reading.names[0].value
2
>>> print(wb.explain("T_{ab} + v_a")[0].error.message)
T is not a tensor

```

## Results

What a program gives back depends on what it computes:

| Result | Is | Holds? | Shown |
|---|---|---|---|
| a number or an expression | a SymPy value: `48*M**2/r**6` | not a claim | `48M²/r⁶` |
| a claim of scalars | `True`, `False`, or SymPy's `Eq`, `Ne`, `<`, ... | `holds(value)` | `x² + y² = 1` |
| a tensor with components | a [`Tensor`](#tensor) | not a claim | its name, then a line for each component |
| a metric | a [`Metric`](#metric), a `Tensor` | not a claim | its line element |
| an equation of tensors | a [`TensorEquation`](#tensorequation) | `.holds()`, `bool()` | whether it holds, and where not |
| index notation | an [`IndexEquation` or `IndexExpression`](#index-notation) | `.holds()`, `bool()` | the notation |
| several values | a list | | |
| a statement of a document | a [`Step`](#workbook), from `wb.each` | | |
| a checked document | a [`Report`](#checking-papers) of `Finding`s, from `wb.check` | `Verdict`s | |
| a geodesic | a [`Geodesic`](#geodesics) or the `geodesic_equations` | | its span, or an equation for each coordinate |

## Does it hold?

`relativisticpy.holds(value)` answers for any claim: `True`, `False`, or `None`
where it is not decided, as when SymPy cannot show the sides equal and numbers
cannot show that they differ. Equations of tensors and of index notation have
the same answer as a method, and `bool()` agrees with it:

```python
>>> from relativisticpy import holds
>>> vacuum = Workbook()
>>> vacuum.expr(r"""
...     Coordinates := [t, r, \theta, \phi]
...     g_{\mu\nu} := Schwarzschild(M)
... """)
>>> holds(vacuum.expr(r"R_{\mu\nu} = 0")), bool(vacuum.expr(r"R_{\mu\nu} = 0"))
(True, True)
>>> vacuum.expr(r"\Gamma^{a}_{bc} = 0").holds()
False
>>> holds(vacuum.expr(r"\Gamma^{r}_{\theta\theta} = -r"))   # 2M - r is not -r
False
>>> holds(Workbook().expr(r"G_{\mu\nu} = 8\pi G T_{\mu\nu}"))   # a law, not an identity
False

```

Where an equation is not decided, `bool()` is an `UndecidedError`, a
`RelativisticPyError` and a `TypeError`, rather than a guess. A program that states
such a claim gets the same error where the claim is written, rather than the claim
shown as if it were an answer. A value that is no claim, such as `48*M**2/r**6`, is a
`TypeError` for `holds`.

## How results look

Every result is shown in four forms, by `relativisticpy.display`, the one
printer that the command line, the prompt, notebooks, `repr` and the report of
a check all use:

| Function | What it gives |
|---|---|
| `display.unicode(value)` | mathematics for a terminal: `48M²/r⁶`; it is what `repr` shows |
| `display.text(value)` | ASCII in the language's own syntax: `48*M^2/r^6` |
| `display.latex(value)` | LaTeX for MathJax, KaTeX and papers: `\frac{48 M^{2}}{r^{6}}` |
| `display.data(value)` | a JSON-serialisable description: its `kind`, its three forms and its structure |
| `display.html(value)` | HTML for notebooks: numerical data as a table |

```python
>>> from relativisticpy import display
>>> K = vacuum.expr(r"R_{abcd} R^{abcd}")
>>> display.unicode(K), display.text(K), display.latex(K)
('48M²/r⁶', '48*M^2/r^6', '\\frac{48 M^{2}}{r^{6}}')
>>> display.data(K)["kind"]
'expression'

```

The text and LaTeX forms read back: the language parses them as the value they
show, and so does the Unicode form wherever the language reads Unicode.
Euler's number is `e` and the imaginary unit `i`, so that a symbol `E` is never
shown as either; numbers are shown as written, and an integer too long to read
is shortened, with its number of digits:

```python
>>> display.unicode(Workbook().expr("E + e")), display.unicode(Workbook().expr("0.1 + 0.2"))
('E + e', '0.30000000000000004')
>>> display.unicode(Workbook().expr("10000!"))
'2.846259680917054…×10³⁵⁶⁵⁹ (an integer of 35,660 digits)'
>>> display.text(Workbook().expr(r"\lambda + \beta x + M_\odot"))
'M_odot + \\beta*x + \\lambda'
>>> display.latex(Workbook().expr(r"M_\odot + \rho_{\rm vac}"))
'M_{\\odot} + \\rho_{\\mathrm{vac}}'

```

A result that RelativisticPy does not know shows itself in a form by a method
`_display_unicode`, `_display_text`, `_display_latex` or `_display_data`, where
it has one.

## Tensor

A `Tensor` has `components` (a SymPy array whose axes follow `indices`, or a
numerical array), `indices`, `coordinates`, an optional `name`, the `spaces` its
axes range over, and the `source` it is the value of, as written, where it has no
name of its own. Arithmetic follows index notation: `+` aligns operands by index
name, and `*` sums over every index that appears once up and once down. A
tensor is shown by its name and indices, then a line for each non-zero
component, labelled by coordinate; components equal, or opposite, by an
exchange of indices share a line:

```python
>>> from sympy import symbols
>>> from relativisticpy import Tensor
>>> t, x = symbols("t x")
>>> eta = Tensor([[-1, 0], [0, 1]], "_{a b}", (t, x), name="eta")
>>> v = Tensor([2, 3], "^{b}", (t, x), name="v")
>>> eta * v
η_{ab} v^b in (t, x): 2 components non-zero
  η_{tb} v^b = −2
  η_{xb} v^b = 3
>>> eta * v * v.relabel("^{a}")
5
>>> vacuum.expr(r"\Gamma^{a}_{bc}")
Γ^a_{bc} in (t, r, θ, φ): 13 of 64 components non-zero
  Γ^t_{tr} = Γ^t_{rt} = M/(r (r − 2M))
  Γ^r_{tt} = M (r − 2M)/r³
  Γ^r_{rr} = M/(r (2M − r))
  Γ^r_{θθ} = 2M − r
  Γ^r_{φφ} = (2M − r) sin(θ)²
  Γ^θ_{rθ} = Γ^θ_{θr} = 1/r
  Γ^θ_{φφ} = −sin(θ) cos(θ)
  Γ^φ_{rφ} = Γ^φ_{φr} = 1/r
  Γ^φ_{θφ} = Γ^φ_{φθ} = cos(θ)/sin(θ)
>>> vacuum.expr(r"R_{\mu\nu}")
R_{μν} = 0

```

Each line reads back as a claim: `Γ^t_{tr} = M/(r (r − 2M))` is true in this
workbook. Components are taken by position or by name, and a slice is the
tensor of the components it leaves free:

```python
>>> gamma = vacuum.expr(r"\Gamma^{a}_{bc}")
>>> gamma["r", "t", "t"] == gamma[1, 0, 0]
True
>>> gamma["θ", "φ", "φ"]
-sin(theta)*cos(theta)
>>> gamma["r", "t", :]
Γ^r_{tc} in (t, r, θ, φ): 1 of 4 components non-zero
  Γ^r_{tt} = M (r − 2M)/r³

```

| Attribute or method | What it is |
|---|---|
| `T.rank` (also `ndim`), `T.shape`, `T.size`, `T.dim` | the number of indices, of components along each axis, of components, and along every axis where they agree |
| `T[...]` | a component by position or by name (`G['r', 't', 't']`, `w['MSFT']`), or with a slice the tensor of those left free |
| `T.nonzero()` | the non-zero components, keyed by the names of their coordinates or basis elements, or by position |
| `T.labels(axis)` | the names of the components along an axis |
| `T.simplify()`, `T.subs(...)`, `T.diff(x)`, `T.applyfunc(f)` | act on every component |
| `T.relabel("_{c d}")` | the same components under new labels |
| `T.T` | the transpose: the axes and their indices in the other order |
| `T.free_symbols`, `T.dtype` | the symbols of the components; NumPy's type of numerical ones |
| `T.numerical`, `T.to_numpy(**values)`, `T.to_pandas()` | whether the components are numbers; them as NumPy or pandas data, labelled by their spaces |
| `T.latex()` | the LaTeX of `display.latex`; `sympy.latex(T)` gives it too |
| `len(T)`, `iter(T)` | the first axis, as for a NumPy array |

`sympy.sympify(T)` is the array of components, `numpy.asarray(T)` an array of
them, and NumPy's functions keep a numerical tensor's indices: `numpy.exp(A)` is a
tensor. Symbolic components need values to become numbers, and say which:

```python
>>> g = vacuum.expr(r"g_{\mu\nu}")
>>> g.to_numpy(M=1, r=4, theta=0.5)[0, 0]
np.float64(-0.5)

```

Components may also be numbers in an array, as data and what is computed from
it are. Each axis then ranges over one of `tensor.spaces`. Large data is
summarised, never dumped, and notebooks show a table labelled by the spaces'
bases:

```python
>>> import numpy as np
>>> book = Workbook()
>>> book["w^a"] = np.array([0.5, 0.5])
>>> book["w"].to_numpy()
array([0.5, 0.5])
>>> book["A_{ij}"] = np.arange(250000.0).reshape(500, 500)
>>> print(display.unicode(book["A"]).splitlines()[0])
A_{ij}: 500 × 500 float64

```

## Metric

A `Metric` is the tensor \(g_{\mu\nu}\) that knows its geometry, and is shown by
its line element. Each quantity is computed when first asked for, then cached:

```python
>>> from sympy import diag, sin
>>> from relativisticpy import Metric
>>> theta, phi, a = symbols("theta phi a", positive=True)
>>> sphere = Metric(diag(a**2, a**2 * sin(theta)**2), (theta, phi))
>>> sphere
g_{μν} in (θ, φ): ds² = a² dθ² + a² sin(θ)² dφ²
>>> sphere.ricci_scalar
2/a**2
>>> sphere.christoffel.nonzero()
{('theta', 'phi', 'phi'): -sin(theta)*cos(theta), ('phi', 'theta', 'phi'): cos(theta)/sin(theta), ('phi', 'phi', 'theta'): cos(theta)/sin(theta)}

```

| Attribute or method | What it is |
|---|---|
| `inverse`, `determinant` | \(g^{\mu\nu}\) and \(\det g_{\mu\nu}\) |
| `christoffel`, `riemann`, `ricci`, `einstein`, `weyl` | the tensors, with indices `^{a}_{bc}`, `^{a}_{bcd}`, `_{ab}`, `_{ab}`, `_{abcd}` |
| `ricci_scalar`, `kretschmann` | the scalars |
| `line_element` | \(ds^2\) as a SymPy expression in the differentials `dt`, `dr`, `dtheta`, ... |
| `line_element_text(form)` | the line element as a paper writes it, in a form of `display` |
| `geodesic_equations(parameter)` | the geodesic equations, one for each coordinate |
| `Metric.from_line_element(expr, coordinates)` | a metric from \(ds^2\) |

A metric is checked on construction: it must be square, symmetric and
non-degenerate. Its LaTeX is its line element, which the language reads back.

## TensorEquation

An equation between tensors with components, such as the field equations of a
metric, is a `TensorEquation`. It says whether it holds and, if not, where; each
component equal to another by a symmetry of both sides is counted once:

```python
>>> fails = vacuum.expr(r"R_{\mu\nu} = g_{\mu\nu}")
>>> fails.holds(), fails.analysis().failing, fails.analysis().total
(False, 4, 10)
>>> fails.failures()[("t", "t")]
(0, 2*M/r - 1)

```

It is shown by its `source`, the equation as written, which names its sides:

```python
>>> from sympy import Array
>>> from relativisticpy import Indices, TensorEquation
>>> M, r = symbols("M r")
>>> TensorEquation(Array([[0, 0], [0, 0]]), Array([[2*M/r - 1, 0], [0, r/(r - 2*M)]]),
...                Indices(r"_{\mu\nu}"), (t, r), source=r"R_{\mu\nu} = g_{\mu\nu}")
R_{μν} = g_{μν}: differs in 2 of 3 components
  R_{tt} = 0 but g_{tt} = 2M/r − 1
  R_{rr} = 0 but g_{rr} = r/(r − 2M)

```

| Attribute or method | What it is |
|---|---|
| `E.holds()`, `bool(E)` | whether every component holds, as above |
| `E.failures()` | the components where the sides are not shown equal, keyed by coordinate names, each its two sides |
| `E.components()` | the same as SymPy equations, keyed by position |
| `E.equations()` | the equations of those components, in order, to `solve` |
| `E.analysis()` | how many independent components there are, and which fail |
| `E.lhs`, `E.rhs`, `E.indices`, `E.coordinates`, `E.spaces` | the sides' components, and what they range over |
| `E.source` | the equation as written, by which its components are named when shown |
| `E.latex()`, `E.subs(...)` | LaTeX, with `\neq` where it fails; the equation with values put in |

Numerical sides hold where they agree to rounding error. An equation of large
data is summarised, with its largest difference, in milliseconds.

## Index notation

An equation between tensors that have no components, such as
`G_{\mu\nu} = 8\pi G T_{\mu\nu}`, is an `IndexEquation`, and an expression an
`IndexExpression` (see [Solving tensor equations](notation.md)). They are
`relativisticpy.abstract` values, built on `sympy.tensor.tensor`:

| Attribute or method | What it is |
|---|---|
| `E.lhs`, `E.rhs`, `X.value` | the sides of an equation, the value of an expression: SymPy tensor expressions |
| `E.holds()`, `bool(E)` | whether an equation is an identity |
| `X == 0`, `X == F` | whether an expression vanishes, or equals a scalar or another expression |
| `X.is_zero`, `X.holds()` | whether an expression vanishes; as for any number, `bool(X)` is whether it does not |
| `E == F` | whether two equations say the same |
| `E.latex()`, `str(E)` | LaTeX; `repr(E)` and `E.display(form)` the forms of `display` |
| `E.algebra` (also `E.space`) | the `Algebra` they belong to |

```python
>>> efe = Workbook().expr(r"G_{\mu\nu} = 8\pi G T_{\mu\nu}")
>>> efe
IndexEquation(G_{μν} = 8π G T_{μν})
>>> efe.holds()
False
>>> print(efe.rhs)
8*pi*G*T(-mu, -nu)
>>> Workbook().expr(r"simplify(R_{\rho\sigma\mu\nu} + R_{\rho\mu\nu\sigma} + R_{\rho\nu\sigma\mu})") == 0
True

```

`relativisticpy.abstract` also offers the algebra on its own: `canonical`,
`solve`, `trace` and `substitute`. Each works in an `Algebra`, the tensors of a
computation in a given number of dimensions: their indices, the metric, the
Kronecker delta, and a head for every tensor by name and rank.

```python
>>> from relativisticpy.abstract import Algebra, canonical, trace
>>> algebra = Algebra(dim=3)
>>> a, b = algebra.index("a", upper=True), algebra.index("b")
>>> canonical(algebra.delta(a, -a), algebra)
3
>>> trace(algebra.head("R", 2, "ricci")(a, b), algebra)
R
>>> algebra
Algebra(dim=3, metric g, Ricci scalar R, 1 tensor: R)

```

## Space

A `Space` is what an index ranges over: its `name`, its dimension `dim`, the
names of its `basis` if it has them, its `field` (`"real"` or `"complex"`),
whether it is `spacetime`, and whether it was `inferred` from the shape of data
rather than declared. `position(name)` is the number of a basis element:

```python
>>> from relativisticpy import Space
>>> assets = Space("assets", 3, basis=("AAPL", "MSFT", "NVDA"))
>>> assets.position("MSFT"), assets.describe()
(1, 'space assets: ℝ³, basis (AAPL, MSFT, NVDA)')

```

## Checking papers

`wb.check(document)` gives a `Report` (from `relativisticpy.checking`, and
exported by `relativisticpy`) with the document's `name`, the number of its
display `equations`, and its `findings`, in the order of the document:

| Attribute or method | What it is |
|---|---|
| `report.findings` | one `Finding` for each statement, and for each link of a chain |
| `report.count(verdict)` | how many findings met a `Verdict` |
| `report.summary()`, `report.describe()` | how many statements met each verdict; and the whole report, as text |
| `report.ok`, `report.failed(verdicts)` | whether the check passes `--strict`, or fails on the verdicts given |
| `report.to_dict()` | the whole report as data, for JSON |
| `finding.verdict`, `finding.span`, `finding.detail` | what became of a statement, where it was written, and why |
| `finding.line`, `finding.location`, `finding.describe()` | its line, its place as `paper.tex:12:5`, and the finding on one line with its detail beneath |

A `Verdict` is one of `HOLDS`, `DIFFERS`, `RELATES`, `ILL_FORMED`, `DEFINES`,
`COMPUTES`, `NOT_COMPUTED`, `NOT_UNDERSTOOD` and `NOT_PARSED`; its `value` is its
name in words, its `mark` the sign the report shows (✓ ✗ ↔ ✗ = → ~ ? !) and its
`meaning` what it means. [Checking papers](checking.md#verdicts) explains each:

```python
>>> from relativisticpy import Document, Verdict
>>> report = Workbook().check(Document(r"\begin{equation} ds^2 = -dt^2 + a(t)^2 dx^2 \end{equation}"))
>>> [(finding.line, finding.verdict.value) for finding in report.findings]
[(1, 'defines')]
>>> Verdict.DIFFERS.mark
'✗'

```

## Geodesics

`relativisticpy.numeric.geodesic` returns a `Geodesic`, shown by its span and
the range of each coordinate, with `tau`, the coordinates `x` and velocities
`u` (one row per coordinate), each coordinate by name (`orbit["r"]`,
`orbit["θ"]`), `stopped`, and SciPy's `solution`. See [Numerics](numerics.md).
`metric.geodesic_equations()` gives a list of SymPy equations, shown one for
each coordinate:

```python
>>> from relativisticpy import spacetimes
>>> spacetimes.minkowski().geodesic_equations()
t: t''(τ) = 0
x: x''(τ) = 0
y: y''(τ) = 0
z: z''(τ) = 0

```

## Spacetimes

`relativisticpy.spacetimes` has `minkowski`, `schwarzschild`,
`reissner_nordstrom`, `kerr`, `kerr_newman`, `flrw`, `de_sitter` and
`anti_de_sitter`, each returning a `Metric`. Parameters are names, numbers or
SymPy expressions, and `G` and `c` default to one:

```python
>>> spacetimes.schwarzschild("M", G="G", c="c").kretschmann
48*G**2*M**2/(c**4*r**6)

```

## Errors

Every error a user can cause derives from `RelativisticPyError`, and carries its
`message`, the `span` of source text responsible (with its `line`, `column` and
`source.name`), a `hint` where there is a fix and `notes` where there is more to say.
`error.render()` gives the whole diagnostic, as the command line prints it, and
`error.to_dict()` the same as data:

| Error | What it means |
|---|---|
| `ParseError` | the text is not valid RelativisticPy |
| `CompileError` | a statement makes no sense: an unknown tensor, a metric without coordinates |
| `IndexNotationError` | a `CompileError` breaking the rules of index notation |
| `NotComputedError` | a `CompileError` for what is read but not computed, such as `\propto` or a condition |
| `EvaluationError` | a computation failed, such as inverting a degenerate metric |
| `MetricError` | a metric that cannot be one; also a `ValueError` |
| `DataError` | data from Python that cannot be bound as written; also a `ValueError` |
| `UnknownNameError` | `wb[name]` for a name nothing defines; also a `KeyError` |
| `UsageError` | a call from Python given what it cannot take; also a `TypeError` and a `ValueError` |
| `FileError` | a file that does not exist, or is not UTF-8; also an `OSError` |
| `UndecidedError` | `bool()` of an equation whose truth is not decided; also a `TypeError` |
| `InternalError` | a failure of RelativisticPy's own, never of the input, which invites a report |

```python
>>> from relativisticpy import RelativisticPyError
>>> try:
...     Workbook().expr(r"\frac{1}{2")
... except RelativisticPyError as error:
...     print(error.to_dict()["type"], error.to_dict()["line"])
ParseError 1

```

Some calls from Python raise Python's own errors for Python's mistakes: a
`KeyError` for a component name that is not a coordinate, and a `TypeError` for
components given after their indices.

## LaTeX output

`relativisticpy.latex(value)` is `display.latex`: it typesets any result, and
tensors, metrics, equations and geodesics also display themselves, typeset, in Jupyter
and marimo. Their hooks give `display.jupyter_latex(value)` and, for numerical data,
the table of `display.jupyter_html(value)` to Jupyter, and `display.marimo(value)` to
marimo.

```python
>>> from relativisticpy import latex
>>> latex(wb.expr(r"\frac{x^2}{2}"))
'\\frac{x^{2}}{2}'
>>> latex(vacuum.expr(r"R_{\mu\nu} = 0")).endswith(r"\tag*{holds in all 10 components}")
True

```

**Next:** [the language reference](reference.md), for what a program may say, or
[Jupyter and notebooks](jupyter.md), for how results display there.
