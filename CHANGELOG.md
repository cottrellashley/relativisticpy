# Changelog

Notable changes to RelativisticPy. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## 2.0.0 - Unreleased

A rewrite of the whole project around its first idea: equations written as they
are in papers, in LaTeX or in plain text, with an engine that computes them.

### Added

- A language that reads LaTeX, plain text and Unicode mathematics alike, and
  mixes them freely: paste an equation from a paper and it runs.
- Checking papers: `relativisticpy check paper.tex` and `Workbook.check` run
  the display equations of a LaTeX document, or a Markdown notebook, in order
  and give each a verdict: holds, differs, relates, ill-formed, defines,
  computes, not computed, not understood or not parsed. An equation holds when it is proven; one that
  is not differs only where numbers show its sides to differ. Each document is
  checked in a workbook of its own, nothing in it stops the check, and
  `--strict` makes the command fail when an equation differs or cannot be read.
  `relativisticpy.documents` finds a document's equations and expands its
  macros as TeX does, including those of local packages and `\let`, keeping
  every position the file's own.
- The notation of papers: plain TeX (`{a \over b}`, `{\rm d}`, `\matrix`),
  closing punctuation, equations separated by commas and wide spaces, chains of
  relations and continued alignment rows, relations that are read but not
  computed (`\sim`, `\approx`, `\propto`, `\to`, `\in`), ellipses, primed and
  subscripted index labels, and labels on symbols such as `r_+` and `r^*`.
- The context physicists leave implicit: coordinates named by a line element,
  functions written once with their arguments (so that `A'` is \(dA/dr\) once
  `A(r)` is written), numbered indices, and repeated derivatives such as
  `\partial_0^2` and `\partial^2_{tx}`. Every statement of a display is LaTeX,
  `\text{and}` separates equations, `299\,792\,458` is one number, `n!!` is the
  double factorial, and `E^{(2)}` names an order of perturbation.
- The d'Alembertian `\Box`, also written `\nabla^2`, the Lie derivative
  `\pounds_V`, the wedge product of forms, and `\pm`.
- Tensor equations solved as tensor equations: an equation with a tensor that
  has no components, such as \(T_{\mu\nu}\) in Einstein's equations, or any
  tensor equation without a metric, is kept as index notation
  (`relativisticpy.abstract`). `solve` (with trace reversal), `trace`,
  `simplify`, `subs`, `LHS` and `RHS` work on it, using canonical forms under
  the tensors' symmetries and the identities of General Relativity; `holds()`
  says whether it is an identity; and with a metric in force, a tensor defined
  from it has components. Checking a paper verifies its identities in index
  notation.
- Index notation beyond spacetime. An index ranges over a space, declared in the
  notation (`\mathcal{A} := \mathbb{R}^{500}`, `a, b \in \mathcal{A}`) or brought
  by data, and a tensor's type records the space of each axis. Data is bound from
  Python, `wb["R_{ta}"] = frame`, from NumPy, pandas (whose labels name the
  components), SciPy sparse matrices or SymPy, and results give it back with
  `to_numpy` and `to_pandas`. Numerical components are computed with NumPy:
  contractions with `numpy.einsum` in their cheapest order, in slices when too large
  to hold, and as matrix products for sparse matrices.
- Statistics as contractions with a measure: `\mathbb{E}_t[...]`,
  `\operatorname{Var}_t`, `\operatorname{Cov}_t` and the Ledoit–Wolf covariance from
  scikit-learn, each one contraction with what it averages, and weighted by
  `Workbook.space(..., measure=...)`.
- Linear algebra on tensors: inverses, `(\Sigma^{-1})^{ab}`, computed as linear
  solves, by Cholesky factorisation where they can be; `eig` and `spec`, relative to
  a space's metric for a bilinear form, with the few largest or smallest
  eigenvalues of a sparse matrix; `svd`, `det`, `tr`, norms, `diag`, the
  elementwise product `\odot`, and functions of a map such as `\exp(-i H t)`. Any
  space can be given a metric, `g_{ab} := \Sigma_{ab}`; where its metric is the
  identity, an index may be summed in any position.
- JAX: `Workbook(backend="jax")` computes with JAX arrays, and `Workbook.grad`
  differentiates a scalar with respect to bound data, through contractions and
  solves. New extras: `relativisticpy[data]` (pandas and scikit-learn) and
  `relativisticpy[jax]`.
- Blackboard bold is a name of its own (`\mathbb{R}`, ℝ, 𝔼, 𝟙), names listed before
  `\in` are one declaration, `A \odot B` is the elementwise product, and `y_t`
  indexes `y` where `y` is a tensor.
- `Workbook.each`, which runs a program one statement at a time and carries on
  after errors.
- One way results look, everywhere: in Unicode at a terminal, and as text, LaTeX,
  HTML or data (`relativisticpy.display`), the same in Python, at the command line
  and in notebooks. Every equation answers `holds()`. `:=` always defines; `=`
  defines a name that has no meaning yet, and is otherwise a claim, with a note
  saying which it was.
- `relativisticpy help [TOPIC]` prints the documentation offline, and the language
  reference, generated from the language's own tables, is there even without it.
  `--format unicode|text|latex|json` chooses how results are written, and the exit
  status says what happened: 0, 1, 2, 130 after Ctrl-C and 143 after SIGTERM.
- An MCP server for AI agents, `relativisticpy mcp`: tools to compute in sessions,
  verify an equation, check a paper and read the documentation, which is also
  served as resources, and prompts to verify a derivation and check a paper.
  Programs run in worker processes with limits of time and memory, and files are
  read only where `--allow-files` allows (`pip install "relativisticpy[mcp]"`).
- Live mathematics in notebooks, `relativisticpy.live`. Tensors, metrics, equations,
  index notation and geodesics show themselves typeset in marimo and Jupyter, through
  `relativisticpy.display`. `show(source)` runs a program and shows each statement
  typeset with its results, its notes and its errors, in place, as `--render` does, but
  as HTML. `Editor` is a widget for marimo, Jupyter and VS Code that computes a program
  as it is typed, from its first changed statement, in a worker process with a time
  limit, in a copy of the workbook it is given; it underlines each error where it is,
  computes a part of a statement selected, from a menu, and typesets offline with Temml
  (`pip install "relativisticpy[live]"`).
- `Workbook.copy`, a workbook in the same state whose definitions are its own from then
  on.
- A curriculum of mathematics with known answers, `curriculum/`: 407 cases across
  18 domains at five levels, from notation to the frontier, each a program and an
  oracle that computes its answer independently. `benchmarks/curriculum.py` runs
  it, and the test suite keeps every case that passes passing.
- A corpus benchmark, `benchmarks/corpus.py`, measuring how much of real
  papers RelativisticPy reads and what checking them finds.
- Index notation in full: Einstein summation, raising and lowering with the
  metric, traces, partial and covariant derivatives in every notation
  (`\partial_\mu`, `\nabla_\mu`, commas and semicolons), symmetric and
  antisymmetric parts, including brackets that span tensors, and components
  named by coordinate (`\Gamma^{r}_{\theta\theta}`, `g_{tt}`).
- Checks of index notation before anything is computed, and errors that point at
  the source, with a hint where one helps.
- An intermediate representation. Programs compile to a typed graph that is
  computed lazily, shares common subexpressions, remembers every value, and can
  spread component-wise work over processes with `Workbook(workers=N)`. Every
  operation in it is named, and looked up in a registry of implementations as it
  runs, so the graph holds no Python functions, and nothing is computed while
  compiling.
- The Weyl tensor and the Kretschmann scalar as built-ins.
- Known spacetimes: Minkowski, Schwarzschild, Reissner–Nordström, Kerr,
  Kerr–Newman, FLRW, de Sitter and anti-de Sitter, each in geometric units or
  with `G` and `c` after its own parameters. A decimal in a metric is the exact
  fraction it denotes, so curvature stays exact.
- Metrics from line elements, `ds^2 = ...`, including `d\Omega^2`.
- Fast, exact simplification. Whether a component vanishes is decided without
  polynomial GCDs, so the Kerr metric is verified to be a vacuum solution in
  seconds.
- The Python classes `Metric`, `Tensor` and `TensorEquation`, and the module
  `relativisticpy.spacetimes`.
- `relativisticpy.numeric`: NumPy functions from symbolic results, and geodesics
  of particles and light integrated with SciPy
  (`pip install "relativisticpy[numeric]"`).
- The `relativisticpy` command, which runs a program or a Markdown notebook,
  typesets the results with `--render`, and has an interactive prompt.
- Documentation at [relativisticpy.org](https://relativisticpy.org), and worked
  examples. Every example in both runs as part of the test suite.

### Changed

- Arithmetic is exact: `1/2` is one half, not `0.5`.
- `Workbook.exe` always returns a list, the value of every expression in the
  program; `Workbook.expr` returns the last.
- Errors are raised as exceptions (`ParseError`, `CompileError`,
  `IndexNotationError`, `EvaluationError`, `UnknownNameError`, `UsageError`,
  `DataError`, `FileError`, `MetricError`, `NotComputedError` and
  `InternalError`) rather than returned as strings. Each says where it happened,
  with notes and a hint, and is shown as a compiler shows a diagnostic
  (`render()`) or given as data (`to_dict()`).
- `Workbook.markdown` takes the text of a document rather than a path, and
  returns the document with the results typeset; `Workbook.run_file` runs a
  file.
- A bare `g` is the determinant of the metric, and `R` is the Riemann tensor,
  the Ricci tensor or the Ricci scalar according to its number of indices,
  unless the metric is written with them as symbols.
- `x = 5` defines `x`, but an equation about what already has a meaning is a
  claim to check: a component (`g_{tt} = \ldots`), a built-in scalar
  (`R = 0`), a bound tensor, or \(\pi\). An equation for a coordinate in force,
  as `t = 0`, is a condition, which is read but not computed.
- In LaTeX, a run of letters is a product of symbols (`2GM` is `2 G M`); in
  plain text it is one name.
- In arithmetic, lists are vectors and matrices: `2 [1, 2]` is `[2, 4]`.
- An equation that is an identity, such as `e^{2a}(1 + b) = e^{2a} + b e^{2a}`,
  evaluates to `True`, decided exactly.
- `\[ ... \]` and `$$ ... $$` hold one formula, however their lines break.
- RelativisticPy needs Python 3.10 or later. SymPy is its only required
  dependency.

### Removed

- The modules of version 1 (`relativisticpy.core`, `relativisticpy.gr`,
  `relativisticpy.interpreter`, `relativisticpy.symengine` and others). The
  package's entry point is still `relativisticpy.Workbook`.

## 1.2.9 and earlier

The original implementation, whose history is in the git log.
