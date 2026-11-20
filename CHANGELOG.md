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
  and give each a verdict: holds, differs, defines, computes, not computed,
  not understood or not parsed. `relativisticpy.documents` finds a document's
  equations and expands its macros, including those of local packages.
- The notation of papers: plain TeX (`{a \over b}`, `{\rm d}`, `\matrix`),
  closing punctuation, equations separated by commas and wide spaces, chains of
  relations and continued alignment rows, relations that are read but not
  computed (`\sim`, `\approx`, `\propto`, `\to`, `\in`), ellipses, primed and
  subscripted index labels, and labels on symbols such as `r_+` and `r^*`.
- The context physicists leave implicit: coordinates named by a line element,
  functions written once with their arguments, numbered indices, and repeated
  derivatives such as `\partial_0^2`.
- The d'Alembertian `\Box`, the Lie derivative `\pounds_V`, the wedge product of
  forms, and `\pm`.
- Tensor equations solved as tensor equations: an equation with a tensor that
  has no components, such as \(T_{\mu\nu}\) in Einstein's equations, or any
  tensor equation without a metric, is kept as index notation
  (`relativisticpy.abstract`). `solve` (with trace reversal), `trace`,
  `simplify`, `subs`, `LHS` and `RHS` work on it, using canonical forms under
  the tensors' symmetries and the identities of General Relativity; `holds()`
  says whether it is an identity; and with a metric in force, a tensor defined
  from it has components. Checking a paper verifies its identities in index
  notation.
- `Workbook.each`, which runs a program one statement at a time and carries on
  after errors.
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
  spread component-wise work over processes with `Workbook(workers=N)`.
- The Weyl tensor and the Kretschmann scalar as built-ins.
- Known spacetimes: Minkowski, Schwarzschild, Reissner–Nordström, Kerr,
  Kerr–Newman, FLRW, de Sitter and anti-de Sitter.
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
  `IndexNotationError`, `EvaluationError`) rather than returned as strings.
- `Workbook.markdown` takes the text of a document rather than a path, and
  returns the document with the results typeset; `Workbook.run_file` runs a
  file.
- A bare `g` is the determinant of the metric, and `R` is the Riemann tensor,
  the Ricci tensor or the Ricci scalar according to its number of indices.
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

## 1.2.8 and earlier

The original implementation, whose history is in the git log.
