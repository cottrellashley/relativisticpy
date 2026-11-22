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
- Integrations: every function the language knows is a capability that a module of
  `relativisticpy.integrations` declares, a class whose methods compute it with SymPy
  and, where it has one, with arrays. The package is scanned, outside packages
  contribute through the `relativisticpy.integrations` entry points, and a library a
  capability needs is imported only when it is used. A conformance suite holds every
  capability to the pattern in nine checks, saying in words what to fix, and
  `integrations.check` runs it on one ([Integrations](docs/integrations.md)).
- Python objects read by name: `\texttt{prices}` is the object called prices in the scope
  a workbook is given, `Workbook(scope=globals())`, with its attributes, keys and
  positions, `\texttt{prices["NVDA"]}`, read by a grammar of its own that never runs
  Python. Object integrations, a kind of their own, decide what an object becomes: a
  number a scalar, an array data, a pandas table labelled data whose index and columns
  are the bases of spaces, a SciPy sparse matrix sparse data, an xarray DataArray data
  over its named dimensions, a SymPy expression an expression in the program's symbols, a
  list a vector or a matrix, and a function a function, called on numbers and on data
  elementwise. Written with indices, `\texttt{signal}_t`, an object is data with them.
  Data from two objects meets by label, never by position, as data bound
  with `wb["P_{ta}"] = frame` now does too. `show(source, signal=signal)` and
  `Editor(wb, scope=...)` read objects, the editor's going to its worker as data; a
  document checked and a program sent to the MCP server have no scope, and refuse a
  reference, saying so ([Beyond spacetime](docs/spaces.md#python-objects-by-name)).
- Functions read as books write them, each capability declaring its forms in LaTeX and
  Unicode: `\Gamma(x)`, `\mathrm{B}(a, b)`, `\zeta(s)`, the Bessel functions `J_{\nu}(x)`,
  `Y_{\nu}`, `I_{\nu}`, `K_{\nu}`, `j_n` and `y_n`, the Airy functions, `P_n(x)`,
  `P_\ell^m(x)`, `H_n(x)`, `L_n(x)`, `L_n^{(\alpha)}(x)`, `T_n(x)`, `U_n(x)`,
  `Y_\ell^m(\theta, \phi)`, `erfc`, `erfi`, `Ei`, `Si`, `Ci`, Lambert's `W_k(x)`, the
  complete elliptic integrals, `{}_pF_q(a; b; z)`, the Pochhammer symbol `(a)_n`, the normal
  distribution function `\Phi(x)`, the principal logarithm, the unit step `u_c(t)` and the
  Dirac delta, exact with SymPy and numeric with SciPy, printed in the same forms. A book
  name no integration provides yet is refused, saying what it is, never read as an
  unknown function.
- Transforms and rewrites, two kinds of capability. The Laplace, Fourier and z-transforms
  and their inverses, by their book names, `\mathcal{L}\{f(t)\}(s)`,
  `\mathcal{F}\{f\}(\omega)` and `\mathcal{Z}\{x\}(z)`, exact with SymPy, the z-transform
  by summation and its inverse by partial fractions; their variables named by convention,
  and the Fourier transform in the convention a program declares by its integral, shows
  by the 2π of an integral it writes, or its letters choose, said in a note. The discrete
  Fourier transform of data, `\operatorname{fft}(x_t)`, with NumPy or JAX, over a space of
  the frequencies of its index, and `\operatorname{ifft}` back. The transform written as
  its sum, on data, is computed as an FFT, by a rewrite the conformance suite checks gives
  the same numbers.
- Grids written in the notation: a range beside a definition,
  `x_n := n\,h, \quad n = 0, \ldots, N-1`, declares the space of its index, and the
  definition is made for each of its values. A sum whose body writes its index on data,
  or on a list, runs over the whole of it, from its lower limit, as a contraction.
- Matrices as books write them: the trace `\operatorname{tr} A` and the rank, unbracketed
  as `\det A` is, `e^{A t}`, `\exp`, `\log` and `\sqrt` of a matrix, powers `\Lambda^k`
  and `V^{1/2}`, the pseudoinverse `A^+`, the transpose `A^T`, `A^\top` or `Aᵀ` and the
  conjugate transpose `A^\dagger`, the spectral radius `\rho(A)`, the eigenvalues
  `\lambda_{\max}(A)` and `\lambda_k(A)`, the spectrum, the norms `\|A\|_2`, `\|A\|_F`
  and `\|A\|_\infty`, the condition number `\kappa(A)`, the adjugate, `\ker A` and
  `\dim \ker A`, `\operatorname{vec}`, entries `(A^3)_{12}`, the Kronecker product
  `A \otimes B`, the direct sum `A \oplus B`, and matrices written in blocks. `I`,
  `\mathbb{1}`, 𝟙 and `\mathbf{1}` added to a matrix are the identity, a number added to
  a square one is a multiple of it, `\mathbf{1}` in a product is the vector of ones, unless
  the program defines it, and a 1 × 1 product is the number it holds. Each is read by its
  operands (`design/notation.md`, 6, 11, 12 and 23), recorded as a reading, with a note
  where another was possible, and computed exactly with SymPy and numerically with NumPy
  and SciPy, or JAX; given a map `A^a{}_b`, a function of a matrix acts on it, over its
  space.
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
- Assumptions written as books write them: `a > 0`, `0 < x < 1`, `a, b > 0` and
  `x \in \mathbb{R}` say what names are for the rest of a program, as a reading
  `Workbook.explain` shows. Where a program says nothing, an integral or a limit
  that would hold conditions on a parameter is given under the assumption books
  make, a scale positive, with a note and the step's `assuming`. `solve` drops
  the solutions an assumption rules out, saying which, and takes another path
  where SymPy fails inside itself; `dsolve` takes conditions written `y(0) = y_0`.
- Optima as books write them: `\max_{x}`, `\min_{x \in S}`, `\sup_{0 \le x < 1}`,
  `\inf_{x > 0}`, and where they are, `\arg\max_{x, y}`, `\operatorname*{arg\,min}_w` and
  `\underset{x}{\min}`, in plain text `min_{x}`, with the set or the conditions of the
  variables beneath the operator, in `\substack` too, and constraints after "subject to"
  or "s.t.", on the line or the lines after it. Over an index of data, an optimum is the
  greatest or least value along it, and `\arg\max` its label; over variables, real, whole
  numbers, vectors and tensors, it is found by optimizers, a new kind of capability: SymPy's
  first, exactly, from stationary points and the ends of a domain, the Lagrange and KKT
  conditions, the simplex method, or every point of a finite domain; then SciPy's,
  `linprog` and `milp`, convex quadratic programs, and `minimize`, which take first a
  problem written with decimals. A note says which found it and whether it is global, and
  where the answer depends on signs a program does not say, it is found with the fewest
  parameters positive, as books take them. The conformance suite checks that an optimum
  reported is feasible, that the KKT conditions hold at it, and that exact and numeric
  optimizers agree. `A \succ 0` and `w \succeq 0` are generalized inequalities.
- Names read as books write them. A subscript that labels is part of the name,
  as in `\lambda_1`, `M_\odot`, `r_+` and `\sigma_x`, and so is a number or an
  order raised on a name being defined, as in `\sigma^2 :=`; `\Delta x` is one
  name, the change in x. A one-letter subscript the value is written with makes
  a family, `c_n := ...` or `\psi_n(x) := ...`, which `c_1`, `a_{n+1}` and
  `\psi_3(x)` evaluate, and a recurrence with its first members given is
  computed. A name with scripts or a prime defines a function, as `R_{nl}(r) :=`
  and `H'(x) :=` do, `f^{(4)}(0)` is a derivative, and a function named without
  its argument is the function at its parameters. Each reading is recorded for
  `Workbook.explain`, with a note where another was possible.

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
