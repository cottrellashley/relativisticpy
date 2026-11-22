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
  reference, generated from the language's own tables, is there even without it. The
  reference indexes the functions, a line for each family with every name and form, and
  `relativisticpy help NAME` describes one in full, as `relativisticpy help besselj`,
  generated from what its integration declares: its forms, its arguments, what computes
  it, its examples with their answers and its notes. A misspelled name is told the topic
  or the function it may have meant, and the MCP server's `help` tool and its resources,
  `relativisticpy://docs/besselj`, give the same.
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
- Quantum notation as books write it, read by its operands (`design/notation.md`, 13
  and 20). Dirac's kets, bras, brackets and matrix elements, and the physics and braket
  packages' `\ket`, `\bra` and `\braket`: a ket defined as a column is that vector, its
  bra the conjugate transpose, `|a\rangle\langle b|` a matrix and `|00\rangle` the
  tensor product of the kets it names; a function of one variable is a wavefunction,
  whose brackets are integrals over the domain the program states; a bracket a program
  defines, `\langle x|p\rangle := ...`, is a function of its labels, and its conjugate
  written the other way round; labels may be lists, as Clebsch–Gordan coefficients are
  written. Angle brackets without a bar hold what a program defines, the inner product of
  two vectors, or an expectation value in the state it names. `[A, B]` and `\{A, B\}` of
  matrices or operators are the commutator and the anticommutator, and of anything else
  a list and a set, as before. An expression that ends in an operator,
  `\hat{p} := -i\hbar \frac{d}{dx}`, defines an operator, applied by juxtaposition and
  composed by products and powers, and `a_\pm` defines both signs; a claim between
  operators is decided on a test function. An `=` whose left side is written with
  operators and kets nothing defines, `[\hat{x}, \hat{p}] = i\hbar`,
  `a|n\rangle = \sqrt{n}|n-1\rangle` or `\langle m|n\rangle = \delta_{mn}`, states a rule
  for what follows, which the algebra of those operators and kets reduces by, and the
  modulus of whose brackets, `|\langle k|H'|0\rangle|^2`, a sum over a label takes. A
  star before a call's arguments, `\psi^*(x)`, conjugates the call, as `\psi(x)^*` does.
  These replace the refusals of the commutator of matrices, of angle brackets and of
  Dirac's notation.
- Sequences written with square brackets, as Oppenheim and Schafer write discrete-time
  signals (`design/notation.md`, 31). `x[n] := \ldots` defines a sequence of the integer n,
  and `x[3]` and `x[n - 1]` are its values; where nothing defines δ or u, `\delta[n]` and
  `u[n]` are the unit impulse and the unit step, capabilities of a new integration, with a
  note, where the brackets hold an integer as books write a point, so that a variation
  `\delta[(-g)^{-1/2}]` is read as before; brackets after a function of one variable are
  its value; a list, a tensor or data is read as before. A sum over a sequence's index takes
  each impulse at its point and cuts its terms' ranges to where their steps are 1, and runs
  over every integer where no limits are written, as `\sum_k x[k] h[n - k]`; a closed form
  that holds the case of two equal parameters apart is given for the others, with a note. A
  sequence defined through its own members is a difference equation at initial rest,
  computed one member after another, and in closed form by SymPy's `rsolve`, as `solve`
  gives one. An equation between sequences nothing defines is a difference equation, whose
  capital letters are their z-transforms, `H(z) := \frac{Y(z)}{X(z)}`, and `poles` and
  `zeros` give the roots of a transfer function. `H(e^{j\omega}) := \ldots` defines the
  frequency response as a function of ω, written at e^{jω}. A bound on a modulus standing
  alone, as `|z| > \frac{1}{2}`, is an assumption, which decides where a series converges,
  and a claim about samples, as `\cos(7\pi n/5) = \cos(3\pi n/5)` of an integer n, is
  proven where it holds at the integers.
- Vectors and vector calculus as books write them (`design/notation.md`, 8, 16 and 25).
  A list, a tuple or a column of numbers is a vector, `\mathbf{v} := (1, 2, 3)`:
  `\cdot` and `\times` between vectors are the dot and cross products, and between
  numbers multiply; `|\mathbf{v}|` and `\|\mathbf{v}\|` are its length and
  `\mathbf{p}^2` is \(\mathbf{p} \cdot \mathbf{p}\). Derivatives, integrals, evaluation
  bars and limits of a vector are its components', `\mathbf{r}'(t)` and
  `\mathbf{x}_u` among them, and `\mathbf{v}(\mathbf{r}(t))` is a field at a point.
  `\hat{\mathbf{x}}`, `\hat{\imath}` and `\mathbf{i}`, `\mathbf{j}`, `\mathbf{k}`, which
  nothing defines, are Cartesian unit vectors, `\hat{x}` too where no operator is near,
  and `\hat{\mathbf{r}}` and `\mathbf{e}_r` are those of the coordinates in force.
  `\nabla f`, `\nabla \cdot \mathbf{v}`, `\nabla \times \mathbf{v}`, `\nabla^2 f` and
  `\Delta f` are computed from the metric of the coordinates in force, whose scale
  factors a line element gives, or the Cartesian coordinates the program writes in them,
  or their names; without coordinates, in x, y and z. `(\mathbf{u} \cdot \nabla)` acts on
  what follows it. Of a function at a point, `\nabla f(\mathbf{a})` is its gradient and
  `\nabla^2 f(\mathbf{a})` its Hessian, said in a note. An equation between vectors is
  decided component by component, each as a scalar equation is, and so is one between
  tensors, by \(\cosh^2 - \sinh^2 = 1\) too. Integrals over a disk or a ball about the
  origin, `\iint_{x^2 + y^2 \le 1} f\,dA`, and over a box, `\iiint_{[0, 1]^3} f\,dV`, are
  computed. These replace the refusals of the cross product of unit vectors, of the
  derivative of a list and of the absolute value of one.
- Cartesian tensors (`design/notation.md`, 16): where the metric in force is written as
  the identity, `ds^2 = dx^2 + dy^2 + dz^2`, an index repeated in any position is summed,
  as it is in a space declared as \(\mathbb{R}^n\), and there `\epsilon_{ijk}`, which
  nothing defines, is the Levi-Civita symbol. In spacetime, where its sign is a
  convention, it is not read yet.
- Integrals along a curve the program defines, `C(t) := (\cos t, \sin t, 0)` with
  `0 \le t \le 2\pi`: `\oint_C \mathbf{F} \cdot d\mathbf{r}` and `\int_C`, in which
  `d\mathbf{r}` is the curve's tangent, its name its point and x, y and z its components;
  where SymPy finds no closed form in time and only the curves' parameters are left, as
  in Gauss's linking integral, the value is computed numerically, said in a note.
- Evaluation bars and brackets as books write them (`curriculum/ROADMAP.md`, section 7):
  `\left. F \right|_{x=a}` is F at a, and `\left. F \right|_a^b`, `\left[F\right]_a^b`,
  `[F]_{x=a}^{x=b}` and `F \Big|_a^b` are F(b) − F(a). A bar written with `\left.` and
  `\right|` evaluates what they enclose and nothing else, so that
  `x + \left. x^2 \right|_{x=2}` is x + 4, not 6, and a limit around it is taken after it;
  a bar with limits written alone evaluates the side of an equation it is on. Where no
  variable is written, it is that of an integral beside it in a sum, as integration by
  parts writes it, else the one symbol what it evaluates holds, and otherwise the bar is
  refused, with a hint. Each reading is recorded, as `Workbook.explain` shows it.
- Convolutions as signal processing writes them (`design/notation.md`, 27, now adopted):
  `*` or `\ast` between two functions applied at one variable, `f(t) * g(t)`, is their
  convolution, the integral over every τ of f(τ) g(t − τ), and between two sequences at
  points of one index, `x[n] * h[n]`, the sum over every integer k of x[k] h[n − k], each
  with a note; in their operands, u and δ that nothing defines are the unit step and the
  Dirac delta. Between anything else `*` multiplies, as `2*x` does, and `z^*` is a
  conjugate. `\circledast`, and ⊛, is the circular convolution, of the length its subscript
  gives, `\circledast_N`, or else of the longer sequence: at a point,
  `(x \circledast h)[0]`, and as a sequence, `x[n] \circledast h[n]`. These replace the
  refusals of the asterisk between sequences and of `\circledast`. In plain text too,
  `f(t)*g(t)` is a convolution, as the decision says; a product is written side by side,
  as the text form of a result now writes two functions nothing defines, `A(r) B(r)`.
- A curriculum of mathematics with known answers, `curriculum/`: 407 cases across
  18 domains at five levels, from notation to the frontier, each a program and an
  oracle that computes its answer independently. `benchmarks/curriculum.py` runs
  it, and the test suite keeps every case that passes passing. Both run under one hash
  seed, and `benchmarks/curriculum.py --seeds 0,1,2` lists the cases whose outcome
  changes with it, as continuous integration does on every change, as findings.
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
- Number theory and combinatorics, in two integrations: Euler's totient `\varphi(n)`, the
  Möbius function `\mu(n)`, the prime-counting function `\pi(x)`, the divisor functions
  `\sigma_k(n)` and `d(n)`, `\omega(n)`, `\Omega(n)`, von Mangoldt's `\Lambda(n)`, the
  remainder `a \bmod m`, the Stirling numbers `{n \brace k}` and `{n \brack k}`, the
  Catalan, Fibonacci, Bell, Bernoulli and harmonic numbers, the partitions `p(n)` and the
  multinomial coefficient, each exact with SymPy. A letter is the function only where its
  argument is an integer, so that `\varphi(x)` and `\mu(t)` stay a program's own; `\pi(x)`
  stays π times x, and counts primes only of a name assumed an integer; and `H_n` and `B_n`
  are numbers only of such a name, so that `H_0` stays Hubble's constant
  (`design/notation.md`, 32 to 35).
  Sums and products over a condition: `\sum_{d \mid n}`, `\prod_{p \le x}` and
  `\prod_{p \text{ prime}}`, with p ranging over the primes, Euler's products for ζ, and
  the rows of `\substack` (19 and 36). Congruences, `a \equiv b \pmod{m}`, claims decided by
  the remainders, or assumptions of a name, which the Chinese remainder theorem may leave
  one value; `a^{-1} \bmod m`, the inverse; `solve` of congruences; and `[x^n] F(x)`, the
  coefficient of a generating function (30 and 37). The greatest common divisor of a
  whole number and the index of a sum waits for the index's values. `\sum_k` of binomial
  coefficients is over every integer, as Concrete Mathematics writes it, and a sum of
  finitely many terms holds no conditions of convergence, so that the binomial theorem
  and Vandermonde's convolution hold (38).
- Random variables, declared as books declare them: `X \sim \mathcal{N}(\mu, \sigma^2)`,
  `N \sim \operatorname{Poisson}(\lambda)`, `X \sim \operatorname{Bin}(n, p)`, `U(a, b)`,
  `\operatorname{Exp}(\lambda)`, `\Gamma(k, \theta)`, `\chi^2_k`, `t_\nu` and the Bernoulli,
  beta and geometric distributions, each a capability of a new kind, `distribution`, in
  `integrations/sympy_stats.py`, whose forms say what each argument is: `\mathcal{N}(\mu,
  \sigma^2)` takes the variance, and the note of a declaration gives the value of each.
  `X, Y \overset{\text{iid}}{\sim} D` declares independent ones, `X_1, \ldots, X_n
  \overset{\text{iid}}{\sim} D` a sample whose members a sum takes in turn, and
  `X \mid N \sim \operatorname{Bin}(N, p)` or `X \mid Y = 0 \sim D` a law given another.
  `\Pr(X > 1)`, `P(X = 2)`, `\mathbb{P}(A)`, `\mathbb{E}[X]`, `\operatorname{Var}(X)` and
  `\operatorname{Cov}(X, Y)`, with `\mid` the conditional bar inside them, and `f_X(x)`,
  `F_X(x)`, `F_X^{-1}(q)`, `M_X(t)` and `\varphi_X(t)`, are computed exactly by SymPy, by
  the laws of total expectation and probability where a law is given another, and from
  each distribution's declared mean and variance; numerically by SciPy, with a note, where
  SymPy finds no closed form in time and there are no parameters. A probability of one
  variable is found from its distribution function over the values the event holds at, as
  SymPy's own is 0 for P(T ≤ a); where a parameter the program says nothing of could fall
  outside the values the variable takes, it is taken inside them, as books do, and said so.
  `\Pr(A \mid B) := 0.9` gives an event's probability, and a capital letter nothing
  defines, inside an expectation or a variance, is a random variable of any law, so that
  `\operatorname{Var}(aX + b) = a^2 \operatorname{Var}(X)` holds. The conformance suite
  checks that a distribution's density integrates to 1 and that its mean and variance are
  those it declares, by SymPy and by SciPy (`design/notation.md`, 14).

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
- The z-transform of a sequence written with impulses and steps is taken over every n, and
  the inverse z-transform gives a sequence back, with its step and its impulses,
  `\left(\frac{1}{2}\right)^n u[n]`. A transform written without its point in the value
  of a function or a sequence whose parameter is its variable is at that parameter, so
  that `F(2)` of `F(s) := \mathcal{L}\{t^2\}` is 1/4. In plain text, brackets right
  after a name are refused as Python's indexing by the reader, which knows whether the name
  is a sequence, rather than by the parser.
- The trace, the determinant and the other functions of a matrix, of a name nothing
  defines, as `\operatorname{tr}(\ln M)` and `\det M`, are unknown values, tr(log M) and
  det(M), said in a note, rather than the number M would be as a 1 × 1 matrix: a claim
  about a matrix nothing defines, as `\operatorname{tr}(\ln M) = \ln(\det M)`, relates,
  rather than holding by its 1 × 1 case. Of a matrix, a number, or a scalar the program
  defines, a coordinate, a symbol it says something of or a parameter of its metric, they
  are as before.
- `\bar h` is h̄, a symbol of its own, shown as `\bar{h}`, no longer ℏ: an accent whose
  name, put after its letter's, would spell another symbol's name, as hbar spells ℏ's, is
  written in the name with the mark Unicode sets over the letter. Each form of either
  reads back as it, and `\bar h := ...` no longer gives ℏ a value.
- Angle brackets no longer group as brackets do: `\langle x \rangle` is what a program
  defines, an expectation value or an inner product, and refused where it is none of
  them, rather than `x`.
- The same program gives the same answer every time. `relativisticpy`,
  `python -m relativisticpy` and `relativisticpy mcp` start themselves again with
  `PYTHONHASHSEED=0` where it is not set, before they import anything else, and the
  processes they start for their work, of `-j` and of the MCP server, inherit it; on
  Windows, which cannot replace a process, only those processes do. A seed set is kept.
  From Python, start Python with `PYTHONHASHSEED=0` to reproduce a run exactly
  ([Troubleshooting](docs/troubleshooting.md#a-result-changed-from-one-run-to-the-next)).
- `import relativisticpy` imports each of its names when it is first used, so that
  importing a module of the package, as the command does to start, takes milliseconds.
- RelativisticPy needs Python 3.10 or later. SymPy is its only required
  dependency.

### Removed

- The modules of version 1 (`relativisticpy.core`, `relativisticpy.gr`,
  `relativisticpy.interpreter`, `relativisticpy.symengine` and others). The
  package's entry point is still `relativisticpy.Workbook`.

## 1.2.9 and earlier

The original implementation, whose history is in the git log.
