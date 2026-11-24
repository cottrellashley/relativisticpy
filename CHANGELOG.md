# Changelog

Notable changes to RelativisticPy. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## 0.1.0 - 2026-11-23

The first release of RelativisticPy as a language: mathematics written as its books
and papers write it, in LaTeX, plain text or Unicode, with an engine that computes
it. It is a new implementation, and starts the version numbers again. The releases
1.1.9 to 1.2.9 on PyPI were an earlier one, and
[Coming from 1.x](docs/upgrading.md) says what changed since them.

### The language

- **Three notations, mixed freely**: LaTeX, plain text and Unicode. An equation pasted
  from a paper runs as it is: plain TeX, `\left` and `\right`, environments, closing
  punctuation, several equations in a display, and chains of relations.
- **Plain text and Unicode, read as the LaTeX they spell, or refused**: `sin 2x` is
  `\sin 2x`, `d/dx f` and `∂f/∂x` are derivatives, `int_0^1 x^2 dx` and
  `∑_{n=1}^∞ 1/n²` an integral and a sum, `oo` infinity, and Unicode's scripts, roots,
  fractions and floors are LaTeX's, as `f⁽²⁾` is `f^{(2)}`. In plain text `xy`, beside an
  x and a y of the program's own, says that it is one name, and `sinx` says how to write
  sin of x; in LaTeX and Unicode, `sinx` is refused. In plain text, a run of a
  function's parameters in its body is their product, as `xy` in
  `g(x, y) := x e^(xy)`, with a note.
- **LaTeX's brackets, primes and powers**, as books mean them: `(xy)^2` is x²y², and
  `\pi(x + 1)^2` is π(x + 1)²; `(x^3)'` is 3x², and `\frac{d(\sin^2\theta)}{d\theta}` a
  derivative; Euler's `D_x f(x)`, and Stewart's `f_x(1, 2)`, `(f_x)_y`, `f_1`, `f_{12}`
  and `D_1 f`; `J^0` and `F^{0i}` are components, not powers; and `a = b \geq c` is two
  claims. A function the program defines is called whatever spaces stand before its
  brackets, as in `f (3)`; brackets after a name in brackets of its own multiply, as in
  `(a)(b)`; and `x_1x_2` is x₁x₂.
- **`:=` and `=`.** `:=` always defines. `=` defines only what has no meaning yet: a new
  name, function or tensor, or a line element. Otherwise it is a claim, which is decided,
  or an equation to solve. A claim that nothing proves is refused, never shown as true.
  An equation is shown, not solved, and a value after it that holds one of its names says
  so in a note, with how to solve it.
- **Calculus**: derivatives in every notation, integrals, limits, sums and products in
  closed form where there is one, infinite products by their partial products, a series
  that converges under a condition with a note that says so, evaluation bars and
  brackets, `solve`, and `dsolve` with its conditions, which takes the variable of its
  unknown from its arguments, as in `dsolve(y' = r y, y(t))`. Where nothing defines δ,
  `\delta(x - a)` of a point is Dirac's delta, with a note, and `dsolve` solves an
  equation beside it; δ of what is no point, as Carroll's `\delta(g^{-1})`, is not. A
  condition `y(0) = 0` is y at 0, though y was written as a factor before, as in
  `y'' + \lambda y = 0`.
- **Assumptions, written as books write them**: `a > 0`, `0 < x < 1` and
  `x \in \mathbb{R}` hold for the rest of a program. Where a program says nothing, an
  integral or a limit is given under the assumption books make, and a note says so.
  `solve` drops the solutions an assumption rules out, and says which.
- **Names, as books write them**: a subscript that labels, as in `M_\odot` and `r_+`;
  families, `c_n := ...` and then `c_3`; recurrences; primes and accents; `\Delta x`; and
  a function named without its argument. A bar or a star over a quantity the program
  defines, or declares complex, is its conjugate, as `\bar{z}`, `\overline{3 + 4i}` and
  `(3 + 4i)^*` are; over a name that means nothing, `\bar{x}` is a name. A labelled name
  that nothing defines, before names in brackets, is an unknown function, as
  `\Phi_N(r)` is, and so is a name in `\operatorname` that nothing defines, applied to
  what follows it, as in `\operatorname{lb} 8`; a note says each is unknown.
- **Nothing is guessed silently.** A reading that could have been another carries a
  note, `Workbook.explain` shows how each statement is read, and what cannot be read or
  decided is refused, saying where, why and what to write instead. Every decision on
  notation that books write in more than one way is recorded in `design/notation.md`.
  Notes say where an `I` that nothing defines is written as the imaginary unit is, as in
  `3 + 4*I` in plain text; where a name runs into digits, as `j4` after
  `j := \sqrt{-1}`; where brackets after a name with a value multiply it, as `\delta[n]`
  after `\delta := 0.1`; and how LaTeX reads a run of letters before brackets, as
  `foo(x)`.
- **What has no value is refused, never shown as one**: 1/0, Γ(0) and tan(π/2), a limit
  from both sides at a pole, an integral that never settles, a series whose terms tend
  to no limit, a negative power of a singular matrix, the zeros of 0, every point of
  which is one, the unit step of a number that is not real, as `step(1/2 + i)`, and a
  power of a matrix that SymPy leaves undone.
- **Closed forms checked by numbers.** A definite integral or a series in closed form is
  checked against mpmath, and an antiderivative is differentiated back; a closed form
  that numbers refute is refused. `solve` drops a root where the equation, as written,
  divides by zero.

### Mathematics, by field

- **Special functions** by their book names: `\Gamma(x)`, `\zeta(s)`, the Bessel and Airy
  functions, the orthogonal polynomials, spherical harmonics, `{}_pF_q`, the elliptic
  integrals, `\Phi(x)` and more, exact with SymPy and numerical with SciPy; and the
  inverse hyperbolic functions by their ISO names too, `arsinh`, `arcosh` and `artanh`.
- **Matrices**: `\operatorname{tr} A`, `\det A`, `e^{A t}`, `A^T`, `A^\dagger`, `A^+`,
  `\lambda_{\max}(A)`, `\|A\|_F`, `\kappa(A)`, `A \otimes B` and `A \oplus B`, with the
  identity and the vector of ones read by what they stand beside. Artin's `A^t`, Lang's
  `{}^tA` and Goldstein's `\tilde{A}` are transposes, with a note; the spectrum is
  `\operatorname{spec}(A)` or `\sigma(A)`; a product that is 1 × 1 is the number it
  holds, however it is written, `ab` or `a \cdot b`; and `\|x\|_2` is a norm of a vector
  written as a list too. `eig(A)` of a matrix is refused, with a hint to write
  `spec(A)`.
- **Vectors and vector calculus**: the dot and cross products, also as `dot(u, v)` and
  `cross(u, v)`, `\nabla f`, `\nabla \cdot \mathbf{v}`, `\nabla \times \mathbf{v}` and
  `\nabla^2 f` in the coordinates in force, unit vectors, integrals along curves and over
  simple regions, and Cartesian tensors with `\epsilon_{ijk}`.
- **Probability**: random variables declared with `\sim`, eleven distributions, `\Pr`,
  `\mathbb{E}`, `\operatorname{Var}` and `\operatorname{Cov}`, conditioning with `\mid`,
  densities, distribution functions, quantiles and generating functions.
- **Signals**: sequences `x[n]`, the unit impulse and the unit step, difference
  equations, the Laplace, Fourier and z-transforms and their inverses, the FFT of data,
  convolution written `*`, transfer functions and frequency responses.
- **Optimisation**: `\max`, `\min`, `\sup`, `\inf` and `\arg\max`, with constraints after
  "subject to", found exactly with SymPy or numerically with SciPy. A note says how an
  optimum was found, and whether it is global.
- **Number theory and combinatorics**: `\varphi(n)`, `\mu(n)`, the divisor functions, the
  Stirling, Catalan, Bell and Bernoulli numbers, sums over divisors and primes,
  congruences written with `\pmod`, and `[x^n] F(x)`.
- **Quantum mechanics**: Dirac's kets, bras and brackets, commutators and
  anticommutators, operators as values, `\hat{p} := -i\hbar \frac{d}{dx}`, rules such as
  `[\hat{x}, \hat{p}] = i\hbar`, and expectation values.
- **Chemistry and machine learning**: `\mathrm{p}K_a` is −log₁₀ K_a, as chemists write
  it, and ReLU and the softplus are known by their names.
- **General Relativity**: index notation in full, its rules checked before anything is
  computed; metrics from line elements; the Christoffel symbols and the Riemann, Ricci,
  Einstein and Weyl tensors; covariant and Lie derivatives; geodesics; eight known
  spacetimes; and tensor equations solved as tensor equations, in index notation, before
  there are components. A field of the coordinates met with a path is taken along it, as
  the metric is in a Lagrangian `\frac{1}{2} m\, g_{ij} \dot{q}^i \dot{q}^j` after
  `q^i := (\theta(t), \phi(t))`, with a note.

### Data, and Python

- **Index notation over any space**: an index ranges over assets, days or the states of
  a qubit. Data is bound from NumPy, pandas, SciPy sparse matrices or SymPy, and
  statistics, spectra and linear algebra are contractions, computed with NumPy, or with
  JAX, which gives gradients. A letter a tensor of data is written with is its index,
  as a is in `\mathbb{E}_t[X_{ta}]`, though a column is named a too, with a note.
- **Python objects by name**: `\texttt{prices}` is the object of that name in the scope a
  workbook is given, `Workbook(scope=globals())`. A table brings its labels, and data
  from two objects meets by label. A paper that is checked, and the MCP server, have no
  scope.
- **The Python API**: `Workbook`, with `expr`, `exe`, `each`, `explain`, `check` and
  `copy`; results as SymPy values, `Tensor`, `Metric`, and equations that answer
  `holds()`; `relativisticpy.display`, for every form of a result; errors that say where,
  why and what to do; and `relativisticpy.numeric`, for NumPy functions and geodesics.

### Tools

- **The command line**: `relativisticpy FILE` runs a program or a Markdown notebook,
  `--render` typesets a notebook's results, `--format json` gives results as data, and
  without a file there is a prompt. Its exit statuses are documented, and a run gives the
  same answer every time: the command fixes its hash seed.
- **Checking papers**: `relativisticpy check paper.tex` runs the display equations of a
  LaTeX document in order, and gives each a verdict: holds, differs, relates,
  ill-formed, defines, computes, not computed, not understood or not parsed. An
  equation for functions nothing defines relates, as the conditions a derivation finds
  for the functions of a metric do, unless it writes a quantity computed from the
  metric; so does a value compared with an equation between quantities nothing defines,
  as a name defined by a chain of equations is. An equation of index notation without
  components that is no identity relates, as a law does, and is never said to differ.
- **Help, offline**: `relativisticpy help` prints the documentation, a page for the core
  of the language and one for each field, found by the names a person asks for, as
  `relativisticpy help number-theory`; and `relativisticpy help NAME` describes any
  function, from what its integration declares.
- **An MCP server for AI agents**, `relativisticpy mcp`: tools to compute, to verify a
  step, to check a paper and to read the documentation, within limits of time and
  memory. Its instructions say how LaTeX and plain text read a run of letters.
- **Values that say whether they are answers.** Each value, in the MCP server's results
  and in `--format json`, says whether it is a `complete` answer, and if not, its
  `reasons`, as an integral kept as written, or a function nothing defines applied to a
  number, as `N(0.77)` is in LaTeX; and whether it is `approximate`. An equation of index
  notation that is no identity, or an equation for a function nothing defines, has
  `holds: null`: it is not decided. A refusal is the same `error` from `compute`,
  `verify` and `check`, with its place and a hint of what to write instead, and the
  tools' descriptions say what an agent should do with each.
- **Live mathematics in notebooks**: results typeset in Jupyter and marimo, `show`, and
  `Editor`, a widget that computes as it is typed.

### Extending it

- **Integrations**: every function is a capability that a module of
  `relativisticpy.integrations` declares, of one of nine kinds. A conformance suite holds
  each to one pattern, and says in words what to fix. Adding a library is writing one
  module, and there is no configuration.
- **A curriculum**, `curriculum/`: 734 problems across 32 fields at five levels, each
  with an answer computed without RelativisticPy. The tests keep every problem that
  passes passing. A problem that fails says how: a wrong answer, a value kept as
  written with a note that says so, as a function nothing defines at a number, or a claim
  left undecided, a refusal, or a time limit.
- **A notation matrix**, `curriculum/notations/`: 44 constructs of algebra, calculus,
  linear algebra and index notation, each written every way books write it, in LaTeX,
  plain text and Unicode, 3,812 programs in all. Every form must give the answer an
  oracle computes without RelativisticPy, or the refusal its operand calls for, as of
  1/0; `benchmarks/notations.py` runs it, and the tests keep what passes passing.
- **A record of readings**, `benchmarks/readings.py`: how every statement the docs, the
  curriculum and the tests write is read, so that a change to the language lists the
  statements it reads otherwise.
- **A corpus benchmark**, `benchmarks/corpus.py`: how much of real papers is read, and
  what checking them finds.

### Requirements

- Python 3.10 or later. SymPy is the only required dependency, and the extras
  `numeric`, `data`, `jax`, `mcp` and `live` add what each is named for.

## 1.1.9 to 1.2.9

An earlier implementation, published to PyPI, whose history is in the git log.
