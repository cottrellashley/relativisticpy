# Changelog

Notable changes to RelativisticPy. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## 0.1.0 - Unreleased

The first release of RelativisticPy as a language: mathematics written as its books
and papers write it, in LaTeX, plain text or Unicode, with an engine that computes
it. It is a new implementation, and starts the version numbers again. The releases
1.1.9 to 1.2.9 on PyPI were an earlier one, and
[Coming from 1.x](docs/upgrading.md) says what changed since them.

### The language

- **Three notations, mixed freely**: LaTeX, plain text and Unicode. An equation pasted
  from a paper runs as it is: plain TeX, `\left` and `\right`, environments, closing
  punctuation, several equations in a display, and chains of relations.
- **`:=` and `=`.** `:=` always defines. `=` defines only what has no meaning yet: a new
  name, function or tensor, or a line element. Otherwise it is a claim, which is decided,
  or an equation to solve. A claim that nothing proves is refused, never shown as true.
- **Calculus**: derivatives in every notation, integrals, limits, sums and products in
  closed form where there is one, evaluation bars and brackets, series, `solve`, and
  `dsolve` with its conditions.
- **Assumptions, written as books write them**: `a > 0`, `0 < x < 1` and
  `x \in \mathbb{R}` hold for the rest of a program. Where a program says nothing, an
  integral or a limit is given under the assumption books make, and a note says so.
  `solve` drops the solutions an assumption rules out, and says which.
- **Names, as books write them**: a subscript that labels, as in `M_\odot` and `r_+`;
  families, `c_n := ...` and then `c_3`; recurrences; primes and accents; `\Delta x`; and
  a function named without its argument.
- **Nothing is guessed silently.** A reading that could have been another carries a
  note, `Workbook.explain` shows how each statement is read, and what cannot be read or
  decided is refused, saying where, why and what to write instead. Every decision on
  notation that books write in more than one way is recorded in `design/notation.md`.

### Mathematics, by field

- **Special functions** by their book names: `\Gamma(x)`, `\zeta(s)`, the Bessel and Airy
  functions, the orthogonal polynomials, spherical harmonics, `{}_pF_q`, the elliptic
  integrals, `\Phi(x)` and more, exact with SymPy and numerical with SciPy.
- **Matrices**: `\operatorname{tr} A`, `\det A`, `e^{A t}`, `A^T`, `A^\dagger`, `A^+`,
  `\lambda_{\max}(A)`, `\|A\|_F`, `\kappa(A)`, `A \otimes B` and `A \oplus B`, with the
  identity and the vector of ones read by what they stand beside.
- **Vectors and vector calculus**: the dot and cross products, `\nabla f`,
  `\nabla \cdot \mathbf{v}`, `\nabla \times \mathbf{v}` and `\nabla^2 f` in the
  coordinates in force, unit vectors, integrals along curves and over simple regions,
  and Cartesian tensors with `\epsilon_{ijk}`.
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
- **General Relativity**: index notation in full, its rules checked before anything is
  computed; metrics from line elements; the Christoffel symbols and the Riemann, Ricci,
  Einstein and Weyl tensors; covariant and Lie derivatives; geodesics; eight known
  spacetimes; and tensor equations solved as tensor equations, in index notation, before
  there are components.

### Data, and Python

- **Index notation over any space**: an index ranges over assets, days or the states of
  a qubit. Data is bound from NumPy, pandas, SciPy sparse matrices or SymPy, and
  statistics, spectra and linear algebra are contractions, computed with NumPy, or with
  JAX, which gives gradients.
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
  ill-formed, defines, computes, not computed, not understood or not parsed.
- **Help, offline**: `relativisticpy help` prints the documentation, and
  `relativisticpy help NAME` describes any function, from what its integration declares.
- **An MCP server for AI agents**, `relativisticpy mcp`: tools to compute, to verify a
  step, to check a paper and to read the documentation, within limits of time and memory.
- **Live mathematics in notebooks**: results typeset in Jupyter and marimo, `show`, and
  `Editor`, a widget that computes as it is typed.

### Extending it

- **Integrations**: every function is a capability that a module of
  `relativisticpy.integrations` declares, of one of nine kinds. A conformance suite holds
  each to one pattern, and says in words what to fix. Adding a library is writing one
  module, and there is no configuration.
- **A curriculum**, `curriculum/`: 734 problems across 32 fields at five levels, each
  with an answer computed without RelativisticPy. The tests keep every problem that
  passes passing.
- **A corpus benchmark**, `benchmarks/corpus.py`: how much of real papers is read, and
  what checking them finds.

### Requirements

- Python 3.10 or later. SymPy is the only required dependency, and the extras
  `numeric`, `data`, `jax`, `mcp` and `live` add what each is named for.

## 1.1.9 to 1.2.9

An earlier implementation, published to PyPI, whose history is in the git log.
