# What the curriculum fails, and why

*Measured on `main` at 755b881, by `uv run python benchmarks/curriculum.py --failures`,
which lists every case that fails with its reason. This page groups them by cause.*

The curriculum has 734 cases in 32 domains. **528 pass.** Of the other 206, **8 give a
wrong answer**, **6 keep a value as written**, saying so, and **192 are refused**, saying
why.

| Level | Pass | Wrong | Kept | Refused | Cases |
|---|---:|---:|---:|---:|---:|
| 1 notation | 129 | 1 | 0 | 5 | 135 |
| 2 computation | 204 | 0 | 1 | 18 | 223 |
| 3 derivation | 140 | 1 | 2 | 38 | 181 |
| 4 specialist | 54 | 1 | 1 | 73 | 129 |
| 5 frontier | 1 | 5 | 2 | 58 | 66 |
| **all** | **528** | **8** | **6** | **192** | **734** |

Level 1, notation as a textbook prints it, fails six cases: `j` taken for a name in
`sig.notation.imaginary-unit`, and five refusals: `e^{i\theta} = \cos\theta + i\sin\theta`
read as a definition of e, `\log x := \log_{10} x` read as a recursive definition,
`\operatorname{arccosh}`, an index range written `i, j = 1, 2, 3`, and the solid angle
`d\Omega` as a measure.

## Wrong answers

The worst failures: a value given as the answer that is not the answer.

- **An equation shown, then not used** (5): `chem.equilibrium.weak-acid`, `de.bvp.robin`,
  `sr.four-momentum.mandelstam`, `sr.lorentz.wigner-rotation`,
  `thermo.phases.maxwell-construction`. A later line uses a name that an equation before
  it constrains; the equation was shown, not solved, so the name stays free. In four, a
  note now says so, naming the equation and how to solve it (decision 3). Mandelstam's
  identity is a claim, which is shown as an equation, with no note. Needed: a design for
  which equations a program means to solve, and which it only states; and a claim decided
  under the equations stated before it.
- **A name nothing defines, taken as one of the program's own, without a note** (2):
  `j` in \(|1/(1 + j)|\), a name by decision 24, which gives a note only for j in an
  exponent (`sig.notation.imaginary-unit`); and `\lambda^{(5)}`, not found as the member
  of `\lambda^{(k)}` that a loop written in words, `\textbf{for } k = 1, \ldots, 5`, would
  compute (`num.eigen.power-iteration`).
- **Units** (1): units are names, and are not converted
  (`astro.expansion.hubble-time-units`; see `design/debt.md`).

## Kept as written

No answer, and said to be none: `fa.series.convergence` and `sig.random.autocorrelation`
keep sums SymPy finds no closed form for; `fa.transforms.delta` keeps the Fourier
transform of 1, which is 2πδ(ω); `graph.trees.cayley` keeps `I_5` and `J_5`, which the
language does not know as the identity and the matrix of ones;
`num.rounding.cancellation` keeps \(\operatorname{fl}\), rounding to floating point; and
`data.finance.black-scholes-call` keeps Hull's N(x), the normal distribution function,
which is not the language's N (decision 15), a function nothing defines, applied to
numbers.

## Refusals, by cause

Each cause, with the number of cases it blocks, how many at each level from 1 to 5, and
examples.

- **Index notation without components, spaces or ranges** (33; levels 1, 4, 9, 17, 2).
  An index range written in words, `i, j = 1, 2, 3`, is read but not used; a tensor with
  no components, as \(\eta_{\mu\nu}\) or \(\gamma^\mu\), needs coordinates or a space;
  derivatives are not yet part of tensor notation; \(x^\mu(\lambda)\) is a product.
  `la.index.delta-trace`, `sr.lorentz.index-form`, `dg.connections.parallel-transport`.
- **Notation the reader does not know yet** (30; levels 1, 1, 3, 15, 10). Commands such
  as `\slashed`, `\ddagger` and `\limsup`; the standard state, `^\circ`; a subscript on
  what is not a name, as in \(\left(\frac{\partial p}{\partial V}\right)_T\); sets
  written by a condition, \(\{f(z) : |z| = 1\}\); representations named by their
  dimension, \(\mathbf{8} \oplus \mathbf{1}\); `\operatorname{arccosh}`.
  `geom.hyperbolic.distance`, `thermo.maxwell.triple-product`, `qft.dirac.slash`.
- **Calculus not taken yet** (26; levels 0, 4, 6, 7, 9). Contour integrals and residues;
  Jacobians written as fractions; implicit derivatives; the derivative of a list; the
  equations of motion of a Lagrangian written in θ rather than θ(t); the Dirac delta in
  three dimensions; integrals over regions, \(d^4x\); optima over functions or sets.
  `complex.residues.residue-theorem`, `analysis.multivariable.jacobian`,
  `cm.lagrangian.pendulum`.
- **Definitions the language cannot make yet** (18; levels 2, 1, 6, 8, 1). A
  concentration `[\mathrm{A}]`; a product as a name, \(\omega_e x_e\); a family named
  nowhere, `P_l` in a sum; recursive definitions; recurrences whose members are not
  given. `chem.rates.second-order-brackets`, `special.legendre.generating-function`,
  `num.ode.rk4`.
- **Claims SymPy cannot decide** (16; levels 0, 3, 7, 3, 3). Inequalities, order terms
  \(O(x^n)\), and identities the sides of which agree in numbers but are not proven, as
  the functional equation of ζ, and the limit of the heat kernel, Dirac's δ(x).
  `analysis.inequalities.am-gm`, `analysis.taylor.big-o`,
  `special.zeta.functional-equation`, `de.pde.heat-kernel`.
- **Differentials and forms** (15; levels 1, 1, 0, 5, 8). Relations between
  differentials, `dU = T dS - p dV`; the exterior derivative and the Hodge star; Itô
  integrals; the solid angle as a measure. `dg.forms.exterior-derivative`,
  `thermo.potentials.thermodynamic-identity`, `prob.stochastic.ito-integral`.
- **What SymPy does not carry out** (13; levels 0, 1, 3, 2, 7). "SymPy could not compute
  this", and values defined in words, as `\text{the first zero of } \theta`.
  `econ.demand.point-elasticity`,
  `astro.gravitation.kepler-equation`, `statmech.ising.free-energy`.
- **Vectors and matrices as books treat them** (10; levels 0, 2, 3, 4, 1). The transpose
  of a vector, \(\mathbf{x}^T\); a function applied to each entry of a vector; a vector
  compared with its length, `r = |\mathbf{r}|`. `ml.kernels.gram-matrix`,
  `ml.network.forward-pass`, `cm.central.runge-lenz`.
- **Relations read but not decided** (10; levels 0, 0, 0, 5, 5). \(\sim\), \(\propto\),
  \(\approx\), \(\to\) and a colon. `analysis.taylor.stirling`,
  `astro.expansion.cmb-temperature`.
- **Sums over sets, and terms not written** (7; levels 0, 0, 1, 4, 2). \(\sum\) and
  \(\prod\) without limits, over a set given in words; `\cdots` and `&` for terms not
  written. `graph.trees.spectral`, `de.series.frobenius`.
- **Quantifiers and prose** (7; level 5). \(\forall\), \(\exists\), "converges uniformly
  on". `analysis.inequalities.epsilon-delta`, `analysis.uniform.m-test`.
- **Differential equations written without the unknown's variable** (4; levels 0, 1, 0,
  3, 0). `y''` with no `(x)`, \(\dot a = H_0/a\), and a condition SymPy cannot read.
  `astro.expansion.radiation-scale-factor`, `de.series.euler-exponents`.
- **Formalisms with no design yet** (3; level 5). Time-ordered products, the
  presentation of a group, a random process. `qft.interactions.wick`,
  `geom.topology.klein-bottle`, `sig.random.ar1-spectrum`.

## The first roadmap

The first version of this page, of 21 November 2026, ranked what 276 failing cases of
407 needed, and planned the work in waves: wave 0 was seven guards, each turning a way
the language misread notation into a refusal. The code still cites its guards and
waves, and its section 6, on the decisions of notation, became `design/notation.md`.
It is in the history: `git show 8c7cb39:curriculum/ROADMAP.md`.
