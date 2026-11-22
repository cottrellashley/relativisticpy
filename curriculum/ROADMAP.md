# Climbing the curriculum: a roadmap

*21 November 2026 · branch `v2` · from `uv run python benchmarks/curriculum.py --failures`*

The curriculum has **407 cases** in 18 domains (the folders under `curriculum/`) and 5 levels.
**131 pass.** The other 276 fail: **88 give a wrong answer silently**, **187 are refused** and
**1 times out**. This roadmap names what the language must learn for each failing case, ranks those
capabilities, groups the silent wrong answers by cause, and proposes an order of work.

It classifies by what the language must do, not by today's files: the component named for each
capability is a pass of `design/pipeline.md` (parser, reader, typer, lowerer, rewriter, engines),
an integration of `design/integrations.md` (function, map function, reduction, transform, solver,
constructor, object), or the design itself where the notation needs a decision.

## How it was made, and how far to trust it

- Every failing case was read (program, oracle, the reason it fails) and given the capabilities it
  lacks, **first blocker first**, from one shared vocabulary, so that a capability's count means
  something across domains.
- **Verified by rewriting.** For each case, each missing capability was emulated with notation the
  language reads today (`Positive := [a]` for `a > 0`, `transpose(Q)` for `Q^T`, an explicit
  argument for a function named bare, `doit(...)` around a sum, plain names for primed or
  Greek-subscripted ones...), and the variant was run against the case's own oracle:
  - **181 cases verified in full:** the variant passes, so the listed capabilities are exactly
    what is missing;
  - **76 in part:** the run was shown to reach a capability nothing can emulate today
    (Dirac notation, differential forms, `\sim`...), so later blockers may remain;
  - **19 not verified,** mostly level 5, where no design exists to emulate.
- Verification overturned several plausible guesses: the metric's `r` *is* the user's `r` (the
  Reissner–Nordström `[]` comes from `solve` on an unsimplified `g^{rr}`); a user's
  `R_{\mu\nu\rho\sigma} :=` *does* override the built-in; `\operatorname{Cov}_t(r_t, m_t)` fails
  because `r_t` without braces is a plain symbol, not because of the statistic; repeated lower
  indices over a declared `\mathbb{R}^3` are already summed.
- Expected pass counts below are exact consequences of the classification. Each is given with a
  **floor** that counts only cases verified in full.
- The headline claims were re-checked independently of the workers who classified the domains: 12
  variants run against their oracles, one for each of the capabilities at the top of the ranking,
  all pass.
- The data: `classification.json` (every case: outcome, capabilities, first blocker, variant,
  result, cause, evidence), `capabilities.py`, `waves.json`; scripts in `scripts/`; every variant
  program in `work/g1` … `work/g7`. Nothing in the repository was modified.

## Summary

- **Silent wrong answers come mostly from the reader accepting what it does not know.** A book's
  name it lacks becomes an unknown function (`\Gamma(1/2)`, `J_0(1)`, `\mathcal{L}\{t^2\}`,
  `\operatorname{tr} A` as the symbol tr times A): 29 cases. `:=` silently becomes an equation when
  its left side cannot be defined (`\psi_n(x) :=`, `\Delta\phi :=`): 10. A scripted name becomes a
  new symbol (`c_1` after `c_n :=`, `x_i` on a list, `f^{(4)}(0)` as a symbol times 0): 7.
- **A wave of seven cheap guards (wave 0) would turn 61 of the 88 wrong answers into honest
  refusals and flag 16 more,** before any new mathematics.
- **The most needed single capability is a decision:** what the language assumes of parameters
  nothing declares (27 cases need it; 6 need nothing else). Books take widths, masses and ħ
  positive; SymPy takes them complex and returns Piecewise conditions.
- **Special functions by their book names** unlock the most cases alone (11 of the 18 that need
  them), then **functions named without their argument** (8 of 13) and **transpose and dagger** (7 of 10).
- **Waves 1 and 2 remove 76 of the 88 wrong answers** and take the curriculum from 131 to about
  236 passing (floor 211). Waves 3 and 4 take it to about 355 (floor 304). The last 52 cases are
  large formalisms (Dirac notation, forms, random variables, Itô calculus) and level-5 designs.
- **107 of the 187 refusals mislead:** an internal error, "SymPy could not compute this", or a
  message that blames the user for notation the language should read. Principle 4 of
  `design/interface.md` is broken more often than it is kept.

## 1. Totals

### By level

| Level | Passes | Wrong | Refused | Timeout | Cases |
|---|---:|---:|---:|---:|---:|
| 1 notation | 47 | 16 | 16 | 0 | 79 |
| 2 computation | 46 | 37 | 45 | 0 | 128 |
| 3 derivation | 30 | 19 | 42 | 1 | 92 |
| 4 specialist | 8 | 11 | 52 | 0 | 71 |
| 5 frontier | 0 | 5 | 32 | 0 | 37 |
| **all** | 131 | 88 | 187 | 1 | 407 |

### By domain

Each level cell is passes/cases, with the failures as wrong (w), refused (r) and timeouts (t).

| Domain | L1 | L2 | L3 | L4 | L5 | Passes | Wrong | Refused | Timeout | Cases |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| analysis | 5/7 (0w 2r) | 3/11 (5w 3r) | 2/9 (5w 2r) | 0/4 (0w 4r) | 0/4 (0w 4r) | 10 | 10 | 15 | 0 | 35 |
| classical-mechanics | 2/3 (1w 0r) | 2/6 (2w 2r) | 1/4 (0w 3r) | 0/5 (2w 3r) | 0/2 (1w 1r) | 5 | 6 | 9 | 0 | 20 |
| complex-analysis | 1/3 (0w 2r) | 3/4 (1w 0r) | 1/4 (1w 2r) | 0/4 (2w 2r) | 0/1 (0w 1r) | 5 | 4 | 7 | 0 | 16 |
| data | 3/3 (0w 0r) | 1/4 (1w 2r) | 4/6 (1w 1r) | 0/1 (0w 1r) | 0/1 (0w 1r) | 8 | 2 | 5 | 0 | 15 |
| data-sources | 0/3 (0w 3r) | 0/4 (0w 4r) | 0/4 (0w 4r) | – | – | 0 | 0 | 11 | 0 | 11 |
| differential-equations | 0/2 (0w 2r) | 1/6 (0w 5r) | 3/7 (1w 3r) | 0/3 (0w 3r) | 0/2 (1w 1r) | 4 | 2 | 14 | 0 | 20 |
| differential-geometry | 4/4 (0w 0r) | 6/7 (1w 0r) | 1/3 (1w 1r) | 3/7 (0w 4r) | 0/1 (0w 1r) | 14 | 2 | 6 | 0 | 22 |
| electromagnetism | 3/4 (0w 1r) | 1/5 (2w 2r) | 3/5 (0w 2r) | 1/4 (1w 2r) | 0/2 (0w 2r) | 8 | 3 | 9 | 0 | 20 |
| fourier-analysis | 1/2 (1w 0r) | 2/5 (3w 0r) | 1/4 (2w 0r 1t) | 0/3 (1w 2r) | 0/1 (1w 0r) | 4 | 8 | 2 | 1 | 15 |
| general-relativity | 5/8 (1w 2r) | 10/12 (1w 1r) | 5/11 (1w 5r) | 4/7 (1w 2r) | 0/6 (0w 6r) | 24 | 4 | 16 | 0 | 44 |
| linear-algebra | 5/7 (1w 1r) | 1/9 (2w 6r) | 0/3 (0w 3r) | 0/1 (0w 1r) | 0/1 (0w 1r) | 6 | 3 | 12 | 0 | 21 |
| probability-and-statistics | 6/7 (0w 1r) | 7/13 (3w 3r) | 2/4 (1w 1r) | 0/6 (1w 5r) | 0/4 (0w 4r) | 15 | 5 | 14 | 0 | 34 |
| quantum-field-theory | 0/2 (2w 0r) | 0/2 (0w 2r) | 0/1 (0w 1r) | 0/5 (1w 4r) | 0/6 (1w 5r) | 0 | 4 | 12 | 0 | 16 |
| quantum-mechanics | 3/6 (3w 0r) | 1/16 (5w 10r) | 1/12 (3w 8r) | 0/10 (0w 10r) | 0/1 (0w 1r) | 5 | 11 | 29 | 0 | 45 |
| special-functions | 1/7 (6w 0r) | 3/9 (5w 1r) | 0/2 (2w 0r) | 0/5 (0w 5r) | 0/1 (0w 1r) | 4 | 13 | 7 | 0 | 24 |
| special-relativity | 4/4 (0w 0r) | 2/5 (2w 1r) | 3/5 (0w 2r) | 0/2 (1w 1r) | 0/2 (1w 1r) | 9 | 4 | 5 | 0 | 18 |
| statistical-mechanics | 3/4 (1w 0r) | 1/6 (4w 1r) | 3/5 (1w 1r) | 0/2 (1w 1r) | 0/1 (0w 1r) | 7 | 7 | 4 | 0 | 18 |
| vector-calculus | 1/3 (0w 2r) | 2/4 (0w 2r) | 0/3 (0w 3r) | 0/2 (0w 2r) | 0/1 (0w 1r) | 3 | 0 | 10 | 0 | 13 |

### The wrong answers by type

| Type | Cases | What the language did |
|---|---:|---|
| a definite wrong value | 32 | 0, `[]`, False, another number or expression |
| the question handed back | 30 | a Sum, an Integral, an unknown function, a symbol such as `c_1`, shown as the result |
| an undecided claim | 20 | an equation or inequality shown where True or False was asked |
| right, in another form | 5 | a Piecewise of conditions around the right answer |
| invalid notation accepted | 1 | `L_i = I_{ij} \omega_i` relabelled instead of refused |

### Where it is hard

- Level 1 still fails 32 of 79: the notation of textbooks is the gap, not hard mathematics. Half of
  those (16) are silent wrong answers. Half of those (8) are book names read as unknown functions,
  such as Γ(1/2), J_0(1), ζ(2), P_3(x), ℒ{t²} and tr A, and 4 are sums or integrals left undone.
- Level 4 passes 8 of 71 and level 5 none of 37, as designed.
- **Quantum mechanics** (5 of 45) and **quantum field theory** (0 of 16) are the weakest domains;
  **data-sources** (0 of 11) waits first on one capability, a Python scope, in 9 of its 11 cases.
- **Special functions** (13 wrong of 20 failing) and **Fourier analysis** (8 of 11) fail mostly by
  silent non-answers.

## 2. Capabilities, ranked

136 capabilities account for every failure. They are ranked by the cases each unlocks
**alone** (cases that need nothing else), then by the cases it is **needed** for.

- **First:** the cases it blocks first.
- **Wrong:** wrong answers among the cases that need it.
- **Size:** S is a few days, M a week or two, L a month or more.

`*` marks the cases it blocks first. Component names the pass of the pipeline or the kind of
integration that changes; `design` means the notation needs a decision first (section 6).

### The top 20

| # | Capability | Alone | Needed | Wrong | Component | Size |
|---:|---|---:|---:|---:|---|:-:|
| 1 | `special-functions`: \Gamma, B, \zeta, J_\nu, P_n, H_n, Y_l^m, erfc, Log read as special functions by their book names | 11 | 18 | 17 | integration:function | M |
| 2 | `fn-bare`: a function named without its argument, as y after y(x) is defined (in d/dx y, \dot a, y'', products) | 8 | 13 | 2 | reader | M |
| 3 | `transpose-dagger`: the transpose A^T, A^\top and the Hermitian conjugate A^\dagger of a matrix | 7 | 10 | 0 | reader+design | S |
| 4 | `default-assumptions`: what the language assumes of parameters nothing declares: real, and positive where books take a scale (a width, mass, frequency, hbar, time, a count), stated in a note; so moduli, conjugates, convergent integrals and limits come out in closed form, not as Piecewise conditions | 6 | 27 | 16 | design+lowerer | M |
| 5 | `matrix-functions`: functions of a matrix written as on paper: \operatorname{tr} A, \operatorname{rank} A, e^{At}, \exp(-i\theta\sigma_y/2), V^{1/2}, \Lambda^k, \log_2\rho | 6 | 11 | 4 | integration:map-function | M |
| 6 | `label-subscript`: a subscript that labels rather than indexes, on any base: \lambda_1, \delta_0, V_{10}, \sigma_x := matrix, R_\odot, r_+, h_\times (part of the name, not a component) | 5 | 15 | 1 | reader | M |
| 7 | `residues-contours`: contour integrals \oint_{&#124;z&#124;=r} and residues \operatorname{Res}_{z=a} (by Laurent series), principal values | 5 | 6 | 2 | integration+reader | M |
| 8 | `quantifiers`: quantified and prose claims: \forall, \exists, 'converges uniformly on', images of sets | 5 | 6 | 0 | design | L |
| 9 | `identity-matrix`: the identity written I, 1, \mathbf{1}, \mathbb{1} in matrix arithmetic (A - \lambda I), and a number plus a matrix | 4 | 7 | 0 | reader+design | S |
| 10 | `python-scope`: a Python scope for \texttt{name}: Workbook(scope=...) and object conversions | 3 | 9 | 0 | workbook+integration:object | M |
| 11 | `closed-form-sums`: evaluating symbolic and infinite sums that have a closed form, rather than keeping them as written | 3 | 6 | 5 | design+engine | S |
| 12 | `laplace-fourier`: Laplace and Fourier transforms by their book names: \mathcal{L}\{f\}(s), \mathcal{F}, and the pair f \supset F | 3 | 5 | 4 | integration:transform | M |
| 13 | `random-variables`: random variables given by distributions, X \sim Poisson(\lambda), P(X = 2), \mathbb{E}[X], Var(aX + b), X &#124; N | 3 | 5 | 1 | integration:constructor+reduction | L |
| 14 | `curve-tensors`: tensors along a curve, x^\mu(\lambda), V^\mu(\lambda), and d/d\lambda of them in index notation | 3 | 3 | 1 | typer+lowerer | M |
| 15 | `hypergeometric`: hypergeometric functions {}_pF_q(a, b; c; z), with ; separating the parameter lists | 3 | 3 | 0 | parser+integration:function | S |
| 16 | `family`: a name defined with a free subscript is a family: c_n := ..., \psi_n(x) := ..., E_n := ...; then c_1, \psi_3(x), E_1, a_{n+1} evaluate it | 2 | 11 | 7 | reader | M |
| 17 | `vector-algebra`: 3-vectors as lists: &#124;v&#124;, \&#124;v\&#124;, v \cdot w, v \times w, \partial_\theta v and \int v dt component by component | 2 | 11 | 2 | engine | M |
| 18 | `primed-names`: a prime as part of a name: x', t', E', H'(x), F'^{\mu\nu}, k'^\mu, and primed indices \mu' (a new name, not a derivative) | 2 | 11 | 0 | reader+design | M |
| 19 | `vector-calculus`: \nabla f, \nabla \cdot v, \nabla \times v, \nabla^2 on functions of x, y, z, and in curvilinear coordinates from a line element | 2 | 10 | 0 | integration+reader | M |
| 20 | `dirac-notation`: Dirac notation: kets, bras and brackets, &#124;\psi>, <\phi&#124;, <\phi&#124;\psi>, <\phi&#124;A&#124;\psi>, &#124;a><b&#124; | 2 | 9 | 1 | parser+design+engine | L |

Capabilities needed by many cases but rarely enough alone matter as much as the top of the table:

| Capability | Needed | Alone |
|---|---:|---:|
| `default-assumptions` | 27 | 6 |
| `label-subscript` | 15 | 5 |
| `subscripted-function` | 12 | 1 |
| `family` | 11 | 2 |
| `primed-names` | 11 | 2 |
| `vector-algebra` | 11 | 2 |

They sit under the others: `subscripted-function` blocks first in 11 cases.

### Every capability

| # | Capability | What the language must do | Alone | Needed | First | Wrong | Domains | Component | Size | Cases (blocked first marked *) |
|---:|---|---|---:|---:|---:|---:|---|---|:-:|---|
| 1 | `special-functions` | \Gamma, B, \zeta, J_\nu, P_n, H_n, Y_l^m, erfc, Log read as special functions by their book names | 11 | 18 | 17 | 17 | 5 | integration:function | M | *complex.maps.principal-log, qm.oscillator.hermite-wavefunction, *em.sources.legendre, *special.bessel.value, *special.bessel.equation, *special.bessel.half-order, *special.erf.asymptotic, *special.gamma.half, *special.gamma.beta, *special.gamma.reflection, *special.zeta.two, *special.zeta.functional-equation, *special.legendre.p3, *special.hermite.orthogonality, *special.legendre.generating-function, *special.harmonics.y10, *special.harmonics.unsold, *prob.distributions.gamma-variance |
| 2 | `fn-bare` | a function named without its argument, as y after y(x) is defined (in d/dx y, \dot a, y'', products) | 8 | 13 | 10 | 2 | 5 | reader | M | *analysis.derivatives.dot-squared, *de.first-order.exact, *de.second-order.wronskian, *de.second-order.harmonic, *de.second-order.resonance, *de.second-order.system, *de.series.frobenius, qm.angular-momentum.l-squared, qm.oscillator.ground-state, qm.hydrogen.radial-equation, *qft.klein-gordon.dispersion, *gr.cosmology.hubble-matter, *gr.cosmology.radiation-era |
| 3 | `transpose-dagger` | the transpose A^T, A^\top and the Hermitian conjugate A^\dagger of a matrix | 7 | 10 | 9 | 0 | 4 | reader+design | S | *qm.dirac.change-of-basis, qm.perturbation.sum-over-states, *qm.time-evolution.unitarity, *qm.time-evolution.precession, *sr.lorentz.preserves-metric, *la.eigen.spectral, *la.orthogonality.least-squares, *la.orthogonality.qr, *la.orthogonality.svd, *prob.estimation.normal-equations |
| 4 | `default-assumptions` | what the language assumes of parameters nothing declares: real, and positive where books take a scale (a width, mass, frequency, hbar, time, a count), stated in a note; so moduli, conjugates, convergent integrals and limits come out in closed form, not as Piecewise conditions | 6 | 27 | 6 | 16 | 7 | design+lowerer | M | analysis.uniform.interchange, qm.oscillator.hermite-wavefunction, qm.hydrogen.mean-radius, *qm.square-well.normalisation, qm.square-well.orthogonality, qm.square-well.expansion, *qm.scattering.soft-sphere, qm.time-evolution.beats, qm.commutators.gaussian-packet, qm.dirac.momentum-wavefunction, qm.square-well.uncertainty, qm.perturbation.first-order, qm.perturbation.second-order, qm.perturbation.stark, qm.time-evolution.unitarity, *qft.interactions.gaussian, fa.transforms.pair, statmech.canonical.oscillator, *statmech.canonical.ideal-gas, *statmech.canonical.rigid-rotor, statmech.quantum.planck-distribution, statmech.quantum.stefan-boltzmann, *prob.distributions.poisson-limit, prob.distributions.gamma-variance, cm.hamiltonian.canonical-bracket, cm.hamiltonian.action-oscillator, cm.hamiltonian.hamilton-jacobi |
| 5 | `matrix-functions` | functions of a matrix written as on paper: \operatorname{tr} A, \operatorname{rank} A, e^{At}, \exp(-i\theta\sigma_y/2), V^{1/2}, \Lambda^k, \log_2\rho | 6 | 11 | 6 | 4 | 4 | integration:map-function | M | *qm.information.purity, *qm.information.entropy, qm.spin.rotation, qm.time-evolution.unitarity, qm.time-evolution.precession, qft.dirac.trace, *la.eigen.powers, la.eigen.exponential, *la.matrices.trace, *la.matrices.rank, *prob.estimation.correlation-matrix |
| 6 | `label-subscript` | a subscript that labels rather than indexes, on any base: \lambda_1, \delta_0, V_{10}, \sigma_x := matrix, R_\odot, r_+, h_\times (part of the name, not a component) | 5 | 15 | 9 | 1 | 3 | reader | M | *qm.angular-momentum.singlet, qm.oscillator.potential-energy, qm.square-well.uncertainty, qm.perturbation.second-order, qm.perturbation.stark, *qm.scattering.hard-sphere, *qm.spin.pauli-commutator, qm.spin.bra-ket-expectation, *qm.spin.rotation, *gr.black-holes.reissner-nordstrom-temperature, gr.black-holes.kerr-horizon-area, *gr.formalisms.newman-penrose-psi2, *gr.geodesics.light-deflection-sun, *gr.waves.linearised-riemann, *statmech.ising.free-energy |
| 7 | `residues-contours` | contour integrals \oint_{&#124;z&#124;=r} and residues \operatorname{Res}_{z=a} (by Laurent series), principal values | 5 | 6 | 5 | 2 | 2 | integration+reader | M | *complex.maps.argument-principle, *complex.residues.essential-singularity, *complex.residues.residue-theorem, *complex.residues.real-integral, *complex.residues.principal-value, qft.klein-gordon.feynman-propagator |
| 8 | `quantifiers` | quantified and prose claims: \forall, \exists, 'converges uniformly on', images of sets | 5 | 6 | 6 | 0 | 3 | design | L | *analysis.inequalities.epsilon-delta, *analysis.uniform.not-uniform, *analysis.uniform.m-test, *analysis.uniform.uniform-continuity, *complex.maps.joukowski, *gr.black-holes.null-energy-condition |
| 9 | `identity-matrix` | the identity written I, 1, \mathbf{1}, \mathbb{1} in matrix arithmetic (A - \lambda I), and a number plus a matrix | 4 | 7 | 4 | 0 | 5 | reader+design | S | qft.dirac.plane-wave, qft.dirac.clifford, qft.dirac.slash, *statmech.ising.transfer-eigenvalues, *la.eigen.characteristic, *prob.markov.absorbing, *cm.oscillations.principal-moments |
| 10 | `python-scope` | a Python scope for \texttt{name}: Workbook(scope=...) and object conversions | 3 | 9 | 9 | 0 | 1 | workbook+integration:object | M | *data.objects.column, *data.objects.dataframe, *data.objects.key, *data.objects.constants, *data.objects.function, *data.objects.sympy, *data.fft.named, *data.fft.dft-sum, *data.objects.aligned |
| 11 | `closed-form-sums` | evaluating symbolic and infinite sums that have a closed form, rather than keeping them as written | 3 | 6 | 6 | 5 | 4 | design+engine | S | *analysis.series.basel, *analysis.series.harmonic, *qm.hydrogen.degeneracy, *fa.series.convergence, *statmech.canonical.oscillator, *statmech.quantum.planck-distribution |
| 12 | `laplace-fourier` | Laplace and Fourier transforms by their book names: \mathcal{L}\{f\}(s), \mathcal{F}, and the pair f \supset F | 3 | 5 | 5 | 4 | 1 | integration:transform | M | *fa.transforms.pair, *fa.laplace.power, *fa.laplace.derivative, *fa.laplace.convolution, *fa.laplace.step |
| 13 | `random-variables` | random variables given by distributions, X \sim Poisson(\lambda), P(X = 2), \mathbb{E}[X], Var(aX + b), X &#124; N | 3 | 5 | 3 | 1 | 1 | integration:constructor+reduction | L | *prob.bayes.total-expectation, *prob.distributions.poisson-tilde, *prob.moments.variance-affine, prob.stochastic.ito-isometry, prob.stochastic.ou-variance |
| 14 | `curve-tensors` | tensors along a curve, x^\mu(\lambda), V^\mu(\lambda), and d/d\lambda of them in index notation | 3 | 3 | 3 | 1 | 3 | typer+lowerer | M | *gr.geodesics.circular-geodesic, *dg.connections.parallel-transport, *cm.lagrangian.index-form |
| 15 | `hypergeometric` | hypergeometric functions {}_pF_q(a, b; c; z), with ; separating the parameter lists | 3 | 3 | 3 | 0 | 1 | parser+integration:function | S | *special.hypergeometric.logarithm, *special.hypergeometric.gauss, *special.hypergeometric.arcsin |
| 16 | `family` | a name defined with a free subscript is a family: c_n := ..., \psi_n(x) := ..., E_n := ...; then c_1, \psi_3(x), E_1, a_{n+1} evaluate it | 2 | 11 | 2 | 7 | 5 | reader | M | *analysis.series.radius-ratio, *de.pde.heat, de.series.airy, qm.oscillator.hermite-wavefunction, qm.square-well.orthogonality, qm.square-well.expansion, qm.time-evolution.beats, qm.perturbation.sum-over-states, sr.four-momentum.mandelstam, sr.four-momentum.two-body-phase-space, qft.dirac.clifford |
| 17 | `vector-algebra` | 3-vectors as lists: &#124;v&#124;, \&#124;v\&#124;, v \cdot w, v \times w, \partial_\theta v and \int v dt component by component | 2 | 11 | 4 | 2 | 6 | engine | M | qm.scattering.born-yukawa, em.field-tensor.e-dot-b, *em.sources.current-loop, em.sources.lienard-wiechert, em.waves.poynting, sr.four-momentum.two-body-phase-space, *dg.surfaces.theorema-egregium, *vc.integrals.line, vc.integrals.divergence-theorem, vc.integrals.stokes, *cm.central.runge-lenz |
| 18 | `primed-names` | a prime as part of a name: x', t', E', H'(x), F'^{\mu\nu}, k'^\mu, and primed indices \mu' (a new name, not a derivative) | 2 | 11 | 6 | 0 | 4 | reader+design | M | qm.oscillator.potential-energy, *qm.perturbation.first-order, qm.perturbation.sum-over-states, *qm.scattering.born-yukawa, *em.field-tensor.boost, *em.field-tensor.e-dot-b, em.sources.lienard-wiechert, *sr.four-momentum.compton, *sr.lorentz.velocity-composition, sr.lorentz.index-form, qft.interactions.muon-pair |
| 19 | `vector-calculus` | \nabla f, \nabla \cdot v, \nabla \times v, \nabla^2 on functions of x, y, z, and in curvilinear coordinates from a line element | 2 | 10 | 9 | 0 | 3 | integration+reader | M | *analysis.multivariable.lagrange, *em.maxwell.gauss, *em.waves.retarded-green, *vc.integrals.divergence-theorem, vc.integrals.stokes, *vc.integrals.delta, *vc.operators.divergence, *vc.operators.curl, *vc.operators.curl-grad, *vc.operators.curl-cylindrical |
| 20 | `dirac-notation` | Dirac notation: kets, bras and brackets, &#124;\psi>, <\phi&#124;, <\phi&#124;\psi>, <\phi&#124;A&#124;\psi>, &#124;a><b&#124; | 2 | 9 | 9 | 1 | 2 | parser+design+engine | L | *qm.angular-momentum.clebsch-gordan, *qm.dirac.inner-product, *qm.dirac.completeness, *qm.dirac.momentum-wavefunction, *qm.oscillator.potential-energy, *qm.perturbation.sum-over-states, *qm.information.partial-trace, *qm.spin.bra-ket-expectation, *qft.interactions.wick |
| 21 | `decide-identities` | deciding identities that the quick zero test cannot: simplify and trigonometric/hyperbolic expansion, for scalars and for each component of a matrix equation, special-function identities, and numerical evidence at random points under its own verdict | 2 | 7 | 2 | 4 | 3 | engine+design | M | *sr.lorentz.rapidities-add, sr.lorentz.index-form, special.bessel.half-order, special.zeta.functional-equation, special.zeta.euler-product, special.harmonics.unsold, *data.finance.black-scholes-pde |
| 22 | `operators` | differential operators as values: \hat p := -i\hbar d/dx, L_x := ..., (-\hbar^2 d^2/dx^2)\psi(x), applied and composed; commutators [A, B] | 2 | 7 | 3 | 0 | 3 | design+reader+engine | L | *qm.commutators.canonical, *qm.commutators.angular-momentum, qm.commutators.gaussian-packet, *qm.oscillator.ladder-commutator, qm.square-well.uncertainty, em.waves.retarded-green, la.eigen.zeta-determinant |
| 23 | `levi-civita` | the Levi-Civita symbol \epsilon_{ijk}, \epsilon^{\mu\nu\rho\sigma}, with its sign convention | 2 | 5 | 4 | 1 | 4 | integration:constructor | S | *em.field-tensor.dual-invariant, *dg.index-notation.epsilon-delta, vc.operators.curl-curl, *la.index.epsilon-delta, *la.index.bac-cab |
| 24 | `differential-forms` | differential forms: one-forms in dx, d, \wedge, \star, integrals of forms | 2 | 5 | 4 | 0 | 2 | design+engine | L | dg.connections.spin-connection, *dg.forms.exterior-derivative, *dg.forms.hodge-star, *dg.forms.stokes-theorem, *vc.integrals.form |
| 25 | `stochastic-calculus` | Ito calculus: dS = \mu S dt + \sigma S dz, Brownian motion B_t, \int B dB | 2 | 5 | 5 | 0 | 2 | design+integration | L | *prob.stochastic.ito-log, *prob.stochastic.ito-integral, *prob.stochastic.ito-isometry, *prob.stochastic.ou-variance, *data.finance.black-scholes-derivation |
| 26 | `inequality-assumptions` | assumptions written as relations: a > 0, a, b > 0, t > 0, x \in \mathbb{R} | 2 | 4 | 3 | 2 | 3 | reader+design | S | *analysis.inequalities.am-gm, *analysis.integrals.gaussian, *complex.maps.cayley, qft.klein-gordon.feynman-propagator |
| 27 | `extremum` | \max_x, \min_x, \arg\min, \arg\max over a variable or an index, with constraints | 2 | 4 | 3 | 0 | 3 | reader+integration:solver | M | *cm.hamiltonian.legendre, *data.portfolio.argmin, data.objects.key, *data.fft.sampled-tone |
| 28 | `functionals` | operators and functionals with square-bracket arguments, L[y] := ..., R[y], S[q], and functional derivatives \delta S/\delta q | 2 | 4 | 3 | 0 | 3 | parser+reader | M | *de.series.euler-exponents, *de.bvp.rayleigh, *qft.interactions.generating-functional, cm.lagrangian.hamilton-principle |
| 29 | `conjugate-bar` | \bar z, \overline{...} and \bar{m}^\rho as the complex conjugate where the quantity is complex (and \partial/\partial\bar z) | 2 | 3 | 2 | 1 | 2 | reader+design | S | *complex.analytic.wirtinger, *complex.numbers.conjugate, gr.formalisms.newman-penrose-psi2 |
| 30 | `normal-cdf` | N(x) and \Phi(x) as the standard normal distribution function where finance and statistics books mean it | 2 | 2 | 2 | 2 | 2 | design+integration:function | S | *prob.limits.normal-approximation, *data.finance.black-scholes-call |
| 31 | `nth-derivative` | the nth derivative written f^{(n)}(x) | 2 | 2 | 2 | 2 | 2 | reader | S | *analysis.taylor.nth-derivative, *prob.moments.normal-fourth-moment |
| 32 | `limit-dne` | a limit that does not exist is reported as such; a derivative at a point is a limit (a piecewise function's derivative at its seam) | 2 | 2 | 2 | 1 | 1 | engine | S | *analysis.derivatives.not-differentiable, *analysis.limits.does-not-exist |
| 33 | `coordinate-transformations` | changing coordinates: new coordinates defined from old ones (v := t + r^*, t \pm r = tan(...)) transform the metric | 2 | 2 | 2 | 0 | 1 | engine+reader | M | *gr.formalisms.penrose-compactification, *gr.spacetimes.eddington-finkelstein |
| 34 | `inf-sup` | \inf, \sup, \limsup, \liminf, and set-builder sets {f(n) : n = 1, 2, ...} | 2 | 2 | 2 | 0 | 1 | parser+engine | M | *analysis.inequalities.infimum, *analysis.series.limsup |
| 35 | `subscripted-function` | a function whose name is a Greek or built-in letter with a subscript, defined and called: \psi_{100}(r), \psi_1(x), R_{21}(r) (today \psi_1(x) := ... is read as the product \psi_1 x) | 1 | 12 | 11 | 7 | 2 | reader | S | *qm.oscillator.ground-state, *qm.oscillator.hermite-wavefunction, *qm.hydrogen.mean-radius, *qm.square-well.orthogonality, *qm.square-well.expansion, *qm.time-evolution.beats, *qm.hydrogen.radial-equation, *qm.square-well.uncertainty, qm.perturbation.first-order, *qm.perturbation.second-order, *qm.perturbation.stark, *cm.central.circular-orbit |
| 36 | `superscript-label` | a superscript that labels on the left of :=: \sigma^2 := ..., <x^2> := ..., \Delta_0^{(2)}, E_n^{(0)} (a name, not a power or an index) | 1 | 8 | 3 | 1 | 2 | reader | S | *qm.angular-momentum.l-squared, *qm.commutators.gaussian-packet, qm.square-well.uncertainty, qm.perturbation.second-order, qm.perturbation.sum-over-states, qm.information.partial-trace, qm.scattering.born-yukawa, *prob.limits.chebyshev |
| 37 | `index-ranges` | an index range written beside a definition, n = 0, ..., N-1 or i, j = 1, 2, 3, declares the index's space | 1 | 6 | 1 | 1 | 3 | reader+typer | M | fa.transforms.dft, *la.index.delta-trace, data.objects.function, data.grids.uniform, data.fft.dft-sum, data.fft.sampled-tone |
| 38 | `region-integrals` | integrals over a region or with a measure: \iint_{x^2+y^2 \le 1} f dA, \int_D, d^3x, d^4x | 1 | 6 | 1 | 0 | 6 | parser+engine | M | *analysis.multivariable.region, qm.scattering.born-yukawa, sr.four-momentum.two-body-phase-space, qft.interactions.generating-functional, dg.forms.stokes-theorem, vc.integrals.form |
| 39 | `space-index-sums` | \sum_{t=1}^{n} over an index of a declared space binds that index | 1 | 6 | 1 | 0 | 2 | typer | S | data.learning.softmax, *data.learning.pca, data.objects.column, data.objects.function, data.grids.uniform, data.fft.dft-sum |
| 40 | `frontier-qft` | path integrals, Feynman amplitudes, Wick contractions, loop integrals in d dimensions, running couplings | 1 | 5 | 3 | 1 | 2 | design | L | *qm.time-evolution.path-integral, qft.interactions.wick, *qft.interactions.phi4-vertex, *qft.interactions.muon-pair, qft.interactions.loop-integral |
| 41 | `implicit-relations` | an equation as a constraint on what follows: dy/dx after x^3 + y^3 = 6xy, \partial m/\partial h after m = tanh(...), \oint p dq on the orbit | 1 | 4 | 2 | 3 | 3 | design+reader+engine | M | analysis.derivatives.implicit, *statmech.meanfield.curie-weiss, statmech.meanfield.critical-exponent, *cm.hamiltonian.action-oscillator |
| 42 | `distribution-limits` | distributional results: \int e^{-i\omega t} dt = 2\pi\delta(\omega), \lim K = \delta(x), \nabla \cdot (\hat r/r^2) = 4\pi\delta^3 | 1 | 4 | 2 | 1 | 4 | engine | L | *de.pde.heat-kernel, em.waves.retarded-green, *fa.transforms.delta, vc.integrals.delta |
| 43 | `unbraced-index` | a one-letter subscript without braces, r_t, is an index when t is a declared index label, as r_{t} is (today r_t := [...] defines the symbol r_t, and an undefined a_j is the symbol a_j) | 1 | 4 | 2 | 0 | 3 | reader | S | la.index.bac-cab, *prob.markov.stationary-index, data.learning.softmax, *data.portfolio.capm-beta |
| 44 | `decide-inequalities` | deciding an inequality claim for all values of its variables (e^x >= 1 + x, AM-GM) | 1 | 3 | 1 | 2 | 2 | engine | M | *analysis.inequalities.exp-tangent, analysis.inequalities.am-gm, gr.black-holes.null-energy-condition |
| 45 | `compound-names` | \Delta\phi, \Delta x := ... as one quantity's name (today := gives an equation \Delta\phi = ... and later uses are \Delta\cdot\phi) | 1 | 3 | 2 | 1 | 2 | reader+design | S | *gr.geodesics.mercury-perihelion, gr.geodesics.light-bending, *data.grids.uniform |
| 46 | `kronecker-product` | \otimes between matrices is the Kronecker product | 1 | 3 | 1 | 1 | 1 | engine | S | *qm.information.tensor-product, qm.angular-momentum.singlet, qm.information.partial-trace |
| 47 | `poisson-bracket` | the Poisson bracket {f, g} over canonical pairs (x, p_x), and {q, p}_{Q,P} | 1 | 3 | 2 | 1 | 1 | reader+design | S | cm.central.runge-lenz, *cm.hamiltonian.angular-momentum, *cm.hamiltonian.canonical-bracket |
| 48 | `asymptotic-relations` | \sim and \approx as claims: asymptotic equivalence (lim f/g = 1), scaling exponents, approximations to an order | 1 | 3 | 2 | 0 | 3 | design+engine | M | *analysis.taylor.stirling, gr.geodesics.perihelion-advance, *statmech.meanfield.critical-exponent |
| 49 | `operator-algebra` | abstract operators with commutation relations: [\hat x, \hat p] = i\hbar, then [\hat x^3, \hat p] | 1 | 3 | 1 | 0 | 1 | engine | M | *qm.commutators.power-of-x, qm.oscillator.potential-energy, qm.perturbation.sum-over-states |
| 50 | `special-integrals` | definite integrals SymPy leaves undone but which have a closed form: \int_0^\infty x^3/(e^x - 1) dx = \Gamma(4)\zeta(4) | 1 | 2 | 2 | 2 | 1 | engine | S | *statmech.quantum.planck-integral, *statmech.quantum.stefan-boltzmann |
| 51 | `order-terms` | an equation with an O(x^n) term is a claim about an expansion to that order | 1 | 2 | 1 | 1 | 2 | engine | S | *analysis.taylor.big-o, gr.geodesics.light-bending |
| 52 | `piecewise-removable` | no Piecewise branch for a removable singularity: the sinc function | 1 | 2 | 1 | 1 | 1 | engine | S | *fa.transforms.rectangle, fa.transforms.pair |
| 53 | `solid-angle` | d\Omega in an integral is the measure sin\theta d\theta d\phi over the sphere | 1 | 2 | 1 | 1 | 1 | reader | S | *qft.interactions.solid-angle, qft.interactions.muon-pair |
| 54 | `solve-simplify` | solve simplifies first: cancels common factors (so roots that also zero a removable denominator are kept) and applies trigonometric identities (so it never divides by a coefficient that vanishes) | 1 | 2 | 1 | 1 | 2 | engine | S | sr.four-momentum.compton, *gr.black-holes.reissner-nordstrom-horizons |
| 55 | `sum-index-integer` | a summation index is an integer, so integrals and terms with sin(n\pi) simplify (no Piecewise for n = 0) | 1 | 2 | 1 | 1 | 1 | lowerer | S | fa.series.convergence, *fa.series.parseval |
| 56 | `d-variable` | d and D as ordinary variables (a diameter d^2, D^2) when no derivative is meant | 1 | 2 | 1 | 0 | 2 | reader | S | qm.dirac.momentum-wavefunction, *em.sources.loop-moment |
| 57 | `bar-precedence` | an evaluation bar \left. V \right&#124;_{q=p/d} written inside a limit applies before the limit | 1 | 1 | 1 | 1 | 1 | parser | S | *em.sources.dipole |
| 58 | `definition-index-check` | a definition's left side must carry exactly the free indices of its right side (L_i := I_{ij} omega_i is refused, not relabelled by position) | 1 | 1 | 1 | 1 | 1 | typer | S | *cm.oscillations.mismatched-index |
| 59 | `function-of-tensor-body` | a function whose body is computed from tensor components substitutes its argument into them: \omega(r) := -k_\mu u^\mu, then \omega(r_e) | 1 | 1 | 1 | 1 | 1 | lowerer | S | *gr.schwarzschild.redshift |
| 60 | `jacobian-notation` | the Jacobian determinant \partial(x, y)/\partial(r, \theta) | 1 | 1 | 1 | 1 | 1 | reader | S | *analysis.multivariable.jacobian |
| 61 | `label-index-subscript` | a label then an index in one subscript: k_{1\mu} is the vector k_1 with index \mu | 1 | 1 | 1 | 1 | 1 | reader | S | *sr.four-momentum.two-photon-mass |
| 62 | `matrix-factorisation-equations` | a matrix equation with named unknown factors, B_y B_x = R_z(\theta) B(u): the polar decomposition, with R_z(\theta) and B(u) by their book names | 1 | 1 | 1 | 1 | 1 | engine | L | *sr.lorentz.wigner-rotation |
| 63 | `bessel-zeros` | zeros of Bessel functions j_{\nu,k} | 1 | 1 | 1 | 0 | 1 | integration:function | S | *special.bessel.zeros |
| 64 | `embedding-indices` | indices over two spaces in one expression: an embedding X^i(\theta, \phi), i over R^3 and a, b over the surface | 1 | 1 | 1 | 0 | 1 | typer | M | *dg.surfaces.induced-metric |
| 65 | `eval-bracket` | the evaluation bracket [F(x)]_a^b | 1 | 1 | 1 | 0 | 1 | parser | S | *analysis.integrals.by-parts |
| 66 | `exp-not-tensor` | e^{...} is always the exponential: e^{i\theta} is never a tensor e with indices i, \theta | 1 | 1 | 1 | 0 | 1 | reader | S | *complex.numbers.euler |
| 67 | `kl-divergence` | D(p \&#124; q), the relative entropy | 1 | 1 | 1 | 0 | 1 | parser+integration:reduction | S | *prob.information.kl-notation |
| 68 | `line-element-products` | a line element with a square of a sum of differentials, (dr + \beta dt)^2 | 1 | 1 | 1 | 0 | 1 | reader | S | *gr.formalisms.painleve-gullstrand-lapse |
| 69 | `matrix-similarity` | A \sim B between matrices is similarity, decided by Jordan normal forms | 1 | 1 | 1 | 0 | 1 | design+engine | S | *la.eigen.jordan |
| 70 | `periodic-identification` | \tau \sim \tau + \beta as a periodic identification, its period fixed by regularity of the Euclidean metric at the horizon | 1 | 1 | 1 | 0 | 1 | design+engine | L | *gr.black-holes.euclidean-period |
| 71 | `pochhammer` | the Pochhammer symbol (a)_n | 1 | 1 | 1 | 0 | 1 | reader+design | S | *special.hypergeometric.pochhammer |
| 72 | `probability-notation` | \Pr(A \mid B) with values assigned to events | 1 | 1 | 1 | 0 | 1 | parser+reader | S | *prob.bayes.conditional-notation |
| 73 | `staggered-unbraced` | staggered indices after an unbraced script: R_i{}^k reads as R_{i}{}^{k} (R^i{}_j already works) | 1 | 1 | 1 | 0 | 1 | parser | S | *gr.waves.polarisation-rotation |
| 74 | `thermo-partials` | (\partial S/\partial V)_T: a subscript on a parenthesised derivative names what is held fixed | 1 | 1 | 1 | 0 | 1 | parser+reader | S | *statmech.thermo.maxwell-relation |
| 75 | `user-notation` | defining notation: <p, q> := \int ..., \&#124;p\&#124; := ... | 1 | 1 | 1 | 0 | 1 | reader+design | M | *la.orthogonality.gram-schmidt |
| 76 | `ode-unknown` | the unknown of a differential equation named without its argument (dy/dt = r y, t y' + 2y = 4t^2), with conditions written y(0) = y_0, y'(0) = 0 | 0 | 9 | 5 | 1 | 5 | reader | M | de.first-order.growth, de.first-order.linear, *de.first-order.separable, *de.second-order.impulse, *de.bvp.robin, qft.interactions.running-coupling, *gr.geodesics.perihelion-advance, *la.eigen.zeta-determinant, data.finance.black-scholes-derivation |
| 77 | `unit-vectors` | unit vectors \hat{x}, \hat{y}, \hat{z} (and \hat{r}, \hat\varphi in curvilinear coordinates) as basis vectors | 0 | 8 | 1 | 1 | 2 | reader+design | S | *em.waves.poynting, vc.integrals.line, vc.integrals.divergence-theorem, vc.integrals.stokes, vc.integrals.delta, vc.operators.divergence, vc.operators.curl, vc.operators.curl-cylindrical |
| 78 | `accent-script` | a script after an accent applies to the accented name: \dot\theta^2 is (\dot\theta)^2 | 0 | 7 | 4 | 0 | 3 | parser | S | analysis.derivatives.dot-squared, qft.klein-gordon.hamiltonian, *cm.lagrangian.pendulum, *cm.lagrangian.bead-on-hoop, *cm.lagrangian.double-pendulum, cm.lagrangian.noether, *cm.lagrangian.hamilton-principle |
| 79 | `star-conjugate` | a superscript * is the complex conjugate: ψ^*, ψ(x)^*, ψ^*(x) (today u^* is a new name ignoring u's value, and f(x)^* is a power by a symbol named *) | 0 | 6 | 0 | 2 | 1 | reader+design | S | qm.square-well.orthogonality, qm.square-well.expansion, qm.commutators.gaussian-packet, qm.perturbation.first-order, qm.perturbation.second-order, qm.perturbation.stark |
| 80 | `step-delta` | the Heaviside step u_c(t) and the Dirac delta \delta(x) as functions/distributions | 0 | 6 | 0 | 1 | 5 | integration:function | S | de.pde.heat-kernel, de.second-order.impulse, em.waves.retarded-green, sr.four-momentum.two-body-phase-space, qft.interactions.generating-functional, fa.laplace.step |
| 81 | `metric-without-coordinates` | a metric given by its components alone, \eta_{\mu\nu} = diag(-1, 1, 1, 1), and \eta or g as Minkowski where nothing is declared, so indices range over 4 dimensions without coordinates | 0 | 5 | 1 | 1 | 2 | reader+typer+design | M | sr.four-momentum.two-body-phase-space, *sr.lorentz.index-form, qft.dirac.clifford, qft.dirac.trace, qft.klein-gordon.euler-lagrange |
| 82 | `gamma-matrices` | Dirac matrices: \gamma^\mu as a matrix-valued vector, the Clifford algebra, traces of products, \slashed{p} | 0 | 4 | 2 | 1 | 1 | integration:constructor | L | qft.dirac.clifford, *qft.dirac.trace, *qft.dirac.slash, qft.interactions.muon-pair |
| 83 | `differential-numerator` | an integral whose differential is in a fraction's numerator, followed by more factors or terms: \int dr/(...) - \pi, \int dp^0/(2\pi) f | 0 | 4 | 4 | 0 | 3 | parser | S | *sr.four-momentum.two-body-phase-space, *qft.interactions.loop-integral, *qft.klein-gordon.feynman-propagator, *gr.geodesics.light-bending |
| 84 | `ode-conditions` | dsolve with initial and boundary conditions written as equations, y(0) = y_0, y'(0) = 0, choosing the branch they select (today an internal error, even with y(t) written out) | 0 | 4 | 2 | 0 | 1 | reader+integration:solver | S | *de.first-order.growth, *de.first-order.linear, de.first-order.separable, de.second-order.impulse |
| 85 | `typeface-names` | a typeface makes a distinct name: \mathcal{F} is not F, \mathbf{r} is not r, \boldsymbol{\beta} is not \beta (today every typeface is read as the plain letter) | 0 | 4 | 1 | 0 | 2 | reader+design | S | em.field-tensor.dual-invariant, *em.sources.lienard-wiechert, em.waves.retarded-green, cm.central.runge-lenz |
| 86 | `latex-per-statement` | a statement of a LaTeX program without a backslash is read as plain text, so 6xy is the one name xy | 0 | 3 | 2 | 2 | 2 | lexer/reader | S | *analysis.derivatives.implicit, analysis.multivariable.second-derivative-test, *la.eigen.exponential |
| 87 | `list-subscript` | a subscript on a list-valued name selects its element: x_i for x := [2, 4, ...] (base from the sum's range) | 0 | 3 | 2 | 2 | 3 | reader+design | S | *fa.transforms.dft, prob.estimation.sample-variance, *cm.lagrangian.noether |
| 88 | `component-definitions` | defining a tensor one component at a time: e^1 := d\theta, \gamma^0 := (matrix) | 0 | 3 | 3 | 0 | 2 | reader+typer | M | *qft.dirac.plane-wave, *qft.dirac.clifford, *dg.connections.spin-connection |
| 89 | `field-derivatives` | derivatives of a field that has no components, \nabla\phi and \partial_\mu\phi, as quantities of their own: (\nabla\phi)^2, \partial\mathcal{L}/\partial(\partial_\mu\phi) | 0 | 3 | 2 | 0 | 1 | reader+engine | M | qft.interactions.phi4-vertex, *qft.klein-gordon.hamiltonian, *qft.klein-gordon.euler-lagrange |
| 90 | `lagrangian-velocities` | generalised coordinates: q and \dot q independent in \partial L/\partial \dot q, both functions of t under d/dt | 0 | 3 | 0 | 0 | 1 | reader+engine | M | cm.lagrangian.pendulum, cm.lagrangian.noether, cm.lagrangian.hamilton-principle |
| 91 | `geometric-exponential-sums` | sums of exponentials sum e^{-a n} and sum n e^{-a n} (geometric series), which SymPy's summation leaves undone even with a > 0 | 0 | 2 | 0 | 2 | 1 | engine | S | statmech.canonical.oscillator, statmech.quantum.planck-distribution |
| 92 | `series-identities` | deciding identities between a closed form and an infinite series (generating functions, Frobenius solutions), by series coefficients | 0 | 2 | 0 | 2 | 2 | engine | M | em.sources.legendre, special.legendre.generating-function |
| 93 | `abstract-hypotheses` | component-free algebra using stated equations as hypotheses: (p_1 + p_2)^2 with p_i^2 = m_i^2; \partial_\mu T^{\mu\nu} given Maxwell's equations | 0 | 2 | 0 | 1 | 2 | engine | L | em.waves.stress-energy-conservation, sr.four-momentum.mandelstam |
| 94 | `abstract-vectors` | vectors with no components or indices, p_1, p, whose square p^2 is the Minkowski norm p_\mu p^\mu (Mandelstam variables, \slashed{p}^2) | 0 | 2 | 1 | 1 | 2 | reader+engine | M | *sr.four-momentum.mandelstam, qft.dirac.slash |
| 95 | `bvp-eigenvalues` | eigenvalues of a boundary-value problem, \lambda_n, numerically | 0 | 2 | 0 | 1 | 2 | integration:solver | M | de.bvp.robin, la.eigen.zeta-determinant |
| 96 | `function-argument-extent` | an unbracketed function argument stops before a fraction or derivative: \sin\theta \frac{\partial Y}{\partial\theta} is sin θ · ∂Y/∂θ (today the fraction joins the argument) | 0 | 2 | 0 | 1 | 2 | reader+design | S | qm.angular-momentum.l-squared, fa.series.gibbs |
| 97 | `bar-in-integrand` | an evaluation bar inside an integrand, before its differential: \int \left. f \right&#124;_{r=r_+} d\theta | 0 | 2 | 2 | 0 | 2 | parser | S | *gr.black-holes.kerr-horizon-area, *vc.integrals.stokes |
| 98 | `conditional-index-sets` | a sum over an index set given by a condition or by context: \sum_{k \neq 0}, \sum_i over a tuple's components, \sum_{\text{spins}} | 0 | 2 | 1 | 0 | 2 | reader+design | M | qm.perturbation.sum-over-states, *special.zeta.euler-product |
| 99 | `fft` | \operatorname{fft} and its rewrite of the DFT sum | 0 | 2 | 0 | 0 | 1 | integration:transform | S | data.fft.named, data.fft.sampled-tone |
| 100 | `index-solve` | solving an equation in index notation for an unknown tensor, with data: \pi_j = \pi_i P^i_j, \pi_i 1^i = 1 | 0 | 2 | 0 | 0 | 2 | integration:solver | M | dg.connections.spin-connection, prob.markov.stationary-index |
| 101 | `indexed-function-value` | a function whose value carries an index: \operatorname{softmax}(z)_i := ... | 0 | 2 | 1 | 0 | 2 | reader | S | *data.learning.softmax, data.fft.sampled-tone |
| 102 | `unknown-vectors` | an undefined name written with indices of a declared space (a_{j}, \pi_{j}) is an unknown vector: arbitrary in an identity, an unknown in an equation (today it is the scalar symbol a_j) | 0 | 2 | 0 | 0 | 2 | typer | M | la.index.bac-cab, prob.markov.stationary-index |
| 103 | `accented-definitions` | ':=' defines an accented name, \bar x, \hat x, \tilde x, \dot x (today := silently gives the equation \bar x = ..., though '=' defines it) | 0 | 1 | 1 | 1 | 1 | reader | S | *prob.estimation.sample-variance |
| 104 | `letter-run-subscript` | in a LaTeX letter run a subscript belongs to the last letter, as in TeX: iE_1 is i·E_1, ab_1 is a·b_1 (today the run and its subscript are one name, even when E_1 is defined) | 0 | 1 | 0 | 1 | 1 | lexer/reader | S | qm.time-evolution.beats |
| 105 | `parity-split-sums` | series whose terms vanish on one parity class, summed over the other (n = 2k + 1), as Fourier series need | 0 | 1 | 0 | 1 | 1 | engine | M | fa.series.convergence |
| 106 | `pde-by-ansatz` | solving a PDE with a stated ansatz: S = W(q) - alpha t in the Hamilton-Jacobi equation determines W (separation of variables) | 0 | 1 | 1 | 1 | 1 | integration:solver | L | *cm.hamiltonian.hamilton-jacobi |
| 107 | `abstract-definitions` | a tensor defined from tensors that have no components, kept as notation: T^{\mu\nu} := F^{\mu\lambda}F^\nu{}_\lambda - g^{\mu\nu}F^2/4 | 0 | 1 | 1 | 0 | 1 | typer+engine | M | *em.waves.stress-energy-conservation |
| 108 | `abstract-derivatives` | \partial_\mu and \nabla_\mu in component-free index notation, with the Leibniz rule, so that \partial_\mu T^{\mu\nu} can be expanded | 0 | 1 | 0 | 0 | 1 | engine | L | em.waves.stress-energy-conservation |
| 109 | `anticommutator` | the anticommutator {A, B} = AB + BA of matrices | 0 | 1 | 0 | 0 | 1 | reader+design | S | qft.dirac.clifford |
| 110 | `bar-in-definition` | a definition by = whose right side holds an evaluation bar with an = (\left. L \right&#124;_{\dot\theta=0}) defines, instead of silently becoming an equation | 0 | 1 | 0 | 0 | 1 | reader | S | cm.lagrangian.bead-on-hoop |
| 111 | `block-matrices` | block matrices: entries that are matrices, with 1 and 0 standing for identity and zero blocks | 0 | 1 | 0 | 0 | 1 | reader | S | qft.dirac.clifford |
| 112 | `boundary-of-region` | the oriented boundary \partial D of a region as the domain of a contour integral, \oint_{\partial D} | 0 | 1 | 0 | 0 | 1 | engine | M | dg.forms.stokes-theorem |
| 113 | `cartesian-indices` | Cartesian index notation: in a Euclidean space, repeated lower indices are summed (\delta_{ii}, \epsilon_{ijk} a_j), and free indices are still checked | 0 | 1 | 1 | 0 | 1 | typer+design | M | *vc.operators.curl-curl |
| 114 | `clebsch-gordan` | Clebsch–Gordan coefficients written as the bracket ⟨j1 j2; m1 m2 &#124; j1 j2; J M⟩ | 0 | 1 | 0 | 0 | 1 | integration:function | S | qm.angular-momentum.clebsch-gordan |
| 115 | `component-variable` | a component written with a numeric superscript, p^0, used as a variable of its own, not p to the power 0 (today \int_0^1 (p^0)^2 dp^0 = 1) | 0 | 1 | 0 | 0 | 1 | reader+design | S | qft.klein-gordon.feynman-propagator |
| 116 | `derivative-in-function-body` | a function whose body differentiates with respect to its own parameters (D(x, y) := \partial^2 f/\partial x^2 ...; g(x) := d/dx x^3) is differentiated before its arguments are put in; today g(2) differentiates with respect to 2 and fails | 0 | 1 | 0 | 0 | 1 | lowerer | S | analysis.multivariable.second-derivative-test |
| 117 | `derivative-wrt-expression` | a derivative with respect to an expression, d/d log(p/M), by the chain rule | 0 | 1 | 1 | 0 | 1 | engine | S | *qft.interactions.running-coupling |
| 118 | `dominant-term-limits` | limits at infinity governed by the largest term, (1/N) ln(lambda_1^N + lambda_2^N) -> ln lambda_1, which SymPy's gruntz cannot take | 0 | 1 | 0 | 0 | 1 | engine | S | statmech.ising.free-energy |
| 119 | `ellipsis-patterns` | products and sums written with an ellipsis and a general term, 3·5·7⋯(2n+1), read as what they stand for | 0 | 1 | 0 | 0 | 1 | design+reader | M | de.series.frobenius |
| 120 | `field-composition` | a field written in x, y, z evaluated at a point given by a vector function, v(r(t)) | 0 | 1 | 0 | 0 | 1 | reader+engine | S | vc.integrals.line |
| 121 | `functional-determinant` | zeta-regularised determinants of differential operators, det(-d^2/dx^2) | 0 | 1 | 0 | 0 | 1 | engine | L | la.eigen.zeta-determinant |
| 122 | `indexed-variables` | an index ranging over numbered variables: M_{jk} = d^2T/d(phi-dot_j)d(phi-dot_k) defines a matrix from phi_1, phi_2, and phi = 0 sets every phi_j | 0 | 1 | 0 | 0 | 1 | reader+typer | M | cm.lagrangian.double-pendulum |
| 123 | `induced-curvature` | the intrinsic curvature of a hypersurface, {}^{(3)}R, from the induced metric h_{\mu\nu} (today read silently as a symbol named ^{(3)}R) | 0 | 1 | 0 | 0 | 1 | engine (geometry) | M | gr.formalisms.hamiltonian-constraint |
| 124 | `integration-variable-range` | inside \int_a^b ... dx the variable lies between its limits, so limits and simplifications in the integrand may use a < x < b (lim_n n x (1-x^2)^n = 0 on (0, 1)) | 0 | 1 | 1 | 0 | 1 | lowerer+engine | S | *analysis.uniform.interchange |
| 125 | `partial-trace` | the partial trace tr_B over one factor of a tensor product | 0 | 1 | 0 | 0 | 1 | integration:map-function | S | qm.information.partial-trace |
| 126 | `perturbative-ansatz` | an approximate solution u \approx f(\phi; \alpha) substituted into an equation and solved order by order for its parameters | 0 | 1 | 0 | 0 | 1 | design+engine | L | gr.geodesics.perihelion-advance |
| 127 | `perturbative-integrals` | expanding an integral that depends on a small parameter to a given order (light deflection to O(M^2)) instead of integrating exactly | 0 | 1 | 0 | 0 | 1 | engine | L | gr.geodesics.light-bending |
| 128 | `pm-names` | ± and ∓ in a name being defined define both signs: a_\pm := (\mp i\hat p + m\omega x)/..., then a_+ and a_- | 0 | 1 | 0 | 0 | 1 | reader | S | qm.oscillator.ladder-commutator |
| 129 | `radial-fourier-3d` | the 3-D Fourier integral of a radial function over all space, ∫d³x e^{iq·x} V(&#124;x&#124;) = (4π/q)∫ r V(r) sin(qr) dr | 0 | 1 | 0 | 0 | 1 | engine | M | qm.scattering.born-yukawa |
| 130 | `recurrence` | a sequence defined by a recurrence, a_{n+2} := f(a_{n-1}), with an expression in the subscript | 0 | 1 | 1 | 0 | 1 | parser | M | *de.series.airy |
| 131 | `retarded-time` | the retarded bracket [ ... ]_ret, evaluated at the root t' of t - t' = &#124;x - r(t')&#124;/c | 0 | 1 | 0 | 0 | 1 | design+engine | M | em.sources.lienard-wiechert |
| 132 | `riemann-sum-limits` | limits of sums as N \to \infty (Riemann sums, Gibbs) | 0 | 1 | 1 | 0 | 1 | engine | L | *fa.series.gibbs |
| 133 | `space-by-labels` | spaces meet by their labels: a pandas axis named 'asset' and an unnamed axis with the same labels are one space | 0 | 1 | 0 | 0 | 1 | workbook+integration:object | S | data.objects.aligned |
| 134 | `subscript-partials` | subscript partial derivatives f_x, f_{xy} | 0 | 1 | 1 | 0 | 1 | reader+design | S | *analysis.multivariable.second-derivative-test |
| 135 | `trace-by-name` | a two-index tensor's name alone (K, h, T) means its trace with the metric, so K^2 is (K^\mu{}_\mu)^2 | 0 | 1 | 1 | 0 | 1 | design+typer | S | *gr.formalisms.hamiltonian-constraint |
| 136 | `vector-equations` | an equation between vectors is decided component by component: [0, 0] = [0, 0] is refused today, and (a, b, c) = \mathbf{0} silently gives False | 0 | 1 | 0 | 0 | 1 | engine+reader | S | vc.operators.curl-grad |

## 3. The silent wrong answers, grouped by cause

A wrong answer is worse than a refusal: it is believed. The 88 fall into ten causes, by the first
misreading that produces the wrong output. "Values" counts the definite wrong values in the group
(32 in all); the rest hand the question back or leave a claim undecided, presented as a result.

| Group | Cause | Cases | Values | Guard in wave 0 |
|---|---|---:|---:|---|
| A | A book's name the language lacks (or means otherwise) becomes an unknown function or a symbol | 29 | 5 | 28 |
| B | `:=` (or a function-defining `=`) silently becomes an equation, so the definition is lost | 10 | 3 | all |
| C | A scripted name is read as a new symbol, not as a member, an element or a derivative | 7 | 6 | all |
| D | An equation stated as a constraint is forgotten by what follows | 7 | 4 | 4 |
| E | A dependence is lost in a derivative, a substitution or a limit | 5 | 5 | none: fix it |
| F | An operator is read with another meaning | 7 | 7 | all |
| G | A computation left undone or conditional is shown as the answer | 16 | 1 | all |
| H | A claim is shown as an equation instead of being decided | 5 | 0 | all |
| I | A solver drops the roots | 1 | 1 | none: fix it |
| J | Invalid index notation is accepted | 1 | 0 | none: fix it |

### A. A book's name the language lacks (or means otherwise) becomes an unknown function or a symbol (29)

The reader accepts any name followed by brackets as an unknown function, and any unknown name
as a symbol. Examples:

- `\Gamma(1/2)` comes back as `Gamma(1/2)`, `\zeta(2)` as `zeta(2)`, and `\mathcal{L}\{t^2\}` as `L(t^2)`.
- `\operatorname{tr} A` is the symbol tr times the matrix A, and `\operatorname{erfc} x` is erfc·x,
  whose limit gives `oo*sign(erfc)`.
- `Y_1^0(\theta, \phi)` is `Y_1` to the power 0 times the tuple: `[theta, phi]`.
- `\operatorname{Res}_{z=0}` is a symbol `Res_z` multiplied in.
- `N(d_1)` is the language's numerical evaluation, so a Black–Scholes price of 4.76 comes out as 8.42.

Even spelled out, `zeta`, `besselj`, `legendre`, `hermite`, `Ynm`, `erfc`, `beta`, `Heaviside` and
`DiracDelta` are unknown functions that print exactly like SymPy's: `N(besselj(0, 1))` stays
`besselj(0, 1)`. Only `gamma`, `erf` and the documented builtins are SymPy's.

**Cases:** `special.bessel.value`, `special.bessel.equation`, `special.bessel.half-order`, `special.erf.asymptotic` (value), `special.gamma.half`, `special.gamma.beta`, `special.gamma.reflection`, `special.zeta.two`, `special.zeta.functional-equation`, `special.legendre.p3`, `special.hermite.orthogonality`, `special.legendre.generating-function`, `special.harmonics.y10` (value), `complex.maps.principal-log`, `complex.residues.essential-singularity`, `complex.residues.real-integral`, `fa.laplace.power`, `fa.laplace.derivative`, `fa.laplace.convolution`, `fa.laplace.step`, `prob.limits.normal-approximation`, `prob.moments.variance-affine`, `la.matrices.trace` (value), `la.matrices.rank` (value), `dg.index-notation.epsilon-delta`, `em.sources.legendre`, `qft.dirac.trace`, `prob.distributions.gamma-variance`, `data.finance.black-scholes-call` (value).

**Fix:** special-functions, laplace-fourier, residues-contours, matrix-functions, levi-civita, normal-cdf, random-variables, gamma-matrices.

**Guard now:** Reserve the book names (Γ, B, ζ, J_ν, P_n, H_n, Y_l^m, erfc, Log, Φ, Res, tr, rank, Var, ℒ, ε with indices): until an integration provides one, a call of it is refused as not supported yet, never an unknown function. N(x) needs the decision on N (section 6).

### B. `:=` (or a function-defining `=`) silently becomes an equation, so the definition is lost (10)

`\psi_n(x) := \ldots`, `\psi_{100}(r) := \ldots`, `Y_2^1(\theta, \phi) := \ldots`,
`\Delta\phi := \ldots` and `\bar{x} := \ldots` produce an equation such as `Eq(psi_n*x, sin(n*x))`,
shown as an output, and define nothing. Every later use is a plain symbol, so derivatives vanish:
`d^2\psi_0/dx^2 = 0`, and dU_eff/dr = 0 makes `solve` return `[]`. In
`gr.geodesics.mercury-perihelion` the perihelion advance comes out as `2.69e8 Delta phi / pi`.

**Cases:** `qm.oscillator.ground-state`, `qm.oscillator.hermite-wavefunction`, `qm.hydrogen.mean-radius`, `qm.square-well.orthogonality` (value), `qm.square-well.expansion`, `qm.time-evolution.beats`, `qm.angular-momentum.l-squared`, `gr.geodesics.mercury-perihelion`, `prob.estimation.sample-variance` (value), `cm.central.circular-orbit` (value).

**Fix:** subscripted-function, compound-names, accented-definitions, superscript-label, family.

**Guard now:** `:=` always defines or refuses, never yields an equation; `=` whose left side is a name with arguments defines that function or refuses.

### C. A scripted name is read as a new symbol, not as a member, an element or a derivative (7)

Examples:

- `a_{n+1}` becomes a new name `a_n+1`, so a radius of convergence is `zoo`.
- `c_1` after `c_n := \ldots` is a new symbol, returned as the answer.
- `x_n` on a list `x` is a new symbol, so a DFT factors it out.
- `k_{1\mu}` is the symbol `k_1mu`, so a contraction becomes a vector.
- `f^{(4)}(0)` is a symbol `f^(4)` times 0, which is 0.
- `h_+ \cos[\omega(t-z)]` is lexed as *one* symbol, with its spaces and commands, so a
  gravitational wave is a constant and its curvature is 0.

**Cases:** `analysis.series.radius-ratio` (value), `de.pde.heat`, `fa.transforms.dft` (value), `sr.four-momentum.two-photon-mass` (value), `analysis.taylor.nth-derivative` (value), `prob.moments.normal-fourth-moment` (value), `gr.waves.linearised-riemann` (value).

**Fix:** family, list-subscript and index-ranges, label-index-subscript, nth-derivative, label-subscript.

**Guard now:** A scripted name that nothing defines, but whose base is defined (c_1 after c_n, x_i on a list, f^{(4)} on a function, k_{1μ} beside k_1^μ), is refused with a hint; the lexer never makes a name that contains a space or a command (`h_+ \cos[...]`).

### D. An equation stated as a constraint is forgotten by what follows (7)

Examples:

- after `x^3 + y^3 = 6xy`, `dy/dx` is 0;
- after `m = \tanh(\ldots)`, `\partial m/\partial h` is 0;
- `\oint p\,dq` with no contour is the indefinite integral of a constant: `p q`;
- `\lambda_1` after a boundary-value problem, and `\cos\theta` after a matrix equation, are handed back
  as symbols;
- the Mandelstam hypotheses are read and never used.

In `analysis.derivatives.implicit`, `6xy` is also the single name `xy`, because the line has no
backslash and is read as plain text.

**Cases:** `analysis.derivatives.implicit` (value), `statmech.meanfield.curie-weiss` (value), `cm.hamiltonian.action-oscillator` (value), `cm.hamiltonian.hamilton-jacobi` (value), `de.bvp.robin`, `sr.lorentz.wigner-rotation`, `sr.four-momentum.mandelstam`.

**Fix:** latex-per-statement with implicit-relations; pde-by-ansatz, bvp-eigenvalues, matrix-factorisation-equations, abstract-vectors with abstract-hypotheses.

**Guard now:** Differentiating a symbol with respect to a variable that an earlier equation ties it to gives a note or a refusal, not 0; `\oint` with no contour is refused; Mandelstam's undecided claim is caught by the guard of group H.

### E. A dependence is lost in a derivative, a substitution or a limit (5)

- `\bar z` is a new symbol, so `\partial |z|^2/\partial\bar z = 0`.
- Along a curve `q^i = (\theta(t), \phi(t))` the metric is evaluated at the coordinate, not on the
  curve, so the sin cos terms are lost.
- `\omega(r) := -k_\mu u^\mu` substitutes its argument only into an `r` written in its body, not
  into the computed components, so `\omega(r_e) = \omega(r)`.
- `\lim_{d\to0} \left. V \right|_{q=p/d}` takes the limit before the bar: 0.
- A piecewise function's derivative at its seam is taken branch by branch: 0, where it does not exist.

**Cases:** `complex.analytic.wirtinger` (value), `cm.lagrangian.index-form` (value), `gr.schwarzschild.redshift` (value), `em.sources.dipole` (value), `analysis.derivatives.not-differentiable` (value).

**Fix:** conjugate-bar, curve-tensors, function-of-tensor-body, bar-precedence, limit-dne.

### F. An operator is read with another meaning (7)

- `\otimes` between matrices is the matrix product.
- `\hat{\mathbf{x}}` is a symbol `xhat`, and `\times` between symbols is multiplication.
- `\{L_x, L_y\}` is a tuple, compared with `L_z`: False.
- `\langle 0|\ldots|0\rangle` is `0·|...|·0`.
- `d\Omega` is the differential of a variable Ω: Ω times the integrand.
- `\partial_\theta` of a list is left as a Derivative, then multiplied as a scalar.
- `e^{At}` on a line with no backslash makes `At` one name.

**Cases:** `qm.information.tensor-product` (value), `em.waves.poynting` (value), `cm.hamiltonian.angular-momentum` (value), `qft.interactions.wick` (value), `qft.interactions.solid-angle` (value), `dg.surfaces.theorema-egregium` (value), `la.eigen.exponential` (value).

**Fix:** kronecker-product, unit-vectors with vector-algebra, poisson-bracket, dirac-notation, solid-angle, latex-per-statement with matrix-functions.

**Guard now:** Refuse, until built: ⊗ between matrices, × between undefined hatted names, a tuple `{a, b}` compared with a scalar, `⟨…|…⟩` read as products, a differential of a name the integrand never mentions (dΩ), ∂ of a list, e^{matrix}.

### G. A computation left undone or conditional is shown as the answer (16)

- **Sums kept as written by design:** `\sum_{n=1}^\infty 1/n^2` is shown as
  `Sum(n**(-2), (n, 1, oo))`. SymPy's `summation` cannot even do `\sum e^{-an}`.
- **Piecewise conditions:** a parameter of unknown sign gives a Piecewise. In
  `qm.scattering.soft-sphere`, `\lim_{k\to0}` then takes the κ = 0 branch and returns a definite 0.
- **Integrals SymPy leaves undone:** the Bose integral `\int_0^\infty x^3/(e^x - 1)\,dx`.
- **A Jacobian left undone:** `\partial(x, y)/\partial(r, \theta)` comes back as a Derivative of a tuple.

**Cases:** `analysis.series.basel`, `qm.hydrogen.degeneracy`, `statmech.canonical.oscillator`, `statmech.quantum.planck-distribution`, `fa.series.convergence`, `qm.square-well.normalisation`, `analysis.integrals.gaussian`, `qft.interactions.gaussian`, `statmech.canonical.ideal-gas`, `statmech.canonical.rigid-rotor`, `qm.scattering.soft-sphere` (value), `fa.transforms.rectangle`, `statmech.quantum.planck-integral`, `statmech.quantum.stefan-boltzmann`, `fa.transforms.delta`, `analysis.multivariable.jacobian`.

**Fix:** closed-form-sums with geometric-exponential-sums, sum-index-integer and parity-split-sums; default-assumptions with inequality-assumptions and piecewise-removable; special-integrals; distribution-limits; jacobian-notation.

**Guard now:** A result that holds an unevaluated Sum, Integral, Limit or Derivative, or a Piecewise of convergence conditions, carries a note (not computed; under these conditions); a limit never takes the branch a Piecewise has only at the limit point.

### H. A claim is shown as an equation instead of being decided (5)

`e^x \geq 1 + x` comes back as the relation itself. Identities that the quick zero test cannot
prove come back as `Eq`, and so are not decided:

- `B(\zeta_1)B(\zeta_2) = B(\zeta_1 + \zeta_2)`, compared component by component without the
  hyperbolic expansion a scalar claim gets;
- the Black–Scholes PDE, where `simplify` would succeed;
- an `O(x^5)` expansion.

**Cases:** `analysis.inequalities.exp-tangent`, `analysis.inequalities.am-gm`, `analysis.taylor.big-o`, `sr.lorentz.rapidities-add`, `data.finance.black-scholes-pde`.

**Fix:** decide-inequalities with inequality-assumptions, order-terms, decide-identities.

**Guard now:** A claim that is not decided says so, with its reason, instead of printing as an equation.

### I. A solver drops the roots (1)

`solve(g^{rr} = 0, r)` receives the unsimplified `r^2 f^2/(r^2 - 2Mr + Q^2)`. Both horizons also zero
the denominator, so SymPy discards them and returns `[]`: a black hole with no horizon. The display of
`g^{rr}` is simplified, which hides it. Primed names hide the same fault in
`sr.four-momentum.compton`: with plain names, `solve` divides by a coefficient that vanishes by
cos² + sin² = 1 and returns `zoo`.

**Cases:** `gr.black-holes.reissner-nordstrom-horizons` (value).

**Fix:** solve-simplify.

### J. Invalid index notation is accepted (1)

`L_i = I_{ij}\omega_i` sums over i and renames the free j to i by position, instead of refusing the
mismatch the case expects.

**Cases:** `cm.oscillations.mismatched-index`.

**Fix:** definition-index-check.

### SymPy pitfalls an integration must not pass through

Several wrong answers are SymPy's behaviour shown as RelativisticPy's. The integrations to come must
guard against these, and each deserves a test of its own:
- `residue(z**2*exp(1/z), z, 0)` returns 0; the residue is 1/6. Residues at essential singularities
  need the Laurent series.
- `limit(sin(1/h), h, 0)` returns `AccumBounds(-1, 1)` as if it were a value; it does not exist.
- `solve` on an unsimplified rational expression drops roots that also zero a removable denominator.
- `solve` keeps terms that cancel only by an identity, then divides by their zero coefficient (`zoo`).
- A limit of a Piecewise takes the branch that holds only at the limit point.
- `summation` leaves geometric series in `e^{-an}` undone, even with `a > 0`.
- Integrals of parameters with unknown sign come back as Piecewise conditions, with `polar_lift`.

## 4. The order of work, in waves

Each wave is a set of capabilities. Inside a wave they are ordered by cases unlocked per unit of
effort, with wrong answers counted twice, so wrong answers go first. The effort units are S = 1,
M = 3 and L = 8. Capabilities that would turn refusals into silent wrong answers if shipped alone are
**bundled** (`+`), and each bundle ships whole:
- `accent-script + lagrangian-velocities + functionals`: fixing `\dot\theta^2` alone makes the
  pendulum give `g l m \sin\theta`, a hoop 0, and `\delta S/\delta q(t)` become `S/q(t)`;
- `levi-civita + typeface-names`: with ε alone, the dual invariant is silently `2E^2 - 2B^2`,
  because `\mathcal{F}` is read as `F`.

The expected count is a consequence of the classification. The floor counts only cases whose
variant passed.

| Wave | Theme | Capabilities | Effort | Newly passing | Expected pass | Floor | Of the 88 wrong, still failing |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Guards: refuse rather than misread | 7 guards | ~8 | 0 | 131 | 131 | 88: 61 now refused, 16 flagged, 11 silent |
| 1 | Wrong answers: the high-yield capabilities | 25 | 39 | 76 | **207** | 194 | 35 |
| 2 | Wrong answers: the long tail | 20 | 32 | 29 | **236** | 211 | 12 |
| 3 | Refusals: notation the reader should already know | 33 | 37 | 55 | **291** | 258 | 10 |
| 4 | Refusals: medium capabilities | 30 | 87 | 64 | **355** | 304 | 7 |
| 5 | Large formalisms | 20 | 97 | 35 | **390** | 311 | 3 |
| 6 | The frontier (level 5 designs) | 8 | 64 | 17 | **407** | 312 | 0 |

After wave 2, 12 of the original 88 still fail:
- `qm.oscillator.ground-state` and `qm.angular-momentum.l-squared` wait only on wave 3's bare
  function names, and are refusals by then;
- `cm.lagrangian.index-form` and `fa.series.convergence` wait on wave 4;
- the rest need large formalisms or frontier designs: Mandelstam's hypotheses, the Wigner rotation,
  Dirac traces, Wick's theorem, the Fourier transform of a constant, the variance of a random
  variable, Hamilton–Jacobi by separation, and a Sturm–Liouville eigenvalue.

Four of them would still answer silently despite the guards: `cm.lagrangian.index-form` (wrong
components), `cm.hamiltonian.hamilton-jacobi` (`[]`), `de.bvp.robin` (λ_1 handed back) and
`sr.lorentz.wigner-rotation` (cos θ handed back). They are the case for a last guard: a value
that is only an undefined symbol the program asked for is noted, not shown as an answer.

### Wave 0: guards, no new mathematics

Each guard is small. None adds a passing case, but together they make 61 of the 88 wrong answers
honest refusals (the harness counts them refused), flag 16 more with a note (still counted wrong,
no longer silent), and leave 11 silent until their fixes land:

1. **`:=` defines or refuses**, never yields an equation. An `=` whose left side is a name with
   arguments defines that function or refuses. (group B, 10 cases)
2. **Reserved book names.** Γ, B, ζ, J_ν, P_n, H_n, Y_l^m, erfc, Log, Φ, Res, tr, rank, Var, ℒ, ℱ,
   and ε with indices, are refused as "not supported yet" until an integration provides them; never
   an unknown function. The same applies to their spelled-out forms (`zeta`, `besselj`, ...). (group
   A, 28 cases)
3. **Scripted names whose base is defined.** `c_1` after `c_n`, `x_i` on a list, `f^{(n)}` on a
   function, and `k_{1\mu}` beside `k_1^\mu` are refused with a hint. A name never contains a space
   or a command. (group C, 7)
4. **Operators read with another meaning are refused** until built: ⊗ of matrices, × of undefined
   hatted names, a tuple compared with a scalar, ⟨…|…⟩, a differential of a name the integrand never
   mentions, ∂ of a list, e^{matrix}. (group F, 7)
5. **Undecided claims say so**, as `UndecidedError` in a program, instead of printing an equation.
   (group H, 5, and Mandelstam in group D)
6. **Constrained symbols.** A derivative of a symbol that an earlier equation ties to the variable
   is refused or noted, and `\oint` with no contour is refused. (group D, 3 of 7; with guard 5, 4)
7. **Undone results carry a note:** an unevaluated Sum, Integral, Limit or Derivative, or a Piecewise
   of conditions. A limit never takes a branch that holds only at the limit point. (group G, 16)

### Wave 1: Wrong answers: the high-yield capabilities

**Effort 39. Pass count 207 (floor 194). Of the 88 wrong answers, 35 still fail.**

In order: `special-functions` (M) → `closed-form-sums` (S) → `default-assumptions` (M) → `subscripted-function` (S) → `geometric-exponential-sums` (S) → `nth-derivative` (S) → `normal-cdf` (S) → `special-integrals` (S) → `inequality-assumptions` (S) → `conjugate-bar` (S) → `poisson-bracket` (S) → `decide-identities` (M) → `limit-dne` (S) → `matrix-functions` (M) → `label-subscript` (M) → `kronecker-product` (S) → `levi-civita` (S) + `typeface-names` (S) → `family` (M) → `star-conjugate` (S) → `piecewise-removable` (S) → `laplace-fourier` (M) → `step-delta` (S) → `letter-run-subscript` (S) → `sum-index-integer` (S).

Newly passing: `analysis.derivatives.not-differentiable`, `analysis.integrals.gaussian`, `analysis.series.basel`, `analysis.series.radius-ratio`, `analysis.limits.does-not-exist`, `analysis.series.harmonic`, `analysis.taylor.nth-derivative`, `complex.analytic.wirtinger`, `complex.numbers.conjugate`, `complex.maps.cayley`, `complex.maps.principal-log`, `de.pde.heat`, `qm.oscillator.hermite-wavefunction`, `qm.hydrogen.degeneracy`, `qm.hydrogen.mean-radius`, `qm.square-well.normalisation`, `qm.square-well.orthogonality`, `qm.square-well.expansion`, `qm.information.tensor-product`, `qm.scattering.soft-sphere`, `qm.time-evolution.beats`, `qm.angular-momentum.singlet`, `qm.perturbation.stark`, `qm.information.purity`, `qm.information.entropy`, `qm.scattering.hard-sphere`, `qm.spin.pauli-commutator`, `qm.spin.rotation`, `em.field-tensor.dual-invariant`, `sr.lorentz.rapidities-add`, `qft.interactions.gaussian`, `gr.black-holes.reissner-nordstrom-temperature`, `gr.formalisms.newman-penrose-psi2`, `gr.geodesics.light-deflection-sun`, `gr.waves.linearised-riemann`, `dg.index-notation.epsilon-delta`, `special.bessel.value`, `special.bessel.equation`, `special.bessel.half-order`, `special.erf.asymptotic`, `special.gamma.half`, `special.gamma.beta`, `special.gamma.reflection`, `special.zeta.two`, `special.zeta.functional-equation`, `special.legendre.p3`, `special.hermite.orthogonality`, `special.harmonics.y10`, `special.harmonics.unsold`, `fa.series.parseval`, `fa.transforms.rectangle`, `fa.transforms.pair`, `fa.laplace.power`, `fa.laplace.derivative`, `fa.laplace.convolution`, `fa.laplace.step`, `statmech.canonical.oscillator`, `statmech.canonical.ideal-gas`, `statmech.canonical.rigid-rotor`, `statmech.quantum.planck-integral`, `statmech.quantum.planck-distribution`, `statmech.quantum.stefan-boltzmann`, `la.eigen.powers`, `la.index.epsilon-delta`, `la.matrices.trace`, `la.matrices.rank`, `prob.distributions.poisson-limit`, `prob.distributions.gamma-variance`, `prob.estimation.correlation-matrix`, `prob.limits.normal-approximation`, `prob.moments.normal-fourth-moment`, `cm.central.circular-orbit`, `cm.hamiltonian.angular-momentum`, `cm.hamiltonian.canonical-bracket`, `data.finance.black-scholes-call`, `data.finance.black-scholes-pde`.

### Wave 2: Wrong answers: the long tail

**Effort 32. Pass count 236 (floor 211). Of the 88 wrong answers, 12 still fail.**

In order: `residues-contours` (M) → `compound-names` (S) → `latex-per-statement` (S) → `order-terms` (S) → `solid-angle` (S) → `solve-simplify` (S) → `implicit-relations` (M) → `bar-precedence` (S) → `definition-index-check` (S) → `function-of-tensor-body` (S) → `jacobian-notation` (S) → `label-index-subscript` (S) → `vector-algebra` (M) → `unit-vectors` (S) → `decide-inequalities` (M) → `series-identities` (M) → `index-ranges` (M) → `list-subscript` (S) → `accented-definitions` (S) → `function-argument-extent` (S).

Newly passing: `analysis.derivatives.implicit`, `analysis.inequalities.exp-tangent`, `analysis.inequalities.am-gm`, `analysis.multivariable.jacobian`, `analysis.taylor.big-o`, `complex.maps.argument-principle`, `complex.residues.essential-singularity`, `complex.residues.residue-theorem`, `complex.residues.real-integral`, `complex.residues.principal-value`, `em.sources.dipole`, `em.sources.current-loop`, `em.sources.legendre`, `em.waves.poynting`, `sr.four-momentum.two-photon-mass`, `qft.interactions.solid-angle`, `gr.black-holes.reissner-nordstrom-horizons`, `gr.geodesics.mercury-perihelion`, `gr.schwarzschild.redshift`, `dg.surfaces.theorema-egregium`, `special.legendre.generating-function`, `fa.transforms.dft`, `statmech.meanfield.curie-weiss`, `la.eigen.exponential`, `la.index.delta-trace`, `prob.estimation.sample-variance`, `cm.central.runge-lenz`, `cm.hamiltonian.action-oscillator`, `cm.oscillations.mismatched-index`.

### Wave 3: Refusals: notation the reader should already know

**Effort 37. Pass count 291 (floor 258). Of the 88 wrong answers, 10 still fail.**

In order: `transpose-dagger` (S) → `identity-matrix` (S) → `fn-bare` (M) → `superscript-label` (S) → `hypergeometric` (S) → `space-index-sums` (S) → `primed-names` (M) → `unbraced-index` (S) → `bar-in-integrand` (S) → `d-variable` (S) → `indexed-function-value` (S) → `eval-bracket` (S) → `exp-not-tensor` (S) → `line-element-products` (S) → `staggered-unbraced` (S) → `thermo-partials` (S) → `pochhammer` (S) → `bessel-zeros` (S) → `kl-divergence` (S) → `probability-notation` (S) → `matrix-similarity` (S) → `field-composition` (S) → `integration-variable-range` (S) → `dominant-term-limits` (S) → `differential-numerator` (S) → `component-variable` (S) → `bar-in-definition` (S) → `vector-equations` (S) → `derivative-wrt-expression` (S) → `pm-names` (S) → `clebsch-gordan` (S) → `subscript-partials` (S) → `derivative-in-function-body` (S).

Newly passing: `analysis.integrals.by-parts`, `analysis.multivariable.second-derivative-test`, `analysis.uniform.interchange`, `complex.numbers.euler`, `de.first-order.exact`, `de.second-order.wronskian`, `de.second-order.harmonic`, `de.second-order.resonance`, `de.second-order.system`, `qm.angular-momentum.l-squared`, `qm.oscillator.ground-state`, `qm.dirac.change-of-basis`, `qm.hydrogen.radial-equation`, `qm.perturbation.first-order`, `qm.perturbation.second-order`, `qm.time-evolution.unitarity`, `qm.time-evolution.precession`, `em.field-tensor.boost`, `em.field-tensor.e-dot-b`, `em.sources.loop-moment`, `sr.four-momentum.compton`, `sr.lorentz.preserves-metric`, `sr.lorentz.velocity-composition`, `qft.klein-gordon.dispersion`, `qft.klein-gordon.feynman-propagator`, `gr.black-holes.kerr-horizon-area`, `gr.cosmology.hubble-matter`, `gr.cosmology.radiation-era`, `gr.formalisms.painleve-gullstrand-lapse`, `gr.waves.polarisation-rotation`, `special.bessel.zeros`, `special.hypergeometric.pochhammer`, `special.hypergeometric.logarithm`, `special.hypergeometric.gauss`, `special.hypergeometric.arcsin`, `statmech.ising.transfer-eigenvalues`, `statmech.ising.free-energy`, `statmech.thermo.maxwell-relation`, `vc.integrals.line`, `la.eigen.characteristic`, `la.eigen.spectral`, `la.eigen.jordan`, `la.orthogonality.least-squares`, `la.orthogonality.qr`, `la.orthogonality.svd`, `prob.bayes.conditional-notation`, `prob.estimation.normal-equations`, `prob.information.kl-notation`, `prob.limits.chebyshev`, `prob.markov.absorbing`, `cm.oscillations.principal-moments`, `data.learning.softmax`, `data.learning.pca`, `data.portfolio.capm-beta`, `data.grids.uniform`.

### Wave 4: Refusals: medium capabilities

**Effort 87. Pass count 355 (floor 304). Of the 88 wrong answers, 7 still fail.**

In order: `vector-calculus` (M) → `python-scope` (M) → `curve-tensors` (M) → `fft` (S) → `extremum` (M) → `accent-script` (S) + `lagrangian-velocities` (M) + `functionals` (M) → `space-by-labels` (S) → `region-integrals` (M) → `metric-without-coordinates` (M) → `asymptotic-relations` (M) → `field-derivatives` (M) → `coordinate-transformations` (M) → `inf-sup` (M) → `parity-split-sums` (M) → `operators` (L) → `ode-unknown` (M) → `ode-conditions` (S) → `bvp-eigenvalues` (M) → `operator-algebra` (M) → `component-definitions` (M) → `conditional-index-sets` (M) → `unknown-vectors` (M) → `index-solve` (M) → `embedding-indices` (M) → `user-notation` (M) → `recurrence` (M) → `cartesian-indices` (M) → `indexed-variables` (M).

Newly passing: `analysis.derivatives.dot-squared`, `analysis.inequalities.infimum`, `analysis.series.limsup`, `analysis.multivariable.lagrange`, `analysis.multivariable.region`, `analysis.taylor.stirling`, `de.first-order.growth`, `de.first-order.linear`, `de.first-order.separable`, `de.second-order.impulse`, `de.series.airy`, `de.series.euler-exponents`, `de.bvp.rayleigh`, `de.bvp.robin`, `qm.commutators.canonical`, `qm.commutators.angular-momentum`, `qm.commutators.power-of-x`, `qm.commutators.gaussian-packet`, `qm.oscillator.ladder-commutator`, `qm.square-well.uncertainty`, `em.maxwell.gauss`, `sr.four-momentum.two-body-phase-space`, `sr.lorentz.index-form`, `qft.dirac.plane-wave`, `qft.interactions.generating-functional`, `qft.interactions.running-coupling`, `qft.klein-gordon.hamiltonian`, `qft.klein-gordon.euler-lagrange`, `gr.formalisms.penrose-compactification`, `gr.geodesics.circular-geodesic`, `gr.spacetimes.eddington-finkelstein`, `dg.connections.parallel-transport`, `dg.surfaces.induced-metric`, `special.zeta.euler-product`, `fa.series.convergence`, `statmech.meanfield.critical-exponent`, `vc.integrals.divergence-theorem`, `vc.integrals.stokes`, `vc.operators.divergence`, `vc.operators.curl`, `vc.operators.curl-grad`, `vc.operators.curl-cylindrical`, `vc.operators.curl-curl`, `la.index.bac-cab`, `la.orthogonality.gram-schmidt`, `prob.markov.stationary-index`, `cm.hamiltonian.legendre`, `cm.lagrangian.pendulum`, `cm.lagrangian.bead-on-hoop`, `cm.lagrangian.double-pendulum`, `cm.lagrangian.index-form`, `cm.lagrangian.noether`, `cm.lagrangian.hamilton-principle`, `data.portfolio.argmin`, `data.objects.column`, `data.objects.dataframe`, `data.objects.key`, `data.objects.constants`, `data.objects.function`, `data.objects.sympy`, `data.fft.named`, `data.fft.dft-sum`, `data.fft.sampled-tone`, `data.objects.aligned`.

### Wave 5: Large formalisms

**Effort 97. Pass count 390 (floor 311). Of the 88 wrong answers, 3 still fail.**

In order: `dirac-notation` (L) → `partial-trace` (S) → `distribution-limits` (L) → `random-variables` (L) → `stochastic-calculus` (L) → `differential-forms` (L) → `boundary-of-region` (M) → `radial-fourier-3d` (M) → `retarded-time` (M) → `ellipsis-patterns` (M) → `gamma-matrices` (L) → `abstract-vectors` (M) → `abstract-hypotheses` (L) → `riemann-sum-limits` (L) → `anticommutator` (S) → `block-matrices` (S) → `trace-by-name` (S) → `induced-curvature` (M) → `abstract-definitions` (M) → `abstract-derivatives` (L).

Newly passing: `de.pde.heat-kernel`, `de.series.frobenius`, `qm.angular-momentum.clebsch-gordan`, `qm.dirac.inner-product`, `qm.dirac.completeness`, `qm.dirac.momentum-wavefunction`, `qm.oscillator.potential-energy`, `qm.perturbation.sum-over-states`, `qm.information.partial-trace`, `qm.scattering.born-yukawa`, `qm.spin.bra-ket-expectation`, `em.sources.lienard-wiechert`, `em.waves.stress-energy-conservation`, `em.waves.retarded-green`, `sr.four-momentum.mandelstam`, `qft.dirac.clifford`, `qft.dirac.trace`, `qft.dirac.slash`, `gr.formalisms.hamiltonian-constraint`, `dg.connections.spin-connection`, `dg.forms.exterior-derivative`, `dg.forms.hodge-star`, `dg.forms.stokes-theorem`, `fa.series.gibbs`, `fa.transforms.delta`, `vc.integrals.form`, `vc.integrals.delta`, `prob.bayes.total-expectation`, `prob.distributions.poisson-tilde`, `prob.moments.variance-affine`, `prob.stochastic.ito-log`, `prob.stochastic.ito-integral`, `prob.stochastic.ito-isometry`, `prob.stochastic.ou-variance`, `data.finance.black-scholes-derivation`.

### Wave 6: The frontier (level 5 designs)

**Effort 64. Pass count 407 (floor 312). Of the 88 wrong answers, 0 still fail.**

In order: `quantifiers` (L) → `frontier-qft` (L) → `matrix-factorisation-equations` (L) → `pde-by-ansatz` (L) → `periodic-identification` (L) → `perturbative-ansatz` (L) → `perturbative-integrals` (L) → `functional-determinant` (L).

Newly passing: `analysis.inequalities.epsilon-delta`, `analysis.uniform.not-uniform`, `analysis.uniform.m-test`, `analysis.uniform.uniform-continuity`, `complex.maps.joukowski`, `qm.time-evolution.path-integral`, `sr.lorentz.wigner-rotation`, `qft.interactions.wick`, `qft.interactions.phi4-vertex`, `qft.interactions.muon-pair`, `qft.interactions.loop-integral`, `gr.black-holes.euclidean-period`, `gr.black-holes.null-energy-condition`, `gr.geodesics.light-bending`, `gr.geodesics.perihelion-advance`, `la.eigen.zeta-determinant`, `cm.hamiltonian.hamilton-jacobi`.

### Notes on the waves

- **Wave 1 is mostly integrations and two readings.**
  - The special functions, the Laplace and Fourier transforms, the matrix functions, the residues
    and the Levi-Civita symbol are integrations of the kinds `design/integrations.md` already
    describes: function, transform, map function and constructor. They are the natural first users
    of the pattern.
  - The readings are subscripted function names and families. They are a change to the reader that
    the integrations also need: `J_\nu(x)`, `P_n(x)` and `Y_l^m` are subscripted names called with
    arguments.
- **`default-assumptions` and `closed-form-sums` are decisions before they are code** (section 6).
  Evaluating sums by default needs a time budget: `fa.series.parseval` timed out trying to sum a
  Piecewise series, and evaluating more sums makes that more likely, so ship it with
  `sum-index-integer`.
- **`decide-identities` must give its own verdict when it rests on numbers.** "Holds at 5 random
  points" is evidence, not proof (the oracles of `special.zeta.functional-equation` and
  `data.finance.black-scholes-pde` check only numerically).
- **Wave 3 is the reader's backlog:** transpose and dagger, the identity matrix, functions named bare,
  labels in superscripts, primes, hypergeometric `;`, and thirty small forms. It has the best yield
  per effort after wave 1, but comes after wave 2 because wrong answers come first.
- **Wave 4 holds the medium formalisms:** vector calculus, a Python scope, tensors along curves,
  extrema, Lagrangian mechanics, ODEs with conditions, operators, coordinate changes, and a metric
  given by its components alone.
- **Waves 5 and 6 need designs first.** The large formalisms are Dirac notation, random variables,
  Itô calculus, differential forms, Dirac matrices and component-free algebra with hypotheses. The
  frontier is quantifiers, path integrals and Feynman amplitudes, perturbative expansions and
  functional determinants. Level 5 exists to record these gaps, and nothing should be rushed there.

## 5. Cases that look mis-specified

No oracle was found to compute a wrong value. But 27 cases ask for something the program does not
say, or are judged more strictly than the mathematics.

### Claims that are false as written: true only under a condition the program omits (4)

These matter most. A language that answered True for them as written would be wrong.

- **`em.sources.legendre`:** the expansion of 1/|r − r'| converges only for r' < r. The program
  never says so, and the oracle checks only points with r' < r. Add `r' < r`.
- **`special.legendre.generating-function`:** the identity holds for |t| < 1 (and |x| ≤ 1). The
  oracle compares formal coefficients only. Add `|t| < 1`, or say the claim is about formal series.
- **`special.bessel.half-order`:** `J_{1/2}(x) = \sqrt{2/(\pi x)} \sin x` holds for x > 0. On SymPy's
  principal branches the sides differ in sign for x < 0: `simplify` leaves `sqrt(x)*sqrt(1/x) - 1`.
  The oracle declares x positive. Add `x > 0`.
- **`special.harmonics.unsold`:** the sum is 5/(4π) for real angles. The oracle declares θ and φ
  real and the program does not. Add the reality, or adopt decision 1 below.

### Oracles that assume what the program never states (16)

Each oracle declares its parameters positive or real. The program states nothing, exactly as the
book does, and the identity, the integral or the limit depends on it:
`qft.interactions.gaussian`, `statmech.canonical.oscillator`, `statmech.canonical.ideal-gas`,
`statmech.canonical.rigid-rotor`, `statmech.quantum.planck-distribution`,
`statmech.ising.free-energy`, `cm.hamiltonian.canonical-bracket`,
`cm.hamiltonian.action-oscillator`, `prob.distributions.poisson-limit` (it needs k ≥ 0, not only
n > 0), `prob.distributions.gamma-variance`, `qm.square-well.normalisation`,
`qm.time-evolution.unitarity` (U†U = 1 only for real t), `qm.oscillator.hermite-wavefunction`,
`qm.commutators.gaussian-packet`, `qm.hydrogen.mean-radius`, `qm.perturbation.stark`.

They are well specified if the project adopts decision 1 (section 6), and they become its tests. If
it does not, each program should state its assumptions, which would fit the cases to the language,
against the curriculum's own rule.

### Judged more strictly than the mathematics (5)

These need changes to `benchmarks/curriculum.py`, not to the cases.

- **`fa.transforms.rectangle`:** the language's Piecewise has the oracle's sin(πs)/(πs) for s ≠ 0
  and its limit 1 at s = 0. That is more complete than the oracle, whose expression is undefined at
  0. `_left_undone` rejects every Piecewise. Accept a Piecewise whose generic branch is the expected
  value and whose other branches are its limits; the language should drop removable branches anyway
  (`piecewise-removable`).
- **`de.first-order.separable` and `de.second-order.impulse`** (and growth and linear, once they
  run): the expected value is SymPy's `Eq` from `dsolve`, and the harness can compare two `Eq` only
  structurally, because `Eq - Eq` raises. An equivalent solution in another form, such as the book's
  form for impulse, would be judged wrong. Compare equations by the difference of their sides.
- **`cm.lagrangian.noether` and `cm.lagrangian.index-form`:** the oracles write velocities as
  `Derivative(x(t), t)`, so a right answer in dotted symbols cannot be compared. Either the language
  produces derivatives of functions of t (`lagrangian-velocities`, which this roadmap assumes) or the
  harness maps a dotted name to the derivative.

### Programs that leave something unwritten (2)

- **`fa.transforms.dft`:** k is never given a range. Its sister `data.fft.dft-sum` writes
  `, k = 0, \ldots, N-1`. Add it, so that the case tests one idea and not range inference too.
- **`qm.scattering.born-yukawa`:** uses r' without saying r' = |x'|. Write `V(|\mathbf{x}'|)`.

## 6. Notation decisions the project must make

These are the places where textbooks disagree, or a reading is ambiguous, and where the language today
guesses, usually silently. Each names the capabilities that wait on it. The recommendations follow
the interface's rule: never guess silently. Where a reading is chosen, say so in a note.

### 1. What a parameter nothing declares is

**Waiting on it:** `default-assumptions` (27 cases), `inequality-assumptions` (4).

**Why it is open.** SymPy takes every symbol complex, so integrals come back as Piecewise conditions with `polar_lift`, and |f|² and conjugates never simplify. Books never say that a width, a mass, ħ, a temperature or a frequency is positive, or that a time or an angle is real.

**Recommendation.** Keep symbols general. When a result would be a Piecewise of conditions, or a modulus or conjugate that a sign or reality assumption removes, give the value under that assumption and state it in a note and in the result's data ("assuming a > 0"). Declarations override. Never assume against a declaration. The coordinates of a metric and the variable of an integral over real limits are real. Complex analysis keeps z complex because nothing there needs otherwise.

### 2. A relation on a lone name: assumption or claim?

**Waiting on it:** `inequality-assumptions`: `a > 0`, `a, b > 0`, `t > 0`, `x \in \mathbb{R}`.

**Why it is open.** Today `a > 0` is printed and assumes nothing. `a, b > 0` splits at the comma into `a` and `b > 0`. `x \in \mathbb{R}` is refused, and `x \in \mathbb{R}^3` declares x an index (the regression the review recorded).

**Recommendation.** A relation between undefined names and a constant, standing alone, is an assumption, distributed over a list of names, with a note. `\in \mathbb{R}` (ℂ, ℤ, ℕ, no exponent) is an assumption. `\in \mathbb{R}^n` declares a space only for names used as index labels.

### 3. When `=` between expressions is a claim, an equation or a constraint

**Waiting on it:** `decide-identities` (7), `implicit-relations` (4), `decide-inequalities` (3), `order-terms` (2).

**Why it is open.** Today such an `=` is an equation to solve, shown as `Eq` unless a quick zero test proves it. A later `dy/dx` ignores it. Curriculum cases write identities (the Black–Scholes PDE, B(ζ₁)B(ζ₂) = B(ζ₁+ζ₂)) and constraints (x³ + y³ = 6xy) with the same sign.

**Recommendation.** Decide every relation whose sides are fully known. It is True when it holds identically, by simplification, expansion, and then numerical evidence under a verdict of its own ("holds numerically"); otherwise it stays an equation, "not an identity". An equation between undefined names that a later derivative or integral uses makes them dependent (y an implicit function of x), with a note. An inequality claim is decided for all real values of its free names.

### 4. Should sums with a closed form be computed?

**Waiting on it:** `closed-form-sums` (6), `geometric-exponential-sums` (2), `sum-index-integer` (2).

**Why it is open.** By design, sums with symbolic or infinite limits are kept as written until `doit`, while finite numeric ones are computed. The review found `\sum_{n=1}^{3} n = 6` judged as differing. Every book means the value.

**Recommendation.** Compute a sum or product when a closed form is found within a time budget. Otherwise keep it as written, with a note. A summation index is an integer.

### 5. What a subscript means

**Waiting on it:** `label-subscript` (15), `subscripted-function` (12), `family` (11), `unbraced-index` (4), `list-subscript` (3), and five more.

**Why it is open.** It can be part of a name (r_s, λ_1, σ_x, R_⊙, r_+), an index (T_{μν}), a family parameter (c_n, ψ_n(x)), an element (x_i of a list), a partial derivative (f_{xy} in Stewart), a list of labels (j_{ν,k}) or the length of a Pochhammer symbol ((a)_n). Today Greek and built-in bases with a label become components (`\lambda_1 :=` is refused, R_⊙ is the Ricci tensor), `\psi_1(x)` is a product, and `c_1` after `c_n` is a new symbol.

**Recommendation.** Read it in this order, and say which reading was taken where another was possible:

1. a declared index label, or an index of a tensor in force, is an index, braced or not;
2. on the left of `:=`, a one-letter subscript that occurs on the right makes a family, so `c_1`, `a_{n+1}` and `\psi_3(x)` evaluate it;
3. on a list, it is an element, counted from the lower limit of the sum that ranges over the whole list (decision 17);
4. after a parenthesised expression it is a Pochhammer length;
5. otherwise it is part of the name, whatever the base: Greek letters, numbers, signs, ⊙. A built-in tensor claims a subscripted name only while a metric is in force;
6. `f_x` is a partial derivative only on a defined function of x, with nothing defining `f_x`;
7. in a letter run the subscript binds to the last letter, as in TeX (`iE_1` is i·E_1);
8. a comma separates labels on a name that is not a tensor.

### 6. What a superscript means

**Waiting on it:** `transpose-dagger` (10), `superscript-label` (8), `star-conjugate` (6), `nth-derivative` (2), and four more.

**Why it is open.** It can be a power, an index, a label (Y_l^m, E^{(0)}), a derivative order (f^{(4)}), a transpose (A^T), a conjugate (ψ^*) or part of a name (r^*, the tortoise coordinate). Today `Q^T` is refused as a power, `\sigma^2 :=` as a component, and `\psi^*` is a power by a symbol named `*`.

**Recommendation.** Read it as follows:

1. on the left of `:=` a superscript is part of the name being defined (`\sigma^2`, `\langle x^2 \rangle`, `E_n^{(0)}`), with a note;
2. `^{(n)}` on a function is its nth derivative, and on a new name an order label;
3. `^T`, `^\top` and `^\intercal` on a matrix are its transpose when T has no value, and `^\dagger` is always the conjugate transpose;
4. `^*` after a call or a defined complex quantity is the conjugate, and on a new name part of the name;
5. on a special function's name it is a label;
6. `e^{...}` is always the exponential;
7. a two-index tensor's name alone is its trace, as R is, so K² is (K^μ_μ)².

### 7. What a prime means

**Waiting on it:** `primed-names` (11).

**Why it is open.** It can be a derivative or another frame (x', t', E', F'^{μν}, k'^μ, μ'). Today a prime is always a derivative, so `x' :=` is refused and `\mathbf{r}(t') :=` silently becomes an equation.

**Recommendation.** A prime is a derivative only on a function of one variable: one the program defined, or an unknown function written with its argument. Elsewhere, and always on the left of `:=`, it is part of the name. On an index it marks another frame's index.

### 8. What accents mean

**Waiting on it:** `unit-vectors` (8), `accent-script` (7), `conjugate-bar` (3), `lagrangian-velocities` (3), `accented-definitions` (1).

**Why it is open.** \bar can be a conjugate (complex analysis, \bar m^ρ), a mean (statistics) or the Dirac adjoint. \hat can be a unit vector, an operator or an estimator. \dot can be a time derivative or a generalised velocity. Today they are new names: `\bar z` is zbar, `\hat{\mathbf{x}}` is xhat, and `\dot\theta^2` is the dot of θ².

**Recommendation.** Read each by what it sits on and where:

1. `\bar` of a defined complex quantity, or of z in a complex expression, is the conjugate, and ∂/∂\bar z the Wirtinger derivative;
2. an accented name on the left of `:=` is a new name (`\bar x :=` the mean);
3. `\hat x` and `\hat{\mathbf{x}}`, with x a coordinate, are unit vectors, while `\hat p :=` names an operator;
4. `\dot q` is independent of q in ∂L/∂\dot q and is dq/dt under d/dt, and a script after an accent applies to the accented name.

### 9. Do typefaces make different names?

**Waiting on it:** `typeface-names` (4).

**Why it is open.** Today `\mathbf{r}` is r and `\mathcal{F}` is F, so `\mathbf{N} :=` redefines the builtin N, and `\mathcal{F}^{\alpha\beta} := \ldots` overwrites F. Mechanics writes \mathbf{r} and r = |\mathbf{r}| on one line.

**Recommendation.** Yes, a typeface makes a different name. `\mathcal{L}\{\ldots\}`, with braces, is the Laplace transform; `\mathcal{L} :=` is a Lagrangian density.

### 10. Is LaTeX decided per statement or per program?

**Waiting on it:** `latex-per-statement` (3).

**Why it is open.** Today a statement without a backslash in a LaTeX program is read as plain text, so `6xy` is the name `xy`, `-ka` is `ka`, and `e^{At}` makes `At` one name. The review recorded this; three cases here, and two variants written for this roadmap, met it.

**Recommendation.** A program given as LaTeX (a `latex` fence, a document, `--latex`) is LaTeX in every statement.

### 11. How far an unbracketed function argument runs

**Waiting on it:** `function-argument-extent` (2).

**Why it is open.** Today `\sin (2k-1) x` is x·sin(2k − 1), and `\sin\theta \frac{\partial Y}{\partial\theta}` is sin(θ·∂Y/∂θ).

**Recommendation.** Follow the books: the argument is the following product of factors, up to an operator, a fraction, a derivative or another function. So `\sin (2k-1)x` is sin((2k−1)x) and `\sin\theta\,\frac{a}{b}` is sin θ·a/b. Give a note where both readings are plausible.

### 12. The identity, and numbers beside matrices

**Waiting on it:** `identity-matrix` (7).

**Why it is open.** Books write `A - \lambda I`, `\mathbf{1}`, `\mathbb{1}`, and the Dirac `(\gamma^0 E - \gamma^3 p - m)`. Today `I` is a plain symbol (the imaginary unit is `i`), and a number added to a matrix is refused.

**Recommendation.** An undefined `I`, `\mathbf{1}`, `\mathbb{1}` or 𝟙 in matrix arithmetic is the identity of the matching size, with a note; a defined `I` keeps its value. A number added to a square matrix inside a matrix equation is that multiple of the identity, with a note.

### 13. Brackets

**Waiting on it:** `dirac-notation` (9), `operators` (7), `poisson-bracket` (3), `anticommutator`, `user-notation`, `kl-divergence`.

**Why it is open.** `\{f, g\}` can be a Poisson bracket, an anticommutator or a set; `[A, B]` a commutator, a list or an interval; `\langle\cdot\rangle` an expectation value, an inner product or Dirac notation; `\|` a norm or the separator in D(p‖q). Today `\{L_x, L_y\}` is a tuple, so a true Poisson bracket is judged False.

**Recommendation.** Decide by the operands:

1. `\{f, g\}` of scalar functions is a Poisson bracket, with canonical pairs by name (x with p_x) or given as `_{q,p}`; of matrices or operators it is an anticommutator; otherwise a set;
2. `[A, B]` of operators or matrices is a commutator;
3. `\langle\ldots\rangle` with a bar is Dirac notation; without, an expectation value or a user's bracket (`\langle p, q\rangle :=`);
4. `D(p \| q)` is the relative entropy, in bits (Cover and Thomas).

### 14. What `\sim` and `\approx` claim

**Waiting on it:** `random-variables` (5), `asymptotic-relations` (3), `matrix-similarity`, `periodic-identification`, `perturbative-ansatz`.

**Why it is open.** `\sim` can be asymptotic equivalence (Stirling), a scaling law (m ∼ (T_c − T)^β), 'distributed as' (X ∼ Poisson(λ)), similarity of matrices, or an identification (τ ∼ τ + β). Today it is read and not computed.

**Recommendation.** Decide by the operands:

1. a random variable and a distribution: distributed as;
2. two matrices: similar;
3. a coordinate and itself plus a constant: an identification;
4. two functions with an implied limit: asymptotic equivalence (lim f/g = 1);
5. a power law with an unknown exponent: a scaling law, solved for the exponent;
6. anything else: refused.

`\approx` with unknown parameters is an ansatz to solve order by order; otherwise it stays unchecked.

### 15. Book names that collide with the language's or with each other

**Waiting on it:** `special-functions` (18), `step-delta` (6), `normal-cdf` (2).

**Why it is open.** N(x) is the normal distribution function in Hull and numerical evaluation in the language; Black–Scholes silently gives 8.42 instead of 4.76. Φ is the normal distribution function or a potential. Γ is Euler's gamma or a Christoffel symbol, and H_n the physicists' or the probabilists' Hermite polynomial. u_c(t) is a step or a velocity, δ_0 a phase shift or the Kronecker delta, π_j a vector or the constant.

**Recommendation.** A user's definition always wins. Otherwise:

1. `\Gamma(x)` with one bracketed argument and no indices is Euler's gamma; with indices it stays Christoffel;
2. `\mathrm{B}(a, b)` is the Beta function, and `H_n` the physicists' Hermite polynomial (probabilists write He_n);
3. `u_c(t)` is the unit step where u is not defined;
4. `N(x)` and `\Phi(x)` are the standard normal distribution function, with a note. This retires `N` as numerical evaluation in LaTeX, where `evalf` and `numerical` remain;
5. a name with one index (`\delta_0`, `\pi_j`) is not the two-index Kronecker delta nor the constant.

### 16. Index notation without a spacetime

**Waiting on it:** `index-ranges` (6), `levi-civita` (5), `metric-without-coordinates` (5), `unknown-vectors` (2), `cartesian-indices`, `embedding-indices`, `definition-index-check`.

**Why it is open.** `\eta_{\mu\nu} = \operatorname{diag}(-1, 1, 1, 1)` needs coordinates today. Under a declared Euclidean metric, repeated lower indices are refused. `i, j = 1, 2, 3` is a statement 'read but not computed'. The sign of ε^{0123} is +1 in Jackson and in this curriculum, and −1 in Peskin; MTW and Carroll fix ε_{0123} = +√|g|. A definition relabels free indices by position.

**Recommendation.** Apply these rules:

1. accept a metric given by components alone;
2. sum repeated lower indices under a Euclidean metric (Cartesian tensors), and check free indices on both sides of every definition;
3. read `i, j = 1, 2, 3` and `n = 0, \ldots, N-1` as declaring a range;
4. default ε to Carroll's convention, and let a statement such as `\epsilon^{0123} = 1` set it, with a note;
5. give an index whose components outnumber the coordinates a space of that size;
6. read an undefined name with declared indices as an unknown vector, not a scalar.

### 17. Where a list's positions start

**Waiting on it:** `list-subscript` (3).

**Why it is open.** A sample mean writes `\sum_{i=1}^{n} x_i`; a DFT writes `\sum_{n=0}^{N-1} x_n`. Both on a plain list.

**Recommendation.** Take the base from the sum when it ranges over the whole list. Otherwise refuse, with a hint, rather than guess.

### 18. When d is a differential

**Waiting on it:** `d-variable` (2).

**Why it is open.** Today `d^2` with nothing after it is ∂², and `D^2` is ∇², because D is the covariant derivative. So a current loop of diameter d is refused.

**Recommendation.** d is a differential only before a variable, inside an integral, a derivative or a line element. Anywhere else, and in an evaluation bar's list of values, it is a variable. D is the covariant derivative only with an index.

### 19. Ellipses and index sets given by a condition

**Waiting on it:** `conditional-index-sets` (2), `ellipsis-patterns` (1).

**Why it is open.** Books write 3·5·7⋯(2n+1), `\prod_{p \text{ prime}}` and `\sum_{k \neq 0}`.

**Recommendation.** Infer an arithmetic progression that ends in a general term, with a note, and refuse any other ellipsis. Accept a condition under a sum or product as its index set.

### 20. Operators as values, and rules written with `=`

**Waiting on it:** `operators` (7), `operator-algebra` (3).

**Why it is open.** Books write `\hat p := -i\hbar \frac{d}{dx}`, `[\hat x, \hat p] = i\hbar` and `a_-|n\rangle = \sqrt{n}|n-1\rangle`. Writing each operator as a function of its operand, `Lx(g) := ...`, already makes three cases pass.

**Recommendation.** A definition whose right side is a differential operator defines an operator, applied by juxtaposition and composed by products. An `=` whose left side is a pattern in undefined operators or kets defines a rule for what follows, not a claim.

### 21. Transform conventions

**Waiting on it:** `laplace-fourier` (5).

**Why it is open.** ℒ maps t to s by convention. Fourier transforms differ: Bracewell writes e^{−2πixs}, physics e^{−iωt}. `\supset` (Bracewell's pair) is also the superset.

**Recommendation.** Name the variables by convention (t → s for ℒ; x → s or t → ω for ℱ, by the letters used). Take the Fourier convention from a declaration, or from the 2π in a written integral. `f \supset F` between functions is a Fourier pair.

### 22. Quantified and prose claims

**Waiting on it:** `quantifiers` (6, all level 5).

**Why it is open.** ∀, ∃, 'converges uniformly on' and images of sets are refused today.

**Recommendation.** Keep them refused until a design exists. The first step worth taking is 'for all x in a set' as a claim decided as identities and inequalities are.

## Appendix: every failing case

The first blocker is the first capability the run meets. "Verified" says whether a variant that
emulates every listed capability passed the oracle (full), reached a capability nothing can emulate
(partial), or could not be tried (none). Variants, causes and evidence are in `classification.json`.

| Case | L | Outcome | First blocker | Everything it needs | Verified |
|---|:-:|---|---|---|---|
| `analysis.derivatives.dot-squared` | 1 | refused | `fn-bare` | fn-bare, accent-script | full |
| `analysis.derivatives.implicit` | 3 | wrong (wrong-value) | `latex-per-statement` | latex-per-statement, implicit-relations | full |
| `analysis.derivatives.not-differentiable` | 3 | wrong (wrong-value) | `limit-dne` | limit-dne | partial |
| `analysis.inequalities.am-gm` | 3 | wrong (undecided-claim) | `inequality-assumptions` | inequality-assumptions, decide-inequalities | partial |
| `analysis.inequalities.epsilon-delta` | 5 | refused | `quantifiers` | quantifiers | none |
| `analysis.inequalities.exp-tangent` | 3 | wrong (undecided-claim) | `decide-inequalities` | decide-inequalities | partial |
| `analysis.inequalities.infimum` | 4 | refused | `inf-sup` | inf-sup | full |
| `analysis.integrals.by-parts` | 1 | refused | `eval-bracket` | eval-bracket | full |
| `analysis.integrals.gaussian` | 2 | wrong (right-other-form) | `inequality-assumptions` | inequality-assumptions | full |
| `analysis.limits.does-not-exist` | 2 | refused | `limit-dne` | limit-dne | partial |
| `analysis.multivariable.jacobian` | 2 | wrong (unevaluated) | `jacobian-notation` | jacobian-notation | full |
| `analysis.multivariable.lagrange` | 3 | refused | `vector-calculus` | vector-calculus | full |
| `analysis.multivariable.region` | 4 | refused | `region-integrals` | region-integrals | full |
| `analysis.multivariable.second-derivative-test` | 2 | refused | `subscript-partials` | subscript-partials, latex-per-statement, derivative-in-function-body | full |
| `analysis.series.basel` | 2 | wrong (unevaluated) | `closed-form-sums` | closed-form-sums | full |
| `analysis.series.harmonic` | 2 | refused | `closed-form-sums` | closed-form-sums | full |
| `analysis.series.limsup` | 4 | refused | `inf-sup` | inf-sup | full |
| `analysis.series.radius-ratio` | 3 | wrong (wrong-value) | `family` | family | full |
| `analysis.taylor.big-o` | 2 | wrong (undecided-claim) | `order-terms` | order-terms | partial |
| `analysis.taylor.nth-derivative` | 2 | wrong (wrong-value) | `nth-derivative` | nth-derivative | full |
| `analysis.taylor.stirling` | 4 | refused | `asymptotic-relations` | asymptotic-relations | full |
| `analysis.uniform.interchange` | 3 | refused | `integration-variable-range` | integration-variable-range, default-assumptions | full |
| `analysis.uniform.m-test` | 5 | refused | `quantifiers` | quantifiers | none |
| `analysis.uniform.not-uniform` | 5 | refused | `quantifiers` | quantifiers | none |
| `analysis.uniform.uniform-continuity` | 5 | refused | `quantifiers` | quantifiers | none |
| `cm.central.circular-orbit` | 2 | wrong (wrong-value) | `subscripted-function` | subscripted-function | full |
| `cm.central.runge-lenz` | 3 | refused | `vector-algebra` | vector-algebra, typeface-names, poisson-bracket | full |
| `cm.hamiltonian.action-oscillator` | 4 | wrong (wrong-value) | `implicit-relations` | implicit-relations, default-assumptions | full |
| `cm.hamiltonian.angular-momentum` | 2 | wrong (wrong-value) | `poisson-bracket` | poisson-bracket | full |
| `cm.hamiltonian.canonical-bracket` | 4 | refused | `poisson-bracket` | poisson-bracket, default-assumptions | full |
| `cm.hamiltonian.hamilton-jacobi` | 5 | wrong (wrong-value) | `pde-by-ansatz` | pde-by-ansatz, default-assumptions | full |
| `cm.hamiltonian.legendre` | 4 | refused | `extremum` | extremum | full |
| `cm.lagrangian.bead-on-hoop` | 3 | refused | `accent-script` | accent-script, bar-in-definition | full |
| `cm.lagrangian.double-pendulum` | 3 | refused | `accent-script` | accent-script, indexed-variables | full |
| `cm.lagrangian.hamilton-principle` | 5 | refused | `accent-script` | accent-script, functionals, lagrangian-velocities | partial |
| `cm.lagrangian.index-form` | 4 | wrong (wrong-value) | `curve-tensors` | curve-tensors | full |
| `cm.lagrangian.noether` | 4 | refused | `list-subscript` | list-subscript, accent-script, lagrangian-velocities | full |
| `cm.lagrangian.pendulum` | 2 | refused | `accent-script` | accent-script, lagrangian-velocities | full |
| `cm.oscillations.mismatched-index` | 1 | wrong (accepted-invalid) | `definition-index-check` | definition-index-check | partial |
| `cm.oscillations.principal-moments` | 2 | refused | `identity-matrix` | identity-matrix | full |
| `complex.analytic.wirtinger` | 4 | wrong (wrong-value) | `conjugate-bar` | conjugate-bar | full |
| `complex.maps.argument-principle` | 4 | refused | `residues-contours` | residues-contours | partial |
| `complex.maps.cayley` | 3 | refused | `inequality-assumptions` | inequality-assumptions | full |
| `complex.maps.joukowski` | 5 | refused | `quantifiers` | quantifiers | none |
| `complex.maps.principal-log` | 4 | wrong (unevaluated) | `special-functions` | special-functions | full |
| `complex.numbers.conjugate` | 1 | refused | `conjugate-bar` | conjugate-bar | full |
| `complex.numbers.euler` | 1 | refused | `exp-not-tensor` | exp-not-tensor | full |
| `complex.residues.essential-singularity` | 2 | wrong (unevaluated) | `residues-contours` | residues-contours | partial |
| `complex.residues.principal-value` | 4 | refused | `residues-contours` | residues-contours | partial |
| `complex.residues.real-integral` | 3 | wrong (undecided-claim) | `residues-contours` | residues-contours | full |
| `complex.residues.residue-theorem` | 3 | refused | `residues-contours` | residues-contours | partial |
| `data.finance.black-scholes-call` | 2 | wrong (wrong-value) | `normal-cdf` | normal-cdf | full |
| `data.finance.black-scholes-derivation` | 5 | refused | `stochastic-calculus` | stochastic-calculus, ode-unknown | partial |
| `data.finance.black-scholes-pde` | 3 | wrong (undecided-claim) | `decide-identities` | decide-identities | full |
| `data.learning.pca` | 3 | refused | `space-index-sums` | space-index-sums | full |
| `data.learning.softmax` | 2 | refused | `indexed-function-value` | indexed-function-value, unbraced-index, space-index-sums | full |
| `data.portfolio.argmin` | 4 | refused | `extremum` | extremum | full |
| `data.portfolio.capm-beta` | 2 | refused | `unbraced-index` | unbraced-index | full |
| `data.fft.dft-sum` | 3 | refused | `python-scope` | python-scope, index-ranges, space-index-sums | partial |
| `data.fft.named` | 2 | refused | `python-scope` | python-scope, fft | partial |
| `data.fft.sampled-tone` | 3 | refused | `extremum` | extremum, index-ranges, fft, indexed-function-value | partial |
| `data.grids.uniform` | 2 | refused | `compound-names` | compound-names, index-ranges, space-index-sums | partial |
| `data.objects.aligned` | 3 | refused | `python-scope` | python-scope, space-by-labels | full |
| `data.objects.column` | 1 | refused | `python-scope` | python-scope, space-index-sums | full |
| `data.objects.constants` | 2 | refused | `python-scope` | python-scope | full |
| `data.objects.dataframe` | 1 | refused | `python-scope` | python-scope | full |
| `data.objects.function` | 3 | refused | `python-scope` | python-scope, index-ranges, space-index-sums | partial |
| `data.objects.key` | 1 | refused | `python-scope` | python-scope, extremum | partial |
| `data.objects.sympy` | 2 | refused | `python-scope` | python-scope | full |
| `de.bvp.rayleigh` | 4 | refused | `functionals` | functionals | full |
| `de.bvp.robin` | 5 | wrong (unevaluated) | `ode-unknown` | ode-unknown, bvp-eigenvalues | none |
| `de.first-order.exact` | 3 | refused | `fn-bare` | fn-bare | full |
| `de.first-order.growth` | 1 | refused | `ode-conditions` | ode-conditions, ode-unknown | full |
| `de.first-order.linear` | 2 | refused | `ode-conditions` | ode-conditions, ode-unknown | full |
| `de.first-order.separable` | 2 | refused | `ode-unknown` | ode-unknown, ode-conditions | partial |
| `de.pde.heat` | 3 | wrong (unevaluated) | `family` | family | full |
| `de.pde.heat-kernel` | 5 | refused | `distribution-limits` | distribution-limits, step-delta | partial |
| `de.second-order.harmonic` | 2 | refused | `fn-bare` | fn-bare | full |
| `de.second-order.impulse` | 3 | refused | `ode-unknown` | ode-unknown, step-delta, ode-conditions | partial |
| `de.second-order.resonance` | 2 | refused | `fn-bare` | fn-bare | full |
| `de.second-order.system` | 2 | refused | `fn-bare` | fn-bare | full |
| `de.second-order.wronskian` | 1 | refused | `fn-bare` | fn-bare | full |
| `de.series.airy` | 3 | refused | `recurrence` | recurrence, family | full |
| `de.series.euler-exponents` | 4 | refused | `functionals` | functionals | full |
| `de.series.frobenius` | 4 | refused | `fn-bare` | fn-bare, ellipsis-patterns | full |
| `dg.connections.parallel-transport` | 4 | refused | `curve-tensors` | curve-tensors | full |
| `dg.connections.spin-connection` | 4 | refused | `component-definitions` | component-definitions, differential-forms, index-solve | partial |
| `dg.forms.exterior-derivative` | 4 | refused | `differential-forms` | differential-forms | partial |
| `dg.forms.hodge-star` | 4 | refused | `differential-forms` | differential-forms | partial |
| `dg.forms.stokes-theorem` | 5 | refused | `differential-forms` | differential-forms, region-integrals, boundary-of-region | partial |
| `dg.index-notation.epsilon-delta` | 2 | wrong (undecided-claim) | `levi-civita` | levi-civita | full |
| `dg.surfaces.induced-metric` | 3 | refused | `embedding-indices` | embedding-indices | full |
| `dg.surfaces.theorema-egregium` | 3 | wrong (wrong-value) | `vector-algebra` | vector-algebra | full |
| `em.field-tensor.boost` | 2 | refused | `primed-names` | primed-names | full |
| `em.field-tensor.dual-invariant` | 4 | refused | `levi-civita` | levi-civita, typeface-names | full |
| `em.field-tensor.e-dot-b` | 3 | refused | `primed-names` | primed-names, vector-algebra | full |
| `em.maxwell.gauss` | 2 | refused | `vector-calculus` | vector-calculus | full |
| `em.sources.current-loop` | 3 | refused | `vector-algebra` | vector-algebra | full |
| `em.sources.dipole` | 2 | wrong (wrong-value) | `bar-precedence` | bar-precedence | full |
| `em.sources.legendre` | 4 | wrong (undecided-claim) | `special-functions` | special-functions, series-identities | partial |
| `em.sources.lienard-wiechert` | 5 | refused | `typeface-names` | typeface-names, primed-names, vector-algebra, retarded-time | partial |
| `em.sources.loop-moment` | 1 | refused | `d-variable` | d-variable | full |
| `em.waves.poynting` | 2 | wrong (wrong-value) | `unit-vectors` | unit-vectors, vector-algebra | full |
| `em.waves.retarded-green` | 5 | refused | `vector-calculus` | vector-calculus, typeface-names, operators, step-delta, distribution-limits | partial |
| `em.waves.stress-energy-conservation` | 4 | refused | `abstract-definitions` | abstract-definitions, abstract-derivatives, abstract-hypotheses | partial |
| `fa.laplace.convolution` | 3 | wrong (undecided-claim) | `laplace-fourier` | laplace-fourier | full |
| `fa.laplace.derivative` | 2 | wrong (undecided-claim) | `laplace-fourier` | laplace-fourier | full |
| `fa.laplace.power` | 1 | wrong (unevaluated) | `laplace-fourier` | laplace-fourier | full |
| `fa.laplace.step` | 4 | wrong (unevaluated) | `laplace-fourier` | laplace-fourier, step-delta | full |
| `fa.series.convergence` | 3 | wrong (unevaluated) | `closed-form-sums` | closed-form-sums, sum-index-integer, parity-split-sums | partial |
| `fa.series.gibbs` | 4 | refused | `riemann-sum-limits` | riemann-sum-limits, function-argument-extent | partial |
| `fa.series.parseval` | 3 | timeout | `sum-index-integer` | sum-index-integer | full |
| `fa.transforms.delta` | 5 | wrong (unevaluated) | `distribution-limits` | distribution-limits | none |
| `fa.transforms.dft` | 2 | wrong (wrong-value) | `list-subscript` | list-subscript, index-ranges | partial |
| `fa.transforms.pair` | 4 | refused | `laplace-fourier` | laplace-fourier, default-assumptions, piecewise-removable | partial |
| `fa.transforms.rectangle` | 2 | wrong (right-other-form) | `piecewise-removable` | piecewise-removable | partial |
| `gr.black-holes.euclidean-period` | 5 | refused | `periodic-identification` | periodic-identification | partial |
| `gr.black-holes.kerr-horizon-area` | 3 | refused | `bar-in-integrand` | bar-in-integrand, label-subscript | full |
| `gr.black-holes.null-energy-condition` | 5 | refused | `quantifiers` | quantifiers, decide-inequalities | partial |
| `gr.black-holes.reissner-nordstrom-horizons` | 2 | wrong (wrong-value) | `solve-simplify` | solve-simplify | full |
| `gr.black-holes.reissner-nordstrom-temperature` | 3 | refused | `label-subscript` | label-subscript | full |
| `gr.cosmology.hubble-matter` | 1 | refused | `fn-bare` | fn-bare | full |
| `gr.cosmology.radiation-era` | 3 | refused | `fn-bare` | fn-bare | full |
| `gr.formalisms.hamiltonian-constraint` | 5 | refused | `trace-by-name` | trace-by-name, induced-curvature | full |
| `gr.formalisms.newman-penrose-psi2` | 4 | refused | `label-subscript` | label-subscript, conjugate-bar | full |
| `gr.formalisms.painleve-gullstrand-lapse` | 2 | refused | `line-element-products` | line-element-products | full |
| `gr.formalisms.penrose-compactification` | 5 | refused | `coordinate-transformations` | coordinate-transformations | full |
| `gr.geodesics.circular-geodesic` | 4 | refused | `curve-tensors` | curve-tensors | full |
| `gr.geodesics.light-bending` | 5 | refused | `differential-numerator` | differential-numerator, compound-names, order-terms, perturbative-integrals | partial |
| `gr.geodesics.light-deflection-sun` | 1 | refused | `label-subscript` | label-subscript | full |
| `gr.geodesics.mercury-perihelion` | 1 | wrong (unevaluated) | `compound-names` | compound-names | full |
| `gr.geodesics.perihelion-advance` | 5 | refused | `ode-unknown` | ode-unknown, asymptotic-relations, perturbative-ansatz | partial |
| `gr.schwarzschild.redshift` | 3 | wrong (wrong-value) | `function-of-tensor-body` | function-of-tensor-body | full |
| `gr.spacetimes.eddington-finkelstein` | 3 | refused | `coordinate-transformations` | coordinate-transformations | full |
| `gr.waves.linearised-riemann` | 4 | wrong (wrong-value) | `label-subscript` | label-subscript | full |
| `gr.waves.polarisation-rotation` | 3 | refused | `staggered-unbraced` | staggered-unbraced | full |
| `la.eigen.characteristic` | 2 | refused | `identity-matrix` | identity-matrix | full |
| `la.eigen.exponential` | 2 | wrong (wrong-value) | `latex-per-statement` | latex-per-statement, matrix-functions | full |
| `la.eigen.jordan` | 4 | refused | `matrix-similarity` | matrix-similarity | partial |
| `la.eigen.powers` | 3 | refused | `matrix-functions` | matrix-functions | full |
| `la.eigen.spectral` | 2 | refused | `transpose-dagger` | transpose-dagger | full |
| `la.eigen.zeta-determinant` | 5 | refused | `ode-unknown` | ode-unknown, operators, bvp-eigenvalues, functional-determinant | partial |
| `la.index.bac-cab` | 3 | refused | `levi-civita` | levi-civita, unbraced-index, unknown-vectors | full |
| `la.index.delta-trace` | 1 | refused | `index-ranges` | index-ranges | full |
| `la.index.epsilon-delta` | 2 | refused | `levi-civita` | levi-civita | full |
| `la.matrices.rank` | 2 | wrong (wrong-value) | `matrix-functions` | matrix-functions | partial |
| `la.matrices.trace` | 1 | wrong (wrong-value) | `matrix-functions` | matrix-functions | full |
| `la.orthogonality.gram-schmidt` | 3 | refused | `user-notation` | user-notation | full |
| `la.orthogonality.least-squares` | 2 | refused | `transpose-dagger` | transpose-dagger | full |
| `la.orthogonality.qr` | 2 | refused | `transpose-dagger` | transpose-dagger | full |
| `la.orthogonality.svd` | 2 | refused | `transpose-dagger` | transpose-dagger | full |
| `prob.bayes.conditional-notation` | 4 | refused | `probability-notation` | probability-notation | full |
| `prob.bayes.total-expectation` | 4 | refused | `random-variables` | random-variables | partial |
| `prob.distributions.gamma-variance` | 2 | wrong (unevaluated) | `special-functions` | special-functions, default-assumptions | full |
| `prob.distributions.poisson-limit` | 1 | refused | `default-assumptions` | default-assumptions | full |
| `prob.distributions.poisson-tilde` | 4 | refused | `random-variables` | random-variables | partial |
| `prob.estimation.correlation-matrix` | 2 | refused | `matrix-functions` | matrix-functions | full |
| `prob.estimation.normal-equations` | 2 | refused | `transpose-dagger` | transpose-dagger | full |
| `prob.estimation.sample-variance` | 2 | wrong (wrong-value) | `accented-definitions` | accented-definitions, list-subscript | full |
| `prob.information.kl-notation` | 4 | refused | `kl-divergence` | kl-divergence | full |
| `prob.limits.chebyshev` | 2 | refused | `superscript-label` | superscript-label | full |
| `prob.limits.normal-approximation` | 2 | wrong (unevaluated) | `normal-cdf` | normal-cdf | full |
| `prob.markov.absorbing` | 3 | refused | `identity-matrix` | identity-matrix | full |
| `prob.markov.stationary-index` | 4 | refused | `unbraced-index` | unbraced-index, unknown-vectors, index-solve | partial |
| `prob.moments.normal-fourth-moment` | 3 | wrong (wrong-value) | `nth-derivative` | nth-derivative | full |
| `prob.moments.variance-affine` | 4 | wrong (undecided-claim) | `random-variables` | random-variables | partial |
| `prob.stochastic.ito-integral` | 5 | refused | `stochastic-calculus` | stochastic-calculus | partial |
| `prob.stochastic.ito-isometry` | 5 | refused | `stochastic-calculus` | stochastic-calculus, random-variables | partial |
| `prob.stochastic.ito-log` | 5 | refused | `stochastic-calculus` | stochastic-calculus | partial |
| `prob.stochastic.ou-variance` | 5 | refused | `stochastic-calculus` | stochastic-calculus, random-variables | partial |
| `qft.dirac.clifford` | 4 | refused | `component-definitions` | component-definitions, block-matrices, gamma-matrices, family, anticommutator, identity-matrix, metric-without-coordinates | partial |
| `qft.dirac.plane-wave` | 2 | refused | `component-definitions` | component-definitions, identity-matrix | full |
| `qft.dirac.slash` | 4 | refused | `gamma-matrices` | gamma-matrices, abstract-vectors, identity-matrix | none |
| `qft.dirac.trace` | 4 | wrong (undecided-claim) | `gamma-matrices` | gamma-matrices, matrix-functions, metric-without-coordinates | none |
| `qft.interactions.gaussian` | 1 | wrong (right-other-form) | `default-assumptions` | default-assumptions | full |
| `qft.interactions.generating-functional` | 5 | refused | `functionals` | functionals, region-integrals, step-delta | none |
| `qft.interactions.loop-integral` | 5 | refused | `differential-numerator` | differential-numerator, frontier-qft | partial |
| `qft.interactions.muon-pair` | 5 | refused | `frontier-qft` | frontier-qft, gamma-matrices, primed-names, solid-angle | none |
| `qft.interactions.phi4-vertex` | 5 | refused | `frontier-qft` | frontier-qft, field-derivatives | none |
| `qft.interactions.running-coupling` | 5 | refused | `derivative-wrt-expression` | derivative-wrt-expression, ode-unknown | none |
| `qft.interactions.solid-angle` | 1 | wrong (wrong-value) | `solid-angle` | solid-angle | full |
| `qft.interactions.wick` | 5 | wrong (wrong-value) | `dirac-notation` | dirac-notation, frontier-qft | none |
| `qft.klein-gordon.dispersion` | 2 | refused | `fn-bare` | fn-bare | full |
| `qft.klein-gordon.euler-lagrange` | 4 | refused | `field-derivatives` | field-derivatives, metric-without-coordinates | none |
| `qft.klein-gordon.feynman-propagator` | 4 | refused | `differential-numerator` | differential-numerator, component-variable, inequality-assumptions, residues-contours | partial |
| `qft.klein-gordon.hamiltonian` | 3 | refused | `field-derivatives` | field-derivatives, accent-script | full |
| `qm.angular-momentum.clebsch-gordan` | 4 | refused | `dirac-notation` | dirac-notation, clebsch-gordan | partial |
| `qm.angular-momentum.l-squared` | 2 | wrong (undecided-claim) | `superscript-label` | superscript-label, fn-bare, function-argument-extent | full |
| `qm.angular-momentum.singlet` | 2 | refused | `label-subscript` | label-subscript, kronecker-product | full |
| `qm.commutators.angular-momentum` | 3 | refused | `operators` | operators | full |
| `qm.commutators.canonical` | 2 | refused | `operators` | operators | full |
| `qm.commutators.gaussian-packet` | 3 | refused | `superscript-label` | superscript-label, star-conjugate, operators, default-assumptions | full |
| `qm.commutators.power-of-x` | 4 | refused | `operator-algebra` | operator-algebra | partial |
| `qm.dirac.change-of-basis` | 2 | refused | `transpose-dagger` | transpose-dagger | full |
| `qm.dirac.completeness` | 4 | refused | `dirac-notation` | dirac-notation | full |
| `qm.dirac.inner-product` | 4 | refused | `dirac-notation` | dirac-notation | full |
| `qm.dirac.momentum-wavefunction` | 4 | refused | `dirac-notation` | dirac-notation, d-variable, default-assumptions | full |
| `qm.hydrogen.degeneracy` | 1 | wrong (unevaluated) | `closed-form-sums` | closed-form-sums | full |
| `qm.hydrogen.mean-radius` | 2 | wrong (unevaluated) | `subscripted-function` | subscripted-function, default-assumptions | full |
| `qm.hydrogen.radial-equation` | 3 | refused | `subscripted-function` | subscripted-function, fn-bare | full |
| `qm.information.entropy` | 2 | refused | `matrix-functions` | matrix-functions | partial |
| `qm.information.partial-trace` | 4 | refused | `dirac-notation` | dirac-notation, kronecker-product, superscript-label, partial-trace | full |
| `qm.information.purity` | 2 | refused | `matrix-functions` | matrix-functions | full |
| `qm.information.tensor-product` | 1 | wrong (wrong-value) | `kronecker-product` | kronecker-product | full |
| `qm.oscillator.ground-state` | 2 | wrong (undecided-claim) | `subscripted-function` | subscripted-function, fn-bare | full |
| `qm.oscillator.hermite-wavefunction` | 2 | wrong (unevaluated) | `subscripted-function` | subscripted-function, family, special-functions, default-assumptions | full |
| `qm.oscillator.ladder-commutator` | 3 | refused | `operators` | operators, pm-names | full |
| `qm.oscillator.potential-energy` | 4 | refused | `dirac-notation` | dirac-notation, primed-names, label-subscript, operator-algebra | partial |
| `qm.perturbation.first-order` | 2 | refused | `primed-names` | primed-names, subscripted-function, star-conjugate, default-assumptions | full |
| `qm.perturbation.second-order` | 3 | refused | `subscripted-function` | subscripted-function, label-subscript, superscript-label, star-conjugate, default-assumptions | full |
| `qm.perturbation.stark` | 3 | refused | `subscripted-function` | subscripted-function, label-subscript, star-conjugate, default-assumptions | full |
| `qm.perturbation.sum-over-states` | 4 | refused | `dirac-notation` | dirac-notation, primed-names, operator-algebra, transpose-dagger, superscript-label, family, conditional-index-sets | partial |
| `qm.scattering.born-yukawa` | 4 | refused | `primed-names` | primed-names, region-integrals, vector-algebra, superscript-label, radial-fourier-3d | partial |
| `qm.scattering.hard-sphere` | 2 | refused | `label-subscript` | label-subscript | full |
| `qm.scattering.soft-sphere` | 3 | wrong (wrong-value) | `default-assumptions` | default-assumptions | full |
| `qm.spin.bra-ket-expectation` | 4 | refused | `dirac-notation` | dirac-notation, label-subscript | full |
| `qm.spin.pauli-commutator` | 2 | refused | `label-subscript` | label-subscript | full |
| `qm.spin.rotation` | 2 | refused | `label-subscript` | label-subscript, matrix-functions | full |
| `qm.square-well.expansion` | 3 | wrong (unevaluated) | `subscripted-function` | subscripted-function, family, star-conjugate, default-assumptions | full |
| `qm.square-well.normalisation` | 1 | wrong (unevaluated) | `default-assumptions` | default-assumptions | full |
| `qm.square-well.orthogonality` | 2 | wrong (wrong-value) | `subscripted-function` | subscripted-function, family, star-conjugate, default-assumptions | full |
| `qm.square-well.uncertainty` | 3 | refused | `subscripted-function` | subscripted-function, superscript-label, operators, label-subscript, default-assumptions | full |
| `qm.time-evolution.beats` | 3 | wrong (unevaluated) | `subscripted-function` | subscripted-function, family, letter-run-subscript, default-assumptions | full |
| `qm.time-evolution.path-integral` | 5 | refused | `frontier-qft` | frontier-qft | none |
| `qm.time-evolution.precession` | 3 | refused | `transpose-dagger` | transpose-dagger, matrix-functions | full |
| `qm.time-evolution.unitarity` | 2 | refused | `transpose-dagger` | transpose-dagger, matrix-functions, default-assumptions | full |
| `special.bessel.equation` | 2 | wrong (undecided-claim) | `special-functions` | special-functions | full |
| `special.bessel.half-order` | 2 | wrong (undecided-claim) | `special-functions` | special-functions, decide-identities | partial |
| `special.bessel.value` | 1 | wrong (unevaluated) | `special-functions` | special-functions | full |
| `special.bessel.zeros` | 4 | refused | `bessel-zeros` | bessel-zeros | partial |
| `special.erf.asymptotic` | 2 | wrong (wrong-value) | `special-functions` | special-functions | full |
| `special.gamma.beta` | 1 | wrong (unevaluated) | `special-functions` | special-functions | full |
| `special.gamma.half` | 1 | wrong (unevaluated) | `special-functions` | special-functions | full |
| `special.gamma.reflection` | 2 | wrong (undecided-claim) | `special-functions` | special-functions | full |
| `special.harmonics.unsold` | 2 | refused | `special-functions` | special-functions, decide-identities | partial |
| `special.harmonics.y10` | 1 | wrong (wrong-value) | `special-functions` | special-functions | full |
| `special.hermite.orthogonality` | 2 | wrong (unevaluated) | `special-functions` | special-functions | full |
| `special.hypergeometric.arcsin` | 4 | refused | `hypergeometric` | hypergeometric | partial |
| `special.hypergeometric.gauss` | 4 | refused | `hypergeometric` | hypergeometric | partial |
| `special.hypergeometric.logarithm` | 4 | refused | `hypergeometric` | hypergeometric | partial |
| `special.hypergeometric.pochhammer` | 4 | refused | `pochhammer` | pochhammer | full |
| `special.legendre.generating-function` | 3 | wrong (undecided-claim) | `special-functions` | special-functions, series-identities | partial |
| `special.legendre.p3` | 1 | wrong (unevaluated) | `special-functions` | special-functions | full |
| `special.zeta.euler-product` | 5 | refused | `conditional-index-sets` | conditional-index-sets, decide-identities | partial |
| `special.zeta.functional-equation` | 3 | wrong (undecided-claim) | `special-functions` | special-functions, decide-identities | partial |
| `special.zeta.two` | 1 | wrong (unevaluated) | `special-functions` | special-functions | full |
| `sr.four-momentum.compton` | 3 | refused | `primed-names` | primed-names, solve-simplify | full |
| `sr.four-momentum.mandelstam` | 4 | wrong (undecided-claim) | `abstract-vectors` | abstract-vectors, abstract-hypotheses, family | none |
| `sr.four-momentum.two-body-phase-space` | 5 | refused | `differential-numerator` | differential-numerator, region-integrals, metric-without-coordinates, family, vector-algebra, step-delta | none |
| `sr.four-momentum.two-photon-mass` | 2 | wrong (wrong-value) | `label-index-subscript` | label-index-subscript | full |
| `sr.lorentz.index-form` | 4 | refused | `metric-without-coordinates` | metric-without-coordinates, primed-names, decide-identities | full |
| `sr.lorentz.preserves-metric` | 2 | refused | `transpose-dagger` | transpose-dagger | full |
| `sr.lorentz.rapidities-add` | 2 | wrong (undecided-claim) | `decide-identities` | decide-identities | full |
| `sr.lorentz.velocity-composition` | 3 | refused | `primed-names` | primed-names | full |
| `sr.lorentz.wigner-rotation` | 5 | wrong (unevaluated) | `matrix-factorisation-equations` | matrix-factorisation-equations | none |
| `statmech.canonical.ideal-gas` | 2 | wrong (right-other-form) | `default-assumptions` | default-assumptions | full |
| `statmech.canonical.oscillator` | 2 | wrong (unevaluated) | `closed-form-sums` | closed-form-sums, geometric-exponential-sums, default-assumptions | partial |
| `statmech.canonical.rigid-rotor` | 2 | wrong (right-other-form) | `default-assumptions` | default-assumptions | full |
| `statmech.ising.free-energy` | 3 | refused | `label-subscript` | label-subscript, dominant-term-limits | partial |
| `statmech.ising.transfer-eigenvalues` | 2 | refused | `identity-matrix` | identity-matrix | full |
| `statmech.meanfield.critical-exponent` | 5 | refused | `asymptotic-relations` | asymptotic-relations, implicit-relations | partial |
| `statmech.meanfield.curie-weiss` | 4 | wrong (wrong-value) | `implicit-relations` | implicit-relations | partial |
| `statmech.quantum.planck-distribution` | 2 | wrong (unevaluated) | `closed-form-sums` | closed-form-sums, geometric-exponential-sums, default-assumptions | partial |
| `statmech.quantum.planck-integral` | 1 | wrong (unevaluated) | `special-integrals` | special-integrals | partial |
| `statmech.quantum.stefan-boltzmann` | 3 | wrong (unevaluated) | `special-integrals` | special-integrals, default-assumptions | partial |
| `statmech.thermo.maxwell-relation` | 4 | refused | `thermo-partials` | thermo-partials | full |
| `vc.integrals.delta` | 5 | refused | `vector-calculus` | vector-calculus, unit-vectors, distribution-limits | partial |
| `vc.integrals.divergence-theorem` | 3 | refused | `vector-calculus` | vector-calculus, unit-vectors, vector-algebra | full |
| `vc.integrals.form` | 4 | refused | `differential-forms` | differential-forms, region-integrals | partial |
| `vc.integrals.line` | 3 | refused | `vector-algebra` | vector-algebra, unit-vectors, field-composition | full |
| `vc.integrals.stokes` | 3 | refused | `bar-in-integrand` | bar-in-integrand, vector-calculus, unit-vectors, vector-algebra | full |
| `vc.operators.curl` | 1 | refused | `vector-calculus` | vector-calculus, unit-vectors | full |
| `vc.operators.curl-curl` | 4 | refused | `cartesian-indices` | cartesian-indices, levi-civita | full |
| `vc.operators.curl-cylindrical` | 2 | refused | `vector-calculus` | vector-calculus, unit-vectors | full |
| `vc.operators.curl-grad` | 2 | refused | `vector-calculus` | vector-calculus, vector-equations | full |
| `vc.operators.divergence` | 1 | refused | `vector-calculus` | vector-calculus, unit-vectors | full |
