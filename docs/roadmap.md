# Roadmap

RelativisticPy began with one idea: Einstein's equations should look exactly as
they do in papers, with an engine that can compute them. These are the plans
that grew from it, and where each stands.

## Done

- **Equations that look like LaTeX.** Plain-text and LaTeX notation, mixed
  freely; paste an equation from a paper and it runs.
- **A context-sensitive grammar** for tensors: `G_{a=0 b}` selects components
  rather than building a list, and `x^{a}` is a component or a power depending
  on what `x` is.
- **The built-in tensors**: metric, Christoffel symbols, Riemann, Ricci, Einstein
  and Weyl tensors, Ricci and Kretschmann scalars, the Kronecker delta.
- **Covariant derivatives**, in `\nabla_\mu` and semicolon notation, and partial
  derivatives in `\partial_\mu` and comma notation.
- **Symmetric and antisymmetric parts**, `T_{(ab)}` and `T_{[ab]}`, including
  brackets that span tensors.
- **Defining tensors** from components or from expressions, with indices
  reordered to match.
- **Metrics from line elements**, `ds^2 = ...`, including `d\Omega^2`.
- **The geodesic equation**, symbolically, and integrated numerically with SciPy.
- **Known spacetimes**: Minkowski, Schwarzschild, Reissner–Nordström, Kerr,
  Kerr–Newman, FLRW, de Sitter and anti-de Sitter.
- **Markdown notebooks**: run the equations of a Markdown document and get it
  back with the results typeset.
- **An intermediate representation** with an executor that computes lazily,
  shares common subexpressions and works in parallel.
- **Good errors**, located in the source, with hints.
- **Checking papers**: run a paper's equations and see which hold. Sean
  Carroll's lecture notes are the benchmark: 98.7% of their equations parse.
- **Tensor equations solved as tensor equations**: `solve`, `trace`,
  `simplify` and `subs` in index notation, with the identities of General
  Relativity, before there are components, and components after.
- **Beyond spacetime**: indices over spaces of any size, data from NumPy and
  pandas, statistics as contractions with a measure, spectra and linear algebra,
  sparse data, and JAX for gradients; portfolios, quantum systems, graphs and
  regressions in the notation of General Relativity.

## Next

- **Derivatives in index notation**: \(\nabla_\mu\) and \(\partial_\mu\) as
  part of the notation, with the Leibniz rule, metric compatibility and the
  contracted Bianchi identity \(\nabla^\mu G_{\mu\nu} = 0\), so that
  conservation laws can be derived rather than only checked.
- **Abstract definitions**: `T_{\mu\nu} := (\rho + p) u_\mu u_\nu + p g_{\mu\nu}`
  with a velocity that has no components yet, kept as notation until it has.
- **A visual engine, with [Manim](https://www.manim.community).** The
  animation library behind 3Blue1Brown, driven by RelativisticPy: derivations
  that animate step by step, as chains of equalities already are; orbits and
  light rays integrated by `relativisticpy.numeric`; light cones tilting
  towards a horizon; embedding diagrams and Penrose diagrams. Visuals would be
  a scene library, `relativisticpy.visual`, behind an optional
  `relativisticpy[visual]` extra, so the core keeps SymPy as its only
  dependency.
- **More of physics' implicit conventions**: \(x^\mu\) as the coordinates
  themselves, coordinates along a curve, \(x^\mu(\lambda)\), in geodesic
  equations, and \(\eta_{\mu\nu}\) as the Minkowski metric, so that more of a
  paper can be checked.
- **Coordinate transformations**: transform tensors between coordinate systems
  with the Jacobian, e.g. Cartesian to spherical.
- **Energy-momentum tensors** for common sources (perfect fluids, the
  electromagnetic field) as built-ins, to state field equations in full.
- **More spacetimes**: Oppenheimer–Snyder collapse, Lemaître–Tolman–Bondi dust,
  Schwarzschild–de Sitter.
- **Fields on grids**: the components of a tensor field evaluated over a grid
  of points by the numerical kernels, rather than one point at a time, for
  numerical relativity and for pictures.
- **Optimisation in notation**: minimise \(w^a \Sigma_{ab} w^b\) subject to
  constraints, with gradients from JAX; Dirac's bra-ket notation,
  \(\langle\psi|H|\psi\rangle\); more of scikit-learn's estimators.
- **Tensor and spacetime visualisation**: embedding diagrams, light cones,
  Penrose–Carter diagrams.
- **The ADM formalism** and tools for numerical relativity.
- **A language server**, for highlighting and errors in editors.

## Someday

Hawking radiation and the Penrose process as worked modules; analysis of real
data from gravitational-wave observatories; interactive tutorials for people new
to the subject.

Ideas and contributions are welcome:
[open an issue](https://github.com/cottrellashley/relativisticpy/issues).
