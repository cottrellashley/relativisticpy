# Roadmap

> **For:** anyone deciding whether RelativisticPy will do what they need, and anyone
> who would like to help. **You will learn:** what is done, what is being done now, and
> what comes next.

RelativisticPy began with one idea: Einstein's equations should look exactly as they do
in papers, with an engine that can compute them. It grew into a language that reads
mathematics in general, as it is written. These are its plans, and where each stands.
What was left on purpose, and why, is in `design/debt.md` in the repository.

## Done

- **Mathematics as it is written.** LaTeX, plain text and Unicode, mixed freely. Plain
  text and Unicode are read as the LaTeX they spell, or refused. An equation pasted from
  a paper runs as it is.
- **The core**: algebra; calculus, with derivatives in every notation, integrals,
  limits, sums, products, series, `solve` and `dsolve`; linear algebra; and index
  notation over any space, its rules checked before anything is computed.
- **Fields in their books' notation**: special functions, vectors and matrices,
  probability, signals, optimisation, number theory and combinatorics, and quantum
  mechanics, with Dirac's notation and operators.
- **General Relativity**: metrics from line elements; the Christoffel symbols and the
  Riemann, Ricci, Einstein and Weyl tensors; covariant and Lie derivatives; geodesics,
  symbolic and numerical; eight known spacetimes; and tensor equations solved as tensor
  equations, before there are components.
- **Nothing guessed silently.** A reading that could have been another carries a note,
  and `explain` shows how each statement was read. What cannot be read or decided is
  refused, saying where, why and what to write instead. What has no value, as 1/0 or an
  integral that never settles, is refused too, and a closed form that numbers refute is
  not given.
- **Data, and Python**: indices over assets, days or the states of a qubit; data from
  NumPy, pandas, SciPy and SymPy; JAX for gradients; and Python objects by name,
  `\texttt{name}`.
- **Checking papers**: `relativisticpy check` runs a paper's equations and says which
  hold. Sean Carroll's lecture notes are the benchmark: 98.6% of their equations parse,
  and the check finds three mistakes in them.
- **For agents**: an MCP server, to compute, verify a step, check a paper and read the
  documentation. Each value says whether it is a complete answer, and whether it is
  approximate; each refusal says why.
- **Tools**: the command line, Markdown notebooks, Jupyter and marimo, and an editor
  that computes as it is typed. The documentation is offline, in `relativisticpy help`.
- **A pipeline with one job for each pass**, from the lexer to the engines, and
  **integrations to one pattern**: every function a capability that a module declares,
  held to the pattern by a conformance suite, with no configuration.
- **Known answers**: a curriculum of 734 problems in 32 fields, and a notation matrix of
  44 constructs of the core, each written every way books write it. Each answer is
  computed without RelativisticPy, and the tests keep what passes passing.

## Now

- **No silent wrong answer.** Version 0.1.0 waits on the wrong answers known in the
  core, algebra, calculus, linear algebra and index notation: each is to be fixed, or
  refused with what to write instead. Those of other fields are fixed too, or listed in
  the notes of the release. Those known as this is written:
  - a transpose written as Artin writes it, `A^t`, is a power by t, and of a 3 × 3
    matrix a crash; Lang's \({}^tA\) and Griffiths' \(\tilde{A}\) are names;
  - a row times a column is a 1 × 1 matrix written `A \cdot B`, or `AB` in LaTeX,
    and a number written `A B`: one product, two answers;
  - an accent on a defined name makes a new name: after `z := 3 + 4i`, `\bar{z}` is
    `zbar`; and a plain name stands for a bold one: after `\mathbf{w} := (1, 2, 3)`
    and `w := 7`, `\mathbf{w}` is 7;
  - a partial derivative by the number of an argument, `f_1` or `D_1 f`, is a name or a
    product;
  - a name from a paper is read as a product, without a note, as \(\Phi_N(r)\) is;
    and a check of a paper says that a relation between quantities nothing defines
    differs, where it relates them;
  - brackets after a defined number multiply: after `\delta := 0.1`, `\delta[n]` is
    0.1 n;
  - a coordinate along a path, \(q^i := (\theta(t), \phi(t))\), which the metric in a
    Lagrangian does not follow.

  `uv run python benchmarks/curriculum.py --failures` and
  `uv run python benchmarks/notations.py --failures` list every wrong answer they find.
- **The forms agents write.** Refused today, each with a reason: `\|x\|_p` of a vector,
  `eig(A)`, `dot(u, v)`, `cross(u, v)`, `A.T` and `A @ B`.
- **Climbing the curriculum.** What it refuses most, by cause, is in its own
  [roadmap](https://github.com/cottrellashley/relativisticpy/blob/main/curriculum/ROADMAP.md).

## Next

- **A core with no SymPy in it**: an IR of plain terms, one elaborator that ranks the
  readings of a form, and the notation of each field declared as data by its
  integration; then the core rebuilt in Rust.
- **Results as terms, and other engines**: SymPy's answers translated back into the IR,
  so that other engines can compute too, and two engines check each other.
- **Derivatives in index notation**: \(\nabla_\mu\) and \(\partial_\mu\) as part of the
  notation, with the Leibniz rule, metric compatibility and the contracted Bianchi
  identity \(\nabla^\mu G_{\mu\nu} = 0\), so that conservation laws can be derived
  rather than only checked.
- **Abstract definitions**: `T_{\mu\nu} := (\rho + p) u_\mu u_\nu + p g_{\mu\nu}`
  with a velocity that has no components yet, kept as notation until it has.
- **More of physics' implicit conventions**: \(x^\mu\) as the coordinates themselves,
  coordinates along a curve, \(x^\mu(\lambda)\), and \(\eta_{\mu\nu}\) as the
  Minkowski metric, so that more of a paper can be checked.
- **Premises in papers**: a condition a derivation states, as \(R_{\mu\nu} = 0\), so
  that what follows from it is not read as a mistake.
- **Units**: quantities with units, converted and checked for dimensions.
- **Optimisation over a vector written with its index**: minimise
  \(w^a \Sigma_{ab} w^b\) subject to constraints, with gradients from JAX.
- **Coordinate transformations**, with the Jacobian; **energy-momentum tensors** for
  perfect fluids and the electromagnetic field; and **more spacetimes**:
  Oppenheimer–Snyder collapse, Lemaître–Tolman–Bondi dust, Schwarzschild–de Sitter.
- **A visual engine, with [Manim](https://www.manim.community)**: derivations that
  animate step by step, orbits and light rays, light cones, embedding diagrams and
  Penrose diagrams, behind an optional `relativisticpy[visual]` extra.
- **Fields on grids**, for numerical relativity and for pictures, and the ADM
  formalism.
- **A language server**, for highlighting, errors and "compute this" in editors; and
  **pages that compute with no server**, in the reader's browser.
- **An evaluation**: how a model does with the `verify` tool and without it.

## Someday

Hawking radiation and the Penrose process as worked modules; analysis of real
data from gravitational-wave observatories; interactive tutorials for people new
to the subject.

Ideas and contributions are welcome:
[open an issue](https://github.com/cottrellashley/relativisticpy/issues).
