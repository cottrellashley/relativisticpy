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
- **Fields in their books' notation**, [each with a page](language.md#fields): special
  functions, vectors and matrices, probability, signals, optimisation, number theory and
  combinatorics, and quantum mechanics, with Dirac's notation and operators.
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
  the notes of the release. Those known as this is written, none of which says so in a
  note:
  - a plain name stands for a bold one: after `\mathbf{w} := (1, 2, 3)` and `w := 7`,
    `\mathbf{w}` is 7 (decision 9 of `design/notation.md`);
  - beside a path, \(q^i := (\theta(t), \phi(t))\), a sum of scalars keeps the
    coordinate: the value of `L - m g_0 R \cos\theta` holds cos θ beside θ(t), and its
    note says only that θ is taken along the path in L; its derivatives by the path are
    right;
  - a claim after equations that hold its names is not decided under them: Mandelstam's
    \(s + t + u = \sum_i m_i^2\), after \(p_1 + p_2 = p_3 + p_4\) and
    \(p_i^2 = m_i^2\), is shown as an equation;
  - `C(n, k)`, as some books write the binomial coefficient, is an unknown function of
    the names n and k;
  - `j`, which nothing defines, is a name in \(\left|\frac{1}{1 + j}\right|\), where an
    engineer means the imaginary unit: decision 24 gives a note only in an exponent;
  - `\operatorname{Var} X`, without brackets, is Var times X, even of a random variable;
  - a loop written in words, `\textbf{for } k = 1, \ldots, 5`, is read as a statement
    about k, so `\lambda^{(5)}`, after it, is a name;
  - units are names: \(\mathrm{Mpc}\,\mathrm{s}/(70\,\mathrm{Gyr}\,\mathrm{km})\) is
    not converted (`design/debt.md`).

  Three fail with an internal error, where they should give a value or a refusal:
  `\mathbb{E}[x]` of a list, after `x := [1, 2, 3, 6]`; an equation of index notation
  whose identity SymPy cannot tell, as Carroll's (1.69), described as data, in
  `--format json` and the MCP server's `compute`, and judged, in the check of the notes;
  and, in that check, (7.89), \(-\infty < u < +\infty\) after (7.88) defines u, where
  SymPy compares NaN.

  `uv run python benchmarks/curriculum.py --failures` and
  `uv run python benchmarks/notations.py --failures` list every wrong answer they find.
- **The forms agents write.** Refused today, each with a reason: `eig(A)` of a matrix,
  with a hint to write `spec(A)`; `A.T`; `A @ B`; and `||x||_1` typed in plain text, where
  `\|x\|_1` works.
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
  Minkowski metric, so that more of a paper can be checked; and a field met with a path
  taken along it in the IR, in every sum.
- **Premises in papers**: a premise a derivation states in prose, as
  \(R_{\mu\nu} = 0\), and an identity that holds at a point, as in Riemann normal
  coordinates, so that neither reads as a mistake. An equation for the functions a
  metric is written with relates already.
- **Equations a program means to solve**: a design for which equations to solve unasked,
  and which only state; and a claim decided under the equations stated before it.
- **What an agent reads**: a note that carries its kind, which a value's `reasons` are
  read from, and `compute`'s claims judged as `verify` judges them.
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
