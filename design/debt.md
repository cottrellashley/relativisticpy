# Technical debt

What was left on purpose before 0.1.0, why, and what would pay it. The rule that
decided: a change is made where it removes a silent wrong answer, or makes one construct
read the same in every notation. A change that would take weeks of work on the core, and
change no answer now, is left, and written down here.

Silent wrong answers are not debt: they are bugs, and the [roadmap](../docs/roadmap.md)
lists those known.

## The core

**One core, with no SymPy in it.** The passes are modules of their own
([the pipeline](pipeline.md)), but the reader and the lowerer are still classes stacked
on one another, whose readings are tried in turn and lowered through methods each class
overrides, and the IR carries SymPy objects. The plan: an IR of plain terms, one
elaborator that ranks the readings of a form, the notation of each field declared as
data by its integration, and a core that imports nothing of SymPy or NumPy, which could
then be rebuilt in Rust.

- *Why it was left:* it is weeks of work, and changes no answer by itself. Each
  misreading was fixed instead in the pass that made it, and `benchmarks/readings.py`
  shows every statement a change reads otherwise.
- *What would pay it:* a `core/` package, held by a test to import nothing of SymPy or
  NumPy, and the passes moved into it one at a time, each move reading every statement as
  before.

**Results as terms, and engines besides SymPy.** One engine, SymPy, computes. A closed
form of a definite integral or a series is checked against mpmath, and an antiderivative
is differentiated back (`runtime/crosscheck.py`); a closed form that numbers refute is
refused.

- *Why it was left:* results as terms need the core first.
- *What would pay it:* one translator for each engine, from terms to its expressions and
  from its answers back to terms. Two engines could then check each other, and a
  disagreement would never be shown as an answer.

**What the checks of closed forms do not cover.** Integrals and sums with parameters,
and multiple integrals, are not checked. An integral that oscillates, as
\(\sin x / x\) does, or has a kink the check cannot find, stays unchecked rather than
refused.

- *Why it was left:* a parameter must be sampled where the program's assumptions and
  SymPy's conditions allow; a sample outside them would refuse a right answer.
- *What would pay it:* sampling parameters within the bounds the program assumes.

**A field is taken along a path at run time.** Where a product meets a field of the
coordinates and quantities on a path, as \(g_{ij}\dot q^i \dot q^j\) does after
\(q^i := (\theta(t), \phi(t))\), the runtime takes the field to the path's point
(`runtime/paths.py`), with a note. A sum of scalars has no types, so the value of
`V := L - m g_0 R \cos\theta` keeps cos θ beside θ(t), without a note; its derivatives by
the path are right, as the gradient takes it along the path.

- *Why it was left:* the runtime knows the coordinates, from the operands' types, and the
  path, from their values, so the fix was made there, before a sum, a product, a
  contraction or a gradient whose operands have types.
- *What would pay it:* an operation `along(field, path)` in the IR, made by the lowering,
  which would reach a sum of scalars too.

## Equations

**An equation shown is not solved.** A name an equation holds stays a symbol after it,
as x does after \(x^2/(0.10 - x) = 1.8 \times 10^{-5}\), and a value that holds it later
says so in a note, with how to solve it (decision 3 of [notation](notation.md)). An
equation of a function written without its variable, as \(y'' + \lambda y = 0\), is not
solved either. Four cases of the curriculum get no answer this way, with the note;
Mandelstam's identity, a claim after such equations, is shown as an equation, without
one.

- *Why it was left:* which equations a program means to solve, and which it only states,
  needs a design of its own. An equation of one unknown with one root under the
  assumptions could be solved, but not safely in general.
- *What would pay it:* that design; and a claim decided under the equations stated before
  it.

## Checking papers

**A premise stated in prose reads as a mistake.** The check takes an equation of known
quantities for an identity. It now tells a condition from an identity where the
difference of the sides holds only functions the program introduced as unknowns, as
\(\beta(t, r)\) of a metric written with it: such an equation is one for those
functions, and relates. So five steps of Carroll's derivation of Birkhoff's theorem, as
\(\partial_0\beta = 0\), relate now. What is left: an equation that writes a quantity
computed from the metric, as \(R_{00}\), and holds only under a premise stated in prose,
as \(R_{\mu\nu} = 0\), which the first link of Carroll's (7.21) does; and an identity that
holds only at a point, as (3.77) does in Riemann normal coordinates. Three of the notes'
six statements that differ are such.

- *Why it was left:* telling them apart needs the premises of the derivation, which the
  prose states in words. Reading "curvature = 0" as a condition would also turn a wrong
  identity, written as \(R_{ab} - X_{ab} = 0\), into one that relates.
- *What would pay it:* a way to state a premise, or a verdict for an equation that
  follows from those before it; and a way to say that an equation holds at a point.

## Units

**Units are names.** \(\mathrm{Mpc}\) is a symbol, and
\(\mathrm{Mpc}\,\mathrm{s}/(70\,\mathrm{Gyr}\,\mathrm{km})\) is not converted, so the
curriculum's `astro.expansion.hubble-time-units` gets a wrong answer.

- *Why it was left:* a system of units is a field of its own.
- *What would pay it:* an integration that declares units and their conversions, and
  checks dimensions.

## Time limits

**Three timers.** `core._budget`, `checking._time_limit` and `mcp._alarm` each set the
same alarm signal. The first two ring again every half second until their block ends;
`mcp._alarm` rings once. `checking._time_limit` and `mcp._alarm` stop the timer as they
end, so a timer set around them stops too.

- *Why it was left:* one timer touches every time limit, and none of the three is known
  to fail now.
- *What would pay it:* one module of time limits: a stack of the limits in force, one
  handler, and an alarm that belongs to its block.

**A budget is a constant of the engine.** An integral has 60 seconds
(`core._INTEGRAL_BUDGET`), so a slower machine can keep as written what a faster one
computes.

- *Why it was left:* the time a computation is given belongs in the request, which is
  the core's work.
- *What would pay it:* a budget given with the request, and a value kept for want of time
  never judged a wrong answer.

## What an agent gets

- **A reason is found by its words.** Of a value's `reasons`, `unknown function` and
  `time budget` are recognised in the wording of the notes (`steps.py`). The workbook's
  note of a function nothing defines shares its phrase with `steps.py`,
  `steps.NOTHING_DEFINES`; the others are matched as they are written, so a note reworded
  drops its reason, and the tests fail. A note should carry its kind, given where it is
  made, in `reading/` and `runtime/`.
- **A difference has no `complete` or `approximate`.** `verify` and `check` give the
  difference of a claim's sides as `display.data` describes it, without the fields a
  value of `compute` has. The verdict on it is decided by numbers, and is right.
- **A value defined earlier carries no reason.** After `h := foo(x)`, `h + 1` is complete:
  the note that `foo` is an unknown function went with the definition. At a number, as
  `foo(2)`, the workbook says so again wherever the value is shown.
- **`compute` and `verify` can call one claim two things.** A relation of names nothing
  defines, as `x^2 = 4`, has `holds: false` in `compute`'s data, meaning no identity,
  where `verify` says it relates. An equation of index notation that is no identity, as
  `R_{ab} = 0` without a metric, agrees: `holds: null` in `compute`, relates in `verify`.
  `compute` could carry `verify`'s verdict for each claim.
- **A tool error is text.** An MCP call that cannot be done at all is `isError`, with its
  hint on a line of its own, not an `error` as data.
- **The editor marks no value incomplete.** It shows the notes, as before.

## Code

- **Private names reached from other modules.** `core._budget`, `core._OverBudget`,
  `assumptions._general` and `assumptions._assumed` are reached from the integrations of
  distributions and optimizers, as `tests/integrations/test_imports.py` lists in
  `STILL_PRIVATE`. Renaming them would have broken the work being merged. Public names,
  `core.budget` and `core.OverBudget`, would pay it, and empty that list.
- **Polynomials of decimals at very high degree are slow.** Their exact path goes
  through fractions. Evaluating them with mpmath would pay it.

## Not measured

- **Whether `verify` helps a model.** An evaluation of a model with and without the tool
  needs access to a model's API, and has not been run.
