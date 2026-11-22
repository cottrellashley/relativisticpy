# The pipeline: one job for each pass

RelativisticPy reads a program in stages. The front end is cleanly separated already.
After it, one compiler of 6,090 lines resolves names, decides what ambiguous notation
means, checks index notation, plans computations, does index-notation algebra and emits
the IR, all in one recursive pass. Its IR carries Python functions at 54 places, and it
computes values while compiling at 7. This design separates those jobs. Engines then run
an IR they can read, and each stage can be inspected and tested on its own.

## The passes

| Pass | Takes | Gives | Its one job |
|---|---|---|---|
| lexer | text | tokens | characters into tokens |
| parser | tokens | syntax tree | the grammar of the language |
| reader | syntax tree, session, Python scope | readings | what each name and each ambiguous form means: a product or a call, an index or a power, a definition or a claim; with a note wherever the less likely reading was taken |
| typer | readings | typed readings | index notation and spaces: free and summed indices, positions, shapes; the only place their rules live |
| lowerer | typed readings | IR | a closed IR of named operations; no Python inside it, and nothing computed |
| rewriter | IR | IR | faster forms of what was written: a solve for an inverse times a vector, an FFT for a Fourier sum |
| engines | IR | values | each named operation computed by an implementation chosen for its values: exact with SymPy, numerical with NumPy or the array API, differentiable with JAX |
| display | values | text, LaTeX, data | how every result looks |

The `Workbook` orchestrates and holds the session; it decides nothing itself. What passes
between stages can be inspected:

- `wb.tokens`, `wb.parse` and `wb.ir`, as now;
- `wb.explain(source)`, which shows the readings: how each statement was read, and why.

## The closed IR

An operation of the IR is a name, its arguments and its attributes, such as
`call("besselj", [ν, x])`, `contract(...)`, `notation.solve(...)` or `fft(...)`. Every
name refers to a core operation or to a capability of an integration
(`design/integrations.md`). No operation holds a Python function. So an engine can read
the whole program before computing it, JAX can trace it, a rewrite can recognise a
pattern in it, and the IR can be printed, compared and saved.

Nothing is computed while compiling. What the compiler needs to know comes from a
declaration instead:

- a spacetime declares its coordinates;
- a space written `\mathbb{R}^{3}` declares its dimension;
- a count that depends on data is checked by the engine.

Index-notation algebra (solve, substitute, trace and simplify on notation) becomes
operations that the engines run, like any other.

## Steps

Each step keeps everything as it is: the test suite, the curriculum's record of passing
cases, and the verdicts on Carroll's notes, statement by statement.

1. **Name every operation.** Replace the Python functions in `Call` and `Map` with names
   in a registry of implementations, at first a core registry beside the kernels.
2. **Compute nothing while compiling.** Spacetimes declare their coordinates. Dimensions
   come from what is written. Counts are checked by the engine.
3. **Run notation algebra in the engines,** as operations of the IR.
4. **Split the compiler** into `reading/`, `typing/` and `lowering/`. These are moves of
   code, each with the tests that already pin its behaviour.
5. **One place for the rules of index notation.** The checker's "ill-formed" asks the
   typer, instead of analysing the syntax again. *Done:* the rule of a term is
   `typing.written.contraction`, which the typer keeps as notation is compiled, and a
   check asks `typing.written.broken_rule` whether a statement surely breaks the rules
   as written. The verdict keeps its own words, which the tests and the reports pin.
6. **Bind data in a module of its own,** with its spaces, so that the workbook only
   orchestrates. *Done:* `compiler.binding` binds data, and the data of Python objects,
   into the session, with the space of each axis, and declares spaces; the lowering
   imports it as any module, where it imported the workbook's private helpers inside its
   functions to get round a cycle.
7. **Turn the registry into integrations,** discovered as the design of integrations
   describes, with the conformance suite.

## Found while building the steps

What the work on live mathematics needed from the core, which it did without, for now:

- **One way to list a program's statements and steps without running it.** The command
  line, the MCP server and the editor each do it with private helpers of the workbook
  and the command line (`_in_order`, `_pieces`, `_record`, `_snapshot`, `_defined`). It
  belongs with step 6, in a module of its own that all three use.
- **`Session.fork()` should give a copy its own functions.** `Workbook.copy()` rebinds
  the closures of user functions to the copy's names itself.
- **Keys that do not depend on the process.** The IR numbers its constants and
  `abstract` its dummy indices from process-wide counters, so a worker that receives a
  pickled workbook must number its own from far away (2^62) to avoid collisions.
- **A public way to fork an executor,** or to read the values it holds, which
  `Workbook._portable` reads privately today.
- **A reference that stays one reading as integrations arrive.** Every capability was a
  row of the generated reference, which passed 40,000 characters on 22 November 2026.
  Done: the page is an index, a line for each family of capabilities with their names
  and forms, and `relativisticpy help <name>` describes each one in full, from its
  declaration (`reference.capability`): its forms, its arguments, its examples and what
  computes it. A new capability adds its names to a line of the page, not a row.

## Where integrations plug in

- **Written forms:** read by the reader, printed by display.
- **Signatures:** read by the typer.
- **Implementations:** run by the engines.
- **Rewrites:** applied by the rewriter.
- **Object conversions:** what the reader makes of a Python object named in
  `\texttt{...}`.

An integration touches no pass's code, only the registry each pass consults.
