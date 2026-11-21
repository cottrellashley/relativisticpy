# The interface of RelativisticPy

What a person or an agent sees when they use RelativisticPy, and the rules it follows.
Five audits of the user experience (21 November 2026) found that the computation was
right far more often than the presentation of it. This is the contract the interface
keeps from now on. The language itself is described in the [docs](../docs/); this page
is about how RelativisticPy talks back.

## Principles

1. **Show what was done.** Every result says which statement it came from, and is
   shown in the notation it was written in.
2. **Never guess silently.** Where a statement could mean two things, RelativisticPy
   reads it one way and says so, in a note, or refuses and says why.
3. **Never lose work.** One failing statement never hides the results before it, and
   no error ends an interactive session.
4. **Every error says what, where, why and how to fix it,** in the user's notation.
   A failure of RelativisticPy's own is called that, never blamed on the user.
5. **One answer to one question.** "Does it hold?" has one answer, `holds()`, on
   every kind of equation.
6. **Everything can be learnt from the tool itself,** offline: `relativisticpy help`.
7. **Everything a person can read, a program can read:** JSON output, stable error
   types, documented exit codes.

## How results look

`relativisticpy.display` prints every result, in four forms, used everywhere:

| Form | Used by | Example |
|---|---|---|
| `unicode` | terminals, the prompt, `repr` | `48M²/r⁶`, `Γ^t_{tr} = M/(r(r − 2M))` |
| `text` | ASCII consoles, `--format text` | `48*M^2/r^6`, `\Gamma^{t}_{tr} = M/(r*(r - 2*M))` |
| `latex` | notebooks, `--latex`, `--render`, papers | `\frac{48 M^{2}}{r^{6}}` |
| `data` | `--format json`, agents | `{"kind": "scalar", "text": ..., "latex": ...}` |

- The `text` and `latex` forms read back: parsing them gives the same value. Tests
  hold every printer to this.
- Notation is the user's: index labels as written (`\mu`, μ in `unicode`), never
  `mu` or `_{mu}_{nu}`; Euler's number is `e` and the imaginary unit `i`, so that
  `E + e` is never shown as `E + E`; `π`, `∞`, `∂`.
- Numbers are shown as written: `0.3`, not `0.300000000000000`. An integer too long to
  read is shortened, with its number of digits.
- A tensor is shown by its name and indices, then one self-contained line per
  non-zero component, labelled by coordinate: `Γ^t_{tr} = Γ^t_{rt} = M/(r(r − 2M))`.
  Components equal by a symmetry share a line. A result without a name is never
  given one, such as `T`, that means something else.
- Numerical data is summarised: its shape, its spaces and a preview, never a dump of
  every number. Notebooks get an HTML table.
- An equation of tensors says plainly whether it holds and, if not, where:
  `R_{\mu\nu} = g_{\mu\nu}: differs in 4 of 10 components, R_{tt} = 0 but g_{tt} = …`.

## What `=` means

Papers use `=` both to define and to claim, so RelativisticPy decides by what the left
side already means:

- `:=` always defines, or redefines.
- `=` defines only what has no meaning yet: a new name (`x = 2`), a new function
  (`f(x) = x^2`), a new tensor (`T_{\mu\nu} = \ldots`), a line element.
- Otherwise `=` is a claim, checked: a name that has a value (`K = 48M^2/r^5` after
  `K` was defined), a component (`g_{tt} = \ldots`), a built-in (`R = 0`), a constant
  (`\pi = 3.14159`). The result carries a note: *K already has a value, so `=`
  compares; to give K a new value, write `K := ...`*.
- A name in its own value (`R = 2/R^2`) is an equation, never a definition.
- An equation for a coordinate in force (`t = 0`) is a condition: read, not computed.
- A name that the metric in force is written with, defined afterwards ("where
  \(\Delta = r^2 - 2Mr + a^2\)"), takes that value in the metric from then on.

## Errors

```
IndexNotationError: the two sides have different free indices: _{\mu\nu} on the left, _{\mu} on the right
 --> paper.tex:12:5
   |
12 | R_{\mu\nu} = g_{\mu}
   |     ^^^^^^^^^^^^^^^^
note: a free index appears once in every term
hint: write the same free indices on both sides, as in R_{\mu\nu} = g_{\mu\nu}
```

- A headline that says what is wrong, in lower case without a full stop, in the
  user's notation. One rule has one phrasing.
- `note:` says why, or points at a related place, such as the definition an error
  comes from. `hint:` gives a fix, in the user's language.
- Every line of a span that covers several lines is shown.
- No SymPy, NumPy or Python text is shown as if it were RelativisticPy's: known
  failures are translated, and the rest are quoted in a note.
- A failure that is RelativisticPy's own is an `internal error`, with an invitation
  to report it, and never ends a session or a check.
- From Python, every error a user can cause is a `RelativisticPyError`, and notebooks
  show it without RelativisticPy's internal frames.

## Checking papers

| Verdict | Meaning | Fails `--strict` |
|---|---|---|
| ✓ holds | proven equal | no |
| ✗ differs | every quantity is known, and the sides are not equal: a mistake, or a condition | yes |
| ↔ relates | the sides involve quantities nothing defines, so this is a relation to solve, not an identity | no |
| ✗ ill-formed | the notation breaks the rules of index notation | yes |
| = defines | a definition | no |
| → computes | an expression, computed | no |
| ~ not computed | a relation such as `\sim`, a condition, or the time limit | yes, for a time limit |
| ? not understood | something RelativisticPy cannot do yet, with the reason | no |
| ! not parsed | not read, with the reason | yes |

- What fails is listed by default; whatever is not listed is counted, and the report
  says how to list it (`--all`).
- `--lines` runs what precedes the range, so that its definitions are known, and
  reports only the range.
- Findings are located as `paper.tex:12:5`, carry their hint, and are never cut short
  when the output is not a terminal.
- `--format json` gives the whole report as data, and `--format github` gives
  annotations for GitHub Actions.

## The command line

- `relativisticpy FILE` runs a program or a Markdown notebook, printing each result as
  soon as it is computed, labelled with its statement. It stops at the first error
  unless given `--keep-going`. `-` reads standard input.
- `relativisticpy` with no file starts the prompt in a terminal. From a pipe, standard
  input is read as a program, with no prompts.
- `relativisticpy check FILE` checks a paper.
- `relativisticpy help [TOPIC]` prints the documentation, offline.
- At the prompt: `help`, `exit`, Ctrl-C cancels the line or the computation and keeps
  every definition, and LaTeX stays open until its `\end{...}` closes.
- `--format unicode|text|latex|json` chooses the form; `--latex` is short for
  `--format latex`.

| Exit status | Meaning |
|---|---|
| 0 | done, and everything succeeded |
| 1 | a statement failed, or a check failed by `--strict` |
| 2 | the command line was wrong: an unknown option or a missing file |
| 130 | interrupted with Ctrl-C |

## Learning the language

- **A reference generated from the language's own tables:** every command,
  environment, function, constant, spacetime, statistic and verdict. A test keeps it
  complete.
- **The documentation in the installed package,** printed by `relativisticpy help`.
- **`llms.txt` and `llms-full.txt` on the docs site,** for agents.
- **A section in the README for agents:**
  - how to learn the language offline;
  - the JSON outputs;
  - the exit codes;
  - the pitfalls, such as raw strings for LaTeX in Python.
