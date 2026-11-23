# The command line

The `relativisticpy` command runs programs and Markdown notebooks, checks the
equations of papers, prints this documentation offline, and gives a prompt to
compute at. `python -m relativisticpy` is the same command. `relativisticpy mcp`
serves the same to AI agents, as tools: see [the MCP server](mcp.md).

```bash
relativisticpy schwarzschild.relpy             # run a program, each result as it comes
relativisticpy notes.md                        # run the maths of a Markdown notebook
relativisticpy notes.md --render -o notes.md   # typeset each block's results after it
relativisticpy -c '\int_0^1 x^2 \, dx'         # run code given on the command line
printf 'x := 2\nx^3\n' | relativisticpy       # run standard input
relativisticpy check paper.tex --strict        # check the equations of a paper
relativisticpy help                            # the documentation, offline
relativisticpy help besselj                    # one function of the language, in full
relativisticpy                                 # the prompt, in a terminal
```

The examples below run as they are shown, as the tests of this page:
`main(arguments)` is the command, given its arguments, and returns its exit status.

```python
>>> from relativisticpy.cli import main
>>> main(["-c", r"\int_0^1 x^2 \, dx"])
1/3
0

```

## Running a program

`relativisticpy FILE` runs a program, one statement at a time, and prints each
result as soon as it is computed. A program is a file of statements, `.relpy` by
convention; a Markdown notebook, `.md` or `.markdown`, runs its blocks of maths in
order (see below). `-c CODE` runs code given on the command line, and `-` or a pipe
runs standard input, named `<stdin>`. A LaTeX document (`.tex`, or any file with
`\documentclass`) is checked rather than run, with `relativisticpy check`.

Each result is labelled with the statement it came from: where it is, as editors and
compilers write it, then the statement as written. A definition says what it
defines:

```python
>>> main(["-c", r"""g_{\mu\nu} := Schwarzschild(M)
... \Gamma^{r}_{tt}
... R_{\mu\nu} = 0
... x := 2; x^3"""])
<command line>:1:1: g_{\mu\nu} := Schwarzschild(M)
  defines the metric g_{μν} and the coordinates t, r, θ, φ
<command line>:2:1: \Gamma^{r}_{tt}
  M (r − 2M)/r³
<command line>:3:1: R_{\mu\nu} = 0
  R_{μν} = 0: holds in all 10 components
<command line>:4:1: x := 2
  defines x
<command line>:4:9: x^3
  8
0

```

A program of one statement prints its result alone, as the first example does.

### Errors

Errors go to standard error, after everything printed before them. A run stops at
the first error, with every result before it already printed, and says what did not
run:

```text
$ relativisticpy plain.relpy
plain.relpy:1:1: x := 3
  defines x
plain.relpy:2:1: x^2
  9
plain.relpy:3:1: y := x + 1
  defines y
plain.relpy:4:1: y > 2
  True
EvaluationError: cannot differentiate with respect to a number
 --> plain.relpy:5:1
  |
5 | diff(sin(x), x)
  | ^^^^^^^^^^^^^^^
hint: differentiate with respect to a symbol: a name defined as a number, as by x := 3, is that number
stopped at the first error, at plain.relpy:5:1: the statement after it did not run; --keep-going (-k) runs it
```

Statements that cannot be read are known before anything runs: after a stop, each
of them is reported too. `--keep-going` (`-k`) runs every statement that can be read,
and reports every error, in the order of the program:

```text
$ relativisticpy errors.relpy -k
errors.relpy:1:1: a := 1
  defines a
CompileError: T is not a tensor
 --> errors.relpy:2:1
  |
2 | T_{ab} u^a
  | ^^^^^^
hint: define it first, for example T_{ab} := [...]
errors.relpy:3:1: b := 2
  defines b
ParseError: unknown LaTeX command \frca
 --> errors.relpy:4:1
  |
4 | \frca{1}{2}
  | ^^^^^
hint: did you mean \frac?
errors.relpy:5:1: a + b
  3
2 of 5 statements failed: errors.relpy:2:1, errors.relpy:4:1
```

Either way the exit status is 1 if a statement failed. A failure of RelativisticPy's
own is an `internal error`, which says so and invites a report; it never stops a
`--keep-going` run, nor the prompt. `RELATIVISTICPY_DEBUG=1` adds its Python
traceback.

Ctrl-C stops a run and says where: `interrupted after 3.1 s computing
kerr.relpy:4:1, R^{a}_{bcd}; the results above are complete`, with exit status 130.
SIGTERM, as a harness stops a command that takes too long, does the same with status
143, and stops the worker processes of `-j` at once.

### Forms of output

`--format` chooses how results are printed, as
[`relativisticpy.display`](python.md) shows them:

| Form | What it is |
|---|---|
| `unicode` | mathematics for a terminal, `48M²/r⁶`: the default |
| `text` | plain ASCII in the language's own syntax, which reads back as the same value: `48*M^2/r^6`; the default where the console cannot show Unicode |
| `latex` | LaTeX, `\frac{48 M^{2}}{r^{6}}`, labelled by LaTeX comments; `--latex` is short for it |
| `json` | JSON Lines: one object per statement |

```python
>>> main(["-c", r"y := 2; \frac{1}{y}", "--latex"])
% <command line>:1:1: y := 2
% defines y
% <command line>:1:9: \frac{1}{y}
\frac{1}{2}
0
>>> main(["-c", r"\theta^2", "--format", "text"])
\theta^2
0

```

With `--format json`, each statement is a line of JSON, written as soon as it is
computed. Its keys:

| Key | What it holds |
|---|---|
| `file`, `line`, `column`, `end_line`, `end_column` | where the statement is, from 1 |
| `statement` | the statement as written |
| `values` | each value it showed, as `relativisticpy.display.data` describes it: its `kind`, `text`, `unicode` and `latex` forms, and its structure; and whether it is `complete`, the `reasons` it is not, and whether it is `approximate` (below) |
| `defines` | what it defined: each with its `name`, `kind` and how it is `written` |
| `notes` | how it was read, where that could surprise, and what its computation did, as dropping the solutions an assumption rules out |
| `assuming` | what its values are given under that books assume and the program did not say, as `a > 0` |
| `error` | `null`, or the error as `RelativisticPyError.to_dict()` gives it: its `type`, `message`, place, `notes` and `hint` |

```python
>>> main(["-c", "x := 2; x^3", "--format", "json"])
{"file": "<command line>", "line": 1, "column": 1, "end_line": 1, "end_column": 7, "statement": "x := 2", "values": [], "defines": [{"name": "x", "kind": "value", "written": "x"}], "notes": [], "assuming": [], "error": null}
{"file": "<command line>", "line": 1, "column": 9, ..., "values": [{"kind": "number", "digits": 1, "value": 8, "text": "8", "unicode": "8", "latex": "8", "complete": true, "reasons": [], "approximate": false}], ...}
0

```

A value is `complete` unless the program kept part of it as written, as it could not
carry it out; `reasons` then says why, each in a word or two:

| Reason | What is kept as written |
|---|---|
| `integral`, `sum`, `product`, `limit` | one SymPy found no closed form of |
| `derivative` | a derivative SymPy could not take |
| `unknown function` | a call of a function RelativisticPy does not know, as `foo(x)` |
| `time budget` | besides one of the above: SymPy was stopped looking for its closed form, at the end of the time it is given |

The statement's `notes` say the same in words. `approximate` is `true` for a value with a
floating-point number in it, from a decimal written, as `0.1`, or a numerical method: an
answer to the digits shown, not exactly.

```python
>>> main(["-c", r"\int_0^1 \sin(\sin x) \, dx", "--format", "json"])
{..., "values": [{"kind": "expression", ..., "complete": false, "reasons": ["integral"], "approximate": false}], ..., "notes": ["an integral is left as written: SymPy could not compute it"], ...}
0

```

Errors are in the JSON, not repeated on standard error; what a person needs besides,
such as `stopped at the first error`, is on standard error.

### Standard input

Standard input that is not a terminal is read to its end as one program named
`<stdin>`, with no banner and no prompts; errors go to standard error, and a failure
exits with status 1. If nothing comes for two seconds, as when a harness leaves
standard input open, the command says on standard error that it is reading it, to
its end. `-i` starts the prompt instead, or after running a FILE or CODE, with what
it defined.

## Markdown notebooks

A Markdown notebook runs its blocks of maths, in order, as one program: `$$...$$`,
`\[...\]`, display environments such as `\begin{equation}`, and fenced blocks of
` ```math `; a fenced block of ` ```relativisticpy ` is a program, run as written.
Code, and maths in code, is not run. Errors point at the notebook's own lines.

`--render` prints the notebook with each block's results typeset after it, in LaTeX,
between two comments that a reader of the rendered page does not see:

```markdown
$$
a := 2 \\
a^2
$$

<!-- relativisticpy:result -->
$$
4
$$
<!-- /relativisticpy:result -->
```

- **Every block is rendered.** A statement that fails is typeset as its error, in a
  block of code where the result would be, and the blocks after it still run. The
  errors are reported on standard error too, and the exit status is 1.
- **Rendering again replaces the results**, rather than running them: rendering a
  rendered notebook gives the same document. A run, or a check, of a rendered
  notebook skips them.
- **`-o FILE`** writes the document to FILE, which may be the notebook itself: the
  file is written whole or not at all.

`--ir` prints the IR, the computation a program compiles to, without computing it.
The blocks of a notebook compile in order, so that each sees what those before it
define, and the IR of each is headed by where it is: `# notes.md:12`.

## The prompt

`relativisticpy` with no file, in a terminal, starts the prompt:

```text
RelativisticPy 0.1.0: type help for help, who for what is defined, exit or Ctrl-D to leave.
rp> g_{\mu\nu} := Schwarzschild(M)
defines the metric g_{μν} and the coordinates t, r, θ, φ
rp> R_{abcd} R^{abcd}
48M²/r⁶
```

Definitions persist from one input to the next, and no error ends the session.

| Key or command | What it does |
|---|---|
| Enter | runs the input, unless a bracket, `\left`, `\begin{...}`, `$$` or `\[` is still open, or the line ends with an operator, `=`, `:=` or a single `\`; then the input goes on, at a `...` prompt |
| pasting | lines pasted at once are one input, so a line element over two lines is one statement |
| Ctrl-C | clears the input, or stops a computation; every definition is kept |
| Ctrl-D | leaves, at an empty prompt; on an unfinished input, runs what there is |
| Tab | completes built-in functions, spacetimes, constants, LaTeX commands, and what you defined |
| Up, Down | the inputs before, kept from one session to the next |
| `help`, `help TOPIC` | what the language looks like and the topics; one topic, or one function in full |
| `who` | what is defined, by kind: coordinates, metric, functions, tensors, values, spaces |
| `reset` | forgets every definition, once you say yes |
| `exit`, `quit` | leaves |

Each input is named by its number in errors, `--> <input 3>:1:5`. Input that looks
like Python, as `wb = Workbook()` or a line pasted from a Python session, gets a hint
rather than a result: the prompt reads the language itself. The history is kept in
`$XDG_STATE_HOME/relativisticpy/history` (by default
`~/.local/state/relativisticpy/history`), or in the file `RELATIVISTICPY_HISTORY`
names; set it to nothing to keep none. Line editing, history and completion use
Python's `readline` where it is installed, and the prompt works without it.

## Checking papers

`relativisticpy check FILE...` runs the equations of LaTeX documents and Markdown
notebooks in order, as a reader would, and says what became of each, as
[Checking papers](checking.md) explains. It takes:

- **files:** a `.md` or `.markdown` file is a notebook, and anything else is LaTeX;
- **patterns,** such as `'chapters/*.tex'`, where the shell does not expand them;
- **folders,** searched for the LaTeX documents that no other includes (a chapter
  that `main.tex` reads with `\input` is checked through it) and the notebooks with
  maths in them;
- **`-`,** standard input, read as LaTeX.

```text
$ relativisticpy check paper.tex --strict
paper.tex: 3 equations, 3 statements, checked in 0.1 s
  ✓ holds               1   proven equal
  ✗ differs             1   of known quantities, whose sides are not equal: a mistake, or a condition
  = defines             1   definitions

paper.tex:6:18: ✗ differs  \Gamma^\theta_{r\theta} = r
    the left side minus the right is 1/r − r

Not listed: 1 holds, 1 defines. --all lists every statement.

failed by --strict: 1 differs (paper.tex:6:18)
```

What fails `--strict` is listed; the rest is counted, and `--all` lists it. Nothing
is cut short when the output is not a terminal; in a terminal, long lines are
wrapped to its width. Several documents are reported one after another, and a last
line totals them. A file that cannot be read is reported, the others are checked,
and the exit status is 2.

| Option | What it does |
|---|---|
| `--all` | lists every statement, with its reason, not only those that fail |
| `--lines RANGE` | checks only these lines of one document: `120:340`, `13`, `10:` or `:10`. What comes before them still runs, for its definitions |
| `--time-limit SECONDS` | the time each statement may take, or `inf` (default 30) |
| `--strict` | exits with status 1 if an equation differs or is ill-formed, a statement is not parsed or is over its time limit, or a document has no equations |
| `--fail-on VERDICTS` | exits with status 1 on these verdicts, separated by commas: `differs`, `relates`, `ill-formed`, `defines`, `computes`, `not-computed`, `not-understood`, `not-parsed`, and the conditions `time-limit` and `empty`. With `--strict`, either fails it |
| `--format unicode` | the report as above: the default |
| `--format text` | the same, with marks in ASCII: `+ holds`, `x differs`, `<> relates`, `-> computes` |
| `--format json` | each report as one line of JSON: `Report.to_dict()`, with the `policy` it was checked against and whether it `failed` it |
| `--format github` | the summary, then an annotation of GitHub Actions for each finding listed: an error for what differs, is ill-formed or is not parsed, a warning for what is not computed or not understood |
| `-j N` | processes for component-wise work, as for a run |

```text
$ relativisticpy check paper.tex --format github
paper.tex: 3 equations, 3 statements, checked in 0.1 s
  ✓ holds               1   proven equal
  ✗ differs             1   of known quantities, whose sides are not equal: a mistake, or a condition
  = defines             1   definitions
::error file=paper.tex,line=6,col=18,endLine=6,endColumn=45,title=RelativisticPy%3A differs::\Gamma^\theta_{r\theta} = r%0Athe left side minus the right is 1/r − r
```

On a terminal, a check that takes more than a second shows where it is on standard
error: `⠼ checking paper.tex, line 9102 of 12011 · 1 min 12 s`.

## Reading the documentation

`relativisticpy help` prints what the language looks like and the topics of this
documentation; `relativisticpy help TOPIC` prints one, as Markdown, offline. A topic
may be asked for by another name, as `check` for `checking`, and a topic that does
not exist is told what it may have meant. At the prompt, `help` and `help TOPIC` do
the same. `relativisticpy --help` and `relativisticpy check --help` list every
option, with examples.

[The language reference](reference.md) lists every function of the language by name,
a line for each family. `relativisticpy help NAME` describes one in full, generated from
what its integration declares: the forms it is written in, its arguments, what computes
it, with SymPy or a library on data and what installs that library, its examples with
their answers and where each answer comes from, and its notes. Another of its names
does the same, as `arcsin` for `asin`, and so does its name in another case, where only
one function has it. A page comes first: `relativisticpy help latex` is the page on
LaTeX, and `relativisticpy help 'latex()'`, with brackets, the function.

```text
$ relativisticpy help besselj
# besselj

The Bessel function of the first kind, of order `\nu`.

- **Written:** `besselj(nu, x)`, `J_{\nu}(x)`
- **Arguments:** 2, `nu` and `x`
- **Takes:** numbers and expressions
- **Computed by:** SymPy, exactly; on data, SciPy (`pip install "relativisticpy[numeric]"`)
- **Kind:** a function, declared by the integration `relativisticpy.integrations.bessel`: Bessel and Airy functions

## Examples

- `J_0(1)` gives `0.7651976865579666` (scipy.special.jv(0, 1.0))
- `J_{1/2}(\pi/2)` gives `2/π` (DLMF 10.16.1)

## Notes

- Given a tensor, it applies to each component, and given data, to each number.
```

A name that is neither a topic nor a function is told what it may have meant, among
both: `relativisticpy help besselk2` suggests `besselk`.

## Options of a run

| Option | What it does |
|---|---|
| `FILE` | a program, a Markdown notebook, or `-` for standard input |
| `-c CODE` | runs CODE: statements separated by line breaks or `;`. Given more than once, the pieces are lines of one program. What follows `-c` is code, even `-x` |
| `-k`, `--keep-going` | runs every statement, reporting every error, rather than stopping at the first |
| `-i`, `--interactive` | starts the prompt after FILE or CODE, or with standard input that is not a terminal |
| `--format FORM` | `unicode`, `text`, `latex` or `json` |
| `--latex` | `--format latex` |
| `--render` | prints a Markdown notebook with its results typeset |
| `-o FILE` | with `--render`, writes the document to FILE |
| `--ir` | prints the IR instead of computing |
| `-j N`, `--workers N` | processes that share component-wise work, such as simplifying the components of a tensor: a number, or `0` or `auto` for one per CPU (default 1) |
| `--color WHEN` | `auto` (the default), `always` or `never` |
| `-V`, `--version` | the versions of RelativisticPy, Python, SymPy and NumPy, for a report |
| `-h`, `--help` | every option, with examples |

Options that do not go together are refused, never ignored: a FILE and `-c`,
`--render` and `--ir`, `--render` of something that is not a notebook, `-o` without
`--render`, `--format json` at the prompt, `--keep-going` with nothing to run.
Options of a command may come before it: `relativisticpy -j 4 check paper.tex`. A
file named `check` or `help` is written `./check`.

## Exit status

| Status | Meaning |
|---|---|
| 0 | done, and everything succeeded; a check that passed its policy, or had none |
| 1 | a statement failed, or a check failed its policy (`--strict`, `--fail-on`) |
| 2 | the command line was wrong: an unknown option, options that do not go together, or a file that cannot be read (missing, a folder, not UTF-8, or not a program) |
| 130 | interrupted with Ctrl-C |
| 143 | stopped with SIGTERM |

A false equation in a run is a result, not a failure: to fail on it, check it.

## Environment

| Variable | What it does |
|---|---|
| `NO_COLOR` | set to anything, no colour; nor with `TERM=dumb` |
| `FORCE_COLOR` | set to anything but `0`, colour even where the output is not a terminal |
| `RELATIVISTICPY_DEBUG` | `1` shows the Python traceback of an internal error |
| `RELATIVISTICPY_HISTORY` | the file the prompt keeps its history in; empty for none |
| `XDG_STATE_HOME` | where the history is kept otherwise, as `relativisticpy/history` |
| `PYTHONHASHSEED` | the seed of Python's hashes, which order SymPy's sets of symbols. Where it is not set, the command starts itself again with `0` before it imports anything else, so that a program gives the same answer every time, and the processes it starts for its work, of `-j` and of `relativisticpy mcp`, inherit it; `RELATIVISTICPY_RESTARTED` marks the command started again. A value set, as `random`, is kept. On Windows, which cannot replace a process, only the processes it starts get `0`: set it yourself to reproduce a run there |
