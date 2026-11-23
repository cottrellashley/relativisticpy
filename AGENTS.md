# For coding agents

RelativisticPy is a language that reads mathematics as it is written and computes it.
It is built to be extended by agents: a library is added as a module that follows one
pattern, and a suite says, in words, whether it does. This page is what an agent needs
before changing anything. [CONTRIBUTING.md](CONTRIBUTING.md) has the same for people.

## The checks

All of them pass before a change is done, and after every commit:

```bash
uv sync
uv run pytest                                # tests, and every example in the docs
uv run ruff check && uv run ruff format --check
uv run mypy
uv run --group docs mkdocs build --strict
uv run python benchmarks/curriculum.py       # no case that passed may fail
uv run python benchmarks/notations.py        # no cell of the notation matrix either
```

A change to the lexer, the parser or the reader also says which statements it reads
otherwise: `uv run python benchmarks/readings.py --write before.tsv` before it, `--write
after.tsv` after it, then `--compare before.tsv after.tsv`. Each statement listed should be
one the change meant to change.

Python 3.10 to 3.14 are supported, and CI runs them all, with `PYTHONHASHSEED=0`. Use
no syntax or library feature newer than 3.10.

## Where things are

A program goes through one pass after another, each with one job
([how it works](docs/architecture.md)):

| Pass | Where | Its job |
|---|---|---|
| lexer, parser | `src/relativisticpy/language/` | text into a syntax tree |
| reader | `reading/` | what each name and ambiguous form means, with a note where another reading was possible |
| typer | `typing/` | the rules of index notation and spaces |
| lowerer | `lowering/` | an IR of named operations; nothing is computed while compiling |
| engines | `runtime/`, `integrations/` | each operation computed, exactly or numerically |
| display | `display.py` | every result, as Unicode, text, LaTeX, HTML or data |

The designs are in `design/`. `design/notation.md` holds every decision on notation that
books write in more than one way: a new reading of ambiguous notation is a decision
recorded there, not a guess in the code.

## Adding a library

1. **Find the gap.** `uv run python benchmarks/curriculum.py --failures` lists what the
   language cannot do yet, and which problems wait on it.
2. **Write `src/relativisticpy/integrations/<library>.py`,** following an integration of
   the same kind. [Integrations](docs/integrations.md) has the pattern, step by step.
   Nothing else is edited: the package is scanned, and there is no configuration.
3. **Run `uv run pytest tests/integrations -k <name>`.** The conformance suite says what
   is missing or wrong, in words. Fix what it says until it passes.
4. **Run the curriculum.** `uv run python benchmarks/curriculum.py --update` records the
   problems the integration makes pass.

## Rules

- **Never guess silently.** A reading that could have been another carries a note, or is
  refused with what to write instead ([the interface](design/interface.md)). A wrong
  answer given without a warning is the worst bug there is.
- **Answers come from outside.** An example's answer is a closed form with its source, or
  the library called directly, never what the code under test returned.
- **Never fit a problem to the program.** A curriculum case is written as its book writes
  it. If the language cannot read it, the case fails, and that is its use.
- **What passed keeps passing.** No case in `curriculum/passing.txt` may fail, and a
  paper's verdicts change only for a stated reason (`benchmarks/corpus.py`).
- **Say it plainly.** Messages, docstrings and docs are short sentences in the user's
  notation: what is wrong, where, why, and what to write instead.
