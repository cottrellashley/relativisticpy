# Contributing

Bug reports, ideas and pull requests are all welcome.
[Open an issue](https://github.com/cottrellashley/relativisticpy/issues) to start
a conversation, or to report notation that should work and doesn't. An equation
copied from a paper that fails to run is a bug.

## Setting up

RelativisticPy uses [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/cottrellashley/relativisticpy
cd relativisticpy
uv sync
```

## Checks

Continuous integration runs these on every pull request:

```bash
uv run pytest                            # the tests, and every example in the docs
uv run ruff check                        # lint
uv run ruff format --check               # formatting
uv run mypy                              # types
uv run --group docs mkdocs build --strict
```

It runs the tests on Python 3.10 to 3.14, under one hash seed, `PYTHONHASHSEED=0`, as the
command line runs itself: use nothing newer than Python 3.10 has.

`uv run pytest -m "not slow"` skips the few tests that take more than a few
seconds. pytest also runs the examples in docstrings, `README.md`, `docs/` and
`examples/`, so documentation cannot drift from the code.

The editor of `relativisticpy.live` is also tested in a browser: a marimo notebook,
driven by Node's Playwright in Chromium. CI runs it; to run it yourself, install
Playwright for Node, with `npm install --global playwright` and `npx playwright install
chromium`, then:

```bash
uv run --group notebooks pytest tests/live tests/test_notebooks.py
```

Without marimo, Node or Playwright, those tests are skipped, and say why.

## Where things go

[How it works](https://relativisticpy.org/architecture/) describes the pipeline,
from text to tokens, syntax tree, IR and results, and where to add a function, a
LaTeX command, a spacetime or an operation.

Every stage has its own tests under `tests/`. A change comes with tests for the
stage it touches and, for anything a user can write, an end-to-end test in
`tests/e2e` that states the notation and what it means.

Where one thing is written in two ways, as `\frac{dy}{dx}` and `\frac{d}{dx} y`, the two
must agree. `tests/equivalences` holds such pairs, each with its value computed without
RelativisticPy. A pair that does not agree yet is marked to fail, with the finding it
comes from, and the fix of the finding removes the mark.

## Adding a library

Every function of the language is a capability that an integration declares: one module
of `src/relativisticpy/integrations/`, following one pattern. There is no configuration,
and nothing else to edit. [Integrations](https://relativisticpy.org/integrations/) shows
how to write one, and the conformance suite says, in words, what is missing or wrong:

```bash
uv run pytest tests/integrations -k NAME
```

## The curriculum

`curriculum/` holds what the language should read: problems written as their books write
them, each with an answer computed without RelativisticPy
([its README](curriculum/README.md)).

```bash
uv run python benchmarks/curriculum.py --failures   # what fails, and why
uv run python benchmarks/curriculum.py --update     # record what passes now
uv run python benchmarks/notations.py               # each construct, written every way
```

A problem that passes is recorded in `curriculum/passing.txt`, and the tests keep it
passing. A new problem is written as its book writes it, never as the language happens to
read it: one that fails says what to build next.

## Notation

Books write the same thing in more than one way, and the same way for more than one
thing. How the language reads each such form is a decision, recorded in
`design/notation.md` with its reasons. A change to what a form means starts there.

## Conventions

- Results are exact, and tests compare them exactly. Prefer an identity that
  must hold, such as $\nabla_\mu G^{\mu\nu} = 0$, over a hard-coded expression.
- Errors are for the people who write the mathematics: say what is wrong in the
  language of the notation, point at the source, and suggest the fix when there is one.
- Nothing is guessed silently: a reading that could have been another carries a note,
  or is refused, saying what to write instead.
- Code is typed (`mypy --strict`) and formatted with `ruff format`.
