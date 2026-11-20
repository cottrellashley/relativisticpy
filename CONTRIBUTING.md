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

`uv run pytest -m "not slow"` skips the few tests that take more than a few
seconds. pytest also runs the examples in docstrings, `README.md`, `docs/` and
`examples/`, so documentation cannot drift from the code.

## Where things go

[How it works](https://relativisticpy.org/architecture/) describes the pipeline,
from text to tokens, syntax tree, IR and results, and where to add a function, a
LaTeX command, a spacetime or an operation.

Every stage has its own tests under `tests/`. A change comes with tests for the
stage it touches and, for anything a user can write, an end-to-end test in
`tests/e2e` that states the notation and what it means.

## Conventions

- Results are exact, and tests compare them exactly. Prefer an identity that
  must hold, such as $\nabla_\mu G^{\mu\nu} = 0$, over a hard-coded expression.
- Errors are for physicists: say what is wrong in the language of the notation,
  point at the source, and suggest the fix when there is one.
- Code is typed (`mypy --strict`) and formatted with `ruff format`.
