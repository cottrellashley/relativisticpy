# Integrations: every computation library, by one pattern

The language is only as expressive as what it can compute with. SymPy, NumPy, SciPy,
scikit-learn and JAX are a start, and the libraries of science keep growing. This
design makes adding one routine work: as much code as it takes, but code that follows
one pattern and is verified automatically. An agent can write it from this page and
the tests' messages alone.

There is no configuration: no registry of strings, no YAML, no lists to keep in sync.
An integration is a Python module. Its flexibility comes from the strictness of the
pattern and from the tests that hold every integration to it.

## What an integration is

- **A module** in `relativisticpy/integrations/`, named after what it integrates:
  `sympy_functions.py`, `scipy_special.py`, `numpy_fft.py`, `sklearn_covariance.py`,
  `spacetimes.py`.
- **It declares capabilities:** what the language can now write and compute.
- **Discovery:**
  - The package is scanned, so adding the file is enough; no list needs editing.
  - Packages outside RelativisticPy contribute through the `relativisticpy.integrations`
    entry-point group, Python's standard mechanism.
- **Optional dependencies:** a library is imported when a capability is first used, and
  its absence is an error that names what to install.

Every capability is of one of a few kinds. Each kind is a protocol, with its own part
of the conformance suite.

| Kind | What it is | Examples |
|---|---|---|
| function | scalar in, scalar out; applied component by component to tensors and data automatically | Bessel functions, `\operatorname{erf}`, `\Gamma` |
| map function | acts on a map, a tensor with one index up and one down | `\exp(H)`, `\sqrt{\Sigma}`, inverses, `\operatorname{eig}` |
| reduction | binds an index, as a sum does | `\mathbb{E}_t`, `\operatorname{Var}_t`, `\operatorname{LedoitWolf}_t` |
| transform | acts on a function of a variable | Fourier, Laplace, Hilbert |
| solver | equations in, solutions out | `solve`, `dsolve`, ODE integration, roots, optimisation |
| constructor | builds an object the language works with | spacetimes, distributions, spaces |
| backend | an array library | NumPy, JAX, CuPy, PyTorch, Dask, through the Python array API standard |
| source | where large sets of numbers come from | CSV, Parquet, Excel, JSON, NumPy files, HDF5 and netCDF, SQL, the tables of a Markdown notebook, pandas and xarray objects |
| rewrite | a faster way to compute what is written | the Fourier transform written as its sum, computed as an FFT; an inverse times a vector, computed as a solve |

### Written forms

Every capability declares how it is written: in LaTeX (`J_{\nu}(x)`,
`\operatorname{J}_{\nu}(x)`), in plain text (`besselj(nu, x)`) and in Unicode. The
same declaration does two jobs:

- **Reading.** The parser reads the general forms (a command,
  `\operatorname{name}_{sub}^{sup}(...)`, `\mathcal{F}\{f\}(\omega)`), so most
  capabilities need no change to the parser.
- **Printing.** The printer writes a call the way it was declared, so every result
  reads back.

## Large sets of numbers

Data is written in the notation, not only bound from Python:

- **Tables.** A file named in `\texttt{...}` is a table, as LaTeX typesets a path.
  - `P_{ta} := \texttt{prices.csv}` is the whole table. Its row labels are the basis of
    `t`, and its column names that of `a`, as a pandas DataFrame bound from Python gives
    them.
  - `x_t := \texttt{signal.csv}[x]` is one column.
  - Data from two tables meets by label, never by position.
  - Each kind of file is a source integration. It is read lazily, and in chunks when
    large, and it is only ever read: nothing in a file runs.
  - Paths are relative to the document.
- **Grids.** `x_n := n\,\Delta x, \quad n = 0, \ldots, N-1` defines a vector, as
  mathematics does: the range of an index declares its space.
- **Samples.** `f_n := \sin(2\pi \cdot 5\, t_n)` samples a function on a grid, and
  `\epsilon_t \sim \mathcal{N}(0, 1)` draws from a distribution, with a seed so that a
  result can be reproduced.
- **Transforms.** `\operatorname{fft}(x_t)` names the FFT. The same transform written as
  its sum, `\sum_{t=0}^{N-1} x_t\, e^{-2\pi i k t/N}`, is computed as an FFT too, by a
  rewrite that the FFT's integration declares. Write the mathematics; the engine finds
  the fast algorithm.

Grids, samples and ranges are forms of the language, lowered to operations of the IR
that the engines compute lazily. Tables and transforms are integrations. The cases in
`curriculum/data-sources/` fix this notation, and fail until it is built.

## An example

```python
"""SciPy's special functions."""

import sympy

from relativisticpy.integrations import Example, function


@function(
    "besselj",
    written=[r"J_{\nu}(x)", r"\operatorname{J}_{\nu}(x)", "besselj(nu, x)"],
    summary="the Bessel function of the first kind, of order \\nu",
    requires="scipy",
    examples=[
        Example(r"J_0(0)", 1),
        Example(r"J_{1/2}(\pi/2)", 2 / sympy.pi),
    ],
)
class BesselJ:
    def exact(self, nu, x):
        return sympy.besselj(nu, x)

    def numeric(self, xp, nu, x):  # xp: the array namespace of the data
        from scipy.special import jv

        return jv(nu, x)
```

## Verification

The conformance suite runs for every capability that the scan finds:

1. **Declaration.**
   - Names are unique across every integration.
   - Every written form parses to this capability.
   - There is a summary, at least two examples, and the dependency is declared.
2. **Examples.** Each runs in the language and gives its known answer. The answer comes
   from outside the integration: the library called directly, or a closed form with its
   source.
3. **Agreement.**
   - The exact and numeric implementations agree at sampled points.
   - NumPy and JAX agree where both are installed.
4. **Lifting.** A function applied to a tensor, or to data, gives its results component
   by component, and keeps the spaces.
5. **Errors.** Each of these is a `RelativisticPyError` whose message says what to do:
   - the wrong number of arguments;
   - the wrong kind of argument;
   - a dependency that is missing.
6. **Display.** A call prints in all four forms, and each form reads back.
7. **Reference.** The generated reference describes the capability from its
   declaration.

An integration is done when the suite passes. Nothing else is edited.

## How an agent adds a library

1. **Find the gap.** `uv run python benchmarks/curriculum.py --failures` names what the
   language cannot do yet: an unknown function, a notation it does not read.
2. **Write `integrations/<library>.py`,** following an integration of the same kind.
3. **Run `uv run pytest tests/integrations -k <library>`.** The conformance suite says,
   in words, what is missing or wrong.
4. **Run the curriculum.** The cases the integration makes pass join
   `curriculum/passing.txt`.

The same steps can happen while someone is working. An agent connected through the MCP
server can meet a name it does not know, write its integration in a sandbox, and load
it once the suite passes. The pattern and the suite are what make that safe.

## Migration

Today's tables become the first integrations, with their behaviour unchanged:

- `compiler/builtins.py`: SymPy's functions, algebra and calculus, and the spacetimes;
- the statistics (`_STATISTICS`) and linear algebra (`_LINEAR_ALGEBRA`) of the compiler;
- the numerical dispatch in `runtime/kernels.py` and `tensors/numerical.py`.

The compiler and the runtime then ask the integrations for a name, instead of their own
tables.

## What stays in the core

Integrations plug into the core and never change it. The core is:

- the language: the lexer and parser;
- index notation and spaces;
- the IR and the runtime;
- display;
- errors.

A capability that needs new syntax, such as bra–ket notation, is a change to the
language first. After that, it is an integration like any other.
