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
  - The package is scanned, so adding the file is enough; no list needs editing. A
    module whose name starts with an underscore is the framework, not an integration.
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
| object | what a Python object, read by name, becomes | NumPy, JAX, CuPy and PyTorch arrays; pandas, polars and Arrow tables; xarray with its named dimensions; SciPy sparse matrices; SymPy expressions; Python functions |
| rewrite | a faster way to compute what is written | the Fourier transform written as its sum, computed as an FFT; an inverse times a vector, computed as a solve |

### Written forms

Every capability declares how it is written: in LaTeX (`latex=r"J_{\nu}(x)"`, and
`\operatorname{J}_{\nu}(x)` too if books write it so) and in Unicode
(`unicode="J_ν(x)"`). In plain text a capability is its name, called with its
parameters, `besselj(nu, x)`: the name is the plain text, so there is nothing to keep in
step. The same declaration does two jobs:

- **Reading.** Each form is written in the language and parsed once, by its own parser,
  into a pattern: the name the call is written with and how it is set (upright, as
  `\operatorname{erfc}` or `\mathrm{B}`, calligraphic, a Greek letter, or plainly), its
  scripts before, below and above, and its arguments, each a variable named after a
  parameter of `exact`. The reader takes a call a program writes apart in the same way,
  whether the parser joined it into a name (`J_0(1)`), read it as a product
  (`J_{\nu}(x)`), or as a bracket with a script (`(a)_n`), and reads it as the capability
  whose pattern it has. The parser needs no change for most capabilities.
- **Printing.** The printer writes a call the way it was declared, so every result
  reads back.

The reader decides as `design/notation.md` says, and records each decision as a reading
that `Workbook.explain` shows:

- a user's definition always wins: a name given a value, or defined as a function;
- index notation wins: a script that is an index label of the statement is an index,
  and `\Gamma` with indices stays the Christoffel symbol, since no form has them;
- a name of one letter called with the coordinates, as `N(t)` in a metric, is a function
  of them;
- a name of one letter written alone, as `\Gamma(x)` or `\Phi(x)`, is read as the
  capability with a note, since a program could mean a function of its own.

A book name no integration provides yet is reserved where integrations live
(`integrations/book_names.py`), with what it is and its forms, and a program that writes
it is refused as not supported yet, saying what it is, never read as an unknown function
or a symbol. A capability with the reservation's name retires it.

## Large sets of numbers: Python objects

The language does not read files. Python does, in its own cells, with any library there
is, and the language reads the objects:

```latex
P_{ta} := \texttt{prices}                 % a pandas DataFrame: rows label t, columns a
x_t := \texttt{prices["NVDA"]}            % keys and attributes reach inside an object
w_n := \texttt{hann}(n, N), \quad n = 0, \ldots, N-1   % a Python function, called
```

### Scope

`\texttt{name}` is the object called `name` in the Python scope a workbook is given:
`Workbook(scope=globals())`, or a notebook's own namespace. In Marimo, the plugin passes
the cell's namespace and declares what the cell reads, so that it reruns when an object
changes.

### Reading only

A reference is a name followed by attributes and keys, never Python to evaluate. A
document checked with `relativisticpy check` has no scope unless one is given, so a paper
cannot reach into Python.

### Conversion by type

What an object becomes is decided by object integrations, one for each family of types,
following the same verified pattern as every other integration:

- a number is a scalar;
- an array or a table is data;
- a DataFrame's index and columns, and xarray's named dimensions, are the bases of
  spaces;
- a SymPy expression is an expression;
- a function is callable.

Standard protocols cover new libraries without code of their own: the array API, the
DataFrame interchange protocol, `__array__` and `_sympy_`. Data from two objects meets
by label, never by position.

What implementing it settled:

- **`\texttt{...}` is one name, as written.** The lexer reads `\texttt{prices["NVDA"]}`
  as one name, so that `\texttt{model.predict}(x)` is a call and `\texttt{c}^2` a power,
  as with any name; the compiler reads it as a Python object before anything else, so
  that it is never a name of the language's.
- **An object becomes one of four things:** a `Scalar`, `Data` with the labels and the
  name of each axis, an `Array`, as the language's lists are, or a `PythonFunction`. An
  object integration declares the classes it takes by name, never importing their
  library, or a protocol by its method; the most specific class decides, then a
  protocol, a function last. Its examples give the objects their programs read, and four
  checks hold it to the pattern: declaration, examples, errors and reference.
- **Data defined with indices is bound as `wb[...] = data` binds it,** by the same
  function, so that `P_{ta} := \texttt{prices}` and `wb["P_{ta}"] = prices` give the same
  spaces; labelled data meets the space whose elements its labels are, whatever its axis
  is named, for both. Data read inline has indices no one wrote, over spaces named after
  its axes, or after the object.
- **An object is read anew each time a program names it,** and copied, so that the
  workbook holds data, never the object. A SymPy expression's symbols are the program's,
  by name, so that `\frac{d}{dx}` differentiates with respect to the program's x.
- **What crosses to the editor's worker is data.** The objects a program names are read
  where the scope is, as each run starts, and carried as what they became. A function
  crosses by its module's name, where the module is installed; one defined in a notebook
  cannot, and the editor computes in the notebook's own process, saying why.

### Grids and samples

Grids and samples are written in the notation:

- `x_n := n\,\Delta x, \quad n = 0, \ldots, N-1` defines a vector: the range of an
  index declares its space.
- `\epsilon_t \sim \mathcal{N}(0, 1)` draws from a distribution, with a seed, so that a
  result can be reproduced.

They are forms of the language, lowered to operations of the IR that the engines compute
lazily.

### Transforms

`\operatorname{fft}(x_t)` names the FFT. The same transform written as its sum,
`\sum_{t=0}^{N-1} x_t\, e^{-2\pi i k t/N}`, is computed as an FFT too, through a rewrite
that the FFT's integration declares. Write the mathematics; the engine finds the fast
algorithm.

The cases in `curriculum/data-sources/` fix this notation, and fail until it is built.

## An example

```python
"""SciPy's special functions."""

import sympy

from relativisticpy.integrations import Example, function


@function(
    "besselj",
    latex=r"J_{\nu}(x)",
    summary=r"the Bessel function of the first kind, of order \nu",
    requires="scipy",
    examples=[
        Example("J_0(0)", 1, "DLMF 10.8.1"),
        Example(r"J_{1/2}(\pi/2)", 2 / sympy.pi, "DLMF 10.16.1"),
    ],
)
class BesselJ:
    def exact(self, nu, x):
        return sympy.besselj(nu, x)

    def numeric(self, xp, nu, x):  # xp: the array namespace of the data
        from scipy.special import jv

        return jv(nu, x)
```

What implementing it settled:

- **The signature of `exact` is the declaration of the parameters:** how many arguments
  a call takes, the names its written forms use, and the plain text it is called with,
  `besselj(nu, x)`. There is no table of arities.
- **An example names its source,** a book and equation or the library called directly,
  so that its answer is known from outside the integration.
- **Each kind says how the engines treat it:** a function lifts to tensors and data
  unless it takes its arguments whole, as `min` does (`lifts=False`), and gives a
  scalar unless it gives another value, as `latex` gives text (`scalar=False`).
- **The registry names what computes a capability by the capability's name,** so a
  worker process finds it again, and a missing library is an error that names the extra
  to install, found in the package's own metadata.
- **A function lifts over whichever argument is a tensor,** not only the first: the x of
  `J_0(x_t)` is data, its order a number. The IR's `Map` says where the component goes.
- **A form may write what is known when the program is read,** as the counts p and q of
  `{}_pF_q(a; b; z)`: `exact` takes them by keyword and checks them, and a capability
  whose calls print other than by its parameters in order says how, with an `arguments`
  method.
- **Exact gives the closed form where there is one:** a Bessel function of half an odd
  integer as the elementary function it is, a hypergeometric function through
  `hyperexpand`, a spherical harmonic of given degree written out.

## Verification

The conformance suite runs for every capability that the scan finds, each check for the
kinds it applies to. `relativisticpy.integrations.check` runs it on one capability, so an
agent checks a capability before it joins an integration.

1. **Declaration.**
   - Names are unique across every integration, and each is one name in the language.
   - Every written form parses to this capability.
   - There is a summary, at least two examples, each with its source, and the library it
     needs is declared and imported only when it is used.
2. **Examples.** Each runs in the language, calls the capability, and gives its known
   answer. The answer comes from outside the integration: the library called directly,
   or a closed form with its source.
3. **Agreement.**
   - The exact and numeric implementations agree at points: those the capability
     declares, or 0.3, 0.7 and 1.6 for each argument.
   - Numeric takes arrays of the points at once.
   - NumPy and JAX agree where both are installed.
4. **Lifting.** A function applied to a tensor, or to data, gives its results component
   by component, and keeps the spaces.
5. **Errors.** Each of these is a `RelativisticPyError` whose message says what to do:
   - the wrong number of arguments;
   - a dependency that is missing.

   The wrong kind of argument, a list given to a function or text in quotes to anything
   else, is computed or refused so, never a failure of RelativisticPy's own: what a
   function can take is its own to say, as `doit` of a list is the list.
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

Today's tables became the first integrations, with their behaviour unchanged:

- the builtins of `runtime/functions.py`: the elementary functions, algebra and calculus,
  equations, matrices and numbers, and the spacetimes, a module of
  `relativisticpy/integrations/` for each;
- the numerical dispatch of `tensors/numerical.py`: a function NumPy has itself declares
  `numeric`, and the engines translate the rest through SymPy.

The registry, the compiler and the reference now ask the integrations for a name. The
operations the compiler emits for notation (`core.*`, `linalg.*`, `stats.*`,
`notation.*`) stay in the registry: they are the core, not integrations.

## What stays in the core

Integrations plug into the core and never change it. The core is:

- the language: the lexer and parser;
- index notation and spaces;
- the IR and the runtime;
- display;
- errors.

A capability that needs new syntax, such as bra–ket notation, is a change to the
language first. After that, it is an integration like any other.
