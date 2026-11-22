# Integrations

> **For:** contributors and agents who add what the language can compute, from SymPy,
> SciPy or any other library. **You will learn:** what an integration is, how to write
> one step by step, and how the conformance suite checks it. **Before this:**
> [How it works](architecture.md).

Every function the language knows, from `sin` to `Schwarzschild`, is a *capability* that
an *integration* declares. An integration is a Python module in
`src/relativisticpy/integrations/`, named after what it integrates: `elementary.py`,
`algebra.py`, `spacetimes.py`. There is no configuration: no registry of strings and no
list to keep in sync. The package is scanned, so adding the module is enough, and the
conformance suite holds every capability to one pattern, saying in words what to fix.

## What a capability declares

A capability is a class whose methods compute it, decorated with its kind. The kind says
how the language treats it:

| Kind | What it is | Examples |
|---|---|---|
| `function` | scalar in, scalar out; applied to each component of a tensor, and to data | `sin`, `simplify`, `gamma` |
| `map_function` | a function of a matrix | `det`, `inverse`, `trace` |
| `solver` | equations in, solutions out | `solve`, `dsolve` |
| `constructor` | builds an object the language works with | `Schwarzschild`, `Eq`, `symbols` |

The decorator takes, by name:

- `summary`: one phrase, as the [reference](reference.md) lists it;
- `examples`: two or more `Example(program, answer, source)`, each a program that uses the
  capability and the value of its last output, known from *outside* the integration: a
  closed form with its place in a book, `"DLMF 10.8.1"`, or the library called directly,
  `"scipy.special.jv(0, 1.5)"`;
- `requires`: the library it needs beyond SymPy, as pip installs it, if any;
- `aliases`: other names it is called by, as `arcsin` for `asin`;
- for a function, `points`: arguments it is checked at, where it has a value. Without
  them, each argument takes 0.3, 0.7 and 1.6. And `lifts=False` for a function that takes
  its arguments whole, as `min` does.

The class has an `exact` method, which computes with SymPy, and may have a `numeric`
method, which computes with arrays: `xp` is the array module of the data, NumPy's or
JAX's. The parameters of `exact` are the declaration of the arguments: how many a call
takes, and their names in the reference. A library is imported inside the methods, never
at the top of the module, so that the language runs without it; when it is missing, the
user is told what to install.

## Written forms

A capability is called by its name in plain text, `gamma(x)`, and is written in books in
forms of its own, which it declares in the language itself: `latex=r"\Gamma(x)"` and
`unicode="Γ(x)"`. Each form is parsed once, by the language's own parser, into a pattern:
the name it is written with and how that name is set, its scripts, and its arguments,
each named after a parameter of `exact`. A program that writes the same shape is read as
the capability, and a result prints in the first form, so that it reads back:

```python
>>> from relativisticpy import Workbook, display
>>> wb = Workbook()
>>> wb.expr(r"\Gamma\left(\tfrac{1}{2}\right)")
sqrt(pi)
>>> display.latex(wb.expr(r"\Gamma(x)"))
'\\Gamma\\left(x\\right)'

```

The reader follows the notation decisions of the project (`design/notation.md`, 15): a
definition in the program always wins, a script that is an index label is an index, so
that `\Gamma^{\lambda}_{\mu\nu}` stays the Christoffel symbol, and a name of one letter
called with the coordinates, as the lapse `N(t)` in a metric, is a function of them.
`Workbook.explain` shows each reading, and a note says where a name of one letter could
have meant a function of the program's own.

A name books write for a function no integration provides yet is *reserved* in
`integrations/book_names.py`, with what it is: a program that writes it is refused,
saying so, rather than reading it as an unknown function. A capability of the same name
retires the reservation.

## Adding one, step by step

1. **Find the gap.** `uv run python benchmarks/curriculum.py --failures` names what the
   language cannot do yet: an unknown function, a notation it does not read.
2. **Write the capability,** in the integration of its library, following a capability of
   the same kind. Here is a complete one, the logistic function, exact with SymPy and
   numeric with SciPy:

    ```python
    >>> import sympy
    >>> from relativisticpy.integrations import Example, function
    >>> @function(
    ...     "expit",
    ...     summary="the logistic function, 1/(1 + e^{-x})",
    ...     requires="scipy",
    ...     examples=[
    ...         Example("expit(0)", sympy.Rational(1, 2), "standard"),
    ...         Example("expit(2)", 0.8807970779778823, "scipy.special.expit(2.0)"),
    ...     ],
    ... )
    ... class Logistic:
    ...     def exact(self, x):
    ...         return 1 / (1 + sympy.exp(-x))
    ...
    ...     def numeric(self, xp, x):
    ...         from scipy.special import expit
    ...
    ...         return expit(x)

    ```

3. **Check it.** `relativisticpy.integrations.check` runs the conformance suite on one
   capability, before it is part of an integration, and gives the failures of each check,
   none where it passes:

    ```python
    >>> from relativisticpy import integrations
    >>> integrations.check(Logistic)
    {'declaration': [], 'examples': [], 'agreement': [], 'lifting': [], 'errors': [],
     'display': [], 'reference': []}

    ```

    In the package, `uv run pytest tests/integrations -k expit` runs the same checks.
4. **Run the curriculum.** `uv run python benchmarks/curriculum.py --update` records the
   cases the capability makes pass in `curriculum/passing.txt`, and
   `uv run python -m relativisticpy.reference --write docs/reference.md` writes the
   reference again, from the declarations.

Nothing else is edited: the registry, the compiler and the reference read the
declarations.

## The conformance suite

`tests/integrations` runs seven checks for every capability the scan finds. A failure
names the capability and says what to do:

1. **Declaration.** Its names are its own, and the language reads each as one name; it
   has a summary, at least two examples, each with its source, and declares the library
   it needs, which is imported only when it is used.
2. **Examples.** Each runs in the language, calls the capability, and gives its answer.
3. **Agreement.** `exact` and `numeric` agree at each point, `numeric` takes arrays of the
   points at once, and JAX agrees with NumPy where it is installed.
4. **Lifting.** A function applied to a tensor, or to data, gives its value at each
   component, over the same space.
5. **Errors.** The wrong number of arguments and a missing library are refused, saying
   what to do; the wrong kind of argument, a list to a function or text in quotes to
   anything else, never fails inside RelativisticPy.
6. **Display.** A call prints in text, Unicode, LaTeX and data, and each reads back as the
   same value.
7. **Reference.** The language reference describes it, from its declaration.

A capability that breaks the pattern is told, for instance:

```text
brokenj's example brokenj(0) gives 1, not 2: fix exact, or the example's answer
brokenj: numeric and exact differ at (x=0.3): numeric gives 0.955336489126, exact
  0.977626246538; fix numeric, or, if the point is outside where it is defined, declare
  points=[...] inside
brokenj: its module imports scipy when it is imported; import it inside exact or
  numeric, and declare requires="scipy", so that the language runs without it
```

## Integrations outside RelativisticPy

A package of its own contributes through the entry-point group
`relativisticpy.integrations`, Python's standard mechanism. Each entry point names a
module of capabilities, a capability, or a list of them:

```toml
[project.entry-points."relativisticpy.integrations"]
mylibrary = "mypackage.relativisticpy_integration"
```

The capabilities are found when the language first needs them. An entry point that fails
to load is a warning that names it, never a language that does not start.

**Next:** [the language reference](reference.md), which lists every capability.
