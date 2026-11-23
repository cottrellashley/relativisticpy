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
| `map_function` | a function of a matrix; given a map written with its indices, of its matrix of components | `det`, `rank`, `expm` |
| `solver` | equations in, solutions out | `solve`, `dsolve` |
| `constructor` | builds an object the language works with | `Schwarzschild`, `Eq`, `symbols` |
| `transform` | a function of a variable in, a function of another out; or data over an index in, data over a new space out | `LaplaceTransform`, `FourierTransform`, `fft` |
| `rewrite` | a faster computation of what a program writes, to the same numbers | the DFT written as its sum, computed by `fft` |
| `optimizer` | a problem of optimisation in, its optimum out, with how it was found and whether it is global | SymPy's stationary points and simplex method, SciPy's `linprog` and `minimize` |
| `distribution` | the law of a random variable, written after `\sim` | `\mathcal{N}(\mu, \sigma^2)`, `\operatorname{Poisson}(\lambda)`, `\chi^2_k` |
| `python_object` | what a Python object read by name, `\texttt{prices}`, becomes | NumPy arrays, pandas tables, SymPy expressions, functions |

The decorator takes, by name:

- `summary`: one phrase, with which `relativisticpy help NAME` begins its description;
- `examples`: two or more `Example(program, answer, source)`, each a program that uses the
  capability and the value of its last output, known from *outside* the integration: a
  closed form with its place in a book, `"DLMF 10.8.1"`, or the library called directly,
  `"scipy.special.jv(0, 1.5)"`;
- `requires`: the library it needs beyond SymPy, as pip installs it, if any;
- `aliases`: other names it is called by, as `arcsin` for `asin`;
- for a function, `points`: arguments it is checked at, where it has a value. Without
  them, each argument takes 0.3, 0.7 and 1.6. And `lifts=False` for a function that takes
  its arguments whole, as `min` does; and `integers`, the parameters of a function of
  integers, as `n` of Euler's totient: a form written with a letter, `\varphi(n)`, reads
  as it only where they are integers, and a call prints by its name where they are not;
- for a function of a matrix, `points` too, each matrix a list of its rows, as
  `points=[([[2, 1], [1, 3]],)]`, and `scalar=True` where its value is a number, as a
  trace's. It acts on a map written with its indices, `A^{a}{}_{b}`, unless `lifts=False`,
  where its value is over other spaces, as a Kronecker product's is.

The class has an `exact` method, which computes with SymPy, and may have a `numeric`
method, which computes with arrays: `xp` is the array module of the data, NumPy's or
JAX's. The parameters of `exact` are the declaration of the arguments: how many a call
takes, and their names in the reference. A library is imported inside the methods, never
at the top of the module, so that the language runs without it; when it is missing, the
user is told what to install.

A `transform` of functions declares the `variables` it maps by convention, a pair for each
convention the letters choose, as ℱ takes `("x", "s")` or `("t", "omega")`; the transform
that undoes it, its `inverse`; and, as `points`, functions it is checked on. Its class has
an `exact(self, f, t=None, s=None)` method, and an `integral` method, which writes it as
the integral or the sum it stands for, unevaluated: the language names the variables and
reads the transform as books write it, `\mathcal{L}\{f\}(s)`, from its forms. A
transform of data has instead a `space(self, over)` method, which gives the space
its result is over, of the samples' frequencies for `fft`, and `numeric(self, xp, x)`,
which computes it along the first axis. The transforms of functions are declared in
`integrations/transforms.py`, and the FFT, with its rewrite, in `integrations/fft.py`.

A `rewrite` declares, in its `latex` form, the sum it stands for, and the capability that
`computes` it faster; its class has a `matches(self, kernel, n, k, size)` method, which,
given the kernel of a sum over data in the positions of its two indices, gives the factor
by which the sum is that computation, or `None`. The DFT written as its sum,
\(\sum_{n=0}^{N-1} x_n e^{-2\pi i k n/N}\), is so computed by `fft`.

An `optimizer` finds the optimum of a `Problem`: the greatest or least value of an
objective, a SymPy expression of its variables, real or whole numbers, subject to
equalities, each 0 where it holds, and inequalities, each at least 0. Its class has a
`takes(self, problem)` method, which says whether it solves a problem, and an
`optimum(self, problem)` method, which gives a `Found`, the value, the point, how it was
found and whether it is proven global, or `None`, so that the next optimizer is asked. It
declares `problems` it is checked at, and `numerical=True` where its answers are numbers
found numerically. An optimum a program writes, `\min_{x} f(x)`, is never a call: the
optimizers are asked in turn as it runs, the exact ones first. SymPy's are in
`integrations/sympy_optima.py`, and SciPy's in `integrations/scipy_optimize.py`:

```python
>>> import sympy
>>> from relativisticpy import integrations
>>> from relativisticpy.integrations import Example, Found, Problem, optimizer
>>> x = sympy.Symbol("x", real=True)
>>> @optimizer(
...     "vertex",
...     summary="the vertex of a parabola, at -b/2a",
...     problems=[Problem("min", x**2 - 4*x + 7, (x,))],
...     examples=[
...         Example(r"\min_{x} \left(x^2 - 4x + 7\right)", 3, "standard: (x - 2)^2 + 3"),
...         Example(r"\arg\min_{x} \left(x^2 - 2x\right)", 1, "standard: (x - 1)^2 - 1"),
...     ],
... )
... class Vertex:
...     def takes(self, problem):
...         return problem.degree() == 2 and len(problem.variables) == 1
...
...     def optimum(self, problem):
...         (v,) = problem.variables
...         a, b = sympy.Poly(problem.objective, v).all_coeffs()[:2]
...         point = -b / (2 * a)
...         return Found(problem.at((point,)), (point,), "at -b/2a", proven=bool(a > 0))
>>> integrations.check(Vertex, ["feasibility", "optimality"])
{'feasibility': [], 'optimality': []}

```

A `distribution` is the law a random variable is declared with, `X \sim
\mathcal{N}(\mu, \sigma^2)` (see [the language](language.md#random-variables)). Its `exact`
gives SymPy's law of its parameters, and its `numeric` SciPy's frozen law; its class has
`mean` and `variance` methods, the closed forms of its moments, whose source it declares as
`moments`, and may have `mgf(t, ...)`, its moment-generating function, where SymPy's law has
none. Its forms are read after `\sim` alone, so that `\Gamma(k, \theta)` never takes the
place of a function written `\Gamma(a, x)`, and their variables are its parameters in order:
`\sigma^2` in `\mathcal{N}(\mu, \sigma^2)` says that the second argument is the square of
the second parameter. The distributions are in `integrations/sympy_stats.py`:

```python
>>> from relativisticpy.integrations import distribution
>>> @distribution(
...     "Laplace",
...     latex=r"\operatorname{Laplace}(\mu, b)",
...     summary="the Laplace distribution, of mean μ and scale b",
...     requires="scipy",
...     moments="Johnson, Kotz and Balakrishnan (1995), chapter 24",
...     examples=[
...         Example("X \\sim \\operatorname{Laplace}(0, 1)\n\\operatorname{Var}(X)", 2,
...                 "scipy.stats.laplace.var()"),
...         Example("X \\sim \\operatorname{Laplace}(0, 1)\n\\Pr(X > 1)", 0.18393972058572117,
...                 "scipy.stats.laplace.sf(1)"),
...     ],
...     points=[(0, 1), (1, 2)],
... )
... class Laplace:
...     def exact(self, mu, b):
...         from sympy.stats.crv_types import LaplaceDistribution
...
...         return LaplaceDistribution(mu, b)
...
...     def numeric(self, xp, mu, b):
...         from scipy.stats import laplace
...
...         return laplace(mu, b)
...
...     def mean(self, mu, b):
...         return mu
...
...     def variance(self, mu, b):
...         return 2 * b**2
>>> integrations.check(Laplace, ["examples", "agreement", "density", "moments"])
{'examples': [], 'agreement': [], 'density': [], 'moments': []}

```

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
have meant a function of the program's own. A function of a matrix written with one letter,
as the spectral radius `\rho(A)`, is read so only where its argument is written as a
matrix, so that `\rho(1 + 3w)` stays a density's; and a form may write the name of a
function as its command, as the `\max` of `\lambda_{\max}(A)`, which a call writes as it
is. The notation of matrices that is no call, as `A^T` and `e^{A t}`, is read in
`reading/matrices.py`, by its operands.

A name books write for a function no integration provides yet is *reserved* in
`integrations/book_names.py`, with what it is: a program that writes it is refused,
saying so, rather than reading it as an unknown function. A capability of the same name
retires the reservation.

## Adding one, step by step

1. **Find the gap.** `uv run python benchmarks/curriculum.py --failures` names what the
   language cannot do yet: an unknown function, a notation it does not read.
2. **Write the capability,** in the integration of its library, following a capability of
   the same kind: `bessel.py`, `polynomials.py` and `special_functions.py` are functions
   written as books write them, exact with SymPy and numeric with SciPy. Here is a complete
   one, the logistic function:

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

`tests/integrations` runs thirteen checks for every capability the scan finds, each where
it applies to its kind. A failure names the capability and says what to do:

1. **Declaration.** Its names are its own, and the language reads each as one name; it
   has a summary, at least two examples, each with its source, and declares the library
   it needs, which is imported only when it is used. Each written form reads as it, under
   a name that is no other capability's: `N(x)` could not be the normal distribution
   function, as `N` is numerical evaluation.
2. **Examples.** Each runs in the language, calls the capability, and gives its answer;
   an optimizer's is found by it.
3. **Agreement.** `exact` and `numeric` agree at each point, `numeric` takes arrays of the
   points at once, and JAX agrees with NumPy where it is installed; of a function of a
   matrix, entry by entry, at each matrix; of an optimizer, at each problem it declares,
   the optimizers of the other kind, exact or numeric, that take it find the same value,
   and none a better one than an optimum proven global; of a distribution, its density and
   distribution function, as the language computes them exactly, agree with SciPy's.
4. **Lifting.** A function applied to a tensor, or to data, gives its value at each
   component, over the same space, and applied to data past its points too: below 0, far
   beyond them, between whole numbers and off the real line, where it gives the value
   `exact` gives, complex where that is, or refuses where `exact` refuses, never nan or a
   real part alone. A function of a matrix, applied to a map, gives its value at the map's
   matrix, a number or a map over the same space.
5. **Errors.** The wrong number of arguments and a missing library are refused, saying
   what to do; the wrong kind of argument, a list to a function or text in quotes to
   anything else, never fails inside RelativisticPy.
6. **Display.** A call prints in text, Unicode, LaTeX and data, and each reads back as the
   same value.
7. **Reference.** The language reference indexes it, by its names and forms, and
   `relativisticpy help NAME` describes it in full, both from its declaration.
8. **Transform.** A transform of functions agrees with the integral or the sum it stands
   for at points, and its inverse gives back what it transformed; a transform of data
   agrees with its sum, its inverse undoes it, it keeps Parseval's relation, and its
   result is over a space of the size and the labels it declares.
9. **Rewrite.** Each example of a rewrite is computed by what it names, and gives the
   numbers it gives computed as written.
10. **Feasibility.** At each problem an optimizer declares, the point it reports keeps
    every constraint, and the value it reports is the objective there.
11. **Optimality.** At that point, the conditions of optimality hold: of a smooth problem,
    the KKT conditions, the gradient of the objective a combination of the gradients of the
    constraints that hold there with equality, with multipliers of the right signs; over
    finitely many points, no feasible point is better.
12. **Density.** At each of its points, a distribution's density integrates, or sums, to 1,
    by SymPy, or by SciPy where SymPy cannot in time.
13. **Moments.** There, its mean and variance, as SymPy and SciPy compute them, are those
    it declares.

A capability that breaks the pattern is told, for instance:

```text
brokenj's example brokenj(0) gives 0, not 2: fix exact, or the example's answer
brokenj: numeric and exact differ at (x=0.3): numeric gives 0.955336489126, exact
  0.0141169980066; fix numeric, or, if the point is outside where it is defined, declare
  points=[...] inside
brokenj: its module imports scipy when it is imported; import it inside exact or
  numeric, and declare requires="scipy", so that the language runs without it
brokenj: fresnels(x) prints in latex as S\left(x\right), which reads back as S(x), in
  which S is a function the language does not know; declare a latex form the language
  reads as brokenj, as latex="J_{\nu}(x)"
```

A form may write a parameter that is known when the program is read, as the counts p and
q of `{}_pF_q(a; b; z)`: `exact` takes it by keyword, `def exact(self, a, b, z, *, p=None,
q=None)`, and checks it. A capability whose calls print other than by its parameters in
order says how, with a method `arguments(self, call)` that gives the value of each, as
the hypergeometric function does in `special_functions.py`.

## Python objects

An object integration says what a family of Python objects becomes when a program reads
one by name, as `\texttt{prices}` (see [Beyond spacetime](spaces.md#python-objects-by-name)):
a `Scalar`, a number or an expression; `Data`, an array with the labels and the name of
each axis, where the object has them, by which data from two objects meets; an `Array`, a
vector or a matrix, as the language's lists are; or a `PythonFunction`, called on numbers
and elementwise on data. It declares the classes it `takes`, by their module and name,
which takes their subclasses too, or a protocol, by its method, as `"__array__"`. The
most specific class an object has decides, then a protocol, a function last. Nothing is
imported to tell: no object of a library's class exists without the library. Its
examples are given the objects their programs read:

```python
>>> import datetime
>>> from relativisticpy.integrations import Scalar, python_object
>>> @python_object(
...     "durations",
...     takes="datetime.timedelta",
...     summary="a duration: its seconds, a number",
...     examples=[
...         Example(r"\texttt{lap} / 60", 3, "three minutes",
...                 given=lambda: {"lap": datetime.timedelta(minutes=3)}),
...         Example(r"\texttt{day}", 86400, "a day",
...                 given=lambda: {"day": datetime.timedelta(days=1)}),
...     ],
... )
... class Durations:
...     def becomes(self, value):
...         return Scalar(value.total_seconds())
>>> integrations.check(Durations)
{'declaration': [], 'examples': [], 'errors': [], 'reference': []}

```

Its checks are four: the declaration; the examples, each run in a workbook given its
objects; that what its objects become, called, added and given to a function, is computed
or refused, never a failure inside RelativisticPy; and the reference, which lists what
each object becomes. `numpy_arrays.py`, `pandas_tables.py` and `python.py` are
integrations of this kind.

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

**Next:** [the language reference](reference.md), which lists every capability, each of
which `relativisticpy help NAME` describes in full.
