r"""Integrations: what the language can write and compute, a module for each library.

The language is only as expressive as what it computes with. An integration is a module
of this package, named after what it integrates, that declares *capabilities*: a class for
each, whose methods compute it, decorated with its kind. There is no configuration to
edit: adding the module is enough, as the package is scanned, and packages outside
RelativisticPy contribute through the entry-point group ``relativisticpy.integrations``.

The kinds are:

- :func:`function`: scalar in, scalar out, applied to each component of a tensor and to
  data, as ``sin`` and the Bessel functions are;
- :func:`map_function`: a function of a matrix, as ``det`` and ``inverse``;
- :func:`solver`: equations in, solutions out, as ``solve``;
- :func:`constructor`: what builds an object the language works with, as a spacetime;
- :func:`transform`: a function of a variable in, a function of another out, as the Laplace
  transform; or data over an index in, data over a new space out, as the FFT;
- :func:`rewrite`: a faster computation of what a program writes, as the discrete Fourier
  transform written as its sum, computed as an FFT;
- :func:`optimizer`: what finds the optimum of a problem, as SymPy's stationary points or
  SciPy's linear programming, which ``\min_x f(x)`` and ``\arg\max_w`` consult in turn;
- :func:`python_object`: what a Python object becomes when a program reads it by name, as
  ``\texttt{prices}`` reads a DataFrame as labelled data.

Each capability gives :class:`Example`\ s, programs whose answers are known from outside
the integration, and the conformance suite, ``tests/integrations``, holds every capability
to the pattern, saying in words what to fix. ``docs/integrations.md`` tells how to add one.

    >>> from relativisticpy import integrations
    >>> sine = integrations.find("sin")
    >>> sine.kind, sine.arity, sine.text()
    ('function', (1, 1), 'sin(x)')
    >>> integrations.check(sine, ["examples", "agreement"])
    {'examples': [], 'agreement': []}
"""

from relativisticpy.integrations._conformance import CHECKS, Check, applies, check, usable
from relativisticpy.integrations._declarations import (
    Capability,
    Constructor,
    Example,
    Function,
    MapFunction,
    Reservation,
    Solver,
    constructor,
    function,
    map_function,
    reserved,
    solver,
)
from relativisticpy.integrations._discovery import (
    GROUP,
    Exactly,
    Numerically,
    capabilities,
    conversions,
    find,
    installed,
    on_change,
    reservations,
)
from relativisticpy.integrations._objects import (
    Array,
    Data,
    ObjectConversion,
    PythonFunction,
    PythonScope,
    Scalar,
    python_object,
)
from relativisticpy.integrations._optimizers import Found, Optimizer, Problem, optimizer
from relativisticpy.integrations._transforms import Rewrite, Transform, rewrite, transform

__all__ = [
    "CHECKS",
    "GROUP",
    "Array",
    "Capability",
    "Check",
    "Constructor",
    "Data",
    "Exactly",
    "Example",
    "Found",
    "Function",
    "MapFunction",
    "Numerically",
    "ObjectConversion",
    "Optimizer",
    "Problem",
    "PythonFunction",
    "PythonScope",
    "Reservation",
    "Rewrite",
    "Scalar",
    "Solver",
    "Transform",
    "applies",
    "capabilities",
    "check",
    "constructor",
    "conversions",
    "find",
    "function",
    "installed",
    "map_function",
    "on_change",
    "optimizer",
    "python_object",
    "reservations",
    "reserved",
    "rewrite",
    "solver",
    "transform",
    "usable",
]
