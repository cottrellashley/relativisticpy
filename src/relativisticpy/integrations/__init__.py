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
- :func:`constructor`: what builds an object the language works with, as a spacetime.

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
    find,
    installed,
    on_change,
    reservations,
)

__all__ = [
    "CHECKS",
    "GROUP",
    "Capability",
    "Check",
    "Constructor",
    "Exactly",
    "Example",
    "Function",
    "MapFunction",
    "Numerically",
    "Reservation",
    "Solver",
    "applies",
    "capabilities",
    "check",
    "constructor",
    "find",
    "function",
    "installed",
    "map_function",
    "on_change",
    "reservations",
    "reserved",
    "solver",
    "usable",
]
