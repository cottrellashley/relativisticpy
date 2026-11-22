"""The functions a program can call by name, and the constants it can name.

Every builtin is a capability an integration declares (:mod:`relativisticpy.integrations`),
registered as an entry of the registry of implementations
(:mod:`relativisticpy.runtime.registry`) under each name the language calls it by: the
compiler reads what it needs to know from the entry, how many arguments it takes and
whether it applies to each component of a tensor, and emits an operation that names it.
"""

from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType
from typing import Any

import sympy

from relativisticpy import integrations, spacetimes
from relativisticpy.runtime.registry import REGISTRY, Implementation

#: A function callable from the language: its entry in the registry.
Builtin = Implementation

_BUILTINS: dict[str, Builtin] = {}
_SPACETIMES: dict[str, Callable[..., Any]] = {}


def _tabulate() -> None:
    """The builtins and spacetimes, again whenever the integrations change."""
    _BUILTINS.clear()
    _BUILTINS.update((name, entry) for name, entry in REGISTRY.items() if entry.builtin)
    _SPACETIMES.clear()
    for capability in integrations.capabilities():
        if isinstance(capability, integrations.Constructor) and spacetimes.declares(
            capability.exact
        ):
            _SPACETIMES[capability.name] = capability.exact


_tabulate()
integrations.on_change(_tabulate)

#: Every builtin function, by the name the language calls it.
BUILTINS = MappingProxyType(_BUILTINS)

#: The names of the builtins that are spacetimes, such as ``Schwarzschild``.
SPACETIMES = _SPACETIMES.keys()

#: Names with a fixed value: the constants of mathematics.
CONSTANTS = MappingProxyType(
    {
        "pi": sympy.pi,
        "e": sympy.E,
        "i": sympy.I,
        "oo": sympy.oo,
        "infty": sympy.oo,
        "True": sympy.true,
        "False": sympy.false,
    }
)
