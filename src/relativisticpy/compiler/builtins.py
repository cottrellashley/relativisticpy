"""The functions a program can call by name, and the constants it can name.

Every builtin is an entry of the registry of implementations
(:mod:`relativisticpy.runtime.registry`), under the name the language calls it by: the
compiler reads what it needs to know from the entry, how many arguments it takes and
whether it applies to each component of a tensor, and emits an operation that names it.
"""

from __future__ import annotations

from types import MappingProxyType

import sympy

from relativisticpy.runtime import functions
from relativisticpy.runtime.registry import REGISTRY, Implementation

#: A function callable from the language: its entry in the registry.
Builtin = Implementation

#: Every builtin function, by the name the language calls it.
BUILTINS = MappingProxyType({name: entry for name, entry in REGISTRY.items() if entry.builtin})

# The families of builtins, as the language reference lists them.
_ELEMENTARY = functions.ELEMENTARY
_ALGEBRA = functions.ALGEBRA
_VALUES = functions.VALUES
_SPACETIMES = functions.SPACETIMES

#: The names of the builtins that are spacetimes, such as ``Schwarzschild``.
SPACETIMES: tuple[str, ...] = tuple(_SPACETIMES)

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
