"""The registry of implementations: what computes each operation a program names.

Each :class:`Implementation` holds the function that computes an operation, with what
the compiler needs to know of it: how many arguments it takes, whether it applies to
each component of a tensor, and what kind of value it gives.

A builtin is registered under the name the language calls it by: ``sin``, ``solve``,
``Schwarzschild``. A new function is an entry here, with its implementation in
:mod:`relativisticpy.runtime.functions`.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from relativisticpy.runtime import functions

#: What an operation gives: ``"scalar"``, a scalar where its arguments are scalars (a
#: value otherwise, as ``sin`` of a list is); ``"value"``, any other value, as a list, an
#: equation or a matrix; ``"argument"``, a value of the type of its first argument, as a
#: derivative of each component of a tensor is; ``"tensor"``, a tensor, whose indices and
#: spaces the compiler works out.
Result = Literal["scalar", "value", "argument", "tensor"]


@dataclass(frozen=True)
class Implementation:
    """An operation the IR names, and the function that computes it.

    ``func`` is given the values of the operation's arguments, in order, then its
    attributes, by name. It takes from ``min_args`` to ``max_args`` arguments (``None``:
    any number). A ``componentwise`` operation given a tensor applies to every component,
    as ``simplify(R_{abcd})`` simplifies each, in parallel when the workbook has workers.
    ``result`` is the kind of value it gives (see :data:`Result`). A ``builtin`` is
    called by its name in the language, and ``parameters`` names its parameters in
    messages, as a spacetime's. Text an operation that ``takes_names`` is given is names,
    as for ``symbols('x y')``, which never runs it; any other refuses text written in
    quotes, which SymPy would run as Python.
    """

    name: str
    func: Callable[..., Any]
    result: Result = "value"
    min_args: int = 0
    max_args: int | None = None
    componentwise: bool = False
    builtin: bool = False
    takes_names: bool = False
    parameters: tuple[str, ...] = ()

    @property
    def scalar(self) -> bool:
        """Whether it gives a scalar for scalar arguments."""
        return self.result == "scalar"

    def check_arity(self, count: int) -> str | None:
        """An error message if ``count`` arguments are wrong for this function:
        ``Schwarzschild(M, G, c) takes at most 3 arguments, not 4``."""
        low, high = self.min_args, self.max_args
        if low <= count and (high is None or count <= high):
            return None
        if high == low:
            expected = f"{low}"
        elif high is None:
            expected = f"at least {low}"
        elif low == 0:
            expected = f"at most {high}"
        elif high == low + 1:
            expected = f"{low} or {high}"
        else:
            expected = f"{low} to {high}"
        plural = "" if expected in ("1", "at least 1", "at most 1") else "s"
        return f"{self.signature()} takes {expected} argument{plural}, not {count}"

    def signature(self) -> str:
        """How the function is called: ``Schwarzschild(M, G, c)``, or ``solve()``."""
        return f"{self.name}({', '.join(self.parameters)})"


def lookup(name: str) -> Implementation:
    """The implementation registered as ``name``."""
    try:
        return _ENTRIES[name]
    except KeyError:
        raise KeyError(f"no operation is registered as {name!r}") from None


# -- the builtins ----------------------------------------------------------------------


def _builtins() -> Iterator[Implementation]:
    """The functions a program calls by name, in the order the language lists them."""
    for name, func in functions.ELEMENTARY.items():
        yield Implementation(name, func, "scalar", 1, 1, componentwise=True, builtin=True)
    for name, func in functions.ALGEBRA.items():
        low, high = functions.ARITY.get(name, (1, 1))
        yield Implementation(name, func, "scalar", low, high, componentwise=True, builtin=True)
    for name, func in functions.VALUES.items():
        low, high = functions.ARITY.get(name, (1, None))
        result: Result = "scalar" if name in functions.SCALAR_VALUES else "value"
        yield Implementation(
            name, func, result, low, high, builtin=True, takes_names=name == "symbols"
        )
    for name, factory in functions.SPACETIMES.items():
        low, high = functions.arity(factory)
        parameters = tuple(inspect.signature(factory).parameters)
        spacetime = functions.Spacetime(factory)
        yield Implementation(
            name, spacetime, "value", low, high, builtin=True, parameters=parameters
        )


_ENTRIES: dict[str, Implementation] = {}
for _entry in _builtins():
    if _entry.name in _ENTRIES:  # pragma: no cover - a mistake in the tables above
        raise ValueError(f"two operations are registered as {_entry.name!r}")
    _ENTRIES[_entry.name] = _entry

#: Every operation, by its name: the builtins, in the order the language lists them.
REGISTRY = MappingProxyType(_ENTRIES)
