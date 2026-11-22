"""The registry of implementations: what computes each operation the IR names.

An operation of the IR that calls a function, :class:`~relativisticpy.ir.ops.Call` or
:class:`~relativisticpy.ir.ops.Map`, holds a name and attributes, never a Python
function. The engines look the name up here. Its :class:`Implementation` says how many
arguments it takes, whether it applies to each component of a tensor, and what kind of
value it gives.

The names are of two kinds:

- a builtin is registered under the name the language calls it by: ``sin``, ``solve``,
  ``Schwarzschild``;
- what the compiler emits for notation is registered in a namespace: ``core.sum`` for
  :math:`\\sum`, ``core.subs`` for an evaluation bar, ``core.diff`` for a derivative,
  ``linalg.solve`` for an inverse times a vector, ``stats.ledoit_wolf``.

A new function is an entry here, with its implementation in
:mod:`relativisticpy.runtime.functions` or :mod:`relativisticpy.runtime.core`.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from relativisticpy.diffgeom import geodesic_equations
from relativisticpy.runtime import core, functions
from relativisticpy.tensors import linalg

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


# -- what the compiler emits for notation ---------------------------------------------------


def _core() -> Iterator[Implementation]:
    def entry(
        name: str,
        func: Callable[..., Any],
        result: Result,
        low: int = 0,
        high: int | None = None,
        *,
        componentwise: bool = False,
        takes_names: bool = False,
    ) -> Implementation:
        return Implementation(
            name, func, result, low, high, componentwise=componentwise, takes_names=takes_names
        )

    # collections
    yield entry("core.list", core.make_list, "value")
    yield entry("core.tuple", core.make_tuple, "value")
    yield entry("core.item", core.item, "value", 2, 2, takes_names=True)
    yield entry("core.part", core.item, "tensor", 2, 2, takes_names=True)
    yield entry("core.piecewise", core.piecewise, "scalar")
    # calculus, each applied to every component of a tensor
    yield entry("core.limits", core.limits, "value", 3, 3)
    yield entry("core.sum", core.summation, "argument", 4, 4, componentwise=True)
    yield entry("core.prod", core.product, "argument", 4, 4, componentwise=True)
    yield entry("core.limit", core.limit, "scalar", 4, 4, componentwise=True)
    yield entry("core.diff", core.differentiate, "argument", 2, componentwise=True)
    yield entry("core.derivative_at", core.derivative_at, "scalar", 4, 4)
    yield entry("core.gradient", core.gradient, "tensor", 2, 2)
    yield entry("core.subs", core.substitute, "argument", 1, componentwise=True)
    yield entry("core.time_derivative", core.time_derivative, "argument", 2, 2, componentwise=True)
    # functions, relations and logic
    yield entry("core.function", core.function, "scalar")
    yield entry("core.lt", core.less, "scalar", 2, 2)
    yield entry("core.gt", core.greater, "scalar", 2, 2)
    yield entry("core.le", core.less_equal, "scalar", 2, 2)
    yield entry("core.ge", core.greater_equal, "scalar", 2, 2)
    yield entry("core.eq", core.equal, "scalar", 2, 2)
    yield entry("core.ne", core.unequal, "scalar", 2, 2)
    yield entry("core.not", core.logical_not, "scalar", 1, 1)
    yield entry("core.and", core.logical_and, "scalar", 2)
    yield entry("core.or", core.logical_or, "scalar", 2)
    # the metric
    yield entry("core.where", core.where, "value", 3, 3)
    yield entry("core.where_function", core.where_function, "value", 3, 3)
    yield entry("core.line_element", core.line_element, "scalar", 2, 2)
    yield entry("core.components", core.components, "value", 1, 1)
    yield entry("core.geodesic_equations", geodesic_equations, "value", 3, 3)
    # values for Python
    yield entry("core.tensor", core.tensor, "value", 1, 1)
    yield entry("core.tensor_equation", core.tensor_equation, "value", 1, 1)
    # spaces
    yield entry("core.identity", core.identity, "tensor", 1, 1)
    yield entry("core.ones", core.ones, "tensor", 1, 1)
    yield entry("core.uniform", core.uniform, "tensor", 1, 1)
    yield entry("core.space_metric", core.space_metric, "tensor", 2, 2)
    yield entry("core.hadamard", core.elementwise_product, "value", 2, 2)
    yield entry("core.conjugate", core.conjugated, "argument", 2, 2)
    yield entry("core.squared_modulus", core.squared_modulus, "argument", 2, 2)
    yield entry("core.real", core.real, "scalar", 1, 1)
    # linear algebra and statistics
    yield entry("linalg.solve", linalg.solve, "tensor", 3, 3)
    yield entry("linalg.hadamard", linalg.hadamard, "argument", 2, 2)
    yield entry("linalg.diag", linalg.diag, "tensor", 1, 1)
    yield entry("linalg.eig", linalg.eig, "value", 4, 4)
    yield entry("linalg.svd", linalg.svd, "value", 2, 2)
    yield entry("linalg.det", linalg.det, "scalar", 2, 2)
    yield entry("linalg.trace", linalg.trace, "scalar", 2, 2)
    yield entry("linalg.function", core.matrix_function, "tensor", 3, 3)
    yield entry("stats.ledoit_wolf", core.ledoit_wolf, "tensor", 1, 1)


_ENTRIES: dict[str, Implementation] = {}
for _entry in (*_builtins(), *_core()):
    if _entry.name in _ENTRIES:  # pragma: no cover - a mistake in the tables above
        raise ValueError(f"two operations are registered as {_entry.name!r}")
    _ENTRIES[_entry.name] = _entry

#: Every operation, by its name: the builtins first, in the order the language lists
#: them, then what the compiler emits for notation.
REGISTRY = MappingProxyType(_ENTRIES)
