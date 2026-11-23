r"""Quantities along a path.

A path in the coordinates is written with each coordinate a function of its parameter,
as ``q^i := (\theta(t), \phi(t))``. A field of the coordinates, as the metric, met with a
quantity on the path, is the field at the path's point: in ``g_{ij} \dot{q}^i \dot{q}^j``,
the metric's θ is θ(t). Tensors at two points neither add nor contract, so there is no
other reading. The same holds of a derivative by the path's coordinates or velocities:
:math:`\partial L / \partial q^i` of a Lagrangian written with the coordinates is taken
along the path.

Where the two meet, the coordinates of the field are replaced by the path's, and a note
says so.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import sympy
from sympy.core.function import AppliedUndef

from relativisticpy import display
from relativisticpy.assumptions import remark
from relativisticpy.ir import Node, ops
from relativisticpy.ir.types import TensorType

#: The calls that take a derivative by their second argument, which may be a path's.
_BY_A_PATH = frozenset({"core.gradient"})


def meets(node: Node) -> bool:
    """Whether ``node`` is where a field may meet a path: a sum, a product, a contraction,
    or a derivative by a vector."""
    if isinstance(node.op, ops.Add | ops.Mul | ops.Scale | ops.Contract):
        return True
    return isinstance(node.op, ops.Call) and node.op.name in _BY_A_PATH


def along(node: Node, operands: Sequence[Node], args: list[Any]) -> list[Any]:
    """``args`` with the coordinates of a field replaced by a path's, where ``node`` meets
    a quantity on a path in its coordinates with a field of them; ``args`` as they are
    otherwise."""
    coordinates = _coordinates([node, *operands])
    if not coordinates:
        return args
    path = _path(args, coordinates)
    if not path:
        return args
    held = {
        symbol: path[symbol.name]
        for value in args
        for expr in _expressions(value)
        for symbol in expr.free_symbols
        if isinstance(symbol, sympy.Symbol) and symbol.name in path
    }
    if not held:
        return args
    names = _listed([display.unicode(symbol) for symbol in held])
    points = _listed([display.unicode(point) for point in held.values()])
    verb = "is" if len(held) == 1 else "are"
    written = " ".join(node.span.text.split()) if node.span is not None else ""
    where = f", in {written}" if written and len(written) <= _SHOWN else ""
    remark(f"{names} {verb} taken along the path, as {points}{where}")
    return [_replaced(value, held) for value in args]


#: The longest statement a note quotes.
_SHOWN = 60


def _listed(items: Sequence[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _coordinates(nodes: Sequence[Node]) -> set[str]:
    """The names of the coordinates that the values of ``nodes`` are over."""
    return {
        name
        for node in nodes
        if isinstance(node.type, TensorType)
        for space in node.type.spaces
        if space.spacetime
        for name in space.basis or ()
    }


def _path(args: Sequence[Any], coordinates: set[str]) -> dict[str, sympy.Expr]:
    """The point of the path that ``args`` hold, by the coordinates' names: each coordinate
    written as a function of one parameter, the same for each, as θ(t) and φ(t). None
    where there is no path, or where two parameters make two."""
    point: dict[str, sympy.Expr] = {}
    for value in args:
        for expr in _expressions(value):
            for applied in expr.atoms(AppliedUndef):
                name = type(applied).__name__
                if name not in coordinates or len(applied.args) != 1:
                    continue
                (parameter,) = applied.args
                if not isinstance(parameter, sympy.Symbol) or parameter.name == name:
                    continue
                if point.setdefault(name, applied) != applied:
                    return {}  # θ(t) and θ(s): two paths
    parameters = {applied.args[0] for applied in point.values()}
    return point if len(parameters) == 1 else {}


def _expressions(value: Any) -> Iterator[sympy.Expr]:
    """The SymPy expressions a value holds: itself, or the components of an array."""
    if isinstance(value, sympy.NDimArray):
        for component in sympy.flatten(value.tolist()):
            if isinstance(component, sympy.Basic):
                yield component
    elif isinstance(value, sympy.Basic):
        yield value


def _holds(value: Any, symbols: dict[sympy.Symbol, sympy.Expr]) -> bool:
    return any(expr.has(*symbols) for expr in _expressions(value))


def _replaced(value: Any, rules: dict[sympy.Symbol, sympy.Expr]) -> Any:
    if isinstance(value, sympy.NDimArray):
        if not _holds(value, rules):
            return value
        return value.applyfunc(lambda component: sympy.sympify(component).subs(rules))
    if isinstance(value, sympy.Basic) and value.has(*rules):
        return value.subs(rules)
    return value
