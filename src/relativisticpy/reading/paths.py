r"""A statement on a path, as the reader takes it (design/notation.md, decision 16).

A path in the coordinates is written with each coordinate a function of its parameter, as
``q^i := (\theta(t), \phi(t))``, and a quantity the program defines with it, as a
Lagrangian ``L := \frac{1}{2} m g_{ij} \dot{q}^i \dot{q}^j``, is on the path: its θ is
θ(t), which the engines take it as where a field meets the path. A statement that writes
such a quantity is on the path too, all of it: a coordinate it writes alone, as θ in
``L - m g R \cos\theta``, is the path's θ(t), and so is a coordinate in the value of a
name it writes that is no quantity on the path, as V's in ``L - V``. Each is said in a
note. A statement that writes nothing on the path keeps its coordinates.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import sympy

from relativisticpy.compiler.scope import Scope, TensorBinding, VariableBinding
from relativisticpy.ir import Node, dependencies, ops
from relativisticpy.language import syntax as ast
from relativisticpy.typing.syntax import _walk


@dataclass(frozen=True)
class Along:
    """The path a statement is on: its ``point``, the call of each coordinate's function
    at the parameter, by the coordinate's name, as θ(t) of θ; the ``parameter``; and the
    names the statement writes that are ``on`` it, as L."""

    point: dict[str, Node]
    parameter: sympy.Symbol
    on: tuple[str, ...]


def along(statement: ast.Node, scope: Scope, coordinates: Iterable[str]) -> Along | None:
    """The path ``statement`` is on, where a name it writes is bound to a quantity on one;
    ``None`` where none is, or where two parameters make two paths."""
    names = set(coordinates)
    if not names:
        return None
    point: dict[str, Node] = {}
    on: list[str] = []
    written = dict.fromkeys(node.id for node in _walk(statement) if isinstance(node, ast.Name))
    for name in written:
        binding = scope.lookup(name)
        if not isinstance(binding, VariableBinding | TensorBinding) or name in names:
            continue
        found = _point_of(binding.node, names)
        if found:
            on.append(name)
            for coordinate, call in found.items():
                point.setdefault(coordinate, call)
    parameters = {_parameter(call) for call in point.values()}
    if not point or len(parameters) != 1:
        return None
    (parameter,) = parameters
    return Along(point, parameter, tuple(on))


def held(node: Node, coordinates: Iterable[str]) -> list[str]:
    """The coordinates a value holds as symbols, as V of ``V := m g R \\cos\\theta`` holds
    θ, by what its constants are written with, in the order they are met."""
    names = set(coordinates)
    found: dict[str, None] = {}
    for each in dependencies([node]):
        value = each.op.value if isinstance(each.op, ops.Const) else None
        if isinstance(value, sympy.Basic):
            for symbol in sorted(value.free_symbols, key=str):
                if isinstance(symbol, sympy.Symbol) and symbol.name in names:
                    found.setdefault(symbol.name)
    return list(found)


def _point_of(node: Node, coordinates: set[str]) -> dict[str, Node]:
    """The coordinates' functions of one parameter a value is computed from, as θ(t) and
    φ(t) of a Lagrangian on the path, by the coordinate's name."""
    found: dict[str, Node] = {}
    for each in dependencies([node]):
        op = each.op
        if not (isinstance(op, ops.Call) and op.name == "core.function" and len(each.args) == 1):
            continue
        name = dict(op.attributes).get("name")
        if not isinstance(name, str) or name not in coordinates:
            continue
        parameter = _parameter(each)
        if parameter is not None and parameter.name not in coordinates:
            found.setdefault(name, each)
    return found


def _parameter(call: Node) -> sympy.Symbol | None:
    """The parameter a coordinate's function is called at, t of θ(t); ``None`` for what is
    no symbol."""
    (argument,) = call.args
    value = argument.op.value if isinstance(argument.op, ops.Const) else None
    return value if isinstance(value, sympy.Symbol) else None
