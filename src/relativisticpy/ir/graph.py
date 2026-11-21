"""The IR graph: nodes, and the builder that interns them."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from itertools import count
from typing import Any

from relativisticpy.ir import ops
from relativisticpy.ir.types import Type
from relativisticpy.language.source import Span

#: Fresh keys for constants, one per constant made, in any graph.
_TOKENS = count()


@dataclass(frozen=True, eq=False)
class Node:
    """One value in the graph: an operation applied to other nodes.

    ``type`` is the value's static type, including the names of its free
    indices. ``key`` identifies the *computation* (the operation and the keys of
    its arguments) and ignores the type, so two nodes that differ only in how
    their indices are named share one computed value.
    """

    op: ops.Op
    args: tuple[Node, ...]
    type: Type
    id: int
    key: Hashable = field(repr=False)
    span: Span | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        return f"%{self.id}"


class Graph:
    """Builds nodes, interning them.

    Adding the same operation on the same arguments with the same type returns
    the existing node: common-subexpression elimination for free. ``∂_c g_{ab}``
    appears three times in the Christoffel formula but is computed once.

    Every constant is a computation of its own, with a fresh key: two constants
    are never one, even with the same value, so data bound again after it changed
    in place, or one array bound to two names, is never mistaken for another.
    """

    def __init__(self) -> None:
        self._interned: dict[tuple[Hashable, Type], Node] = {}
        self._ids = count()

    def add(self, op: ops.Op, args: Sequence[Node], type: Type, span: Span | None = None) -> Node:
        key: Hashable
        if isinstance(op, ops.Const):
            key = ("const", next(_TOKENS))
            return Node(op, (), type, next(self._ids), key, span)
        key = (op, tuple(arg.key for arg in args))
        node = self._interned.get((key, type))
        if node is None:
            node = Node(op, tuple(args), type, next(self._ids), key, span)
            self._interned[(key, type)] = node
        return node

    def const(self, value: Any, type: Type, span: Span | None = None) -> Node:
        return self.add(ops.Const(value), (), type, span)

    def retype(self, node: Node, type: Type) -> Node:
        """The same computation with a different static type, e.g. renamed indices."""
        if node.type == type:
            return node
        if isinstance(node.op, ops.Const):
            return Node(node.op, (), type, next(self._ids), node.key, node.span)
        return self.add(node.op, node.args, type, node.span)

    def __len__(self) -> int:
        return len(self._interned)


def dependencies(roots: Iterable[Node]) -> list[Node]:
    """Every node the roots depend on, in an order where arguments come first."""
    ordered: list[Node] = []
    seen: set[int] = set()
    for root in roots:
        stack: list[tuple[Node, bool]] = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            if node.id in seen:
                continue
            if expanded:
                seen.add(node.id)
                ordered.append(node)
            else:
                stack.append((node, True))
                stack.extend((arg, False) for arg in reversed(node.args) if arg.id not in seen)
    return ordered


def render(roots: Sequence[Node]) -> str:
    """List the computation behind ``roots``, one node per line::

    %3 = const [[-1 + 2*M/r, 0, 0, 0], ...]          Value
    %4 = metric(%3)                                  Tensor_{a}_{b}
    %5 = inverse(%4)                                 Tensor^{a}^{b}
    %6 = christoffel(%4, %5)                         Tensor^{a}_{b}_{c}
    out %6
    """
    lines = []
    computed: dict[Hashable, Node] = {}
    for node in dependencies(roots):
        if node.key in computed:  # the same computation with differently named indices
            text = f"{node!r} = {computed[node.key]!r} relabelled"
        else:
            computed[node.key] = node
            args = ", ".join(repr(arg) for arg in node.args)
            text = f"{node!r} = {node.op.label()}" + (f"({args})" if args else "")
        lines.append(f"{text:<56} {node.type}")
    lines.extend(f"out {root!r}" for root in roots)
    return "\n".join(lines)
