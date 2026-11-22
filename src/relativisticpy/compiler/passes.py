"""What the passes of the compiler share as they compile a statement.

The compiler reads a statement in three passes, each in a package of its own: the reader
(:mod:`relativisticpy.reading`) decides what each name and each ambiguous form means; the
typer (:mod:`relativisticpy.typing`) holds the rules of index notation and spaces; the
lowerer (:mod:`relativisticpy.lowering`) emits the IR. They are one walk of the syntax
tree, as reading a form may need what was made of the forms before it, as the scripts on
``re`` need to know whether ``r e`` is a product: the lowerer walks, and asks the reader
and the typer. :class:`Pass` is what they share: the session, the statement's state, and
the walk itself, which the reader and the typer ask the lowerer for.
"""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Sequence

from relativisticpy import assumptions
from relativisticpy.compiler.scope import Check, Context, Operator, Output, Scope
from relativisticpy.compiler.session import Session
from relativisticpy.ir import Node, Type, ops
from relativisticpy.language import syntax as ast
from relativisticpy.language.source import Span
from relativisticpy.reading.readings import Decision, Resolution
from relativisticpy.tensors.spaces import Space

#: What compiling an expression gives: a value, or an operator that acts on what follows.
Term = Node | Operator


class Pass:
    """The state of compiling one statement, in a :class:`~relativisticpy.compiler.Session`
    it updates as it goes."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.graph = session.graph
        # Every component selection made, in order, so that a covariant derivative
        # can tell which components were chosen inside its operand.
        self._selections: list[Node] = []
        self._fresh = itertools.count()  # for index labels no one wrote, as in □ = ∇^a ∇_a
        # The spaces the index labels of the statement being compiled range over, and
        # the labels an expectation binds while its operand is compiled.
        self._labels: dict[str, Space] = {}
        # How often the statement being compiled writes each label as a sub- and superscript.
        self._scripts: dict[str, Counter[str]] = {"_": Counter(), "^": Counter()}
        # The names the statement being compiled writes as factors, not as functions
        # called: G in 2GM, which makes G(p^2 + q^2) in the same statement a product.
        self._factors: frozenset[str] = frozenset()
        # The statement's text, to write index labels as it does.
        self._text = ""
        self._bound: frozenset[str] = frozenset()
        # How the statement being compiled was read, where that could surprise, and the
        # notes of statements that showed nothing to carry them, such as definitions.
        self._notes: list[str] = []
        self._unshown: list[str] = []
        # What the reader decided of the statement being compiled, and why, and how it
        # resolved each name: its reading (see relativisticpy.reading.readings).
        self._decided: dict[Decision, None] = {}  # in order, each once
        self._resolved: dict[Resolution, None] = {}
        self._depth = 0
        # What the definitions compiled need of the values they make, for the workbook to
        # compute as their statements run.
        self._checks: list[Check] = []
        # The identity a term of a sum beside a matrix is read with, as I is in A - \lambda I.
        self._identity: Node | None = None

    def _call(
        self,
        name: str,
        args: Sequence[Node],
        kind: Type,
        span: Span,
        attributes: ops.Attributes = (),
    ) -> Node:
        """The operation registered as ``name`` (see :mod:`relativisticpy.runtime.registry`)
        applied to ``args``, with ``attributes``; one that the bounds a program assumes
        decide something of is given them too, as ``assuming``."""
        if name in assumptions.DECIDED and self.session.bounds:
            bounds = assumptions.in_force(self.session.bounds, self.session.assumptions)
            attributes = (*attributes, ("assuming", bounds))
        return self.graph.add(ops.Call(name, attributes), list(args), kind, span)

    def _geometry(self, quantity: str, span: Span) -> Node:
        return self.session.geometry(quantity, span)

    # -- the walk, which the lowerer does --------------------------------------------

    def statement(
        self, statement: ast.Statement, scope: Scope, outer: Context | None = None
    ) -> list[Output]:
        """Compile a statement: what it shows (see :meth:`Compiler.statement`)."""
        raise NotImplementedError

    def value(self, node: ast.Expr, scope: Scope, ctx: Context) -> Node:
        """Compile an expression that must produce a value."""
        raise NotImplementedError

    def term(self, node: ast.Expr, scope: Scope, ctx: Context) -> Term:
        """Compile an expression, which may be an operator such as ``∂_μ``."""
        raise NotImplementedError

    def _multiply(self, operands: Sequence[Node], span: Span) -> Node:
        """The product of values: scalars, or tensors contracted."""
        raise NotImplementedError

    def _power(self, base: Node, exponent: Node, span: Span) -> Node:
        """A value raised to a power."""
        raise NotImplementedError

    def _kept(self, node: Node) -> Node | None:
        """The notation a node holds, if any, read in the algebra in force."""
        raise NotImplementedError

    def _components_of(self, node: Node, scope: Scope, ctx: Context, span: Span) -> Node:
        """A value kept as notation, turned into components with the metric in force."""
        raise NotImplementedError
