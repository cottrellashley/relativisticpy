r"""The lowering of abstract index notation.

An equation with a tensor that has no components, such as T_{\mu\nu} in Einstein's
equations before T is defined, is kept as notation (:mod:`relativisticpy.abstract`), and
solve, trace, simplify and subs work on it as notation. The lowerer emits the operations
of notation, whose types their rules give (:mod:`relativisticpy.typing.notation`), and
the engines compute them as the statement runs (:mod:`relativisticpy.runtime.notation`).
Where components are needed, in a definition or a component equation, notation is turned
into components with the metric in force, by the lowerer, once the engines have computed
it: :func:`lower_notation`.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from functools import reduce
from typing import Any, Literal

import sympy
from sympy import Symbol

from relativisticpy import abstract
from relativisticpy.abstract import IndexExpression
from relativisticpy.compiler.scope import (
    Check,
    Context,
    FunctionBinding,
    Scope,
    TensorBinding,
    VariableBinding,
)
from relativisticpy.compiler.session import MetricState, Session
from relativisticpy.errors import CompileError, IndexNotationError
from relativisticpy.ir import SCALAR, VALUE, Graph, Node, NotationType, TensorType, Type, ops
from relativisticpy.language import syntax as ast
from relativisticpy.language.parser import split_index_labels
from relativisticpy.language.source import Span
from relativisticpy.reading.families import Families
from relativisticpy.reading.names import _names_builtin
from relativisticpy.tensors import Index, Indices
from relativisticpy.tensors.spaces import Space
from relativisticpy.typing import notation as rules
from relativisticpy.typing.syntax import (
    _NOTATION_OPERATIONS,
    _mentions_tensor,
    _operand,
    _product_factors,
    _signed_terms,
    _tensor_pattern,
)
from relativisticpy.typing.typer import _is_zero


class Notation(Families):
    """The lowering of abstract index notation, and of its components."""

    def _value_or_notation(
        self, node: ast.Expr, scope: Scope, ctx: Context, *, definition: bool = False
    ) -> Node:
        """The value of an expression, or of a link of a chain, or what a name is defined
        as: notation if it can only be read as notation. Without a metric, an equation or
        a definition of tensors can only be."""
        equation = definition or isinstance(node, ast.Equation)
        if self._is_notation(node, scope, ctx, equation=equation):
            return self._notation(node, scope, ctx)
        return self.value(node, scope, ctx)

    def _notation(self, node: ast.Expr, scope: Scope, ctx: Context) -> Node:
        """Notation: what it is, as a value of the program."""
        return self._abstract(node, scope, ctx)

    def _algebra(self) -> Node:
        """The algebra of notation in force, which the engines make the tensors of
        notation in."""
        algebra = self.session.algebra()
        attributes = (("dim", algebra.dim), ("metric", algebra.metric), ("number", algebra.number))
        return self.graph.add(ops.Call("notation.algebra", attributes), (), VALUE)

    def _notation_call(
        self,
        name: str,
        args: Sequence[Node],
        kind: Type,
        span: Span | None,
        attributes: ops.Attributes = (),
    ) -> Node:
        """The operation of notation ``notation.<name>``, of the type its rule gives. It is
        computed as the statement that writes it runs, as the check of a definition is
        (see :class:`~relativisticpy.compiler.scope.Check`), in the order it is written:
        so notation that cannot be computed, as a solve for a tensor it does not contain,
        is refused there, with how the statement was read, before what is written after
        it."""
        node = self.graph.add(ops.Call(f"notation.{name}", attributes), list(args), kind, span)
        self._checks.append(Check(node, span, notes=tuple(self._notes)))
        return node

    def _abstract(self, node: ast.Expr, scope: Scope, ctx: Context) -> Node:
        node = _operand(node)
        if isinstance(node, ast.Name):
            binding = scope.lookup(node.id)
            if (
                isinstance(binding, VariableBinding)
                and (found := self._kept(binding.node)) is not None
            ):
                return found
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _NOTATION_OPERATIONS
        ):
            return self._notation_operation(node, scope, ctx)
        if isinstance(node, ast.Chain):
            raise CompileError("give the links of a chain one at a time", span=node.span)
        self._index_structure(node, scope)  # what breaks the rules, before it is built
        algebra = self._algebra()
        if isinstance(node, ast.Equation):
            lhs = self._tensorial(node.lhs, scope, ctx)
            rhs = self._tensorial(node.rhs, scope, ctx)
            kind = rules.equation(_notation_type(lhs), _notation_type(rhs))
            return self._notation_call("equation", (algebra, lhs, rhs), kind, node.span)
        term = self._tensorial(node, scope, ctx)
        kind = rules.expression(_notation_type(term))
        return self._notation_call("expression", (algebra, term), kind, node.span)

    def _notation_operation(self, node: ast.Call, scope: Scope, ctx: Context) -> Node:
        assert isinstance(node.func, ast.Name)
        name, args = node.func.id, node.args
        if not args:
            raise CompileError(f"{name} needs an equation", span=node.span)
        try:
            subject = self._abstract(args[0], scope, ctx)
            kind = _notation_type(subject)
            if name == "solve":
                if len(args) != 2 or not kind.equation:
                    raise CompileError(
                        "solve takes an equation and the tensor to solve it for",
                        span=node.span,
                        hint="for example solve(E, T_{\\mu\\nu})",
                    )
                self._index_structure(args[1], scope)
                target = self._tensorial(args[1], scope, ctx)
                unknown = self._notation_call(
                    "unknown", (subject, target), target.type, args[1].span
                )
                solved = rules.solve(kind, _notation_type(unknown))
                written = (("unknown", args[1].span.text),)
                return self._notation_call("solve", (subject, unknown), solved, node.span, written)
            if name in ("LHS", "RHS"):
                if not kind.equation:
                    raise CompileError(f"{name} takes an equation", span=node.span)
                which = rules.side(kind, 0 if name == "LHS" else 1)
                return self._notation_call(name.lower(), (subject,), which, node.span)
            if name == "subs":
                rule = self._abstract(args[1], scope, ctx) if len(args) == 2 else None
                if rule is None or not _notation_type(rule).equation:
                    raise CompileError(
                        "subs takes an expression and a rule",
                        span=node.span,
                        hint="for example subs(E, u^\\alpha u_\\alpha = -1)",
                    )
                return self._notation_call("subs", (subject, rule), rules.kept(kind), node.span)
            if name == "trace":
                return self._notation_call("trace", (subject,), rules.trace(kind), node.span)
            return self._notation_call("simplify", (subject,), rules.kept(kind), node.span)
        except CompileError as error:
            if error.span is None:
                error.span = node.span
            raise

    def _tensorial(self, node: ast.Expr, scope: Scope, ctx: Context) -> Node:
        """An expression read as abstract index notation: a term of notation, which the
        engines make a SymPy tensor expression."""
        node = _operand(node)
        algebra = self.session.algebra()
        if isinstance(node, ast.Binary) and node.op in ("+", "-"):
            signed = _signed_terms(node)
            terms = [self._tensorial(term, scope, ctx) for _, term in signed]
            kind = rules.add([_notation_type(term) for term in terms])
            signs = (("signs", tuple(sign for sign, _ in signed)),)
            return self._notation_call("add", terms, kind, node.span, signs)
        if isinstance(node, ast.Binary) and node.op == "*":
            factors = [self._tensorial(factor, scope, ctx) for factor in _product_factors(node)]
            kind = rules.mul([_notation_type(factor) for factor in factors])
            return self._notation_call("mul", factors, kind, node.span)
        if isinstance(node, ast.Binary) and node.op == "/":
            by = self._tensorial(node.right, scope, ctx)
            denominator = self._notation_call("denominator", (by,), by.type, node.right.span)
            numerator = self._tensorial(node.left, scope, ctx)
            kind = rules.quotient(_notation_type(numerator))
            return self._notation_call("div", (numerator, denominator), kind, node.span)
        if isinstance(node, ast.Unary) and node.op in ("-", "+"):
            value = self._tensorial(node.operand, scope, ctx)
            if node.op == "+":
                return value
            return self._notation_call("neg", (value,), value.type, node.span)
        if (isinstance(node, ast.Scripted) and isinstance(node.base, ast.Operator)) or (
            isinstance(node, ast.Derivative | ast.Differential) and _mentions_tensor(node)
        ):
            raise CompileError(
                "derivatives are not yet part of tensor notation; they work on components",
                span=node.span,
            )
        if isinstance(node, ast.Scripted) and (pattern := _tensor_pattern(node)):
            written = self._letter_run(node, scope, ctx)
            if written is not node:
                why = "in LaTeX the indices on a run of letters that names nothing are its last's"
                self._decide("letters", node.span, " ".join(split_index_labels(pattern[0])), why)
                return self._tensorial(written, scope, ctx)
            return self._abstract_tensor(pattern, scope, node.span)
        if isinstance(node, ast.Name):
            binding = scope.lookup(node.id)
            if (
                isinstance(binding, VariableBinding)
                and (found := self._kept(binding.node)) is not None
            ):
                kept = _notation_type(found)
                if kept.equation:
                    raise CompileError(f"{node.id} is an equation", span=node.span)
                return self._notation_call("value", (found,), rules.value(kept), node.span)
            if binding is None and node.id == "R":
                kind = rules.term(algebra)
                return self._notation_call("ricci_scalar", (self._algebra(),), kind, node.span)
            if isinstance(binding, TensorBinding):  # a tensor without indices: its trace
                symbol = self.graph.const(Symbol(node.id), SCALAR, node.span)
                return self._notation_call("scalar", (symbol,), rules.term(algebra), node.span)
        if isinstance(node, ast.Equation):
            raise CompileError("an equation cannot be part of an expression", span=node.span)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name in _NOTATION_OPERATIONS:
                found = self._notation_operation(node, scope, ctx)
                kept = _notation_type(found)
                if kept.equation:
                    raise CompileError(
                        "an equation cannot be part of an expression", span=node.span
                    )
                return self._notation_call("value", (found,), rules.value(kept), node.span)
            builtin = _names_builtin(node.func, latex=ctx.latex)
            function = isinstance(scope.lookup(name), FunctionBinding) or builtin
            if not function and len(node.args) == 1 and _mentions_tensor(node):
                # \kappa(T_{\mu\nu} - ...), as typeset: a product
                times = ast.Binary("*", node.func, node.args[0], implicit=True, span=node.span)
                return self._tensorial(times, scope, ctx)
        if _mentions_tensor(node):  # not a scalar, though nothing else here reads it
            raise IndexNotationError(
                "tensor notation reads sums and products of tensors, and not yet this",
                span=node.span,
            )
        value = self.value(node, scope, ctx)
        if isinstance(value.type, TensorType):
            raise IndexNotationError(
                "in tensor notation, write a tensor with its indices", span=node.span
            )
        if not rules.a_factor(value):
            raise CompileError(
                "this factor is computed from tensors, but tensor notation is read before "
                "anything is computed",
                span=node.span,
                hint="give the tensors of the notation components, so that the equation is "
                "computed with components",
            )
        return self._notation_call("scalar", (value,), rules.term(algebra), node.span)

    def _abstract_tensor(
        self, pattern: tuple[str, list[tuple[str, bool]]], scope: Scope, span: Span
    ) -> Node:
        name, entries = pattern
        binding = scope.lookup(name)
        if isinstance(binding, VariableBinding) and rules.notation_of(binding.node) is not None:
            raise IndexNotationError(
                f"{name} stands for notation, whose indices are its own",
                span=span,
                hint=f"write it without indices, as in trace({name})",
            )
        quantity = None
        if not isinstance(binding, TensorBinding):
            quantity = self._builtin_tensor(name, len(entries))
        head = _NOTATION_NAMES.get(quantity or "", name)
        indices = Indices(Index(label, upper) for label, upper in entries)
        algebra = self.session.algebra()
        attributes: ops.Attributes = (("name", head), ("indices", indices))
        if quantity is not None:
            attributes += (("quantity", quantity),)
        else:  # a tensor of one's own, whose components notation's may need
            self.session.notation_tensors.setdefault(algebra, set()).add((head, len(entries)))
        kind = rules.tensor(algebra, indices)
        return self._notation_call("tensor", (self._algebra(),), kind, span, attributes)

    def _components_of(self, node: Node, scope: Scope, ctx: Context, span: Span) -> Node:
        """A value kept as notation, turned into components with the metric in force."""
        found = self._kept(node)
        if found is None:
            return node
        kept = _notation_type(found)
        if kept.equation:
            raise CompileError(
                "an equation in index notation has no components",
                span=span,
                hint="take one side of it, as in RHS(E)",
            )
        return self._lower(found, kept, scope, span)

    def _lower(self, notation: Node, kept: NotationType, scope: Scope, span: Span) -> Node:
        """Notation turned into components with the metric in force. Each term of the
        notation is written in the language, with its scalar coefficient bound to a name,
        and compiled; but the terms are known only once the notation is computed, so the
        lowerer does this as the statement runs (see :func:`lower_notation`), given the
        metric and the tensors in scope that the notation may be written with. The
        components' type comes from notation's rules: its free indices, in the order
        they are written."""
        spaces = [self._notation_space(index.name, span) for index in kept.free]
        for index, space in zip(kept.free, spaces, strict=True):
            if space.dim or space.basis is None:  # not the space of no coordinates
                self._check_label(index.name, space, span)
        kind = rules.components(kept.free, spaces)
        session = self.session
        references = [session.metric.node] if session.metric is not None else []
        tensors: dict[str, Indices | None] = {}
        for name, _ in sorted(session.notation_tensors.get(kept.algebra, ())):
            binding = scope.lookup(name)
            if name in tensors or not isinstance(binding, TensorBinding | VariableBinding):
                continue
            tensors[name] = binding.indices if isinstance(binding, TensorBinding) else None
            references.append(binding.node)
        labels = {
            label: declared
            for label in kept.free.names
            if (declared := self._labels.get(label) or session.labels.get(label)) is not None
        }
        context = Lowering(
            kind=kind,
            span=span,
            text=self._text,
            coordinates=session.coordinates,
            names=tuple(session.names.items()),
            metric=session.metric.names if session.metric is not None else None,
            dropped=session.dropped,
            tensors=tuple(tensors.items()),
            labels=tuple(labels.items()),
            bound=self._bound,
        )
        lower = ops.Call("notation.lower", (("context", context),))
        lowered = self.graph.add(lower, (notation, *references), VALUE, span)
        self._checks.append(Check(lowered, span, notes=tuple(self._notes)))
        return self.graph.add(ops.Call("notation.components"), (lowered,), kind, span)

    def _notation_components(self, kept: Any, scope: Scope, ctx: Context, span: Span) -> Node:
        """Notation, computed, turned into components: each term written in the language,
        with its scalar coefficient bound to a name, and compiled. The lowerer does this
        as the statement runs (see :func:`lower_notation`)."""
        value, space = kept.value, kept.space
        traces = space.traces()
        inner = scope.child()
        fresh = itertools.count()
        summands: list[ast.Expr] = []
        for coefficient, tensors in abstract.terms(value):
            present = [symbol for symbol in traces if coefficient.has(symbol)]
            try:
                monomials = (
                    sympy.Poly(coefficient, *present).terms() if present else [((), coefficient)]
                )
            except sympy.PolynomialError:
                raise CompileError(
                    "cannot turn this into components: a trace appears other than as a factor",
                    span=span,
                ) from None
            for powers, scalar in monomials:
                name = f"#c{next(fresh)}"
                inner.bind(name, VariableBinding(self.graph.const(sympy.sympify(scalar), SCALAR)))
                factors: list[ast.Expr] = [ast.Name(name, verbatim=True, span=span)]
                for symbol, power in zip(present, powers, strict=True):
                    for _ in range(power):
                        factors.append(_trace_syntax(symbol, traces[symbol], fresh, span))
                labels: dict[str, str] = {}
                for tensor in tensors:
                    factors.append(_tensor_syntax(tensor, space, labels, fresh, span))
                summands.append(
                    reduce(lambda a, b: ast.Binary("*", a, b, implicit=True, span=span), factors)
                )
        if not summands:
            return self.graph.const(sympy.S.Zero, SCALAR, span)
        # The built-in tensors and R are bound to the geometry itself, not reached by name:
        # the metric may use R as a symbol, and the Einstein tensor may be called E.
        used: set[str] = set()
        for coefficient, tensors in abstract.terms(value):
            used.update(q for t in tensors if (q := _quantity(t.head, space)) is not None)
            if coefficient.has(space.ricci_scalar):
                used.add("ricci_scalar")
        for quantity in sorted(used):
            node = self._geometry(quantity, span)
            binding: VariableBinding | TensorBinding = VariableBinding(node)
            if isinstance(node.type, TensorType):
                binding = TensorBinding(node, node.type.indices)
            inner.bind(f"#{quantity}", binding)
        total = reduce(lambda a, b: ast.Binary("+", a, b, span=span), summands)
        return self.value(total, inner, replace(ctx, latex=False))

    def _kept(self, node: Node) -> Node | None:
        """The notation a node holds, if any, in the algebra in force: notation kept before
        the coordinates changed, or the metric was renamed, is read in this one."""
        kept = rules.notation_of(node)
        if kept is None:
            return None
        algebra = self.session.algebra()
        if kept.algebra == algebra:
            return node
        tensors = self.session.notation_tensors
        tensors.setdefault(algebra, set()).update(tensors.get(kept.algebra, ()))
        kind = rules.rebased(kept, algebra)
        return self._notation_call("rebase", (self._algebra(), node), kind, node.span)


@dataclass(frozen=True)
class Lowering:
    """What turning notation into components needs of the statement that writes it,
    besides the nodes of the metric and of the tensors it may be written with: the
    components' type, where they are written, the statement's text, to write labels as it
    does, the coordinates, the names of the built-in tensors, the names the metric is
    written with (``None`` without a metric), where the metric was dropped, the tensors
    bound to the names notation may write (each with its indices, or ``None`` for a
    value), the spaces declared for the free indices, and the labels an expectation
    binds there."""

    kind: Type
    span: Span
    text: str
    coordinates: tuple[Symbol, ...] | None
    names: tuple[tuple[str, str], ...]
    metric: frozenset[str] | None
    dropped: Span | None
    tensors: tuple[tuple[str, Indices | None], ...]
    labels: tuple[tuple[str, Space], ...]
    bound: frozenset[str]

    def __str__(self) -> str:
        """How the IR shows it: the components' type and the tensors they may need."""
        written = ", ".join(name for name, _ in self.tensors)
        return f"{self.kind}" + (f" of {written}" if written else "")


def lower_notation(kept: IndexExpression, references: Sequence[Node], context: Lowering) -> Node:
    """The IR of the components of ``kept``, notation the engines have computed, as the
    statement in ``context`` would have them: compiled in a session of its own, with the
    metric and tensors the ``references`` are the nodes of. Of the components' type, as
    notation's rules gave it; zero, of notation that vanishes."""
    session = Session(
        graph=Graph(),
        coordinates=context.coordinates,
        names=dict(context.names),
        dropped=context.dropped,
    )
    nodes = list(references)
    if context.metric is not None:
        session.metric = MetricState(nodes.pop(0), context.metric)
    scope = Scope()
    for (name, indices), node in zip(context.tensors, nodes, strict=True):
        binding = TensorBinding(node, indices) if indices is not None else VariableBinding(node)
        scope.bind(name, binding)
    from relativisticpy.compiler.compiler import Compiler  # the passes, when the engines ask

    compiler = Compiler(session)
    compiler._depth = 1  # a statement's part, whose notes are the statement's
    compiler._text, compiler._labels, compiler._bound = (
        context.text,
        dict(context.labels),
        context.bound,
    )
    node = compiler._notation_components(kept, scope, Context(), context.span)
    kind = context.kind
    if not isinstance(kind, TensorType):
        return node
    if not isinstance(node.type, TensorType):
        if _is_zero(node):  # notation that vanishes, whatever indices it has
            return session.graph.add(ops.ToArray(kind.shape), (node,), kind, context.span)
        raise IndexNotationError(
            f"the components of this notation have no free indices, where it has "
            f"{compiler._written_indices(kind.indices)}",
            span=context.span,
        )
    return compiler._align(node, kind.indices, context.span, what="notation")


def _notation_type(node: Node) -> NotationType:
    """The type of a value or term of notation."""
    assert isinstance(node.type, NotationType)
    return node.type


# The names built-in tensors have in notation, whatever they were written as.
_NOTATION_NAMES = {
    "riemann": "R", "ricci": "R", "einstein": "G", "weyl": "C", "christoffel": "Gamma",
}  # fmt: skip


def _tensor_syntax(
    tensor: Any, space: Any, labels: dict[str, str], fresh: Iterator[int], span: Span
) -> ast.Scripted:
    """A tensor of notation, written as the language writes it, dummies renamed. A
    built-in tensor is written with the name its geometry is bound to, as #riemann."""
    quantity = _quantity(tensor.head, space)
    name = f"#{quantity}" if quantity is not None else tensor.head.name
    scripts: list[ast.Script] = []
    run: list[ast.IndexItem] = []
    position: Literal["^", "_"] | None = None
    for index in tensor.indices:
        label = index.name
        if label.startswith("L_"):
            label = labels.setdefault(label, f"#l{next(fresh)}")
        kind: Literal["^", "_"] = "^" if index.is_up else "_"
        if position is not None and kind != position:
            scripts.append(
                ast.Script(position, None, ast.IndexList(tuple(run), span=span), span=span)
            )
            run = []
        position = kind
        run.append(ast.IndexItem(label, span=span))
    if run and position is not None:
        scripts.append(ast.Script(position, None, ast.IndexList(tuple(run), span=span), span=span))
    return ast.Scripted(ast.Name(name, verbatim=True, span=span), tuple(scripts), span=span)


def _quantity(head: Any, space: Any) -> str | None:
    """The built-in tensor a head of notation stands for: the metric, the Kronecker delta,
    or a tensor of the metric's geometry; ``None`` for a tensor of one's own."""
    if head == space.metric:
        return "metric"
    if head == space.delta:
        return "kronecker"
    quantity: str | None = space.quantity(head)
    return quantity


def _trace_syntax(symbol: Symbol, head: Any, fresh: Iterator[int], span: Span) -> ast.Expr:
    """The trace a scalar of notation stands for: the Ricci scalar, bound to the name
    #ricci_scalar, or T^{a}{}_{a}."""
    if head is None:
        return ast.Name("#ricci_scalar", verbatim=True, span=span)
    label = f"#t{next(fresh)}"
    up = ast.Script(
        "^", None, ast.IndexList((ast.IndexItem(label, span=span),), span=span), span=span
    )
    down = ast.Script(
        "_", None, ast.IndexList((ast.IndexItem(label, span=span),), span=span), span=span
    )
    return ast.Scripted(ast.Name(head.name, verbatim=True, span=span), (up, down), span=span)
