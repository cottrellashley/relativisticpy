"""The lowering of tensors with indices: indices put on a tensor, raised and lowered with
the metric of their space, traces, (anti)symmetrisations and components chosen; and
derivatives with indices, partial, covariant and Lie, and the wave operator. Raising or
lowering an index is a contraction with the metric; a covariant derivative is a partial
derivative plus one contraction with the Christoffel symbols per index.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from relativisticpy.compiler.scope import (
    Context,
    IndexDerivative,
    LieDerivative,
    Scope,
    TensorBinding,
)
from relativisticpy.errors import CompileError, IndexNotationError
from relativisticpy.ir import SCALAR, VALUE, Node, TensorType, Type, ops
from relativisticpy.language import syntax as ast
from relativisticpy.language.source import Span
from relativisticpy.language.vocabulary import GREEK
from relativisticpy.lowering.notation import Notation
from relativisticpy.reading.names import _elements
from relativisticpy.tensors import Index, Indices
from relativisticpy.tensors.spaces import Space, spacetime
from relativisticpy.typing.notation import _passed
from relativisticpy.typing.syntax import _bracket_groups, _plain_label
from relativisticpy.typing.typer import TensorSource, _spaces


class Tensors(Notation):
    """The lowering of tensors with indices, and of their derivatives."""

    def _access(self, source: TensorSource, node: ast.Scripted, scope: Scope, ctx: Context) -> Node:
        """Put indices on a tensor: raise or lower them, take components, derivatives
        (``T_{a,b}``, ``T_{a;b}``), traces and (anti)symmetrisations."""
        entries = self._entries(node, f"the indices of {source.name}")
        main = [(item, upper) for item, upper in entries if item.derivative is None]
        derivatives = [(item, upper) for item, upper in entries if item.derivative is not None]
        if any(item.derivative is None for item, _ in entries[len(main) :]):
            raise IndexNotationError(
                "derivative indices come after the tensor's own: T_{ab;c}", span=node.span
            )
        rank = len(source.indices)
        if len(main) != rank:
            example = source.name + "".join(str(i) for i in source.indices)
            raise IndexNotationError(
                f"{source.name} has {rank} ind{'ex' if rank == 1 else 'ices'}, but "
                f"{len(main)} {'is' if len(main) == 1 else 'are'} given",
                span=node.span,
                hint=f"for example {example}",
            )
        spaces = _spaces(source.node)
        if spaces is None or any(space.spacetime for space in spaces):
            self.session.require_coordinates(node.span)
        if spaces is None:
            spaces = (self.session.spacetime(node.span),) * rank
        for (item, _), space in zip(main, spaces, strict=True):
            if _plain_label(item):
                self._check_label(str(item.label), space, node.span)
        current = Indices(Index(f"#{k}", index.upper) for k, index in enumerate(source.indices))
        result = source.node  # a scalar, \phi in \phi_{;\mu}, stays one
        if rank:
            result = self.graph.retype(source.node, TensorType(current, spaces))
        for position, (_, upper) in enumerate(main):
            if current[position].upper != upper:
                try:
                    result, current = self._move_index(result, current, position, node.span)
                except CompileError as error:  # V_a of V^{a}, with no metric to lower it
                    if self.session.metric is not None:
                        raise
                    defined = self._written_label(source.name) + self._written_indices(
                        source.indices
                    )
                    way = "raises" if upper else "lowers"
                    raise CompileError(
                        f"{source.name} is defined as {defined}; writing {node.span.text} "
                        f"{way} an index, which needs a metric",
                        span=node.span,
                        hint=error.hint,
                        notes=error.notes,
                    ) from None
        for number, (item, upper) in enumerate(derivatives):
            covariant = item.derivative == ";"
            label = f"#d{number}"
            result = self._derivative_of(result, current, label, covariant, node.span)
            moved = (*range(1, len(current) + 1), 0)
            current = Indices([*current, Index(label)])
            result = self._permute(result, moved, current)
            if upper:
                result, current = self._move_index(result, current, len(current) - 1, node.span)
        return self._label(result, [*main, *derivatives], node.span)

    def _label(
        self,
        result: Node,
        entries: Sequence[tuple[ast.IndexItem, bool]],
        span: Span,
        fixed: Mapping[int, int] | None = None,
    ) -> Node:
        """Name the axes after the items written, take traces and (anti)symmetrisations,
        then fix components: those the items name, and any already ``fixed`` by position.
        Components are fixed last so that a covariant derivative of the result can still
        reach the whole tensor (see :meth:`_index_derivative`)."""
        assert isinstance(result.type, TensorType)
        spaces = result.type.spaces
        labels: list[Index] = []
        values: dict[str, int] = {}
        for position, (item, upper) in enumerate(entries):
            if fixed and position in fixed:
                value: int | None = fixed[position]
            else:
                value = self._index_value(item, spaces[position])
            if value is not None:
                values[f"#v{position}"] = value
                labels.append(Index(f"#v{position}", upper))
            else:
                labels.append(Index(str(item.label), upper))
        result = self.graph.retype(result, TensorType(Indices(labels), spaces))
        result = self._trace(result, span)
        result = self._brackets(result, entries, span)
        return self._select(result, values, span)

    def _select(self, node: Node, values: dict[str, int], span: Span) -> Node:
        """Fix the axes named in ``values`` to those components."""
        if not values or not isinstance(node.type, TensorType):
            return node
        indices = node.type.indices
        fixed = tuple(
            (axis, values[index.name]) for axis, index in enumerate(indices) if index.name in values
        )
        remaining = Indices(index for index in indices if index.name not in values)
        spaces = tuple(s for s, i in zip(node.type.spaces, indices, strict=True) if i in remaining)
        kind: Type = TensorType(remaining, spaces) if remaining else SCALAR
        selection = self.graph.add(ops.Select(fixed), (node,), kind, span)
        self._selections.append(selection)
        return selection

    def _move_index(
        self, node: Node, indices: Indices, position: int, span: Span
    ) -> tuple[Node, Indices]:
        """Raise or lower one index with the metric of its space: spacetime's metric, one
        declared for the space, or the identity, which leaves the components as they are."""
        index = indices[position]
        assert isinstance(node.type, TensorType)
        space = node.type.spaces[position]
        if space.spacetime:
            metric = self._geometry("metric" if index.upper else "inverse", span)
        elif (declared := self.session.space_metrics.get(space)) is not None:
            metric = declared if index.upper else self._inverse(declared, span)
        else:
            renamed = Indices([index.flip() if p == position else i for p, i in enumerate(indices)])
            return self.graph.retype(node, node.type.with_indices(renamed)), renamed
        placeholder = Index(f"#m{position}", not index.upper)
        metric_indices = Indices([placeholder, index.flip()])
        output = Indices([placeholder if p == position else i for p, i in enumerate(indices)])
        moved = self._contract([(node, indices), (metric, metric_indices)], span, output=output)
        renamed = Indices([index.flip() if p == position else i for p, i in enumerate(indices)])
        assert isinstance(moved.type, TensorType)
        return self.graph.retype(moved, moved.type.with_indices(renamed)), renamed

    def _permute(self, node: Node, axes: Sequence[int], indices: Indices) -> Node:
        assert isinstance(node.type, TensorType)
        kind = TensorType(indices, tuple(node.type.spaces[axis] for axis in axes))
        if list(axes) == list(range(len(axes))):
            return self.graph.retype(node, kind)
        return self.graph.add(ops.Permute(tuple(axes)), (node,), kind, node.span)

    def _align(self, value: Node, indices: Indices, span: Span, what: str = "a sum") -> Node:
        """``value``'s components with axes in the order of ``indices``: those of the
        other terms of ``what``, a sum, the other side of an equation or a definition, or
        the other factor of an elementwise product."""
        assert isinstance(value.type, TensorType)
        axes = self._alignment(value.type.indices, indices, span, what)
        return self._permute(value, axes, indices)

    def _contract(
        self,
        factors: Sequence[tuple[Node, Indices]],
        span: Span,
        output: Indices | None = None,
    ) -> Node:
        """Multiply tensors, summing each index that appears once up and once down, or,
        in a space whose metric is the identity, twice in any position (see
        :meth:`~relativisticpy.typing.typer.Typer._contraction`)."""
        op, kind = self._contraction(factors, span, output)
        return self.graph.add(op, [node for node, _ in factors], kind, span)

    def _inverse(self, node: Node, span: Span) -> Node:
        """The inverse of a two-index tensor, whose components are the inverse matrix. Of
        a form, with both indices down, it has both up, as :math:`(\\Sigma^{-1})^{ab}`
        does; of a map it is a map, :math:`(A^{-1})^a{}_b`. It goes the other way between
        two spaces: its first index ranges over the space of the tensor's second."""
        assert isinstance(node.type, TensorType)
        assert node.type.rank == 2
        first, second = node.type.indices
        if first.upper == second.upper:
            first, second = first.flip(), second.flip()
        indices = Indices([Index(second.name, first.upper), Index(first.name, second.upper)])
        kind = TensorType(indices, node.type.spaces[::-1])
        return self.graph.add(ops.Inverse(), (node,), kind, span)

    def _name_axes(self, value: Node, indices: Indices, span: Span) -> Node:
        """Name the axes of a value whose indices no one wrote, such as the eigenvectors
        eig returns, by position after the indices of a definition."""
        assert isinstance(value.type, TensorType)
        current = value.type.indices
        if len(current) != len(indices) or not all(i.name.startswith("#") for i in current):
            return value
        for position, wanted in enumerate(indices):
            if current[position].upper != wanted.upper:
                value, current = self._move_index(value, current, position, span)
        assert isinstance(value.type, TensorType)
        for wanted, space in zip(indices, value.type.spaces, strict=True):
            self._check_label(wanted.name, space, span)
        return self.graph.retype(value, value.type.with_indices(indices))

    def _trace(self, node: Node, span: Span) -> Node:
        """Sum over indices repeated within one tensor: ``R^{a}{}_{bac}``."""
        if not isinstance(node.type, TensorType):
            return node
        names = node.type.indices.names
        if len(set(names)) == len(names):
            return node
        return self._contract([(node, node.type.indices)], span)

    def _brackets(
        self, node: Node, entries: Sequence[tuple[ast.IndexItem, bool]], span: Span
    ) -> Node:
        """(Anti)symmetrise over bracketed indices that open and close within ``entries``."""
        for kind, labels, _ in _bracket_groups([entries], span, closed=False):
            node = self._symmetrize(node, kind, labels, span)
        return node

    def _spanning_brackets(self, factors: Sequence[ast.Expr], node: Node, span: Span) -> Node:
        """Brackets opened in one factor and closed in another, as in the Bianchi identity
        ``\\nabla_{[\\lambda} R_{\\rho\\sigma]\\mu\\nu}``."""
        groups = []
        entries = []
        for factor in factors:
            if isinstance(factor, ast.Scripted):
                own = [
                    (i, s.kind == "^") for s in factor.scripts if s.indices for i in s.indices.items
                ]
                entries.append(own)
        flat = [item for own in entries for item, _ in own]
        if not any(item.opens or item.closes for item in flat):
            return node
        for kind, labels, spans_factors in _bracket_groups(entries, span, closed=True):
            if spans_factors:
                groups.append((kind, labels))
        for kind, labels in groups:
            node = self._symmetrize(node, kind, labels, span)
        return node

    def _symmetrize(self, node: Node, kind: str, labels: Sequence[str], span: Span) -> Node:
        axes = self._symmetrized(node.type, labels, span)
        op = ops.Symmetrize(axes, antisymmetric=kind == "[")
        return self.graph.add(op, (node,), node.type, span)

    def _ones(self, label: tuple[str, bool], span: Span, space: Space | None = None) -> Node:
        """𝟙_a: the covector whose components are all one, which sums what it meets."""
        name, upper = label
        space = space or self._space_of(name, span)
        kind = TensorType(Indices([Index(name, upper)]), (space,))
        return self.graph.add(
            ops.Call("core.ones"), (self.graph.const(space.dim, VALUE),), kind, span
        )

    def _lie_operator(self, node: ast.Scripted) -> LieDerivative:
        """``\\pounds_V``: the subscript names the vector field."""
        script = node.scripts[0]
        name = script.body.id if isinstance(script.body, ast.Name) else None
        if len(node.scripts) != 1 or script.kind != "_" or name is None:
            raise CompileError(
                "\\pounds needs one subscript, the vector field it is along",
                span=node.span,
                hint="for example \\pounds_V T^{\\mu}",
            )
        return LieDerivative(node.span, name)

    def _dalembertian(self, operand: Node, span: Span, written: str = "\\Box") -> Node:
        """``□X = ∇^a ∇_a X``."""
        self.session.require_coordinates(span, written)
        item = ast.IndexItem(f"#box{next(self._fresh)}", span=span)
        inner = self._index_derivative(operand, item, False, True, span)
        return self._index_derivative(inner, item, True, True, span)

    def _lie_derivative(self, vector: str, operand: Node, span: Span, scope: Scope) -> Node:
        """The Lie derivative along ``V``:
        :math:`£_V T^{a}{}_{b} = V^c \\partial_c T^{a}{}_{b} - (\\partial_c V^a) T^{c}{}_{b}
        + (\\partial_b V^c) T^{a}{}_{c}`, with one such term for every index."""
        space = self.session.spacetime(span, "a Lie derivative")
        binding = scope.lookup(vector)
        if not isinstance(binding, TensorBinding) or len(binding.indices) != 1:
            raise CompileError(
                f"\\pounds_{vector} needs {vector} to be a vector field",
                span=span,
                hint=f"define it first, for example {vector}^{{\\mu}} := [1, 0, 0, 0]",
            )
        if not binding.indices[0].upper:
            raise IndexNotationError(
                f"\\pounds_{vector} is along a vector, with its index up", span=span
            )
        field_ = self.graph.retype(binding.node, TensorType(Indices([Index("#v", True)]), (space,)))
        selection: tuple[tuple[int, int], ...] = ()
        if isinstance(operand.op, ops.Select):  # as for ∇: differentiate, then select
            selection, operand = operand.op.fixed, operand.args[0]
        names = operand.type.indices if isinstance(operand.type, TensorType) else Indices()
        current = Indices(Index(f"#{k}", i.upper) for k, i in enumerate(names))
        if names:
            assert isinstance(operand.type, TensorType)
            operand = self.graph.retype(operand, operand.type.with_indices(current))
        kind = TensorType(current, (space,) * len(current)) if current else SCALAR
        derivative = self._derivative_of(operand, current, "#c", False, span)
        terms = [
            (
                1,
                self._contract(
                    [
                        (field_, Indices([Index("#c", True)])),
                        (derivative, Indices([Index("#c"), *current])),
                    ],
                    span,
                    output=current,
                ),
            )
        ]
        field_derivative = self._derivative_of(
            field_, Indices([Index("#v", True)]), "#c", False, span
        )
        for position, index in enumerate(current):
            swapped = Indices(
                Index("#e", index.upper) if p == position else i for p, i in enumerate(current)
            )
            if index.upper:  # - (∂_e V^a) T^{..e..}
                field_indices, sign = Indices([Index("#e"), Index(index.name, True)]), -1
            else:  # + (∂_b V^e) T_{..e..}
                field_indices, sign = Indices([Index(index.name), Index("#e", True)]), 1
            terms.append(
                (
                    sign,
                    self._contract(
                        [(field_derivative, field_indices), (operand, swapped)],
                        span,
                        output=current,
                    ),
                )
            )
        result = self.graph.add(
            ops.Add(tuple(s for s, _ in terms)), [t for _, t in terms], kind, span
        )
        if not names:
            return result
        entries = [(ast.IndexItem(name.name, span=span), name.upper) for name in names]
        return self._label(result, entries, span, dict(selection))

    def _derivative_operator(self, node: ast.Scripted, *, covariant: bool) -> IndexDerivative:
        """``\\partial_\\mu`` or ``\\nabla^\\mu``: one derivative for each index. A number
        written as a superscript is an order, never an index: ``\\partial_0^2`` and
        ``\\partial^2_0`` are \\partial_0 \\partial_0, and with no subscript
        ``\\partial^2`` is \\partial^\\mu \\partial_\\mu and ``\\nabla^2`` is
        \\nabla^\\mu \\nabla_\\mu, which is \\Box."""
        scripts = node.scripts
        for script in scripts:  # \partial_{}: no derivative at all
            if script.body is None and (script.indices is None or not script.indices.items):
                where = "subscript" if script.kind == "_" else "superscript"
                raise CompileError(
                    f"empty {where}",
                    span=script.span,
                    hint=f"write the index of the derivative, as in {node.base.span.text}_\\mu",
                )
        orders = [s for s in scripts if s.kind == "^" and isinstance(s.body, ast.Number)]
        if len(orders) == 1 and len(scripts) <= 2:
            order = orders[0]
            assert isinstance(order.body, ast.Number)
            count = int(order.body.text) if order.body.text.isdigit() else 0
            rest = tuple(s for s in scripts if s is not order)
            symbol = "\\nabla" if covariant else "\\partial"
            if not rest:  # \nabla^2: the wave operator, with an index no one wrote
                if count != 2:
                    raise CompileError(
                        f"{symbol}^{order.body.text} has no meaning: a number on {symbol} is "
                        f"the order of a derivative, and {symbol}^2 is {symbol}^\\mu {symbol}_\\mu",
                        span=node.span,
                        hint=f"for a component, name it, as in {symbol}^t or {symbol}^{{\\mu=0}}",
                    )
                self.session.require_coordinates(node.span, f"{symbol}^2")
                item = ast.IndexItem(f"#box{next(self._fresh)}", span=node.span)
                return IndexDerivative(node.span, covariant, ((item, True), (item, False)))
            once = self._entries(replace(node, scripts=rest), "a derivative")
            if count < 1 or len(once) not in (1, count):
                raise IndexNotationError(
                    f"a derivative of order {order.body.text} takes one index, or {count}",
                    span=node.span,
                    hint=f"for example {symbol}_t^2 or {symbol}^2_{{tx}}",
                )
            # \partial_0^2 is \partial_0 \partial_0; \partial^2_{tx} is \partial_t \partial_x
            entries = once * count if len(once) == 1 else once
        else:
            entries = self._entries(node, "a derivative")
        for item, _ in entries:
            if item.derivative is not None:
                raise IndexNotationError("unexpected derivative index", span=item.span)
        return IndexDerivative(node.span, covariant, tuple(entries))

    def _index_derivative(
        self, operand: Node, item: ast.IndexItem, upper: bool, covariant: bool, span: Span
    ) -> Node:
        """``∂_μ X`` or ``∇_μ X``: the derivative gets the first axis."""
        if _elements(operand) is not None:  # \partial_\theta \mathbf{x} of a list x
            by = self._written_label(item.label) if item.label else None
            raise self._derivative_of_a_list(operand, [], span, by)
        if (
            self.session.coordinates is None
            and item.label
            and not isinstance(operand.type, TensorType)
        ):
            if item.label in GREEK:  # \partial_\mu \phi, nothing declared: by a variable \mu
                label = self._written_label(item.label)
                self._note(
                    f"there are no coordinates, so a derivative with the index {label} is one "
                    f"with respect to a variable named {label}; declare coordinates to make "
                    f"{label} an index"
                )
            variable = self.graph.const(self.session.symbol(item.label), SCALAR)
            return self._differentiate(operand, [(variable, 1)], span)
        self.session.require_coordinates(span, "a derivative with an index")
        selection: tuple[tuple[int, int], ...] = ()
        if covariant and isinstance(operand.op, ops.Select):
            # ∇ acts on the whole tensor and the components asked for are taken after:
            # ∇_μ T^{μt} is the t component of ∇_μ T^{μν}, which needs every component.
            selection, operand = operand.op.fixed, operand.args[0]
        if isinstance(operand.type, TensorType):
            current = Indices(Index(f"#{k}", i.upper) for k, i in enumerate(operand.type.indices))
            names = operand.type.indices
            operand = self.graph.retype(operand, operand.type.with_indices(current))
        else:
            current, names = Indices(), Indices()
        result = self._derivative_of(operand, current, "#d", covariant, span)
        indices = Indices([Index("#d"), *current])
        if upper:
            result, indices = self._move_index(result, indices, 0, span)
        entries = [(item, upper)] + [
            (ast.IndexItem(name.name, span=span), name.upper) for name in names
        ]
        fixed = {axis + 1: value for axis, value in selection}
        return self._label(result, entries, span, fixed)

    def _derivative_of(
        self, operand: Node, indices: Indices, label: str, covariant: bool, span: Span
    ) -> Node:
        """The derivative of ``operand`` (with axes ``indices``) as a new first axis
        labelled ``label``; covariant derivatives add one connection term per index."""
        coordinates = self.session.require_coordinates(span, "a derivative")
        new = Index(label)
        own = _spaces(operand) or ()
        kind = TensorType(Indices([new, *indices]), (spacetime(coordinates), *own))
        result = self.graph.add(ops.Partial(coordinates), (operand,), kind, span)
        if not covariant or not indices:
            return result
        gamma = self._geometry("christoffel", span)
        terms: list[tuple[int, Node]] = [(1, result)]
        for position, index in enumerate(indices):
            dummy = Index("#e", index.upper)
            swapped = Indices(dummy if p == position else i for p, i in enumerate(indices))
            if index.upper:  # + Γ^{a}_{c e} T^{..e..}
                gamma_indices = Indices([index, new, dummy.flip()])
                sign = 1
            else:  # - Γ^{e}_{c a} T_{..e..}
                gamma_indices = Indices([dummy.flip(), new, index])
                sign = -1
            term = self._contract(
                [(gamma, gamma_indices), (operand, swapped)], span, output=kind.indices
            )
            terms.append((sign, term))
        signs = tuple(sign for sign, _ in terms)
        return self.graph.add(ops.Add(signs), [t for _, t in terms], kind, span)

    def _differentiate(self, expr: Node, variables: Sequence[tuple[Node, int]], span: Span) -> Node:
        if any(_elements(node) is not None for node in (expr, *(v for v, _ in variables))):
            raise self._derivative_of_a_list(expr, [v for v, _ in variables], span)
        self._check_conjugate_variable(expr, [v for v, _ in variables], span)  # \bar{z}
        vectors = [v for v, _ in variables if isinstance(v.type, TensorType)]
        if vectors:
            vector = vectors[0]
            assert isinstance(vector.type, TensorType)
            if len(variables) != 1 or variables[0][1] != 1 or vector.type.rank != 1:
                raise IndexNotationError(
                    "a derivative with respect to a vector takes one first derivative",
                    span=span,
                )
            if isinstance(expr.type, TensorType):
                raise IndexNotationError(
                    "only a scalar can be differentiated with respect to a vector", span=span
                )
            index = vector.type.indices[0].flip()
            kind = TensorType(Indices([index]), vector.type.spaces)
            return self.graph.add(ops.Call("core.gradient"), (expr, vector), kind, span)
        orders = self.graph.const(tuple(order for _, order in variables), VALUE)
        args = [expr, orders, *(v for v, _ in variables)]
        if isinstance(expr.type, TensorType):
            return self.graph.add(ops.Map("core.diff"), args, expr.type, span)
        return self._call("core.diff", args, _passed(expr.type), span)
