r"""The lowering of statistics over an index: :math:`\mathbb{E}_t`, variances, covariances
and the Ledoit–Wolf estimate. A statistic binds its index as a sum does, and is a
contraction with the measure of the index's space.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from relativisticpy.compiler.scope import Context, Scope
from relativisticpy.errors import CompileError, IndexNotationError
from relativisticpy.ir import SCALAR, VALUE, Node, TensorType, Type, ops
from relativisticpy.language import syntax as ast
from relativisticpy.language.source import Span
from relativisticpy.lowering.tensors import Tensors
from relativisticpy.reading.names import _statistic
from relativisticpy.tensors import Index, Indices
from relativisticpy.tensors.indices import latex_label
from relativisticpy.tensors.spaces import Space
from relativisticpy.typing.syntax import _operand
from relativisticpy.typing.typer import _complex, _indices


class Statistics(Tensors):
    """The lowering of statistics over an index."""

    def _statistic_factors(
        self, factors: list[ast.Expr], scope: Scope, ctx: Context
    ) -> list[ast.Expr | Node]:
        """Each statistic, 𝔼_t, Var_t or Cov_t, with the factor after it it acts on,
        compiled to one node; the other factors as they are."""
        grouped: list[ast.Expr | Node] = []
        position = 0
        while position < len(factors):
            factor = factors[position]
            statistic = _statistic(factor)
            if statistic is None:
                if position + 1 < len(factors):
                    self._check_statistic_name(factor, scope)
                grouped.append(factor)
                position += 1
                continue
            if position + 1 == len(factors):
                raise CompileError(
                    f"{factor.span.text} needs something to act on, as in "
                    f"{factor.span.text}[X_{{{statistic[1]}}}]",
                    span=factor.span,
                )
            operand = factors[position + 1]
            span = factor.span.to(operand.span)
            grouped.append(self._statistic(*statistic, operand, span, scope, ctx))
            position += 2
        return grouped

    def _statistic(
        self, kind: str, label: str, operand: ast.Expr, span: Span, scope: Scope, ctx: Context
    ) -> Node:
        r"""``\mathbb{E}_t[X]``, ``\operatorname{Var}_t(X)``, ``\operatorname{Cov}_t(X, Y)``
        and the Ledoit–Wolf covariance, ``\operatorname{LedoitWolf}_t(R_{ta})``. The
        statistic binds ``t`` as a sum does: inside it, every factor shares ``t``."""
        operand = _operand(operand)
        items = operand.items if isinstance(operand, ast.Tuple | ast.List) else (operand,)
        expected = 2 if kind == "covariance" else 1
        if len(items) != expected:
            written = span.text.split("(")[0].split("[")[0]
            hint = None
            if kind == "expectation" and len(items) == 2:  # E_t[X, Y]: a covariance
                arguments = ", ".join(item.span.text for item in items)
                hint = f"for a covariance, write \\operatorname{{Cov}}_{label}({arguments})"
            raise CompileError(
                f"{written} takes {'one expression' if expected == 1 else 'two expressions'}, "
                f"but {len(items)} {'is' if len(items) == 1 else 'are'} given",
                span=span,
                hint=hint,
            )
        with self._binding(label):
            nodes = [self.value(_operand(item), scope, ctx) for item in items]
        for node in nodes:
            if not (isinstance(node.type, TensorType) and label in node.type.indices.names):
                raise IndexNotationError(
                    f"a statistic over {latex_label(label)} needs the index "
                    f"{latex_label(label)} in what it acts on",
                    span=span,
                )
        if kind == "expectation":
            return self._expectation(nodes[0], label, span)
        if kind == "ledoit-wolf":
            return self._ledoit_wolf(nodes[0], label, span)
        # Complex data is conjugated as NumPy's var and cov do: the variance is of
        # |X - E X|^2, and the covariance conjugates its second argument.
        centered = [self._centered(node, label, span) for node in nodes]
        if kind == "variance":
            assert isinstance(centered[0].type, TensorType)
            complex_space = self.graph.const(_complex(centered[0].type), VALUE)
            square = self.graph.add(
                ops.Call("core.squared_modulus"),
                (centered[0], complex_space),
                centered[0].type,
                span,
            )
            return self._expectation(square, label, span)
        with self._binding(label):
            product = self._multiply([centered[0], self._conjugate(centered[1], span)], span)
        return self._expectation(product, label, span)

    def _centered(self, node: Node, label: str, span: Span) -> Node:
        """``X - \\mathbb{E}_t[X]``, the mean spread along ``t`` again."""
        mean = self._expectation(node, label, span)
        with self._binding(label):
            kept, subtracted = self._broadcast([node, mean], span)
            assert isinstance(kept.type, TensorType)
            subtracted = self._align(subtracted, kept.type.indices, span)
            return self.graph.add(ops.Add((1, -1)), (kept, subtracted), kept.type, span)

    def _expectation(self, node: Node, label: str, span: Span) -> Node:
        """The average over ``label`` with its space's measure: a contraction with the
        measure, fused with the contraction that made ``node``, and taken term by term of a
        sum, so that the product of everything averaged is never formed: the average of
        R_{ta} R_{tb} - \\mu_a \\mu_b is two contractions, neither over days, assets and
        assets."""
        assert isinstance(node.type, TensorType)
        names = node.type.indices.names
        if label not in names:
            return node
        index = node.type.indices[names.index(label)]
        space = node.type.space(label)
        measure = self._measure(space, Index(label, not index.upper), span)
        remaining = Indices(i for i in node.type.indices if i.name != label)
        spaces = tuple(
            s for i, s in zip(node.type.indices, node.type.spaces, strict=True) if i.name != label
        )
        kind: Type = TensorType(remaining, spaces) if remaining else SCALAR
        if (parts := self._averaged_parts(node, label, span)) is not None:
            if remaining:
                parts = [self._align(part, remaining, span) for part in parts]
            if isinstance(node.op, ops.Add):
                return self.graph.add(ops.Add(node.op.signs), parts, kind, span)
            if isinstance(node.op, ops.Scale):  # what multiplies it is constant in label
                return self.graph.add(ops.Scale(), [*node.args[:-1], parts[0]], kind, span)
            return parts[0]
        if isinstance(node.op, ops.Contract):
            number = node.op.output[names.index(label)]
            op = ops.Contract(
                (*node.op.labels, (number,)),
                tuple(n for n in node.op.output if n != number),
                node.op.sizes,
            )
            return self.graph.add(op, (*node.args, measure), kind, span)
        assert isinstance(measure.type, TensorType)
        with self._unbinding(label):
            return self._contract(
                [(node, node.type.indices), (measure, measure.type.indices)], span
            )

    def _averaged_parts(self, node: Node, label: str, span: Span) -> list[Node] | None:
        """The averages over ``label`` of the tensors that a sum adds, a product with
        scalars scales or a permutation reorders, each with the indices it has in
        ``node``; ``None`` for any other tensor."""
        assert isinstance(node.type, TensorType)
        indices = node.type.indices
        parts: list[tuple[Node, Indices]]
        if isinstance(node.op, ops.Permute):  # the result's axis i is the argument's axes[i]
            unpermuted = list(indices)
            for position, axis in enumerate(node.op.axes):
                unpermuted[axis] = indices[position]
            parts = [(node.args[0], Indices(unpermuted))]
        elif isinstance(node.op, ops.Add):
            parts = [(term, indices) for term in node.args]
        elif isinstance(node.op, ops.Scale):
            parts = [(node.args[-1], indices)]
        else:
            return None
        if not all(isinstance(part.type, TensorType) for part, _ in parts):
            return None
        averaged = []
        for part, named in parts:
            assert isinstance(part.type, TensorType)
            renamed = self.graph.retype(part, TensorType(named, part.type.spaces))
            averaged.append(self._expectation(renamed, label, span))
        return averaged

    def _measure(self, space: Space, index: Index, span: Span) -> Node:
        """The weights an expectation over ``space`` averages with: those declared for it,
        or the same for every element."""
        kind = TensorType(Indices([index]), (space,))
        declared = self.session.measures.get(space)
        if declared is not None:
            return self.graph.retype(declared, kind)
        return self.graph.add(
            ops.Call("core.uniform"), (self.graph.const(space.dim, VALUE),), kind, span
        )

    def _ledoit_wolf(self, node: Node, label: str, span: Span) -> Node:
        """The Ledoit–Wolf shrinkage estimate of the covariance of ``node`` over
        ``label``, from scikit-learn: better conditioned than the sample covariance when
        there are few samples for the number of variables."""
        assert isinstance(node.type, TensorType)
        if node.type.rank != 2:
            raise IndexNotationError(
                "the Ledoit–Wolf covariance is of samples of a vector, X_{ta}", span=span
            )
        names = node.type.indices.names
        measure = self.session.measures.get(node.type.space(label))
        if measure is not None and not _uniform_weights(measure):
            raise CompileError(
                "scikit-learn's Ledoit–Wolf estimate weighs every sample alike, but the "
                f"samples over {latex_label(label)} are weighted by a measure",
                span=span,
                hint="take the covariance with the measure: \\operatorname{Cov}_t(X_{ta}, X_{tb})",
            )
        other = 1 - names.index(label)
        if other == 0:
            node = self._permute(node, (1, 0), Indices(reversed(node.type.indices)))
        assert isinstance(node.type, TensorType)
        space = node.type.spaces[1]
        kind = TensorType(Indices([Index("#a"), Index("#b")]), (space, space))
        return self.graph.add(ops.Call("stats.ledoit_wolf"), (node,), kind, span)

    def _broadcast(self, values: list[Node], span: Span) -> list[Node]:
        """Inside an expectation over t, a term without t is constant in t, as \\mu_a is
        in R_{ta} - \\mu_a: it is spread along t, so that the terms can be added."""
        widest: dict[str, tuple[Index, Space]] = {}
        for value in values:
            if isinstance(value.type, TensorType):
                for index, space in zip(value.type.indices, value.type.spaces, strict=True):
                    widest.setdefault(index.name, (index, space))
        spread: list[Node] = []
        for value in values:
            have = value.type.indices.names if isinstance(value.type, TensorType) else ()
            missing = [name for name in widest if name not in have]
            if missing and all(name in self._bound for name in missing):
                factors = [(value, _indices(value))] if have else []
                for name in missing:
                    index, space = widest[name]
                    ones = self._ones((name, index.upper), span, space)
                    factors.append((ones, Indices([index])))
                if have:
                    value = self._contract(factors, span)
                else:
                    product = self._contract(factors, span) if len(factors) > 1 else factors[0][0]
                    value = self._multiply([value, product], span)
            spread.append(value)
        return spread

    def _conjugate(self, tensor: Node, span: Span) -> Node:
        """The complex conjugate of a tensor: of components that are numbers, always, and
        of SymPy's over a complex space, where a symbol need not be real."""
        assert isinstance(tensor.type, TensorType)
        complex_space = self.graph.const(_complex(tensor.type), VALUE)
        call = ops.Call("core.conjugate")
        return self.graph.add(call, (tensor, complex_space), tensor.type, span)

    @contextlib.contextmanager
    def _binding(self, label: str) -> Iterator[None]:
        saved = self._bound
        self._bound = saved | {label}
        try:
            yield
        finally:
            self._bound = saved

    @contextlib.contextmanager
    def _unbinding(self, label: str) -> Iterator[None]:
        saved = self._bound
        self._bound = saved - {label}
        try:
            yield
        finally:
            self._bound = saved


def _uniform_weights(measure: Node) -> bool:
    """Whether a measure declared for a space weighs its elements alike."""
    from relativisticpy.tensors import numerical

    if not isinstance(measure.op, ops.Const):
        return False
    weights = numerical.numpy().asarray(measure.op.value, dtype=float)
    return bool(numerical.numpy().allclose(weights, weights.mean(), rtol=1e-12, atol=0.0))
