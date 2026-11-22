"""The lowering of linear algebra on tensors: an inverse times what it multiplies as a
solve, eigen- and singular value decompositions, determinants, traces, and functions of
maps such as the exponential.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import sympy

from relativisticpy.compiler.scope import Context, Scope
from relativisticpy.errors import CompileError, IndexNotationError
from relativisticpy.ir import SCALAR, VALUE, Node, TensorType, Type, ops
from relativisticpy.language import syntax as ast
from relativisticpy.language.source import Span
from relativisticpy.lowering.statistics import Statistics
from relativisticpy.tensors import Index, Indices
from relativisticpy.tensors.spaces import Space
from relativisticpy.typing.syntax import _mentions_tensor
from relativisticpy.typing.typer import _KNOWN_FIRST, _indices, _shares


class LinearAlgebra(Statistics):
    """The lowering of linear algebra on tensors."""

    def _solves(self, tensors: list[Node], span: Span) -> list[Node]:
        """Replace an inverse and what it multiplies, :math:`(A^{-1})^{ab} b_b`, by the
        solution of :math:`A x = b`: solving is faster and more accurate than inverting,
        and with a symmetric :math:`A` it is a Cholesky solve. What the inverse multiplies
        is every factor joined to it through one of its indices, contracted first: in
        :math:`(G^{-1})^{kl} X_{tl} y_t`, the vector :math:`X_{tl} y_t`. It is a solve only
        where that index is written once more and the other is not among those factors;
        otherwise, as in the trace :math:`(\\Sigma^{-1})^{ab} \\Sigma_{ab}`, the product
        is contracted as it is written."""
        tensors = list(tensors)
        written: set[int] = set()  # inverses that are no solve
        while not self._bound:
            position = next(
                (
                    k
                    for k, t in enumerate(tensors)
                    if isinstance(t.op, ops.Inverse)
                    and t.id not in written
                    and any(j != k and _shares(o, t) for j, o in enumerate(tensors))
                ),
                None,
            )
            if position is None:
                break
            inverse = tensors[position]
            plan = self._solve_plan(tensors, position)
            if plan is None:
                written.add(inverse.id)
                continue
            along, kept, transpose, joined = plan
            rhs = tensors[joined[0]]
            if len(joined) > 1:
                rhs = self._contract([(tensors[k], _indices(tensors[k])) for k in joined], span)
            assert isinstance(rhs.type, TensorType)
            assert isinstance(inverse.type, TensorType)
            names = rhs.type.indices.names
            axis = names.index(along.name)
            order = [axis, *(i for i in range(len(names)) if i != axis)]
            rhs = self._permute(rhs, order, Indices(rhs.type.indices[i] for i in order))
            assert isinstance(rhs.type, TensorType)
            space = inverse.type.space(kept.name)
            indices = Indices([kept, *rhs.type.indices[1:]])
            kind = TensorType(indices, (space, *rhs.type.spaces[1:]))
            solution = self.graph.add(
                ops.Call("linalg.solve"),
                (inverse.args[0], rhs, self.graph.const(transpose, VALUE)),
                kind,
                span,
            )
            used = {position, *joined}
            tensors = [t for k, t in enumerate(tensors) if k not in used] + [solution]
        return tensors

    def _solve_plan(
        self, tensors: Sequence[Node], position: int
    ) -> tuple[Index, Index, bool, list[int]] | None:
        """How the inverse at ``position`` is a solve: the index it sums with what it
        multiplies, the index it keeps, whether that is its transpose's solve, and the
        positions of the factors joined to it through the index; ``None`` where it is
        no solve."""
        first, second = _indices(tensors[position])
        if first.name == second.name:
            return None
        others = [k for k in range(len(tensors)) if k != position]
        for along, kept, transpose in ((second, first, False), (first, second, True)):
            carriers = [k for k in others if along.name in _indices(tensors[k]).names]
            if sum(_indices(tensors[k]).names.count(along.name) for k in carriers) != 1:
                continue
            joined, pending = [carriers[0]], [carriers[0]]
            while pending:  # every factor joined to it through shared indices
                names = set(_indices(tensors[pending.pop()]).names)
                for k in others:
                    if k not in joined and names & set(_indices(tensors[k]).names):
                        joined.append(k)
                        pending.append(k)
            if not any(kept.name in _indices(tensors[k]).names for k in joined):
                return along, kept, transpose, sorted(joined)
        return None

    def _linear_algebra(self, name: str, node: ast.Call, scope: Scope, ctx: Context) -> Node | None:
        """Linear algebra on a tensor: ``eig`` and ``spec`` (its spectrum), ``svd``,
        ``det``, ``tr``, ``inv``, ``diag`` of a vector, and ``exp``, ``sqrt`` and ``log``
        of a map named without indices, :math:`e^{-iHt}`. ``None`` leaves the call to the
        builtins, as for a matrix written as a list."""
        argument = node.args[0]
        if name in ("exp", "sqrt", "log") and _mentions_tensor(argument):
            # Written with its indices: one component at a time, by the builtins, except
            # log, whose builtin takes a base and no tensor, as \ln does.
            if name != "log" or len(node.args) > 2:
                return None
            args = [self.value(arg, scope, ctx) for arg in node.args]
            if not isinstance(args[0].type, TensorType):
                return None
            return self.graph.add(ops.Map("log"), args, args[0].type, node.span)
        if name not in ("eig", "spec", "svd", "tr") and not self._names_tensor(argument, scope):
            return None  # a builtin of plain matrices and numbers, as diag(1, x^2)
        tensor = self.value(argument, scope, ctx)
        if not isinstance(tensor.type, TensorType):
            return None
        span = node.span
        if name == "diag":
            if tensor.type.rank != 1:
                raise IndexNotationError("diag makes a matrix of a vector", span=span)
            index = tensor.type.indices[0]
            indices = Indices([Index("#r", index.upper), Index("#c", index.upper)])
            space = tensor.type.spaces[0]
            kind = TensorType(indices, (space, space))
            return self.graph.add(ops.Call("linalg.diag"), (tensor,), kind, span)
        if name == "inv":
            if tensor.type.rank != 2:
                raise IndexNotationError("only a tensor with two indices has an inverse", span=span)
            return self._inverse(tensor, span)
        if name in ("eig", "svd"):
            parts = self._decomposition(node, scope, ctx)
            assert parts is not None
            return self.graph.add(ops.Call("core.tuple"), parts, VALUE, span)
        operator, rows, metric = self._as_map(tensor, span, form=name == "spec")
        if name in ("det", "tr", "trace"):
            return self.graph.add(
                ops.Call("linalg.det" if name == "det" else "linalg.trace"),
                (operator, self.graph.const(rows, VALUE)),
                SCALAR,
                span,
            )
        if name in ("exp", "sqrt", "log"):
            return self.graph.add(
                ops.Call("linalg.function"),
                (self.graph.const(name, VALUE), operator, self.graph.const(rows, VALUE)),
                operator.type,
                span,
            )
        parts = self._eig(operator, rows, metric, node, scope, ctx, values_only=True)
        return parts[0]

    def _as_map(
        self, tensor: Node, span: Span, *, form: bool = True
    ) -> tuple[Node, int, Node | None]:
        """A tensor as a map, its upper indices first: the tensor, the number of its row
        axes and, for a bilinear form with both indices down, the metric its eigenvalues
        are taken relative to; without ``form``, such a tensor is its matrix of
        components, as for the determinant of g_{ab}."""
        assert isinstance(tensor.type, TensorType)
        indices, spaces = tensor.type.indices, tensor.type.spaces
        upper = [i for i, index in enumerate(indices) if index.upper]
        lower = [i for i, index in enumerate(indices) if not index.upper]
        if len(indices) == 2 and len(upper) != 1 and not form:
            return tensor, 1, None
        if not upper and len(lower) == 2:  # a bilinear form
            space = spaces[0]
            if spaces[1] != space:
                raise IndexNotationError(
                    "a bilinear form takes two vectors of one space", span=span
                )
            if space.spacetime:
                metric: Node | None = self._geometry("metric", span)
            else:
                metric = self.session.space_metrics.get(space)
            return tensor, 1, metric
        if [spaces[i] for i in upper] != [spaces[i] for i in lower]:
            raise IndexNotationError(
                "a map takes its upper indices to its lower ones over the same spaces, "
                "as A^{a}{}_{b} does",
                span=span,
                hint="raise or lower an index, as in eig(\\Sigma^{a}{}_{b})",
            )
        order = upper + lower
        permuted = self._permute(tensor, order, Indices(indices[i] for i in order))
        return permuted, len(upper), None

    def _decomposition(self, value: ast.Expr, scope: Scope, ctx: Context) -> list[Node] | None:
        """The parts of ``eig`` (eigenvalues and eigenvectors) or ``svd`` (U, s and V),
        as typed tensors; ``None`` for any other value."""
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in ("eig", "svd")
            and scope.lookup(value.func.id) is None
            and value.args
        ):
            return None
        tensor = self.value(value.args[0], scope, ctx)
        if not isinstance(tensor.type, TensorType):
            raise CompileError(
                f"{value.func.id} takes a tensor written with its indices", span=value.span
            )
        if value.func.id == "svd":
            return self._svd(tensor, value)
        operator, rows, metric = self._as_map(tensor, value.span)
        return self._eig(operator, rows, metric, value, scope, ctx, values_only=False)

    def _eig(
        self,
        operator: Node,
        rows: int,
        metric: Node | None,
        call: ast.Call,
        scope: Scope,
        ctx: Context,
        *,
        values_only: bool,
    ) -> list[Node]:
        assert isinstance(operator.type, TensorType)
        size = math.prod(space.dim for space in operator.type.spaces[:rows])
        count = None
        if len(call.args) > 1:
            number = self.value(call.args[1], scope, ctx)
            count = self._constant(number)
            if count is None or not sympy.sympify(count).is_integer or count == 0:
                raise CompileError(
                    "the number of eigenvalues is a whole number: the largest that many, "
                    "or the smallest if it is negative",
                    span=call.args[1].span,
                    hint=None if isinstance(number.op, ops.Const) else _KNOWN_FIRST,
                )
            count = int(count)
            if abs(count) > size:
                raise CompileError(
                    f"this map has {size} eigenvalues, not {abs(count)}", span=call.args[1].span
                )
        modes = Space(f"eigenvectors of {call.args[0].span.text}", abs(count) if count else size)
        args = [operator, self.graph.const(rows, VALUE), metric or self.graph.const(None, VALUE)]
        args.append(self.graph.const(count, VALUE))
        both = self.graph.add(ops.Call("linalg.eig"), args, VALUE, call.span)
        values = self._part(both, 0, TensorType(Indices([Index("#k")]), (modes,)), call.span)
        if values_only:
            return [values]
        row_indices = Indices(Index(f"#r{i}", True) for i in range(rows))
        kind = TensorType(
            Indices([*row_indices, Index("#k")]), (*operator.type.spaces[:rows], modes)
        )
        return [values, self._part(both, 1, kind, call.span)]

    def _svd(self, tensor: Node, call: ast.Call) -> list[Node]:
        """``U_{tk}, s_k, V_{ak} := \\operatorname{svd}(R_{ta})``: the first index's
        space, the singular values and the second's."""
        assert isinstance(tensor.type, TensorType)
        if tensor.type.rank != 2:
            raise IndexNotationError("svd takes a tensor with two indices", span=call.span)
        first, second = tensor.type.indices
        rows, columns = tensor.type.spaces
        modes = Space(f"singular vectors of {call.args[0].span.text}", min(rows.dim, columns.dim))
        both = self.graph.add(
            ops.Call("linalg.svd"), (tensor, self.graph.const(1, VALUE)), VALUE, call.span
        )
        mode = Index("#k")
        return [
            self._part(
                both,
                0,
                TensorType(Indices([Index("#r", first.upper), mode]), (rows, modes)),
                call.span,
            ),
            self._part(both, 1, TensorType(Indices([mode]), (modes,)), call.span),
            self._part(
                both,
                2,
                TensorType(Indices([Index("#c", second.upper), mode]), (columns, modes)),
                call.span,
            ),
        ]

    def _part(self, whole: Node, position: int, kind: Type, span: Span) -> Node:
        index = self.graph.const(position, VALUE)
        return self.graph.add(ops.Call("core.part"), (whole, index), kind, span)
