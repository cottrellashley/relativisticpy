"""The walk of an expression: each form of the syntax lowered to the IR. At each name and
each ambiguous form the walk asks the reader what it means, and at each index the typer
whether it keeps the rules.
"""

from __future__ import annotations

import contextlib
import itertools
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import replace
from typing import Any, Literal

import sympy
from sympy import Symbol

from relativisticpy import integrations
from relativisticpy.compiler.builtins import BUILTINS, Builtin
from relativisticpy.compiler.passes import Term
from relativisticpy.compiler.scope import (
    Context,
    Dalembertian,
    DifferentialMarker,
    FunctionBinding,
    FunctionReference,
    IndexDerivative,
    LeibnizDerivative,
    LieDerivative,
    Operator,
    Scope,
    TensorBinding,
    VariableBinding,
)
from relativisticpy.compiler.session import Session
from relativisticpy.diffgeom import differential
from relativisticpy.errors import CompileError, IndexNotationError
from relativisticpy.ir import (
    SCALAR,
    VALUE,
    EquationType,
    Node,
    NotationType,
    TensorType,
    ValueType,
    dependencies,
    ops,
)
from relativisticpy.language import syntax as ast
from relativisticpy.language.parser import split_index_labels
from relativisticpy.language.source import Span
from relativisticpy.language.vocabulary import FUNCTION_WORDS, GREEK
from relativisticpy.lowering.linalg import LinearAlgebra
from relativisticpy.reading import evaluations
from relativisticpy.reading.names import (
    _LINEAR_ALGEBRA,
    _TRANSPOSES,
    BOLD_ONE,
    _a_matrix,
    _a_pair,
    _an_index,
    _bold_one,
    _bracketed_order,
    _decoration,
    _in_brackets,
    _letter,
    _mentions_symbol,
    _named,
    _names_builtin,
    _names_omega,
    _shown,
    _side_by_side,
    _slashed,
    _splittable,
    _subscripted_name,
    _symbols_written,
    _unprimed,
    _written_with,
    accented,
)
from relativisticpy.reading.reader import BookCall
from relativisticpy.reading.unknowns import read_as_functions
from relativisticpy.tensors import Index, Indices
from relativisticpy.typing.notation import _passed
from relativisticpy.typing.syntax import (
    _NOTATION_OPERATIONS,
    _has_derivative,
    _is_minus_one,
    _operand,
    _plain_label,
    _product_factors,
    _rank,
    _tensor_pattern,
    _tensor_pattern_of_one,
    _walk,
)
from relativisticpy.typing.typer import TensorSource, _indices, _listed, shown_indices


class Expressions(LinearAlgebra):
    """The walk of an expression, lowered to the IR."""

    def __init__(self, session: Session) -> None:
        super().__init__(session)
        # The sums being lowered, innermost last: an evaluation bar written without its
        # variable is in that of an integral beside it in its sum.
        self._sums: list[ast.Binary] = []

    def value(self, node: ast.Expr, scope: Scope, ctx: Context) -> Node:
        """Compile an expression that must produce a value. A builtin's name with nothing
        to act on, as N is in N^2 (the lapse) or \\frac{N}{2}, is a symbol."""
        result = self.term(node, scope, ctx)
        if isinstance(result, FunctionReference) and scope.lookup(result.name) is None:
            return self._named_alone(result)
        if isinstance(result, FunctionReference) and (
            alone := self._at_parameters(result, scope, ctx)
        ):
            return alone  # f + 1, of f(x) := ...: f at its parameters
        if isinstance(result, Operator):
            raise self._nothing_to_act_on(result, scope)
        return result

    def term(self, node: ast.Expr, scope: Scope, ctx: Context) -> Term:
        """Compile an expression, which may be an operator such as ``∂_μ``."""
        handler: Callable[[Any, Scope, Context], Term] = getattr(self, f"_{type(node).__name__}")
        return handler(node, scope, ctx)

    def _Number(self, node: ast.Number, scope: Scope, ctx: Context) -> Node:
        if _bold_one(node):
            defined = scope.lookup(BOLD_ONE)
            if isinstance(defined, VariableBinding):  # \mathbf{1} := ..., the program's own
                self._resolve_binding(node.span, defined)
                return defined.node
            if self._identity is not None:  # A - \lambda \mathbf{1}
                return self._identity
        text = node.text
        number = (
            sympy.Integer(int(text)) if _INTEGER.fullmatch(text.lstrip("-")) else sympy.Float(text)
        )
        return self.graph.const(number, SCALAR, node.span)

    def _String(self, node: ast.String, scope: Scope, ctx: Context) -> Node:
        return self.graph.const(node.value, VALUE, node.span)

    def _Dots(self, node: ast.Dots, scope: Scope, ctx: Context) -> Node:
        raise CompileError(
            f"{node.span.text} stands for terms that are not written, so it cannot be computed",
            span=node.span,
        )

    def _Relation(self, node: ast.Relation, scope: Scope, ctx: Context) -> Node:
        if node.op == "≡":  # A \equiv B, where A cannot be defined: an identity
            return self._Equation(ast.Equation(node.lhs, node.rhs, span=node.span), scope, ctx)
        if node.op == "∈":
            raise self._membership(node)
        raise CompileError(
            f"'{node.op}' is read but not computed: RelativisticPy decides equations and "
            "inequalities",
            span=node.span,
        )

    def _Chain(self, node: ast.Chain, scope: Scope, ctx: Context) -> Node:
        raise CompileError(
            "a chain of relations, a = b = c, is a statement of its own", span=node.span
        )

    def _Operator(self, node: ast.Operator, scope: Scope, ctx: Context) -> Term:
        if node.kind == "box":
            return Dalembertian(node.span)
        if node.kind == "lie":
            raise CompileError(
                "\\pounds needs the vector field it is along",
                span=node.span,
                hint="for example \\pounds_V T^{\\mu}",
            )
        symbol = "\\partial" if node.kind == "partial" else "\\nabla"
        raise CompileError(
            f"{symbol} needs an index or a variable",
            span=node.span,
            hint=f"for example {symbol}_\\mu",
        )

    def _Unary(self, node: ast.Unary, scope: Scope, ctx: Context) -> Node:
        if node.op == "*":
            raise self._star(node)
        if node.op in ("±", "∓"):  # ±x is both x and -x
            negative = ast.Unary("-", node.operand, span=node.span)
            pair = (node.operand, negative) if node.op == "±" else (negative, node.operand)
            return self.value(ast.List(pair, span=node.span), scope, ctx)
        operand = self.value(_operand(node.operand), scope, ctx)
        if node.op == "+":
            return operand
        if node.op == "-" and isinstance(operand.op, ops.Const) and operand.type == SCALAR:
            return self.graph.const(-operand.op.value, SCALAR, node.span)
        if node.op == "not":
            return self._call("core.not", [operand], SCALAR, node.span)
        minus_one = self.graph.const(sympy.S.NegativeOne, SCALAR)
        return self._multiply([minus_one, operand], node.span)

    def _Binary(self, node: ast.Binary, scope: Scope, ctx: Context) -> Term:
        if node.op in ("+", "-"):
            return self._sum(node, scope, ctx)
        if node.op in ("±", "∓"):  # a ± b is both a + b and a - b
            plus = ast.Binary("+", node.left, node.right, span=node.span)
            minus = ast.Binary("-", node.left, node.right, span=node.span)
            pair = (plus, minus) if node.op == "±" else (minus, plus)
            return self.value(ast.List(pair, span=node.span), scope, ctx)
        if node.op == "∧":
            return self._wedge(node, scope, ctx)
        if node.op == "∘":
            raise CompileError("the composition of maps is not supported", span=node.span)
        if node.op == "⊙":
            return self._hadamard(node, scope, ctx)
        if node.op == "⊕":
            return self._direct_sum(node, scope, ctx)
        if node.op == "*":
            return self._product(node, scope, ctx)
        if node.op in ("and", "or"):
            args = [self.value(node.left, scope, ctx), self.value(node.right, scope, ctx)]
            return self._call(f"core.{node.op}", args, SCALAR, node.span)
        self._check_derivative_by_an_expression(node, scope)  # \frac{d \ln p}{d(1/T)}
        self._check_misread_derivative(node, scope)  # d*y/dx, drate/dx
        left = self.value(_operand(node.left), scope, ctx)
        right = self.value(_operand(node.right), scope, ctx)
        if node.op == "/":
            return self._divide(left, right, node.span)
        return self._power(left, right, node.span)

    def _hadamard(self, node: ast.Binary, scope: Scope, ctx: Context) -> Node:
        """``A_{ta} \\odot B_{ta}``: the elementwise product, of tensors with the same free
        indices, summing nothing. Index notation has no other way to say it. Of lists or
        matrices, as ``[1, 2] \\odot [3, 4]``, it is elementwise too, never their product."""
        left = self.value(_operand(node.left), scope, ctx)
        right = self.value(_operand(node.right), scope, ctx)
        if any(isinstance(value.type, ValueType) for value in (left, right)) and not any(
            isinstance(value.type, TensorType) for value in (left, right)
        ):
            call = ops.Call("core.hadamard")
            return self.graph.add(call, (left, right), VALUE, node.span)
        if not (isinstance(left.type, TensorType) and isinstance(right.type, TensorType)):
            return self._multiply([left, right], node.span)
        right = self._align(right, left.type.indices, node.span, what="an elementwise product")
        return self.graph.add(ops.Call("linalg.hadamard"), (left, right), left.type, node.span)

    def _wedge(self, node: ast.Binary, scope: Scope, ctx: Context) -> Node:
        """The wedge product of forms written with indices:
        :math:`(A \\wedge B)_{\\mu\\nu} = 2 A_{[\\mu} B_{\\nu]}`."""
        left = self.value(_operand(node.left), scope, ctx)
        right = self.value(_operand(node.right), scope, ctx)
        ranks = [
            len(value.type.indices) if isinstance(value.type, TensorType) else 0
            for value in (left, right)
        ]
        if not any(ranks):
            raise CompileError(
                "a wedge product needs forms written with their indices",
                span=node.span,
                hint="for example A_{\\mu} \\wedge B_{\\nu}",
            )
        product = self._multiply([left, right], node.span)
        assert isinstance(product.type, TensorType)
        indices = product.type.indices
        if len(indices) != sum(ranks) or any(index.upper for index in indices):
            raise IndexNotationError(
                "a wedge product is of forms: their indices are all down and all different",
                span=node.span,
            )
        if 0 in ranks:
            return product
        result = self._symmetrize(product, "[", list(indices.names), node.span)
        p, q = ranks
        factor = sympy.factorial(p + q) / (sympy.factorial(p) * sympy.factorial(q))
        return self._multiply([self.graph.const(factor, SCALAR), result], node.span)

    def _sum(self, node: ast.Binary, scope: Scope, ctx: Context) -> Node:
        # The terms in order, without recursion: a sum may have thousands of them.
        terms: list[tuple[int, ast.Expr]] = []
        pending: list[tuple[int, ast.Expr]] = [(1, node)]
        while pending:
            sign, expr = pending.pop()
            if isinstance(expr, ast.Binary) and expr.op in ("+", "-"):
                pending.append((sign if expr.op == "+" else -sign, expr.right))
                pending.append((sign, expr.left))
            else:
                terms.append((sign, expr))
        self._sums.append(node)
        try:
            values = self._terms_of_a_sum(terms, scope, ctx)  # A - \lambda I
        finally:
            self._sums.pop()
        matrices = any(_a_matrix(v) or isinstance(v.type, TensorType) for _, v, _ in values)
        if not ctx.latex and not matrices:  # x + y*I in plain text: no imaginary unit, said so
            self._note_capital_i([term for _, term in terms], scope, sum_=True)
        tensors = [value for _, value, _ in values if isinstance(value.type, TensorType)]
        if not tensors:
            kind = VALUE if any(v.type == VALUE for _, v, _ in values) else SCALAR
            signs = tuple(sign for sign, _, _ in values)
            return self.graph.add(ops.Add(signs), [v for _, v, _ in values], kind, node.span)
        if self._bound:
            spread = self._broadcast([value for _, value, _ in values], node.span)
            values = [
                (sign, new, term) for (sign, _, term), new in zip(values, spread, strict=True)
            ]
            tensors = [value for _, value, _ in values if isinstance(value.type, TensorType)]
        self._check_terms([(value, term) for _, value, term in values], node.span, scope)
        reference = tensors[0].type
        assert isinstance(reference, TensorType)
        kept: list[tuple[int, Node]] = []
        for sign, value, term in values:
            if isinstance(value.type, TensorType):
                kept.append((sign, self._align(value, reference.indices, term.span)))
        signs = tuple(sign for sign, _ in kept)
        return self.graph.add(ops.Add(signs), [value for _, value in kept], reference, node.span)

    def _product(self, node: ast.Binary, scope: Scope, ctx: Context) -> Term:
        if (kronecker := self._kronecker(node, scope, ctx)) is not None:  # A \otimes B
            return kronecker
        if any(sign in node.span.text for sign in ("times", "×")):  # \times
            self._check_product_signs(node, scope, ctx)
        factors: list[ast.Expr] = []
        left: ast.Expr = node
        while (
            isinstance(left, ast.Binary)
            and left.op == "*"
            and not self._kronecker_written(left, scope, ctx)
        ):
            factors.insert(0, left.right)
            left = left.left
        factors.insert(0, left)
        if ctx.latex:
            factors = _slashed(factors)
        factors = [part for factor in factors for part in _unsigned(factor)]
        if not ctx.latex:  # 4*I, in plain text: I is no imaginary unit, said so
            self._note_capital_i(factors, scope, sum_=False)
        self._check_misread_integral(factors, scope)  # integral x^2 dx
        factors = self._compound_names(factors, scope, ctx)  # \Delta x, one name
        factors = self._calls_of_names(factors, scope)  # \psi_n(x), of the program's own
        factors = self._book_factors(factors, scope)  # J_{\nu}(x) is a call
        factors = self._euler_factors(factors, scope, ctx)  # D f(x) is f'(x)
        grouped = self._statistic_factors(factors, scope, ctx)
        grouped = self._ones_vectors(grouped, scope, ctx)  # \mathbf{1}^T A \mathbf{1}
        grouped = self._identities_in_a_product(grouped, scope, ctx)  # I \sigma_x
        if len(grouped) == 1 and isinstance(grouped[0], Node):
            return grouped[0]
        terms: list[Term] = []
        chosen: list[list[Node]] = []  # the components each factor selects
        previous: ast.Expr | Node | None = None
        for factor in grouped:
            mark = len(self._selections)
            if isinstance(factor, Node):  # a statistic, already compiled
                terms.append(factor)
                chosen.append([])
                previous = factor
                continue
            if isinstance(previous, ast.Expr) and terms and _an_index(previous, factor):
                last = terms[-1]
                if isinstance(last, Node) and last.type == VALUE:  # [1, 2][0], as Python does
                    raise CompileError(
                        "a list is not indexed with [ ]",
                        span=previous.span.to(factor.span),
                        notes=[
                            f"after a list, {factor.span.text} is a factor: "
                            f"{previous.span.text}{factor.span.text} multiplies the list by it"
                        ],
                        hint="to name the elements of a list, write them on the left of :=, "
                        "as in a, b := [1, 2]",
                    )
            previous = factor
            inner = scope
            if terms and isinstance(terms[-1], DifferentialMarker) and _names_omega(factor):
                # in a line element d\Omega is the solid angle, whatever \Omega means
                inner = scope.child()
                inner.bind("Omega", VariableBinding(self.graph.const(Symbol("Omega"), SCALAR)))
            term = self.term(_operand(factor), inner, ctx)
            if isinstance(term, Node):
                term = self._components_of(term, inner, ctx, factor.span)
            terms.append(term)
            chosen.append(self._selections[mark:])
        # Operators act on what follows them, innermost (rightmost) first, as far as each
        # reaches. As in physics texts, ∂_μ acts on the next factor (∂_μφ ∂^μφ is
        # (∂_μφ)(∂^μφ)) while d/dx, and an operator defined with it, acts on the rest of the
        # term (d/dx x² y³ is d/dx (x² y³), and \hat{p} x ψ is \hat{p} (x ψ)).
        stack: list[Node] = []
        selected: list[list[Node]] = []
        for position in reversed(range(len(terms))):
            term, selections = terms[position], chosen[position]
            if isinstance(term, Operator):
                alone = None
                if not stack and isinstance(term, FunctionReference):
                    alone = self._at_parameters(term, scope, ctx)  # 2 f, of f(x) := ...
                if alone is not None:
                    stack.append(alone)
                    selected.append([])
                    continue
                if not stack:
                    raise self._nothing_to_act_on(term, scope)
                if term.reach == "term":
                    rest = self._multiply(stack[::-1], node.span)
                    after = grouped[position + 1 :]
                    self._check_operand(term, after, rest, node.span, scope, ctx)
                    stack = [self._apply(term, rest, node.span, scope, ctx)]
                    selected = [[s for group in selected for s in group]]
                else:
                    whole = isinstance(term, Dalembertian | LieDerivative)
                    if whole or (isinstance(term, IndexDerivative) and term.covariant):
                        self._require_whole(stack[-1], selected[-1], term.span)
                    stack[-1] = self._apply(term, stack[-1], node.span, scope, ctx)
            else:
                stack.append(term)
                selected.append(selections)
        result = self._one_by_one(self._multiply(stack[::-1], node.span), node.span)
        return self._spanning_brackets(factors, result, node.span)

    def _multiply(self, operands: Sequence[Node], span: Span) -> Node:
        if len(operands) == 1:
            return operands[0]
        scalars = [o for o in operands if not isinstance(o.type, TensorType)]
        tensors = [o for o in operands if isinstance(o.type, TensorType)]
        if not tensors:
            kind = VALUE if any(isinstance(o.type, ValueType) for o in scalars) else SCALAR
            return self.graph.add(ops.Mul(), scalars, kind, span)
        tensors = self._solves(tensors, span)
        product = tensors[0]
        if len(tensors) > 1:
            product = self._contract([(t, _indices(t)) for t in tensors], span)
        if not scalars:
            return product
        return self.graph.add(ops.Scale(), [*scalars, product], product.type, span)

    def _divide(self, numerator: Node, denominator: Node, span: Span) -> Node:
        if isinstance(denominator.type, TensorType):
            raise IndexNotationError("cannot divide by a tensor", span=span)
        minus_one = self.graph.const(sympy.S.NegativeOne, SCALAR)
        kind = _passed(denominator.type)
        reciprocal = self.graph.add(ops.Pow(), (denominator, minus_one), kind, span)
        return self._multiply([numerator, reciprocal], span)

    def _power(self, base: Node, exponent: Node, span: Span) -> Node:
        if isinstance(base.type, TensorType):
            inverse = isinstance(exponent.op, ops.Const) and exponent.op.value == -1
            metric = isinstance(base.op, ops.Geometry) and base.op.quantity == "metric"
            raise self._tensor_power(
                self.session.names["metric"] if metric else None,
                base.type.indices,
                "-1" if inverse else None,
                span,
            )
        if isinstance(exponent.type, TensorType):
            raise IndexNotationError("cannot raise to the power of a tensor", span=span)
        if (power := self._matrix_power(base, exponent, span)) is not None:  # e^{A t}, A^k
            return power
        kind = (
            VALUE if any(isinstance(t, ValueType) for t in (base.type, exponent.type)) else SCALAR
        )
        return self.graph.add(ops.Pow(), (base, exponent), kind, span)

    def _Equation(self, node: ast.Equation, scope: Scope, ctx: Context) -> Node:
        lhs, rhs = self._sides_of_an_equation(node, scope, ctx)  # U^\dagger U = I
        self._check_pair_and_scalar(node, lhs, rhs)
        tensor = next((v.type for v in (lhs, rhs) if isinstance(v.type, TensorType)), None)
        if tensor is not None:
            assert isinstance(tensor, TensorType)
            sides = []
            for v, side in ((lhs, node.lhs), (rhs, node.rhs)):
                if isinstance(v.type, TensorType):
                    sides.append(self._align_sides(v, tensor, lhs, side.span, node.span))
                else:
                    sides.append(self._zero_side(v, tensor, side, lhs is v, scope))
            return self.graph.add(ops.Equation(), sides, EquationType(tensor), node.span)
        return self.graph.add(ops.Equation(), (lhs, rhs), EquationType(SCALAR), node.span)

    def _align_sides(
        self, value: Node, tensor: TensorType, lhs: Node, span: Span, equation: Span
    ) -> Node:
        """One side of a tensor equation, its axes in the order of the first side's."""
        self._check_sides(value, tensor, lhs, equation)
        return self._align(value, tensor.indices, span)

    def _zero_side(
        self, value: Node, tensor: TensorType, side: ast.Expr, left: bool, scope: Scope
    ) -> Node:
        """The side of a tensor equation that is not a tensor: zero, or components."""
        if value.type == VALUE:
            return self.graph.add(ops.ToArray(tensor.shape), (value,), tensor, side.span)
        self._check_zero_side(value, tensor, side, left, scope)
        return value

    def _Comparison(self, node: ast.Comparison, scope: Scope, ctx: Context) -> Node:
        first = node.operands[0]
        if isinstance(first, ast.Tuple) and not any(
            isinstance(item, _WRITTEN_OUT) for item in first.items
        ):  # a, b > 0 is a > 0 and b > 0
            each = [replace(node, operands=(name, *node.operands[1:])) for name in first.items]
            return self._call(
                "core.and", [self.value(c, scope, ctx) for c in each], SCALAR, node.span
            )
        operands = [self.value(o, scope, ctx) for o in node.operands]
        self._check_ordered(node, operands)
        relations = [
            self._call(_COMPARE[op], [a, b], SCALAR, node.span)
            for op, a, b in zip(node.ops, operands, operands[1:], strict=False)
        ]
        if len(relations) == 1:
            return relations[0]
        return self._call("core.and", relations, SCALAR, node.span)

    def _check_ordered(self, node: ast.Comparison, operands: Sequence[Node]) -> None:
        """Refuse an inequality of a tuple or a matrix, as ``(1, 2) < 3``, or ``x < 3`` of
        ``x := (1, 2)``: an inequality orders numbers, and computing it of what has no order
        would fail (guard 4 of wave 0)."""
        for position, op in enumerate(node.ops):
            for side in (position, position + 1):
                kind = _unordered(operands[side]) if op in _ORDERS else None
                if kind is not None:
                    shown = node.operands[side].span.text
                    raise CompileError(
                        f"{node.span.text} is an inequality, which orders numbers, and "
                        f"{shown} is {kind}",
                        span=node.span,
                        hint=f"compare each of its items, or its length, |{shown}|",
                    )

    def _Call(self, node: ast.Call, scope: Scope, ctx: Context) -> Node:
        func = node.func
        self._note_unbracketed(node, ctx)  # sin x y, in plain text: sin(x y)
        if isinstance(func, ast.Prime):
            return self._prime(func, node.args, node.span, scope, ctx)
        if isinstance(func, ast.Primed):  # (x^3)'(2)
            return self._primed_at(func, node.args, node.span, scope, ctx)
        if (called := self._call_of_a_name(node, scope, ctx)) is not None:  # \psi_3(x)
            return called
        if not isinstance(func, ast.Name):  # J_{\nu}(x), made one call by the reader
            book = self._book_call(node, scope)
            assert book is not None
            return self._capability(book, node.span, scope, ctx)
        if func.id in _NOTATION_OPERATIONS and self._is_notation(node, scope, ctx, operation=True):
            return self._notation(node, scope, ctx)
        name = func.id
        binding = scope.lookup(name)
        if isinstance(binding, FunctionBinding):
            line = binding.definition.span.line
            why = f"{name} is a function, defined on line {line}"
            self._decide(CALL, node.span, f"a call of {name}", why)
            args = [self.value(a, scope, ctx) for a in node.args]
            value = self._inline(binding, args, node.span, ctx)
            if not ctx.calls:  # the call that is written, not those of its body
                defined = f"{func.span.text} is the function defined on line {line}"
                self._check_numbers(func.span.text, args, value, node.span, defined)
            return value
        if binding is None and name in _BIG_OPERATOR_CALLS:  # sum(f, (n, 1, oo)), as SymPy
            return self.value(_big_operator_call(node), scope, ctx)
        trace = binding is None and name in ("trace", "tr", "Tr") and len(node.args) == 1
        if trace and (traced := self._metric_trace(node.args[0], scope, ctx)) is not None:
            return traced
        linear = {"Tr": "tr", "inverse": "inv"}.get(name, name)  # as \operatorname{tr}, inv
        if binding is None and linear in _LINEAR_ALGEBRA and node.args:
            done = self._linear_algebra(linear, node, scope, ctx)
            if done is not None:
                return done
        if self._depends(node, binding):  # \alpha(r), once \alpha is defined, or a(t)
            assert isinstance(binding, VariableBinding)
            arguments = ", ".join(arg.span.text for arg in node.args)
            why = f"{name} has a value, which depends on {arguments}"
            self._decide(CALL, node.span, f"the value of {name}", why)
            return binding.node
        if (
            func.verbatim
            and binding is None
            and not _names_builtin(func, latex=ctx.latex)
            and (book := self._book_call(node, scope))
        ):  # \mathrm{p}K_\mathrm{a}: an upright name books write a capability with is it
            return self._capability(book, node.span, scope, ctx)
        multiplies = self._multiplies(node, binding, scope, ctx)
        if multiplies == "product":  # r(\partial_1 \beta - 1), with r a coordinate
            times = ast.Binary("*", func, _group(node), implicit=True, span=node.span)
            return self.value(times, scope, ctx)
        if binding is not None:
            raise CompileError(f"{name} is not a function", span=func.span)
        if name == "geodesic_equations":
            return self._geodesic_equations(node, scope, ctx)
        if name == "geodesic":  # integrated numerically, from Python
            raise CompileError(
                "geodesics are integrated from Python, with relativisticpy.numeric.geodesic",
                span=node.span,
                hint="in the language, geodesic_equations(\\tau) gives the equations they solve",
            )
        if name == "eig" and (eigenvalues := self._eigenvalues_of(node, scope, ctx)) is not None:
            return eigenvalues  # eig(A) of a matrix, as MATLAB writes its eigenvalues
        if name in ("eig", "svd"):  # what _linear_algebra leaves: no tensor
            count = ", and how many eigenvalues if not all" if name != "svd" else ""
            hint = f"for example {name}(\\Sigma^{{a}}{{}}_{{b}})"
            if name == "eig" and len(node.args) == 1:  # eig(A), MATLAB's eigenvalues
                written = node.args[0].span.text
                hint = (
                    f"for the eigenvalues of {written}, write spec({written}); eig decomposes a "
                    "tensor written with its indices, as eig(\\Sigma^{a}{}_{b})"
                )
            raise CompileError(
                f"{name}() takes a tensor written with its indices{count}",
                span=node.span,
                hint=hint,
            )
        builtin = _names_builtin(func, latex=ctx.latex)  # \gamma(r) is an unknown function
        if not builtin and (book := self._book_call(node, scope)) is not None:
            return self._capability(book, node.span, scope, ctx)  # \Gamma(x), J_0(x)
        if builtin:
            self._check_keywords(BUILTINS[name], node)
        if multiplies == "letters":  # 2Gm(r) in LaTeX is 2 G m(r)
            self._note_letters_called(node, scope)
            *leading, last = self._euler_letters(node, scope, ctx)
            factors = [self.value(n, scope, ctx) for n in leading]
            factors.append(self.value(last, scope, ctx))
            return self._multiply(factors, node.span)
        if builtin:
            why = f"{name} is a builtin function"
            self._decide(CALL, node.span, f"a call of the builtin {name}", why)
            unknowns = read_as_functions(self, node.args, scope) if name == "dsolve" else None
            with unknowns or contextlib.nullcontext():  # dsolve(dy/dt = r y): y is y(t)
                args = [self.value(a, scope, ctx) for a in node.args]
            return self._builtin(BUILTINS[name], args, node.span)
        why = f"nothing defines {func.span.text}"
        self._decide(CALL, node.span, f"a call of the unknown function {func.span.text}", why)
        self._check_unknown_function(func, node)
        args = [self.value(a, scope, ctx) for a in node.args]
        for arg, source in zip(args, node.args, strict=True):
            if isinstance(arg.type, TensorType):
                raise CompileError(f"cannot pass a tensor to {name}()", span=source.span)
            if isinstance(arg.type, ValueType):  # f([1, 2]): of numbers, not of a list
                raise CompileError(
                    f"{func.span.text} is a function RelativisticPy does not know, so it takes "
                    "numbers and symbols, not a list",
                    span=source.span,
                )
        values = [arg.op.value if isinstance(arg.op, ops.Const) else None for arg in args]
        coordinates = self.session.coordinates or ()
        names = {c.name for c in coordinates}
        # Later, a bare name means this; but a coordinate, as r in r(t), stays a coordinate.
        if values and all(value in coordinates for value in values) and name not in names:
            self.session.functions[name] = tuple(values)
        return self._call("core.function", args, SCALAR, node.span, (("name", name),))

    def _eigenvalues_of(self, node: ast.Call, scope: Scope, ctx: Context) -> Node | None:
        """``eig(A)`` of a matrix written without indices: its eigenvalues, each as often as
        it is one, as MATLAB's ``eig(A)`` gives them and ``spec(A)`` writes them, with a
        note; ``None`` for anything else, which is refused as before. A tensor written with
        its indices, as ``eig(\\Sigma^{a}{}_{b})``, keeps NumPy's decomposition, its
        eigenvalues and eigenvectors (design/notation.md, decision 42)."""
        if len(node.args) != 1:
            return None
        matrix = self.value(node.args[0], scope, ctx)
        if not _a_matrix(matrix):
            return None
        written = node.args[0].span.text
        why = f"{written} is a matrix written without indices, of which MATLAB's eig is this"
        note = (
            f"eig({written}) is read as the eigenvalues of {written}, as MATLAB gives them, "
            f"spec({written}); for the eigenvectors too, write the matrix with its indices, "
            "as eig(A^{a}{}_{b})"
        )
        self._decide(CALL, node.span, f"the eigenvalues of {written}", why, note)
        self._note(note)
        found = self._map_function("spec", [matrix], node.span)
        assert found is not None
        return found

    def _depends(self, node: ast.Call, binding: object) -> bool:
        """Whether ``a(t)`` names a value the program gave ``a`` that depends on what is in
        its brackets, or on coordinates, as ``\\alpha(r)`` does once ``\\alpha`` is
        defined: then it is that value."""
        names = {c.name for c in self.session.coordinates or ()}
        return isinstance(binding, VariableBinding) and all(
            isinstance(a, ast.Name) and (a.id in names or _mentions_symbol(binding.node, a.id))
            for a in node.args
        )

    def _multiplies(
        self, node: ast.Call, binding: object, scope: Scope, ctx: Context
    ) -> Literal["product", "letters"] | None:
        """Whether a name with brackets after it, which names no function the program
        defines, multiplies them: ``"product"`` where the name multiplies the brackets, as
        \\pi does in ``\\pi(x + 1)`` and x in ``x(x + 1)`` (decision 28); ``"letters"``
        where, in LaTeX, it is a run of letters that names nothing, of which only the last
        is called, as ``Gm(r)`` is G m(r); ``None`` where it is a call. Each reading is
        recorded with why."""
        func = node.func
        assert isinstance(func, ast.Name)
        name = func.id
        names = {c.name for c in self.session.coordinates or ()}
        one_name = len(node.args) == 1 and isinstance(node.args[0], ast.Name)
        product = (name in names and not one_name) or self._a_product(func, node.args, binding, ctx)
        if product:
            why = product if isinstance(product, str) else f"{name} is a coordinate"
            self._decide(CALL, node.span, "a product", why)
            return "product"
        if binding is not None or not ctx.latex:
            return None
        if not (_splittable(func) or self._labelled_run(func, scope) is not None):
            return None  # Gm(r), or Gm_1(r), a run with a subscript on its last letter
        if _names_builtin(func, latex=ctx.latex) or self._book_call(node, scope) is not None:
            return None
        if _in_brackets(func):  # (xy)(x + 1): the letters in brackets are one factor
            why = "in LaTeX a run of letters that names nothing is a product, here in brackets"
            self._decide(CALL, node.span, "a product", why)
            return "product"
        labels = self._letters_of(func, scope)
        read = f"{' '.join(labels[:-1])} times {labels[-1]} called"
        why = "in LaTeX only the last letter of a run that names nothing is called"
        self._decide("letters", node.span, read, why)
        return "letters"

    def _note_letters_called(self, node: ast.Call, scope: Scope) -> None:
        """Say how a run of three letters or more that nothing defines, with brackets after
        it, is read in LaTeX, as ``foo(x)`` is f o o(x), where a function foo may be meant,
        as plain text reads it (design/notation.md, decision 41.3)."""
        func = node.func
        assert isinstance(func, ast.Name)
        labels = self._letters_of(func, scope)
        if len(labels) < 3:  # Gm(r), G m(r) as physics writes it
            return
        inside = ", ".join(arg.span.text for arg in node.args)
        letters = " ".join(self._written_label(label) for label in labels[:-1])
        self._note(
            f"{node.span.text} is read as {letters} times {labels[-1]}({inside}), as LaTeX "
            f"sets a run of letters; for a function {func.id}, write "
            f"\\operatorname{{{func.id}}}({inside})"
        )

    def _raised_product(self, node: ast.Scripted, scope: Scope, ctx: Context) -> Node | None:
        """A power written on brackets after a name that multiplies them: a power of the
        brackets alone, as TeX sets it, so that ``\\pi(x + 1)^2`` is π (x + 1)² and
        ``x(x + 1)^2`` is x (x + 1)²; and on a run of letters of which only the last is
        called, a power of that call, so that ``Gm(r)^2`` is G m(r)². ``None`` for a power
        of a call, as ``f(x)^2`` is, and for any other script."""
        call = node.base
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and "_" not in call.func.id  # \psi_3(x), a member of a family, is called
            and call.func.id not in _NOTATION_OPERATIONS
            and all(_a_power(script) for script in node.scripts)
        ):
            return None
        binding = scope.lookup(call.func.id)
        if isinstance(binding, FunctionBinding) or self._depends(call, binding):
            return None
        linear = {"Tr": "tr", "inverse": "inv"}.get(call.func.id, call.func.id)
        if binding is None and (call.func.id in ("trace", "tr", "Tr") or linear in _LINEAR_ALGEBRA):
            return None
        multiplies = self._multiplies(call, binding, scope, ctx)
        if multiplies is None:
            return None
        if multiplies == "product":
            group = _bracketed(call)
            raised = ast.Scripted(group, node.scripts, span=group.span.to(node.span))
            times = ast.Binary("*", call.func, raised, implicit=True, span=node.span)
            return self.value(times, scope, ctx)
        self._note_letters_called(call, scope)
        *leading, last = self._euler_letters(call, scope, ctx)
        factors = [self.value(n, scope, ctx) for n in leading]
        raised = ast.Scripted(last, node.scripts, span=last.span.to(node.span))
        factors.append(self.value(raised, scope, ctx))
        return self._multiply(factors, node.span)

    def _euler_factors(self, factors: list[ast.Expr], scope: Scope, ctx: Context) -> list[ast.Expr]:
        """Factors with Euler's D before a function applied made its derivative there, as
        Stewart writes ``D f(x)`` for f'(x) (see :meth:`_euler`), and ``D_x``, which the
        lexer joins into one name, the derivative ``D_{x}`` before what follows it."""
        merged: list[ast.Expr] = []
        position = 0
        while position < len(factors):
            factor = factors[position]
            following = factors[position + 1] if position + 1 < len(factors) else None
            derivative = None
            if following is not None and _side_by_side(factor, following):  # not D \cdot f(x)
                derivative = self._euler(factor, following, scope, ctx)
            if following is not None and derivative is None:
                numbered = None
                if _side_by_side(factor, following):  # D_1 f, not D_1 \cdot f
                    numbered = self._numbered_derivative(factor, following, scope, ctx)
                if numbered is not None:
                    merged.append(numbered)
                    position += 2
                    continue
                operator = self._joined_derivative(factor, scope)  # D_x f(x)
                if operator is not None:
                    merged.append(operator)
                    position += 1
                    continue
            merged.append(derivative or factor)
            position += 2 if derivative is not None else 1
        return merged

    def _letters_of(self, func: ast.Name, scope: Scope) -> list[str]:
        """The letters of a run that names nothing, ``Gm`` or ``Gm_1``, the subscript on the
        last."""
        return self._labelled_run(func, scope) or split_index_labels(func.id)

    def _euler_letters(self, node: ast.Call, scope: Scope, ctx: Context) -> list[ast.Expr]:
        """The factors of a run of letters of which the last is called, as
        :func:`_letters_called` gives them, with Euler's D before the call made its
        derivative, as ``Df(x)`` is f'(x)."""
        assert isinstance(node.func, ast.Name)
        *leading, last = _letters_called(node, self._letters_of(node.func, scope))
        if leading and (derivative := self._euler(leading[-1], last, scope, ctx)) is not None:
            return [*leading[:-1], derivative]
        return [*leading, last]

    def _euler(
        self, operator: ast.Expr, operand: ast.Expr, scope: Scope, ctx: Context
    ) -> ast.Expr | None:
        """``D f(x)``: Euler's D, which nothing defines, before a function of one variable
        applied, a function the program defines or one it does not know, is its derivative
        there, f'(x), with a note (decision 18). ``None`` for anything else: D with a value,
        and D before anything but a call, which multiplies."""
        if not (
            isinstance(operator, ast.Name)
            and operator.id == "D"
            and self._undefined(operator, scope)
            and isinstance(operand, ast.Call)
            and isinstance(operand.func, ast.Name)
            and len(operand.args) == 1
        ):
            return None
        func = operand.func
        binding = scope.lookup(func.id)
        if isinstance(binding, FunctionBinding):
            if len(binding.definition.params) != 1:
                return None
        elif binding is not None or _names_builtin(func, latex=ctx.latex):
            return None
        elif self._multiplies(operand, binding, scope, ctx) is not None:
            return None  # D x(x + 1): x multiplies, so D is a factor too
        applied = f"{self._written_label(func.id)}({operand.args[0].span.text})"
        primed = applied.replace("(", "'(", 1)
        self._note(
            f"D {applied} is read as Euler's D, the derivative {primed}; for D times "
            f"{applied}, write D \\cdot {applied}"
        )
        prime = ast.Prime(func, 1, span=func.span)
        return ast.Call(prime, operand.args, span=operand.span)

    def _metric_trace(self, argument: ast.Expr, scope: Scope, ctx: Context) -> Node | None:
        """``trace(R_{\\mu\\nu})``: the two indices of a tensor on spacetime contracted with
        the metric, as ``trace`` does for notation, rather than the sum of the diagonal
        components, which depends on the coordinates. ``None`` for any other argument."""
        if not self._names_tensor(argument, scope):
            return None
        tensor = self.value(argument, scope, ctx)
        kind = tensor.type
        if not (isinstance(kind, TensorType) and kind.rank == 2):
            return None
        if not all(space.spacetime for space in kind.spaces):
            return None
        indices = Indices(Index(f"#t{k}", index.upper) for k, index in enumerate(kind.indices))
        node = self.graph.retype(tensor, kind.with_indices(indices))
        if indices[0].upper == indices[1].upper:  # T^{a}{}_{a}, raised with the metric
            node, indices = self._move_index(node, indices, 1, argument.span)
        assert isinstance(node.type, TensorType)
        paired = Indices([indices[0], Index(indices[0].name, indices[1].upper)])
        return self._trace(self.graph.retype(node, node.type.with_indices(paired)), argument.span)

    def _builtin(
        self, builtin: Builtin, args: list[Node], span: Span, attributes: ops.Attributes = ()
    ) -> Node:
        if message := builtin.check_arity(len(args)):
            raise CompileError(message, span=span)
        pair = next((arg for arg in args if _a_pair(arg)), None)
        if pair is not None and builtin.name in _of_numbers():  # \sqrt{(a, b)}
            raise self._function_of_a_pair(builtin.name, pair, span)
        if any(_a_relation(arg) for arg in args) and builtin.name in _of_numbers():
            self._check_numbers(builtin.name, args, None, span)  # \sin(1 < 2)
        if (function := self._function_of_a_matrix(builtin.name, args, span)) is not None:
            return function  # \exp, \log or \sqrt of a matrix
        self._check_unknown_matrix(builtin.name, args, span)  # solve(A^T P + P A = -I, P)
        if (lifted_map := self._map_function(builtin.name, args, span)) is not None:
            return lifted_map  # \operatorname{rank} A^a{}_b, of a map
        if builtin.componentwise:  # each component of the first tensor among the arguments
            lifted = next((k for k, a in enumerate(args) if isinstance(a.type, TensorType)), None)
            if lifted is not None:
                tensor, rest = args[lifted], args[:lifted] + args[lifted + 1 :]
                op = ops.Map(builtin.name, attributes, lifted)
                return self.graph.add(op, [tensor, *rest], tensor.type, span)
        args = [self._as_object(arg) for arg in args]
        kind = SCALAR if builtin.scalar and all(a.type == SCALAR for a in args) else VALUE
        return self._call(builtin.name, args, kind, span, attributes)

    def _check_numbers(
        self,
        function: str,
        args: Sequence[Node],
        value: Node | None,
        span: Span,
        defined: str | None = None,
    ) -> None:
        """Refuse a relation given to a function of numbers, as ``f(1 < 2)`` of
        ``f(x) := x^2``: true or false, it is no number, and computing with it would fail.
        ``value`` is what a function the program defines makes of its arguments: it is one of
        numbers where it does arithmetic with them, while a relation it only hands on, as
        to ``solve``, is as it was; ``defined`` says where. Of a random variable, as
        ``P(X < 0)`` of a P the program defines, the probability of the event is meant, as
        the hint says."""
        relations = [arg for arg in args if _a_relation(arg)]
        if relations and value is not None:
            numbers = _of_numbers()
            computed = {
                arg.id
                for node in dependencies([value])
                if _arithmetic(node, numbers)
                for arg in node.args
            }
            relations = [relation for relation in relations if relation.id in computed]
        if not relations:
            return
        relation = relations[0]
        written = relation.span.text if relation.span is not None else "a relation"
        notes = [defined] if defined is not None else []
        notes.append(f"a relation, as {written} is, is true or false, not a number")
        hint = None
        if _random(relation):
            hint = f"for the probability of the event, write \\Pr({written})"
        raise CompileError(
            f"{span.text} gives {function} the relation {written}, and {function} takes a number",
            span=span,
            notes=notes,
            hint=hint,
        )

    def _capability(self, book: BookCall, span: Span, scope: Scope, ctx: Context) -> Node:
        """A call written as books write a capability, as ``J_{\\nu}(x)``, lowered as a
        call of it by its name, with the parameters its form gives by name."""
        traced = None
        if book.capability.name == "trace" and len(book.arguments) == 1:  # tr R_{\mu\nu}
            traced = self._metric_trace(book.arguments[0], scope, ctx)
        if traced is not None:
            return traced
        args = [self.value(arg, scope, ctx) for arg in book.arguments]
        return self._builtin(BUILTINS[book.capability.name], args, span, book.keywords)

    def _as_object(self, node: Node) -> Node:
        """Wrap tensor components in a :class:`Tensor` for a function that wants one."""
        if isinstance(node.type, TensorType):
            wrap = ops.Call("core.tensor", self._wrapped(node.type.indices))
            return self.graph.add(wrap, (node,), VALUE, node.span)
        if isinstance(node.type, EquationType) and isinstance(node.type.of, TensorType):
            wrap = ops.Call("core.tensor_equation", self._wrapped(node.type.of.indices))
            return self.graph.add(wrap, (node,), VALUE, node.span)
        return node

    def _wrapped(self, indices: Indices) -> ops.Attributes:
        """The attributes of an operation that gives tensor components to a function that
        wants a :class:`~relativisticpy.tensors.Tensor`: its indices, as they are shown,
        and the coordinates."""
        return (
            ("indices", shown_indices(indices)),
            ("coordinates", self.session.coordinates or ()),
        )

    def _inline(self, binding: FunctionBinding, args: list[Node], span: Span, ctx: Context) -> Node:
        definition = binding.definition
        if definition.name in ctx.calls:
            raise CompileError(
                f"{definition.name} calls itself; recursive functions are not supported",
                span=span,
            )
        if len(args) != len(definition.params):
            count = len(definition.params)
            raise CompileError(
                f"{definition.name}() takes {count} argument{'s' * (count != 1)}, not {len(args)}",
                span=span,
            )
        scope = binding.closure.child()
        for param, arg in zip(definition.params, args, strict=True):
            scope.bind(param, VariableBinding(arg))
        calls = (*ctx.calls, definition.name)
        inner = Context(latex=definition.latex, calls=calls, body=definition)
        if isinstance(definition.body, ast.Block):
            result = None
            for statement in definition.body.statements:
                outputs = self.statement(statement, scope, inner)
                if outputs:
                    result = outputs[-1].node
            if result is None:
                raise CompileError(
                    f"{definition.name} does not return a value",
                    span=definition.span,
                    hint="end the function body with an expression",
                )
            return self._at_arguments(result, definition, args, span)
        with self._as_written_in(definition):
            value = self.value(definition.body, scope, inner)
        return self._at_arguments(value, definition, args, span)

    @contextlib.contextmanager
    def _as_written_in(self, definition: ast.FunctionDefinition) -> Iterator[None]:
        """Read what a definition holds as the statement that makes it is written, wherever
        it is used: a hat in the body of a function is read by the hats its definition
        writes, not by those of the statement that calls it."""
        saved, self._text = self._text, definition.span.text
        try:
            yield
        finally:
            self._text = saved

    def _at_arguments(
        self, value: Node, definition: ast.FunctionDefinition, args: Sequence[Node], span: Span
    ) -> Node:
        """A function's value at its arguments where a parameter is a coordinate, as r is
        of ``\\omega(r) := -k_\\mu u^\\mu``: the components of the tensors in its body are
        written in the coordinate, which binding the parameter does not reach, so the
        argument is put in for the coordinate too, and \\omega(r_e) is \\omega at r_e."""
        coordinates = {c.name: c for c in self.session.coordinates or ()}
        pairs: list[Node] = []
        for param, arg in zip(definition.params, args, strict=True):
            coordinate = coordinates.get(param)
            same = isinstance(arg.op, ops.Const) and arg.op.value == coordinate
            if coordinate is not None and not same:
                pairs += [self.graph.const(coordinate, SCALAR), arg]
        if not pairs or isinstance(value.type, NotationType):  # notation has no components
            return value
        return self._substituted(value, pairs, span)

    def _prime(
        self, prime: ast.Prime, args: tuple[ast.Expr, ...], span: Span, scope: Scope, ctx: Context
    ) -> Node:
        """``f'(x)``: differentiate ``f`` with respect to its argument, then apply it; but a
        function the program defines under the primed name, as ``H'(x) := ...``, is that."""
        name = prime.func.id
        primed = scope.lookup(name + "'" * prime.order)
        if isinstance(primed, FunctionBinding):  # H'(x) := ..., a function of its own
            return self._inline(primed, [self.value(a, scope, ctx) for a in args], span, ctx)
        if len(args) != 1:
            raise CompileError(f"{name}{chr(39) * prime.order}(x) takes one argument", span=span)
        point = self.value(args[0], scope, ctx)
        dummy = self.graph.const(sympy.Dummy(name), SCALAR)
        binding = scope.lookup(name)
        if isinstance(binding, FunctionBinding):
            body = self._inline(binding, [dummy], span, ctx)
        elif binding is not None:
            raise CompileError(f"{name} is not a function", span=prime.func.span)
        elif _names_builtin(prime.func, latex=ctx.latex):
            body = self._builtin(BUILTINS[name], [dummy], span)
        else:
            body = self._call("core.function", [dummy], SCALAR, span, (("name", name),))
        order = self.graph.const(prime.order, VALUE)
        return self._call("core.derivative_at", [body, dummy, order, point], SCALAR, span)

    def _geodesic_equations(self, node: ast.Call, scope: Scope, ctx: Context) -> Node:
        if len(node.args) > 1:
            raise CompileError(
                f"geodesic_equations(\\tau) takes at most 1 argument, not {len(node.args)}",
                span=node.span,
            )
        parameter = (
            self.value(node.args[0], scope, ctx)
            if node.args
            else self.graph.const(sympy.Symbol("tau"), SCALAR)
        )
        christoffel = self._geometry("christoffel", node.span)
        coordinates = self.graph.const(self.session.coordinates, VALUE)
        return self._call(
            "core.geodesic_equations", [christoffel, coordinates, parameter], VALUE, node.span
        )

    def _require_whole(self, operand: Node, selections: Sequence[Node], span: Span) -> None:
        """A covariant derivative acts on whole tensors. Components chosen as the operand
        is accessed, as in ∇_μ T^{μt}, are taken after the derivative (see
        :meth:`_index_derivative`); components chosen inside it, as in ∇_μ (A^μ B^t),
        are an error rather than a guess at which was meant."""
        inner = operand.args[0] if isinstance(operand.op, ops.Select) else operand
        inside = {node.id for node in dependencies([inner])}
        if any(selection.id in inside for selection in selections):
            raise IndexNotationError(
                "a covariant derivative acts on whole tensors, not on components chosen "
                "inside its operand",
                span=span,
                hint="define the operand as a tensor first, as in X^{\\mu\\nu} := "
                "A^{\\mu} B^{\\nu}, then write \\nabla_\\mu X^{\\mu t}; or name a "
                "component to use it as a function, as in f := g_{tt}",
            )

    def _apply(
        self, operator_: Operator, operand: Node, span: Span, scope: Scope, ctx: Context
    ) -> Node:
        """Apply an operator to the value after it; ``span`` is the expression written."""
        if isinstance(operator_, FunctionReference):
            binding = scope.lookup(operator_.name)
            if isinstance(binding, FunctionBinding):
                return self._inline(binding, [operand], span, ctx)
            return self._builtin(BUILTINS[operator_.name], [operand], span)
        if isinstance(operator_, LeibnizDerivative):
            return self._differentiate(operand, operator_.variables, span)
        if isinstance(operator_, DifferentialMarker):
            return self._differential(operand, span)
        if isinstance(operator_, Dalembertian):
            return self._dalembertian(operand, span, operator_.span.text)
        if isinstance(operator_, LieDerivative):
            return self._lie_derivative(operator_.vector, operand, span, scope)
        assert isinstance(operator_, IndexDerivative)
        result = operand
        for item, upper in reversed(operator_.items):
            result = self._index_derivative(result, item, upper, operator_.covariant, span)
        return result

    def _Derivative(self, node: ast.Derivative, scope: Scope, ctx: Context) -> Term:
        if node.written is not None and (
            defined := [name for name in node.written.names if scope.lookup(name) is not None]
        ):  # dy/dx, after dx := 0.01: the quotient of what the program defines
            shown = _listed([_shown(name, ctx.latex) for name in defined])
            verb = "is" if len(defined) == 1 else "are"
            note = f"{shown} {verb} defined, so {node.span.text} is a quotient, not a derivative"
            why = f"the program defines {shown}"
            self._decide("derivative or quotient", node.span, "a quotient", why, note)
            self._note(note)
            return self.term(node.written.expr, scope, ctx)
        if (coordinate := self._along_a_path(node, scope)) is not None:
            raise CompileError(
                f"the derivative of the coordinate {coordinate} along a path is read "
                "but not computed",
                span=node.span,
                hint=r"write the path's coordinates as functions of its parameter, as in r(\tau)",
            )
        variables = tuple((self.value(v, scope, ctx), order) for v, order in node.variables)
        if node.expr is None:
            return LeibnizDerivative(node.span, variables, node.partial)
        expr = node.expr
        binding = scope.lookup(expr.id) if isinstance(expr, ast.Name) else None
        alone = (
            isinstance(expr, ast.Name)
            and self._undefined(expr, scope)
            and not (ctx.latex and _splittable(expr))  # GM in LaTeX is G M, no one name
        )
        if isinstance(binding, FunctionBinding) and isinstance(expr, ast.Name):
            # \frac{d^2 f}{dx^2}, with f(x) defined: the derivative of f(x)
            expr = self._applied(expr, binding, node)
        elif isinstance(expr, ast.Name) and alone:
            by = [v for v, _ in node.variables if isinstance(v, ast.Name)]
            names, variables_written = {expr.id: expr.span.text}, {v.id: v.span.text for v in by}
            self._check_tied(names, variables_written, node.span, alone=expr.span)  # dy/dx
            written = list(dict.fromkeys(v.span.text for v in by))
            if written and expr.id not in {v.id for v in by}:
                shown = expr.span.text
                self._note(
                    f"{shown} does not depend on {_listed(written)}, so its derivative is 0; "
                    f"for a function, write {shown}({', '.join(written)})"
                )
        value = self.value(expr, scope, ctx)
        if not alone:  # \frac{d(y^2)}{dx}, of what is written with y
            self._check_differentiated(
                [node.expr],
                value,
                self._differentiated_by(variables),
                node.span,
                scope,
                ctx,
                partial=node.partial,
            )
        return self._differentiate(value, variables, node.span)

    def _note_unbracketed(self, node: ast.Call, ctx: Context) -> None:
        """Say how far a function written in letters without brackets reaches, in plain
        text, where it takes a product of names: ``sin x y`` is sin(x y), as LaTeX reads
        ``\\sin x y`` (design/notation.md, decisions 11 and 40), where it could be sin(x) y.
        A number before a name, as in ``sin 2x``, is no such product."""
        func = node.func
        if ctx.latex or len(node.args) != 1 or not isinstance(func, ast.Name):
            return
        written, argument = func.span.text, node.args[0]
        if written not in FUNCTION_WORDS:
            return
        between = Span(func.span.source, func.span.end, argument.span.start).text
        factors = _product_factors(argument)
        names = [factor for factor in factors if isinstance(factor, ast.Name)]
        if any(bracket in between for bracket in "([{") or len(names) < 2:
            return  # sin(x y), or sin 2x
        first = factors[0].span.text
        rest = " ".join(factor.span.text for factor in factors[1:])
        self._note(
            f"{written} {argument.span.text} is read as {written}({argument.span.text}), as "
            f"LaTeX reads it; for {written}({first}) times {rest}, write {written}({first}) {rest}"
        )

    def _differentiated_by(self, variables: Sequence[tuple[Node, int]]) -> dict[str, str]:
        """The names of the variables a derivative is by, each as the statement writes it."""
        return {
            variable.op.value.name: self._written_label(variable.op.value.name)
            for variable, _ in variables
            if isinstance(variable.op, ops.Const) and isinstance(variable.op.value, Symbol)
        }

    def _check_operand(
        self,
        operator_: Operator,
        written: Sequence[ast.Expr | Node],
        operand: Node,
        span: Span,
        scope: Scope,
        ctx: Context,
    ) -> None:
        """Check what an operator that reaches the rest of its term is applied to, as it is
        ``written``, before it is applied: a derivative keeps guard 6, so that
        ``\\frac{d}{dx} y`` after x^3 + y^3 = 6xy is refused as ``\\frac{dy}{dx}`` is."""
        if isinstance(operator_, LeibnizDerivative):
            by = self._differentiated_by(operator_.variables)
            partial = operator_.partial
            self._check_differentiated(written, operand, by, span, scope, ctx, partial=partial)

    def _check_differentiated(
        self,
        written: Sequence[ast.Expr | Node],
        operand: Node,
        by: Mapping[str, str],
        span: Span,
        scope: Scope,
        ctx: Context,
        *,
        partial: bool,
    ) -> None:
        """Refuse the derivative, by the variables ``by`` names, of what is ``written`` with a
        name nothing defines that an earlier equation ties to one of them, as
        ``\\frac{d}{dx} y`` and ``\\frac{d}{dx}(x y)`` after x^3 + y^3 = 6xy (guard 6), as
        ``\\frac{dy}{dx}`` is refused. Only the names written count, not those of the values
        of what they name, nor y of y(x), a function of x. A ``partial`` derivative of what
        is written with its variable holds the other names constant, as it says:
        ``\\frac{\\partial}{\\partial x}(x y)`` is y."""
        if not self.session.relations or not by:
            return
        symbols = set(_symbols_written(operand))
        names = {
            name: self._written_label(name)
            for name in _written_names(written, latex=ctx.latex)
            if name in symbols and name not in by and scope.lookup(name) is None
        }
        constant = not set(by) & _written_with(operand)
        if names and (constant or not partial):
            self._check_tied(names, by, span, constant=constant)

    @staticmethod
    def _applied(name: ast.Name, binding: FunctionBinding, node: ast.Derivative) -> ast.Expr:
        """A function written alone in a derivative, as f in \\frac{df}{dx}, applied to
        what it is differentiated by: f(x), or f(x, y) by its parameters' names."""
        params = binding.definition.params
        variables = [v for v, _ in node.variables if isinstance(v, ast.Name)]
        names = list(dict.fromkeys(v.id for v in variables))
        if len(params) == 1 and len(names) == 1:
            args: tuple[ast.Expr, ...] = (variables[0],)
        elif names and set(names) <= set(params):
            args = tuple(ast.Name(param, span=name.span) for param in params)
        else:
            return name
        return ast.Call(name, args, span=name.span)

    def _Differential(self, node: ast.Differential, scope: Scope, ctx: Context) -> Node:
        raise CompileError(
            "a differential d{...} needs a denominator, as in d{f}/d{x}", span=node.span
        )

    def _Integral(self, node: ast.Integral, scope: Scope, ctx: Context) -> Node:
        if node.domain is not None or any(bound.dimension for bound in node.bounds):
            raise CompileError(
                "integrals over regions of spacetime, as in d^4x, are not supported yet",
                span=node.span,
            )
        if node.span.text.lstrip().startswith(("\\oint", "∮")):  # \oint p\, dq
            raise self._closed_integral(node)
        inner = scope.child()
        limits = []
        for bound in node.bounds:
            symbol = self._bound_variable(bound.variable, inner)
            if bound.lower is None and bound.upper is None:
                limits.append(symbol)
                continue
            if bound.lower is None or bound.upper is None:
                raise CompileError("give both limits of a definite integral", span=bound.span)
            lower = self.value(bound.lower, scope, ctx)
            upper = self.value(bound.upper, scope, ctx)
            limits.append(self._call("core.limits", [symbol, lower, upper], VALUE, bound.span))
        integrand = self.value(node.integrand, inner, ctx)
        self._check_measure(node, integrand, scope)
        if isinstance(integrand.type, TensorType):
            return self.graph.add(
                ops.Map("integrate"), [integrand, *limits], integrand.type, node.span
            )
        return self._call("integrate", [integrand, *limits], _passed(integrand.type), node.span)

    def _BigOperator(self, node: ast.BigOperator, scope: Scope, ctx: Context) -> Node:
        if (each := self._term_by_term(node, scope, ctx)) is not None:  # \lambda_a, defined
            return each
        if node.lower is None or node.upper is None:
            hint = f"for example \\{node.op}_{{n=0}}^{{N}} a_n"
            label = node.variable.id if isinstance(node.variable, ast.Name) else None
            indexed = [  # the tensors that carry the index, as w in \sum_a w^a
                current
                for current in _walk(node.body)
                if isinstance(current, ast.Scripted)
                and (pattern := _tensor_pattern(current))
                and any(name == label for name, _ in pattern[1])
                and isinstance(scope.lookup(pattern[0]), TensorBinding)
            ]
            if node.op == "sum" and label is not None and indexed:  # \sum_a w^a
                pattern = _tensor_pattern(indexed[0])
                assert pattern is not None
                written = indexed[0].span.text
                ones = f"1_{label}" if dict(pattern[1])[label] else f"1^{label}"
                hint = (
                    f"to sum the components of {written}, contract it with the covector of "
                    f"ones: {written} {ones}"
                )
            raise CompileError(f"\\{node.op} needs limits", span=node.span, hint=hint)
        inner = scope.child()
        variable = self._bound_variable(node.variable, inner)
        lower = self.value(node.lower, scope, ctx)
        upper = self.value(node.upper, scope, ctx)
        body = self.value(node.body, inner, ctx)
        name = "core.sum" if node.op == "sum" else "core.prod"
        args = [body, variable, lower, upper]
        if isinstance(body.type, TensorType):  # component by component, as an integral is
            return self.graph.add(ops.Map(name), args, body.type, node.span)
        return self._call(name, args, _passed(body.type), node.span)

    def _Limit(self, node: ast.Limit, scope: Scope, ctx: Context) -> Node:
        inner = scope.child()
        variable = self._bound_variable(node.variable, inner)
        point = self.value(node.point, scope, ctx)
        body = self.value(node.body, inner, ctx)
        direction = self.graph.const(node.direction or "+-", VALUE)
        if isinstance(body.type, TensorType):
            return self.graph.add(
                ops.Map("core.limit"), [body, variable, point, direction], body.type, node.span
            )
        return self._call("core.limit", [body, variable, point, direction], SCALAR, node.span)

    def _bound_variable(self, variable: ast.Expr, scope: Scope) -> Node:
        """Bind a summation, integration or limit variable as a plain symbol."""
        if not isinstance(variable, ast.Name):
            raise CompileError("expected a variable", span=variable.span)
        symbol = self.graph.const(self.session.symbol(variable.id), SCALAR, variable.span)
        scope.bind(variable.id, VariableBinding(symbol))
        return symbol

    def _Root(self, node: ast.Root, scope: Scope, ctx: Context) -> Node:
        if node.degree is None:  # \sqrt{\Sigma}, of a tensor named without indices: a matrix's
            call = ast.Call(ast.Name("sqrt", span=node.span), (node.radicand,), span=node.span)
            if (root := self._linear_algebra("sqrt", call, scope, ctx)) is not None:
                return root
        radicand = self.value(node.radicand, scope, ctx)
        if node.degree is None:
            return self._builtin(BUILTINS["sqrt"], [radicand], node.span)
        degree = self.value(node.degree, scope, ctx)
        return self._builtin(BUILTINS["root"], [radicand, degree], node.span)

    def _Factorial(self, node: ast.Factorial, scope: Scope, ctx: Context) -> Node:
        return self._builtin(
            BUILTINS["factorial"], [self.value(node.operand, scope, ctx)], node.span
        )

    def _Abs(self, node: ast.Abs, scope: Scope, ctx: Context) -> Node:
        return self._absolute(node, self.value(node.operand, scope, ctx))

    def _absolute(self, node: ast.Abs, operand: Node) -> Node:
        """What bars around a value write: the length of a vector, the norm of a matrix, or
        the absolute value."""
        if isinstance(operand.type, TensorType) and operand.type.rank == 1:
            return self._norm(operand, node.span)
        if (norm := self._norm_of(node, operand)) is not None:  # \|A\| of a matrix
            return norm
        return self._builtin(BUILTINS["Abs"], [operand], node.span)

    def _norm(self, vector: Node, span: Span) -> Node:
        """``\\|v\\|``, the length of a vector with the metric of its space:
        :math:`\\sqrt{g_{ab} \\bar v^a v^b}`, with the conjugate of complex components."""
        assert isinstance(vector.type, TensorType)
        index = vector.type.indices[0]
        lowered, _ = self._move_index(vector, vector.type.indices, 0, span)
        twin = Indices([Index(index.name, not index.upper)])
        conjugate = self._conjugate(vector, span)
        square = self._contract([(conjugate, vector.type.indices), (lowered, twin)], span)
        square = self.graph.add(ops.Call("core.real"), (square,), SCALAR, span)
        half = self.graph.const(sympy.Rational(1, 2), SCALAR)
        return self.graph.add(ops.Pow(), (square, half), SCALAR, span)

    def _Substitution(self, node: ast.Substitution, scope: Scope, ctx: Context) -> Node:
        expr = self.value(node.expr, scope, ctx)
        pairs = []
        for target, value in node.bindings:
            pairs += [self.value(target, scope, ctx), self.value(value, scope, ctx)]
        return self._substituted(expr, pairs, node.span)

    def _substituted(self, expr: Node, pairs: list[Node], span: Span) -> Node:
        """``expr`` with every substitution made at once: in each component of a tensor,
        and in each side of an equation."""
        if isinstance(expr.op, ops.Equation):
            sides = [self._substituted(side, pairs, span) for side in expr.args]
            return self.graph.add(ops.Equation(), sides, expr.type, span)
        if isinstance(expr.type, TensorType):
            return self.graph.add(ops.Map("core.subs"), [expr, *pairs], expr.type, span)
        return self._call("core.subs", [expr, *pairs], _passed(expr.type), span)

    def _Evaluation(self, node: ast.Evaluation, scope: Scope, ctx: Context) -> Node:
        """An evaluation bar or bracket: what it evaluates at a point, or its value at the
        upper limit minus its value at the lower, in the variable written with them; else in
        that of an integral beside it in a sum; else in the one symbol what it evaluates is
        written with (see :mod:`relativisticpy.reading.evaluations`)."""
        written = evaluations.enclosed(node.expr)
        limits = [node.lower] if node.upper is None else [node.upper, node.lower]
        variable, note = node.variable, None
        why = f"its variable is written with its {'point' if node.upper is None else 'limits'}"
        if variable is None and (beside := self._integral_beside(node)) is not None:
            variable = beside
            why = (
                f"no variable is written, and {beside.span.text} is that of the integral beside it"
            )
        symbol: Node | None
        if isinstance(variable, ast.Name):
            inner = scope
            if scope.lookup(variable.id) is not None:  # as an integral's, its variable is its own
                inner = scope.child()
                symbol = self._bound_variable(variable, inner)
            else:
                symbol = self.graph.const(self.session.symbol(variable.id), SCALAR, variable.span)
            value = self.value(node.expr, inner, ctx)
            shown = variable.span.text
            others = [name for name in _symbols_written(value) if name != variable.id]
            if node.variable is None and others:
                other = _shown(others[0], ctx.latex)
                note = (
                    f"{node.span.text} is evaluated in {shown}, the variable of the integral "
                    f"beside it; for {other}, write it with the limits: "
                    f"{_with_variable(node, other)}"
                )
        elif variable is not None:
            value = self.value(node.expr, scope, ctx)
            symbol, shown = self.value(variable, scope, ctx), variable.span.text
        else:
            value = self.value(node.expr, scope, ctx)
            symbol, shown = self._evaluated_in(node, value, written, ctx)
            why = f"no variable is written, and {written} is written with {shown} alone"
            if symbol is None:
                why = f"{written} is written with no symbol, so it is the same at every point"
                note = (
                    f"{node.span.text} is the same wherever it is evaluated, as {written} is "
                    "written with no symbol; to evaluate it in a variable, write it with it: "
                    f"{_with_variable(node, 'x')}"
                )
        if len(limits) > 1 and isinstance(value.type, EquationType):
            raise CompileError(
                f"{node.span.text} takes the difference of the values of an equation at two "
                "points, which is no equation",
                span=node.span,
                hint="write the bar with \\left. and \\right| around what it evaluates, as in "
                f"\\left. ... \\right|{_with_variable(node, 'x')}",
            )
        if note is not None:
            self._note(note)
        at = [f"{shown} = {limit.span.text}" for limit in limits]
        reading = f"{written} at {at[0]}" + (f", minus at {at[1]}" if len(at) > 1 else "")
        self._decide(evaluations.EVALUATION, node.span, reading, why, note)
        values = [
            value
            if symbol is None
            else self._substituted(value, [symbol, self.value(limit, scope, ctx)], node.span)
            for limit in limits
        ]
        if len(values) == 1:
            return values[0]
        return self.graph.add(ops.Add((1, -1)), values, value.type, node.span)

    def _integral_beside(self, node: ast.Evaluation) -> ast.Name | None:
        """The variable of the integral written beside an evaluation in the innermost sum it
        is in, as x of ``\\left[x e^x\\right]_0^1 - \\int_0^1 e^x\\,dx``; ``None`` where there
        is none, or there are integrals in several variables."""
        sum_ = next(
            (s for s in reversed(self._sums) if evaluations.within(node.span, s.span)), None
        )
        if sum_ is None:
            return None
        variables = evaluations.integral_variables(sum_, node.span)
        return variables[0] if len(variables) == 1 else None

    def _evaluated_in(
        self, node: ast.Evaluation, value: Node, written: str, ctx: Context
    ) -> tuple[Node | None, str]:
        """The one symbol what an evaluation evaluates is written with, which it is in where
        no variable is written, and that symbol as written; ``None`` where it is written with
        none, and refused where it is written with several."""
        names = _symbols_written(value)
        shown = [_shown(name, ctx.latex) for name in names]
        if len(names) > 1:
            where = "its point" if node.upper is None else "the limits"
            raise CompileError(
                f"{node.span.text} does not say which variable it is evaluated in, and "
                f"{written} is written with {_listed(shown)}",
                span=node.span,
                hint=f"write the variable with {where}: {_with_variable(node, shown[0])}",
            )
        if not names:
            return None, ""
        return self.graph.const(self.session.symbol(names[0]), SCALAR, node.span), shown[0]

    def _Accent(self, node: ast.Accent, scope: Scope, ctx: Context) -> Node:
        if node.accent in ("dot", "ddot"):
            return self._dotted(node.base, node.accent, node.span, scope, ctx)
        if node.accent == "bar" and (conjugate := self._barred(node, scope, ctx)) is not None:
            return conjugate  # \bar{z} of z := 3 + 4i, and \overline{3 + 4i}
        if node.accent == "tilde" and (transposed := self._tilde_transpose(node, scope, ctx)):
            return transposed  # \tilde{T} of a matrix, as Goldstein writes it
        if not isinstance(node.base, ast.Name):
            raise CompileError(f"\\{node.accent} needs a single symbol", span=node.base.span)
        return self.value(
            ast.Name(accented(node.base.id, node.accent), verbatim=True, span=node.span), scope, ctx
        )

    def _dotted(
        self, dotted: ast.Expr, accent: str, span: Span, scope: Scope, ctx: Context
    ) -> Node:
        """``\\dot{x}``: the time derivative of x. Of a function the program defined of one
        variable, as ``a(t) := t^3``, it is the function's derivative by that variable."""
        binding = scope.lookup(dotted.id) if isinstance(dotted, ast.Name) else None
        if isinstance(binding, FunctionBinding) and len(binding.definition.params) == 1:
            (param,) = binding.definition.params
            why = f"{dotted.span.text} is a function of {param}, defined on line "
            why += str(binding.definition.span.line)
            reading = f"the derivative of {dotted.span.text}({param}) by {param}"
            self._decide("function named alone", span, reading, why)
            variable = self.graph.const(self.session.symbol(param), SCALAR, dotted.span)
            body = self._inline(binding, [variable], span, ctx)
            return self._differentiate(body, ((variable, 1 if accent == "dot" else 2),), span)
        return self._time_derivative(self.value(dotted, scope, ctx), accent, span)

    def _dotted_scripted(self, node: ast.Scripted, scope: Scope, ctx: Context) -> Node:
        """``\\dot{x}^\\mu`` or ``\\dot{a}^2``: an index or a subscript belongs to the dotted
        name, inside the derivative, and a power to the derivative, outside it, as in books:
        \\dot{a}^2 is (da/dt)^2, never d(a^2)/dt."""
        accent = node.base
        assert isinstance(accent, ast.Accent)
        name = accent.base
        inside = [s for s in node.scripts if self._scripts_the_dotted(s, name, scope)]
        powers = [s for s in node.scripts if s not in inside]
        inner = ast.Scripted(name, tuple(inside), span=node.span) if inside else name
        result = self._dotted(
            inner, accent.accent, node.span if inside else accent.span, scope, ctx
        )
        for script in powers:
            if script.body is None:
                raise CompileError("empty superscript", span=script.span)
            why = f"{script.body.span.text} is no index of {name.span.text}"
            reading = f"a power of {accent.span.text}"
            self._decide("index or power", node.span, reading, why)
            result = self._power(result, self.value(script.body, scope, ctx), node.span)
        return result

    def _scripts_the_dotted(self, script: ast.Script, name: ast.Expr, scope: Scope) -> bool:
        """Whether a script after a dotted name is the name's, as a subscript, or an index
        of a tensor, as \\mu in ``\\dot{x}^\\mu``, is; a power is the derivative's. A Greek
        label, or one the statement shows to be an index, stays an index, as before. So does
        everything on a symbol nothing defines, which is refused as before: its dot alone,
        \\dot\\theta, is a symbol of its own, which d/dt would take for a constant, as a
        Lagrangian's velocity is not yet (see curriculum/ROADMAP.md, wave 4)."""
        if script.kind == "_" or self._undefined(name, scope) or self._names_tensor(name, scope):
            return True
        items = script.indices.items if script.indices is not None else ()
        if not items or not all(_plain_label(item) for item in items):
            return False
        greek = any(str(item.label) in GREEK for item in items)
        return greek or self._index_written(script, scope) is not None

    def _time_derivative(self, value: Node, accent: str, span: Span) -> Node:
        order = self.graph.const(1 if accent == "dot" else 2, VALUE)
        if isinstance(value.type, TensorType):
            return self.graph.add(ops.Map("core.time_derivative"), (value, order), value.type, span)
        return self._call("core.time_derivative", [value, order], _passed(value.type), span)

    def _List(self, node: ast.List, scope: Scope, ctx: Context) -> Node:
        items = [self.value(item, scope, ctx) for item in node.items]
        if len(items) == 2 and all(_a_matrix(item) for item in items):  # [K_1, K_2]
            raise self._matrices_in_brackets(node, node.items)
        return self._call("core.list", items, VALUE, node.span)

    def _Tuple(self, node: ast.Tuple, scope: Scope, ctx: Context) -> Node:
        items = [self.value(item, scope, ctx) for item in node.items]
        bracketed = node.span.text.startswith(_MATRIX_BRACKETS)
        if bracketed and len(items) == 2 and all(_a_matrix(item) for item in items):
            raise self._matrices_in_brackets(node, node.items)  # \left[ A, B \right]
        return self._call("core.tuple", items, VALUE, node.span)

    def _Matrix(self, node: ast.Matrix, scope: Scope, ctx: Context) -> Node:
        cells = [self.value(cell, scope, ctx) for row in node.rows for cell in row]
        shape = (len(node.rows), len(node.rows[0]))
        array = self.graph.add(ops.MakeArray(shape), cells, VALUE, node.span)
        if node.environment == "vmatrix":
            return self._builtin(BUILTINS["det"], [array], node.span)
        return array

    def _Piecewise(self, node: ast.Piecewise, scope: Scope, ctx: Context) -> Node:
        args = []
        for value, condition in node.pieces:
            args.append(self.value(value, scope, ctx))
            args.append(
                self.value(condition, scope, ctx)
                if condition is not None
                else self.graph.const(sympy.true, SCALAR)
            )
        return self._call("core.piecewise", args, SCALAR, node.span)

    def _Prime(self, node: ast.Prime, scope: Scope, ctx: Context) -> Node:
        """A primed name written alone. Of a function written once with its one argument,
        as A(r) is, it is the derivative there: A' is dA/dr. Otherwise it is a name of its
        own, as t' is in \\int f(t')\\, dt', or x' in a change of coordinates."""
        name = node.func.id
        primed = name + "'" * node.order
        if isinstance(binding := scope.lookup(primed), VariableBinding):
            return binding.node
        if isinstance(binding, FunctionBinding):  # \rho' of \rho'(x, t) := ...
            return self._function_alone(primed, node, binding, scope, ctx)
        binding = scope.lookup(name)
        if isinstance(binding, FunctionBinding) and len(binding.definition.params) == 1:
            return self._derivative_alone(node, binding, scope, ctx)  # y'' of y(x) := ...
        arguments = self.session.functions.get(name, ())
        if binding is None and len(arguments) == 1:
            at = ast.Name(str(arguments[0]), verbatim=True, span=node.span)
            return self._prime(node, (at,), node.span, scope, ctx)
        if binding is not None or _names_builtin(node.func, latex=ctx.latex):
            raise CompileError(f"{primed} needs an argument, as in {primed}(x)", span=node.span)
        return self.graph.const(self.session.symbol(primed), SCALAR, node.span)

    def _Primed(self, node: ast.Primed, scope: Scope, ctx: Context) -> Node:
        """``(x^3)'``: the derivative of what the brackets hold, by the one variable it is
        written with, as ``f'`` is of a function of one variable, with a note that says
        which; of what is written with no variable, 0."""
        value = self.value(node.operand, scope, ctx)
        variable = self._variable_of(node, value)
        return self._differentiate(value, [(variable, node.order)], node.span)

    def _primed_at(
        self, node: ast.Primed, args: tuple[ast.Expr, ...], span: Span, scope: Scope, ctx: Context
    ) -> Node:
        """``(x^3)'(2)``: the derivative of what the brackets hold, by its one variable,
        at the point in the brackets after it."""
        if len(args) != 1:
            raise CompileError(f"{node.span.text}(x) takes one argument", span=span)
        value = self.value(node.operand, scope, ctx)
        variable = self._variable_of(node, value)
        point = self.value(args[0], scope, ctx)
        order = self.graph.const(node.order, VALUE)
        return self._call("core.derivative_at", [value, variable, order, point], SCALAR, span)

    def _variable_of(self, node: ast.Primed, value: Node) -> Node:
        """The variable primes after brackets differentiate by: the one symbol what the
        brackets hold is written with. Refused where it is written with more than one, as
        ``(xy)'`` is, since the prime does not say which."""
        names = _symbols_written(value)
        written = node.span.text
        if len(names) > 1:
            shown = [self._written_label(name) for name in names]
            raise CompileError(
                f"{written} is written with {_listed(shown)}, so its prime does not say "
                "which variable it is the derivative by",
                span=node.span,
                hint=f"write the variable, as in \\frac{{d}}{{d{shown[0]}}}"
                f"{node.operand.span.text}",
            )
        if not names:  # (3)': of a constant, 0 by any variable
            return self.graph.const(sympy.Dummy("x"), SCALAR)
        (name,) = names
        symbol = next(
            each
            for found in dependencies([value])
            if isinstance(found.op, ops.Const) and isinstance(found.op.value, sympy.Basic)
            for each in found.op.value.free_symbols
            if isinstance(each, Symbol) and each.name == name
        )
        self._note(
            f"{written} is the derivative by {self._written_label(name)}, the one variable "
            "it is written with"
        )
        return self.graph.const(symbol, SCALAR, node.span)

    def _Scripted(self, node: ast.Scripted, scope: Scope, ctx: Context) -> Term:
        if (book := self._book_scripted(node, scope)) is not None:  # (a)_n
            return self._capability(book, node.span, scope, ctx)
        if (raised := self._raised_product(node, scope, ctx)) is not None:  # \pi(x + 1)^2
            return raised
        if (matrix := self._matrix_scripted(node, scope, ctx)) is not None:  # A^T, \|A\|_2
            return matrix
        if (labelled := self._labelled(node, scope, ctx)) is not None:  # \lambda_1, R_\odot
            return labelled
        if (conjugate := self._starred_conjugate(node, scope, ctx)) is not None:  # f(x)^*, z^*
            return conjugate
        every = isinstance(node.base, ast.Prime) and self._a_primed_tensor(node.base, scope)
        if isinstance(node.base, ast.Prime) and isinstance(
            named := _unprimed(node, every=every), ast.Scripted
        ):
            node = named  # R'_{\mu\nu}, or k'^\mu of k'^\mu := ...: the tensor R'
        decorated = _named(node)
        if isinstance(decorated, ast.Name):  # f^{(4)} of a function f: no name of its own
            self._check_order(node, scope)
        if isinstance(decorated, ast.Name) and all(
            not isinstance(s.body, ast.Name) or not _bracketed_order(s.body) for s in node.scripts
        ):  # L^{(2)} or r^*, a name of its own even where L is a tensor; V^{(a)} is an index
            return self.term(decorated, scope, ctx)
        base = node.base
        if isinstance(base, ast.Operator) and base.kind == "lie":
            return self._lie_operator(node)
        if isinstance(base, ast.Operator) and base.kind == "box":
            raise CompileError("\\Box takes no indices", span=node.span)
        if isinstance(base, ast.Operator):
            return self._derivative_operator(node, covariant=base.kind == "nabla")
        if isinstance(base, ast.Accent) and base.accent in ("dot", "ddot"):
            return self._dotted_scripted(node, scope, ctx)
        if (inverse := self._inverse_access(node, scope, ctx)) is not None:
            return inverse
        if (result := self._indexed_result(node, scope, ctx)) is not None:
            return result
        if isinstance(base, ast.Name):
            name = base.id
            binding = scope.lookup(name)
            names = self.session.names
            if (
                binding is None
                and name in (names["partial"], names["covariant"])
                and all(s.indices is not None and s.indices.items for s in node.scripts)
                and self._differentiates(node)
            ):
                return self._derivative_operator(node, covariant=name == names["covariant"])
            rank = _rank(node)
            labels_only = rank is not None and all(
                _plain_label(item)
                for s in node.scripts
                for item in s.indices.items  # type: ignore[union-attr]
            )
            source = None
            if binding is None and labels_only and (pattern := _tensor_pattern(node)):
                source = self._space_tensor(name, pattern[1], node.span)
            if source is None:  # R_{(\mu\nu\rho)} is checked as R_{\mu\nu\rho} is
                strict = rank is not None and all(
                    item.label is not None and item.value is None and item.derivative is None
                    for s in node.scripts
                    for item in s.indices.items  # type: ignore[union-attr]
                )
                source = self._tensor_source(
                    name, binding, rank, node.span, strict=strict, node=node
                )
            differentiated = _has_derivative(node)
            if source is None and rank == 0 and differentiated:  # \phi_{;\mu}
                source = self._scalar_source(base, binding, scope, ctx)
            if source is not None:
                self._check_component(node, base, source)
                why = f"{base.span.text} is {self._tensor_meaning(name, binding, rank)}"
                self._decide("index or power", node.span, "a tensor with its indices", why)
                return self._access(source, node, scope, ctx)
            if differentiated:  # never a symbol named phi_mu
                raise self._underivable(node, base, binding, rank)
            if rank is None:  # g_{\mu\nu}^{-1}: a tensor with its indices, then a power
                self._check_tensor_power(node, base, binding)
        if (ones := _tensor_pattern_of_one(node)) is not None:  # 𝟙_a, or 1_a
            return self._ones(ones, node.span)
        return self._script_math(node, scope, ctx)

    def _inverse_access(self, node: ast.Scripted, scope: Scope, ctx: Context) -> Node | None:
        """``(\\Sigma^{-1})^{ab}``: the inverse of a two-index tensor, written with indices
        of its own, in the positions opposite to the tensor's."""
        base = node.base
        if not (
            isinstance(base, ast.Scripted)
            and len(base.scripts) == 1
            and base.scripts[0].kind == "^"
            and _is_minus_one(base.scripts[0].body)
            and isinstance(base.base, ast.Name)
        ):
            return None
        matrix = self.value(base.base, scope, ctx)
        if not isinstance(matrix.type, TensorType) or matrix.type.rank != 2:
            return None
        inverse = self._inverse(matrix, base.span)
        assert isinstance(inverse.type, TensorType)
        source = TensorSource(inverse, inverse.type.indices, f"{base.base.id}^{{-1}}")
        return self._access(source, node, scope, ctx)

    def _indexed_result(self, node: ast.Scripted, scope: Scope, ctx: Context) -> Node | None:
        """``\\operatorname{diag}(k)_{ij}``: the tensor a call gives, with indices written
        on it, by position. ``f(x)^2`` stays a power: its script is not index labels."""
        base = node.base
        plain = all(
            script.indices is not None and all(_plain_label(item) for item in script.indices.items)
            for script in node.scripts
        )
        if not (plain and isinstance(base, ast.Call) and self._names_tensor(base, scope)):
            return None
        value = self.value(base, scope, ctx)
        if not isinstance(value.type, TensorType):
            return None
        return self._access(
            TensorSource(value, value.type.indices, base.span.text), node, scope, ctx
        )

    def _differential(self, operand: Node, span: Span) -> Node:
        """``d`` applied to a coordinate in a line element: ``d\\theta^2`` is ``(dθ)²``."""
        coordinates = self.session.coordinates or ()
        base, power = operand, None
        if isinstance(operand.op, ops.Pow):
            base, power = operand.args
        symbol = base.op.value if isinstance(base.op, ops.Const) else None
        if symbol in coordinates:
            result = self.graph.const(differential(symbol), SCALAR, span)
        elif isinstance(symbol, Symbol) and symbol.name == "Omega":
            result = self._solid_angle(coordinates, span)
        else:
            raise CompileError("in a line element, d must be followed by a coordinate", span=span)
        if power is not None:
            result = self.graph.add(ops.Pow(), (result, power), SCALAR, span)
        return result

    def _solid_angle(self, coordinates: Sequence[Symbol], span: Span) -> Node:
        r"""The solid angle element :math:`d\Omega = \sqrt{d\theta^2 + \sin^2\theta\, d\phi^2}`,
        for spherical coordinates."""
        names = {c.name: c for c in coordinates}
        if "theta" not in names or "phi" not in names:
            raise CompileError("d\\Omega needs coordinates named \\theta and \\phi", span=span)
        theta, phi = names["theta"], names["phi"]
        element = sympy.sqrt(
            differential(theta) ** 2 + sympy.sin(theta) ** 2 * differential(phi) ** 2
        )
        return self.graph.const(element, SCALAR, span)


#: The decision of whether a name before brackets is called or multiplies.
CALL = "product or call"

_COMPARE = {
    "<": "core.lt",
    ">": "core.gt",
    "<=": "core.le",
    ">=": "core.ge",
    "==": "core.eq",
    "!=": "core.ne",
}

#: The operations whose value is a relation, true or false.
_RELATIONS = frozenset({*_COMPARE.values(), "core.and", "core.or", "core.not"})

#: The comparisons that order what they compare, which no tuple or matrix is.
_ORDERS = frozenset({"<", ">", "<=", ">="})

#: What a list of names before a relation, as a, b in ``a, b > 0``, does not hold: a number,
#: or a tuple, a list or a matrix written out, as 1 and 2 in ``(1, 2) < 3``.
_WRITTEN_OUT = (ast.Number, ast.Tuple, ast.List, ast.Matrix)


def _a_relation(value: Node) -> bool:
    """Whether a value is a relation, true or false, as ``X < 0`` and ``x = 2`` are, known
    from how it is written."""
    if isinstance(value.op, ops.Equation):
        return True
    return isinstance(value.op, ops.Call) and value.op.name in _RELATIONS


def _arithmetic(node: Node, numbers: frozenset[str]) -> bool:
    """Whether an operation computes with numbers: arithmetic, or one of the functions
    called ``numbers``, as the elementary functions are."""
    op = node.op
    if isinstance(op, ops.Add | ops.Mul | ops.Pow | ops.Scale):
        return True
    return isinstance(op, ops.Call | ops.Map) and op.name in numbers


def _random(value: Node) -> bool:
    """Whether a value is written with a random variable, as ``X < 0`` is, of X ~ N(0, 1)."""
    return any(
        isinstance(node.op, ops.Call) and node.op.name == "prob.variable"
        for node in dependencies([value])
    )


def _unordered(value: Node) -> str | None:
    """What a value is, where it is a tuple or a matrix, known without computing it, of
    which an inequality fails: ``a tuple``; ``None`` for anything else, as a number, or a
    list, whose inequality says that it is not defined."""
    if _a_matrix(value):
        return "a matrix"
    if isinstance(value.op, ops.Call) and value.op.name == "core.tuple":
        return "a tuple"
    return None


_INTEGER = re.compile(r"\d+")

#: The brackets that write a commutator or an anticommutator of two matrices.
_MATRIX_BRACKETS = ("[", "\\left[", "\\big[", "\\Big[", "\\{", "\\left\\{", "\\lbrace")


def _of_numbers() -> frozenset[str]:
    """The names of the elementary functions, which take numbers: a pair given to one, as
    to the square root in \\sqrt{(a, b)}, is a pair where none was meant. The modulus is
    not one of them, as |v| of a vector is its length."""
    return frozenset(
        name
        for capability in integrations.capabilities()
        if capability.module.endswith(".elementary")
        for name in capability.names
    ) - {"abs", "Abs"}


def _written_names(written: Sequence[ast.Expr | Node], *, latex: bool) -> list[str]:
    """The names what is ``written`` is written with, each once, read from its syntax: y
    of ``(x y)``, x and y of ``xy`` in LaTeX, where letters side by side multiply, and y_1
    of ``y_1``; none of a factor that is compiled already."""
    found: dict[str, None] = {}
    parts = [part for part in written if isinstance(part, ast.Expr)]
    for node in itertools.chain.from_iterable(map(_walk, parts)):
        if isinstance(node, ast.Name) and not node.verbatim:
            found[node.id] = None
            if latex and _splittable(node):
                found.update(dict.fromkeys(split_index_labels(node.id)))
        elif (
            isinstance(node, ast.Scripted)
            and isinstance(node.base, ast.Name)
            and all(script.kind == "_" for script in node.scripts)
        ):
            found[_subscripted_name(node)] = None
    return list(found)


def _unsigned(factor: ast.Expr) -> list[ast.Expr]:
    """Take the sign off a factor, so that ``-\\partial_\\mu \\phi`` and ``-d t^2`` let
    the operator act on what follows: ``-x`` becomes ``-1``, ``x``."""
    if isinstance(factor, ast.Unary) and factor.op == "-":
        return [ast.Number("-1", span=factor.span), *_unsigned(factor.operand)]
    return [factor]


def _with_variable(node: ast.Evaluation, variable: str) -> str:
    """The scripts of an evaluation, written with a variable, as ``_{x=0}^{x=1}``."""
    lower = f"_{{{variable}={node.lower.span.text}}}"
    return lower if node.upper is None else f"{lower}^{{{variable}={node.upper.span.text}}}"


def _group(node: ast.Call) -> ast.Expr:
    """What the brackets of a call hold, as one operand: its argument, or the pair of its
    arguments."""
    return node.args[0] if len(node.args) == 1 else ast.Tuple(node.args, span=node.span)


def _bracketed(node: ast.Call) -> ast.Expr:
    """What the brackets of a call hold, as :func:`_group` gives it, written in them: with
    their span, so that ``(ab)`` is a run of letters in brackets."""
    span = Span(node.span.source, node.func.span.end, node.span.end)
    return replace(_group(node), span=span)


def _letters_called(node: ast.Call, labels: Sequence[str]) -> list[ast.Expr]:
    """The factors of a call of a run of letters that names nothing, in LaTeX, of which
    ``labels`` are the letters: each but the last, then the last called, as G and m(r) are
    of ``Gm(r)``."""
    func = node.func
    assert isinstance(func, ast.Name)
    spans = [func.span] * len(labels)
    if func.span.text == "".join(labels):  # each letter where it is written
        ends = list(itertools.accumulate(len(label) for label in labels))
        spans = [
            Span(func.span.source, func.span.start + end - len(label), func.span.start + end)
            for label, end in zip(labels, ends, strict=True)
        ]
    leading = [
        _letter(label, span, first=n == 0)
        for n, (label, span) in enumerate(zip(labels[:-1], spans, strict=False))
    ]
    last = _letter(labels[-1], spans[-1], first=False)
    called = Span(node.span.source, spans[-1].start, node.span.end)
    return [*leading, replace(node, func=last, span=called)]


def _a_power(script: ast.Script) -> bool:
    """Whether a script raises what it is written on to a power, as ``^2`` and ``^{-1}``
    do: a superscript that is no mark, as ``^*`` and ``^{(2)}`` are, nor a transpose, as
    ``^T`` is."""
    body = script.body
    if script.kind != "^" or body is None or _decoration(script):
        return False
    return not (isinstance(body, ast.Name) and body.id in (*_TRANSPOSES, "dagger"))


#: The plain-text calls of a sum and a product, by SymPy's names and Python's: ``sum(f, (n,
#: a, b))`` is ``\sum_{n=a}^{b} f`` and ``product(f, (k, a, b))`` is ``\prod_{k=a}^{b} f``.
_BIG_OPERATOR_CALLS: Mapping[str, Literal["sum", "prod"]] = {
    "sum": "sum",
    "summation": "sum",
    "product": "prod",
}


def _big_operator_call(node: ast.Call) -> ast.BigOperator:
    """``sum(f, (n, a, b))``, ``summation(...)`` or ``product(...)``, a sum or a product as
    SymPy writes one, as the LaTeX it stands for, so that each is read as
    ``\\sum_{n=a}^{b} f`` is (design/notation.md, decision 40). With the limits of several
    indices, the first is the innermost, as in SymPy. Anything else is refused."""
    assert isinstance(node.func, ast.Name)
    name, op = node.func.id, _BIG_OPERATOR_CALLS[node.func.id]
    body, *limits = node.args or (None,)
    written = [
        limit.items
        for limit in limits
        if isinstance(limit, ast.Tuple)
        and len(limit.items) == 3
        and isinstance(limit.items[0], ast.Name)
    ]
    if body is None or not limits or len(written) != len(limits):
        index = "k" if op == "prod" else "n"
        what = "multiplies" if op == "prod" else "sums"
        raise CompileError(
            f"{name} takes what it {what}, then its index and the index's limits, in brackets",
            span=node.span,
            hint=f"for example {name}(1/{index}^2, ({index}, 1, oo))",
        )
    result: ast.Expr = body
    for variable, lower, upper in written:
        result = ast.BigOperator(op, variable, lower, upper, result, span=node.span)
    assert isinstance(result, ast.BigOperator)
    return result
