r"""The lowering of vectors and of vector calculus, as :mod:`relativisticpy.reading.vectors`
reads them (design/notation.md, 8, 9, 16 and 25):

- the algebra of vectors, component by component: ``\mathbf{v} \cdot \mathbf{w}``,
  ``\mathbf{v} \times \mathbf{w}``, ``|\mathbf{v}|`` and ``\|\mathbf{v}\|``, and
  ``\mathbf{p}^2``, which is \mathbf{p} \cdot \mathbf{p};
- the calculus of vectors, component by component: ``\frac{d\mathbf{r}}{dt}``,
  ``\partial_\theta \mathbf{x}``, ``\mathbf{r}'(t)``, ``\int \mathbf{v}\,dt``, an
  evaluation bar and a limit; a field at a point, ``\mathbf{v}(\mathbf{r}(t))``, and a
  function of the components of a vector, ``f(\mathbf{x})``;
- the operators of vector calculus, of the frame in force: ``\nabla f``,
  ``\nabla \cdot \mathbf{v}``, ``\nabla \times \mathbf{v}``, ``\nabla^2 f`` and ``\Delta
  f``, and ``(\mathbf{u} \cdot \nabla)``, which acts on what follows it; of a function at
  a point, ``\nabla f(\mathbf{a})`` is its gradient there, and ``\nabla^2 f(\mathbf{a})``
  its Hessian (decision 25);
- an equation between vectors, decided component by component.

``\cdot`` and ``\times`` are products of vectors only between vectors: between numbers,
or a number and a vector, they multiply, as before.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Sequence
from dataclasses import dataclass

import sympy

from relativisticpy.compiler.passes import Term
from relativisticpy.compiler.scope import (
    Context,
    FunctionBinding,
    FunctionReference,
    Operator,
    Scope,
    VariableBinding,
)
from relativisticpy.compiler.session import Session
from relativisticpy.errors import CompileError, RelativisticPyError
from relativisticpy.ir import SCALAR, VALUE, Node, TensorType, Type, ops
from relativisticpy.language import syntax as ast
from relativisticpy.language.source import Span
from relativisticpy.lowering.probability import Probability
from relativisticpy.reading.families import Family
from relativisticpy.reading.labels import label_name
from relativisticpy.reading.names import _a_pair, _named, _written_as_letter, _written_with
from relativisticpy.reading.quantum import OperatorBinding
from relativisticpy.reading.vectors import (
    BOLD,
    CARTESIAN,
    CURVILINEAR,
    VECTOR,
    Frame,
    Vectors,
    bold,
    hatted,
)
from relativisticpy.tensors import Index, Indices
from relativisticpy.tensors.spaces import Space
from relativisticpy.typing.syntax import _operand, _plain_label, _signed_terms, _walk
from relativisticpy.typing.typer import TensorSource

#: The operations that act on each component of a vector they are given first.
_EACH_COMPONENT = frozenset(
    {
        "core.derivative_at",
        "core.time_derivative",
        "core.subs",
        "core.limit",
        "core.sum",
        "integrate",
        "vectors.along",
    }
)


@dataclass(frozen=True)
class Del(Operator):
    """``\\nabla`` with nothing after it, as in ``(\\mathbf{u} \\cdot \\nabla)``: what a
    vector before it makes the derivative along it."""


@dataclass(frozen=True)
class Directional(Operator):
    """``(\\mathbf{a} \\cdot \\nabla)``, the derivative along ``a``, which acts on the
    factor after it."""

    along: Node | None = None


@dataclass(frozen=True)
class _Along:
    """``\\nabla \\mathbf{u}`` after a vector and a dot, as in ``\\mathbf{u} \\cdot \\nabla
    \\mathbf{u}``: the derivative of ``vector`` along the vector before it."""

    vector: Node
    span: Span


#: What a product of vectors is made of: what is written, a value, an operator, or the
#: derivative of a vector waiting for the vector it is along.
_Item = ast.Expr | Node | Operator | _Along


@dataclass(frozen=True)
class _Curve:
    """A curve an integral is along: the ``name`` its function is bound to, as written,
    the symbol of its parameter, its point and its tangent there, and the limits of its
    parameter."""

    name: str
    written: str
    parameter: Node
    point: Node
    tangent: Node
    limits: tuple[Node, Node]


class VectorCalculus(Vectors, Probability):
    """The lowering of vectors, their algebra and their calculus."""

    def __init__(self, session: Session) -> None:
        super().__init__(session)
        # The hats written under scripts, as {\hat\theta}^{(\mu)} is: names, as before,
        # and no unit vectors, by the identity of their syntax.
        self._hats_named: set[int] = set()
        # The tangents of the curves an integral is along, while its integrand is read, by
        # the names of the curves, and the innermost under "": d\mathbf{r} is r'(t) dt.
        self._differentials: dict[str, Node] = {}
        self._differentials_read = 0

    # -- products: \cdot, \times and \nabla ------------------------------------------------

    def _product(self, node: ast.Binary, scope: Scope, ctx: Context) -> Term:
        written = _written_factors(node)
        if (vector := self._vector_product(node, written, scope, ctx)) is not None:
            return vector
        return super()._product(node, scope, ctx)

    def _vector_product(
        self,
        node: ast.Binary,
        written: list[tuple[str, ast.Expr]],
        scope: Scope,
        ctx: Context,
    ) -> Term | None:
        """A product written with ``\\nabla``, or with ``\\cdot`` or ``\\times`` between
        vectors: ∇ acts on the factor after it, by the sign between them, ``\\times`` binds
        before ``\\cdot``, and factors written side by side multiply. ``None`` for any other
        product, which is read as before."""
        self._check_delta_of_a_vector(written, scope)
        operators = any(
            self._acting(written, position, scope, ctx) or _an_operator_in_brackets(factor)
            for position, (_, factor) in enumerate(written)
        )
        along = self._with_differentials(written, scope)
        operators = operators or len(along) != len(written)
        signs = any(joint for joint, _ in written)
        if not operators and not signs:
            return None
        try:
            items = self._nablas_applied(along, scope, ctx)
            runs = self._runs(items, scope, ctx)
        except RelativisticPyError:
            if operators:
                raise
            return None
        vectors = any(
            isinstance(value, Operator | _Along) or self._vector_components(value) is not None
            for _, value in runs
            if isinstance(value, Node | Operator | _Along)
        )
        if not operators and not vectors:
            return None
        return self._signed(runs, node.span)

    def _nabla_kind(self, factor: ast.Expr, scope: Scope, ctx: Context) -> str | None:
        """What operator of vector calculus a factor is: ``\\nabla`` alone, ``\\nabla^2``
        where there is a frame of vector calculus, or ``\\Delta`` where nothing defines it;
        ``None`` for any other factor, as ``\\nabla_\\mu``."""
        if isinstance(factor, ast.Operator) and factor.kind == "nabla":
            return "∇"
        if (
            isinstance(factor, ast.Scripted)
            and isinstance(factor.base, ast.Operator)
            and factor.base.kind == "nabla"
            and len(factor.scripts) == 1
            and factor.scripts[0].kind == "^"
            and isinstance(factor.scripts[0].body, ast.Number)
            and factor.scripts[0].body.text == "2"
        ):
            return "∇²"
        if (
            isinstance(factor, ast.Name)
            and factor.id == "Delta"
            and _written_as_letter(factor)
            and not ctx.line_element
            and self._delta_names_nothing(scope)
        ):
            return "Δ"
        return None

    def _acting(
        self, written: list[tuple[str, ast.Expr]], position: int, scope: Scope, ctx: Context
    ) -> bool:
        """Whether the factor at ``position`` is an operator of vector calculus that acts:
        ∇ or ∇², or \\Delta before a function, not before a name, as in \\Delta x."""
        kind = self._nabla_kind(written[position][1], scope, ctx)
        if kind != "Δ":
            return kind is not None
        after = written[position + 1][1] if position + 1 < len(written) else None
        return after is not None and _a_function_written(after, scope)

    def _delta_names_nothing(self, scope: Scope) -> bool:
        """Whether \\Delta means nothing here, as it does in Kerr's metric, so that before a
        function it is the Laplacian (decision 25 of design/notation.md)."""
        session = self.session
        if scope.lookup("Delta") is not None or "Delta" in session.assumptions:
            return False
        metric = session.metric is not None and "Delta" in session.metric.names
        return not metric and "Delta" not in session.functions

    def _with_differentials(
        self, written: list[tuple[str, ast.Expr]], scope: Scope
    ) -> list[tuple[str, _Item]]:
        """The factors of a product with ``d`` before a curve the integral is along, as in
        ``d\\mathbf{r}_1``, or before a name that names nothing, as ``d\\mathbf{r}`` in
        ``\\oint_C \\mathbf{F} \\cdot d\\mathbf{r}``, read as the curve's tangent, its
        differential by its parameter, whose own differential the integral takes."""
        found: list[tuple[str, _Item]] = []
        position = 0
        while position < len(written):
            joint, factor = written[position]
            following = written[position + 1] if position + 1 < len(written) else None
            tangent = None
            if (
                self._differentials
                and following is not None
                and not following[0]
                and isinstance(factor, ast.Name)
                and factor.id == "d"
                and scope.lookup("d") is None
            ):
                tangent = self._tangent_written(following[1], scope)
            if tangent is not None:
                found.append((joint, tangent))
                self._differentials_read += 1
                position += 2
                continue
            found.append((joint, factor))
            position += 1
        return found

    def _tangent_written(self, written: ast.Expr, scope: Scope) -> Node | None:
        """The tangent of the curve a name after ``d`` is, by its name, or of the innermost
        curve, where the name names nothing else."""
        if isinstance(written, ast.Scripted):
            name = label_name(written)
        elif isinstance(written, ast.Name):
            name = written.id
        else:
            return None
        if name is not None and name in self._differentials:
            return self._differentials[name]
        if name is not None and scope.lookup(name) is None:
            return self._differentials.get("")
        return None

    def _nablas_applied(
        self, written: Sequence[tuple[str, _Item]], scope: Scope, ctx: Context
    ) -> list[tuple[str, _Item]]:
        """The factors of a product with each ∇ applied to the factor after it, from the
        right, so that ``\\nabla \\times \\nabla f`` is the curl of the gradient; a ∇ with
        nothing after it stays an operator, as in ``(\\mathbf{u} \\cdot \\nabla)``."""
        done: list[tuple[str, _Item]] = []
        for joint, factor in reversed(written):
            if not isinstance(factor, ast.Expr):  # a curve's tangent, as d\mathbf{r} is
                done.append((joint, factor))
                continue
            kind = self._nabla_kind(factor, scope, ctx)
            if kind == "Δ" and not (done and _a_function_written(done[-1][1], scope)):
                kind = None  # \Delta x, the change in x
            if kind is None:
                done.append((joint, factor))
                continue
            if not done:
                if kind != "∇":
                    raise CompileError(
                        f"{factor.span.text} needs something to act on, as in {factor.span.text} f",
                        span=factor.span,
                    )
                done.append((joint, Del(factor.span)))
                continue
            sign, operand = done.pop()
            applied = self._nabla_of(kind, factor, sign, operand, joint, scope, ctx)
            done.append((joint, applied))
        return done[::-1]

    def _nabla_of(
        self,
        kind: str,
        nabla: ast.Expr,
        sign: str,
        operand: _Item,
        before: str,
        scope: Scope,
        ctx: Context,
    ) -> _Item:
        """∇ applied, by the sign written between it and its operand: the gradient, the
        divergence or the curl; ∇² and Δ, the Laplacian or, of a function at a point, the
        Hessian."""
        span = nabla.span.to(_span_of(operand, nabla.span))
        if isinstance(operand, Operator | _Along):
            raise CompileError(
                f"{nabla.span.text} acts on a value, not on {_written(operand)}",
                span=span,
            )
        if kind != "∇":
            if sign:
                raise CompileError(
                    f"{nabla.span.text} {sign} has no meaning: {nabla.span.text} is the "
                    "Laplacian, which acts on what follows it",
                    span=span,
                )
            return self._laplacian(kind, nabla, operand, span, scope, ctx)
        if sign == "·":
            return self._divergence(operand, span, scope, ctx)
        if sign == "×":
            return self._curl(operand, span, scope, ctx)
        return self._gradient(operand, span, scope, ctx, after_a_dot=before == "·")

    # -- the operators -----------------------------------------------------------------------

    def _operand_value(self, operand: _Item, scope: Scope, ctx: Context) -> Node:
        if isinstance(operand, Node):
            return operand
        assert isinstance(operand, ast.Expr)
        return self.value(_operand(operand), scope, ctx)

    def _frame_arguments(self, frame: Frame) -> list[Node]:
        return [frame.metric] if frame.metric is not None else []

    def _gradient(
        self, operand: _Item, span: Span, scope: Scope, ctx: Context, *, after_a_dot: bool
    ) -> _Item:
        """``\\nabla f``: the gradient of a function, of the frame in force, or of the
        variables a function is written in where there are no coordinates; of a function
        the program defines at a point, its gradient there, as a column (decision 25)."""
        at = self._function_at_a_point(operand, scope, ctx)
        if at is not None:
            return self._derivatives_at(at, span, ctx, hessian=False)
        frame = self._frame_of(operand, span, scope)
        value = self._operand_value(operand, scope, ctx)
        if self._vector_components(value) is not None:
            if after_a_dot:
                return _Along(value, span)
            raise CompileError(
                f"{span.text} is the gradient of a vector, a tensor of two indices, which is "
                "not taken yet",
                span=span,
                hint="for the derivative along a vector, write (\\mathbf{u} \\cdot \\nabla)"
                "\\mathbf{v}",
            )
        self._check_a_function(value, operand, frame, span, "gradient")
        why = f"∇ is followed by a function, with no sign between them, in {frame.written}"
        self._decide(VECTOR, span, "its gradient", why)
        attributes = (("coordinates", frame.coordinates),)
        args = [value, *self._frame_arguments(frame)]
        return self._call("vectors.gradient", args, VALUE, span, attributes)

    def _divergence(self, operand: _Item, span: Span, scope: Scope, ctx: Context) -> Node:
        """``\\nabla \\cdot \\mathbf{v}``: the divergence of a vector of the frame in force."""
        value = self._operand_value(operand, scope, ctx)
        components = self._vector_components(value)
        if components is None:
            raise self._not_a_vector(operand, value, span, "the divergence")
        frame = self._frame_of_vector(components, span)
        self._check_written_in_the_frame(components, operand, frame, span, "divergence")
        why = f"∇ is followed by \\cdot and a vector, in {frame.written}"
        self._decide(VECTOR, span, "its divergence", why)
        attributes = (("coordinates", frame.coordinates),)
        args = [self._vector_of(components, span), *self._frame_arguments(frame)]
        return self._call("vectors.divergence", args, SCALAR, span, attributes)

    def _curl(self, operand: _Item, span: Span, scope: Scope, ctx: Context) -> Node:
        """``\\nabla \\times \\mathbf{v}``: the curl of a vector of three components."""
        value = self._operand_value(operand, scope, ctx)
        components = self._vector_components(value)
        if components is None:
            raise self._not_a_vector(operand, value, span, "the curl")
        if len(components) != 3:
            raise CompileError(
                f"{span.text} is the curl, which is of a vector of three components, and "
                f"this one has {len(components)}",
                span=span,
            )
        frame = self._frame_of_vector(components, span)
        self._check_written_in_the_frame(components, operand, frame, span, "curl")
        why = f"∇ is followed by \\times and a vector, in {frame.written}"
        self._decide(VECTOR, span, "its curl", why)
        attributes = (("coordinates", frame.coordinates),)
        args = [self._vector_of(components, span), *self._frame_arguments(frame)]
        return self._call("vectors.curl", args, VALUE, span, attributes)

    def _laplacian(
        self, kind: str, nabla: ast.Expr, operand: _Item, span: Span, scope: Scope, ctx: Context
    ) -> Node:
        """``\\nabla^2 f`` or ``\\Delta f``: the Laplacian, of a function or of each
        component of a vector in Cartesian coordinates; with a metric in force, ``\\nabla^2``
        of a function is \\nabla^\\mu \\nabla_\\mu, as before. ``\\nabla^2`` of a function
        the program defines, at a point, is its Hessian there (decision 25)."""
        if kind == "∇²" and (at := self._function_at_a_point(operand, scope, ctx)) is not None:
            return self._derivatives_at(at, span, ctx, hessian=True)
        value = self._operand_value(operand, scope, ctx)
        components = self._vector_components(value)
        if components is not None:
            frame = self._frame_of_vector(components, span)
            if not frame.cartesian:
                raise CompileError(
                    f"{span.text} is the Laplacian of a vector, taken here in Cartesian "
                    f"coordinates, and the coordinates in force are {frame.written}",
                    span=span,
                    hint="write it as \\nabla(\\nabla \\cdot \\mathbf{v}) - \\nabla \\times "
                    "(\\nabla \\times \\mathbf{v})",
                )
            why = "it is followed by a vector, in Cartesian coordinates"
            self._decide(VECTOR, span, "the Laplacian of each component", why)
            each = [self._scalar_laplacian(part, frame, span) for part in components]
            return self._vector_of(each, span)
        if kind == "∇²" and self.session.metric is not None:  # \nabla^\mu \nabla_\mu
            assert isinstance(nabla, ast.Scripted)
            box = self._derivative_operator(nabla, covariant=True)
            return self._apply(box, value, span, scope, ctx)
        frame = self._frame_of(operand, span, scope)
        self._check_a_function(value, operand, frame, span, "Laplacian")
        why = f"{nabla.span.text} is followed by a function, in {frame.written}"
        if kind == "Δ":
            why = f"nothing defines \\Delta, and it is followed by a function, in {frame.written}"
            note = f"{span.text} is read as the Laplacian; to name a change, define \\Delta"
            self._note(note)
            self._decide(VECTOR, span, "its Laplacian", why, note)
        else:
            self._decide(VECTOR, span, "its Laplacian", why)
        return self._scalar_laplacian(value, frame, span)

    def _scalar_laplacian(self, value: Node, frame: Frame, span: Span) -> Node:
        attributes = (("coordinates", frame.coordinates),)
        args = [value, *self._frame_arguments(frame)]
        return self._call("vectors.laplacian", args, SCALAR, span, attributes)

    def _frame_of(self, operand: _Item, span: Span, scope: Scope) -> Frame:
        """The frame of an operator's function: that of the coordinates in force, or, where
        there are none, the Cartesian coordinates the function is written in, as x and y
        of f(x, y), or else x, y and z."""
        frame = self._vector_frame(span, _variables_written(operand, scope))
        if frame is None:
            return self._frame_or_refuse(span, span.text)
        return frame

    def _frame_of_vector(self, components: Sequence[Node], span: Span) -> Frame:
        """The frame of a vector of ``components``: that in force, which must have as many
        coordinates, or, where there are none, x, y and z, or x and y in the plane."""
        variables = CARTESIAN[: len(components)] if self.session.coordinates is None else None
        frame = self._vector_frame(span, variables)
        if frame is None:
            return self._frame_or_refuse(span, span.text)
        if len(frame.coordinates) != len(components):
            raise CompileError(
                f"{span.text} is of a vector of {len(components)} components, and the "
                f"coordinates in force are {frame.written}",
                span=span,
            )
        return frame

    def _check_a_function(
        self, value: Node, operand: _Item, frame: Frame, span: Span, what: str
    ) -> None:
        """Refuse the gradient or the Laplacian of a name that names nothing, which would be
        0: books mean a function of position, as \\Phi is in ``\\nabla \\Phi``; and of a
        function written in r, θ or φ where they are no coordinates, as -GM/r is, whose
        dependence on x, y and z would be lost."""
        symbol = value.op.value if isinstance(value.op, ops.Const) else None
        if not isinstance(symbol, sympy.Symbol) or symbol in frame.coordinates:
            self._check_written_in_the_frame([value], operand, frame, span, what)
            return
        written = operand.span.text if isinstance(operand, ast.Expr) else str(symbol)
        variables = ", ".join(self._written_label(name) for name in frame.names)
        raise CompileError(
            f"{written} names nothing, so it does not depend on {frame.written}, and its "
            f"{what} would be 0",
            span=span,
            hint=f"for a function of position, write {written}({variables})",
        )

    def _check_written_in_the_frame(
        self, values: Sequence[Node], operand: _Item, frame: Frame, span: Span, what: str
    ) -> None:
        """Refuse an operator of Cartesian coordinates on what is written in none of them
        but in the coordinates of another frame, r, θ or φ, which here are symbols."""
        names = frozenset().union(*(_written_with(value) for value in values))
        if not frame.cartesian or names & set(frame.names):
            return
        foreign = sorted(names & CURVILINEAR)
        if not foreign:
            return
        written = operand.span.text if isinstance(operand, ast.Expr) else "it"
        shown = ", ".join(self._written_label(name) for name in foreign)
        raise CompileError(
            f"{written} is written in {shown}, which here is no coordinate, so in "
            f"{frame.written} its {what} would be 0",
            span=span,
            hint="declare the coordinates it is written in with their line element, as "
            "ds^2 = dr^2 + r^2 d\\theta^2 + r^2 \\sin^2\\theta d\\phi^2",
        )

    def _not_a_vector(self, operand: _Item, value: Node, span: Span, what: str) -> CompileError:
        written = operand.span.text if isinstance(operand, ast.Expr) else "this"
        notes = []
        symbol = isinstance(value.op, ops.Const) and isinstance(value.op.value, sympy.Symbol)
        if isinstance(operand, ast.Expr) and bold(operand) and symbol:
            notes.append(f"{written} names nothing, so it is a number, not a vector")
        return CompileError(
            f"{span.text} is {what}, which is of a vector, and {written} is no vector",
            span=span,
            notes=notes,
            hint=f"define it with its components first, as {written} := (v_x, v_y, v_z)",
        )

    # -- a function at a point (decision 25) -----------------------------------------------

    def _function_at_a_point(
        self, operand: _Item, scope: Scope, ctx: Context
    ) -> tuple[FunctionBinding, list[Node], Span] | None:
        """A function the program defines of several variables, written at a point, as
        ``f(1, 1)`` or ``f(\\mathbf{x})``: the function, and the point's coordinates.
        ``None`` for anything else, as f written at its own variables, f(x, y)."""
        if not isinstance(operand, ast.Call) or not isinstance(operand.func, ast.Name):
            return None
        binding = scope.lookup(operand.func.id)
        if not isinstance(binding, FunctionBinding) or len(binding.definition.params) < 2:
            return None
        params = binding.definition.params
        if all(isinstance(arg, ast.Name) for arg in operand.args) and len(operand.args) == len(
            params
        ):
            return None
        values = [self.value(arg, scope, ctx) for arg in operand.args]
        if len(values) == 1 and (components := self._vector_components(values[0])) is not None:
            values = components
        if len(values) != len(params):
            return None
        return binding, values, operand.span

    def _derivatives_at(
        self,
        at: tuple[FunctionBinding, list[Node], Span],
        span: Span,
        ctx: Context,
        *,
        hessian: bool,
    ) -> Node:
        """The gradient, as a column, or the Hessian of a function of several variables at
        a point: by its parameters, then at the point (decision 25)."""
        binding, point, written = at
        params = binding.definition.params
        symbols = tuple(self.session.symbol(param) for param in params)
        body = self._inline(binding, [self.graph.const(s, SCALAR) for s in symbols], span, ctx)
        what = "Hessian" if hessian else "gradient"
        name = binding.definition.name
        note = (
            f"{span.text} is read as the {what} of {name} at {written.text.removeprefix(name)}: "
            f"{name} is a function of the vector of its {len(params)} variables; "
            f"\\Delta {name} is the Laplacian"
        )
        if not hessian:
            note = (
                f"{span.text} is read as the gradient of {name} at "
                f"{written.text.removeprefix(name)}, a column"
            )
        self._note(note)
        why = f"{name} is a function of {len(params)} variables, written at a point"
        self._decide(VECTOR, span, f"its {what} there", why, note)
        pairs = [
            node
            for symbol, value in zip(symbols, point, strict=True)
            for node in (self.graph.const(symbol, SCALAR), value)
        ]
        if hessian:
            attributes = (("coordinates", symbols),)
            matrix = self._call("vectors.hessian", [body], VALUE, span, attributes)
            return self._call("core.subs", [matrix, *pairs], VALUE, span)
        cells = [
            self._substituted(
                self._differentiate(body, [(self.graph.const(s, SCALAR), 1)], span), pairs, span
            )
            for s in symbols
        ]
        return self.graph.add(ops.MakeArray((len(cells), 1)), cells, VALUE, span)

    # -- runs of factors, and the signs between them ------------------------------------------

    def _runs(
        self, items: list[tuple[str, _Item]], scope: Scope, ctx: Context
    ) -> list[tuple[str, Term | _Along]]:
        """The factors between the signs of a product of vectors, each run of them side by
        side multiplied, with the sign written before it."""
        runs: list[tuple[str, list[_Item]]] = []
        for joint, item in items:
            if joint or not runs:
                runs.append((joint, [item]))
            else:
                runs[-1][1].append(item)
        return [(joint, self._run_value(run, scope, ctx)) for joint, run in runs]

    def _run_value(self, run: list[_Item], scope: Scope, ctx: Context) -> Term | _Along:
        """Factors written side by side: what is written, read as a product is, and the
        values and operators of vector calculus among it, multiplied in turn."""
        if len(run) == 1 and not isinstance(run[0], ast.Expr):
            return run[0]
        bracketed = any(
            isinstance(item, ast.Expr) and _an_operator_in_brackets(item) for item in run
        )
        if all(isinstance(item, ast.Expr) for item in run) and not (bracketed and len(run) > 1):
            written = [item for item in run if isinstance(item, ast.Expr)]
            whole = _operand(written[0]) if len(written) == 1 else _product_of(written)
            return self._written_term(whole, scope, ctx)
        terms: list[Term] = []
        for item in run:
            if isinstance(item, _Along):
                raise CompileError(
                    f"{item.span.text} is the derivative of a vector along another, written "
                    "after it and a dot",
                    span=item.span,
                )
            if isinstance(item, ast.Expr):
                terms.append(self._written_term(_operand(item), scope, ctx))
            else:
                terms.append(item)
        values: list[Node] = []
        for term in reversed(terms):  # an operator acts on what follows it
            if isinstance(term, Operator):
                if not values:
                    raise self._nothing_to_act_on(term, scope)
                values[-1] = self._apply(term, values[-1], term.span, scope, ctx)
            else:
                values.append(term)
        span = _span_of(run[0], None).to(_span_of(run[-1], None))
        return self._multiply(values[::-1], span)

    def _written_term(self, written: ast.Expr, scope: Scope, ctx: Context) -> Term:
        """What is written, as a value, or an operator other than a function named alone,
        which is its value at its parameters, as f is of f(x, y) := ...."""
        term = self.term(written, scope, ctx)
        if isinstance(term, FunctionReference):
            return self.value(written, scope, ctx)
        return term

    def _signed(self, runs: list[tuple[str, Term | _Along]], span: Span) -> Term:
        """Runs of factors joined by their signs: ``\\times`` first, then ``\\cdot``, as
        ``\\mathbf{a} \\cdot \\mathbf{b} \\times \\mathbf{c}`` is a triple product."""
        values: list[Term | _Along] = [runs[0][1]]
        crossed = 0
        for joint, value in runs[1:]:
            if joint == "×":
                crossed += 1
                values[-1] = self._cross(values[-1], value, span, chained=crossed > 1)
            else:
                crossed = 0
                values.append(value)
        result = values[0]
        for value in values[1:]:
            result = self._dot(result, value, span)
        if isinstance(result, _Along):
            raise CompileError(
                f"{result.span.text} is the gradient of a vector, which is not taken yet",
                span=result.span,
            )
        return result

    def _cross(
        self, left: Term | _Along, right: Term | _Along, span: Span, *, chained: bool
    ) -> Node:
        """``\\times``: the cross product of two vectors of three components, and between
        numbers, or a number and a vector, multiplication."""
        if not (isinstance(left, Node) and isinstance(right, Node)):
            raise CompileError(
                f"{span.text} writes \\times beside an operator, which acts on a value",
                span=span,
            )
        a, b = self._vector_components(left), self._vector_components(right)
        if a is None or b is None:
            return self._scaled(left, right, a, b, "\\times", span)
        if chained:
            raise CompileError(
                f"{span.text} writes two cross products without brackets, and the cross "
                "product depends on which is taken first",
                span=span,
                hint="write the brackets, as \\mathbf{a} \\times (\\mathbf{b} \\times \\mathbf{c})",
            )
        if len(a) != 3 or len(b) != 3:
            raise CompileError(
                f"{span.text} is a cross product, which is of vectors of three components, "
                f"and these have {len(a)} and {len(b)}",
                span=span,
            )
        why = "\\times is written between two vectors"
        note = (
            "\\times between vectors is read as their cross product, and between numbers as "
            "multiplication"
        )
        self._note(note)
        self._decide(VECTOR, span, "their cross product", why, note)
        components = []
        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            plus = self.graph.add(ops.Mul(), (a[j], b[k]), SCALAR, span)
            minus = self.graph.add(ops.Mul(), (a[k], b[j]), SCALAR, span)
            components.append(self.graph.add(ops.Add((1, -1)), (plus, minus), SCALAR, span))
        return self._vector_of(components, span)

    def _dot(self, left: Term | _Along, right: Term | _Along, span: Span) -> Term:
        """``\\cdot``: the dot product of two vectors, the derivative along a vector before
        ∇, and between numbers, or a number and a vector, multiplication."""
        if isinstance(left, Node) and isinstance(right, Del):
            if self._vector_components(left) is None:
                raise CompileError(
                    f"{span.text} writes \\cdot \\nabla after what is no vector",
                    span=span,
                )
            why = "a vector is followed by \\cdot \\nabla"
            self._decide(VECTOR, span, "the derivative along the vector", why)
            return Directional(span, left)
        if isinstance(left, Node) and isinstance(right, _Along):
            return self._along(left, right.vector, span)
        if not (isinstance(left, Node) and isinstance(right, Node)):
            raise CompileError(
                f"{span.text} writes \\cdot beside an operator, which acts on a value",
                span=span,
            )
        a, b = self._vector_components(left), self._vector_components(right)
        if a is None or b is None:
            return self._scaled(left, right, a, b, "\\cdot", span)
        if len(a) != len(b):
            raise CompileError(
                f"{span.text} is a dot product of vectors of {len(a)} and {len(b)} components",
                span=span,
            )
        why = "\\cdot is written between two vectors"
        note = (
            "\\cdot between vectors is read as their dot product, and between numbers as "
            "multiplication"
        )
        self._note(note)
        self._decide(VECTOR, span, "their dot product", why, note)
        products = [
            self.graph.add(ops.Mul(), pair, SCALAR, span) for pair in zip(a, b, strict=True)
        ]
        return self.graph.add(ops.Add((1,) * len(products)), products, SCALAR, span)

    def _scaled(
        self,
        left: Node,
        right: Node,
        a: list[Node] | None,
        b: list[Node] | None,
        sign: str,
        span: Span,
    ) -> Node:
        """``\\cdot`` or ``\\times`` where one side is no vector: multiplication, said in a
        note beside a vector, where a product of vectors may have been meant."""
        if a is not None or b is not None:
            note = f"{sign} beside a number is read as multiplying the vector by it"
            self._note(note)
            self._decide(
                VECTOR, span, "a vector times a number", f"one side of {sign} is a number", note
            )
        return self._multiply([left, right], span)

    def _along(self, along: Node, operand: Node, span: Span) -> Node:
        """``(\\mathbf{a} \\cdot \\nabla) f``: the derivative of ``f`` along ``a``; of a
        vector, in Cartesian coordinates, whose unit vectors are the same everywhere, that of
        each component."""
        components = self._vector_components(along)
        assert components is not None
        frame = self._frame_of_vector(components, span)
        attributes = (("coordinates", frame.coordinates),)
        vector = self._vector_of(components, span)
        rest = self._frame_arguments(frame)

        def derivative(part: Node) -> Node:
            return self._call(
                "vectors.directional", [vector, part, *rest], SCALAR, span, attributes
            )

        each = self._vector_components(operand)
        if each is None:
            return derivative(operand)
        if not frame.cartesian:
            raise CompileError(
                f"{span.text} is the derivative of a vector along another, taken here in "
                f"Cartesian coordinates, and the coordinates in force are {frame.written}",
                span=span,
                hint="write the vector's components in x, y and z",
            )
        return self._vector_like(operand, [derivative(part) for part in each], span)

    def _apply(
        self, operator_: Operator, operand: Node, span: Span, scope: Scope, ctx: Context
    ) -> Node:
        if isinstance(operator_, Directional):
            assert operator_.along is not None
            return self._along(operator_.along, operand, span)
        if isinstance(operator_, Del):
            raise CompileError(
                f"{operator_.span.text} needs a sign, or a vector and a dot before it, as in "
                "\\nabla \\cdot \\mathbf{v} or (\\mathbf{u} \\cdot \\nabla)",
                span=operator_.span,
            )
        return super()._apply(operator_, operand, span, scope, ctx)

    def _check_delta_of_a_vector(self, written: list[tuple[str, ast.Expr]], scope: Scope) -> None:
        """Refuse ``\\delta^3(\\mathbf{r})``, the Dirac delta in three dimensions, which
        would be read as δ cubed times r: a distribution, which is not supported yet."""
        for (_, factor), (_, after) in itertools.pairwise(written):
            inside = after.span.text.strip()
            if not (
                isinstance(factor, ast.Scripted)
                and isinstance(factor.base, ast.Name)
                and factor.base.id == "delta"
                and scope.lookup("delta") is None
                and inside.startswith(("(", "\\left("))
            ):
                continue
            if inside.removeprefix("\\left").lstrip("( ").startswith(BOLD):
                written_text = factor.span.text + after.span.text
                raise CompileError(
                    f"{written_text} is the Dirac delta in three dimensions, a distribution, "
                    "which is not supported yet",
                    span=factor.span.to(after.span),
                    notes=["read as written, it would be \\delta cubed times the vector"],
                )

    # -- names: unit vectors, and the components of vectors ----------------------------------

    def _Accent(self, node: ast.Accent, scope: Scope, ctx: Context) -> Node:
        found = hatted(node)
        if (
            found is not None
            and isinstance(node.base, ast.Name)
            and id(node) not in self._hats_named
        ):
            letter, written_bold = found
            unnamed = scope.lookup(node.base.id + "hat") is None
            if unnamed and (written_bold or not self._operator_near(node, scope)):
                unit = self._unit_vector(letter, node, scope, written_bold=written_bold)
                if unit is not None:
                    return unit
        return super()._Accent(node, scope, ctx)

    def _operator_near(self, node: ast.Accent, scope: Scope) -> bool:
        """Whether a plain hat is near an operator, and so an operator's: the program
        defines one, as ``\\hat{p} := -i\\hbar \\frac{d}{dx}``, beside which ``\\hat{x}`` is
        the position (:mod:`relativisticpy.reading.quantum`)."""
        if self._position(node, scope) is not None:
            return True
        level: Scope | None = scope
        while level is not None:
            if any(isinstance(binding, OperatorBinding) for _, binding in level.bindings()):
                return True
            level = level.parent
        return False

    def _Name(self, node: ast.Name, scope: Scope, ctx: Context) -> Term:
        if node.id in ("i", "j", "k") and bold(node) and scope.lookup(node.id) is None:
            unit = self._unit_vector(node.id, node, scope, written_bold=True)
            if unit is not None:
                note = (
                    f"{node.span.text} is read as a unit vector of Cartesian coordinates; to "
                    f"name a vector {node.id}, define it"
                )
                self._note(note)
                return unit
        if (partial := self._partial_named(node, scope, ctx)) is not None:  # f_x
            return partial
        return super()._Name(node, scope, ctx)

    def _Scripted(self, node: ast.Scripted, scope: Scope, ctx: Context) -> Term:
        if (unit := self._basis_vector(node, scope)) is not None:  # \mathbf{e}_r
            return unit
        if (epsilon := self._levi_civita(node, scope, ctx)) is not None:  # \epsilon_{ijk}
            return epsilon
        if (partial := self._subscript_partial(node, scope, ctx)) is not None:  # \mathbf{x}_u
            return partial
        if hatted(node.base) is not None:  # {\hat\theta}^{(\mu)}: a name, as before
            self._hats_named.add(id(node.base))
        return super()._Scripted(node, scope, ctx)

    def _is_notation(
        self,
        node: ast.Expr,
        scope: Scope,
        ctx: Context,
        *,
        equation: bool = False,
        operation: bool = False,
    ) -> bool:
        """Whether an expression is notation, as :meth:`Reader._is_notation` decides, with
        \\epsilon the Levi-Civita symbol where :meth:`_levi_civita` reads it so: it has
        components, and is no new tensor."""
        if any(self._levi_civita_space(current, scope) for current in _walk(node)):
            scope = scope.child()
            scope.bind("epsilon", VariableBinding(self.graph.const(sympy.Integer(1), SCALAR)))
        return super()._is_notation(node, scope, ctx, equation=equation, operation=operation)

    def _levi_civita_space(self, node: ast.Node, scope: Scope) -> Space | None:
        """The space \\epsilon with indices is the Levi-Civita symbol of, where nothing
        defines \\epsilon and it has as many indices as the space has dimensions, a space
        whose metric is the identity; ``None`` for anything else."""
        if not isinstance(node, ast.Scripted):
            return None
        base = node.base
        if not (isinstance(base, ast.Name) and base.id == "epsilon"):
            return None
        if scope.lookup("epsilon") is not None:
            return None
        labels = []
        for script in node.scripts:
            items = script.indices.items if script.indices is not None else ()
            if not items or not all(_plain_label(item) for item in items):
                return None
            labels += [str(item.label) for item in items]
        space = self._labels.get(labels[0]) or self.session.labels.get(labels[0])
        if space is None and self.session.coordinates is not None:
            space = self.session.spacetime(node.span)
        if space is None or space.dim != len(labels) or not self._euclidean(space):
            return None
        return space

    def _levi_civita(self, node: ast.Scripted, scope: Scope, ctx: Context) -> Term | None:
        """``\\epsilon_{ijk}`` where nothing defines \\epsilon, with as many indices as the
        space they range over has dimensions, a space whose metric is the identity: the
        Levi-Civita symbol, +1 of 1, 2, 3 in any position, as in Cartesian coordinates and
        in \\mathbb{R}^3 (decision 16 of design/notation.md). ``None`` for anything else, as
        \\epsilon of spacetime, whose sign is a convention."""
        space = self._levi_civita_space(node, scope)
        if space is None:
            return None
        size = space.dim
        indices = Indices([Index(name) for name in "abcdefgh"[:size]])
        kind = TensorType(indices, (space,) * size)
        symbol = self._call("vectors.levi_civita", [self.graph.const(size, VALUE)], kind, node.span)
        why = (
            f"nothing defines \\epsilon, and its {size} indices range over a space of {size} "
            "dimensions whose metric is the identity"
        )
        self._decide(VECTOR, node.span, "the Levi-Civita symbol", why)
        return self._access(TensorSource(symbol, indices, "\\epsilon"), node, scope, ctx)

    def _basis_vector(self, node: ast.Scripted, scope: Scope) -> Node | None:
        """``\\mathbf{e}_r`` or ``\\hat{\\mathbf{e}}_\\theta``, where nothing defines them:
        the unit vector along the coordinate the subscript names."""
        base = node.base
        letter_bold = (isinstance(base, ast.Name) and base.id == "e" and bold(base)) or (
            hatted(base) or ("", False)
        )[0] == "e"
        if not letter_bold or len(node.scripts) != 1 or node.scripts[0].kind != "_":
            return None
        script = node.scripts[0]
        items = script.indices.items if script.indices is not None else ()
        if len(items) != 1 or items[0].label is None:
            return None
        label = str(items[0].label)
        if scope.lookup(f"e_{label}") is not None or scope.lookup("e") is not None:
            return None
        return self._unit_vector(label, node, scope, written_bold=True)

    def _subscript_partial(self, node: ast.Scripted, scope: Scope, ctx: Context) -> Node | None:
        """``\\mathbf{x}_u`` or ``\\mathbf{x}_{uv}`` of a function the program defines of u
        and v, where nothing defines the name: its partial derivatives, at its parameters
        (decision 5.6 of design/notation.md)."""
        base = _named(node.base)  # \bar{\mathbf{x}}_u, of \bar{\mathbf{x}}(u, v) := ...
        if not isinstance(base, ast.Name) or len(node.scripts) != 1 or node.scripts[0].kind != "_":
            return None
        script = node.scripts[0]
        items = script.indices.items if script.indices is not None else ()
        labels = [str(item.label) for item in items if item.label is not None]
        if len(labels) != len(items):
            return None
        return self._partial_by(base.id, node.base.span.text, labels, node, scope, ctx)

    def _partial_named(self, node: ast.Name, scope: Scope, ctx: Context) -> Node | None:
        """``f_x`` or ``f_{xy}``, one name as the lexer joins it, of a function the program
        defines of x and y: its partial derivatives, as :meth:`_subscript_partial`."""
        function, underscore, labels = node.id.partition("_")
        if not (underscore and function and labels.isalpha()) or scope.lookup(node.id):
            return None
        return self._partial_by(function, function, [labels], node, scope, ctx)

    def _partial_by(
        self,
        function: str,
        written_function: str,
        labels: list[str],
        node: ast.Expr,
        scope: Scope,
        ctx: Context,
    ) -> Node | None:
        """The partial derivatives of the function the program defines as ``function``, by
        the parameters ``labels`` name in turn, a run of letters each naming one, at its
        parameters; ``None`` where it defines no such function, as a family is, or defines
        the name written."""
        binding = scope.lookup(function)
        if not isinstance(binding, FunctionBinding) or isinstance(binding.definition, Family):
            return None
        params = binding.definition.params
        if not all(label in params for label in labels):
            labels = [letter for label in labels for letter in label]
        if not labels:
            return None
        if not set(labels) <= set(params) or scope.lookup(f"{function}_{''.join(labels)}"):
            return None
        written = node.span.text
        shown = ", ".join(params)
        by = " and ".join(dict.fromkeys(labels))
        why = f"{written_function} is a function of {shown}, and nothing defines {written}"
        reading = f"the partial derivative of {written_function}({shown}) by {by}"
        note = f"{written} is read as {reading}"
        self._note(note)
        self._decide(VECTOR, node.span, reading, why, note)
        symbols = [self.graph.const(self.session.symbol(param), SCALAR) for param in params]
        body = self._inline(binding, symbols, node.span, ctx)
        variables = [(symbols[params.index(label)], 1) for label in labels]
        return self._differentiate(body, variables, node.span)

    def _absolute(self, node: ast.Abs, operand: Node) -> Node:
        components = self._vector_components(operand)
        if components is None:
            return super()._absolute(node, operand)
        why = "bars are written around a vector"
        self._decide(VECTOR, node.span, "its length", why)
        return self._call(
            "vectors.norm", [self._vector_of(components, node.span)], SCALAR, node.span
        )

    def _power(self, base: Node, exponent: Node, span: Span) -> Node:
        components = self._vector_components(base)
        two = isinstance(exponent.op, ops.Const) and exponent.op.value == 2
        written = span.text.lstrip().startswith((*BOLD, "\\vec", "{\\bf"))
        if components is None or not two or not written:  # [1, 2]^2, refused as before
            return super()._power(base, exponent, span)
        note = f"{span.text}, the square of a vector, is read as its dot product with itself"
        self._note(note)
        self._decide(VECTOR, span, "its dot product with itself", "it is a vector squared", note)
        squares = [self.graph.add(ops.Mul(), (part, part), SCALAR, span) for part in components]
        return self.graph.add(ops.Add((1,) * len(squares)), squares, SCALAR, span)

    # -- calculus, component by component --------------------------------------------------

    def _call(
        self,
        name: str,
        args: Sequence[Node],
        kind: Type,
        span: Span,
        attributes: ops.Attributes = (),
    ) -> Node:
        if name in _EACH_COMPONENT and args and (each := self._vector_components(args[0])):
            rest = list(args[1:])
            of_one = super()._call  # named here: a comprehension has no super() before 3.12
            parts = [of_one(name, [part, *rest], SCALAR, span, attributes) for part in each]
            return self._vector_like(args[0], parts, span)
        return super()._call(name, args, kind, span, attributes)

    def _differentiate(self, expr: Node, variables: Sequence[tuple[Node, int]], span: Span) -> Node:
        components = self._vector_components(expr)
        by_a_list = any(self._vector_components(variable) for variable, _ in variables)
        if components is None or by_a_list:  # \frac{\partial(x, y)}{\partial(r, \theta)}
            return super()._differentiate(expr, variables, span)
        self._check_fixed_frame(variables, span)
        of_one = super()._differentiate
        parts = [of_one(part, variables, span) for part in components]
        return self._vector_like(expr, parts, span)

    def _index_derivative(
        self, operand: Node, item: ast.IndexItem, upper: bool, covariant: bool, span: Span
    ) -> Node:
        components = self._vector_components(operand)
        if components is None:
            return super()._index_derivative(operand, item, upper, covariant, span)
        of_one = super()._index_derivative
        parts = [of_one(part, item, upper, covariant, span) for part in components]
        if any(isinstance(part.type, TensorType) for part in parts):
            raise CompileError(
                f"{span.text} differentiates a vector by a coordinate, which index notation "
                "writes with the vector's own index",
                span=span,
                hint="write the derivative by a variable, as \\frac{\\partial}{\\partial t}",
            )
        return self._vector_like(operand, parts, span)

    def _check_fixed_frame(self, variables: Sequence[tuple[Node, int]], span: Span) -> None:
        """Refuse the derivative of a vector by a coordinate whose unit vectors change with
        it, as r̂ does with θ: that is not the derivative of its components."""
        frame = self._vector_frame(span)
        if frame is None or frame.cartesian or self.session.coordinates is None:
            return
        for variable, _ in variables:
            value = variable.op.value if isinstance(variable.op, ops.Const) else None
            if value in frame.coordinates:
                shown = self._written_label(str(value))
                raise CompileError(
                    f"{span.text} differentiates a vector by {shown}, and the unit vectors "
                    f"of {frame.written} change with {shown}, so it is not the derivative of "
                    "its components",
                    span=span,
                    hint="write the vector in Cartesian components, or with indices, as "
                    "\\nabla_\\mu V^\\nu",
                )

    # -- equations between vectors ------------------------------------------------------------

    def _sides_of_an_equation(
        self, node: ast.Equation, scope: Scope, ctx: Context
    ) -> tuple[Node, Node]:
        lhs, rhs = super()._sides_of_an_equation(node, scope, ctx)
        left, right = self._vector_components(lhs), self._vector_components(rhs)
        if left is None and right is None:
            return lhs, rhs
        sides = ((lhs, rhs, node.lhs, node.rhs), (rhs, lhs, node.rhs, node.lhs))
        for vector, other, vector_written, written in sides:
            if self._vector_components(other) is None and other.type == SCALAR:
                if _a_pair(vector) and not (_known_zero(other) and bold(written)):
                    try:  # \{L_x, L_y\} = L_z or [a, b] = 0: its own check says what it is
                        self._check_pair_and_scalar(node, lhs, rhs)
                    except CompileError as refused:
                        refused.notes.extend(self._typeface_notes(node.span))
                        raise
                    return lhs, rhs
                if _known_zero(other):
                    continue
                raise CompileError(
                    f"{node.span.text} equates a vector with {written.span.text}, a number",
                    span=node.span,
                    notes=[*_called_as_a_product(vector_written), *self._typeface_notes(node.span)],
                    hint="write the vector it equals, as \\mathbf{0} or (0, 0, 0)",
                )
        if left is None and right is not None and _known_zero(lhs):
            left = self._zeros(right, node.lhs.span)
        if right is None and left is not None and _known_zero(rhs):
            right = self._zeros(left, node.rhs.span)
        if left is None or right is None:  # A \mathbf{x}, a matrix's product, as before
            return (
                self._array_of(lhs, left, node.lhs.span),
                self._array_of(rhs, right, node.rhs.span),
            )
        if len(left) != len(right):
            raise CompileError(
                f"{node.span.text} equates vectors of {len(left)} and {len(right)} components",
                span=node.span,
            )
        why = "both sides are vectors"
        self._decide(VECTOR, node.span, "an equation of each component", why)
        return self._array(left, node.lhs.span), self._array(right, node.rhs.span)

    def _zeros(self, like: list[Node], span: Span) -> list[Node]:
        """``\\mathbf{0}`` or 0 beside a vector: the vector of zeros of its size."""
        return [self.graph.const(sympy.Integer(0), SCALAR, span) for _ in like]

    def _typeface_notes(self, span: Span) -> list[str]:
        """Say so where a letter is written both bold and plain, as r and \\mathbf{r} are in
        ``r = |\\mathbf{r}|``: they are one name, as typefaces do not make names of their
        own yet (decision 9 of design/notation.md)."""
        text = span.text
        bold_letters = set(
            re.findall(r"\\(?:mathbf|boldsymbol|bm)\s*\{\s*\\?([A-Za-z]+)\s*\}", text)
        )
        plain = re.sub(r"\\(?:mathbf|boldsymbol|bm)\s*\{[^}]*\}", " ", text)
        both = sorted(
            letter
            for letter in bold_letters
            if re.search(rf"(?<![\\A-Za-z]){letter}(?![A-Za-z])", plain)
        )
        return [
            f"\\mathbf{{{letter}}} and {letter} are one name, as typefaces do not make names "
            "of their own yet"
            for letter in both
        ]

    def _array_of(self, value: Node, components: list[Node] | None, span: Span) -> Node:
        """A side of an equation as an array, where it is a vector, in its shape, so that it
        is compared with the other side component by component: a list is no array."""
        if components is None or not isinstance(value.op, ops.Call):
            return value
        return self._array(components, span)

    def _array(self, components: Sequence[Node], span: Span) -> Node:
        return self.graph.add(ops.MakeArray((len(components),)), list(components), VALUE, span)

    # -- integrals over a region or along a curve ------------------------------------------

    def _Integral(self, node: ast.Integral, scope: Scope, ctx: Context) -> Node:
        if node.domain is not None and not node.bounds:
            curve = self._curve_named(node.domain, scope, ctx)
            if curve is not None:  # \oint_C \mathbf{F} \cdot d\mathbf{r}
                return self._along_a_curve(node, curve, scope, ctx)
        if node.domain is not None and (over := self._over_a_region(node, scope, ctx)):
            return over
        return super()._Integral(node, scope, ctx)

    def _curve_named(self, domain: ast.Expr, scope: Scope, ctx: Context) -> _Curve | None:
        """The curve an integral is along, written below it: a function of one parameter
        the program defines, whose value is a vector, as ``\\mathbf{r}_1(s) := (\\cos s,
        \\sin s, 0)``, with the range the program states for its parameter, as
        ``0 \\le s \\le 2\\pi``. ``None`` where the name is of no such function."""
        if isinstance(domain, ast.Scripted):
            name = label_name(domain)
        elif isinstance(domain, ast.Name):
            name = domain.id
        else:
            return None
        binding = scope.lookup(name) if name is not None else None
        if name is None or not isinstance(binding, FunctionBinding):
            return None
        if isinstance(binding.definition, Family) or len(binding.definition.params) != 1:
            return None
        (param,) = binding.definition.params
        parameter = self.graph.const(sympy.Dummy(param, real=True), SCALAR, domain.span)
        point = self._inline(binding, [parameter], domain.span, ctx)
        if self._vector_components(point) is None:
            return None
        lower, upper = self._domain(param)
        if not (sympy.sympify(lower).is_finite and sympy.sympify(upper).is_finite):
            shown = self._written_label(param)
            raise CompileError(
                f"{domain.span.text} is a curve of {shown}, and the program states no range of "
                f"{shown} for an integral along it",
                span=domain.span,
                hint=f"state it, as 0 \\le {shown} \\le 2\\pi",
            )
        tangent = self._differentiate(point, [(parameter, 1)], domain.span)
        limits = (self.graph.const(lower, SCALAR), self.graph.const(upper, SCALAR))
        return _Curve(name, domain.span.text, parameter, point, tangent, limits)

    def _along_a_curve(self, node: ast.Integral, curve: _Curve, scope: Scope, ctx: Context) -> Node:
        """``\\oint_C \\mathbf{F} \\cdot d\\mathbf{r}``: the integral over the parameter of
        the curve C, with the curve's name its point, ``d`` before it, or before a name
        that names nothing, its tangent, and x, y and z its components, so that a field is
        taken along it."""
        inner = scope.child()
        inner.bind(curve.name, VariableBinding(curve.point))
        saved, read = self._differentials, self._differentials_read
        self._differentials = {**saved, curve.name: curve.tangent, "": curve.tangent}
        try:
            integrand = self.value(node.integrand, inner, ctx)
            used = self._differentials_read > read
        finally:
            self._differentials = saved
        if not used:
            raise CompileError(
                f"{node.span.text} is along the curve {curve.written}, and its integrand has no "
                "differential along it",
                span=node.span,
                hint=f"write it, as \\mathbf{{F}} \\cdot d\\mathbf{{r}}, or d{curve.written}",
            )
        components = self._vector_components(curve.point)
        assert components is not None
        frame = self._vector_frame(node.span, CARTESIAN[: len(components)])
        if frame is not None and frame.cartesian and len(frame.coordinates) == len(components):
            pairs = [
                each
                for coordinate, component in zip(frame.coordinates, components, strict=True)
                for each in (self.graph.const(coordinate, SCALAR), component)
            ]
            integrand = self._substituted(integrand, pairs, node.span)
        why = f"{curve.written} is a function of one variable whose value is a vector"
        self._decide(VECTOR, node.span, f"an integral along {curve.written}", why)
        lower, upper = curve.limits
        args = [integrand, curve.parameter, lower, upper]
        attributes = (("inside", True),) if saved else ()  # \oint_{r_1} \oint_{r_2}
        return self._call("vectors.along", args, SCALAR, node.span, attributes)

    def _over_a_region(self, node: ast.Integral, scope: Scope, ctx: Context) -> Node | None:
        """``\\iint_{x^2 + y^2 \\le 1} f\\,dA`` or ``\\iiint_{[0, 1]^3} f\\,dV``: an integral
        over a disk or a ball about the origin, in polar or spherical coordinates, or over a
        box, as integrals in turn. ``None`` for any other region, which is refused as before."""
        size = _measure_size(node)
        if size is None:
            return None
        ball = _ball(node.domain, size)
        box = _box(node.domain, size)
        if ball is None and box is None:
            return None
        frame = self._vector_frame(node.span, list(ball[0]) if ball else CARTESIAN[:size])
        if frame is None or not frame.cartesian or len(frame.coordinates) != size:
            return None
        inner = scope.child()
        integrand = self.value(node.integrand, inner, ctx)
        written = node.domain.span.text if node.domain is not None else ""
        if box is not None:
            limits = [
                self._call(
                    "core.limits",
                    [
                        self.graph.const(coordinate, SCALAR),
                        self.value(low, scope, ctx),
                        self.value(high, scope, ctx),
                    ],
                    VALUE,
                    node.span,
                )
                for coordinate, (low, high) in zip(frame.coordinates, box, strict=True)
            ]
            reading = f"the box {written}, an integral over each coordinate in turn"
            self._decide(VECTOR, node.span, reading, "its region is written as a box")
            return self._call("integrate", [integrand, *limits], SCALAR, node.span)
        assert ball is not None
        bound = ball[1]
        squared = (  # R^2: the radius is R, a length
            isinstance(bound, ast.Scripted)
            and len(bound.scripts) == 1
            and bound.scripts[0].kind == "^"
            and isinstance(bound.scripts[0].body, ast.Number)
            and bound.scripts[0].body.text == "2"
        )
        if isinstance(bound, ast.Scripted) and squared:
            radius = self.value(bound.base, scope, ctx)
        else:
            half = self.graph.const(sympy.Rational(1, 2), SCALAR)
            radius = self._power(self.value(bound, scope, ctx), half, node.span)
        return self._over_a_ball(integrand, frame, radius, written, node.span)

    def _over_a_ball(
        self, integrand: Node, frame: Frame, radius: Node, written: str, span: Span
    ) -> Node:
        """An integral over the disk or the ball of ``radius`` about the origin, in polar or
        spherical coordinates, with their Jacobians."""
        rho = sympy.Dummy("rho", nonnegative=True)
        theta, phi = sympy.Dummy("theta", real=True), sympy.Dummy("phi", real=True)
        if len(frame.coordinates) == 2:
            cartesian = [rho * sympy.cos(theta), rho * sympy.sin(theta)]
            jacobian = rho
            ranges = [(rho, None), (theta, 2 * sympy.pi)]
            what = "the disk"
        else:
            cartesian = [
                rho * sympy.sin(theta) * sympy.cos(phi),
                rho * sympy.sin(theta) * sympy.sin(phi),
                rho * sympy.cos(theta),
            ]
            jacobian = rho**2 * sympy.sin(theta)
            ranges = [(rho, None), (theta, sympy.pi), (phi, 2 * sympy.pi)]
            what = "the ball"
        note = f"{written} is read as {what} about the origin, integrated in polar coordinates"
        if len(frame.coordinates) == 3:
            note = note.replace("polar", "spherical")
        self._note(note)
        self._decide(
            VECTOR,
            span,
            f"{what} about the origin",
            "its region is a sum of squares within a bound",
            note,
        )
        pairs = [
            node
            for coordinate, value in zip(frame.coordinates, cartesian, strict=True)
            for node in (self.graph.const(coordinate, SCALAR), self.graph.const(value, SCALAR))
        ]
        moved = self._substituted(integrand, pairs, span)
        weighted = self._multiply([moved, self.graph.const(jacobian, SCALAR)], span)
        zero = self.graph.const(sympy.Integer(0), SCALAR)
        limits = [
            self._call(
                "core.limits",
                [
                    self.graph.const(variable, SCALAR),
                    zero,
                    radius if upper is None else self.graph.const(upper, SCALAR),
                ],
                VALUE,
                span,
            )
            for variable, upper in ranges
        ]
        return self._call("integrate", [weighted, *limits], SCALAR, span)

    # -- calls: fields at a point, functions of a vector -----------------------------------

    def _Call(self, node: ast.Call, scope: Scope, ctx: Context) -> Node:
        if (why := self._coordinate_before_brackets(node.func, node.args, scope)) is not None:
            self._decide("product or call", node.span, "a product", why)
            group = node.args[0] if len(node.args) == 1 else ast.Tuple(node.args, span=node.span)
            times = ast.Binary("*", node.func, group, implicit=True, span=node.span)
            return self.value(times, scope, ctx)
        if (composed := self._at_a_vector(node, scope, ctx)) is not None:
            return composed
        return super()._Call(node, scope, ctx)

    def _at_a_vector(self, node: ast.Call, scope: Scope, ctx: Context) -> Node | None:
        """A call with one vector as its argument: a field the program defines at that
        point, as ``\\mathbf{v}(\\mathbf{r}(t))`` is v with x, y and z the components of
        r(t); and a function of as many variables as the vector has components, at them,
        as ``f(\\mathbf{x})`` is f(x_1, x_2). ``None`` for any other call."""
        func = node.func
        if not isinstance(func, ast.Name) or len(node.args) != 1:
            return None
        binding = scope.lookup(func.id)
        if not isinstance(binding, FunctionBinding | VariableBinding):
            return None
        if isinstance(binding, FunctionBinding) and len(binding.definition.params) < 2:
            return None
        if isinstance(binding, VariableBinding) and binding.node.type not in (SCALAR, VALUE):
            return None
        argument = node.args[0]
        named = scope.lookup(argument.id) if isinstance(argument, ast.Name) else None
        if isinstance(argument, ast.Name) and not isinstance(named, VariableBinding):
            return None  # \alpha(r): the value of \alpha, which depends on r
        try:
            point = self.value(argument, scope, ctx)
        except RelativisticPyError:
            return None
        components = self._vector_components(point)
        if components is None:
            return None
        if isinstance(binding, FunctionBinding):
            if len(components) != len(binding.definition.params):
                return None
            why = (
                f"{func.span.text} is a function of {len(components)} variables, and "
                f"{argument.span.text} a vector of as many components"
            )
            self._decide(VECTOR, node.span, "the function at the vector's components", why)
            return self._inline(binding, components, node.span, ctx)
        frame = self._vector_frame(node.span, CARTESIAN[: len(components)])
        if frame is None or len(frame.coordinates) != len(components):
            return None
        why = f"{func.span.text} is a field in {frame.written}, and {argument.span.text} a point"
        self._decide(VECTOR, node.span, f"{func.span.text} at the point", why)
        pairs = [
            value
            for coordinate, component in zip(frame.coordinates, components, strict=True)
            for value in (self.graph.const(coordinate, SCALAR), component)
        ]
        return self._substituted(binding.node, pairs, node.span)


def _written_factors(node: ast.Binary) -> list[tuple[str, ast.Expr]]:
    """The factors of a product as written, each with the sign written before it, ``·``,
    ``×`` or none: what brackets hold is one factor, and a minus sign is a factor of -1, so
    that ``-\\nabla \\Phi`` is -1 times the gradient."""
    found: list[tuple[str, ast.Expr]] = []
    pending: list[tuple[str, ast.Expr]] = [("", node)]
    while pending:
        joint, current = pending.pop()
        whole = current is node  # brackets around the whole product only group it
        if (
            isinstance(current, ast.Binary)
            and current.op == "*"
            and (whole or not _bracketed(current))
        ):
            between = Span(current.span.source, current.left.span.end, current.right.span.start)
            pending.append((_sign(between.text), current.right))
            pending.append((joint, current.left))
            continue
        if isinstance(current, ast.Unary) and current.op == "-":
            found.append((joint, ast.Number("-1", span=current.span)))
            pending.append(("", current.operand))
            continue
        found.append((joint, current))
    return found


def _sign(between: str) -> str:
    """The sign written between two factors: ``·`` for ``\\cdot``, ``×`` for ``\\times``, and
    none for anything else, as a space or ``*``."""
    if "\\times" in between or "×" in between:
        return "×"
    if any(dot in between.replace("\\cdots", "") for dot in ("\\cdot", "·", "⋅")):
        return "·"
    return ""


_BRACKETS = (("\\left(", "\\right)"), ("\\left[", "\\right]"), ("(", ")"), ("[", "]"))


def _called_as_a_product(written: ast.Expr) -> list[str]:
    """Say so where a vector is a name times a list, as ``\\rho^2(r, \\theta)`` is read,
    where a function of them may have been meant."""
    if not (isinstance(written, ast.Binary) and written.op == "*"):
        return []
    if not isinstance(written.right, ast.Tuple | ast.List):
        return []
    name, listed = written.left.span.text, written.right.span.text
    return [
        f"{written.span.text} is read as {name} times the list {listed}; for a function of "
        f"them, define it, as {name}{listed} := ..."
    ]


def _known_zero(value: Node) -> bool:
    """Whether a value is the number 0, as written: 0 or \\mathbf{0}."""
    return isinstance(value.op, ops.Const) and value.op.value == 0


def _bracketed(node: ast.Expr) -> bool:
    """Whether brackets hold the whole of what is written, as in ``(\\mathbf{u} \\cdot
    \\nabla)``, but not in ``(a)(b)``."""
    text = node.span.text.strip()
    for opening, closing in _BRACKETS:
        if text.startswith(opening) and text.endswith(closing):
            depth = 0
            for character in text[len(opening) : len(text) - len(closing)]:
                depth += (character in "([" and 1) or (character in ")]" and -1) or 0
                if depth < 0:
                    return False
            return depth == 0
    return False


def _an_operator_in_brackets(factor: ast.Expr) -> bool:
    """Whether brackets hold an operator of vector calculus, ending in ∇ with nothing after
    it, as ``(\\mathbf{u} \\cdot \\nabla)`` does, which acts on the factor after them."""
    if not (isinstance(factor, ast.Binary) and factor.op == "*" and _bracketed(factor)):
        return False
    last = _written_factors(factor)[-1][1]
    return isinstance(last, ast.Operator) and last.kind == "nabla"


def _product_of(factors: Sequence[ast.Expr]) -> ast.Expr:
    """Factors written side by side, as the product they are."""
    product = factors[0]
    for factor in factors[1:]:
        span = product.span.to(factor.span)
        product = ast.Binary("*", product, factor, implicit=True, span=span)
    return product


def _span_of(item: _Item, default: Span | None) -> Span:
    """Where an item of a product is written, or ``default`` for a value made of none."""
    span: Span | None = item.span
    if span is None:
        assert default is not None
        return default
    return span


def _written(item: _Item) -> str:
    """What an operator acts on, as written, for messages."""
    span: Span | None = item.span
    return span.text if span is not None else "an operator"


def _a_function_written(item: _Item, scope: Scope) -> bool:
    """Whether what follows \\Delta is a function: one the program defines, named alone, a
    call, or brackets; \\Delta before a name, as \\Delta x, is the change in it."""
    if not isinstance(item, ast.Expr):
        return isinstance(item, Node)
    if isinstance(item, ast.Name):
        return isinstance(scope.lookup(item.id), FunctionBinding)
    return isinstance(item, ast.Call) or _bracketed(item)


#: The measures of regions, by how many dimensions they are of: dA, dS of a plane region,
#: dV and d\tau of a solid.
_MEASURES = {"A": 2, "S": 2, "V": 3, "tau": 3}


def _measure_size(node: ast.Integral) -> int | None:
    """How many dimensions an integral over a region is of, as its measure says: 2 of dA,
    3 of dV or d^3x; ``None`` for any other measure, or one with limits."""
    if len(node.bounds) != 1:
        return None
    bound = node.bounds[0]
    if bound.lower is not None or bound.upper is not None:
        return None
    if isinstance(bound.dimension, ast.Number) and bound.dimension.text in ("2", "3"):
        return int(bound.dimension.text)
    if bound.dimension is None and isinstance(bound.variable, ast.Name):
        return _MEASURES.get(bound.variable.id)
    return None


def _ball(region: ast.Expr | None, size: int) -> tuple[tuple[str, ...], ast.Expr] | None:
    """The coordinates and the square of the radius of a disk or a ball about the origin,
    written as ``x^2 + y^2 \\le 1``; ``None`` for any other region."""
    if not (isinstance(region, ast.Comparison) and region.ops in (("<=",), ("<",))):
        return None
    lhs, bound = region.operands
    names = []
    for sign, term in _signed_terms(lhs):
        squared = (
            sign > 0
            and isinstance(term, ast.Scripted)
            and isinstance(term.base, ast.Name)
            and len(term.scripts) == 1
            and term.scripts[0].kind == "^"
            and isinstance(term.scripts[0].body, ast.Number)
            and term.scripts[0].body.text == "2"
        )
        if not squared:
            return None
        assert isinstance(term, ast.Scripted)
        assert isinstance(term.base, ast.Name)
        names.append(term.base.id)
    if len(names) != size or len(set(names)) != size:
        return None
    return tuple(names), bound


def _box(region: ast.Expr | None, size: int) -> list[tuple[ast.Expr, ast.Expr]] | None:
    """The limits of each coordinate of a box, written ``[0, 1]^3`` or ``[0, 1] \\times
    [0, 2]``; ``None`` for any other region."""
    if (
        isinstance(region, ast.Scripted)
        and len(region.scripts) == 1
        and region.scripts[0].kind == "^"
        and isinstance(region.scripts[0].body, ast.Number)
        and region.scripts[0].body.text == str(size)
    ):
        interval = _interval(region.base)
        return [interval] * size if interval is not None else None
    if isinstance(region, ast.Binary) and region.op == "*":
        written = _written_factors(region)
        intervals = [_interval(factor) for _, factor in written]
        crossed = all(joint == "×" for joint, _ in written[1:])
        if crossed and len(intervals) == size and all(intervals):
            return [interval for interval in intervals if interval is not None]
    return None


def _interval(written: ast.Expr) -> tuple[ast.Expr, ast.Expr] | None:
    """The ends of a closed interval, ``[a, b]``."""
    if isinstance(written, ast.List) and len(written.items) == 2:
        return written.items[0], written.items[1]
    return None


def _variables_written(operand: _Item, scope: Scope) -> list[str] | None:
    """The variables a function is written in, where they are Cartesian coordinates: those
    of a function the program defines, named alone, as f of f(x, y) := ..., or those named
    in a call, as x, y and z of f(x, y, z); ``None`` for anything else."""
    names: list[str] = []
    if isinstance(operand, ast.Name):
        binding = scope.lookup(operand.id)
        if isinstance(binding, FunctionBinding):
            names = list(binding.definition.params)
    elif isinstance(operand, ast.Call) and all(isinstance(arg, ast.Name) for arg in operand.args):
        names = [arg.id for arg in operand.args if isinstance(arg, ast.Name)]
    if names and set(names) <= set(CARTESIAN) and len(names) >= 2:
        return names
    return None
