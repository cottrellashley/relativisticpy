"""The reader: what each name and each ambiguous form of a statement means.

Names are resolved against the scope, the session and the coordinates. Every ambiguous
reading is decided here: a definition or a claim, notation or components, a product or a
call, an index or a power, a run of letters, a constant or a symbol, a function named
without its argument. Where the less likely reading is taken, the reader says so in a
note. The lowerer asks the reader as it walks a statement.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Sequence
from dataclasses import replace
from functools import reduce
from typing import Literal

from relativisticpy.compiler.builtins import BUILTINS, CONSTANTS, Builtin
from relativisticpy.compiler.passes import Term
from relativisticpy.compiler.scope import (
    Context,
    DifferentialMarker,
    FunctionBinding,
    FunctionReference,
    Operator,
    Scope,
    TensorBinding,
    VariableBinding,
)
from relativisticpy.compiler.session import GEOMETRY_POSITIONS, describe
from relativisticpy.errors import CompileError, IndexNotationError, ParseError, RelativisticPyError
from relativisticpy.ir import SCALAR, VALUE, Node, TensorType, ValueType, ops
from relativisticpy.language import syntax as ast
from relativisticpy.language.parser import split_index_labels
from relativisticpy.language.source import Span
from relativisticpy.language.vocabulary import GREEK
from relativisticpy.reading.names import (
    _BUILT_IN_SCALARS,
    _BUILTIN_TENSORS,
    _CLAIMED,
    _CONCEPTS,
    _DECLARATIONS,
    _ROLE_TENSORS,
    _SHARED_NAMES,
    _SPACETIME_NAMES,
    _STATISTIC_ALIASES,
    _STATISTICS,
    _TRANSPOSES,
    _a_word,
    _built_in,
    _close,
    _decoration,
    _greek_labels,
    _has_a_value,
    _is_line_element,
    _letter,
    _mark,
    _mentions_symbol,
    _named,
    _names_builtin,
    _names_inside,
    _other_names,
    _splittable,
    _styled,
    _subscripted_name,
    _unprimed,
    _writes_tensor,
    _written_with,
)
from relativisticpy.tensors import Index, Indices
from relativisticpy.tensors.spaces import Space
from relativisticpy.typing import notation as rules
from relativisticpy.typing.syntax import (
    _indexed_name,
    _plain_label,
    _tensor_pattern,
    _used,
    _walk,
    _written,
)
from relativisticpy.typing.typer import _EXAMPLE_LABELS, TensorSource, Typer, _listed


class Reader(Typer):
    """What each name and each ambiguous form means, decided as the lowerer meets it, and
    the notes on how a statement was read where that could surprise."""

    def _reading(
        self, equation: ast.Equation, scope: Scope, ctx: Context
    ) -> tuple[Literal["define", "function", "claim", "notation"], str | None]:
        """Whether ``lhs = rhs`` defines its left side, and if not, the note that says why
        it is a claim where it could have been read as a definition.

        ``=`` defines only what has no meaning yet: a new name, a new function written
        with plain names as its arguments, ``f(x) = x^2``, a new tensor, or the metric,
        by a line element or while there is none. A name the metric in force is written
        with has no value, so ``\\Delta = ...`` defines it, as "where" does after a line
        element. Everything else is a claim, to check: a name, tensor or function that
        already has a value (``:=`` gives it a new one), a component such as ``g_{tt}``, a
        built-in such as ``R``, and a constant such as ``\\pi``. A name in its own value, as
        in ``R = 2/R^2``, makes an equation, never a definition."""
        target, value = _named(_unprimed(equation.lhs)), equation.rhs
        written = equation.lhs.span.text
        if _is_line_element(target):
            return "define", None
        if isinstance(target, ast.Prime):  # t' = \gamma(t - vx), a new coordinate
            primed = target.func.id + "'" * target.order
            if isinstance(scope.lookup(primed), VariableBinding):
                return "claim", _has_a_value(written)
            if self._names_a_symbol(target, scope) and not self._in_own_value(
                primed, value, scope, ctx
            ):
                return "define", None
            return "claim", None
        if isinstance(target, ast.Name):
            return self._name_reading(target, written, value, scope, ctx)
        if isinstance(target, ast.Call):
            return self._function_reading(target, value, scope, ctx)
        if not isinstance(target, ast.Scripted) or not isinstance(target.base, ast.Name):
            return "claim", None
        if any(s.indices is None for s in target.scripts):
            return "claim", None
        items = [item for s in target.scripts for item in s.indices.items]  # type: ignore[union-attr]
        if not all(_plain_label(item) for item in items):
            return "claim", None
        labels = [str(item.label) for item in items]
        if len(set(labels)) != len(labels) or any(map(self._names_component, labels)):
            return "claim", None  # a component, or a trace: an equation about what is there
        name = target.base.id
        if name == self.session.names["metric"] and len(items) == 2:
            space = self._labels.get(labels[0]) or self.session.labels.get(labels[0])
            if space is not None and not space.spacetime:
                in_force = space in self.session.space_metrics
            else:
                in_force = self.session.metric is not None
            if not in_force:
                return "define", None
            return "claim", _built_in(written, "the metric in force", f"{written} := ...")
        binding = scope.lookup(name)
        if binding is not None:  # u^\mu = [2, 0] after u^\mu = [1, 0]
            tensor = isinstance(binding, TensorBinding) or (
                isinstance(binding, VariableBinding) and isinstance(binding.node.type, TensorType)
            )
            return "claim", _has_a_value(written) if tensor else None
        if self._builtin_tensor(name, len(items)) is not None:
            return "claim", None
        if _writes_tensor(value, name):  # T_{\mu\nu} = T_{\nu\mu}: a symmetry, not a definition
            return "claim", None
        return "define", None

    def _name_reading(
        self, target: ast.Name, written: str, value: ast.Expr, scope: Scope, ctx: Context
    ) -> tuple[Literal["define", "claim", "notation"], str | None]:
        """How ``x = ...`` is read: see :meth:`_reading`."""
        name = target.id
        if name in _DECLARATIONS:  # Coordinates = [t, r] declares, as := does
            return "define", None
        if name in _CLAIMED:  # \pi = 3.14159: a claim about the constant
            return "claim", _built_in(written, "a constant", f"{written} := ...")
        if scope.lookup(name) is not None:  # x = 3 after x := 2
            return "claim", _has_a_value(written)
        if _indexed_name(target, scope) is not None:  # y_t of a tensor y: a component
            return "claim", None
        quantity = self._builtin_scalar(name)
        if quantity is not None:  # R = 0 with a metric in force
            return "claim", _built_in(written, _BUILT_IN_SCALARS[quantity], f"{written} := ...")
        ricci = name in ("R", self.session.names["ricci"])
        if ricci and self.session.metric is None:  # R = -\kappa T, as notation
            return "notation", _built_in(written, "the Ricci scalar", f"{written} := ...")
        if self._in_own_value(name, value, scope, ctx):
            return "claim", (
                f"{written} appears on both sides, so this is an equation, not a "
                f"definition of {written}"
            )
        return "define", None

    def _function_reading(
        self, target: ast.Call, value: ast.Expr, scope: Scope, ctx: Context
    ) -> tuple[Literal["function", "claim"], str | None]:
        """How ``f(x) = ...`` is read: a definition of a function with the arguments as
        its parameters, where they are plain names and ``f`` has no meaning yet; a claim
        otherwise, as ``f(2) = 4`` and ``r(\\tau) = ...`` of a coordinate are."""
        func = target.func
        if not isinstance(func, ast.Name):
            return "claim", None
        name = func.id
        binding = scope.lookup(name)
        if binding is not None:
            params = binding.definition.params if isinstance(binding, FunctionBinding) else ()
            again = f"{name}({', '.join(params)}) := ..." if params else f"{name} := ..."
            note = (
                f"{func.span.text} already has a value, so = compares it; to give "
                f"{func.span.text} a new value, write {again}"
            )
            return "claim", note
        names = [arg.id for arg in target.args if isinstance(arg, ast.Name)]
        if not names or len(names) != len(target.args) or len(set(names)) != len(names):
            return "claim", None  # f(2) = 4: a claim about the value at 2
        if ctx.latex and _splittable(func) and name not in BUILTINS:
            return "claim", None  # 2Gm(r) = ...: a product of letters
        coordinates = {c.name for c in self.session.coordinates or ()}
        if name in coordinates or self._never_a_function(func, ctx):
            return "claim", None  # r(\tau) = ...: a coordinate along a path
        if name in self._factors:  # G(p^2 + q^2) beside 2GM: a product
            return "claim", None
        if any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
            for node in _walk(value)
        ):
            note = (
                f"{func.span.text} appears on both sides, so this is an equation, not a definition"
            )
            return "claim", note
        return "function", None

    def _never_a_function(self, func: ast.Name, ctx: Context) -> bool:
        """Whether a name, bound to nothing in the program, is never a function of what is
        in brackets after it: a builtin is one already, and a constant, such as ``e``, ``i``
        or one declared with ``Constants := [G]``, multiplies them."""
        name = func.id
        if _names_builtin(func, latex=ctx.latex) or name in CONSTANTS:
            return True
        declared = self.session.assumptions.get(name)
        return declared is not None and bool(declared.assumptions0.get("constant"))

    def _in_own_value(self, name: str, value: ast.Expr, scope: Scope, ctx: Context) -> bool:
        """Whether ``name`` is written in ``value`` other than as a function called there,
        as R is in 2/R^2 but \\beta is not in \\beta(r), or is in the value of a name
        written there, as \\rho is in p after p := w\\rho."""
        other = _other_names(value)  # f in f(x), x in x' and h in \bar h name other things
        for node in _walk(value):
            if isinstance(node, ast.Prime):  # x' is a name of its own, maybe written with x
                written = node.func.id + "'" * node.order
            elif isinstance(node, ast.Accent | ast.Scripted) and isinstance(
                named := _named(node), ast.Name
            ):
                written = named.id  # \bar h, E^{(2)}
            elif isinstance(node, ast.Name) and id(node) not in other:
                written = node.id
            else:
                continue
            if written == name:
                return True
            binding = scope.lookup(written)
            if binding is None and isinstance(node, ast.Name) and ctx.latex and _splittable(node):
                known = node.id in BUILTINS or node.id in CONSTANTS
                if not known and name in split_index_labels(node.id):
                    return True  # K in 2GK
            if isinstance(binding, VariableBinding | TensorBinding) and _mentions_symbol(
                binding.node, name
            ):
                return True
        return False

    def _states_identity(self, statement: ast.Define, scope: Scope) -> bool:
        """Whether a definition written with ``\\equiv`` says what a built-in already is,
        as ``R_{\\mu\\nu} \\equiv R^\\alpha{}_{\\mu\\alpha\\nu}`` does: then it is a claim to
        check, which leaves the built-in as it is, rather than a new meaning for its name.
        The metric is defined this way, as any new tensor is."""
        if len(statement.targets) != 1:
            return False
        target, value = statement.targets[0], statement.value
        written = Span(statement.span.source, target.span.end, value.span.start).text
        if "\\equiv" not in written and "≡" not in written:
            return False
        if isinstance(target, ast.Name):
            return scope.lookup(target.id) is None and self._builtin_scalar(target.id) is not None
        pattern = _tensor_pattern(target) if isinstance(target, ast.Scripted) else None
        if pattern is None or scope.lookup(pattern[0]) is not None:
            return False
        return self._builtin_tensor(pattern[0], len(pattern[1])) not in (None, "metric")

    def _names_component(self, label: str) -> bool:
        """Whether an index label names a component: a coordinate's name, as ``t`` in
        ``g_{tt}``, unless the label ranges over a space other than spacetime."""
        space = self._labels.get(label) or self.session.labels.get(label)
        if space is not None and not space.spacetime:
            return False
        return any(c.name == label for c in self.session.coordinates or ())

    def _along_a_path(self, node: ast.Derivative, scope: Scope) -> str | None:
        """The coordinate a derivative is of, when it is by a parameter that is no
        coordinate, as dt/d\\lambda along a geodesic is: the coordinate is no function
        of it, so the derivative would be zero, which is not what is meant."""
        coordinates = {c.name for c in self.session.coordinates or ()}

        def free(node: ast.Expr | None) -> str | None:
            if isinstance(node, ast.Name) and scope.lookup(node.id) is None:
                return node.id
            return None

        coordinate = free(node.expr)
        if coordinate is None or coordinate not in coordinates:
            return None
        parameters = [free(variable) for variable, _ in node.variables]
        along = all(p is not None and p not in coordinates for p in parameters)
        return node.expr.span.text if along and node.expr is not None else None

    def _names_coordinate(self, target: ast.Expr, scope: Scope) -> bool:
        """Whether ``target`` is the bare name of a coordinate in force, bound to nothing
        else in the program."""
        return (
            isinstance(target, ast.Name)
            and scope.lookup(target.id) is None
            and any(c.name == target.id for c in self.session.coordinates or ())
        )

    def _names_a_symbol(self, prime: ast.Prime, scope: Scope) -> bool:
        """Whether a primed name stands for a symbol of its own, as t' does, rather than
        the derivative of a function, as A' does where A(r) is written."""
        name = prime.func.id
        if isinstance(scope.lookup(name + "'" * prime.order), VariableBinding):
            return True
        return (
            scope.lookup(name) is None
            and len(self.session.functions.get(name, ())) != 1
            and not _names_builtin(prime.func)
        )

    def _written_in_metric(self, name: str) -> bool:
        """Whether the metric in force is written with ``name`` as a symbol or a function
        that has no value, as with \\Delta in ``ds^2 = -\\Delta dt^2 + ...``: a definition of
        it then gives it that value there too. A coordinate never is. It is read from what
        the metric is computed from, without computing it (see :func:`_written_with`), so a
        name that cancels, as x in 1 + x - x, counts too."""
        metric = self.session.metric
        if metric is None or name not in metric.names:
            return False
        if any(c.name == name for c in self.session.coordinates or ()):
            return False
        return name in _written_with(metric.node)

    def _declares(self, relation: ast.Relation) -> bool:
        """``a, b \\in \\mathcal{A}``: declare that the labels range over a space, and say
        so; ``False`` for any other relation, which is read but not computed. Only a space
        defined by name declares its members: in a paper, ``x \\in \\mathbb{R}^3`` says
        what x is, and x may well be a coordinate."""
        names = relation.lhs.items if isinstance(relation.lhs, ast.Tuple) else (relation.lhs,)
        if relation.op != "∈" or not all(isinstance(n, ast.Name) for n in names):
            return False
        rhs = relation.rhs
        space = self.session.spaces.get(rhs.id) if isinstance(rhs, ast.Name) else None
        if space is None:
            return False
        for label in names:
            assert isinstance(label, ast.Name)
            self.session.labels[label.id] = space
        return True

    def _is_notation(
        self,
        node: ast.Expr,
        scope: Scope,
        ctx: Context,
        *,
        equation: bool = False,
        operation: bool = False,
    ) -> bool:
        """Whether an expression is to be read as notation: it mentions a new tensor (a
        name with two or more indices and no components), or, without a metric, something
        kept as notation.

        In an ``equation`` or a definition, or the argument of an ``operation`` on
        notation such as ``simplify``, built-in tensors count too when there is no
        metric, and so do new tensors with one index when a contraction, as in
        u^\\alpha u_\\alpha, shows that index notation is in use; but only where the
        indices follow the rules of the notation and none is a coordinate, since
        \\Gamma^\\theta_{r\\theta} = 1/r is a component. A tensor written with a coordinate
        as an index never counts: ``G_{tt} = 8\\pi T_{tt}`` is about components, which T
        has not yet; nor does the tensor ``solve`` is asked to solve for. In an operation,
        something kept as notation always counts."""
        builtin = single = False
        labels: set[str] = set()
        for current in _used(node):
            if isinstance(current, ast.Name):
                binding = scope.lookup(current.id)
                notation = (
                    isinstance(binding, VariableBinding)
                    and rules.notation_of(binding.node) is not None
                )
                # With a metric, notation is computed where it meets components.
                if notation and (operation or self.session.metric is None):
                    return True
            elif isinstance(current, ast.Scripted) and (pattern := _tensor_pattern(current)):
                written = self._letter_run(current, scope, ctx)
                if isinstance(written, ast.Binary):  # G T_{\mu\nu}: the tensor is the last
                    assert isinstance(written.right, ast.Scripted)
                    pattern = _tensor_pattern(written.right) or pattern
                name, entries = pattern
                if any(self._names_component(label) for label, _ in entries):
                    return False
                labels.update(label for label, _ in entries)
                if scope.lookup(name) is None and name not in BUILTINS and name not in CONSTANTS:
                    quantity = self._builtin_tensor(name, len(entries))
                    first = entries[0][0]
                    over = self._labels.get(first) or self.session.labels.get(first)
                    if quantity in ("kronecker", "metric") and over and not over.spacetime:
                        pass  # the identity or metric of a space of data: it has components
                    elif quantity is not None:
                        builtin = True
                    elif any(self._builtin_tensor(name, rank) for rank in range(5)):
                        pass  # a built-in with the wrong number of indices: an error
                    elif len(entries) >= 2:
                        return True
                    else:
                        single = True
        builtin = builtin and self.session.metric is None
        if not (equation or operation) or not (builtin or single):
            return False
        if any(c.name in labels for c in self.session.coordinates or ()):
            return False
        try:
            _, summed = self._index_structure(node, scope)
        except IndexNotationError:
            return False
        return builtin or summed

    def _letter_run(self, node: ast.Scripted, scope: Scope, ctx: Context) -> ast.Expr:
        """In LaTeX, ``8\\pi GT_{\\mu\\nu}`` is G times T_{\\mu\\nu}: the indices on a run of
        letters that names nothing belong to its last letter, as a script does."""
        base = node.base
        if not (ctx.latex and isinstance(base, ast.Name) and _splittable(base)):
            return node
        name = base.id
        known = name in BUILTINS or name in CONSTANTS or scope.lookup(name) is not None
        if known or any(self._builtin_tensor(name, rank) for rank in range(5)):
            return node
        letters = split_index_labels(name)
        if len(letters) < 2:
            return node
        factors: list[ast.Expr] = [
            _letter(letter, base.span, first=n == 0) for n, letter in enumerate(letters[:-1])
        ]
        factors.append(replace(node, base=_letter(letters[-1], base.span, first=False)))
        return reduce(lambda a, b: ast.Binary("*", a, b, implicit=True, span=node.span), factors)

    def _Name(self, node: ast.Name, scope: Scope, ctx: Context) -> Term:
        name = node.id
        binding = scope.lookup(name)
        if isinstance(binding, VariableBinding | TensorBinding):
            kept = self._kept(binding.node)
            if kept is not None and kept is not binding.node:  # kept in another algebra
                return kept
            return binding.node
        if binding is None and (indexed := _indexed_name(node, scope)) is not None:
            return self.term(indexed, scope, ctx)  # y_t, where y is a tensor: y_{t}
        if isinstance(binding, FunctionBinding):
            return FunctionReference(node.span, name)
        if ctx.line_element and name == "d":
            return DifferentialMarker(node.span)
        if name == "d" and node.verbatim:  # {\rm d}t = ... outside a line element
            raise CompileError(
                "a relation between differentials such as {\\rm d}t is read but not computed",
                span=node.span,
            )
        session = self.session
        if name == "Coordinates" and session.coordinates is not None:
            return self.graph.const(list(session.coordinates), VALUE, node.span)
        if (quantity := self._builtin_scalar(name)) is not None:
            return self._geometry(quantity, node.span)
        self._note_symbol(node)
        if name in session.assumptions:
            return self.graph.const(session.assumptions[name], SCALAR, node.span)
        if name in CONSTANTS:
            return self.graph.const(CONSTANTS[name], SCALAR, node.span)
        if (arguments := session.functions.get(name)) is not None:  # \alpha is \alpha(t, r)
            args = [self.graph.const(argument, SCALAR) for argument in arguments]
            return self._call("core.function", args, SCALAR, node.span, (("name", name),))
        # In LaTeX a run of letters is a product, as typeset maths reads it: re^{x} is
        # r e^{x}. A builtin keeps its name when called, as in diff(f, x).
        if ctx.latex and _splittable(node):
            return self._latex_letters(node, scope, ctx)
        # A builtin applies to what follows it, as sin does in sin x; in LaTeX only if it
        # is written as a command, \operatorname{sign} x, so that N dt^2 is a product.
        if _names_builtin(node, latex=ctx.latex) and (node.verbatim or not ctx.latex):
            return FunctionReference(node.span, name)
        return self.graph.const(session.symbol(name), SCALAR, node.span)

    def _note_symbol(self, node: ast.Name) -> None:
        """Say so where a name bound to nothing is read as a symbol though it could be
        taken for something else: R and g where the metric is written with them, or
        where the metric was dropped, and the spelled-out names of built-ins, as
        RicciScalar."""
        name, written, session = node.id, node.span.text, self.session
        built_in = name in ("R", session.names["ricci"], session.names["metric"])
        if built_in and session.metric is not None and name in session.metric.names:
            meant = "the Ricci scalar is Ric" if name == "R" else "its determinant is det(g_{ab})"
            if name != session.names["ricci"]:
                self._note(
                    f"the metric is written with {written}, so {written} is that symbol; {meant}"
                )
        elif built_in and session.metric is None and session.dropped is not None:
            self._note(f"{written} is a symbol here: {session.dropped_note()[0]}")
        concept = _CONCEPTS.get(name.lower())
        if concept is not None and name not in BUILTINS:
            self._note(f"{written} is not defined, so it is a symbol; did you mean {concept}?")

    def _builtin_scalar(self, name: str) -> str | None:
        """The geometric quantity that ``name``, bound to nothing in the program, denotes
        with a metric in force: the metric determinant ``g``, the Ricci scalar ``Ric`` or
        ``R``, or the Kretschmann scalar. A name the metric itself is written with, such as
        the radius R of a sphere, or a coordinate's name, stays a symbol; ``Ric`` is always
        the Ricci scalar."""
        session = self.session
        if session.metric is None:
            return None
        own = name in session.metric.names or any(c.name == name for c in session.coordinates or ())
        if name == session.names["metric"] and not own:
            return "determinant"
        if name == session.names["ricci"] or (name == "R" and not own):
            return "ricci_scalar"
        if name == "Kretschmann":
            return "kretschmann"
        return None

    def _latex_letters(self, node: ast.Name, scope: Scope, ctx: Context) -> Node:
        r"""In LaTeX, ``GM`` is :math:`G \cdot M`: one factor per letter (or Greek name)."""
        factors = []
        offset = node.span.start if node.span.text == node.id else None
        for position, label in enumerate(split_index_labels(node.id)):
            span = node.span
            if offset is not None:
                span = Span(node.span.source, offset, offset + len(label))
                offset += len(label)
            factors.append(self.value(_letter(label, span, first=position == 0), scope, ctx))
        return self._multiply(factors, node.span)

    def _builtin_tensor(self, name: str, rank: int | None) -> str | None:
        """The geometric quantity a built-in tensor name denotes with ``rank`` indices."""
        names = self.session.names
        if name == names["metric"]:
            return "metric" if rank == 2 else None
        candidates: dict[int, str] = {}
        for role, by_rank in _ROLE_TENSORS.items():
            if names[role] == name:
                candidates.update(by_rank)
        for count, quantity in _BUILTIN_TENSORS.get(name, {}).items():
            candidates.setdefault(count, quantity)
        if name == "C":  # most texts write the Weyl tensor C_{abcd}
            candidates.setdefault(4, "weyl")
        return candidates.get(rank) if rank is not None else None

    def _built_in_as(self, name: str, rank: int) -> str:
        """What a built-in tensor's name means with ``rank`` indices, as it is written:
        the Christoffel connection C^{a}_{bc}."""
        quantity = self._builtin_tensor(name, rank) or ""
        positions = GEOMETRY_POSITIONS.get(quantity, (False,) * rank)
        labels = "abcd"[:rank]
        indices = [Index(label, upper) for label, upper in zip(labels, positions, strict=True)]
        return f"{describe(quantity)} {name}{self._written_indices(indices)}"

    def _tensor_source(
        self,
        name: str,
        binding: object,
        rank: int | None,
        span: Span,
        *,
        strict: bool = False,
        node: ast.Scripted | None = None,
    ) -> TensorSource | None:
        """What ``name`` with ``rank`` indices refers to, if it is a tensor. With
        ``strict`` (the scripts are index labels), a built-in tensor given the wrong
        number of indices is an error rather than an ordinary symbol. A name that is also
        common for a tensor of one's own, as C for a covariance, says what it is built in
        as, and how to use it for one's own."""
        if isinstance(binding, TensorBinding):
            return TensorSource(binding.node, binding.indices, name)
        if isinstance(binding, VariableBinding) and isinstance(binding.node.type, TensorType):
            return TensorSource(binding.node, binding.node.type.indices, name)
        if binding is not None or rank is None:
            return None
        written = node.base.span.text if node is not None else name
        quantity = self._builtin_tensor(name, rank)
        if quantity is None:
            ranks = [n for n in range(1, 5) if self._builtin_tensor(name, n)]
            if ranks and strict:
                expected = " or ".join(str(n) for n in ranks)
                notes: list[str] = []
                hint = None
                if name in _SHARED_NAMES:  # W_{ij}, a matrix of weights
                    built = _listed([self._built_in_as(name, n) for n in ranks], "or")
                    notes.append(f"{written} is a built-in name: {built}")
                    own = node.span.text if node is not None else written
                    hint = f"to use {written} for a tensor of your own, define it first: "
                    hint += f"{own} := [...]"
                raise IndexNotationError(
                    f"{written} has {expected} indices, but {rank} "
                    f"{'is' if rank == 1 else 'are'} given",
                    span=span,
                    notes=notes,
                    hint=hint,
                )
            return None
        try:
            geometry = self._geometry(quantity, span)
        except CompileError as error:  # G_{kl} = X_{tk} X_{tl}: G is the Einstein tensor
            here = node.span.text if node is not None else written
            error.notes.append(f"{here} is {describe(quantity)}, which is built in")
            pattern = _tensor_pattern(node) if node is not None else None
            if pattern is not None and any(
                (space := self._labels.get(label) or self.session.labels.get(label)) is not None
                and not space.spacetime
                for label, _ in pattern[1]
            ):  # its indices range over data: a tensor of one's own is meant
                error.hint = f"to use {written} for a tensor of your own, define it: {here} := ..."
            raise
        assert isinstance(geometry.type, TensorType)
        return TensorSource(geometry, geometry.type.indices, name)

    def _scalar_source(
        self, base: ast.Name, binding: object, scope: Scope, ctx: Context
    ) -> TensorSource | None:
        """A scalar written with derivative indices only, as ``\\phi_{;\\mu}``: one defined
        in the program, a built-in one such as ``R``, or a function of the coordinates
        written once with its arguments. It has no indices of its own."""
        if isinstance(binding, VariableBinding) and binding.node.type == SCALAR:
            return TensorSource(binding.node, Indices(), base.id)
        written = self._builtin_scalar(base.id) is not None or base.id in self.session.functions
        if binding is None and written:
            return TensorSource(self.value(base, scope, ctx), Indices(), base.id)
        return None

    def _space_tensor(
        self, name: str, entries: Sequence[tuple[str, bool]], span: Span
    ) -> TensorSource | None:
        """The Kronecker delta and the metric of a space other than spacetime: the
        identity, or the metric declared for the space, as in g_{ab} := \\Sigma_{ab}. Without
        coordinates, the metric written with labels of no known space is that of the one
        space given a metric, if one is."""
        quantity = self._builtin_tensor(name, len(entries))
        if quantity not in ("kronecker", "metric"):
            return None
        space = self._labels.get(entries[0][0]) or self.session.labels.get(entries[0][0])
        if space is None and quantity == "metric":
            space = self._only_space_metric()
        if space is None:  # spacetime's, or an error that says what the index needs
            what = "the metric" if quantity == "metric" else "the Kronecker delta"
            space = self._space_of(entries[0][0], span, what)
        if space.spacetime:
            return None
        declared = self.session.space_metrics.get(space)
        if quantity == "metric" and declared is not None:
            assert isinstance(declared.type, TensorType)
            return TensorSource(declared, declared.type.indices, name)
        upper = quantity == "kronecker"
        indices = Indices([Index("a", upper), Index("b")])
        node = self.graph.add(
            ops.Call("core.identity"),
            (self.graph.const(space.dim, VALUE),),
            TensorType(indices, (space, space)),
            span,
        )
        return TensorSource(node, indices, name)

    def _only_space_metric(self) -> Space | None:
        """The one space given a metric, where there are no coordinates and so no metric
        of spacetime that g could also be; ``None`` where there are none or several."""
        defined = list(self.session.space_metrics)
        return defined[0] if len(defined) == 1 and self.session.coordinates is None else None

    def _undefined(self, base: ast.Expr, scope: Scope) -> bool:
        """Whether a name means nothing yet, so that scripts on it name a symbol or raise it
        to a power: bound to nothing, and no coordinate, constant, declared symbol, function
        of the coordinates or built-in scalar."""
        if not isinstance(base, ast.Name) or scope.lookup(base.id) is not None:
            return False
        name, session = base.id, self.session
        return not (
            any(c.name == name for c in session.coordinates or ())
            or name in CONSTANTS
            or name in session.assumptions
            or name in session.functions
            or self._builtin_scalar(name) is not None
        )

    def _index_written(self, script: ast.Script) -> str | None:
        """The label of a superscript that the statement shows to be an index: one it
        also writes as a subscript, as in x^\\mu x_\\mu, or on a tensor, or a Greek
        letter it raises on more than one name, as in p^\\mu N^\\nu = n m U^\\mu U^\\nu.
        A Greek superscript written once is a power, as in t^\\alpha."""
        items = script.indices.items if script.indices is not None else ()
        if len(items) != 1 or not _plain_label(items[0]):
            return None
        label = str(items[0].label)
        lowered, raised = self._scripts["_"][label], self._scripts["^"][label]
        index = lowered or label in self._labels or (label in GREEK and raised > 1)
        return label if index else None

    def _check_component(self, node: ast.Scripted, base: ast.Name, source: TensorSource) -> None:
        """Refuse a tensor with one index written with a bare number of 2 or more as its
        superscript, as L^2: its component with upper index 2, or L squared? Where there is
        no such component, as with L^4 and four coordinates, it is a power, refused as one.
        In braces, L^{2}, the number is an index, and L^0 and L^1 are never powers."""
        if len(source.indices) != 1 or len(node.scripts) != 1:
            return
        script = node.scripts[0]
        if script.kind != "^" or not re.fullmatch(r"\^\s*[2-9]", script.span.text):
            return
        name, number = base.span.text, script.span.text.lstrip("^ ")
        if len(split_index_labels(name.lstrip("\\"))) > 1:  # L of GML^2, a run of letters
            name = self._written_label(base.id)
        kind = source.node.type
        dim = kind.spaces[0].dim if isinstance(kind, TensorType) and kind.spaces else None
        if isinstance(dim, int) and int(number) >= dim:
            raise self._tensor_power(None, source.indices, None, node.span)
        raise IndexNotationError(
            f"{name} is a tensor, so {name}^{number} could be its component with upper index "
            f"{number} or a power of it",
            span=node.span,
            hint=f"for the component, write {name}^{{{number}}}; to multiply {name} by itself, "
            f"contract it, as in {name}_{{a}} {name}^{{a}}",
        )

    def _check_tensor_power(self, node: ast.Scripted, base: ast.Name, binding: object) -> None:
        """Refuse a tensor written with its indices and then raised to a power, as
        g_{\\mu\\nu}^{-1}, which would otherwise be read as a symbol g_{\\mu\\nu}."""
        indexed = [s for s in node.scripts if s.indices is not None and s.indices.items]
        powers = [s for s in node.scripts if s.indices is None and s.kind == "^"]
        if not indexed or not powers or len(indexed) + len(powers) != len(node.scripts):
            return
        items = [(item, s.kind == "^") for s in indexed for item in s.indices.items]  # type: ignore[union-attr]
        if not all(_plain_label(item) for item, _ in items):
            return
        tensor = isinstance(binding, TensorBinding) or (
            isinstance(binding, VariableBinding) and isinstance(binding.node.type, TensorType)
        )
        quantity = self._builtin_tensor(base.id, len(items)) if binding is None else None
        if not tensor and quantity is None:
            return
        indices = [Index(str(item.label), upper) for item, upper in items]
        exponent = powers[0].body.span.text if powers[0].body is not None else None
        metric = base.id if quantity == "metric" else None
        raise self._tensor_power(metric, indices, exponent, node.span)

    def _underivable(
        self, node: ast.Scripted, base: ast.Name, binding: object, rank: int | None
    ) -> CompileError:
        """The error for a comma or semicolon derivative of what has none to take."""
        written = base.span.text
        if binding is not None:
            return CompileError(
                f"{written} is not a scalar or a tensor, so it has no derivative", span=node.span
            )
        if not rank:
            return CompileError(
                f"{written} is not defined, so it has no derivative",
                span=node.span,
                hint=f"define it first, for example {written} := ...",
            )
        # T^{\mu\nu}_{;\nu}: the tensor as written, without the derivative's index
        own = [
            Index(str(item.label), script.kind == "^")
            for script in node.scripts
            if script.indices is not None
            for item in script.indices.items
            if item.derivative is None and item.label is not None
        ]
        example = (
            self._written_indices(own) if own else "_{" + "".join(_EXAMPLE_LABELS[:rank]) + "}"
        )
        return CompileError(
            f"{written} is not a tensor",
            span=node.span,
            hint=f"define it first, for example {written}{example} := [...]",
        )

    def _not_a_tensor(
        self,
        node: ast.Scripted,
        base: ast.Name,
        subscripts: Sequence[ast.Script],
        items: Sequence[ast.IndexItem],
    ) -> CompileError:
        """``T_{ab}`` where ``T`` names no tensor: define it first, or, where the
        subscript is a word, as in ``T_{eff}``, it names a symbol, written so."""
        name = base.span.text
        written = _written(subscripts)
        run = written.lstrip("_").strip("{}")
        if len(subscripts) == 1 and _a_word(run):  # T_{eff}: a name, not three indices
            labels = _listed([str(item.label) for item in items])
            return CompileError(
                f"{name} is not a tensor",
                span=node.span,
                notes=[f"in braces each letter is an index: {labels}"],
                hint=f"to name a symbol, write {name}_{{\\rm {run}}} or {name}_{{\\text{{{run}}}}}",
            )
        if any(self._names_component(str(item.label)) for item in items):  # T_{tt}
            written = "_{" + "".join(_EXAMPLE_LABELS[: len(items)]) + "}"
        return CompileError(
            f"{name} is not a tensor",
            span=node.span,
            hint=f"define it first, for example {name}{written} := [...]",
        )

    def _note_power(self, base: ast.Name, script: ast.Script) -> None:
        """Say so where a superscript on a name that means nothing looks like indices, a
        Greek label or several, as in T^{\\mu\\nu}, but is read as a power; or where the
        statement also writes the name with the label as a subscript, as u^a u_a."""
        written = base.span.text
        items = script.indices.items if script.indices is not None else ()
        if not items or not all(_plain_label(item) for item in items):
            return
        labels = [str(item.label) for item in items]
        power = f"{written}{script.span.text}"
        tensor = f"{written}^{{{''.join(self._written_label(label) for label in labels)}}}"
        if any(label in GREEK for label in labels) or len(labels) > 1:
            self._note(
                f"{written} is not defined, so {power} is read as {written} to a power; to use "
                f"a tensor, define it first: {tensor} := [...]"
            )
        elif len(labels) == 1 and f"{base.id}_{labels[0]}" in self._factors:  # u^a u_a
            self._note(
                f"{written} is not defined, so {power} is a power and {written}_{labels[0]} one "
                f"symbol; to use a tensor, define it first: {tensor} := [...]"
            )

    def _script_math(self, node: ast.Scripted, scope: Scope, ctx: Context) -> Node:
        """Scripts on something that is not a tensor: subscripts name a symbol
        (``r_{s}`` is ``r_s``) and superscripts are powers."""
        base, scripts = node.base, list(node.scripts)
        if ctx.latex and isinstance(base, ast.Name) and _splittable(base):
            whole = self.term(base, scope, ctx)
            labels = split_index_labels(base.id)
            if (
                isinstance(whole, Node)
                and isinstance(whole.op, ops.Mul | ops.Scale)  # Scale: a letter is a tensor
                and len(whole.args) == len(labels)
            ):
                # re^{x} is r e^{x}: in LaTeX a script binds to the last letter only, which
                # may be a tensor, as in Rg_{\mu\nu}, or GML^2 where L is one
                last = _letter(labels[-1], base.span, first=False)
                rest = self.value(replace(node, base=last), scope, ctx)
                return self._multiply([*whole.args[:-1], rest], node.span)
        decorations = [s for s in scripts if s.kind == "^" and _decoration(s)]
        if decorations and isinstance(base, ast.Name):  # r^* is a name, not a power
            marks = "".join(_mark(s) for s in decorations)
            base = ast.Name(f"{base.id}^{marks}", verbatim=True, span=base.span)
            scripts = [s for s in scripts if s not in decorations]
        for script in scripts:  # x_{} or x^{}: nothing to read
            if script.body is None and (script.indices is None or not script.indices.items):
                where = "subscript" if script.kind == "_" else "superscript"
                raise CompileError(
                    f"empty {where}", span=script.span, hint="write what it holds, or remove it"
                )
        subscripts = [s for s in scripts if s.kind == "_"]
        powers = [s for s in scripts if s.kind == "^"]
        written_base = base.span.text if isinstance(base, ast.Name) else "this"
        if subscripts:
            if not isinstance(base, ast.Name):
                raise CompileError("only a name can have a subscript", span=node.span)
            items = [i for s in subscripts if s.indices for i in s.indices.items]
            if len(items) > 1 and all(_plain_label(i) for i in items):
                raise self._not_a_tensor(node, base, subscripts, items)
            if self._undefined(base, scope) and _greek_labels(subscripts):  # u_\alpha
                shown = node.span.text if not powers else written_base + _written(subscripts)
                self._note(
                    f"{written_base} is not defined, so {shown} is one symbol, not a "
                    f"component; to use a tensor, define it first: {shown} := [...]"
                )
            symbol_name = _subscripted_name(ast.Scripted(base, tuple(subscripts), span=node.span))
            value = self.value(ast.Name(symbol_name, verbatim=True, span=node.span), scope, ctx)
        else:
            value = self.value(base, scope, ctx)
        exponents = []
        for script in powers:
            if script.body is None or isinstance(script.body, ast.Equation):
                raise CompileError(f"{written_base} is not a tensor", span=script.span)
            if isinstance(base, ast.Name) and (label := self._index_written(script)):
                shown = self._written_label(label)
                raise CompileError(
                    f"{written_base} is not a tensor, so {shown} on it is not an index",
                    span=node.span,
                    hint=f"define it first, for example {written_base}^{{{shown}}} := [...]",
                )
            if not subscripts and isinstance(base, ast.Name) and self._undefined(base, scope):
                self._note_power(base, script)
            if (
                isinstance(script.body, ast.Name)
                and script.body.id in _TRANSPOSES
                and scope.lookup(script.body.id) is None
                and isinstance(value.type, ValueType)
            ):  # A^T of a matrix: a power, which is not what is meant
                raise CompileError(
                    f"{node.span.text} is read as {written_base} to the power "
                    f"{script.body.span.text}",
                    span=node.span,
                    notes=["with indices, the transpose of A_{ab} is A_{ba}"],
                    hint=f"for the transpose, write transpose({written_base})",
                )
            if (
                isinstance(base, ast.Name)
                and scope.lookup(base.id) is None
                and self._builtin_scalar(base.id) == "determinant"
            ):  # g^{-1} is 1/det g, which is easily taken for the inverse metric
                self._note(
                    f"{written_base} alone is the determinant of the metric, so {node.span.text} "
                    f"is a power of it; the inverse metric is {written_base}^{{\\mu\\nu}}"
                )
            exponents.append(self.value(script.body, scope, ctx))
        if not exponents:
            return value
        exponent = exponents[-1]
        for previous in reversed(exponents[:-1]):  # x^a^b is x^(a^b)
            exponent = self._power(previous, exponent, node.span)
        return self._power(value, exponent, node.span)

    def _a_product(
        self, func: ast.Name, args: tuple[ast.Expr, ...], binding: object, ctx: Context
    ) -> bool:
        """Whether ``name(...)`` multiplies what is in the brackets rather than calls a
        function. A constant never is a function, ``e`` and ``i`` as much as a name declared
        with ``Constants := [G]``; nor is a name the statement writes as a factor too, as G
        is in ``2GM + G(p^2 + q^2)``, unless it is called with the coordinates, as
        \\alpha(t, r) is; nor, with more than names in the brackets, one written as a factor
        before, as \\rho in \\rho(1 + 3w) after p = w\\rho. In LaTeX a name with a value is
        a factor too, as are the cases of :meth:`_a_factor`."""
        name = func.id
        if binding is not None:
            return ctx.latex and isinstance(binding, VariableBinding)  # y(3), y defined
        if _names_builtin(func, latex=ctx.latex):
            return False  # sin(x)
        if self._never_a_function(func, ctx):
            return True  # e(x + 1)
        coordinates = {c.name for c in self.session.coordinates or ()}
        names = [arg.id for arg in args if isinstance(arg, ast.Name)]
        if name in coordinates:
            return False  # r(\tau): a coordinate along a path, as the caller reads it
        if names and len(names) == len(args) and set(names) <= coordinates:
            return False  # \alpha(t, r): a function of the coordinates
        if name in self._factors:
            return True
        written = name in self.session.factors and name not in self.session.functions
        if written and len(names) != len(args):
            return True
        one_name = len(args) == 1 and isinstance(args[0], ast.Name)
        return ctx.latex and self._a_factor(name, args, one_name)

    def _a_factor(self, name: str, args: tuple[ast.Expr, ...], one_name: bool) -> bool:
        """Whether, in LaTeX, ``name(...)`` is a product: where the name is written inside
        its own brackets, as M is in M(r - 2M), since nothing is its own argument; or
        where it is a parameter of the metric, as M is in Schwarzschild's, and the
        brackets hold more than a name: M(1 - 2M/r) is M times 1 - 2M/r."""
        if any(_names_inside(name, arg) for arg in args):
            return True
        metric = self.session.metric
        return (
            metric is not None
            and name in metric.names
            and name not in self.session.functions
            and not one_name
        )

    def _note(self, note: str) -> None:
        """Say how the statement being compiled was read: once, however often it is."""
        if note not in self._notes:
            self._notes.append(note)

    def _nothing_to_act_on(self, operator_: Operator, scope: Scope) -> CompileError:
        """The error for an operator, or a function, written with nothing after it."""
        written = operator_.describe()
        if isinstance(operator_, FunctionReference):
            binding = scope.lookup(operator_.name)
            params = binding.definition.params if isinstance(binding, FunctionBinding) else ("x",)
            return CompileError(
                f"{written} is a function, so it needs an argument, as in "
                f"{written}({', '.join(params)})",
                span=operator_.span,
            )
        return CompileError(
            f"{written} needs something to act on, as in {written} f", span=operator_.span
        )

    def _star(self, node: ast.Unary) -> RelativisticPyError:
        """A ``*`` with nothing on its left: the Hodge dual, *F, which is not supported,
        where a statement or an operand starts with it; otherwise an operator written
        twice or a \\cdot with nothing to multiply, a typo."""
        written = node.span.text
        star = next(
            (command for command in ("\\cdot", "\\times", "\\ast") if written.startswith(command)),
            written[:1],
        )
        before = node.span.source.text[: node.span.start].rstrip()
        doubled = before != "" and (
            before[-1] in "+-*/·×" or before.endswith(("\\cdot", "\\times"))
        )
        if star == "*" and not doubled:  # *F
            return CompileError("the Hodge dual is not supported yet", span=node.span)
        hodge = "a * before a form, as in *F, is the Hodge dual, which is not supported yet"
        return ParseError(
            f"'{star}' has nothing on its left",
            span=Span(node.span.source, node.span.start, node.span.start + len(star)),
            hint="remove it, or write what it multiplies",
            notes=[hodge] if star == "*" else [],
        )

    def _membership(self, node: ast.Relation) -> CompileError:
        """``a \\in \\mathcal{B}`` of no declared space, or ``x \\in \\mathbb{R}^3``: read but
        not computed, saying why, and how labels are made to range over a space."""
        member, space = node.lhs.span.text, node.rhs
        styled = _styled(space.span)  # \mathcal{B}, not B
        written = Span(node.span.source, node.span.start, styled.end).text
        if isinstance(space, ast.Name) and space.id not in self.session.spaces:
            name = styled.text
            return CompileError(
                f"{written} is read but not computed: {name} is not a declared space",
                span=node.span,
                hint=f"to have {member} range over a space, declare it first, as in "
                f"{name} := \\mathbb{{R}}^{{3}}",
            )
        return CompileError(
            f"{written} says what {member} is, which is read but not computed",
            span=node.span,
            hint=f"to have {member} range over a space of its own, name the space, as in "
            "\\mathcal{A} := \\mathbb{R}^{3}, and write "
            f"{member} \\in \\mathcal{{A}}",
        )

    def _check_unknown_function(self, func: ast.Name, node: ast.Call) -> None:
        """A call of a function no one defined, which stays an unknown function: refused
        where its name is a builtin's or a spacetime's misspelt, as simplfy, or a built-in
        quantity's spelled out, as RicciScalar(g); said so in a note where its name is a
        word rather than a letter, as foo(x); as it is for f(x) or \\alpha(t, r)."""
        name, written = func.id, func.span.text
        if name == "print":
            arguments = ", ".join(arg.span.text for arg in node.args)
            raise CompileError(
                "print is not needed: each result is shown as it is computed",
                span=func.span,
                hint=f"write {arguments}" if arguments else None,
            )
        if name in GREEK or name in BUILTINS or len(name) == 1:  # \gamma(r), f(x)
            letter = name in GREEK or len(name) == 1  # not a builtin here, as \gamma is not
            if letter and not all(isinstance(a, ast.Name) for a in node.args):
                # \gamma(t - vx): a function of t - vx, or \gamma times it, as typeset?
                inside = ", ".join(arg.span.text for arg in node.args)
                self._note(
                    f"{written} is not defined, so {node.span.text} is an unknown function "
                    f"applied to {inside}; for a product, write {written} \\cdot ({inside})"
                )
            return
        concept = _CONCEPTS.get(name.lower())
        close = _close(name, [*BUILTINS, *_SPACETIME_NAMES]) if concept is None else None
        if concept is not None or close is not None:
            suggestion = concept if concept is not None else f"{close}(...)"
            raise CompileError(
                f"unknown function {written}", span=func.span, hint=f"did you mean {suggestion}?"
            )
        if len(name) > 1 and name not in GREEK and name.rstrip("'") not in GREEK:
            self._note(
                f"{written} is not a function RelativisticPy knows, so {node.span.text} is an "
                "unknown function"
            )

    def _check_keywords(self, builtin: Builtin, node: ast.Call) -> None:
        """Refuse an argument given by name, as n=5 in series(sin(x), x, 0, n=5), which
        would be an equation passed in its place: arguments are given in order."""
        try:
            parameters = list(inspect.signature(builtin.func).parameters)
        except (TypeError, ValueError):  # a function whose signature cannot be read
            return
        for position, arg in enumerate(node.args):
            if not (
                isinstance(arg, ast.Equation)
                and isinstance(arg.lhs, ast.Name)
                and arg.lhs.id in parameters
            ):
                continue
            hint = None
            if parameters.index(arg.lhs.id) == position:
                hint = f"write {node.span.text.replace(arg.span.text, arg.rhs.span.text)}"
            raise CompileError(
                f"{builtin.name}() takes its arguments in order, not by name, as "
                f"{arg.span.text} gives one",
                span=arg.span,
                hint=hint,
            )

    def _check_statistic_name(self, factor: ast.Expr, scope: Scope) -> None:
        """``\\operatorname{E}_t`` or ``\\operatorname{Covv}_t`` before what it acts on: a
        statistic misspelt is refused with the right one; any other unknown operator
        written so is read as a symbol, and says so."""
        if not (
            isinstance(factor, ast.Scripted)
            and isinstance(factor.base, ast.Name)
            and factor.base.verbatim
            and scope.lookup(factor.base.id) is None
            and len(factor.scripts) == 1
            and factor.scripts[0].kind == "_"
            and factor.base.id not in BUILTINS
        ):
            return
        name, label = factor.base.id, factor.scripts[0].span.text.lstrip("_")
        meant = _STATISTIC_ALIASES.get(name)
        if meant is None and (close := _close(name, _STATISTICS, shortest=3)) is not None:
            meant = f"\\operatorname{{{close}}}"
        if meant is not None:
            raise CompileError(
                f"\\operatorname{{{name}}} is not a statistic RelativisticPy knows",
                span=factor.span,
                hint=f"did you mean {meant}_{label}?",
            )
        if len(name) > 1:
            self._note(
                f"\\operatorname{{{name}}} is not an operator RelativisticPy knows, so "
                f"{factor.span.text} is read as a symbol, {name}_{label}; the statistics are "
                "\\mathbb{E}, \\operatorname{Var}, \\operatorname{Cov} and "
                "\\operatorname{LedoitWolf}"
            )

    def _check_spacetime_name(self, value: ast.Expr, scope: Scope, ctx: Context) -> None:
        """Refuse, where a metric is defined, a spacetime misspelt, as ``Schwarzchild(M)``,
        spelt as in Python, ``schwarzschild(M)``, or named without being called."""
        func = value.func if isinstance(value, ast.Call) else value
        if not isinstance(func, ast.Name) or scope.lookup(func.id) is not None:
            return
        name, written = func.id, func.span.text
        listed = ", ".join(list(_SPACETIME_NAMES)[:-1]) + " and " + list(_SPACETIME_NAMES)[-1]
        known = f"the spacetimes are {listed}"
        if not isinstance(value, ast.Call):
            if name in _SPACETIME_NAMES:
                raise CompileError(
                    f"{written} is a spacetime: call it, as in {name}()", span=value.span
                )
            spelt = _close(name, _SPACETIME_NAMES)
            if spelt is not None:
                raise CompileError(
                    f"unknown spacetime {written}",
                    span=value.span,
                    hint=f"did you mean {spelt}()?",
                    notes=[known],
                )
            return
        if name in BUILTINS or len(name) < 2 or name in GREEK:
            return
        args = ", ".join(arg.span.text for arg in value.args)
        same = next((s for s in _SPACETIME_NAMES if s.lower() == name.lower()), None)
        if same is not None:
            raise CompileError(
                f"unknown spacetime {written}",
                span=func.span,
                hint=f"in the language, spacetimes are capitalised: {same}({args})",
                notes=[known],
            )
        spelt = _close(name, _SPACETIME_NAMES)
        if spelt is not None or name[:1].isupper():
            raise CompileError(
                f"unknown spacetime {written}",
                span=func.span,
                hint=f"did you mean {spelt}({args})?" if spelt else None,
                notes=[known],
            )

    def _named_alone(self, reference: FunctionReference) -> Node:
        """A builtin's name with nothing to act on, as N is in N^2 (the lapse) or
        \\frac{N}{2}: a symbol, and said so where the name is a word."""
        name = reference.name
        if len(name) > 1 and name not in GREEK:  # simplify + 1, where N/2 is the lapse
            self._note(
                f"{reference.describe()} is a function, so alone it is read as a symbol; to "
                f"apply it, write {reference.describe()}(...)"
            )
        return self.graph.const(self.session.symbol(name), SCALAR, reference.span)
