r"""The lowering of optima: ``\max_{x} f(x)``, ``\arg\min_{w} f(w)``, ``\sup`` and ``\inf``.

An optimum over an index of data, ``\max_t x_t``, binds the index as a sum does, and is the
greatest or the least value along its axis; ``\arg\max_k |X_k|`` is the label of the
element where it is. An optimum over variables is a problem: each variable is bound, as a
real symbol, or as the components of a vector or of a tensor, what the optimum is of and
its constraints are lowered with them, and the problem is solved as the program runs, by
the optimizers the integrations declare (:mod:`relativisticpy.integrations._optimizers`),
in turn, the first that finds it saying how, and whether it is global.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Any

from sympy import Symbol

from relativisticpy import assumptions, display
from relativisticpy.compiler.scope import Context, Scope, TensorBinding, VariableBinding
from relativisticpy.errors import CompileError, IndexNotationError
from relativisticpy.ir import SCALAR, VALUE, Node, TensorType, ops
from relativisticpy.language import syntax as ast
from relativisticpy.language.source import Span
from relativisticpy.lowering.transforms import Transforms
from relativisticpy.reading import assumptions as reading
from relativisticpy.tensors import Index, Indices
from relativisticpy.tensors.spaces import Space
from relativisticpy.typing.syntax import _walk

#: The decision of what an optimum is over.
OPTIMUM = "optimum"
#: What each operator finds, and whether it is the point.
_FINDS = {
    "max": ("max", "value"),
    "min": ("min", "value"),
    "sup": ("sup", "value"),
    "inf": ("inf", "value"),
    "argmax": ("max", "point"),
    "argmin": ("min", "point"),
}
#: What each operator is called, as a note says it.
_CALLED = {
    "max": "the maximum",
    "min": "the minimum",
    "sup": "the supremum",
    "inf": "the infimum",
    "argmax": "where the maximum is",
    "argmin": "where the minimum is",
}
#: The relations a constraint is written with, as the optimizers take them; ⪰ and ⪯ are
#: componentwise, as between vectors.
_RELATIONS = {
    "==": "=", "<": "<", "<=": "<=", ">": ">", ">=": ">=", "!=": "!=",
    "⪰": ">=", "⪯": "<=", "≻": ">", "≺": "<",
}  # fmt: skip
#: The operation that finds an optimum, by what it gives: a number, where it is, of several
#: variables or of a vector, or a tensor.
_OPERATIONS = {"tensor": "optimum.components", "column": "optimum.point", "list": "optimum.point"}
#: The generalized inequalities of a cone, as Boyd and Vandenberghe write them.
_CONES = frozenset({"⪰", "⪯", "≻", "≺"})
#: The sets of values written in braces, as {0, 1}.
_BRACES = ("\\{", "\\left\\{", "\\lbrace", "\\left\\lbrace")


@dataclass(frozen=True)
class _Variable:
    """A variable of an optimum, as it is bound: by its ``name``, the ``symbols`` of its
    components, one for a number, and ``index``, the index and space of a tensor it is
    written as, or ``size``, the length of a vector; ``values``, the finitely many each
    component takes, where it takes few; and ``bounds``, what its set bounds it by, as
    the 0 < x and x < 1 of x \\in (0, 1), relations of its symbol."""

    name: str
    symbols: tuple[Symbol, ...]
    size: int | None = None
    index: tuple[Index, Space] | None = None
    values: tuple[Any, ...] = ()
    bounds: tuple[Any, ...] = ()


class Optima(Transforms):
    """The lowering of optima, over an index of data or over variables."""

    def _Optimum(self, node: ast.Optimum, scope: Scope, ctx: Context) -> Node:
        label = self._over_an_index(node, scope)
        if label is not None:
            return self._extreme(node, label, scope, ctx)
        return self._problem(node, scope, ctx)

    def _Relation(self, node: ast.Relation, scope: Scope, ctx: Context) -> Node:
        """A generalized inequality, as ``A \\succ 0``: of a matrix and 0, its definiteness;
        of vectors, the inequality of each component."""
        if node.op not in _CONES:
            return super()._Relation(node, scope, ctx)
        lhs, rhs = self.value(node.lhs, scope, ctx), self.value(node.rhs, scope, ctx)
        written = " ".join(node.span.text.split())
        attributes = (("relation", _RELATIONS[node.op]), ("written", written))
        return self._call("optimum.order", [lhs, rhs], SCALAR, node.span, attributes)

    # -- over an index of data --------------------------------------------------------------

    def _over_an_index(self, node: ast.Optimum, scope: Scope) -> str | None:
        """The index an optimum is over, where it is over one, as ``\\max_t x_t`` and
        ``\\max_a v^a`` are: one label, written as an index of data, of a tensor, or of what
        a transform of data gives, as ``\\arg\\max_k |\\operatorname{fft}(x)_k|``, or one
        declared to range over a space. A power, as x^n, is no index."""
        if len(node.variables) != 1 or node.constraints:
            return None
        variable = node.variables[0]
        if not isinstance(variable, ast.Name):
            return None
        label = variable.id
        if self._carriers(label, node.body, scope):
            return label
        for each in _walk(node.body):
            if not (isinstance(each, ast.Scripted) and label in _labels_on(each)):
                continue
            if label in self.session.labels or isinstance(each.base, ast.Call):
                return label
        return None

    def _extreme(self, node: ast.Optimum, label: str, scope: Scope, ctx: Context) -> Node:
        """The greatest or least value along the axis of an index, or its label where it is,
        binding the index as a sum does: inside, every factor shares it."""
        with self._binding(label):
            body = self.value(node.body, scope, ctx)
        written = _operator(node)
        if not (isinstance(body.type, TensorType) and label in body.type.indices.names):
            raise IndexNotationError(
                f"{written} is over the index {self._written_label(label)}, but what it is of "
                "has no such index",
                span=node.span,
            )
        axis = body.type.indices.names.index(label)
        space = body.type.spaces[axis]
        kept = [index for k, index in enumerate(body.type.indices) if k != axis]
        spaces = tuple(s for k, s in enumerate(body.type.spaces) if k != axis)
        if kept:
            kind: Any = TensorType(Indices(kept), spaces)
            name = "optimum.along"
        elif node.op in ("argmax", "argmin"):
            kind, name = VALUE, "optimum.label"
        else:
            kind, name = SCALAR, "optimum.extreme"
        found = _CALLED[node.op]
        why = f"what it is of is written with {self._written_label(label)} as an index of data"
        self._decide(OPTIMUM, node.span, f"{found} over the elements of {space}", why)
        attributes = (
            ("op", node.op),
            ("axis", axis),
            ("labels", tuple(space.basis or ())),
            ("written", written),
        )
        return self._call(name, [body], kind, node.span, attributes)

    # -- over variables ---------------------------------------------------------------------

    def _problem(self, node: ast.Optimum, scope: Scope, ctx: Context) -> Node:
        """An optimum over variables: each bound, with what its set and the conditions
        beneath the operator say of it, what it is of and its constraints lowered with them,
        and the problem left to the optimizers, as the program runs."""
        written = _operator(node)
        names = [self._variable_name(each, written) for each in node.variables]
        conditions = list(node.constraints)
        self._refuse_functions(node, names, written)
        inner = scope.child()
        variables = []
        for name, written_as in zip(names, node.variables, strict=True):
            variable = self._variable(name, written_as, node, conditions, scope, ctx)
            self._bind(variable, inner, written_as)
            variables.append(variable)
        body = self.value(_products(node.body, names), inner, ctx)
        relations: list[str] = []
        sides: list[Node] = []
        for condition in conditions:
            for relation, lhs, rhs in self._links(condition, variables, written):
                relations.append(relation)
                sides += [self.value(lhs, inner, ctx), self.value(rhs, inner, ctx)]
        limits = [bound for variable in variables for bound in variable.bounds]
        for bound in [*limits, *self._assumed_of(variables, written, node.span)]:
            relations.append(_RELATIONS[bound.rel_op])  # 0 < x and x < 1, of x \in (0, 1)
            sides += [self.graph.const(side, SCALAR) for side in (bound.lhs, bound.rhs)]
        unknowns = [self.graph.const(s, SCALAR) for v in variables for s in v.symbols]
        sense, goal = _FINDS[node.op]
        kind, shape = self._answer(goal, variables, node)
        domains = tuple(
            ("values", v.values) if v.values else () for v in variables for _ in v.symbols
        )
        over = ", ".join(self._written_label(v.name) for v in variables)
        why = "what it is of is written with " + over + ", which nothing else binds"
        self._decide(OPTIMUM, node.span, f"{_CALLED[node.op]} over {over}", why)
        attributes = (
            ("sense", sense),
            ("goal", goal),
            ("unknowns", len(unknowns)),
            ("relations", tuple(relations)),
            ("domains", domains),
            ("shape", shape),
            ("written", written),
        )
        operation = _OPERATIONS.get(shape[0], "optimum.find")
        return self._call(operation, [body, *unknowns, *sides], kind, node.span, attributes)

    def _assumed_of(self, variables: Sequence[_Variable], written: str, span: Span) -> list[Any]:
        """The bounds the program assumes of the names an optimum is over, as x \\leq 1 of
        x \\in [0, 1], which SymPy cannot keep on a symbol as it keeps a sign: constraints
        of the optimum, as relations of its variables' symbols, which a note says. A bound
        on a vector is on each of its components."""
        symbols = {variable.name: variable.symbols for variable in variables}
        found: list[Any] = []
        kept: list[Any] = []
        for bound in assumptions.in_force(self.session.bounds, self.session.assumptions):
            if getattr(bound, "rel_op", None) not in ("<", "<=", ">", ">="):
                continue
            named = {s: symbols[s.name] for s in bound.free_symbols if s.name in symbols}
            if not named:
                continue
            if len(named) > 1 and any(len(each) > 1 for each in named.values()):
                raise CompileError(
                    f"{written} is over vectors of which the program assumes "
                    f"{display.unicode(bound)}, which relates more than one of them",
                    span=span,
                    hint="state it as a constraint of the optimum, after \\text{subject to}",
                )
            kept.append(bound)
            for k in range(max(len(each) for each in named.values())):
                found.append(bound.xreplace({s: each[k] for s, each in named.items()}))
        if kept:
            names = sorted({s.name for bound in kept for s in bound.free_symbols} & set(symbols))
            facts = {
                name: dict(getattr(self.session.assumptions.get(name), "_assumptions_orig", {}))
                for name in names
            }
            shown = assumptions.listed([display.unicode(Symbol(name)) for name in names])
            said = assumptions.said(facts, kept)
            self._note(f"{written} keeps what the program assumes of {shown}: {said}")
        return found

    def _variable_name(self, node: ast.Expr, written: str) -> str:
        """The name a variable of an optimum is written as: ``x``, ``x_1``, ``x_{1}``,
        ``\\mathbf{x}``."""
        if isinstance(node, ast.Name):
            return node.id
        if (
            isinstance(node, ast.Scripted)
            and isinstance(node.base, ast.Name)
            and len(node.scripts) == 1
            and node.scripts[0].kind == "_"
            and isinstance(node.scripts[0].body, ast.Name | ast.Number)
        ):
            body = node.scripts[0].body
            label = body.id if isinstance(body, ast.Name) else body.text
            return f"{node.base.id}_{label}"
        raise CompileError(
            f"{written} is over {node.span.text}, which is no name: an optimum is over names",
            span=node.span,
            hint="write the variables beneath the operator as names, as \\min_{x, y}",
        )

    def _refuse_functions(self, node: ast.Optimum, names: Sequence[str], written: str) -> None:
        """Refuse an optimum over a function, as ``\\min_u \\int_0^1 u(t)^2 dt``, whose
        variable is called with arguments that are not written with it."""
        for each in _walk_all(node):
            if (
                isinstance(each, ast.Call)
                and isinstance(each.func, ast.Name)
                and each.func.id in names
                and not any(_mentions(arg, each.func.id) for arg in each.args)
            ):
                raise CompileError(
                    f"{written} is over a function, {each.func.id}, as {each.span.text} shows: "
                    "an optimum over a space of functions, as in the calculus of variations "
                    "and optimal control, is not supported yet",
                    span=each.span,
                    hint="write the function by its parameters, as u(t) = a t + b, and take "
                    "the optimum over them, as \\min_{a, b}",
                )

    def _variable(
        self,
        name: str,
        written: ast.Expr,
        node: ast.Optimum,
        conditions: list[ast.Expr],
        scope: Scope,
        ctx: Context,
    ) -> _Variable:
        """A variable as the optimum takes it: what its set and its conditions say of it,
        as a symbol with facts; a vector, where its set is \\mathbb{R}^n or \\{0, 1\\}^n,
        or a matrix it is multiplied by gives its length; or a tensor, where it is written
        with an index of a space. The conditions of a membership are taken from
        ``conditions``, those it leaves are constraints."""
        facts: dict[str, bool] = {"real": True}
        size: int | None = None
        values: tuple[Any, ...] = ()
        bounds: list[Any] = []
        for condition in list(conditions):
            if not (isinstance(condition, ast.Relation) and condition.op == "∈"):
                continue
            members = (
                condition.lhs.items if isinstance(condition.lhs, ast.Tuple) else (condition.lhs,)
            )
            if not any(_names(member, name) for member in members):
                continue
            taken = self._set_of(name, condition, scope, ctx)
            if taken is None:
                raise CompileError(
                    f"{_operator(node)} is over {condition.rhs.span.text}, a set the language "
                    "does not know: an optimum is over real or whole numbers, an interval, "
                    "\\mathbb{R}^n, or finitely many values, as \\{0, 1\\}^n",
                    span=condition.span,
                )
            more, size, values, limits = taken
            facts.update(more)
            bounds.extend(limits)
            if len(members) == 1:
                conditions.remove(condition)
        for condition in conditions:  # x \geq 0 makes x nonnegative, as an assumption would
            if isinstance(condition, ast.Comparison) and _on(condition, name):
                found = reading.read(
                    condition, lambda operand: name if _names(operand, name) else None
                )
                if found is not None:
                    facts.update(found.facts.get(name, {}))
        before = self.session.assumptions.get(name)
        index = self._index_of(name, node, scope)
        if index is not None:
            size = index[1].dim
        elif size is None:
            size = self._size_from_products(name, node, scope, ctx)
        try:
            if size is None:
                symbol = assumptions.symbol(name, facts, before)
                put = [bound.xreplace({Symbol(name): symbol}) for bound in bounds]
                return _Variable(name, (symbol,), values=values, bounds=tuple(put))
            symbols = tuple(
                assumptions.symbol(f"{name}_{k}", facts, before) for k in range(1, size + 1)
            )
        except ValueError as error:
            raise CompileError(
                f"what {_operator(node)} says of {written.span.text} contradicts what was "
                "assumed of it",
                span=written.span,
                notes=[str(error)],
            ) from None
        put = [bound.xreplace({Symbol(name): s}) for bound in bounds for s in symbols]
        return _Variable(name, symbols, size, index, values, tuple(put))

    def _set_of(
        self, name: str, relation: ast.Relation, scope: Scope, ctx: Context
    ) -> tuple[dict[str, bool], int | None, tuple[Any, ...], list[Any]] | None:
        """What ``x \\in S`` says of a variable: the facts of its symbol, the length of a
        vector, the values it takes, and the bounds an interval puts on it, relations of
        the plain symbol of its name; ``None`` for a set the language does not know."""
        numbers = relation.rhs
        if _in_braces(numbers) and isinstance(numbers, ast.Tuple):  # \{0, 1\}
            return {}, None, self._values(numbers, scope, ctx), []
        if isinstance(numbers, ast.Scripted) and len(numbers.scripts) == 1:
            script = numbers.scripts[0]
            base = numbers.base
            if script.kind == "^" and script.body is not None and not _signs(script.body):
                size = self._whole(script.body, scope, ctx)
                if size is None:
                    return None
                if _in_braces(base) and isinstance(base, ast.Tuple):  # \{0, 1\}^n
                    return {}, size, self._values(base, scope, ctx), []
                single = replace(relation, lhs=ast.Name(name, span=relation.lhs.span), rhs=base)
                found = reading.read(single, lambda operand: name)
                if found is None:  # \mathbb{R}^n
                    return None
                return found.facts.get(name, {}), size, (), []
        single = replace(relation, lhs=ast.Name(name, span=relation.lhs.span))
        found = reading.read(single, lambda operand: name)
        if found is None:
            return None
        return found.facts.get(name, {}), None, (), list(found.bounds)

    def _values(self, items: ast.Tuple, scope: Scope, ctx: Context) -> tuple[Any, ...]:
        values = [self._folded(self.value(item, scope, ctx)) for item in items.items]
        if any(value is None for value in values):
            raise CompileError(
                f"the values in {items.span.text} are numbers known before anything is computed",
                span=items.span,
            )
        return tuple(values)

    def _whole(self, node: ast.Expr, scope: Scope, ctx: Context) -> int | None:
        value = self._folded(self.value(node, scope, ctx))
        if value is None or not getattr(value, "is_integer", False) or value < 1:
            return None
        return int(value)

    def _index_of(self, name: str, node: ast.Optimum, scope: Scope) -> tuple[Index, Space] | None:
        """The index a variable is written with, as the a of ``w^{a} \\Sigma_{ab} w^{b}``,
        and the space it ranges over, where it is written with one of a declared space."""
        for each in _walk_all(node):
            if not (
                isinstance(each, ast.Scripted)
                and isinstance(each.base, ast.Name)
                and each.base.id == name
            ):
                continue
            for script in each.scripts:
                items = script.indices.items if script.indices is not None else ()
                if len(items) != 1 or items[0].label is None:
                    continue
                space = self.session.labels.get(items[0].label)
                if space is not None:
                    return Index(items[0].label, script.kind == "^"), space
        return None

    def _size_from_products(
        self, name: str, node: ast.Optimum, scope: Scope, ctx: Context
    ) -> int | None:
        """The length of a vector a variable is, where it is multiplied by a matrix whose
        shape is known when the program is read: the columns of A in ``A x``, the rows of
        b in ``b^T y``."""
        for each in _walk_all(node):
            if not (isinstance(each, ast.Binary) and each.op == "*"):
                continue
            left, right = _last_factor(each.left), _first_factor(each.right)
            if _names(right, name):
                shape = self._shape(left, scope)
                if shape is not None:
                    return shape[1]
            if _names(_untransposed(left), name):
                shape = self._shape(right, scope)
                if shape is not None:
                    return shape[0]
        return None

    def _shape(self, node: ast.Expr, scope: Scope) -> tuple[int, int] | None:
        """The shape of a matrix a name is bound to, transposed as written, where it is
        known when the program is read."""
        transposed = node is not _untransposed(node)
        named = _untransposed(node)
        if not isinstance(named, ast.Name):
            return None
        binding = scope.lookup(named.id)
        if not isinstance(binding, VariableBinding):
            return None
        op = binding.node.op
        shape = op.shape if isinstance(op, ops.MakeArray | ops.ToArray) else None
        if shape is None or len(shape) != 2:
            return None
        rows, columns = shape
        return (columns, rows) if transposed else (rows, columns)

    def _bind(self, variable: _Variable, scope: Scope, written: ast.Expr) -> None:
        """Bind a variable in the scope of what the optimum is of: its name to its symbol,
        to a column of the symbols of its components, or to a tensor of them; and the name
        of each component, x_1, to its symbol."""
        consts = [self.graph.const(s, SCALAR, written.span) for s in variable.symbols]
        if variable.index is not None:
            index, space = variable.index
            kind = TensorType(Indices([index]), (space,))
            array = self.graph.add(ops.MakeArray((space.dim,)), consts, kind, written.span)
            scope.bind(variable.name, TensorBinding(array, Indices([index])))
        elif variable.size is not None:
            shape = (variable.size, 1)
            scope.bind(
                variable.name,
                VariableBinding(self.graph.add(ops.MakeArray(shape), consts, VALUE, written.span)),
            )
        else:
            scope.bind(variable.name, VariableBinding(consts[0]))
            return
        for symbol, const in zip(variable.symbols, consts, strict=True):
            scope.bind(symbol.name, VariableBinding(const))

    def _links(
        self, condition: ast.Expr, variables: Sequence[_Variable], written: str
    ) -> Iterator[tuple[str, ast.Expr, ast.Expr]]:
        """Each relation of a constraint, with its two sides: a chain, 0 \\le x \\le 3, gives
        two, and a list of names on the left, x_1, x_2 \\geq 0, one for each name."""
        if isinstance(condition, ast.Equation):
            yield from _each("=", condition.lhs, condition.rhs)
            return
        if isinstance(condition, ast.Comparison):
            for left, op, right in zip(
                condition.operands, condition.ops, condition.operands[1:], strict=False
            ):
                yield from _each(_RELATIONS[op], left, right)
            return
        if isinstance(condition, ast.Relation) and condition.op in _RELATIONS:
            yield from _each(_RELATIONS[condition.op], condition.lhs, condition.rhs)
            return
        if isinstance(condition, ast.Relation) and condition.op == "∈":
            members = (
                condition.lhs.items if isinstance(condition.lhs, ast.Tuple) else (condition.lhs,)
            )
            known = {v.name for v in variables}
            if all(any(_names(member, n) for n in known) for member in members):
                return  # what it says is in the variables' symbols
        raise CompileError(
            f"{condition.span.text} is no constraint {written} can keep: write equations and "
            "inequalities",
            span=condition.span,
            hint="as in \\text{subject to} \\quad x + y = 1, \\quad x \\geq 0",
        )

    def _answer(
        self, goal: str, variables: Sequence[_Variable], node: ast.Optimum
    ) -> tuple[Any, tuple[str, ...]]:
        """The type of an optimum's value and the shape of where it is: a number; a list of
        the values of several variables; a column, of a vector; or a tensor, of a variable
        written with an index."""
        if goal == "value":
            return SCALAR, ("value",)
        if len(variables) == 1:
            only = variables[0]
            if only.index is not None:
                index, space = only.index
                return TensorType(Indices([index]), (space,)), ("tensor",)
            if only.size is not None:
                return VALUE, ("column",)
            return SCALAR, ("scalar",)
        if any(v.size is not None for v in variables):
            raise CompileError(
                f"{_operator(node)} is over several variables, one of them a vector: where its "
                "optimum is cannot be given as one value",
                span=node.span,
                hint="take the optimum over the vector alone, or over numbers",
            )
        return VALUE, ("list",)


# -- what is written --------------------------------------------------------------------


def _operator(node: ast.Optimum) -> str:
    """The operator of an optimum with what is beneath it, as written: ``\\min_{x_1, x_2}``."""
    text = node.span.text
    body = node.body.span
    if body.source is node.span.source and node.span.start <= body.start <= node.span.end:
        text = node.span.source.text[node.span.start : body.start]
    return " ".join(text.split()).rstrip("\\;,: ")


def _walk_all(node: ast.Optimum) -> Iterator[ast.Node]:
    """Every node of what an optimum is of and of its constraints."""
    yield from _walk(node.body)
    for constraint in node.constraints:
        yield from _walk(constraint)


def _labels_on(node: ast.Scripted) -> set[str]:
    return {
        str(item.label)
        for script in node.scripts
        if script.indices is not None
        for item in script.indices.items
        if item.label is not None
    }


def _names(node: ast.Expr, name: str) -> bool:
    """Whether a node writes the name ``name``: x, or x_{1} as x_1."""
    if isinstance(node, ast.Name):
        return node.id == name
    if (
        isinstance(node, ast.Scripted)
        and isinstance(node.base, ast.Name)
        and len(node.scripts) == 1
    ):
        script = node.scripts[0]
        body = script.body
        label = body.id if isinstance(body, ast.Name) else getattr(body, "text", None)
        return script.kind == "_" and f"{node.base.id}_{label}" == name
    return False


def _mentions(node: ast.Expr, name: str) -> bool:
    return any(isinstance(each, ast.Expr) and _names(each, name) for each in _walk(node))


def _on(condition: ast.Comparison, name: str) -> bool:
    return any(_names(operand, name) for operand in condition.operands)


def _in_braces(node: ast.Expr) -> bool:
    return node.span.text.lstrip().startswith(_BRACES)


def _signs(node: ast.Expr) -> bool:
    """Whether a script is a sign, as the + of \\mathbb{R}^+."""
    return isinstance(node, ast.Name) and node.id in ("+", "-", "*")


def _each(relation: str, lhs: ast.Expr, rhs: ast.Expr) -> Iterator[tuple[str, ast.Expr, ast.Expr]]:
    """A relation, on each of a list of names written on its left, as x_1, x_2 \\geq 0."""
    if isinstance(lhs, ast.Tuple):
        for item in lhs.items:
            yield relation, item, rhs
    else:
        yield relation, lhs, rhs


def _untransposed(node: ast.Expr) -> ast.Expr:
    """A matrix written transposed, A^T or A^\\top, without its transpose."""
    if (
        isinstance(node, ast.Scripted)
        and len(node.scripts) == 1
        and node.scripts[0].kind == "^"
        and isinstance(node.scripts[0].body, ast.Name)
        and node.scripts[0].body.id in ("T", "top", "intercal")
    ):
        return node.base
    return node


def _last_factor(node: ast.Expr) -> ast.Expr:
    while isinstance(node, ast.Binary) and node.op == "*":
        node = node.right
    return node


def _first_factor(node: ast.Expr) -> ast.Expr:
    while isinstance(node, ast.Binary) and node.op == "*":
        node = node.left
    return node


def _products(node: ast.Expr, names: Sequence[str]) -> ast.Expr:
    """What an optimum is of, with a variable written before brackets that hold it, as
    ``x(1 - x)``, read as a product, not a call (design/notation.md, decision 28)."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in names
        and len(node.args) == 1
        and _mentions(node.args[0], node.func.id)
    ):
        operand = _products(node.args[0], names)
        return ast.Binary("*", node.func, operand, implicit=True, span=node.span)
    if isinstance(node, ast.Binary):
        return replace(node, left=_products(node.left, names), right=_products(node.right, names))
    if isinstance(node, ast.Unary):
        return replace(node, operand=_products(node.operand, names))
    return node
