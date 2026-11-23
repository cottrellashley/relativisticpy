r"""Quantum notation, read by its operands (``design/notation.md``, decisions 13 and 20),
each reading recorded with why, and a note where another was possible:

- a ket a program defines as a column, ``|0\rangle := \begin{pmatrix} 1 \\ 0 \end{pmatrix}``,
  is that vector and its bra the conjugate transpose, so that ``\langle\phi|\psi\rangle`` is
  their inner product, ``\langle\phi|A|\psi\rangle`` a row times a matrix times a column and
  ``|a\rangle\langle b|`` a matrix; ``|00\rangle``, of kets it defines, is
  ``|0\rangle \otimes |0\rangle``;
- a function of one variable named as books name a wavefunction, ``\psi(x) := ...``, or a
  wavefunction defined as ``\langle x|\alpha\rangle := ...``, is a state, whose brackets are
  integrals, :math:`\int \phi^*\psi\, dx`, over the domain the program states for x; in
  them, a hat on x that nothing defines is the position, as in ``\langle \hat{x} \rangle``;
- a bracket a program defines, ``\langle x|p\rangle := ...``, is a function of the labels its
  value is written with, and written the other way round, ``\langle p|x\rangle``, its
  conjugate; ``\langle j_1, j_2; m_1, m_2|j_1, j_2; j, m\rangle`` is a Clebsch–Gordan
  coefficient;
- angle brackets without a bar hold what a program defines, ``\langle x^2\rangle := ...`` or
  ``\langle p, q\rangle := ...``; else the inner product of two vectors, or the expectation
  value of an operator in the state the program names, divided by the square of the state's
  norm, as a state need not be normalised;
- an expression that ends in an operator with nothing to act on, as ``-i\hbar\frac{d}{dx}``,
  is an operator: defined, it is a value, applied by juxtaposition to the rest of the term,
  as ``\frac{d}{dx}`` is, composed by products and powers, and a claim between operators is
  decided on a test function; beside it, a hat on the variable it differentiates by is the
  position, as \hat{x} is beside ``\hat{p} := -i\hbar \frac{d}{dx}``;
- ``[A, B]`` of operators or matrices is their commutator and ``\{A, B\}`` their
  anticommutator; of anything else, a list and a set, as before;
- an ``=`` whose left side holds operators and kets nothing defines, as
  ``[\hat{x}, \hat{p}] = i\hbar``, ``a_-|n\rangle = \sqrt{n}|n - 1\rangle`` or
  ``\langle m|n\rangle = \delta_{mn}``, states a rule for what follows, which the algebra of
  those operators and kets keeps (:mod:`relativisticpy.runtime.quantum`).

The lowering of each is :mod:`relativisticpy.lowering.quantum`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, fields, replace
from typing import Any, ClassVar, Literal

import sympy
from sympy.physics.quantum import Operator as AbstractOperator

from relativisticpy.compiler.builtins import CONSTANTS
from relativisticpy.compiler.scope import (
    Binding,
    Context,
    FunctionBinding,
    Operator,
    Scope,
    VariableBinding,
)
from relativisticpy.ir import Node, dependencies, ops
from relativisticpy.language import syntax as ast
from relativisticpy.language.syntax import to_sexpr
from relativisticpy.reading.functions import Functions
from relativisticpy.reading.names import (
    _a_matrix,
    _matrix_shape,
    _named,
    _subscripted_name,
    _written_with,
)
from relativisticpy.typing.syntax import _operand, _product_factors

#: The decision of how a bracket or an operator is read.
QUANTUM = "bracket or operator"

#: The name an operator's operand is bound to while the operator is applied to it, which no
#: program can write.
OPERAND = "·operand"

#: The names books give a state, a wavefunction or a spinor: a function of one variable, or
#: a column, so named is the state of angle brackets that name none.
STATE_NAMES = frozenset({"psi", "Psi", "chi"})

#: The names books give a state, but a potential or a phase as often: a function or a
#: column so named is a state only where a ket or a subscript names it, as in
#: ``\langle x \rangle_\phi``.
NAMED_STATE_NAMES = frozenset({"phi", "Phi"})

#: How brackets of two items are written, by what they are of operators and matrices.
_COMMUTATORS = (
    "[",
    "\\left[",
    "\\big[",
    "\\Big[",
    "\\bigl[",
    "\\Bigl[",
    "\\lbrack",
    "\\left\\lbrack",
)
_ANTICOMMUTATORS = ("\\{", "\\left\\{", "\\lbrace", "\\left\\lbrace", "\\big\\{", "\\Big\\{",
                    "\\bigl\\{", "\\Bigl\\{")  # fmt: skip


@dataclass(frozen=True)
class ShownBinding(VariableBinding):
    """A value bound under a name no program writes, as a ket's, ``|0⟩``, or a rule's, and
    how it is ``shown``: as it is written, ``|0\\rangle``."""

    shown: str


@dataclass(frozen=True)
class ShownFunction(FunctionBinding):
    """A function bound under a name no program writes, as the bracket
    ``\\langle p, q\\rangle := ...`` is, and how it is ``shown``: as it is written."""

    shown: str


@dataclass(frozen=True)
class OperatorBinding(ShownFunction):
    """An operator a program defines, as ``\\hat{p} := -i\\hbar \\frac{d}{dx}``: a function of
    what it acts on, its operand, and the ``expression`` it is written as, which acts on the
    operand term by term and factor by factor. The functions whose definitions came
    ``before`` it are read as they were where they are written, without it: in their bodies,
    a hat on the variable it differentiates by is no position. They are known by their
    definitions, which a fork of the session keeps."""

    expression: ast.Expr
    before: tuple[ast.FunctionDefinition, ...] = field(default=(), compare=False, repr=False)


@dataclass(frozen=True)
class BracketBinding(ShownFunction):
    """A bracket a program defines, as ``\\langle x|p\\rangle := ...``: a function of the
    labels its value is written with, its parameters. ``bra`` and ``ket`` are its labels as
    written, as a bracket written later is matched against them."""

    bra: ast.Expr
    ket: ast.Expr


@dataclass(frozen=True)
class Operation(Operator):
    """An operator written as an expression, as ``-i\\hbar \\frac{d}{dx}``, ``\\hat{p}^2`` or
    ``[x, \\hat{p}]``, which acts on the rest of the term after it, as ``\\frac{d}{dx}``
    does, so that ``\\hat{p} x \\psi(x)`` is :math:`\\hat{p}(x\\psi)`; its names are those of
    ``scope``, and it is read as LaTeX if it is written in it. One the program defines is
    read as its ``definition`` is written."""

    expression: ast.Expr
    scope: Scope
    latex: bool
    definition: ast.FunctionDefinition | None = None
    reach: ClassVar[Literal["factor", "term"]] = "term"


@dataclass(frozen=True)
class Rule:
    """A rule an ``=`` states for operators and kets nothing defines (decision 20), of one of
    three ``kind``: ``commutator``, ``action`` of operators on a ket, or ``inner`` product of
    a bra and a ket; with the ``labels`` of its kets that stand for any label, as n does in
    ``a_-|n\\rangle = \\sqrt{n}|n - 1\\rangle``."""

    kind: Literal["commutator", "action", "inner"]
    labels: tuple[str, ...]


def ket_name(label: ast.Expr) -> str:
    """The name a ket's definition is bound to: ``|0⟩`` of ``|0\\rangle := ...``."""
    return f"|{to_sexpr(label)}⟩"


def bracket_name(bra: ast.Expr, ket: ast.Expr) -> str:
    """The name a bracket's definition is bound to: ``⟨x|p⟩``."""
    return f"⟨{to_sexpr(bra)}|{to_sexpr(ket)}⟩"


def expectation_name(operand: ast.Expr) -> str:
    """The name of what angle brackets a program defines hold: ``⟨(^ x 2)⟩`` of
    ``\\langle x^2 \\rangle := ...``."""
    return f"⟨{to_sexpr(operand)}⟩"


def inner_name(count: int) -> str:
    """The name of a bracket a program defines of ``count`` names, as
    ``\\langle p, q\\rangle := ...``, a function of them: ``⟨,⟩``."""
    return "⟨" + "," * (count - 1) + "⟩"


#: The prefix of the names rules are bound to, each followed by its number.
RULE = "⟨rule "


def name_written(expr: ast.Expr) -> str | None:
    """The name something is written as, if it is one: ``phat`` of ``\\hat{p}``, ``L_x`` of
    ``L_x`` and ``a_-`` of ``a_-``; ``None`` for anything else."""
    named = _named(expr)
    if isinstance(named, ast.Name):
        return named.id
    if (
        isinstance(named, ast.Scripted)
        and isinstance(named.base, ast.Name)
        and all(script.kind == "_" for script in named.scripts)
    ):
        return _subscripted_name(named)
    return None


def bracket_kind(node: ast.List | ast.Tuple) -> Literal["commutator", "anticommutator"] | None:
    """What brackets of two items are written as: the commutator's square brackets, the
    anticommutator's braces, or neither, as round brackets are."""
    written = node.span.text.lstrip()
    if isinstance(node, ast.List) or written.startswith(_COMMUTATORS):
        return "commutator"
    if written.startswith(_ANTICOMMUTATORS):
        return "anticommutator"
    return None


def items_of(label: ast.Expr) -> list[ast.Expr]:
    """The items of a label, each group's in turn: ``j_1, j_2, j, m`` of
    ``j_1, j_2; j, m``."""
    if isinstance(label, ast.Tuple):
        return [item for group in label.items for item in items_of(group)]
    return [label]


def groups_of(label: ast.Expr) -> list[list[ast.Expr]] | None:
    """The groups of a label that semicolons separate, each a list of its items; ``None``
    for a label of one group."""
    if not (isinstance(label, ast.Tuple) and all(isinstance(g, ast.Tuple) for g in label.items)):
        return None
    return [list(group.items) for group in label.items if isinstance(group, ast.Tuple)]


def label_names(label: ast.Expr) -> list[str]:
    """The names a label is written with, as n of ``|n - 1\\rangle`` and n' of
    ``\\langle n'|``, in order and each once."""
    found: dict[str, None] = {}
    for node in _nodes(label):
        if isinstance(node, ast.Prime):
            found[node.func.id + "'" * node.order] = None
        elif isinstance(node, ast.Name) and not node.verbatim:
            found[node.id] = None
    return list(found)


def _nodes(node: ast.Node) -> Iterator[ast.Node]:
    """Every node of a syntax tree, a prime's name as one with it."""
    yield node
    if isinstance(node, ast.Prime):
        return
    for item in fields(node):
        value = getattr(node, item.name)
        children = value if isinstance(value, tuple) else (value,)
        for child in children:
            if isinstance(child, ast.Node):
                yield from _nodes(child)


def signed(expr: ast.Expr, upper: bool) -> ast.Expr:
    """``expr`` with each ``\\pm`` and ``\\mp`` in it the sign it has above, ``upper``, or
    below: of ``\\mp i\\hat{p} + m\\omega x``, ``-i\\hat{p} + m\\omega x`` above."""
    signs = {"±": "+" if upper else "-", "∓": "-" if upper else "+"}

    def rewritten(node: Any) -> Any:
        if isinstance(node, tuple):
            return tuple(rewritten(item) for item in node)
        if not isinstance(node, ast.Node):
            return node
        changes = {
            item.name: rewritten(getattr(node, item.name))
            for item in fields(node)
            if item.name != "span"
        }
        if isinstance(node, ast.Unary | ast.Binary) and node.op in signs:
            changes["op"] = signs[node.op]
        return replace(node, **changes)

    result: ast.Expr = rewritten(expr)
    return result


def abstract(node: Node) -> bool:
    """Whether a value is an operator nothing defines, read as one by a rule."""
    return isinstance(node.op, ops.Const) and isinstance(node.op.value, AbstractOperator)


def column(node: Node) -> bool:
    """Whether a value is a matrix of one column, known without computing it: a ket."""
    shape = _matrix_shape(node)
    return shape is not None and shape[1] in (1, None) and _a_matrix(node)


class QuantumNotation(Functions):
    """How quantum notation is read: Dirac's, brackets of two items and operators."""

    #: The variables of the wavefunctions whose expectation values or matrix elements are
    #: being read, along which a hat is the position, as \hat{x} is in
    #: ``\langle \hat{x} \rangle`` of ψ(x).
    _state_variables: frozenset[str] = frozenset()

    # -- operators -----------------------------------------------------------------------

    def _acts(self, expr: ast.Expr, scope: Scope) -> bool:
        """Whether what is written is an operator, which acts on what follows it: a
        derivative with nothing to act on, as ``\\frac{d}{dx}``; an operator the program
        defines; a power of one; a product that ends in one; a sum, a quotient or a sign of
        one; or the commutator of operators."""
        expr = _operand(expr)
        if isinstance(expr, ast.Derivative):
            return expr.expr is None
        if isinstance(expr, ast.Binary):
            if expr.op in _SUMS:
                return any(self._acts(term, scope) for term in _terms(expr))
            if expr.op == "*":
                return self._acts(_product_factors(expr)[-1], scope)
            if expr.op == "/":
                return self._acts(expr.left, scope)
            return False
        if isinstance(expr, ast.Unary):
            return expr.op in ("-", "+", "±", "∓") and self._acts(expr.operand, scope)
        if isinstance(expr, ast.List | ast.Tuple):
            if len(expr.items) != 2 or bracket_kind(expr) is None:
                return False
            return any(self._acts(item, scope) for item in expr.items)
        if isinstance(expr, ast.Scripted) and _power(expr) is not None:
            return self._acts(expr.base, scope)
        return self._operator_binding(expr, scope) is not None

    def _operator_binding(self, expr: ast.Expr, scope: Scope) -> OperatorBinding | None:
        """The operator a name written means, if the program defines it as one."""
        name = name_written(expr)
        binding = scope.lookup(name) if name is not None else None
        return binding if isinstance(binding, OperatorBinding) else None

    def _variables_of(self, expr: ast.Expr, scope: Scope) -> list[str]:
        """The variables an operator differentiates by, those of the operators it is written
        with too, in order of their names: x, y and z of the angular momentum."""
        return sorted(
            {
                variable.id
                for derivative in self._derivatives_of(expr, scope)
                for variable, _ in derivative.variables
                if isinstance(variable, ast.Name)
            }
        )

    def _derivatives_of(self, expr: ast.Expr, scope: Scope) -> list[ast.Derivative]:
        """The derivatives an operator is written with, those of the operators it is written
        with too."""
        found: list[ast.Derivative] = []
        pending: list[tuple[ast.Node, Scope]] = [(expr, scope)]
        seen: set[int] = set()
        while pending:
            node, where = pending.pop()
            for current in _nodes(node):
                if isinstance(current, ast.Derivative):
                    found.append(current)
                if isinstance(current, ast.Expr):
                    binding = self._operator_binding(current, where)
                    if binding is not None and id(binding) not in seen:
                        seen.add(id(binding))
                        pending.append((binding.expression, binding.closure))
        return found

    def _vacuous(self, expression: ast.Expr, operand: Node, scope: Scope) -> list[str]:
        """The names an operator would act on in vain: those its operand is written with,
        where nothing defines any of them and none is a variable it differentiates by, so
        that what it makes of them is 0 for no reason a book means, as of f g in
        ``(a\\frac{d}{d\\lambda})(fg)``; none where the operand depends on what it can."""
        variables = set(self._variables_of(expression, scope))
        if not variables:
            return []
        if any(
            isinstance(node.op, ops.Call) and node.op.name == "core.function"
            for node in dependencies([operand])
        ):
            return []
        names = sorted(_written_with(operand))
        coordinates = {c.name for c in self.session.coordinates or ()}
        free = [
            name
            for name in names
            if scope.lookup(name) is None and name not in coordinates and name not in CONSTANTS
        ]
        if not free or len(free) != len(names) or set(names) & variables:
            return []
        return free

    def _position(self, accent: ast.Accent, scope: Scope, ctx: Context) -> str | None:
        """The variable ``\\hat{x}``, which nothing defines, is the position along, which
        acts by multiplying, decided by its letter: that of the wavefunction it is written in
        angle brackets of, as in ``\\langle \\hat{x} \\rangle``, or one an operator in force
        differentiates by, as x is beside ``\\hat{p} := -i\\hbar \\frac{d}{dx}``; ``None``
        for any other, as \\hat{y} is there."""
        if accent.accent != "hat" or not isinstance(accent.base, ast.Name):
            return None
        variable = accent.base.id
        if scope.lookup(variable + "hat") is not None:
            return None
        if variable in self._state_variables or variable in self._differentiated(scope, ctx):
            return variable
        return None

    def _differentiated(self, scope: Scope, ctx: Context) -> set[str]:
        """The variables the operators in force differentiate by: those of every operator the
        program defines, but, in the body of a function, of those it defined before the
        function, as a definition is read as it was where it is written."""
        found: set[str] = set()
        for _, binding in _bindings(scope):
            if not isinstance(binding, OperatorBinding):
                continue
            if any(ctx.body is earlier for earlier in binding.before):
                continue
            found.update(self._variables_of(binding.expression, binding.closure))
        return found

    # -- states ---------------------------------------------------------------------------

    def _states(self, scope: Scope, *, named: bool = False) -> list[tuple[str, Binding]]:
        """The states the program defines, in the order it defines them: its kets, the
        wavefunctions it defines as brackets, as ``\\langle x|\\alpha\\rangle := ...``, and the
        functions of one variable and the columns named as books name a state, ψ, Ψ or χ;
        and φ or Φ too among those a subscript may name, with ``named``, as they are
        potentials and phases as often (see :data:`NAMED_STATE_NAMES`)."""
        names = STATE_NAMES | NAMED_STATE_NAMES if named else STATE_NAMES
        found = []
        for name, binding in _bindings(scope):
            base = name.partition("_")[0]
            if name.startswith("|"):
                found.append((name, binding))
            elif isinstance(binding, BracketBinding):
                if _wavefunction(binding) is not None:
                    found.append((name, binding))
            elif isinstance(binding, OperatorBinding) or base not in names:
                continue
            elif isinstance(binding, FunctionBinding):
                if len(binding.definition.params) == 1:
                    found.append((name, binding))
            elif isinstance(binding, VariableBinding) and column(binding.node):
                found.append((name, binding))
        return found

    def _wavefunction_of(self, label: ast.Expr, scope: Scope) -> tuple[FunctionBinding, str] | None:
        """The wavefunction a label names, and its variable: a function of one variable the
        program defines, or the bracket ``\\langle x|\\alpha\\rangle`` it defines for
        ``|\\alpha\\rangle``; ``None`` for anything else."""
        name = name_written(label)
        binding = scope.lookup(name) if name is not None else None
        if (
            isinstance(binding, FunctionBinding)
            and not isinstance(binding, OperatorBinding | BracketBinding)
            and len(binding.definition.params) == 1
        ):
            return binding, binding.definition.params[0]
        for _, bracket in _bindings(scope):
            if isinstance(bracket, BracketBinding) and to_sexpr(bracket.ket) == to_sexpr(label):
                variable = _wavefunction(bracket)
                if variable is not None:
                    return bracket, variable
        return None

    def _names_a_state(self, label: ast.Expr, scope: Scope) -> bool:
        """Whether a label names a state: a ket the program defines, a column or a
        wavefunction."""
        if scope.lookup(ket_name(label)) is not None:
            return True
        name = name_written(label)
        named = scope.lookup(name) if name is not None else None
        if isinstance(named, VariableBinding) and column(named.node):
            return True
        return self._wavefunction_of(label, scope) is not None

    def _domain(self, variable: str) -> tuple[Any, Any]:
        """Where a wavefunction's variable ranges: between the bounds the program states for
        it, as ``0 \\leq x \\leq a``, or over the whole line."""
        symbol = self.session.symbol(variable)
        lower: Any = 0 if symbol.is_nonnegative else -sympy.oo
        upper: Any = 0 if symbol.is_nonpositive else sympy.oo
        for bound, _ in self.session.bounds:
            relation = sympy.sympify(bound)
            if not isinstance(relation, sympy.core.relational.Relational):
                continue
            left, right = relation.lhs, relation.rhs
            if left == symbol and isinstance(relation, sympy.Ge | sympy.Gt):
                lower = right
            elif left == symbol and isinstance(relation, sympy.Le | sympy.Lt):
                upper = right
            elif right == symbol and isinstance(relation, sympy.Le | sympy.Lt):
                lower = left
            elif right == symbol and isinstance(relation, sympy.Ge | sympy.Gt):
                upper = left
        return lower, upper

    # -- rules ----------------------------------------------------------------------------

    def _free_operator(self, expr: ast.Expr, scope: Scope) -> bool:
        """Whether something is written as an operator a rule may be stated for: a name that
        means nothing yet, or one a rule already reads as an operator, maybe under a
        dagger, as ``a^\\dagger``."""
        if isinstance(expr, ast.Scripted) and _daggered(expr):
            return self._free_operator(expr.base, scope)
        name = name_written(expr)
        if name is None or name in CONSTANTS or not name[:1].isalpha():
            return False
        binding = scope.lookup(name)
        if isinstance(binding, VariableBinding):
            return abstract(binding.node)
        if binding is not None:
            return False
        session = self.session
        return not (
            any(c.name == name for c in session.coordinates or ())
            or name in session.functions
            or self._builtin_scalar(name) is not None
        )

    def _marked_operator(self, expr: ast.Expr, scope: Scope) -> bool:
        """Whether something is written as an operator: under a hat or a dagger, as
        ``\\hat{x}`` and ``a^\\dagger``, or read as one by a rule already. A plain name, as S
        in Carroll's ``[S, T] = 0`` of vector fields, may be anything."""
        if isinstance(expr, ast.Scripted) and _daggered(expr):
            return True
        if isinstance(expr, ast.Accent) and expr.accent == "hat":
            return True
        name = name_written(expr)
        binding = scope.lookup(name) if name is not None else None
        return isinstance(binding, VariableBinding) and abstract(binding.node)

    def _free_label(self, label: ast.Expr, scope: Scope) -> bool:
        """Whether a label names no ket nor bracket the program defines, nor a state."""
        if scope.lookup(ket_name(label)) is not None:
            return False
        name = name_written(label)
        return name is None or scope.lookup(name) is None

    def _rule_written(self, equation: ast.Equation, scope: Scope) -> Rule | None:
        """The rule an ``=`` states, if it states one (decision 20): its left side is the
        commutator of operators nothing defines, one of them written as one, as
        ``[\\hat{x}, \\hat{p}] = i\\hbar``; such operators acting on a ket nothing defines, as
        ``a_-|n\\rangle = \\sqrt{n}|n-1\\rangle``; or the bracket of a bra and a ket nothing
        defines, as ``\\langle m|n\\rangle = \\delta_{mn}``. ``None`` for anything else, as a
        claim about what the program defines."""
        lhs, rhs = _operand(equation.lhs), equation.rhs
        if isinstance(lhs, ast.List | ast.Tuple):
            if len(lhs.items) != 2 or bracket_kind(lhs) != "commutator":
                return None
            if isinstance(_operand(rhs), ast.List | ast.Tuple):
                return None
            if all(self._free_operator(item, scope) for item in lhs.items) and any(
                self._marked_operator(item, scope) for item in lhs.items
            ):
                return Rule("commutator", ())
            return None
        if isinstance(lhs, ast.BraKet) and lhs.operator is None:
            labels = (lhs.bra, lhs.ket)
            if all(self._free_label(label, scope) for label in labels) and not self._bracket_found(
                lhs, scope
            ):
                names = [name for label in labels for name in label_names(label)]
                return Rule("inner", tuple(dict.fromkeys(_free(names, scope))))
            return None
        factors = _product_factors(lhs) if isinstance(lhs, ast.Binary) else []
        if len(factors) < 2 or not isinstance(factors[-1], ast.Ket):
            return None
        ket = factors[-1]
        if not self._free_label(ket.label, scope):
            return None
        if all(self._free_operator(factor, scope) for factor in factors[:-1]):
            return Rule("action", tuple(_free(label_names(ket.label), scope)))
        return None

    def _bracket_found(self, node: ast.BraKet, scope: Scope) -> bool:
        """Whether a bracket the program defines is written, either way round."""
        return any(
            scope.lookup(bracket_name(bra, ket)) is not None
            for bra, ket in ((node.bra, node.ket), (node.ket, node.bra))
        )

    def _rules(self, scope: Scope) -> list[Node]:
        """The rules stated so far, in the order they were stated."""
        found = [
            (int(name[len(RULE) : -1]), binding.node)
            for name, binding in _bindings(scope)
            if name.startswith(RULE) and isinstance(binding, VariableBinding)
        ]
        return [node for _, node in sorted(found, key=lambda pair: pair[0])]

    # -- brackets ---------------------------------------------------------------------------

    def _bracket_definition(
        self, bra: ast.Expr, ket: ast.Expr, scope: Scope
    ) -> tuple[BracketBinding, list[ast.Expr], bool] | None:
        """The bracket the program defines that ``\\langle bra|ket\\rangle`` is, the labels to
        give its parameters, and whether it is written the other way round, so that it is
        the conjugate: ``\\langle p|x\\rangle`` of ``\\langle x|p\\rangle := ...``. Where several
        match, the one whose fixed labels are written wins, as
        ``\\langle x|\\alpha\\rangle := ...`` does of ``\\langle x|\\alpha\\rangle`` over
        ``\\langle x|p\\rangle := ...``; then the way round that gives each parameter its own
        name."""
        best: tuple[tuple[int, int], BracketBinding, list[ast.Expr], bool] | None = None
        for _, binding in _bindings(scope):
            if not isinstance(binding, BracketBinding):
                continue
            for swapped in (False, True):
                written = (ket, bra) if swapped else (bra, ket)
                matched = _matched(binding, *written)
                if matched is None:
                    continue
                arguments, score = matched
                if any(self._names_a_state(argument, scope) for argument in arguments):
                    continue  # a parameter is a label, as x is, never a state, as \alpha is
                if best is None or score > best[0]:
                    best = (score, binding, arguments, swapped)
        if best is None:
            return None
        return best[1], best[2], best[3]


#: The operations of a sum, whose terms an operator acts on each.
_SUMS = ("+", "-", "±", "∓")


def _terms(node: ast.Expr) -> list[ast.Expr]:
    """The terms of a sum, without recursion, as a sum may have thousands of them."""
    found: list[ast.Expr] = []
    pending = [node]
    while pending:
        expr = pending.pop()
        if isinstance(expr, ast.Binary) and expr.op in _SUMS:
            pending.extend((expr.right, expr.left))
        else:
            found.append(expr)
    return found


def _power(node: ast.Scripted) -> int | None:
    """The whole power a single superscript raises its base to, as 2 in ``\\hat{p}^2``;
    ``None`` for any other script."""
    if len(node.scripts) != 1 or node.scripts[0].kind != "^":
        return None
    body = node.scripts[0].body
    if isinstance(body, ast.Number) and body.text.isdigit() and int(body.text) > 0:
        return int(body.text)
    return None


def _daggered(node: ast.Scripted) -> bool:
    """Whether a script is a dagger alone, as of ``a^\\dagger``."""
    if len(node.scripts) != 1 or node.scripts[0].kind != "^":
        return False
    body = node.scripts[0].body
    return isinstance(body, ast.Name) and body.id == "dagger"


def _free(names: list[str], scope: Scope) -> list[str]:
    """The names nothing defines, which stand in a rule for any label."""
    return [name for name in names if scope.lookup(name) is None and name not in CONSTANTS]


def _bindings(scope: Scope) -> list[tuple[str, Binding]]:
    """Every name bound where ``scope`` is, the outermost first, each as it is seen there."""
    levels = []
    current: Scope | None = scope
    while current is not None:
        levels.append(current.bindings())
        current = current.parent
    seen: dict[str, Binding] = {}
    for level in reversed(levels):
        seen.update(level)
    return list(seen.items())


def _wavefunction(binding: BracketBinding) -> str | None:
    """The variable of a bracket defined as a wavefunction, as x of
    ``\\langle x|\\alpha\\rangle := ...``: its bra a parameter, its ket not."""
    params = binding.definition.params
    bra = binding.bra.id if isinstance(binding.bra, ast.Name) else None
    if len(params) == 1 and bra == params[0]:
        return bra
    return None


def _matched(
    binding: BracketBinding, bra: ast.Expr, ket: ast.Expr
) -> tuple[list[ast.Expr], tuple[int, int]] | None:
    """The labels a bracket defined as ``binding`` gives its parameters when written
    ``\\langle bra|ket\\rangle``, and how well it matches: how many labels it fixes are
    written, and how many parameters are written with their own names; ``None`` where a
    label it fixes is written otherwise."""
    params = binding.definition.params
    given: dict[str, ast.Expr] = {}
    fixed = 0
    for defined, written in ((binding.bra, bra), (binding.ket, ket)):
        if isinstance(defined, ast.Name) and defined.id in params:
            given[defined.id] = written
        elif to_sexpr(defined) != to_sexpr(written):
            return None
        else:
            fixed += 1
    arguments = [given[param] for param in params]
    own = sum(
        1
        for param, written in given.items()
        if isinstance(written, ast.Name) and written.id == param
    )
    return arguments, (fixed, own)
