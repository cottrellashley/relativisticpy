r"""Assumptions, read: a relation on names nothing defines, standing alone, as ``a > 0``,
``0 < x < 1``, ``a, b > 0`` or ``x \in \mathbb{R}``, says what the names are for the rest of
the program, as books state the regime they work in (design/notation.md, decision 2).

A relation is an assumption when it relates names nothing defines to constants, as
``a > r > 0`` does. A relation between names alone, as ``k_B T > h c B``, or about a name
that has a value, is a condition or a claim, as before. A list of names, ``a, b > 0``,
shares the relation. So does a membership of a set of numbers, ``\mathbb{R}``,
``\mathbb{Z}``, ``\mathbb{N}``, ``\mathbb{Q}`` or ``\mathbb{C}``, maybe with a bound, as
``\mathbb{Z}_{>0}`` or ``\mathbb{R}^+``, or of an interval, as ``x \in (0, 1)``.
``x \in \mathbb{R}^3`` says what x is in another sense, and is read as before.

Each name gets the facts SymPy keeps on a symbol (positive, nonnegative, real, an
integer); what SymPy cannot keep, as x < 1 or a > r, is a bound, kept in the session.
:mod:`relativisticpy.assumptions` applies both.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import sympy
from sympy import Symbol

from relativisticpy import assumptions as applying
from relativisticpy.compiler.scope import Context, Scope, TensorBinding, VariableBinding
from relativisticpy.errors import CompileError, RelativisticPyError
from relativisticpy.ir import SCALAR, VALUE, NotationType, ops
from relativisticpy.language import syntax as ast
from relativisticpy.reading.names import _mentions_symbol, _named

if TYPE_CHECKING:
    from relativisticpy.lowering.definitions import Definitions

#: The decision of whether a relation standing alone is an assumption.
ASSUMPTION = "assumption or claim"

#: What each set of numbers says of its members.
_SETS: dict[str, dict[str, bool]] = {
    "ℝ": {"real": True},
    "ℂ": {"complex": True},
    "ℤ": {"integer": True},
    "ℕ": {"integer": True, "nonnegative": True},  # with 0, as ISO 80000-2 has it
    "ℚ": {"rational": True},
}
#: What a sign in a set's script says of its members: ℝ^+, ℤ_-, ℕ^*.
_SIGNED: dict[str, dict[str, bool]] = {
    "+": {"positive": True},
    "-": {"negative": True},
    "*": {"nonzero": True},
}
#: The constants a bound may be written with.
_CONSTANTS: dict[str, sympy.Expr] = {
    "pi": sympy.pi,
    "e": sympy.E,
    "infty": sympy.oo,
    "oo": sympy.oo,
}
_FLIPPED = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "!=": "!="}
_RELATIONS: dict[str, Callable[[Any, Any], Any]] = {
    "<": sympy.Lt,
    "<=": sympy.Le,
    ">": sympy.Gt,
    ">=": sympy.Ge,
    "!=": sympy.Ne,
}


@dataclass(frozen=True)
class Assumption:
    """What a relation assumes. ``facts`` holds, by name, in the order written, what SymPy
    keeps on its symbol, as ``{"positive": True}``; ``bounds`` what it cannot keep, as
    x < 1 or a > r, each a relation of plain symbols named as the names, and constants."""

    facts: dict[str, dict[str, bool]] = field(default_factory=dict)
    bounds: tuple[Any, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        """The names it is about, in the order written."""
        named = list(self.facts)
        for bound in self.bounds:
            named.extend(s.name for s in sorted(bound.free_symbols, key=str))
        return tuple(dict.fromkeys(named))


def read(relation: ast.Expr, name_of: Callable[[ast.Expr], str | None]) -> Assumption | None:
    """What ``relation`` assumes, if it is an assumption; ``None`` otherwise. ``name_of``
    gives the name an operand writes, where it is a name nothing defines, and ``None``
    for anything else."""
    if isinstance(relation, ast.Comparison):
        return _comparison(relation, name_of)
    if isinstance(relation, ast.Relation) and relation.op == "∈" and numbers(relation):
        return _membership(relation, name_of)
    return None


# -- comparisons ------------------------------------------------------------------------


def _comparison(
    relation: ast.Comparison, name_of: Callable[[ast.Expr], str | None]
) -> Assumption | None:
    """``a > 0``, ``0 < x \\leq 1``, ``a > r > 0`` or ``a, b > 0``: names and constants
    in a chain, at least one of each."""
    if "==" in relation.ops:
        return None
    first, *rest = relation.operands
    listed = first.items if isinstance(first, ast.Tuple) else (first,)
    heads = [_term(item, name_of) for item in listed]
    terms = [_term(operand, name_of) for operand in rest]
    if None in heads or None in terms or (len(heads) > 1 and not _all_names(heads)):
        return None
    chains = [[head, *terms] for head in heads]
    if _all_names(chains[0]) or not any(isinstance(term, Symbol) for term in chains[0]):
        return None  # names alone, as T_h > T_c, or constants alone: no assumption
    found = Assumption()
    for chain in chains:
        found = _joined(found, _chain(chain, list(relation.ops)))
    return found


def _all_names(terms: Sequence[Any]) -> bool:
    return all(isinstance(term, Symbol) for term in terms)


def _term(operand: ast.Expr, name_of: Callable[[ast.Expr], str | None]) -> Any:
    """An operand of a chain as SymPy has it: the symbol of a name, or a real constant;
    ``None`` for anything else."""
    name = name_of(operand)
    if name is not None:
        return Symbol(name)
    value = _constant(operand)
    if value is None or value.is_extended_real is not True:
        return None
    return value


def _chain(terms: list[Any], relations: list[str]) -> Assumption:
    """What a chain of comparisons of names and constants assumes. In one direction, as
    ``0 < x < 1`` or ``a > r > 0``, each name is bounded by the nearest constant on each
    side; otherwise each link is read alone."""
    links = list(zip(terms, relations, terms[1:], strict=False))
    between = [
        _RELATIONS[op](left, right)  # a > r, as written
        for left, op, right in links
        if isinstance(left, Symbol) and isinstance(right, Symbol)
    ]
    if set(relations) <= {">", ">="}:
        terms = terms[::-1]
        relations = [_FLIPPED[op] for op in relations[::-1]]
    if not set(relations) <= {"<", "<="}:
        found = Assumption()
        for left, op, right in links:
            found = _joined(found, _link(left, op, right))
        return found
    facts: dict[str, dict[str, bool]] = {}
    bounds: list[Any] = list(between)
    for position, term in enumerate(terms):
        if not isinstance(term, Symbol):
            continue
        facts[term.name] = {"real": True}
        lower = _nearest(terms, relations, position, -1)
        upper = _nearest(terms, relations, position, 1)
        if lower is not None:
            _below(term, *lower, facts[term.name], bounds)
        if upper is not None:
            _above(term, *upper, facts[term.name], bounds)
    return Assumption(facts, tuple(bounds))


def _nearest(
    terms: Sequence[Any], relations: Sequence[str], position: int, step: int
) -> tuple[Any, bool] | None:
    """The nearest constant to the term at ``position``, before it or after it, and
    whether what lies between is strict: in ``0 < r < a``, 0 is below a, strictly."""
    strict = False
    at = position
    while 0 <= at + step < len(terms):
        between = relations[at] if step > 0 else relations[at - 1]
        strict = strict or between == "<"
        at += step
        if not isinstance(terms[at], Symbol):
            return terms[at], strict
    return None


def _below(
    name: Symbol, lower: Any, strict: bool, facts: dict[str, bool], bounds: list[Any]
) -> None:
    """What a lower bound says: x > 0 that x is positive, x >= 0 that it is nonnegative,
    and x > 1 a bound besides."""
    if lower == -sympy.oo:
        return
    if lower > 0 or (lower == 0 and strict):
        facts["positive"] = True
    elif lower == 0:
        facts["nonnegative"] = True
    if lower != 0:
        bounds.append(sympy.Gt(name, lower) if strict else sympy.Ge(name, lower))


def _above(
    name: Symbol, upper: Any, strict: bool, facts: dict[str, bool], bounds: list[Any]
) -> None:
    """What an upper bound says, as :func:`_below` does of a lower one."""
    if upper == sympy.oo:
        return
    if upper < 0 or (upper == 0 and strict):
        facts["negative"] = True
    elif upper == 0:
        facts["nonpositive"] = True
    if upper != 0:
        bounds.append(sympy.Lt(name, upper) if strict else sympy.Le(name, upper))


def _link(left: Any, op: str, right: Any) -> Assumption:
    """One comparison of a chain read alone: of a name and a constant, or of two names."""
    if not isinstance(left, Symbol):
        left, op, right = right, _FLIPPED[op], left
    if not isinstance(left, Symbol):
        return Assumption()
    if isinstance(right, Symbol):
        return Assumption({}, (_RELATIONS[op](left, right),))
    if op == "!=":
        if right == 0:
            return Assumption({left.name: {"nonzero": True}})
        return Assumption({}, (sympy.Ne(left, right),))
    facts: dict[str, bool] = {"real": True}
    bounds: list[Any] = []
    if op in (">", ">="):
        _below(left, right, op == ">", facts, bounds)
    else:
        _above(left, right, op == "<", facts, bounds)
    return Assumption({left.name: facts}, tuple(bounds))


def _joined(one: Assumption, other: Assumption) -> Assumption:
    facts = {name: dict(found) for name, found in one.facts.items()}
    for name, found in other.facts.items():
        facts.setdefault(name, {}).update(found)
    bounds = tuple(dict.fromkeys((*one.bounds, *other.bounds)))
    return Assumption(facts, bounds)


# -- memberships ------------------------------------------------------------------------


def _membership(
    relation: ast.Relation, name_of: Callable[[ast.Expr], str | None]
) -> Assumption | None:
    """``x \\in \\mathbb{R}``, ``\\theta, \\phi \\in \\mathbb{R}``, ``n \\in
    \\mathbb{Z}_{>0}`` or ``x \\in (0, 1)``."""
    listed = relation.lhs.items if isinstance(relation.lhs, ast.Tuple) else (relation.lhs,)
    names = [name_of(item) for item in listed]
    if not names or None in names:
        return None
    members = _members(relation.rhs)
    if members is None:
        return None
    facts, chain = members
    found = Assumption()
    for name in names:
        assert name is not None
        found = _joined(found, Assumption({name: dict(facts)}))
        if chain is not None:
            terms, relations = chain
            symbol = Symbol(name)
            placed = [symbol if term is None else term for term in terms]
            found = _joined(found, _chain(placed, list(relations)))
    return found


_Chain = tuple[list[Any], list[str]]


def numbers(relation: ast.Relation) -> bool:
    """Whether a membership is of a set of numbers, which it assumes of a name, as
    ``\\mathbb{R}``, ``\\mathbb{Z}_{>0}`` or the interval ``(0, 1)``, written so; not the
    values a label takes, as in ``\\mu = 0, 1``."""
    listed = isinstance(relation.rhs, ast.Tuple | ast.List)
    return (relation.written != "" or not listed) and _members(relation.rhs) is not None


def _members(numbers: ast.Expr) -> tuple[dict[str, bool], _Chain | None] | None:
    """What a set says of its members: the facts, and a chain of comparisons with the
    place of a member left as ``None``, as ``0 < · < 1`` for the interval (0, 1)."""
    if isinstance(numbers, ast.Name):
        facts = _SETS.get(numbers.id)
        return (dict(facts), None) if facts is not None else None
    if isinstance(numbers, ast.Tuple | ast.List) and len(numbers.items) == 2:
        ends = [_constant(item) for item in numbers.items]
        if any(end is None or end.is_extended_real is not True for end in ends):
            return None
        relation = "<" if isinstance(numbers, ast.Tuple) else "<="
        return {}, ([ends[0], None, ends[1]], [relation, relation])
    if not (
        isinstance(numbers, ast.Scripted)
        and isinstance(numbers.base, ast.Name)
        and numbers.base.id in _SETS
        and len(numbers.scripts) == 1
    ):
        return None
    facts = dict(_SETS[numbers.base.id])
    script = numbers.scripts[0]
    body = script.body
    if isinstance(body, ast.Name) and body.id in _SIGNED:  # ℝ^+, ℤ_-, ℕ^*
        signed = dict(_SIGNED[body.id])
        if numbers.base.id == "ℕ" and body.id in ("+", "*"):
            signed = {"positive": True}
        return {**facts, **signed}, None
    if isinstance(body, ast.Number) and script.kind == "_" and body.text in ("0", "1"):
        start = "nonnegative" if body.text == "0" else "positive"  # ℕ_0, ℕ_1
        return {**facts, start: True}, None
    if isinstance(body, ast.Comparison) and len(body.operands) == 2:  # ℤ_{>0}
        bound = _constant(body.operands[1])
        if bound is None or bound.is_extended_real is not True:
            return None
        return facts, ([None, bound], [body.ops[0]])
    return None


# -- constants --------------------------------------------------------------------------


def _constant(node: ast.Expr) -> Any:
    """The value of an operand written with numbers and constants alone, as ``-\\pi/2``,
    ``+\\infty`` or ``10^{-3}``; ``None`` for anything else."""
    if isinstance(node, ast.Number):
        return (
            sympy.Rational(node.text)
            if "." in node.text or "e" in node.text.lower()
            else (sympy.Integer(node.text))
        )
    if isinstance(node, ast.Name):
        return _CONSTANTS.get(node.id)
    if isinstance(node, ast.Unary) and node.op in ("-", "+"):
        inner = _constant(node.operand)
        return None if inner is None else (-inner if node.op == "-" else inner)
    if isinstance(node, ast.Binary) and node.op in ("+", "-", "*", "/", "**"):
        left, right = _constant(node.left), _constant(node.right)
        if left is None or right is None:
            return None
        return _ARITHMETIC[node.op](left, right)
    if isinstance(node, ast.Scripted) and len(node.scripts) == 1:
        script = node.scripts[0]
        base = _constant(node.base)
        exponent = _constant(script.body) if script.body is not None else None
        if script.kind == "^" and base is not None and exponent is not None:
            return base**exponent
        return None
    if isinstance(node, ast.Root):
        radicand = _constant(node.radicand)
        degree = _constant(node.degree) if node.degree is not None else sympy.Integer(2)
        if radicand is None or degree is None:
            return None
        return sympy.root(radicand, degree)
    return None


_ARITHMETIC: dict[str, Callable[[Any, Any], Any]] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
    "**": lambda a, b: a**b,
}


# -- taking an assumption -----------------------------------------------------------------


def assume(reader: Definitions, relation: ast.Expr, scope: Scope, ctx: Context) -> bool:
    """Take a relation standing alone as an assumption, if it is one: record it as the
    statement's reading and in the session, and give its names what it says from now on.
    ``False`` for a relation that is no assumption, which is read as before."""
    if scope is not reader.session.globals or not isinstance(
        relation, ast.Comparison | ast.Relation
    ):
        return False
    spans: dict[str, ast.Expr] = {}

    def name_of(operand: ast.Expr) -> str | None:
        name = _lone_name(reader, operand, scope, ctx)
        if name is not None:
            spans.setdefault(name, operand)
        return name

    assumption = read(relation, name_of)
    if assumption is None or not assumption.names:
        return False
    session = reader.session
    written = " ".join(relation.span.text.split())
    coordinates = {c.name for c in session.coordinates or ()}
    symbols: dict[str, Symbol] = {}
    for name, facts in assumption.facts.items():
        if name in coordinates or not facts:
            continue
        try:
            symbols[name] = applying.symbol(name, facts, session.assumptions.get(name))
        except ValueError as error:
            raise CompileError(
                f"{written} contradicts what was assumed before",
                span=relation.span,
                notes=[str(error)],
            ) from None
    kept = [
        bound
        for name, facts in assumption.facts.items()
        if name in coordinates
        for bound in applying.as_bounds(Symbol(name), facts)
    ]
    kept.extend(assumption.bounds)
    used = _used_before(reader, symbols, scope)
    for name, symbol in symbols.items():
        session.assumptions[name] = symbol
    session.bounds.extend((bound, relation.span) for bound in kept)
    _carry(reader, symbols, scope)
    if isinstance(relation, ast.Relation):
        why = "it says which numbers names nothing defines are, standing alone"
    else:
        why = "it relates names nothing defines to constants, standing alone, as books state "
        why += "the regime they work in"
    said = applying.said(assumption.facts, assumption.bounds)
    note = f"{written} is an assumption: from here on, {said}"
    reader._decide(ASSUMPTION, relation.span, "an assumption", why, note, written=written)
    for name in assumption.names:
        operand = spans.get(name)
        facts = assumption.facts.get(name, {})
        meaning = "a coordinate" if name in coordinates else "a symbol"
        if facts:
            meaning += f", assumed {applying.described(facts)} from here on"
        shown = operand.span.text if operand is not None else name
        reader._resolve(operand.span if operand is not None else None, meaning, written=shown)
    held = [name for name in assumption.facts if name in coordinates]
    if held:
        one = len(held) == 1
        reader._note(
            f"{applying.listed(held)} {'is a coordinate' if one else 'are coordinates'}, whose "
            f"{'symbol stays as it is' if one else 'symbols stay as they are'}: what is assumed "
            f"of {'it' if one else 'them'} is used where it decides something, as by solve"
        )
    if used:
        one = len(used) == 1
        reader._note(
            f"{applying.listed(used)} {'was' if one else 'were'} used before: what was defined "
            f"with {'it' if one else 'them'} takes the assumption from here on, and what was "
            "shown stays as it was"
        )
    return True


def _lone_name(reader: Definitions, operand: ast.Expr, scope: Scope, ctx: Context) -> str | None:
    """The name an operand writes, where it is a lone name nothing defines, as ``a``,
    ``\\rho_0`` or ``k_\\mathrm{B}``, read as the reader reads it; ``None`` for anything
    else, as a name with a value, a product of letters or a call."""
    target = _named(operand)
    base = target.base if isinstance(target, ast.Scripted) else target
    if not isinstance(base, ast.Name) or scope.lookup(base.id) is not None:
        return None
    if isinstance(target, ast.Name) and target.id in _CONSTANTS:
        return None
    checker = type(reader)(reader.session.fork())
    checker._depth = 1  # its notes are not this statement's
    try:
        node = checker.value(operand, scope, ctx)
    except RelativisticPyError:
        return None
    value = node.op.value if isinstance(node.op, ops.Const) else None
    if not isinstance(value, Symbol) or scope.lookup(value.name) is not None:
        return None
    return str(value.name)


def _used_before(reader: Definitions, symbols: dict[str, Symbol], scope: Scope) -> list[str]:
    """The names an assumption gives facts to that what the program defined is written
    with: that takes the assumption from now on."""
    used = []
    for name in symbols:
        for _, binding in scope.bindings():
            if isinstance(binding, VariableBinding | TensorBinding) and _mentions_symbol(
                binding.node, name
            ):
                used.append(name)
                break
        else:
            if reader._written_in_metric(name):
                used.append(name)
    return used


def _carry(reader: Definitions, symbols: dict[str, Symbol], scope: Scope) -> None:
    """Give what the program defined, and the metric in force, the symbols an assumption
    makes, in place of the ones they are written with: from now on, a value written with
    a is written with a that is positive. What was computed stays as it was."""
    if not symbols:
        return
    graph = reader.graph
    attributes: ops.Attributes = (("symbols", tuple(symbols.values())),)
    for name, binding in scope.bindings():
        if not isinstance(binding, VariableBinding | TensorBinding):
            continue
        node = binding.node
        if isinstance(node.type, NotationType):
            continue
        if not any(_mentions_symbol(node, written) for written in symbols):
            continue
        carried = graph.add(ops.Call("core.assume", attributes), (node,), node.type, node.span)
        if isinstance(binding, TensorBinding):
            scope.bind(name, TensorBinding(carried, binding.indices))
        else:
            scope.bind(name, VariableBinding(carried))
    for name, symbol in symbols.items():
        if reader._written_in_metric(name):  # as a definition after it would give it a value
            written = (graph.const(name, VALUE), graph.const(symbol, SCALAR))
            reader._rewrite_metric(ops.Call("core.where"), written)
