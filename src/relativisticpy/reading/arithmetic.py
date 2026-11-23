r"""Number theory, read: the letters books give arithmetic functions, and the sets a
condition under a sum or a product gives its index (design/notation.md, decisions 19 and
32 to 36).

``\varphi(36)`` is Euler's totient and ``\mu(d)``, of the index of a sum, the Möbius
function, but ``\varphi(x)`` of a real x stays a function of the program's, and
``\mu(t)`` too. An argument is an integer where it is written as a whole number above 0, as
a name the program assumes an integer, ``n \in \mathbb{N}``, or as the index of a sum or a
product, or is made of these by sums, differences, products and powers. Two readings ask
more: the letter of a constant, as the π of ``\pi(n)``, which physics multiplies, and a
subscript, as the n of the Bernoulli number ``B_n``, where a number names a constant, as
``H_0`` names Hubble's and ``B_0`` a field. Each reads so only of names the program assumes
integers, never of a number or of the index of a sum, which ``B_n`` of a Fourier series
is written with.

A condition under a sum gives its index its values: ``d \mid n`` the divisors of n,
``0 \le k \le n`` the integers between, ``k \le n`` those from 1, as number theory sums,
``k \neq 0`` every integer but 0, and ``p \text{ prime}`` the primes. The index p ranges
over the primes wherever no least value is written, as in ``\prod_{p \mid n}`` and
``\sum_{p \le x}``. Each other condition, as ``\gcd(k, n) = 1`` in a row of
``\substack``, is one each value must meet.

``\equiv`` with a modulus on its right, ``a \equiv b \pmod{m}``, is a congruence: a claim,
or, of a name nothing defines, standing alone, an assumption, as ``x > 0`` is; the
congruences assumed of a name, with its bounds, may leave it one value, by the Chinese
remainder theorem.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import sympy

from relativisticpy.compiler.builtins import CONSTANTS
from relativisticpy.compiler.scope import Scope, VariableBinding
from relativisticpy.integrations import Capability
from relativisticpy.integrations import _forms as forms
from relativisticpy.ir import ops
from relativisticpy.language import syntax as ast
from relativisticpy.typing.syntax import _walk

if TYPE_CHECKING:
    from relativisticpy.reading.reader import Reader


def of_integers(
    reader: Reader,
    capability: Capability,
    match: forms.Match,
    node: ast.Expr,
    written: str,
    scope: Scope,
) -> str | None:
    """The note that says a form written with a letter is read as ``capability``, where the
    arguments it reads as integers are integers; ``None`` where one is not, and the letter
    is read as it was before. ``\\pi(10)`` is π times 10, and says how to count primes."""
    shape = match.pattern.shape
    constant = shape.head in CONSTANTS
    subscripted = shape.kind == "subscripted"
    named = [match.bindings[name] for name in capability.integers if name in match.bindings]
    for argument in named:
        if constant and not _assumed(reader, argument, scope):
            if isinstance(argument, ast.Number):
                count = argument.span.text
                reader._note(
                    f"{node.span.text} is read as {written} times {count}; the number of "
                    f"primes up to {count} is {capability.name}({count})"
                )
            return None
        if subscripted and not (_named(argument) and _integral(reader, argument, scope)):
            return None
        if not (constant or subscripted) and not _an_integer(reader, argument, scope):
            return None
    if subscripted and _family_of(reader, shape.head, scope):
        return None
    shown = ", ".join(_written(argument, node) for argument in named)
    what = "symbol" if subscripted else "function"
    return (
        f"{node.span.text} is read as {capability.brief}, {capability.name}, as {shown} is an "
        f"integer; for a {what} of your own, define {written} first"
    )


def _written(argument: ast.Node, node: ast.Expr) -> str:
    """An argument as it is written: the n of ``B_n``, which the lexer joined to its name
    and so shares its place, as its name."""
    if argument.span == node.span and isinstance(argument, ast.Name):
        return argument.id
    return argument.span.text


def _an_integer(reader: Reader, node: ast.Node, scope: Scope) -> bool:
    """Whether an argument is an integer: a whole number above 0, or made of whole numbers
    and names that are integers, by assumption or as the index of a sum."""
    if isinstance(node, ast.Number):
        return node.text.isdigit() and int(node.text) > 0
    return _integral(reader, node, scope, bound=True)


def _assumed(reader: Reader, node: ast.Node, scope: Scope) -> bool:
    """Whether an argument is a name the program assumes an integer, as n is after
    ``n \\in \\mathbb{N}``, which no sum binds."""
    return isinstance(node, ast.Name) and _integral(reader, node, scope)


def _integral(reader: Reader, node: ast.Node, scope: Scope, *, bound: bool = False) -> bool:
    """Whether an expression is an integer by how it is written: whole numbers, and names
    that are integers, in sums, differences, products and powers to whole numbers. A name
    bound in the program counts where ``bound`` allows, as the index of a sum does."""
    if isinstance(node, ast.Number):
        return node.text.isdigit()
    if isinstance(node, ast.Name):
        return _integer_name(reader, node.id, scope, bound=bound)
    if isinstance(node, ast.Unary) and node.op in ("-", "+"):
        return _integral(reader, node.operand, scope, bound=bound)
    if isinstance(node, ast.Binary) and node.op in ("+", "-", "*"):
        return all(_integral(reader, side, scope, bound=bound) for side in (node.left, node.right))
    if isinstance(node, ast.Scripted) and len(node.scripts) == 1 and node.scripts[0].kind == "^":
        exponent = node.scripts[0].body
        whole = isinstance(exponent, ast.Number) and exponent.text.isdigit()
        return whole and _integral(reader, node.base, scope, bound=bound)
    return False


def _integer_name(reader: Reader, name: str, scope: Scope, *, bound: bool) -> bool:
    """Whether a name is an integer: assumed one, or, where ``bound`` allows, bound to an
    integer, or to the index of a sum or a product being read (decision 4)."""
    binding = scope.lookup(name)
    if binding is not None:
        if not bound or not isinstance(binding, VariableBinding):
            return False
        if reader._a_sum_index(name, scope):
            return True
        value = binding.node.op.value if isinstance(binding.node.op, ops.Const) else None
        return isinstance(value, sympy.Basic) and value.is_integer is True
    symbol = reader.session.assumptions.get(name)
    return symbol is not None and symbol.is_integer is True


def _named(node: ast.Node) -> bool:
    """Whether an expression writes a name, as ``2n`` does and ``10`` does not."""
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Unary):
        return _named(node.operand)
    if isinstance(node, ast.Binary):
        return _named(node.left) or _named(node.right)
    return isinstance(node, ast.Scripted) and _named(node.base)


def _family_of(reader: Reader, letter: str, scope: Scope) -> bool:
    """Whether the program names something with the letter and a subscript, as ``B_n`` or
    ``B_1``: a family of its own, which the letter with any subscript is."""
    names = [*reader.session.functions, *reader.session.assumptions]
    level: Scope | None = scope
    while level is not None:
        names += [name for name, _ in level.bindings()]
        level = level.parent
    return any(name.startswith(f"{letter}_") for name in names)


# -- the index sets given by a condition ------------------------------------------------


#: The words that say what an integer is, as ``p \text{ prime}`` and ``k \text{ odd}`` do.
_WORDS = frozenset({"prime", "odd", "even"})


@dataclass(frozen=True)
class IndexSet:
    """The values a condition under a sum or a product gives its index: the ``index``; the
    number whose divisors they are, ``of``; the least and the greatest, ``lower`` and
    ``upper``, each excluded where it is ``strict``, and ``None`` where none is written;
    whether they are the ``primes``; the values ``excluded``, as 0 is by k ≠ 0; and the
    ``conditions`` each value meets besides, as gcd(k, n) = 1. ``notes`` say how it was
    read where that could surprise."""

    index: ast.Name
    of: ast.Expr | None = None
    lower: ast.Expr | None = None
    upper: ast.Expr | None = None
    strict: tuple[bool, bool] = (False, False)
    primes: bool = False
    excluded: tuple[ast.Expr, ...] = ()
    conditions: tuple[ast.Expr, ...] = ()
    notes: tuple[str, ...] = field(default=(), compare=False)


def index_set(condition: ast.Expr) -> IndexSet | None:
    """The values ``condition``, written under a sum or a product, gives its index, which it
    writes first: ``d \\mid n``, ``0 \\le k \\le n``, ``p \\le x``, ``k \\neq 0``,
    ``p \\text{ prime}``, or several of these in the rows of ``\\substack``; ``None`` where
    it names no index, which is said as before."""
    parts = _conjuncts(condition)
    index = _index_in(parts[0])
    if index is None:
        return None
    found = IndexSet(index)
    for part in parts:
        found = _narrowed(found, part)
    return _completed(found)


def every_integer(node: ast.BigOperator) -> IndexSet | None:
    """``\\sum_k \\binom{r}{k} \\binom{s}{n - k}``: a sum over an index alone, with no limits,
    whose range Concrete Mathematics leaves to its binomial coefficients, which vanish where
    their lower index is negative: over every integer, summed where they are not 0 (decision
    38). ``None`` for any other sum written so, as over an index of a tensor or of data,
    ``\\sum_a w^a``, which no binomial coefficient bounds, or one written with the index as
    a subscript."""
    index = node.variable
    if not isinstance(index, ast.Name) or node.lower is not None or node.upper is not None:
        return None
    nodes = list(_walk(node.body))
    if any(
        isinstance(each, ast.Scripted)
        and any(script.kind == "_" and _mentions(script, index.id) for script in each.scripts)
        for each in nodes
    ):
        return None
    bounded = any(
        isinstance(each, ast.Call)
        and isinstance(each.func, ast.Name)
        and each.func.id == "binomial"
        and len(each.args) == 2
        and _mentions(each.args[1], index.id)
        for each in nodes
    )
    if not bounded:
        return None
    note = (
        f"{index.span.text} runs over every integer, as Concrete Mathematics sums over it: "
        "its binomial coefficients leave the terms where they are not 0"
    )
    return IndexSet(index, notes=(note,))


def _conjuncts(condition: ast.Expr) -> list[ast.Expr]:
    if isinstance(condition, ast.Binary) and condition.op == "and":
        return [*_conjuncts(condition.left), *_conjuncts(condition.right)]
    return [condition]


def _index_in(part: ast.Expr) -> ast.Name | None:
    """The index a condition is about: what divides, ``d`` of ``d \\mid n``; the name a
    comparison bounds, in the middle of a chain or else first; or what a word describes, the
    ``p`` of ``p \\text{ prime}``."""
    if isinstance(part, ast.Relation) and part.op == "∣":
        return part.lhs if isinstance(part.lhs, ast.Name) else None
    if isinstance(part, ast.Comparison):
        operands = part.operands
        middle = operands[1] if len(operands) == 3 else None
        for operand in (middle, *operands):
            if isinstance(operand, ast.Name):
                return operand
        return None
    if (word := described(part)) is not None:
        return word[0]
    return None


def described(part: ast.Expr) -> tuple[ast.Name, str] | None:
    """A name and the word that says what it is, as ``p \\text{ prime}`` writes them."""
    if not (isinstance(part, ast.Binary) and part.op == "*"):
        return None
    name, word = part.left, part.right
    if isinstance(name, ast.Name) and isinstance(word, ast.Name) and word.id in _WORDS:
        return (name, word.id) if word.verbatim else None
    return None


def _narrowed(found: IndexSet, part: ast.Expr) -> IndexSet:
    """What one condition says of the values: their bounds, that they divide a number or
    are primes, a value they leave out, or, of anything else, a condition each meets."""
    index = found.index.id
    if isinstance(part, ast.Relation) and part.op == "∣" and _is(part.lhs, index):
        return replace(found, of=part.rhs)
    if (word := described(part)) is not None and word[0].id == index and word[1] == "prime":
        return replace(found, primes=True)
    if isinstance(part, ast.Comparison) and (bounded := _bounded(found, part)) is not None:
        return bounded
    return replace(found, conditions=(*found.conditions, part))


def _bounded(found: IndexSet, part: ast.Comparison) -> IndexSet | None:
    """The bounds a comparison puts on the index, written beside it: ``a \\le k \\le b``,
    ``k < b``, ``a \\le k`` or ``k \\neq 0``; ``None`` where it says more, as k^2 < n
    does."""
    index = found.index.id
    links = list(zip(part.operands, part.ops, part.operands[1:], strict=False))
    if not any(_is(left, index) or _is(right, index) for left, _, right in links):
        return None
    for left, written, right in links:
        relation: str = written
        if _is(right, index):  # a <= k: as k >= a
            left, relation, right = right, _FLIPPED[relation], left
        if not _is(left, index) or _mentions(right, index):
            return None
        if relation == "!=":
            found = replace(found, excluded=(*found.excluded, right))
        elif relation in ("<", "<="):
            found = replace(found, upper=right, strict=(found.strict[0], relation == "<"))
        elif relation in (">", ">="):
            found = replace(found, lower=right, strict=(relation == ">", found.strict[1]))
        else:
            return None
    return found


_FLIPPED: dict[str, str] = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "!=": "!=", "==": "=="}


def _completed(found: IndexSet) -> IndexSet:
    """The values with what books leave unwritten: p ranges over the primes where no least
    value is written, and an index bounded only above starts at 1, as number theory's sums
    over n ≤ x do."""
    name, notes = found.index.span.text, list(found.notes)
    if found.index.id == "p" and found.lower is None and not found.primes:
        found = replace(found, primes=True)
        notes.append(f"{name} ranges over the primes, as number theory writes it")
    if found.lower is None and found.upper is not None and found.of is None and not found.primes:
        found = replace(found, lower=ast.Number("1", span=found.index.span))
        notes.append(f"{name} runs from 1, as number theory sums over {name} ≤ x")
    return replace(found, notes=tuple(notes))


def _is(node: ast.Expr, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _mentions(node: ast.Node, name: str) -> bool:
    """Whether an expression writes the name."""
    return any(isinstance(each, ast.Name) and each.id == name for each in _walk(node))


# -- coefficients ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Coefficient:
    """``[x^n] F(x)``: the coefficient of the ``power`` of the ``variable`` x in the power
    series of what follows the bracket, ``of``, written ``written``."""

    variable: ast.Name
    power: ast.Expr
    of: ast.Expr
    written: str


def coefficient(product: ast.Binary) -> Coefficient | None:
    """The coefficient a product writes, as combinatorics writes it: a bracket of one power
    of a name first, and what follows it; ``None`` for anything else, as a list times a
    value."""
    factors: list[ast.Expr] = []
    left: ast.Expr = product
    while isinstance(left, ast.Binary) and left.op == "*":
        factors.insert(0, left.right)
        left = left.left
    first, rest = left, factors
    if not (rest and isinstance(first, ast.List) and len(first.items) == 1):
        return None
    found = power_of(first.items[0])
    if found is None:
        return None
    variable, order = found
    if not isinstance(variable, ast.Name):
        return None
    of = rest[0]
    for factor in rest[1:]:
        of = ast.Binary("*", of, factor, implicit=True, span=of.span.to(factor.span))
    return Coefficient(variable, order, of, first.span.text)


# -- congruences ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Congruence:
    """``a \\equiv b \\pmod{m}``: that a and b leave the same remainder on division by the
    ``modulus`` m (design/notation.md, 30)."""

    lhs: ast.Expr
    rhs: ast.Expr
    modulus: ast.Expr


def congruence(relation: ast.Expr) -> Congruence | None:
    """The congruence a relation writes: ``\\equiv`` with a modulus on its right, which is
    of all that side, as in ``a \\equiv b + c \\bmod m``; ``None`` for anything else, as
    ``\\equiv`` without a modulus, which defines."""
    if not (isinstance(relation, ast.Relation) and relation.op == "≡"):
        return None
    split = _modulus_of(relation.rhs)
    if split is None:
        return None
    rhs, modulus = split
    return Congruence(relation.lhs, rhs, modulus)


def _modulus_of(side: ast.Expr) -> tuple[ast.Expr, ast.Expr] | None:
    """A side of a congruence without its modulus, and the modulus: ``b`` and ``m`` of
    ``b \\pmod{m}``, and ``b + c`` and ``m`` of ``b + c \\bmod m``."""
    if isinstance(side, ast.Binary) and side.op == "mod":
        return side.left, side.right
    if isinstance(side, ast.Binary) and side.op in ("+", "-"):
        inner = _modulus_of(side.right)
        if inner is not None:
            rest, modulus = inner
            joined = ast.Binary(side.op, side.left, rest, span=side.left.span.to(rest.span))
            return joined, modulus
    return None


def power_of(node: ast.Expr) -> tuple[ast.Expr, ast.Expr] | None:
    """The base and the exponent of a power, ``a^{k}`` or ``a ** k``, whose remainder is
    computed by repeated squaring, and of ``a^{-1}`` as the inverse; ``None`` for anything
    else."""
    if isinstance(node, ast.Binary) and node.op == "**":
        return node.left, node.right
    if isinstance(node, ast.Scripted) and len(node.scripts) == 1:
        script = node.scripts[0]
        if script.kind == "^" and script.body is not None:
            return node.base, script.body
    return None


def negative(node: ast.Expr) -> bool:
    """Whether an exponent is written negative, as the -1 of ``a^{-1}`` is."""
    if isinstance(node, ast.Unary):
        return node.op == "-"
    return isinstance(node, ast.Number) and node.text.startswith("-")


def residues(symbol: sympy.Symbol, bounds: Sequence[Any]) -> tuple[int, int] | None:
    """The remainder and the modulus the congruences assumed of a symbol leave it, together,
    by the Chinese remainder theorem, as x ≡ 2 (mod 3) and x ≡ 3 (mod 5) leave 8 modulo 15;
    ``None`` where none is assumed, or they contradict each other."""
    from sympy.ntheory.modular import solve_congruence

    found = [
        (int(bound.rhs), int(bound.lhs.args[1]))
        for bound in bounds
        if isinstance(bound, sympy.Eq)
        and isinstance(bound.lhs, sympy.Mod)
        and bound.lhs.args[0] == symbol
    ]
    if not found:
        return None
    together = solve_congruence(*found)
    return None if together is None else (int(together[0]), int(together[1]))


def pinned(symbol: sympy.Symbol, bounds: Sequence[Any]) -> tuple[int, int, int, int, int] | None:
    """The one integer the congruences and the bounds assumed of a symbol leave it, as
    0 ≤ x < 105 leaves x ≡ 23 (mod 105) only 23, with the remainder, the modulus and the
    least and greatest values it was found between; ``None`` where they leave it none, or
    more than one."""
    together = residues(symbol, bounds)
    if together is None:
        return None
    remainder, modulus = together
    low, high = _ends(symbol, bounds)
    if low is None or high is None:
        return None
    first = low + (remainder - low) % modulus
    if first > high or first + modulus <= high:
        return None
    return first, remainder, modulus, low, high


def _ends(symbol: sympy.Symbol, bounds: Sequence[Any]) -> tuple[int | None, int | None]:
    """The least and the greatest integer the bounds and the sign assumed of a symbol allow
    it; ``None`` for an end they leave open."""
    low: int | None = 0 if symbol.is_nonnegative else None
    high: int | None = 0 if symbol.is_nonpositive else None
    if symbol.is_positive:
        low = 1
    if symbol.is_negative:
        high = -1
    for bound in bounds:
        if not isinstance(bound, sympy.core.relational.Relational) or isinstance(
            bound, sympy.Eq | sympy.Ne
        ):
            continue
        small, large = (
            (bound.lhs, bound.rhs) if bound.rel_op in ("<", "<=") else (bound.rhs, bound.lhs)
        )
        strict = bound.rel_op in ("<", ">")
        if small == symbol and large.is_number:
            end = int(sympy.ceiling(large)) - 1 if strict else int(sympy.floor(large))
            high = end if high is None else min(high, end)
        elif large == symbol and small.is_number:
            end = int(sympy.floor(small)) + 1 if strict else int(sympy.ceiling(small))
            low = end if low is None else max(low, end)
    return low, high
