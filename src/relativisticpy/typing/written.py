r"""Index notation as written: the rule of one term, and how what is written surely breaks
the rules, read from the syntax alone.

The rule of a term is :func:`contraction`'s: an index is written once, and is free, or
twice, once up and once down, and is summed. The typer keeps it as notation is compiled,
in its own words, and :func:`broken_rule` keeps it as a check of a paper judges what the
paper writes (:mod:`relativisticpy.checking`), whatever its tensors are::

    >>> from relativisticpy.language import parse
    >>> [statement] = parse(r"A_\mu B^\mu C_\mu = 0").statements
    >>> print(broken_rule(statement).message)
    the index \mu appears 3 times in one term

A check blames what is written only where it surely breaks a rule, so only plain notation
is judged: tensors written as a name with index labels, numbers and symbols, sums,
products, quotients by scalars and equations. A derivative, a function or an accent may be
read otherwise, and a Latin index twice in one position is the Euclidean convention.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from relativisticpy.abstract import latex_labels
from relativisticpy.language import syntax as ast
from relativisticpy.language.source import Span
from relativisticpy.language.vocabulary import GREEK, GREEK_VARIANTS
from relativisticpy.tensors.indices import latex_label

#: An index as written in a term: its label, and whether it is upper.
Entry = tuple[str, bool]


class Repeated(Exception):
    """An index written in one term more often than the rules allow: three times or more,
    or twice in one position. ``positions`` say where it is written, in order: upper or
    not."""

    def __init__(self, label: str, positions: Sequence[bool]) -> None:
        super().__init__(label)
        self.label = label
        self.positions = tuple(positions)


def contraction(entries: Iterable[Entry]) -> tuple[frozenset[Entry], frozenset[str]]:
    """The free indices of a term, from every index written in it, and the labels it sums:
    an index written once is free, and one written twice, once up and once down, is summed.

    Raises :class:`Repeated` for the first label, in the order the labels are written, that
    breaks the rule: written three times or more, or twice in one position."""
    positions: dict[str, list[bool]] = {}
    for label, upper in entries:
        positions.setdefault(label, []).append(upper)
    free: set[Entry] = set()
    summed: set[str] = set()
    for label, found in positions.items():
        if len(found) > 2 or (len(found) == 2 and found[0] == found[1]):
            raise Repeated(label, found)
        if len(found) == 2:
            summed.add(label)
        else:
            free.add((label, found[0]))
    return frozenset(free), frozenset(summed)


@dataclass(frozen=True)
class BrokenRule:
    """A rule of index notation that what is written surely breaks: the ``message`` that
    says which, the ``hint`` that says how to keep it, and the ``span`` where it is broken."""

    message: str
    hint: str
    span: Span


def broken_rule(statement: ast.Statement, coordinates: Iterable[str] = ()) -> BrokenRule | None:
    """How a statement surely breaks the rules of index notation as written, read from its
    syntax alone, whatever its tensors are: an index three times in one term, a Greek index
    twice in one position, or terms or sides with different free indices; ``None`` if it
    does not, or may not. ``coordinates`` are the names of the coordinates in force, which
    name components, as ``t`` does in ``g_{tt}``."""
    if not isinstance(statement, ast.Evaluate):
        return None
    try:
        _free(statement.expr, frozenset(coordinates))
    except _Broken as broken:
        return broken.rule
    except (_Unsure, RecursionError):  # too deep to judge is not judged
        return None
    return None


class _Unsure(Exception):
    """Notation the syntax alone cannot judge."""


class _Broken(Exception):
    """Notation that surely breaks a rule, as :class:`BrokenRule` says."""

    def __init__(self, message: str, hint: str, span: Span) -> None:
        super().__init__(message)
        self.rule = BrokenRule(message, hint, span)


def _free(node: ast.Expr, coordinates: frozenset[str]) -> frozenset[Entry] | None:
    """The free indices of plain notation, with their positions; ``None`` for zero, which
    has those of the other side or term."""
    if isinstance(node, ast.Equation):
        lhs, rhs = _free(node.lhs, coordinates), _free(node.rhs, coordinates)
        if lhs is not None and rhs is not None and lhs != rhs:
            raise _Broken(
                "the two sides have different free indices: "
                f"{_scripts(lhs)} on the left, {_scripts(rhs)} on the right",
                "write the same free indices on both sides",
                node.span,
            )
        return lhs if lhs is not None else rhs
    if isinstance(node, ast.Binary) and node.op in ("+", "-"):
        common = None
        for term in (node.left, node.right):
            free = _free(term, coordinates)
            if free is not None and common is not None and free != common:
                raise _Broken(
                    "the terms of a sum have different free indices: "
                    f"{_scripts(common)} and {_scripts(free)}",
                    "every term of a sum has the same free indices",
                    node.span,
                )
            common = free if free is not None else common
        return common
    if isinstance(node, ast.Number) and node.text.strip("0.") == "":
        return None
    return _contract(_entries(node, coordinates), node.span)


def _entries(node: ast.Expr, coordinates: frozenset[str]) -> list[Entry]:
    """Every index written in a factor, summed or free, with its position."""
    if isinstance(node, ast.Name) and "_" in node.id:
        raise _Unsure  # B_k: a name, or B with the index k, written without braces
    if isinstance(node, ast.Number | ast.Name):
        return []
    if isinstance(node, ast.Unary) and node.op in ("-", "+"):
        return _entries(node.operand, coordinates)
    if isinstance(node, ast.Binary) and node.op == "*":
        return _entries(node.left, coordinates) + _entries(node.right, coordinates)
    if isinstance(node, ast.Binary) and node.op == "/":
        if _entries(node.right, coordinates):
            raise _Unsure
        return _entries(node.left, coordinates)
    if (isinstance(node, ast.Binary) and node.op in ("+", "-")) or isinstance(node, ast.Equation):
        free = _free(node, coordinates)  # a sum in a product: its free indices
        return sorted(free) if free is not None else []
    if isinstance(node, ast.Scripted) and isinstance(node.base, ast.Name):
        return _tensor_entries(node, coordinates)
    raise _Unsure


def _tensor_entries(node: ast.Scripted, coordinates: frozenset[str]) -> list[Entry]:
    """The indices of a name with scripts: index labels, components, or a power of a
    scalar. A label that is not surely an index, as the H of \\Omega_H, the prime of
    x^{\\mu'} in another frame, or {\\rm T}, makes the notation one not to judge."""
    entries: list[Entry] = []
    for script in node.scripts:
        if _LABELLED.search(script.span.text):
            raise _Unsure  # T^{\mu\nu}_{\rm dust}, \Pi_{\rm c}: words, not indices
        items = script.indices.items if script.indices is not None else None
        if items is not None and all(_index(item, coordinates) is not None for item in items):
            for item in items:
                label = _index(item, coordinates)
                if label:  # not a component, as g_{tt} or \Gamma^1_{00} has
                    entries.append((label, script.kind == "^"))
            continue
        numbers = items is not None and all(item.value is not None for item in items)
        body = script.body if script.kind == "^" and len(node.scripts) == 1 else None
        if body is not None and (numbers or not _entries(body, coordinates)):
            return []  # a power of a scalar, or a component
        raise _Unsure
    return entries


def _index(item: ast.IndexItem, coordinates: frozenset[str]) -> str | None:
    """An index item's label if it is surely an index, a Greek letter or a lowercase
    Latin one; ``""`` if it names a component, by a number or a coordinate; ``None``
    for anything else, which may be a label rather than an index. Without coordinates,
    a letter papers name coordinates with, as in ``g_{\\theta\\theta}``, may be either."""
    if item.derivative is not None or item.opens or item.closes:
        return None
    if item.value is not None:
        return "" if isinstance(item.value, ast.Number) else None
    label = item.label
    if label is None:
        return None
    if label in coordinates:
        return ""
    if not coordinates and label in _COORDINATE_LETTERS:
        return None
    if label in GREEK or label in GREEK_VARIANTS or re.fullmatch("[a-z]", label):
        return label
    return None


#: The commands that set a script as a word or a label rather than as indices.
_LABELLED = re.compile(
    r"\\(?:rm|mathrm|text\w*|mbox|hbox|operatorname|it|mathit|bf|mathbf|sf|mathsf|tt|cal"
    r"|mathcal|hat|bar|tilde|vec|dot)(?![A-Za-z])"
)

#: The letters papers name coordinates with, which, where no coordinates are known, may
#: name components rather than be indices: g_{\theta\theta}, \Gamma^r_{rr}.
_COORDINATE_LETTERS = frozenset(
    {"t", "x", "y", "z", "r", "u", "v", "w", "theta", "vartheta", "phi", "varphi"}
    | {"tau", "chi", "psi", "eta", "xi", "zeta"}
)


def _contract(entries: list[Entry], span: Span) -> frozenset[Entry]:
    """The free indices of a product, from every index written in it, by the rule of a
    term; a Latin index twice in one position is not judged, as the Euclidean convention
    writes it so."""
    try:
        free, _ = contraction(entries)
    except Repeated as repeated:
        written = latex_label(repeated.label)
        count = len(repeated.positions)
        if count > 2:
            raise _Broken(
                f"the index {written} appears {count} times in one term",
                "an index appears once, or twice to be summed",
                span,
            ) from None
        unprimed = repeated.label.rstrip("'")
        if unprimed not in GREEK and unprimed not in GREEK_VARIANTS:
            raise _Unsure from None  # a_i b_i: the Euclidean convention, perhaps
        raise _Broken(
            f"the index {written} appears twice in one term, both times "
            + ("up" if repeated.positions[0] else "down"),
            "a summed index is written once up and once down",
            span,
        ) from None
    return free


def _scripts(free: frozenset[Entry]) -> str:
    """Free indices as scripts, ``^{\\mu}_{\\nu}``, or ``none``."""
    ordered = sorted(free)
    upper = latex_labels([label for label, up in ordered if up])
    lower = latex_labels([label for label, up in ordered if not up])
    written = (f"^{{{upper}}}" if upper else "") + (f"_{{{lower}}}" if lower else "")
    return written or "none"
