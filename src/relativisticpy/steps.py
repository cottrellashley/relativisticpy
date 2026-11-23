r"""The statements of a program and the steps they run as, listed before any runs, and what
came of each step: what it defined, in words, and the step as data.

:meth:`Workbook.each <relativisticpy.workbook.Workbook.each>` runs a program a step at a
time: a step for each statement, one for each link of a chain of relations, ``a = b = c``,
and one for each statement that cannot be read. :func:`statements_of` lists the statements
before anything runs, each with its steps, and :func:`steps_of` the steps alone: the
command line lists them so to say what did not run, and the editor to run a program again
from its first statement that changed. As a program runs, :func:`snapshot` and
:func:`defined_between` tell what each step defined, which :func:`in_words` says in words;
and :func:`as_data` gives each step as data, as ``relativisticpy --format json``, the MCP
server and the editor give it::

    >>> from relativisticpy import Workbook
    >>> from relativisticpy.language import parse
    >>> [span.text for _, span in steps_of(parse("x := 2; a = b = c"))]
    ['x := 2', 'a = b', 'b = c']
    >>> wb = Workbook()
    >>> before = snapshot(wb)
    >>> [step] = wb.each("y := 3")
    >>> in_words(defined_between(before, snapshot(wb)))
    'defines y'

Each value says whether it is a complete answer, and whether it is approximate:
:func:`incomplete` gives the reasons a value is not complete, as ``integral`` where SymPy
found no closed form of one, and :func:`approximate` whether it holds floating-point
numbers::

    >>> [step] = Workbook().each(r"\int_0^1 \sin(\sin x) \, dx")
    >>> value = as_data(step)["values"][0]
    >>> value["complete"], value["reasons"], value["approximate"]
    (False, ['integral'], False)
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import sympy
from sympy.core.function import AppliedUndef

from relativisticpy import display
from relativisticpy.errors import ParseError, place
from relativisticpy.language import Span
from relativisticpy.language import syntax as ast
from relativisticpy.language.syntax import Program
from relativisticpy.tensors import Tensor, arrays, numerical
from relativisticpy.tensors.equation import TensorEquation
from relativisticpy.tensors.indices import latex_label

if TYPE_CHECKING:
    from relativisticpy.workbook import Step, Workbook

__all__ = [
    "LEFT_UNDONE",
    "REASONS",
    "Defined",
    "Planned",
    "Snapshot",
    "approximate",
    "as_data",
    "defined_between",
    "in_words",
    "incomplete",
    "left_undone",
    "snapshot",
    "statements_of",
    "steps_of",
    "value_data",
    "written_in",
]


# -- before a program runs ---------------------------------------------------------------


@dataclass(frozen=True)
class Planned:
    """A statement of a program, before it runs: what is written there, read as a
    ``statement``, or the error that says why it cannot be read; its ``span``; and its
    ``steps``, each what runs and where: the statement, each link of a chain of relations as
    a statement of its own, or the error."""

    statement: ast.Statement | ParseError
    span: Span
    steps: tuple[tuple[ast.Statement | ParseError, Span], ...]


def statements_of(program: Program) -> list[Planned]:
    """The statements of a program, in the order they are written, each with the steps
    :meth:`Workbook.each <relativisticpy.workbook.Workbook.each>` gives for it. A statement
    that cannot be read is the text between what comes before it and what comes after it,
    and one step, its error."""
    planned = []
    for item, span in _in_order(program):
        if isinstance(item, ParseError):
            steps: tuple[tuple[ast.Statement | ParseError, Span], ...] = ((item, span),)
        else:
            steps = tuple((piece, piece.span) for piece in _pieces(item))
        planned.append(Planned(item, span, steps))
    return planned


def steps_of(program: Program) -> list[tuple[ast.Statement | ParseError, Span]]:
    """The steps :meth:`Workbook.each <relativisticpy.workbook.Workbook.each>` gives for a
    program, in order, with their spans: each statement, each link of a chain as a
    statement of its own, and each statement that cannot be read, as its error."""
    return [step for statement in statements_of(program) for step in statement.steps]


def _pieces(statement: ast.Statement) -> list[ast.Statement]:
    """A statement, or each link of a chain of relations as a statement of its own."""
    if isinstance(statement, ast.Evaluate) and isinstance(statement.expr, ast.Chain):
        return [
            ast.Evaluate(link, span=link.span, latex=statement.latex)
            for link in statement.expr.links()
        ]
    return [statement]


#: What separates statements, and the space around them: what a statement that cannot be
#: read does not begin or end with.
_BETWEEN = " \t\r\n;"


def _in_order(program: Program) -> list[tuple[ast.Statement | ParseError, Span]]:
    """The statements of a program and the errors of those that could not be read, in
    the order they are written, each with the span of its statement.

    A statement that cannot be read is the text between what comes before it and what
    comes after it: from the end of the statement before it, or, after another that
    cannot be read either, from the start of the line its error is on; to the start of
    what follows. Separators, space and comments are left out at either end."""
    if not program.errors:
        return [(statement, statement.span) for statement in program.statements]
    items = program.in_order()
    source = program.span.source
    text = source.text

    def own(item: ast.Statement | ParseError) -> Span | None:
        """The span of an item, if it is in this program's text."""
        span = item.span
        return span if span is not None and span.source is source else None

    starts: list[int] = []
    for position, item in enumerate(items):
        previous = items[position - 1] if position else None
        before = own(previous) if previous is not None else None
        done = before.end if before is not None else 0
        span = own(item)
        if not isinstance(item, ParseError):
            starts.append(span.start if span is not None else done)
        elif isinstance(previous, ParseError) and span is not None:
            line = text.rfind("\n", 0, span.start) + 1  # its own statement starts on its line
            starts.append(max(line, min(done, span.start)))
        else:
            starts.append(done)
    found: list[tuple[ast.Statement | ParseError, Span]] = []
    for position, item in enumerate(items):
        if not isinstance(item, ParseError):
            found.append((item, item.span))
            continue
        end = starts[position + 1] if position + 1 < len(items) else len(text)
        start = _skip_between(text, starts[position], end)
        while end > start and text[end - 1] in _BETWEEN:
            end -= 1
        if text.endswith("\\\\", start, end):  # a row break ends a row, not a statement
            end -= 2
            while end > start and text[end - 1] in _BETWEEN:
                end -= 1
        span = own(item)
        if end <= start and span is not None:  # nothing but the error itself
            start, end = span.start, span.end
        found.append((item, Span(source, start, max(start, end))))
    return found


def _skip_between(text: str, start: int, end: int) -> int:
    """Where a statement starts, from ``start``: after separators, space, row breaks and
    lines that are comments."""
    while start < end:
        if text[start] in _BETWEEN:
            start += 1
        elif text.startswith("\\\\", start):
            start += 2
        elif text[start] == "%":  # a comment, to the end of its line
            newline = text.find("\n", start)
            start = end if newline == -1 else newline + 1
        else:
            break
    return min(start, end)


# -- as a program runs -------------------------------------------------------------------


@dataclass(frozen=True)
class Snapshot:
    """What a workbook defines at a moment, to tell what a step defined: its
    ``coordinates``, if it has any, and its other ``names``, each with its kind, how it is
    written and where."""

    coordinates: tuple[str, ...] | None
    names: dict[str, tuple[str, str, Span | None]]


@dataclass(frozen=True)
class Defined:
    """Something a step defined: its ``name``, its ``kind`` and how it is ``written``, as
    :meth:`Workbook.names <relativisticpy.workbook.Workbook.names>` lists them, and whether
    it was defined before and is defined ``again``."""

    name: str
    kind: str
    written: str
    again: bool


def snapshot(workbook: Workbook) -> Snapshot:
    """What a workbook defines now; nothing, if that cannot be told, which only says less
    of what a step defined."""
    try:
        coordinates = workbook.coordinates
        names = {
            name: (definition.kind, definition.written, definition.span)
            for name, definition in workbook.names().items()
            if definition.kind != "coordinate"
        }
    except Exception:  # what a step defined is said, never a reason to stop
        return Snapshot(None, {})
    return Snapshot(tuple(str(c) for c in coordinates) if coordinates else None, names)


def defined_between(before: Snapshot, after: Snapshot) -> list[Defined]:
    """What changed from one snapshot to the next: the coordinates, if they did, and each
    name defined or defined anew, in the order the workbook lists them."""
    found: list[Defined] = []
    if after.coordinates and after.coordinates != before.coordinates:
        found.extend(
            Defined(name, "coordinate", latex_label(name), bool(before.coordinates))
            for name in after.coordinates
        )
    for name, (kind, written, span) in after.names.items():
        old = before.names.get(name)
        if old is None or old[:2] != (kind, written) or old[2] != span:
            found.append(Defined(name, kind, written, old is not None))
    return found


def in_words(defined: Sequence[Defined], form: str = "unicode") -> str:
    """What a step defined, in words, each name written in ``form``: ``defines x``,
    ``redefines x``, ``defines the metric g_{μν} and the coordinates t, r, θ, φ``."""
    coordinates = [written_in(d.written, form) for d in defined if d.kind == "coordinate"]
    phrases = []
    for definition in defined:
        written = written_in(definition.written, form)
        if definition.kind == "metric":
            phrases.append(f"the metric {written}")
        elif definition.kind == "space":
            phrases.append(f"the space {written}")
        elif definition.kind != "coordinate":
            phrases.append(written)
    if coordinates:
        phrases.append("the coordinates " + ", ".join(coordinates))
    verb = "redefines" if all(d.again for d in defined) else "defines"
    return f"{verb} {_listed(phrases)}"


def written_in(written: str, form: str) -> str:
    """A name as it is written, ``g_{\\mu\\nu}``, in a form: ``g_{μν}`` in Unicode, and as
    it is in any other."""
    if form != "unicode":
        return written
    try:
        return display.written_form(written, "unicode")
    except Exception:  # how a name is shown never stops a run
        return written


def as_data(step: Step, defined: Sequence[Defined] = ()) -> dict[str, Any]:
    """A step as data, as ``relativisticpy --format json`` prints it: where its statement
    is, what it is, and what came of it, with what it ``defined`` and each value as
    :func:`relativisticpy.display.data` describes it, and besides whether it is
    ``complete``, the ``reasons`` it is not (see :func:`incomplete`), and whether it is
    ``approximate``."""
    where = place(step.span)
    return {
        "file": where["file"],
        "line": where["line"],
        "column": where["column"],
        "end_line": where["end_line"],
        "end_column": where["end_column"],
        "statement": where["text"],
        "values": [value_data(value, step.notes) for value in step.values or []],
        "defines": [{"name": d.name, "kind": d.kind, "written": d.written} for d in defined],
        "notes": list(step.notes),
        "assuming": list(step.assuming),
        "error": step.error.to_dict() if step.error is not None else None,
    }


def value_data(value: Any, notes: Sequence[str] = ()) -> dict[str, Any]:
    """A value as data: as :func:`relativisticpy.display.data` describes it, and whether it
    is a complete answer, why not, and whether it is approximate, told by the value and
    the ``notes`` of the step that gave it."""
    described = display.data(value)
    reasons = incomplete(value, notes)
    described["complete"] = not reasons
    described["reasons"] = reasons
    described["approximate"] = approximate(value)
    return described


# -- whether a value is an answer --------------------------------------------------------

#: What a value may keep as written, as SymPy could not carry it out, by the reason
#: :func:`incomplete` gives for it, with the note a step says it in.
LEFT_UNDONE: dict[str, str] = {
    "integral": "an integral is left as written: SymPy could not compute it",
    "sum": "a sum is left as written: SymPy found no closed form for it",
    "product": "a product is left as written: SymPy found no closed form for it",
    "limit": "a limit is left as written: SymPy could not compute it",
    "derivative": "a derivative is left as written: SymPy could not take it",
}

#: Every reason a value is not a complete answer, in the order :func:`incomplete` gives
#: them: what of it is kept as written; a call of a function the program does not know,
#: kept as it is; and the time a closed form is given, which ran out before one was found.
REASONS = (*LEFT_UNDONE, "unknown function", "time budget")

#: How a step's note says a call is of a function the program does not know, which it
#: keeps as an unknown function (the reader's ``_check_unknown_function``).
_UNKNOWN_FUNCTION = "is not a function RelativisticPy knows"
#: How a step's note says that SymPy found no closed form in the time it is given, of a
#: sum, a product or an integral, which is kept as written (``runtime.core`` and
#: ``assumptions.integrated``).
_PAST_THE_BUDGET = re.compile(r"SymPy found no closed form of the \w+ in the \S+ s it is given")


def incomplete(value: Any, notes: Sequence[str] = ()) -> list[str]:
    """Why a value is not a complete answer, as :data:`REASONS` names them: none for one
    that is. A value is incomplete where the program kept part of it as written, as it
    could not carry it out: what SymPy left undone (see :func:`left_undone`), a call of a
    function the program does not know, or a closed form whose time ran out, as the
    ``notes`` of its step say.

    >>> x = sympy.Symbol("x")
    >>> incomplete(sympy.Integral(sympy.exp(x**3), (x, 0, 1)))
    ['integral']
    >>> incomplete(sympy.sin(x)), incomplete(sympy.Derivative(sympy.Function("f")(x), x))
    ([], [])
    """
    reasons = left_undone(value)
    parts = symbolic(value)
    if any(_UNKNOWN_FUNCTION in note for note in notes) and any(
        part.atoms(AppliedUndef) for part in parts
    ):
        reasons.append("unknown function")
    if reasons and any(_PAST_THE_BUDGET.search(note) for note in notes):
        reasons.append("time budget")
    return reasons


def left_undone(value: Any) -> list[str]:
    """What of a value SymPy left undone, kept as written, by its reason in
    :data:`LEFT_UNDONE`: an integral, a sum, a product or a limit it did not carry out, and
    a derivative of what is no function; in a list or a mapping, in a tensor's
    components, or in the sides of a tensor equation too. The derivative of an unknown
    function, as f'(x), is the answer, not something left undone."""
    found: dict[str, None] = {}
    for part in symbolic(value):
        if part.has(sympy.Integral):
            found["integral"] = None
        if part.has(sympy.Sum):
            found["sum"] = None
        if part.has(sympy.Product):
            found["product"] = None
        if part.has(sympy.Limit):
            found["limit"] = None
        if any(not d.expr.has(AppliedUndef) for d in part.atoms(sympy.Derivative)):
            found["derivative"] = None
    return [reason for reason in LEFT_UNDONE if reason in found]


def approximate(value: Any) -> bool:
    """Whether a value holds floating-point numbers, as a decimal written in it or a
    numerical method gives: an approximation, which may be complete, but is not exact.

    >>> approximate(sympy.Float(0.5) * sympy.Symbol("x")), approximate(sympy.Rational(1, 2))
    (True, False)
    """
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, float | complex):
        return True
    if isinstance(value, list | tuple):
        return any(approximate(item) for item in value)
    if isinstance(value, Mapping):  # the solutions of a system, by unknown
        return any(approximate(item) for pair in value.items() for item in pair)
    if isinstance(value, Tensor):
        return approximate(value.components)
    if isinstance(value, TensorEquation):
        return approximate(value.lhs) or approximate(value.rhs)
    tabled = getattr(value, "to_numpy", None)  # a table or a series of pandas: its numbers
    if (type(value).__module__ or "").startswith("pandas") and callable(tabled):
        value = tabled()
    if numerical.is_numerical(value):
        kind = getattr(getattr(value, "dtype", None), "kind", "O")
        if kind == "O":  # numbers of Python's or SymPy's, in an array of objects
            return any(approximate(item) for item in arrays.flatten(value))
        return kind in "fc"
    return any(part.has(sympy.Float) for part in symbolic(value))


def symbolic(value: Any) -> list[Any]:
    """The values of SymPy a value holds: itself, the items of a list, the keys and the
    values of a mapping, a tensor's components or the sides of a tensor equation, where
    they are SymPy's."""
    if isinstance(value, list | tuple):
        return [part for item in value for part in symbolic(item)]
    if isinstance(value, Mapping):
        return [part for pair in value.items() for item in pair for part in symbolic(item)]
    if isinstance(value, Tensor):
        return [] if value.numerical else symbolic(value.components)
    if isinstance(value, TensorEquation):
        return [] if value.numerical else [*symbolic(value.lhs), *symbolic(value.rhs)]
    if isinstance(value, sympy.NDimArray):
        return [component for component in arrays.flatten(value) if hasattr(component, "has")]
    if isinstance(value, sympy.Basic | sympy.MatrixBase):
        return [value]
    return []


def _listed(names: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]
