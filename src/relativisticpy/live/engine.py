r"""What the editor computes with: a program run in copies of a workbook, a statement at a
time, keeping the workbook as it is after each statement.

A program typed again runs from its first changed statement: those before it keep what
came of them, and the workbook they left. Each run starts from a copy, so that a
half-typed definition never reaches the workbook the editor was given. A part of a
statement, as a selection marks it, is computed in a copy of the workbook as it stands
at that statement.

The engine computes where it is called: in the editor's worker process, whose main
thread lets a time limit stop a statement (see :mod:`relativisticpy.live.worker`), or,
where there is no process to start, in the notebook's own.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from relativisticpy.checking import _time_limit
from relativisticpy.cli import _defined, _defines_text, _record, _snapshot
from relativisticpy.errors import (
    NotComputedError,
    ParseError,
    RelativisticPyError,
    UsageError,
    internal_error,
)
from relativisticpy.language import Source, Span, parse
from relativisticpy.language import syntax as ast
from relativisticpy.live import _written
from relativisticpy.tensors.equation import holds
from relativisticpy.workbook import Step, Workbook, _in_order, _pieces

#: What a part of a statement may be asked for, and the program that computes it.
ACTIONS: dict[str, str] = {
    "compute": "{}",
    "simplify": "simplify({})",
    "expand": "expand({})",
    "holds": "{}",
    "latex": "{}",
}

#: What a computed part is called where its errors say where it is.
SELECTION = "<selection>"


@dataclass(frozen=True)
class _Planned:
    """A statement of a program, before it runs: where it is, and the span of each step
    :meth:`Workbook.each` will give for it, one for each link of a chain."""

    start: int
    end: int
    steps: tuple[tuple[int, int], ...]


@dataclass
class _Statement:
    """A statement of the last run: where it is, what came of it, and a copy of the
    workbook as it was after it, which nothing changes."""

    start: int
    end: int
    steps: list[Step]
    entries: list[dict[str, Any]]
    after: Workbook


class Engine:
    """Runs programs in copies of a workbook, as the editor's source changes, and computes
    parts of their statements.

    :meth:`run` gives what came of each statement as data, the entries of
    ``relativisticpy --format json``, with where each statement and its error are in the
    source. :meth:`compute` gives what came of a part of a statement."""

    def __init__(self, workbook: Workbook | None = None) -> None:
        self.workbook = workbook if workbook is not None else Workbook()
        self.source = ""
        self.statements: list[_Statement] = []

    def load(self, workbook: Workbook) -> None:
        """Start from another workbook: every statement runs again."""
        self.workbook = workbook
        self.source = ""
        self.statements = []

    # -- running a program ----------------------------------------------------------

    def run(self, text: str, name: str = "<editor>", limit: float | None = None) -> dict[str, Any]:
        """Run a program, from its first statement that changed since the last run, in
        ``limit`` seconds at most; what came of every statement, as data."""
        started = time.monotonic()
        deadline = math.inf if limit is None else started + limit
        source = Source(text, name)
        plan = _plan(source)
        notes: list[str] = []
        if plan is None:  # the program cannot be read at all: its error says why
            self.source, self.statements = text, []
            return self._result(text, _unreadable(source), notes, started)
        kept = self._kept(text, plan)
        done = self.statements[:kept]
        unfinished: list[dict[str, Any]] = []
        try:
            if kept < len(plan):
                unfinished = self._continue(source, plan, done, deadline, limit, notes)
        finally:  # what finished is kept, even when the run is stopped
            self.source, self.statements = text, done
        entries = [entry for statement in done for entry in statement.entries]
        return self._result(text, entries + unfinished, notes, started)

    def _kept(self, text: str, plan: list[_Planned]) -> int:
        """How many statements of the last run stand as they were: written the same, with
        the same text before them, and read the same."""
        common = _common_prefix(text, self.source)
        kept = 0
        for old, new in zip(self.statements, plan, strict=False):
            if (old.start, old.end) != (new.start, new.end) or old.end > common:
                break
            if [(s.span.start, s.span.end) for s in old.steps] != list(new.steps):
                break
            kept += 1
        if kept and kept < len(plan):  # the rest must read alone as they read after them
            alone = _plan(Source(_blanked(text, plan[kept].start), "<check>"))
            if alone != plan[kept:]:
                return 0
        return kept

    def _continue(
        self,
        source: Source,
        plan: list[_Planned],
        done: list[_Statement],
        deadline: float,
        limit: float | None,
        notes: list[str],
    ) -> list[dict[str, Any]]:
        """Run the statements from ``len(done)`` on, adding each that finishes to
        ``done``; the entries of one that did not finish, as it ran over its time."""
        first = len(done)
        workbook = (done[-1].after if done else self.workbook).copy()
        if first:  # the statements before it stand, so the rest is read alone
            source = Source(_blanked(source.text, plan[first].start), source.name)
        steps = workbook.each(source, recover=True)
        for index, planned in enumerate(plan[first:], start=first):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                notes.append(_not_run(plan[index:], source, limit))
                return []
            ran: list[Step] = []
            entries: list[dict[str, Any]] = []
            try:
                with _time_limit(None if math.isinf(remaining) else remaining):
                    for _ in planned.steps:
                        before = _snapshot(workbook)
                        step = next(steps, None)
                        if step is None:  # read otherwise than planned, which it never is
                            return entries
                        ran.append(step)
                        entries.append(_entry(step, _snapshot(workbook), before))
            except TimeoutError:
                span = Span(source, planned.start, planned.end)
                entries.append(_failed(_over_time(limit, span), span))
                notes.append(_not_run(plan[index + 1 :], source, limit))
                return entries
            done.append(_Statement(planned.start, planned.end, ran, entries, workbook.copy()))
        return []

    @staticmethod
    def _result(
        text: str, entries: list[dict[str, Any]], notes: list[str], started: float
    ) -> dict[str, Any]:
        return {
            "source": text,
            "statements": entries,
            "notes": notes,
            "ok": all(entry["error"] is None for entry in entries),
            "seconds": round(time.monotonic() - started, 3),
        }

    # -- computing a part of a statement --------------------------------------------

    def compute(
        self,
        text: str,
        start: int,
        end: int,
        action: str = "compute",
        limit: float | None = None,
        name: str = "<editor>",
    ) -> dict[str, Any]:
        """Compute a part of a statement of a program, ``text[start:end]``, or the smallest
        expression that holds it, in a copy of the workbook as it stands at that
        statement: its value, simplified or expanded, or whether it holds. The program
        runs first if it is not the one last run, each in ``limit`` seconds."""
        if action not in ACTIONS:
            raise UsageError(f"a part is computed with one of {', '.join(ACTIONS)}, not {action!r}")
        if text != self.source:
            self.run(text, name, limit)
        start, end = _trimmed(text, start, end)
        found = self._statement_at(start, end)
        if found is None:
            return _answer(action, start, end, text[start:end], error=_outside())
        index, step = found
        if type(step.statement) is ast.Statement:  # the statement could not be read
            return _answer(action, start, end, text[start:end], error=step.error)
        part = _part(step, start, end)
        written = text[part.start : part.end]
        program = Source(ACTIONS[action].format(written), SELECTION, latex=step.statement.latex)
        before = self.statements[index - 1].after if index else self.workbook
        workbook = before.copy()
        try:
            with _time_limit(limit):
                steps = list(workbook.each(program, recover=True))
        except TimeoutError:
            return _answer(action, part.start, part.end, written, error=_over_time(limit, None))
        except RelativisticPyError as error:
            return _answer(action, part.start, part.end, written, error=error)
        return _answered(action, part, written, steps)

    def _statement_at(self, start: int, end: int) -> tuple[int, Step] | None:
        """The statement of the last run that holds ``[start, end)``, and the step of it."""
        for index, statement in enumerate(self.statements):
            if statement.start <= start and end <= statement.end:
                inside = [s for s in statement.steps if s.span.start <= start and end <= s.span.end]
                return index, (inside or statement.steps)[0]
        return None


# -- planning a run -------------------------------------------------------------------


def _plan(source: Source) -> list[_Planned] | None:
    """The statements of a program and the steps of each, as :meth:`Workbook.each` will
    run them; ``None`` for a program that cannot be read at all."""
    try:
        program = parse(source, recover=True)
    except RelativisticPyError:
        return None
    plan = []
    for item, span in _in_order(program):
        if isinstance(item, ParseError):
            steps: tuple[tuple[int, int], ...] = ((span.start, span.end),)
        else:
            steps = tuple((piece.span.start, piece.span.end) for piece in _pieces(item))
        plan.append(_Planned(span.start, span.end, steps))
    return plan


def _common_prefix(one: str, other: str) -> int:
    """How many characters two texts begin with alike."""
    differ = (index for index, (a, b) in enumerate(zip(one, other, strict=False)) if a != b)
    return next(differ, min(len(one), len(other)))


def _blanked(text: str, end: int) -> str:
    """``text`` with what comes before ``end`` made blank, its line breaks kept, so that
    the rest is read alone, at the same places."""
    return "".join(c if c == "\n" else " " for c in text[:end]) + text[end:]


def _unreadable(source: Source) -> list[dict[str, Any]]:
    """The entry of a program that cannot be read at all: its error, over all of it."""
    span = Span(source, 0, len(source.text))
    try:
        parse(source, recover=True)
    except RelativisticPyError as error:
        return [_failed(error, span)]
    return []


# -- what came of a statement, as data -----------------------------------------------


def _entry(step: Step, after: Any, before: Any) -> dict[str, Any]:
    """A step as data: the entry of ``relativisticpy --format json``, with where its
    statement and its error are in the source, the LaTeX it is typeset from, what it
    defines in words, and its error as the command line prints it."""
    defined = _defined(before, after) if step.error is None else []
    try:
        entry = _record(step, defined)
    except Exception as error:  # a failure of our own, while describing it: reported
        failure = internal_error(error, span=step.span, during="describing this result")
        entry = _record(Step(step.span, step.statement, error=failure), [])
        step = Step(step.span, step.statement, error=failure)
    entry["start"], entry["end"] = step.span.start, step.span.end
    entry["read"] = type(step.statement) is not ast.Statement
    entry["typeset"] = _written(step)
    entry["definition"] = _defines_text(defined, "unicode") if defined else None
    if step.error is not None:
        entry["error"] = _error(step.error, step.span)
    return entry


def _error(error: RelativisticPyError, span: Span) -> dict[str, Any]:
    """An error as data, with where it is in the source of the statement it came from,
    and as the command line prints it."""
    data = error.to_dict()
    at = error.span if error.span is not None and error.span.source is span.source else span
    data["start"], data["end"] = at.start, max(at.end, at.start)
    data["rendered"] = error.render()
    return data


def _failed(error: RelativisticPyError, span: Span) -> dict[str, Any]:
    """The entry of a statement that failed before it could be a step."""
    step = Step(span, ast.Statement(span=span), error=error)
    entry = _record(step, [])
    entry["start"], entry["end"] = span.start, span.end
    entry["read"] = False
    entry["typeset"] = span.text
    entry["definition"] = None
    entry["error"] = _error(error, span)
    return entry


def _over_time(limit: float | None, span: Span | None) -> NotComputedError:
    """The error of a statement that ran over its time."""
    seconds = f"{limit:g} s" if limit is not None else "its time"
    return NotComputedError(
        f"over the time limit of {seconds}",
        span=span,
        hint="give the editor longer, as with Editor(time_limit=120), or compute this in "
        "steps, defining its parts first",
    )


def _not_run(rest: list[_Planned], source: Source, limit: float | None) -> str:
    """The note on the statements a time limit left unrun."""
    count = f"{len(rest)} statement{'s' if len(rest) != 1 else ''}"
    seconds = f"{limit:g} s" if limit is not None else "its time"
    if not rest:
        return f"the time limit of {seconds} was reached"
    line = source.location(rest[0].start)[0]
    return f"the time limit of {seconds} was reached: {count}, from line {line} on, did not run"


# -- a part of a statement -------------------------------------------------------------


def _trimmed(text: str, start: int, end: int) -> tuple[int, int]:
    """A selection without the space around it, within the text."""
    start, end = max(0, min(start, len(text))), max(0, min(end, len(text)))
    start, end = min(start, end), max(start, end)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _nodes(node: Any) -> Iterator[ast.Node]:
    """Every node of a syntax tree, from its root."""
    if isinstance(node, ast.Node):
        yield node
        for field in dataclasses.fields(node):
            if field.name != "span":
                yield from _nodes(getattr(node, field.name))
    elif isinstance(node, tuple | list):
        for item in node:
            yield from _nodes(item)


def _part(step: Step, start: int, end: int) -> Span:
    """What of a statement a selection asks for: the smallest expression that holds it;
    if none does, the value of a definition, or else the statement."""
    statement = step.statement
    best = step.span
    for node in _nodes(statement):
        span = node.span
        if not isinstance(node, ast.Expr) or span.source is not step.span.source:
            continue
        holds_it = span.start <= start and end <= span.end
        if holds_it and span.end - span.start < best.end - best.start:
            best = span
    if best is step.span and isinstance(statement, ast.Define):
        return statement.value.span
    return best


def _outside() -> UsageError:
    return UsageError(
        "select a part of one statement, among those that ran",
        hint="a selection that spans two statements, or a statement that has not run yet, "
        "cannot be computed alone",
    )


def _answer(
    action: str, start: int, end: int, part: str, *, error: RelativisticPyError | None = None
) -> dict[str, Any]:
    """What came of computing a part: its values, or its error."""
    return {
        "action": action,
        "start": start,
        "end": end,
        "part": part,
        "values": [],
        "notes": [],
        "holds": None,
        "error": None if error is None else _error(error, error.span or Span(Source(part), 0, 0)),
    }


def _answered(action: str, part: Span, written: str, steps: list[Step]) -> dict[str, Any]:
    """What came of computing a part, from the steps of its program."""
    answer = _answer(action, part.start, part.end, written)
    for step in steps:
        if step.error is not None:
            answer["error"] = _error(step.error, step.span)
            return answer
        entry = _record(step, [])
        answer["values"].extend(entry["values"])
        answer["notes"].extend(entry["notes"])
    values = [value for step in steps for value in step.values or []]
    if action == "holds":
        if not values:
            answer["notes"].append("this defines what it names, and claims nothing")
        else:
            try:
                answer["holds"] = holds(values[-1])
            except TypeError:
                answer["notes"].append("this is an expression, not a claim: it has a value")
    return answer
