r"""Checking a document: run its equations in order and say what became of each.

A paper is a program whose statements are its display equations. Checking one
runs them in order in a single :class:`~relativisticpy.Workbook`, so that
definitions carry forward as they do in the text, and gives every statement,
and every link of a chain such as ``a = b = c``, a :class:`Verdict`::

    >>> from relativisticpy import Workbook
    >>> from relativisticpy.documents import Document
    >>> report = Workbook().check(Document(r'''
    ... \begin{equation} ds^2 = -dt^2 + a(t)^2 dx^2 \end{equation}
    ... The connection has \begin{equation} \Gamma^x_{tx} = \frac{\dot a}{a} \ . \end{equation}
    ... '''))
    >>> [finding.verdict for finding in report]
    ['defines', 'holds']
    >>> report.ok
    True

A :class:`Report` is read as text with :meth:`Report.describe`, which lists what
fails, and as data with :meth:`Report.to_dict`. From the command line:
``relativisticpy check notes.tex``.
"""

from __future__ import annotations

import contextlib
import math
import re
import signal
import textwrap
import threading
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import sympy
from sympy.core.function import AppliedUndef

from relativisticpy import display
from relativisticpy.abstract import IndexEquation
from relativisticpy.documents import Document, Equation, math_blocks, read_text
from relativisticpy.errors import (
    EvaluationError,
    FileError,
    InternalError,
    NotComputedError,
    ParseError,
    RelativisticPyError,
    UsageError,
    internal_error,
    place,
)
from relativisticpy.language import Source, Span, parse
from relativisticpy.language import syntax as ast
from relativisticpy.runtime import core
from relativisticpy.runtime.kernels import differs
from relativisticpy.tensors.equation import TensorEquation, UndecidedError
from relativisticpy.tensors.simplification import tidy
from relativisticpy.typing.written import broken_rule

if TYPE_CHECKING:
    from relativisticpy.workbook import Step, Workbook

__all__ = [
    "EMPTY",
    "LISTED",
    "STRICT",
    "TIME_LIMIT",
    "Finding",
    "Report",
    "Verdict",
    "check",
    "check_equations",
    "line_range",
]


class Verdict(str, Enum):
    """What became of a statement. A verdict is its name as a string, so that
    ``Verdict.DIFFERS == "differs"`` and ``report.count("differs")`` work."""

    HOLDS = "holds"
    DIFFERS = "differs"
    RELATES = "relates"
    ILL_FORMED = "ill-formed"
    DEFINES = "defines"
    COMPUTES = "computes"
    NOT_COMPUTED = "not computed"
    NOT_UNDERSTOOD = "not understood"
    NOT_PARSED = "not parsed"

    __hash__ = str.__hash__  # as the string it equals, so either finds it in a dict

    def __str__(self) -> str:
        return self.value

    def __format__(self, spec: str) -> str:
        return format(self.value, spec)

    def __repr__(self) -> str:
        return repr(self.value)

    @property
    def mark(self) -> str:
        """The mark the verdict is shown with: ``✓``, ``✗``, ``↔``, ``=``, ``→``, ``~``,
        ``?`` or ``!``."""
        return _MARKS[self]

    @property
    def meaning(self) -> str:
        """What the verdict means, in a few words."""
        return _MEANINGS[self]


_MARKS = {
    Verdict.HOLDS: "✓",
    Verdict.DIFFERS: "✗",
    Verdict.RELATES: "↔",
    Verdict.ILL_FORMED: "✗",
    Verdict.DEFINES: "=",
    Verdict.COMPUTES: "→",
    Verdict.NOT_COMPUTED: "~",
    Verdict.NOT_UNDERSTOOD: "?",
    Verdict.NOT_PARSED: "!",
}
_MEANINGS = {
    Verdict.HOLDS: "proven equal",
    Verdict.DIFFERS: "of known quantities, whose sides are not equal: a mistake, or a condition",
    Verdict.RELATES: "relations between quantities nothing defines, not identities",
    Verdict.ILL_FORMED: "notation that breaks the rules of index notation",
    Verdict.DEFINES: "definitions",
    Verdict.COMPUTES: "expressions, computed",
    Verdict.NOT_COMPUTED: "relations read but not decided, or over the time limit",
    Verdict.NOT_UNDERSTOOD: "what RelativisticPy cannot do yet",
    Verdict.NOT_PARSED: "not read",
}

#: A condition :meth:`Report.failed` can fail on besides a verdict: a statement over its
#: time limit.
TIME_LIMIT = "time limit"
#: A condition :meth:`Report.failed` can fail on besides a verdict: a document in which
#: no equation was found.
EMPTY = "empty"
#: What ``--strict`` fails on: an equation that differs, notation that is ill-formed, a
#: statement that could not be parsed or was over its time limit, and a document in
#: which no equation was found.
STRICT: frozenset[str] = frozenset(
    {Verdict.DIFFERS, Verdict.ILL_FORMED, Verdict.NOT_PARSED, TIME_LIMIT, EMPTY}
)
#: What a report lists unless asked for everything: what fails ``--strict``.
LISTED: frozenset[str] = frozenset({Verdict.DIFFERS, Verdict.ILL_FORMED, Verdict.NOT_PARSED})

#: How many differing components of a tensor equation a finding shows.
_SHOWN_COMPONENTS = 3
#: How many reasons for "not understood" the summary gives.
_SHOWN_REASONS = 3


@dataclass(frozen=True)
class Finding:
    """The verdict on one statement: where it was written, and why.

    ``span`` is the statement; ``at``, where a finding points inside it, as an error
    does at the token it is about. ``detail`` says why, ``hint`` how to fix it and
    ``notes`` give context. ``difference`` is the left side minus the right, as a
    SymPy value, for an equation that differs or relates; ``reason`` is why a
    statement is not understood, or ``time limit``; ``error`` is the error the
    statement raised, if it raised one, as running it would give it."""

    verdict: Verdict
    span: Span
    detail: str = ""
    hint: str | None = None
    notes: tuple[str, ...] = ()
    at: Span | None = field(default=None, compare=False)
    difference: Any = field(default=None, compare=False)
    reason: str | None = field(default=None, compare=False)
    error: RelativisticPyError | None = field(default=None, compare=False)

    @property
    def file(self) -> str:
        """The file the statement is in."""
        return self.span.source.name

    @property
    def line(self) -> int:
        """The line of the file the finding points at, from 1."""
        return (self.at or self.span).line

    @property
    def column(self) -> int:
        """The column of the file the finding points at, from 1."""
        return (self.at or self.span).column

    @property
    def location(self) -> str:
        """Where the finding points, as editors and compilers write it: ``paper.tex:12:5``."""
        return f"{self.file}:{self.line}:{self.column}"

    @property
    def text(self) -> str:
        """The statement as the file has it, before any macro is expanded."""
        return _written(self.span)

    def describe(self, width: int | None = None) -> str:
        """The finding as the report lists it::

            paper.tex:12:5: ✗ differs  R_{22} = e^{-2\\beta}[r(\\partial_1\\alpha ...
                the left side minus the right is ...

        Nothing is cut short. With a ``width``, the width of a terminal, long lines are
        wrapped under an indent."""
        text = " ".join(self.text.split())
        head = f"{self.location}: {self.verdict.mark} {self.verdict}  {text}"
        body = [line for line in self.detail.split("\n") if line]
        body += [f"note: {note}" for note in self.notes]
        if self.hint:
            body.append(f"hint: {self.hint}")
        excerpt = [f"    {line}" for line in self.at.render().split("\n")[1:]] if self.at else []
        if width is None:
            return "\n".join([head, *excerpt, *(f"    {line}" for line in body)])
        lines = textwrap.wrap(head, width, subsequent_indent=" " * 8) or [head]
        lines += excerpt  # whose caret stays under what it marks
        for line in body:
            lines += textwrap.wrap(
                line, width, initial_indent=" " * 4, subsequent_indent=" " * 6
            ) or [line]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """The finding as data: its verdict, where it points, the statement as written,
        why, the difference of the sides, in each of the forms of
        :mod:`relativisticpy.display`, and the error the statement raised, as
        ``RelativisticPyError.to_dict()`` gives it, or ``None``."""
        where = place(self.at or self.span)
        return {
            "verdict": self.verdict.value,
            "file": self.file,
            "line": where["line"],
            "column": where["column"],
            "end_line": where["end_line"],
            "end_column": where["end_column"],
            "statement": self.text,
            "detail": self.detail,
            "reason": self.reason,
            "hint": self.hint,
            "notes": list(self.notes),
            "difference": None if self.difference is None else _data(self.difference),
            "error": None if self.error is None else self.error.to_dict(),
        }

    def __repr__(self) -> str:
        text = " ".join(self.text.split())
        detail = f"; {self.detail}" if self.detail else ""
        return f"Finding({self.location} {self.verdict.mark} {self.verdict}: {text}{detail})"


@dataclass
class Report:
    """The findings of a check, in the order of the document.

    It is a sequence of :class:`Finding`: ``len(report)``, ``for finding in report``.
    :meth:`describe` gives it as text, :meth:`to_dict` as data, and :meth:`failed`
    says whether it fails a policy such as ``--strict``'s."""

    name: str
    equations: int
    findings: list[Finding] = field(default_factory=list)
    seconds: float = 0.0
    lines: tuple[int | None, int | None] | None = None
    context: int = 0  # statements run before the lines checked, for their definitions
    warnings: list[str] = field(default_factory=list)

    def __iter__(self) -> Iterator[Finding]:
        return iter(self.findings)

    def __len__(self) -> int:
        return len(self.findings)

    def count(self, verdict: Verdict | str) -> int:
        """How many statements met ``verdict``, a :class:`Verdict` or its name."""
        verdict = _verdict(verdict)
        return sum(1 for finding in self.findings if finding.verdict is verdict)

    def counts(self) -> dict[Verdict, int]:
        """How many statements met each verdict that some statement met."""
        found = Counter(finding.verdict for finding in self.findings)
        return {verdict: found[verdict] for verdict in Verdict if found[verdict]}

    def failures(self, on: Iterable[Verdict | str] = STRICT) -> list[Finding]:
        """The findings that fail a policy: those with one of its verdicts, and, if it
        has :data:`TIME_LIMIT`, those over their time limit."""
        policy = _policy(on)
        return [
            finding
            for finding in self.findings
            if finding.verdict in policy or (TIME_LIMIT in policy and finding.reason == TIME_LIMIT)
        ]

    def failed(self, on: Iterable[Verdict | str] = STRICT) -> bool:
        """Whether the check fails a policy: the verdicts and conditions that fail it, by
        default those of ``--strict`` (:data:`STRICT`). :data:`EMPTY` fails a document in
        which no equation was found::

            report.failed()                          # as --strict
            report.failed({"differs", "ill-formed"})  # only for mistakes
        """
        policy = _policy(on)
        return bool(self.failures(policy)) or (EMPTY in policy and self.equations == 0)

    @property
    def ok(self) -> bool:
        """Whether the check passes ``--strict``: nothing differs, is ill-formed, is not
        parsed or is over its time limit, and some equation was found."""
        return not self.failed()

    def summary(self, width: int | None = None) -> str:
        """How many statements met each verdict, what each verdict means, why statements
        were not understood, and how long the check took."""
        equations, statements = _plural(self.equations, "equation"), _plural(len(self), "statement")
        head = f"{self.name}: {equations}, {statements}"
        if self.lines is not None:
            head += f", in {_range_text(self.lines)}"
        lines = [f"{head}, checked in {self.seconds:.1f} s"]
        zero = sum(1 for finding in self.findings if finding.detail == _BOTH_ZERO)
        for verdict, count in self.counts().items():
            meaning = verdict.meaning
            if verdict is Verdict.HOLDS and zero:
                meaning += f", {zero} with both sides zero"
            if verdict is Verdict.NOT_UNDERSTOOD:
                meaning += ", most often:"
            lines.append(f"  {verdict.mark} {verdict:<15} {count:>5}   {meaning}")
            if verdict is Verdict.NOT_UNDERSTOOD:
                lines += self._reasons(width)
        return "\n".join(_cut(line, width) for line in lines)

    def _reasons(self, width: int | None) -> list[str]:
        """The commonest reasons statements were not understood, one to a line."""
        reasons = Counter(
            finding.reason or finding.detail
            for finding in self.findings
            if finding.verdict is Verdict.NOT_UNDERSTOOD
        )
        common = reasons.most_common(_SHOWN_REASONS)
        lines = [f"{'':28}{count:>5}  {reason}" for reason, count in common]
        rest = sum(reasons.values()) - sum(count for _, count in common)
        if rest:
            others = len(reasons) - len(common)
            lines.append(f"{'':28}{rest:>5}  for {_plural(others, 'other reason')}")
        return lines

    def listing(self, all: bool = False) -> list[Finding]:
        """The findings a report lists: every one with ``all``, and otherwise what fails
        (:data:`LISTED`), and statements over their time limit."""
        if all:
            return list(self.findings)
        return self.failures(LISTED | {TIME_LIMIT})

    def describe(self, *, all: bool = False, width: int | None = None) -> str:
        """The report as text: the :meth:`summary`, then the findings it lists (what fails,
        or every finding with ``all``), then what was not listed and how to list it, and
        any warning. Nothing is cut short unless a terminal's ``width`` is given."""
        parts = [self.summary(width)]
        listed = self.listing(all)
        if listed:
            parts.append("\n".join(finding.describe(width) for finding in listed))
        footer = []
        shown = {id(finding) for finding in listed}
        hidden = Counter(f.verdict for f in self.findings if id(f) not in shown)
        if hidden:
            counts = ", ".join(f"{hidden[v]} {v}" for v in Verdict if hidden[v])
            footer.append(f"Not listed: {counts}. --all lists every statement.")
        if self.context and self.lines is not None:
            footer.append(
                f"Also run, for their definitions, and not reported: "
                f"{_plural(self.context, 'statement')} outside {_range_text(self.lines)}."
            )
        footer += [f"warning: {warning}" for warning in self.warnings]
        if footer:
            parts.append("\n".join(_cut(line, width) for line in footer))
        return "\n\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """The whole report as data, for ``--format json`` and for agents: the document,
        the counts of each verdict, the reasons statements were not understood, every
        finding, the warnings, and whether it fails ``--strict``."""
        reasons = Counter(
            finding.reason or finding.detail
            for finding in self.findings
            if finding.verdict is Verdict.NOT_UNDERSTOOD
        )
        return {
            "file": self.name,
            "equations": self.equations,
            "statements": len(self),
            "seconds": round(self.seconds, 3),
            "lines": list(self.lines) if self.lines is not None else None,
            "context": self.context,
            "counts": {verdict.value: count for verdict, count in self.counts().items()},
            "reasons": dict(reasons.most_common()),
            "findings": [finding.to_dict() for finding in self.findings],
            "warnings": list(self.warnings),
            "ok": self.ok,
        }

    def __repr__(self) -> str:
        return self.describe()

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        printer.text(self.describe())


def check(
    workbook: Workbook,
    document: str | Path | Document,
    *,
    lines: tuple[int | None, int | None] | int | None = None,
    time_limit: float | None = None,
) -> Report:
    """Check a :class:`~relativisticpy.documents.Document`, the text of one, or a file: a
    LaTeX document, with the macros of its local packages and of the files it includes,
    or, by its suffix, a Markdown notebook. Text with a line break or ``\\begin`` that is
    not the path of a file is the text of a document. ``lines`` is a range of lines, as
    :func:`line_range` reads it; ``time_limit`` is in seconds, for each statement; an
    infinite one is no limit."""
    time_limit = _seconds(time_limit)
    lines = _lines(lines)
    if isinstance(document, str) and _is_text(document):
        document = Document(document)
    if not isinstance(document, Document):
        path = _path(document)
        text = read_text(path)
        if path.suffix.lower() in (".md", ".markdown"):
            blocks = [Equation(source, "$$") for source in math_blocks(text, str(path))]
            return check_equations(workbook, blocks, str(path), lines, time_limit)
        document = Document.read(path)
    try:
        equations = document.equations()
    except Exception as error:  # a document that cannot be read is reported, as a statement is
        report = Report(document.name, 0)
        span = Span(Source(document.text, document.name), 0, 0)
        detail = f"cannot read the document: {_why(error)}"
        known = error if isinstance(error, RelativisticPyError) else None
        report.findings.append(Finding(Verdict.NOT_PARSED, span, detail, error=known))
        return report
    report = check_equations(workbook, equations, document.name, lines, time_limit)
    report.warnings += [
        f"{document.name}:{line}: the file {name} it includes was not found, so its "
        "equations were not checked"
        for name, line in document.missing
    ]
    return report


def check_equations(
    workbook: Workbook,
    equations: list[Equation],
    name: str,
    lines: tuple[int | None, int | None] | int | None = None,
    time_limit: float | None = None,
) -> Report:
    """Check equations in order. Each section starts afresh, as a reader does: texts
    reuse symbols, and \\Omega in one chapter need not be \\Omega in the next.

    With ``lines``, the equations of the section before the first line run first, so
    that their definitions are known, as a reader who opens a book at a page has read
    the pages before it; only the statements in the lines are reported, of every
    environment that reaches into them.

    The equations run in a new workbook with the same workers, so that the workbook's
    own definitions neither take part nor are lost. Whatever goes wrong with one
    statement is a finding about it; nothing stops the check."""
    started = time.monotonic()
    time_limit = _seconds(time_limit)
    lines = _lines(lines)
    first, last = lines if lines is not None else (None, None)
    chosen = [equation for equation in equations if _overlaps(equation, first, last)]
    reported = {id(equation) for equation in chosen}
    report = Report(name, len(chosen), lines=lines)
    if chosen:
        section = chosen[0].section
        context = [e for e in equations if e.section == section and _before(e, first)]
        run = context + chosen
    else:
        run = []
    with type(workbook)(workers=workbook._workers, backend=workbook._backend) as session:
        state = _State()
        current = run[0].section if run else 0
        for equation in run:
            if equation.section != current:
                session.reset()
                state = _State()
                current = equation.section
            findings = _check_equation(session, equation.source, time_limit, state)
            if id(equation) in reported:
                kept = [finding for finding in findings if _within(finding, first, last)]
                report.context += len(findings) - len(kept)
                report.findings.extend(kept)
            else:
                report.context += len(findings)
    report.seconds = time.monotonic() - started
    report.warnings += _warnings(equations, chosen, name, first, last)
    return report


def line_range(text: str) -> tuple[int | None, int | None]:
    """The lines ``--lines`` takes: ``120:340``, one line, ``13``, or an open range,
    ``10:`` to the end or ``:10`` from the start."""
    first, colon, last = text.strip().partition(":")
    numbers = [part.strip() for part in (first, last)]
    if not all(part.isdigit() or part == "" for part in numbers) or not text.strip(":").strip():
        raise UsageError(
            f"--lines takes line numbers, as in --lines 120:340, 13, 10: or :10, not {text}"
        )
    start = int(numbers[0]) if numbers[0] else None
    end = int(numbers[1]) if numbers[1] else None
    if not colon:
        end = start
    result = (start, end)
    _lines(result)  # refused if it ends before it starts
    return result


# -- one equation ------------------------------------------------------------------------


@dataclass
class _State:
    """What a check knows of a section, beyond its workbook: the names its relations
    such as ``\\mu, \\nu \\in \\{0, 1, 2, 3\\}`` declared, and the statements over their
    time limit, which what follows may need."""

    declared: set[str] = field(default_factory=set)
    expired: list[int] = field(default_factory=list)


def _check_equation(
    session: Workbook, source: Source, time_limit: float | None, state: _State | None = None
) -> list[Finding]:
    """The findings of one equation: one for each statement it makes, as far as they run."""
    state = state if state is not None else _State()
    findings: list[Finding] = []
    whole = _content(source)
    try:
        steps = session.each(source)
        while True:
            with _time_limit(time_limit):  # computing and judging alike
                step = next(steps, None)
                if step is None:
                    break
                try:
                    finding = _finding(step, session, state)
                except Exception as error:  # judging this one failed: the next goes on
                    failure = internal_error(error, span=step.span, during="judging this")
                    finding = _not_understood(step.span, failure, state)
            findings.append(finding)
    except ParseError as error:
        findings.append(_not_parsed(whole, error))
    except TimeoutError:
        state.expired.append(whole.line)
        over = NotComputedError(
            f"over the time limit of {time_limit:g} s",
            span=whole,
            hint="give it longer, as with --time-limit 120, or inf for no limit",
        )
        findings.append(
            Finding(
                Verdict.NOT_COMPUTED, whole, over.message, over.hint, reason=TIME_LIMIT, error=over
            )
        )
    except Exception as error:  # a failure of RelativisticPy's own: reported, not raised
        try:
            parse(source)
        except ParseError as unreadable:  # what does not parse is first of all that
            findings.append(_not_parsed(whole, unreadable))
        except Exception:  # the parser's own failure, which the next lines report
            failure = internal_error(error, span=whole, during="checking this statement")
            findings.append(_not_understood(whole, failure, state))
        else:
            failure = internal_error(error, span=whole, during="checking this statement")
            findings.append(_not_understood(whole, failure, state))
    return findings


def _not_parsed(whole: Span, error: ParseError) -> Finding:
    """A statement that could not be read, pointing at where reading it failed."""
    at = error.span if error.span is not None and error.span.source is whole.source else None
    return Finding(
        Verdict.NOT_PARSED,
        whole,
        error.message,
        error.hint,
        tuple(error.notes),
        at=at,
        reason="not parsed",
        error=error,
    )


def _finding(step: Step, session: Workbook, state: _State) -> Finding:
    error = step.error
    notes = tuple(step.notes)
    if error is not None:
        at = error.span if error.span is not None and error.span != step.span else None
        if at is not None and at.source is not step.span.source:
            at = None
        if isinstance(error, NotComputedError | UndecidedError):  # a claim not decided
            _declare(step.statement, state)
            return Finding(
                Verdict.NOT_COMPUTED,
                step.span,
                error.message,
                error.hint,
                (*notes, *error.notes),
                at=at,
                reason="not computed",
                error=error,
            )
        coordinates = [coordinate.name for coordinate in session.coordinates or ()]
        broken = broken_rule(step.statement, coordinates)
        if broken is not None:  # the notation is at fault, whatever was said of it
            return Finding(
                Verdict.ILL_FORMED,
                step.span,
                broken.message,
                broken.hint,
                notes,
                at=broken.span if broken.span != step.span else None,
                reason="ill-formed",
                error=error,
            )
        return _not_understood(step.span, error, state, at=at, notes=notes)
    if not step.values:
        return Finding(Verdict.DEFINES, step.span, notes=notes)
    try:
        verdict, detail, difference = _judge(step.values[-1], session, state)
    except Exception as failure:  # a check reports what it cannot judge; it does not stop
        error = internal_error(failure, span=step.span, during="judging this statement")
        return _not_understood(step.span, error, state, notes=notes)
    if verdict is Verdict.HOLDS and step.zero:
        detail = _BOTH_ZERO
    elif verdict is Verdict.DIFFERS:
        lhs, rhs = step.sides if step.sides is not None else (None, None)
        if isinstance(lhs, sympy.Expr) and isinstance(rhs, sympy.Expr):
            infinite = _infinite(lhs, rhs)
            if infinite:
                return Finding(Verdict.NOT_COMPUTED, step.span, infinite, notes=notes)
            difference = tidy(lhs - rhs)
        expr = step.statement.expr if isinstance(step.statement, ast.Evaluate) else None
        if isinstance(expr, ast.Comparison) and set(expr.ops) == {"!="}:
            detail = "the two sides are equal"
        elif isinstance(expr, ast.Comparison) and "==" not in expr.ops:
            detail = "the inequality does not hold"
        elif difference is not None:
            verdict, detail = _differs_or_relates(difference, session, state)
        elif _false(step.values[-1]) and step.sides is not None:
            # sides of other kinds, as a value beside an equation a definition gave
            verdict, detail = _sides_differ_or_relate(lhs, rhs, session, state)
    return Finding(verdict, step.span, detail, notes=notes, difference=difference)


def _not_understood(
    span: Span,
    error: RelativisticPyError,
    state: _State,
    *,
    at: Span | None = None,
    notes: Sequence[str] = (),
) -> Finding:
    """A statement RelativisticPy cannot do yet, and why: by the kind of its error, an
    internal error, a computation that failed, or the error's own message."""
    if isinstance(error, InternalError):
        detail, reason = f"internal error: {error.message}", "internal errors of RelativisticPy"
    elif isinstance(error, EvaluationError):
        detail, reason = f"could not compute: {error.message}", "computations that failed"
    else:
        detail = reason = error.message
    context = [*notes, *error.notes]
    if state.expired:  # what an earlier statement would have defined may be missing
        lines = ", ".join(str(line) for line in state.expired)
        context.append(
            f"line {lines} was over the time limit, so what it defines may be unknown here"
        )
    return Finding(
        Verdict.NOT_UNDERSTOOD,
        span,
        detail,
        error.hint,
        tuple(context),
        at=at,
        reason=reason,
        error=error,
    )


def _judge(value: Any, session: Workbook, state: _State) -> tuple[Verdict, str, Any]:
    """Whether a value is a verified identity, a refuted one, a relation, or just a
    value; why; and the difference of an equation's sides, where it has one."""
    if value is True or value is sympy.true:
        return Verdict.HOLDS, "", None
    if _false(value):
        return Verdict.DIFFERS, "the two sides are not equal", None
    if isinstance(value, IndexEquation):  # a law, such as Einstein's, need not be an identity
        if value.holds():
            return Verdict.HOLDS, "an identity of index notation", None
        unknown = _undefined_tensors(value, session)
        detail = f"index notation, not an identity: {_shown(value)}"
        return Verdict.RELATES, f"nothing defines {_listed(unknown)}: {detail}", None
    if isinstance(value, TensorEquation):
        if value.holds():  # bool() would refuse an equation that is not decided
            return Verdict.HOLDS, "", None
        return _tensor_differs(value, session, state)
    if isinstance(value, sympy.Equality):
        lhs, rhs = value.lhs, value.rhs
        if not (isinstance(lhs, sympy.Expr) and isinstance(rhs, sympy.Expr)):
            # a pair, as a chart map \phi_1(x^1, x^2, x^3) \equiv (y^1, y^2) gives one: it
            # relates what nothing defines, and differs only where everything is known
            unknown = _unknowns(sympy.Tuple(lhs, rhs), session, state)
            if unknown:
                relation = f"{_shown(lhs)} and {_shown(rhs)} are related, not shown equal"
                return Verdict.RELATES, f"nothing defines {_listed(unknown)}: {relation}", None
            return Verdict.DIFFERS, f"{_shown(lhs)} is not {_shown(rhs)}", None
        # Not proven equal. Numbers tell an equation that is false from one that is not
        # proven: the first differs wherever it is tried, the second agrees somewhere.
        shown = differs(lhs, rhs)
        if shown:
            return Verdict.DIFFERS, "", tidy(lhs - rhs)
        if shown is None:
            return (
                Verdict.NOT_COMPUTED,
                "not decided: the sides could not be computed as numbers",
                None,
            )
        return Verdict.NOT_COMPUTED, "the sides agree numerically, but this was not proven", None
    return Verdict.COMPUTES, _shown(value), None


def _differs_or_relates(difference: Any, session: Workbook, state: _State) -> tuple[Verdict, str]:
    """An equation whose sides are not equal differs if every quantity in the difference
    of its sides is known: the coordinates, what the metric is written with, and what
    the document declared or defined. Otherwise it relates quantities nothing defines,
    as a law or an equation to solve does, and is not a mistake."""
    shown = f"the left side minus the right is {_shown(difference)}"
    unknown = _unknowns(difference, session, state)
    if not unknown:
        return Verdict.DIFFERS, shown
    return Verdict.RELATES, f"nothing defines {_listed(unknown)}: {shown}"


def _false(value: Any) -> bool:
    """Whether a value is the falsehood a comparison gives, and no more."""
    return value is False or value is sympy.false


def _sides_differ_or_relate(
    lhs: Any, rhs: Any, session: Workbook, state: _State
) -> tuple[Verdict, str]:
    """An equation whose sides are not equal, and have no difference to show, as a value
    beside an equation that a definition gave: it differs if each side is known, and
    relates quantities nothing defines otherwise, as :func:`_differs_or_relates` says."""
    sides = [side for side in (lhs, rhs) if isinstance(side, sympy.Basic)]
    unknown = _unknowns(sympy.Tuple(*sides), session, state) if sides else []
    if not unknown:
        return Verdict.DIFFERS, "the two sides are not equal"
    relation = f"{_shown(lhs)} and {_shown(rhs)} are related, not shown equal"
    return Verdict.RELATES, f"nothing defines {_listed(unknown)}: {relation}"


def _tensor_differs(
    value: TensorEquation, session: Workbook, state: _State
) -> tuple[Verdict, str, Any]:
    """How a tensor equation that does not hold differs: in how many components, and at
    the first few, labelled by coordinates. It relates if every differing component
    involves a quantity nothing defines."""
    components = value.components()
    total = math.prod(getattr(value.lhs, "shape", ()) or (1,))
    shown: list[str] = []
    unknown: list[str] = []
    every_unknown = True
    for index, equation in components.items():
        label = _component(value, index)
        lhs, rhs = (getattr(equation, side) for side in ("lhs", "rhs"))
        if len(shown) < _SHOWN_COMPONENTS:
            shown.append(f"at {label}, {_shown(lhs)} ≠ {_shown(rhs)}")
        if isinstance(lhs, sympy.Expr) and isinstance(rhs, sympy.Expr):
            names = _unknowns(lhs - rhs, session, state)
            unknown += [name for name in names if name not in unknown]
            every_unknown = every_unknown and bool(names)
        else:
            every_unknown = False
    more = len(components) - len(shown)
    detail = f"differs in {len(components)} of {_plural(total, 'component')}: " + "; ".join(shown)
    if more:
        detail += f"; and {more} more"
    if every_unknown and unknown:
        return Verdict.RELATES, f"nothing defines {_listed(unknown)}: {detail}", None
    return Verdict.DIFFERS, detail, None


def _component(value: TensorEquation, index: tuple[int, ...]) -> str:
    """A component's label, by the coordinates or the bases of its spaces: ``tt``, ``rθ``."""
    names = []
    for axis, position in enumerate(index):
        space = value.spaces[axis] if value.spaces is not None else None
        if space is not None and space.basis is not None:
            names.append(str(space.basis[position]))
        elif value.coordinates and (space is None or space.spacetime):
            names.append(_shown(value.coordinates[position]))
        else:
            names.append(str(position))
    joined = "" if all(len(name) == 1 for name in names) else ","
    return joined.join(names)


# -- what is known --------------------------------------------------------------------------


def _unknowns(difference: Any, session: Workbook, state: _State) -> list[str]:
    """The quantities in the difference of an equation's sides that nothing defines, as
    they are shown: symbols and functions that are not coordinates, not what the metric
    is written with, and not declared or defined in the document."""
    if not isinstance(difference, sympy.Basic):
        return []
    known = _known_names(session, state)
    names: dict[str, None] = {}
    for symbol in sorted(difference.free_symbols, key=sympy.default_sort_key):
        if str(symbol) not in known:
            names[_shown(symbol)] = None
    for function in sorted(difference.atoms(AppliedUndef), key=sympy.default_sort_key):
        if function.func.__name__ not in known:
            names[_shown(function)] = None
    return list(names)


def _known_names(session: Workbook, state: _State) -> set[str]:
    """The names a check knows the meaning of: the coordinates, the names the metric is
    written with, symbols declared with assumptions, functions of the coordinates, the
    names relations declared, and every name the document defined."""
    inner = session._session
    known = set(state.declared)
    known.update(coordinate.name for coordinate in inner.coordinates or ())
    if inner.metric is not None:
        known.update(inner.metric.names)
    known.update(inner.assumptions)
    known.update(inner.functions)
    known.update(name for name, _ in inner.globals.bindings())
    return known


def _declare(statement: ast.Statement, state: _State) -> None:
    """Remember the names a relation such as ``\\mu, \\nu \\in \\{0, 1, 2, 3\\}``
    declares: what they range over is known, though not their values."""
    if not isinstance(statement, ast.Evaluate) or not isinstance(statement.expr, ast.Relation):
        return
    if statement.expr.op not in ("∈", "\\in"):
        return
    lhs = statement.expr.lhs
    names = lhs.items if isinstance(lhs, ast.Tuple | ast.List) else (lhs,)
    state.declared.update(name.id for name in names if isinstance(name, ast.Name))


def _undefined_tensors(value: IndexEquation, session: Workbook) -> list[str]:
    """The tensors of an equation in index notation that nothing defines; the metric
    itself, if there is none and the built-in tensors made of it appear."""
    from sympy.tensor.tensor import Tensor as TensorPart

    inner = session._session
    unknown: dict[str, None] = {}
    builtin = False
    for side in (value.lhs, value.rhs):
        for part in sympy.preorder_traversal(side):
            if not isinstance(part, TensorPart):
                continue
            name, rank = str(part.head.name), int(part.head.rank)
            if session._knows_tensor(name, rank) or name == "KD":
                builtin = builtin or inner.globals.lookup(name) is None
                continue
            unknown[_shown(sympy.Symbol(name))] = None
    if not unknown and inner.metric is None:
        unknown["the metric"] = None
    return list(unknown) if unknown else ["the metric" if builtin else "its tensors"]


# -- text ---------------------------------------------------------------------------------

_BOTH_ZERO = "both sides are zero here"
_OPENER = re.compile(r"\s*(?:\\begin\s*\{[^}]*\}|\$\$|\\\[)\s*")
_CLOSER = re.compile(r"\s*(?:\\end\s*\{[^}]*\}|\$\$|\\\])\s*$")


def _content(source: Source) -> Span:
    """The span of an equation without the delimiters of its display, as
    ``\\begin{equation}`` and ``\\end{equation}``, or ``$$``."""
    text = source.text
    opener = _OPENER.match(text)
    start = opener.end() if opener else len(text) - len(text.lstrip())
    closer = _CLOSER.search(text, start)
    end = closer.start() if closer else len(text.rstrip())
    return Span(source, start, max(start, end))


def _written(span: Span) -> str:
    """The text of a span as the file has it, before macros are expanded, from the
    start of a command whose braces it would otherwise open in the middle of, as in
    ``\\mathrm{solve}(x)``."""
    source = span.source
    start, end = source.written_range(span.start, span.end)
    written = source.written if source.written is not None else source.text
    depth = 0
    for char in written[start:end]:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
    opened = depth < 0
    while depth < 0 and start > 0:  # a closing brace without its opening one
        start -= 1
        if written[start] == "{":
            depth += 1
        elif written[start] == "}":
            depth -= 1
    command = re.search(r"\\[A-Za-z]+\s*$", written[:start]) if opened else None
    if command is not None:  # the command whose argument the brace opens: \mathrm{
        start = command.start()
    return written[start:end]


def _shown(value: Any) -> str:
    """A value as a finding shows it, in the notation of :func:`relativisticpy.display.unicode`.
    A value that cannot be shown, as an integer of more digits than Python prints, is
    described instead, never a failure of the check."""
    try:
        return display.unicode(value)
    except Exception:  # printing must never stop a check
        if isinstance(value, int | sympy.Integer) and value != 0:
            number = abs(int(value))
            digits = sympy.integer_log(number, 10)[0] + 1
            return f"{sympy.N(sympy.Integer(value), 16)} (an integer of {digits} digits)"
        return f"a {type(value).__name__} too large to show"


def _data(value: Any) -> Any:
    """A value as data, never a failure of the check."""
    try:
        return display.data(value)
    except Exception:  # printing must never stop a check
        return {"text": _shown(value)}


def _infinite(lhs: sympy.Expr, rhs: sympy.Expr) -> str:
    """Why an equation cannot be decided because a side is infinite or undefined, as
    after a division by zero; empty if neither is."""
    for side, value in (("left", lhs), ("right", rhs)):
        if value.has(sympy.zoo):
            return f"the {side} side is infinite here: a division by zero"
        if value.has(sympy.nan):
            return f"the {side} side is undefined here, as 0/0 is"
    return ""


def _listed(names: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _plural(count: int, noun: str) -> str:
    """``1 equation``, ``2 equations``."""
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _cut(line: str, width: int | None) -> str:
    """A line of a summary, cut to a terminal's width (never, without one)."""
    if width is None or len(line) <= width:
        return line
    return line[: max(1, width - 1)] + "…"


def _why(error: BaseException) -> str:
    """An unexpected error, as a finding's detail: its kind and its message."""
    if isinstance(error, RelativisticPyError):
        return error.message
    message = " ".join(str(error).split())
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def _verdict(verdict: Verdict | str) -> Verdict:
    """A verdict, from itself or its name; a name that is none is refused, never
    counted as nothing."""
    try:
        return Verdict(verdict)
    except ValueError:
        names = ", ".join(v.value for v in Verdict)
        raise UsageError(f"{verdict!r} is not a verdict: the verdicts are {names}") from None


def _policy(on: Iterable[Verdict | str]) -> set[str]:
    """The verdicts and conditions a policy fails on, each checked: verdicts by their
    names, as ``not parsed`` or ``not-parsed``, and :data:`TIME_LIMIT` (or ``timeout``)
    and :data:`EMPTY`."""
    if isinstance(on, str):
        on = [on]
    names = {verdict.value for verdict in Verdict}
    policy: set[str] = set()
    for item in on:
        text = str(item)
        if text in (TIME_LIMIT, "timeout", "time-limit"):
            policy.add(TIME_LIMIT)
        elif text == EMPTY:
            policy.add(EMPTY)
        else:
            policy.add(_verdict(text if text in names else text.replace("-", " ")))
    return policy


# -- files and lines -----------------------------------------------------------------------


def _is_text(document: str) -> bool:
    """Whether a string given to check is the text of a document rather than a path: it
    has a line break or a display in it, and is not the path of a file."""
    looks = "\n" in document or "\\begin" in document or "$$" in document or "\\[" in document
    if not looks:
        return False
    try:
        return not Path(document).is_file()
    except (OSError, ValueError):  # too long, or not a path at all
        return True


def _path(document: Any) -> Path:
    if not isinstance(document, str | Path):
        raise UsageError(
            "check takes a Document, the path of a file or the text of a document, "
            f"not {type(document).__name__}"
        )
    path = Path(document)
    if path.suffix.lower() == ".ipynb":
        raise FileError(
            f"{path} is a Jupyter notebook, which check does not read",
            hint=f"export it as Markdown, as with  jupyter nbconvert --to markdown {path}",
        )
    return path


def _lines(lines: Any) -> tuple[int | None, int | None] | None:
    """A range of lines, checked: ``(first, last)``, either ``None`` for an open end, or
    one line."""
    if lines is None:
        return None
    if isinstance(lines, int) and not isinstance(lines, bool):
        lines = (lines, lines)
    if not (
        isinstance(lines, tuple | list)
        and len(lines) == 2
        and all(n is None or (isinstance(n, int) and not isinstance(n, bool)) for n in lines)
    ):
        raise UsageError(
            f"lines is a line, or (first, last), either of which may be None, not {lines!r}"
        )
    first, last = lines
    if (first is not None and first < 1) or (last is not None and last < 1):
        raise UsageError(f"lines are numbered from 1, so {_range_text((first, last))} has none")
    if first is not None and last is not None and last < first:
        raise UsageError(f"the lines {first}:{last} end before they start")
    return first, last


def _range_text(lines: tuple[int | None, int | None]) -> str:
    """``line 13``, ``lines 9–13``, ``lines 10 to the end``."""
    first, last = lines
    if first == last and first is not None:
        return f"line {first}"
    if last is None:
        return f"lines {first if first is not None else 1} to the end"
    return f"lines {first if first is not None else 1}–{last}"


def _end_line(equation: Equation) -> int:
    source = equation.source
    return source.location(max(0, len(source.text) - 1))[0] if source.text else equation.line


def _overlaps(equation: Equation, first: int | None, last: int | None) -> bool:
    """Whether an equation reaches into the lines."""
    return (first is None or _end_line(equation) >= first) and (
        last is None or equation.line <= last
    )


def _before(equation: Equation, first: int | None) -> bool:
    return first is not None and _end_line(equation) < first


def _within(finding: Finding, first: int | None, last: int | None) -> bool:
    """Whether a finding's statement reaches into the lines."""
    span = finding.span
    end = span.source.location(max(span.start, span.end - 1))[0]
    return (first is None or end >= first) and (last is None or span.line <= last)


def _warnings(
    equations: list[Equation],
    chosen: list[Equation],
    name: str,
    first: int | None,
    last: int | None,
) -> list[str]:
    """What a report warns of: a document in which no equation was found, and lines in
    which none is."""
    if not equations:
        return [
            f"no display equations were found in {name}: check reads LaTeX documents "
            "(equation, align, eqnarray, gather, multline, $$...$$, \\[...\\]) and the $$ "
            "blocks of Markdown notebooks; to run a program, run it: relativisticpy FILE"
        ]
    if not chosen:
        return [f"{_range_text((first, last))} of {name} have no equations in them"]
    return []


# -- the time limit ------------------------------------------------------------------------


def _seconds(time_limit: float | None) -> float | None:
    """A time limit in seconds, or ``None`` for none, which infinity also means, as does
    anything over thirty years: more than the system's timer can count anyway."""
    if time_limit is None or time_limit == math.inf:
        return None
    if isinstance(time_limit, bool) or not isinstance(time_limit, int | float):
        raise UsageError(f"a time limit is a number of seconds, not {time_limit!r}")
    if not time_limit > 0:  # nor nan
        raise UsageError(f"a time limit is a positive number of seconds, not {time_limit}")
    return float(time_limit) if time_limit < 1e9 else None


class _Expired(BaseException):
    """The alarm of a time limit. Like ``KeyboardInterrupt`` it is not an ``Exception``,
    so no ``except Exception`` on its way out, in the executor or in SymPy, takes it for
    a computation that failed."""


@contextlib.contextmanager
def _time_limit(seconds: float | None) -> Iterator[None]:
    """Raise :class:`TimeoutError` if the block runs longer than ``seconds``, and again
    every :data:`core._AGAIN` seconds until it ends: Python loses an exception raised in a
    callback of a weak reference or in a finalizer, and a statement that lost the only one
    would run for ever. Only on platforms with ``SIGALRM``, in the main thread; elsewhere
    there is no limit."""
    usable = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
    if seconds is None or not usable:
        yield
        return

    def expire(signum: int, frame: Any) -> None:
        if not core._collecting(frame):  # where it would be lost, the next alarm raises it
            raise _Expired

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds, core._AGAIN)
    try:
        try:
            yield
        finally:  # an alarm that rings in here is the block's too
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
    except _Expired:
        raise TimeoutError from None
