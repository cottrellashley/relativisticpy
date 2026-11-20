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
    >>> [finding.verdict.value for finding in report.findings]
    ['defines', 'holds']

From the command line: ``relativisticpy check notes.tex``.
"""

from __future__ import annotations

import contextlib
import signal
import threading
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import sympy

from relativisticpy.abstract import IndexEquation
from relativisticpy.documents import Document, Equation, math_blocks
from relativisticpy.errors import CompileError, EvaluationError, ParseError
from relativisticpy.language import Span
from relativisticpy.tensors.equation import TensorEquation
from relativisticpy.tensors.simplification import tidy

if TYPE_CHECKING:
    from relativisticpy.workbook import Step, Workbook


class Verdict(Enum):
    """What became of a statement."""

    HOLDS = "holds"  # an equation that is an identity: verified
    DIFFERS = "differs"  # an equation whose sides differ: an error, or one to solve
    DEFINES = "defines"  # a definition, used by what follows
    COMPUTES = "computes"  # an expression, computed
    NOT_COMPUTED = "not computed"  # a relation such as ≈, read but not decided
    NOT_UNDERSTOOD = "not understood"  # read, but meaningless without more context
    NOT_PARSED = "not parsed"  # not yet RelativisticPy

    @property
    def mark(self) -> str:
        return _MARKS[self]


_MARKS = {
    Verdict.HOLDS: "✓",
    Verdict.DIFFERS: "✗",
    Verdict.DEFINES: "=",
    Verdict.COMPUTES: "→",
    Verdict.NOT_COMPUTED: "~",
    Verdict.NOT_UNDERSTOOD: "?",
    Verdict.NOT_PARSED: "!",
}


@dataclass(frozen=True)
class Finding:
    """The verdict on one statement, where it was written, and why."""

    verdict: Verdict
    span: Span
    detail: str = ""

    @property
    def line(self) -> int:
        return self.span.line

    def describe(self, width: int = 72) -> str:
        """``line 9071  ✓ holds  \\Gamma^0_{00} = \\partial_0 \\alpha``, on one line."""
        text = " ".join(self.span.text.split())
        if len(text) > width:
            text = text[: width - 1] + "…"
        head = f"line {self.line:<6} {self.verdict.mark} {self.verdict.value:<15} {text}"
        return head + (f"\n{'':14}{self.detail}" if self.detail else "")


@dataclass
class Report:
    """The findings of a check, in the order of the document."""

    name: str
    equations: int
    findings: list[Finding] = field(default_factory=list)

    def count(self, verdict: Verdict) -> int:
        return sum(1 for finding in self.findings if finding.verdict is verdict)

    def summary(self) -> str:
        """How many statements met each verdict."""
        counts = Counter(finding.verdict for finding in self.findings)
        zero = sum(1 for finding in self.findings if finding.detail == "both sides are zero here")
        lines = [f"{self.name}: {self.equations} equations, {len(self.findings)} statements"]
        for verdict in Verdict:
            if counts[verdict]:
                line = f"  {verdict.mark} {verdict.value:<15} {counts[verdict]:>5}"
                if verdict is Verdict.HOLDS and zero:
                    line += f"   ({zero} with both sides zero)"
                lines.append(line)
        return "\n".join(lines)


def check(
    workbook: Workbook,
    document: str | Path | Document,
    *,
    lines: tuple[int, int] | None = None,
    time_limit: float | None = None,
) -> Report:
    """Check a :class:`~relativisticpy.documents.Document`, or a file: a LaTeX
    document or, by its suffix, a Markdown notebook."""
    if isinstance(document, Document):
        return check_equations(workbook, document.equations(), document.name, lines, time_limit)
    path = Path(document)
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in (".md", ".markdown"):
        blocks = [Equation(source, "$$") for source in math_blocks(text, str(path))]
        return check_equations(workbook, blocks, str(path), lines, time_limit)
    return check(workbook, Document(text, str(path)), lines=lines, time_limit=time_limit)


def check_equations(
    workbook: Workbook,
    equations: list[Equation],
    name: str,
    lines: tuple[int, int] | None = None,
    time_limit: float | None = None,
) -> Report:
    """Check equations in order. Each section starts afresh, as a reader does: texts
    reuse symbols, and \\Omega in one chapter need not be \\Omega in the next."""
    if lines is not None:
        first, last = lines
        equations = [equation for equation in equations if first <= equation.line <= last]
    report = Report(name, len(equations))
    section = equations[0].section if equations else 0
    for equation in equations:
        if equation.section != section:
            workbook.reset()
            section = equation.section
        source = equation.source
        try:
            steps = workbook.each(source)
            while True:
                with _time_limit(time_limit):
                    step = next(steps, None)
                if step is None:
                    break
                report.findings.append(_finding(step))
        except ParseError as error:
            span = error.span or Span(source, 0, len(source.text))
            report.findings.append(Finding(Verdict.NOT_PARSED, span, error.message))
        except TimeoutError:
            span = Span(source, 0, len(source.text))
            detail = f"took longer than {time_limit:g} s"
            report.findings.append(Finding(Verdict.NOT_COMPUTED, span, detail))
    return report


def _finding(step: Step) -> Finding:
    error = step.error
    if isinstance(error, CompileError) and "is read but not computed" in error.message:
        return Finding(Verdict.NOT_COMPUTED, step.span, error.message)
    if isinstance(error, EvaluationError):
        return Finding(Verdict.NOT_UNDERSTOOD, step.span, f"could not compute: {error.message}")
    if error is not None:
        return Finding(Verdict.NOT_UNDERSTOOD, step.span, error.message)
    if not step.values:
        return Finding(Verdict.DEFINES, step.span)
    try:
        verdict, detail = _judge(step.values[-1])
    except Exception as failure:  # a check reports what it cannot judge; it does not stop
        return Finding(Verdict.NOT_UNDERSTOOD, step.span, f"could not judge: {failure}")
    if verdict is Verdict.HOLDS and step.zero:
        detail = "both sides are zero here"
    elif verdict is Verdict.DIFFERS and step.sides is not None:
        lhs, rhs = step.sides
        if isinstance(lhs, sympy.Expr) and isinstance(rhs, sympy.Expr):
            detail = f"the left side minus the right is {_short(tidy(lhs - rhs))}"
    return Finding(verdict, step.span, detail)


def _judge(value: Any) -> tuple[Verdict, str]:
    """Whether a value is a verified identity, a refuted one, or just a value."""
    if value is True or value is sympy.true:
        return Verdict.HOLDS, ""
    if value is False or value is sympy.false:
        return Verdict.DIFFERS, "the two sides are not equal"
    if isinstance(value, IndexEquation):  # a law, such as Einstein's, need not be an identity
        if value.holds():
            return Verdict.HOLDS, "an identity of index notation"
        return Verdict.COMPUTES, f"index notation, not an identity: {_short(value)}"
    if isinstance(value, TensorEquation):
        if value:
            return Verdict.HOLDS, ""
        index, equation = next(iter(value.components().items()))
        return Verdict.DIFFERS, f"component {index}: {_short(equation)}"
    if isinstance(value, sympy.Equality):
        lhs, rhs = value.lhs, value.rhs
        if isinstance(lhs, sympy.Expr) and isinstance(rhs, sympy.Expr):
            return Verdict.DIFFERS, f"the left side minus the right is {_short(tidy(lhs - rhs))}"
        return Verdict.DIFFERS, f"{_short(lhs)} is not {_short(rhs)}"
    return Verdict.COMPUTES, _short(value)


def _short(value: Any, width: int = 100) -> str:
    text = str(value)
    return text if len(text) <= width else text[: width - 1] + "…"


@contextlib.contextmanager
def _time_limit(seconds: float | None) -> Iterator[None]:
    """Raise :class:`TimeoutError` if the block runs longer than ``seconds``. Only on
    platforms with ``SIGALRM``, in the main thread; elsewhere there is no limit."""
    usable = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
    if seconds is None or not usable:
        yield
        return

    def expire(signum: int, frame: Any) -> None:
        raise TimeoutError

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
