r"""Readings: how the reader read each statement, as :meth:`Workbook.explain` shows them.

A :class:`Reading` is a statement annotated with what the reader decided of it: how each
name was resolved (:class:`Resolution`), each ambiguous form it decided, and why
(:class:`Decision`), and the notes it gave where it took the less likely reading. An
:class:`Explanation` holds the readings of a program, and reads as text::

    >>> from relativisticpy import Workbook
    >>> print(Workbook().explain(r"y = \frac{2GM}{r}"))
    line 1: y = \frac{2GM}{r}
      definition or claim: y = ... is a definition, since y has no meaning yet
      letters: GM is G M, since in LaTeX a run of letters that names nothing is a product
      G: a symbol
      M: a symbol
      r: a symbol
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from relativisticpy import display
from relativisticpy.errors import RelativisticPyError
from relativisticpy.language import syntax as ast
from relativisticpy.language.source import Span


@dataclass(frozen=True)
class Resolution:
    """How a name was resolved, against the scope, the session or the coordinates:
    ``written`` as the statement writes it, ``meaning`` what it was found to be, and,
    where it is known without computing anything, as of a constant, its ``value``."""

    written: str
    meaning: str
    span: Span | None = field(default=None, compare=False, repr=False)
    value: Any = field(default=None, compare=False, repr=False)

    def text(self) -> str:
        """The resolution on one line, with its value where it has one known."""
        shown = f"{self.written}: {self.meaning}"
        if self.value is not None:
            shown += f", {_shown(self.value)}"
        return shown


@dataclass(frozen=True)
class Decision:
    """An ambiguous form, decided: of which ``kind`` the ambiguity is, as ``product or
    call``; what is ``written``; how it was read; ``why``; and the ``note`` the reader
    gave, where it took the less likely reading."""

    kind: str
    written: str
    reading: str
    why: str
    span: Span | None = field(default=None, compare=False, repr=False)
    note: str | None = field(default=None, compare=False, repr=False)

    def text(self) -> str:
        """The decision on one line."""
        return f"{self.kind}: {self.written} is {self.reading}, since {self.why}"


#: The kinds of decision the reader takes, in the order the text lists them.
KINDS = (
    "definition or claim",
    "notation or components",
    "product or call",
    "index or power",
    "letters",
    "constant or symbol",
    "function named alone",
)


@dataclass
class Reading:
    """A statement as the reader read it: how each name was resolved, each decision it
    took, with why, and the notes on how it was read; and the error the statement gives
    when compiled, if it gives one. Nothing is computed to read it."""

    statement: ast.Statement
    names: list[Resolution] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: RelativisticPyError | None = None

    @property
    def span(self) -> Span:
        return self.statement.span

    def text(self) -> str:
        """The reading as lines of text: the statement, then the decisions, the names,
        the notes and the error, indented below it."""
        written = " ".join(self.span.text.split())
        lines = [f"line {self.span.line}: {written}"]
        lines.extend(f"  {decision.text()}" for decision in self.decisions)
        lines.extend(f"  {name.text()}" for name in self.names)
        lines.extend(f"  note: {note}" for note in self.notes)
        if self.error is not None:
            lines.append(f"  {self.error.kind}: {self.error.message}")
        return "\n".join(lines)


@dataclass
class Explanation:
    """The readings of a program's statements, in order (see :meth:`Workbook.explain`).
    ``str()`` gives them as text."""

    readings: list[Reading]

    def __iter__(self) -> Iterator[Reading]:
        return iter(self.readings)

    def __len__(self) -> int:
        return len(self.readings)

    def __getitem__(self, position: int) -> Reading:
        return self.readings[position]

    def text(self) -> str:
        """Every reading, one after another."""
        return "\n".join(reading.text() for reading in self.readings)

    def __str__(self) -> str:
        return self.text()

    def __repr__(self) -> str:
        count = len(self.readings)
        return f"<Explanation of {count} statement{'s' if count != 1 else ''}>"


def _shown(value: Any) -> str:
    """A value as the user writes it, never a failure."""
    try:
        return display.unicode(value)
    except Exception:  # an explanation must not fail on a value it shows
        return repr(value)
