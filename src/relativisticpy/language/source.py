r"""Source text and the spans that point into it.

Every token and syntax-tree node remembers the exact slice of source text it
came from. That is what lets an error message underline the characters that
caused it, however deep in the pipeline the error is discovered.

An excerpt shows every line a span covers, with carets under the text it covers,
aligned by the width the text takes on screen: a tab, a CJK character or a
combining accent moves the carets as it moves the text. A span may carry a label,
and labelled spans of related places, which the excerpt marks with dashes::

     --> <input>:1:16
      |
    1 | Coordinates := [t, r, \theta, \phi
      |                ^ this '[' is never closed
    2 | g_{\mu\nu} := Schwarzschild(M)
      |            -- the next statement starts here
"""

from __future__ import annotations

import unicodedata
from bisect import bisect_right
from dataclasses import dataclass, field, replace
from functools import cached_property
from itertools import pairwise
from typing import NamedTuple

# A span over more lines than this shows its first and last lines, and '...' between.
_LONG = 8
_SHOWN = 3  # the lines shown at each end of a long span
_TAB = 8


class Piece(NamedTuple):
    """Where a part of a source's text came from, when the text is an expansion of
    what was written: from ``start`` in the text on, ``written[written_start:written_end]``,
    copied as it is, or, when not ``copied``, expanded, as a macro is."""

    start: int
    written_start: int
    written_end: int
    copied: bool


@dataclass(frozen=True, eq=False)
class Source:
    """A named piece of program text.

    Sources compare by identity: two inputs with the same text are still
    different sources, and spans from one must never be merged with the other.

    ``origin`` is the line and column at which the text begins in its file, for
    text taken from a larger document such as a Markdown notebook. Positions
    are then reported as the file has them.

    ``latex`` says that the whole text is LaTeX, as the inside of a ``$$`` display
    is: every statement in it is then read as LaTeX, whatever its own tokens.

    ``written`` is, for text made by expanding a document's macros, the text as the
    file has it, and ``pieces`` says where each part of ``text`` came from in it, so
    that positions and excerpts are still the file's.

    ``unreadable`` says why the text cannot be read, and where in it: reading it
    raises a :class:`~relativisticpy.errors.ParseError` there. A document gives
    this to an equation whose macros do not expand.

    ``definitions`` says where each of the document's macros is defined, as
    ``("Gam", "line 7")``, so that an error in an expansion can say so.
    """

    text: str
    name: str = "<input>"
    origin: tuple[int, int] = (1, 1)
    latex: bool = False
    written: str | None = None
    pieces: tuple[Piece, ...] = ()
    unreadable: tuple[str, int, int] | None = None
    definitions: tuple[tuple[str, str], ...] = ()

    def __repr__(self) -> str:
        """``Source('notes.tex', line 12, '\\begin{equation} x = 1 \\end{equation}')``,
        the text shortened when it is long."""
        where = f", line {self.origin[0]}" if self.origin != (1, 1) else ""
        return f"Source({self.name!r}{where}, {_shortened(self.text)!r})"

    @property
    def _file_text(self) -> str:
        return self.text if self.written is None else self.written

    @cached_property
    def _line_starts(self) -> tuple[int, ...]:
        starts = [0]
        starts.extend(i + 1 for i, char in enumerate(self._file_text) if char == "\n")
        return tuple(starts)

    @cached_property
    def _piece_starts(self) -> tuple[int, ...]:
        return tuple(piece.start for piece in self.pieces)

    def written_range(self, start: int, end: int) -> tuple[int, int]:
        """Where the characters ``[start, end)`` of the text are in the text as written:
        the same characters, unless the text is an expansion of what was written."""
        if self.written is None or not self.pieces:
            return start, end
        if start >= len(self.text):
            return len(self.written), len(self.written)
        first = self._piece(start)
        written_start = first.written_start + (start - first.start if first.copied else 0)
        if end <= start:
            return written_start, written_start
        last = self._piece(end - 1)
        written_end = last.written_start + end - last.start if last.copied else last.written_end
        return written_start, max(written_start, written_end)

    def _piece(self, offset: int) -> Piece:
        return self.pieces[max(0, bisect_right(self._piece_starts, offset) - 1)]

    def expansion_at(self, offset: int) -> str | None:
        """The macro call, as written, that the character at ``offset`` was expanded
        from, as ``\\Gam`` for any character of ``\\Gama`` when ``\\Gam`` stands for it;
        ``None`` for a character written as it is."""
        if self.written is None or not self.pieces or offset >= len(self.text):
            return None
        piece = self._piece(offset)
        if piece.copied:
            return None
        return self.written[piece.written_start : piece.written_end]

    def location(self, offset: int) -> tuple[int, int]:
        """Return the 1-based ``(line, column)`` in the file of a character offset."""
        return self._position(self.written_range(offset, offset)[0])

    def _position(self, offset: int) -> tuple[int, int]:
        """The 1-based ``(line, column)`` in the file of an offset in the written text."""
        index = bisect_right(self._line_starts, offset) - 1
        column = offset - self._line_starts[index] + 1
        if index == 0:
            column += self.origin[1] - 1
        return index + self.origin[0], column

    def line_text(self, line: int) -> str:
        """Return the text of a 1-based line of the file, without its line break. On
        the first line, whatever comes before the source in the file is blank."""
        text = self._file_text
        index = line - self.origin[0]
        start = self._line_starts[index]
        end = text.find("\n", start)
        text = text[start:] if end == -1 else text[start:end]
        return " " * (self.origin[1] - 1) + text if index == 0 else text

    @property
    def last_line(self) -> int:
        """The number in the file of the source's last line."""
        return self.origin[0] + len(self._line_starts) - 1


@dataclass(frozen=True)
class Span:
    """A half-open range ``[start, end)`` of character offsets in a source.

    ``label`` says, in an excerpt, what is at the span, after its carets, and
    ``others`` are related places, each with its label, which the excerpt marks with
    dashes: where a bracket was opened, or the other side of an equation. Neither
    takes part in comparing spans. :meth:`labelled` gives a span its labels.
    """

    source: Source = field(repr=False)
    start: int
    end: int
    label: str = field(default="", compare=False, repr=False)
    others: tuple[tuple[Span, str], ...] = field(default=(), compare=False, repr=False)

    @property
    def text(self) -> str:
        return self.source.text[self.start : self.end]

    @property
    def written(self) -> str:
        """The text as the file has it: ``\\dd s^2``, where :attr:`text` has the
        expansion of the document's macro, ``\\mathrm{d} s^2``."""
        start, end = self.source.written_range(self.start, self.end)
        return self.source._file_text[start:end]

    @property
    def line(self) -> int:
        return self.source.location(self.start)[0]

    @property
    def column(self) -> int:
        return self.source.location(self.start)[1]

    def to(self, other: Span) -> Span:
        """Return the smallest span covering both ``self`` and ``other``."""
        if other.source is not self.source:
            raise ValueError("cannot merge spans from different sources")
        return Span(self.source, min(self.start, other.start), max(self.end, other.end))

    def labelled(self, label: str, *others: tuple[Span, str]) -> Span:
        """This span with a label, and labelled spans of related places::

        span.labelled("this '[' is never closed", (after, "the next statement starts here"))
        """
        return replace(self, label=label, others=tuple(others))

    def __repr__(self) -> str:
        """``Span(<input>:1:3, 'a_b')``: where the span starts in its file, and its text."""
        line, column = self.source.location(self.start)
        return f"Span({self.source.name}:{line}:{column}, {_shortened(self.text)!r})"

    def render(self) -> str:
        """Render the span as a source excerpt with the offending text underlined.

        ::

              --> <input>:3:9
               |
             3 | V_{a} + W_{b}
               |         ^^^^^

        Every line the span covers is shown; its label, and its related places with
        their labels, are shown too.
        """
        marks = [_Mark.of(self, self.label, primary=True)]
        marks.extend(_Mark.of(span, label, primary=False) for span, label in self.others)
        sources: list[Source] = []
        for mark in marks:
            if mark.source not in sources:
                sources.append(mark.source)
        width = len(str(max(mark.last[0] for mark in marks)))
        blocks = []
        for number, source in enumerate(sources):
            own = [mark for mark in marks if mark.source is source]
            blocks.append(_block(source, own, width, first=number == 0))
        return "\n".join(blocks)


# -- excerpts ---------------------------------------------------------------------------


def _shortened(text: str, limit: int = 40) -> str:
    """``text`` on one line, shortened to about ``limit`` characters."""
    flat = " ".join(text.split()) if "\n" in text else text
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


@dataclass(frozen=True)
class _Mark:
    """A span as an excerpt marks it: the file positions of its first and last
    characters, its label, and whether it is the error's own place (``^``) or a
    related one (``-``)."""

    source: Source
    first: tuple[int, int]
    last: tuple[int, int]
    label: str
    primary: bool

    @classmethod
    def of(cls, span: Span, label: str, *, primary: bool) -> _Mark:
        source = span.source
        start, end = source.written_range(span.start, span.end)
        text = source._file_text
        while end > start and text[end - 1] in " \t\r\n":  # the text, not the space after it
            end -= 1
        first = source._position(start)
        last = source._position(end - 1) if end > start else first
        return cls(source, first, last, label, primary)


def _block(source: Source, marks: list[_Mark], width: int, *, first: bool) -> str:
    """The excerpt of one source: a header, then each line that a mark covers."""
    primary = marks[0]
    arrow = "-->" if first else ":::"
    gutter = " " * width
    rows = [f"{gutter}{arrow} {source.name}:{primary.first[0]}:{primary.first[1]}", f"{gutter} |"]
    previous = None
    for line in _lines(source, marks):
        if previous is not None and line > previous + 1:
            rows.append("...")
        previous = line
        text = source.line_text(line)
        shown, columns = _expand(text)
        rows.append(f"{line:>{width}} | {shown}".rstrip())
        rows.extend(f"{gutter} | {row}" for row in _marker_rows(line, text, columns, marks))
    return "\n".join(rows)


def _lines(source: Source, marks: list[_Mark]) -> list[int]:
    """The lines to show: those the marks cover, the middle of a long span elided, and a
    single line between two shown, which says more than '...' would."""
    lines: set[int] = set()
    for mark in marks:
        first, last = mark.first[0], mark.last[0]
        if last - first + 1 > _LONG:
            lines.update(range(first, first + _SHOWN))
            lines.update(range(last - _SHOWN + 1, last + 1))
        else:
            lines.update(range(first, last + 1))
    ordered = sorted(lines)
    for before, after in pairwise(ordered):
        if after == before + 2:
            lines.add(before + 1)
    return sorted(line for line in lines if line <= source.last_line)


def _expand(text: str) -> tuple[str, list[int]]:
    """``text`` as it is shown, tabs expanded and control characters made visible, and
    the column on screen at which each of its characters starts, with its end last."""
    shown: list[str] = []
    columns: list[int] = []
    column = 0
    for char in text:
        columns.append(column)
        if char == "\t":
            step = _TAB - column % _TAB
            shown.append(" " * step)
            column += step
            continue
        if unicodedata.category(char) == "Cc":
            char = chr(0x2400 + ord(char)) if ord(char) < 0x20 else "␡"
        shown.append(char)
        column += _width(char)
    columns.append(column)
    return "".join(shown), columns


def _width(char: str) -> int:
    """How many columns of a terminal a character takes: two for a wide one, as CJK
    characters are, and none for a combining accent or an invisible one."""
    if unicodedata.combining(char) or unicodedata.category(char) in ("Mn", "Me", "Cf"):
        return 0
    return 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1


def _marker_rows(line: int, text: str, columns: list[int], marks: list[_Mark]) -> list[str]:
    """The rows of carets and dashes under one line, with their labels."""
    segments = []  # (start, width, marker, label), in screen columns
    for mark in marks:
        if not mark.first[0] <= line <= mark.last[0]:
            continue
        if line == mark.first[0]:
            start = mark.first[1] - 1
        else:  # a span from an earlier line: from the first character on this one
            start = len(text) - len(text.lstrip())
            if start == len(text):
                continue  # a blank line inside the span
        end = mark.last[1] - 1 if line == mark.last[0] else len(text) - 1
        start = min(start, len(text))
        if mark.first == mark.last or end < start:  # an empty span: one caret
            end = start
        left = columns[start]
        right = columns[min(end, len(text) - 1) + 1] if end < len(text) else left + 1
        label = mark.label if line == mark.last[0] else ""
        segments.append((left, max(1, right - left), "^" if mark.primary else "-", label))
    segments.sort(key=lambda segment: segment[0])
    together = all(
        before[0] + before[1] <= after[0] and not before[3] for before, after in pairwise(segments)
    )
    groups = [segments] if together else [[segment] for segment in segments]
    rows = []
    for group in groups:
        row = ""
        for left, size, marker, _ in group:
            row += " " * (left - len(row)) + marker * size
        label = group[-1][3] if group else ""
        rows.append(f"{row} {label}" if label else row)
    return [row for row in rows if row]
