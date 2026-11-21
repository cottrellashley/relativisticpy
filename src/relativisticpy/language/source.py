"""Source text and the spans that point into it.

Every token and syntax-tree node remembers the exact slice of source text it
came from. That is what lets an error message underline the characters that
caused it, however deep in the pipeline the error is discovered.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from functools import cached_property
from typing import NamedTuple


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
    """

    text: str
    name: str = "<input>"
    origin: tuple[int, int] = (1, 1)
    latex: bool = False
    written: str | None = None
    pieces: tuple[Piece, ...] = ()
    unreadable: tuple[str, int, int] | None = None

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


@dataclass(frozen=True)
class Span:
    """A half-open range ``[start, end)`` of character offsets in a source."""

    source: Source = field(repr=False)
    start: int
    end: int

    @property
    def text(self) -> str:
        return self.source.text[self.start : self.end]

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

    def render(self) -> str:
        """Render the span as a source excerpt with the offending text underlined.

        ::

              --> <input>:3:9
               |
             3 | V_{a} + W_{b}
               |         ^^^^^
        """
        start, end = self.source.written_range(self.start, self.end)  # as the file has it
        line, column = self.source._position(start)
        text = self.source.line_text(line)
        last_line = self.source._position(max(start, end - 1))[0]
        width = len(text) - (column - 1) if last_line > line else end - start
        gutter = " " * len(str(line))
        # Tabs are expanded in the line and the carets alike, so the carets stay under
        # the text they mark, whatever the terminal's tab stops.
        before = text[: column - 1].expandtabs()
        underlined = (text[: column - 1 + max(1, width)]).expandtabs()
        caret = " " * len(before) + "^" * max(1, len(underlined) - len(before))
        return (
            f"{gutter}--> {self.source.name}:{line}:{column}\n"
            f"{gutter} |\n"
            f"{line} | {text.expandtabs()}\n"
            f"{gutter} | {caret}"
        )
