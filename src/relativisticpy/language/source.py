"""Source text and the spans that point into it.

Every token and syntax-tree node remembers the exact slice of source text it
came from. That is what lets an error message underline the characters that
caused it, however deep in the pipeline the error is discovered.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from functools import cached_property


@dataclass(frozen=True, eq=False)
class Source:
    """A named piece of program text.

    Sources compare by identity: two inputs with the same text are still
    different sources, and spans from one must never be merged with the other.

    ``origin`` is the line and column at which the text begins in its file, for
    text taken from a larger document such as a Markdown notebook. Positions
    are then reported as the file has them.
    """

    text: str
    name: str = "<input>"
    origin: tuple[int, int] = (1, 1)

    @cached_property
    def _line_starts(self) -> tuple[int, ...]:
        starts = [0]
        starts.extend(i + 1 for i, char in enumerate(self.text) if char == "\n")
        return tuple(starts)

    def location(self, offset: int) -> tuple[int, int]:
        """Return the 1-based ``(line, column)`` in the file of a character offset."""
        index = bisect_right(self._line_starts, offset) - 1
        column = offset - self._line_starts[index] + 1
        if index == 0:
            column += self.origin[1] - 1
        return index + self.origin[0], column

    def line_text(self, line: int) -> str:
        """Return the text of a 1-based line of the file, without its line break. On
        the first line, whatever comes before the source in the file is blank."""
        index = line - self.origin[0]
        start = self._line_starts[index]
        end = self.text.find("\n", start)
        text = self.text[start:] if end == -1 else self.text[start:end]
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
        line, column = self.source.location(self.start)
        text = self.source.line_text(line)
        last_line = self.source.location(max(self.start, self.end - 1))[0]
        width = len(text) - (column - 1) if last_line > line else self.end - self.start
        gutter = " " * len(str(line))
        caret = " " * (column - 1) + "^" * max(1, width)
        return (
            f"{gutter}--> {self.source.name}:{line}:{column}\n"
            f"{gutter} |\n"
            f"{line} | {text}\n"
            f"{gutter} | {caret}"
        )
