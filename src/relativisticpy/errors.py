"""Exceptions raised by RelativisticPy.

Errors caused by user input carry the :class:`~relativisticpy.language.source.Span`
of source text they refer to, and render it as an underlined excerpt::

    ParseError: expected '}' to close '{'
      --> <input>:1:4
       |
     1 | g_{mu nu := 1
       |    ^
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from relativisticpy.language.source import Span


class RelativisticPyError(Exception):
    """Base class for every error raised by RelativisticPy."""

    def __init__(self, message: str, *, span: Span | None = None, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.span = span
        self.hint = hint

    def __str__(self) -> str:
        parts = [self.message]
        if self.span is not None:
            parts.append(self.span.render())
        if self.hint is not None:
            parts.append(f"hint: {self.hint}")
        return "\n".join(parts)


class ParseError(RelativisticPyError):
    """The source text is not valid RelativisticPy (raised by the lexer and parser)."""
