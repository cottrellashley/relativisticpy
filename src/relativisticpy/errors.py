"""Exceptions raised by RelativisticPy.

Errors caused by user input carry the :class:`~relativisticpy.language.source.Span`
of source text they refer to, and render it as an underlined excerpt::

    IndexNotationError: the free indices of a sum must agree: _{a} and _{b}
     --> <input>:3:1
      |
    3 | V_{a} + W_{b}
      | ^^^^^^^^^^^^^

Every error derives from :class:`RelativisticPyError`, and each stage of the
pipeline raises its own kind: the parser raises :class:`ParseError`, the
compiler :class:`CompileError` and the executor :class:`EvaluationError`.
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


class CompileError(RelativisticPyError):
    """A well-formed statement does not make sense: an unknown tensor, a function
    called with the wrong number of arguments, a metric without coordinates."""


class IndexNotationError(CompileError):
    """Tensor indices break the rules of index notation: mismatched free indices in
    a sum, an index repeated three times, two lower indices summed together."""


class EvaluationError(RelativisticPyError):
    """A computation failed while running: a degenerate metric, an equation SymPy
    cannot solve."""
