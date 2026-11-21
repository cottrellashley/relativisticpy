"""Token kinds and the token record produced by the lexer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from relativisticpy.language.source import Span


class TokenKind(Enum):
    """Every kind of token. The value is how the token is described in error messages."""

    _value_: str

    # Literals and names
    NUMBER = "number"
    STRING = "string"
    IDENT = "name"
    COMMAND = "LaTeX command"  # a control word the parser interprets, e.g. \frac or \sin
    BEGIN = r"\begin"  # value: the environment name
    END = r"\end"

    # Arithmetic
    PLUS = "+"
    MINUS = "-"
    STAR = "*"
    DOUBLE_STAR = "**"
    SLASH = "/"
    CARET = "^"
    UNDERSCORE = "_"
    BANG = "!"
    PRIME = "'"
    PLUS_MINUS = "±"
    MINUS_PLUS = "∓"
    WEDGE = "∧"
    COMPOSE = "∘"
    HADAMARD = "⊙"

    # Relations and definitions
    EQUAL = "="
    DEFINE = ":="
    EQ_EQ = "=="
    NOT_EQ = "!="
    LESS = "<"
    GREATER = ">"
    LESS_EQ = "<="
    GREATER_EQ = ">="
    ARROW = "->"
    RELATION = "relation"  # value: the relation, e.g. '~', '≈', '∝', '∈'

    # Brackets
    LPAREN = "("
    RPAREN = ")"
    LBRACKET = "["
    RBRACKET = "]"
    LBRACE = "{"
    RBRACE = "}"
    BAR = "|"

    # Punctuation
    COMMA = ","
    COLON = ":"
    SEMICOLON = ";"
    AMPERSAND = "&"
    PERIOD = "."
    ELLIPSIS = "⋯"

    # Layout
    NEWLINE = "line break"
    ROW_SEP = r"\\"
    EOF = "end of input"


@dataclass(frozen=True)
class Token:
    """A token: its kind, normalised value, source span and leading whitespace.

    ``value`` is the normalised text: the canonical name of an identifier
    (``\\theta`` and ``θ`` both give ``theta``), the digits of a number, the
    name of a LaTeX command without its backslash, or the operator itself.
    ``space_before`` records whether whitespace (or anything typographic the
    lexer discarded) separated this token from the previous one;
    ``explicit_space`` whether that included deliberate LaTeX spacing such as
    ``\\,`` or ``\\quad``, which ends the argument of ``\\sin`` in
    ``\\sin\\theta\\, d\\phi``. ``verbatim`` marks a name written inside
    ``\\mathrm{...}`` or ``\\text{...}``, which is always one symbol however many
    letters it has. ``aligned`` marks a token that follows an alignment marker
    ``&``, as at the start of the continuation row of an ``align``. ``wide_space``
    marks one after wide spacing such as ``\\quad``, which can separate equations.
    """

    kind: TokenKind
    value: str
    span: Span
    space_before: bool = False
    explicit_space: bool = False
    verbatim: bool = False
    aligned: bool = False
    wide_space: bool = False

    def describe(self) -> str:
        """Describe the token for an error message, e.g. ``name 'x'`` or ``'+'``."""
        kind = self.kind
        if kind in (TokenKind.IDENT, TokenKind.NUMBER, TokenKind.STRING):
            return f"{kind.value} {self.value!r}"
        if kind is TokenKind.COMMAND:
            return f"\\{self.value}"
        if kind is TokenKind.RELATION:
            return f"'{self.value}'"
        if kind in (TokenKind.BEGIN, TokenKind.END):
            return f"{kind.value}{{{self.value}}}"
        if kind in (TokenKind.NEWLINE, TokenKind.EOF):
            return str(kind.value)
        return f"'{kind.value}'"

    def __repr__(self) -> str:
        return f"Token({self.kind.name}, {self.value!r}, {self.span.start}:{self.span.end})"
