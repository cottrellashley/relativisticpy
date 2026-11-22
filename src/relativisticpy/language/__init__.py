"""The RelativisticPy language front end: source text to syntax tree.

``tokenize`` turns text into tokens and ``parse`` turns text into a syntax
tree. Neither assigns meaning to what it reads; that is the compiler's job.

Both raise a :class:`~relativisticpy.errors.ParseError` at the first thing they
cannot read. ``parse(source, recover=True)`` goes on instead: it returns a
:class:`~relativisticpy.language.syntax.Program` whose ``statements`` are those that
parse and whose ``errors`` are the errors of those that do not, each in the order of
the source, and ``Program.in_order()`` gives both together, so that a run can report
every error and still run every statement that can be read.
"""

from relativisticpy.language.lexer import tokenize
from relativisticpy.language.parser import parse, parse_expression
from relativisticpy.language.source import Source, Span
from relativisticpy.language.syntax import to_sexpr
from relativisticpy.language.tokens import Token, TokenKind

__all__ = [
    "Source",
    "Span",
    "Token",
    "TokenKind",
    "parse",
    "parse_expression",
    "to_sexpr",
    "tokenize",
]
