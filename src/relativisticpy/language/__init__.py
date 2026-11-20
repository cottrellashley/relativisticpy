"""The RelativisticPy language front end: source text to syntax tree.

``tokenize`` turns text into tokens and ``parse`` turns text into a syntax
tree. Neither assigns meaning to what it reads; that is the compiler's job.
"""

from relativisticpy.language.lexer import tokenize
from relativisticpy.language.source import Source, Span
from relativisticpy.language.tokens import Token, TokenKind

__all__ = ["Source", "Span", "Token", "TokenKind", "tokenize"]
