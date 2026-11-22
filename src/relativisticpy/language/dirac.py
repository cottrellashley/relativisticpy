r"""Dirac's notation as books write it, parsed: kets, bras, their brackets and matrix
elements, and angle brackets around what holds no bar.

The lexer gives angle brackets tokens of their own, ``\langle`` and ``\rangle``, and a bar
the token it always is. Which bar belongs to a bra, to a ket or to an absolute value is
found here, by what closes it:

- ``|\psi\rangle`` is a ket: a bar, a label and an angle bracket;
- ``\langle\phi|`` is a bra, ``\langle\phi|\psi\rangle`` the bracket of a bra and a ket,
  which share their bar, and ``\langle\phi|A|\psi\rangle`` a matrix element;
- ``\langle x \rangle``, with no bar, is what the reader reads as an expectation value, an
  inner product, as of ``\langle u, v \rangle``, or a program's own bracket.

A label is what names a state: a name, a number, a sign alone, as in ``|+\rangle``, an
expression, as in ``|n - 1\rangle``, or several, separated by commas and semicolons, as in
``\langle j_1, j_2; m_1, m_2|``. The physics and braket packages' ``\ket{\psi}``,
``\bra{\phi}``, ``\braket{\phi|\psi}`` (or ``\braket{\phi}{\psi}``), ``\ketbra{a}{b}``,
``\expval{A}`` (or ``\expval{A}{\psi}``) and ``\mel{\phi}{A}{\psi}`` read as what they
print.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from relativisticpy.errors import ParseError
from relativisticpy.language import syntax as ast
from relativisticpy.language import vocabulary as vocab
from relativisticpy.language.tokens import Token, TokenKind

if TYPE_CHECKING:
    from relativisticpy.language.parser import Parser

K = TokenKind

#: The commands of the physics and braket packages that write Dirac's notation.
COMMANDS = vocab.DIRAC_COMMANDS

#: What a program may define in Dirac's notation: a ket, a bracket and what angle brackets
#: hold, as ``|0\rangle :=``, ``\langle x|p\rangle :=`` and ``\langle x^2 \rangle :=``.
DEFINABLE = ast.Ket | ast.BraKet | ast.Expectation

_SIGNS = {K.PLUS: "+", K.MINUS: "-", K.PLUS_MINUS: "±", K.MINUS_PLUS: "∓"}
# What ends an item of a label: its bar, its angle bracket or brace, and what separates items.
_ENDS = frozenset({K.BAR, K.RANGLE, K.RBRACE, K.COMMA, K.SEMICOLON})
# What a ket never spans: the end of a line or of a statement.
_OUTSIDE = frozenset({K.EOF, K.NEWLINE, K.ROW_SEP, K.DEFINE})
_OPENING = frozenset({K.LPAREN, K.LBRACKET, K.LBRACE, K.LANGLE, K.BEGIN})
_CLOSING = frozenset({K.RPAREN, K.RBRACKET, K.RBRACE, K.END})


def at_ket(parser: Parser) -> bool:
    """Whether the bar at the cursor opens a ket, ``|\\psi\\rangle``, rather than an
    absolute value: an angle bracket closes what follows it, at its own depth, before a bar
    does. In ``|\\langle k|H|0\\rangle|^2`` the bars of the bracket are deeper."""
    tokens = parser.tokens
    index = parser._significant(parser.pos) + 1
    depth = 0
    while True:
        kind = tokens[index].kind
        if kind in _OUTSIDE:
            return False
        if kind in _OPENING:
            depth += 1
        elif kind in _CLOSING:
            depth -= 1
            if depth < 0:
                return False
        elif kind is K.RANGLE:
            if depth == 0:
                return True
            depth -= 1
        elif kind is K.BAR and depth == 0:
            return False
        index += 1


def ket(parser: Parser) -> ast.Ket:
    """``|\\psi\\rangle``: the bar, the label and the angle bracket that closes it."""
    opener = parser.advance()
    with parser._nested():
        parser._abs_depth = 1  # a bar ends the label, as it ends an absolute value
        content = label(parser)
        _close(parser, K.RANGLE, opener, "ket")
    return ast.Ket(content, span=parser._span(opener))


def angle(parser: Parser) -> ast.Expr:
    """What an angle bracket opens: an expectation value or an inner product, a bra, the
    bracket of a bra and a ket, or a matrix element (see the module's docstring)."""
    opener = parser.advance()
    return _bracketed(parser, opener, K.RANGLE)


def command(parser: Parser) -> ast.Expr:
    """``\\ket{\\psi}``, ``\\bra{\\phi}``, ``\\braket{\\phi|\\psi}``, ``\\ketbra{a}{b}``,
    ``\\expval{A}`` or ``\\mel{\\phi}{A}{\\psi}``: what each prints, read as written."""
    token = parser.advance()
    name = token.value
    if name == "braket":
        brace = parser.expect(K.LBRACE, after=f"\\{name}")
        found = _bracketed(parser, brace, K.RBRACE)
        if isinstance(found, ast.Expectation):  # \braket{a}{b}, or \braket{a}: <a|a>
            other = _argument(parser, token) if parser.check(K.LBRACE) else found.operand
            found = ast.BraKet(found.operand, None, other, span=parser._span(token))
        return _spanned(found, parser, token)
    first = _argument(parser, token)
    if name == "ket":
        return ast.Ket(first, span=parser._span(token))
    if name == "bra":
        return ast.Bra(first, span=parser._span(token))
    if name == "ketbra":
        second = _argument(parser, token)
        span = parser._span(token)
        bra = ast.Bra(second, span=second.span)
        return ast.Binary("*", ast.Ket(first, span=first.span), bra, implicit=True, span=span)
    if name == "expval":
        if not parser.check(K.LBRACE):
            return ast.Expectation(first, span=parser._span(token))
        state = _argument(parser, token)
        return ast.BraKet(state, first, state, span=parser._span(token))
    middle = _argument(parser, token)  # \mel{\phi}{A}{\psi}
    last = _argument(parser, token)
    return ast.BraKet(first, middle, last, span=parser._span(token))


def label(parser: Parser) -> ast.Expr:
    """The label of a bra or a ket, read up to its bar or angle bracket: one item, or items
    separated by commas, a tuple, whose groups semicolons separate, a tuple of them."""
    groups: list[list[ast.Expr]] = [[_item(parser)]]
    while True:
        if parser.accept(K.COMMA):
            groups[-1].append(_item(parser))
        elif parser.accept(K.SEMICOLON):
            groups.append([_item(parser)])
        else:
            break
    items = [_together(group) for group in groups]
    if len(groups) == 1:
        return items[0]
    return ast.Tuple(tuple(items), span=items[0].span.to(items[-1].span))


def _item(parser: Parser) -> ast.Expr:
    """One item of a label: a sign alone, as in ``|+\\rangle`` and ``\\langle -|``, which is a
    name, or an expression."""
    token = parser.peek()
    if token.kind in _SIGNS and parser.peek(1).kind in _ENDS:
        parser.advance()
        return ast.Name(_SIGNS[token.kind], verbatim=True, span=token.span)
    return parser.expression()


def _together(items: list[ast.Expr]) -> ast.Expr:
    """One item, or several as a tuple."""
    if len(items) == 1:
        return items[0]
    return ast.Tuple(tuple(items), span=items[0].span.to(items[-1].span))


def _bracketed(parser: Parser, opener: Token, closer: TokenKind) -> ast.Expr:
    """What follows an opening angle bracket, or the brace of ``\\braket``, up to its
    ``closer``: a label, then the closer, or a bar and what follows a bra's bar."""
    with parser._nested():
        parser._abs_depth = 1
        first = label(parser)
        if parser.accept(closer):
            return ast.Expectation(first, span=parser._span(opener))
        if not parser.check(K.BAR):
            raise _unclosed(parser, opener, closer)
        parser.advance()
        rest = _after_the_bar(parser, closer)
    if rest is None:  # \langle\phi| alone, a bra
        return ast.Bra(first, span=parser._span(opener))
    operator, second = rest
    return ast.BraKet(first, operator, second, span=parser._span(opener))


def _after_the_bar(parser: Parser, closer: TokenKind) -> tuple[ast.Expr | None, ast.Expr] | None:
    """What follows the bar of a bra inside its brackets: a ket's label and the closer, of
    a bracket; an operator, a bar, a label and the closer, of a matrix element; or nothing
    of either, ``None``, for a bra alone, as in ``|+\\rangle\\langle+| + |-\\rangle\\langle-|``,
    whose ``+`` is no operator."""
    if parser.tokens[parser.pos].kind in _OUTSIDE:  # the line ends after the bra
        return None

    def element() -> tuple[ast.Expr | None, ast.Expr]:
        operator = parser.expression()
        parser.expect(K.BAR)
        content = label(parser)
        parser.expect(closer)
        return operator, content

    def bracket() -> tuple[ast.Expr | None, ast.Expr]:
        content = label(parser)
        parser.expect(closer)
        return None, content

    for alternative in (element, bracket):
        found, _ = parser._attempt(alternative)
        if found is not None:
            return found
    return None


def _argument(parser: Parser, command: Token) -> ast.Expr:
    """A braced argument of one of the :data:`COMMANDS`: a label, or an operator."""
    brace = parser.expect(K.LBRACE, after=f"\\{command.value}")
    with parser._nested():
        content = label(parser)
        parser.expect(K.RBRACE, opener=brace)
    return content


def _spanned(node: ast.Expr, parser: Parser, start: Token) -> ast.Expr:
    """``node``, spanning from ``start`` to what was read last, as a command spans its
    arguments."""
    return replace(node, span=parser._span(start))


def _close(parser: Parser, closer: TokenKind, opener: Token, what: str) -> None:
    """Read the angle bracket that closes a ket, or say what was found instead."""
    if not parser.check(closer):
        raise _unclosed(parser, opener, closer, what)
    parser.advance()


def _unclosed(
    parser: Parser, opener: Token, closer: TokenKind, what: str | None = None
) -> ParseError:
    """The error for angle brackets, or a ket, that are not closed where they should be."""
    found = parser.peek()
    close = "}" if closer is K.RBRACE else closer.value
    if what is None:
        expected = f"'|' or '{close}'"
        what = "bracket"
    else:
        expected = f"'{close}'"
    span = found.span.labelled(f"expected {expected} here", (opener.span, f"the {what} opens here"))
    return ParseError(
        f"expected {expected} to close the {what}, found {found.describe()}",
        span=span,
        hint=r"a ket is written |\psi\rangle, a bra \langle\phi| and their bracket "
        r"\langle\phi|\psi\rangle",
    )
