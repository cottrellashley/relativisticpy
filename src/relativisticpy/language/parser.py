r"""Parsing: tokens in, syntax tree out.

A Pratt (precedence-climbing) parser. Every infix operator has a binding power
in :class:`Power`, from loosest to tightest::

    expr |_{x=a}         substitution
    a = b                equation
    a or b, a and b      logic
    a < b <= c           comparison (chainable)
    a + b, a - b         sum
    a * b, a / b, a b    product (juxtaposition is multiplication)
    -a, +a               sign
    a ** b               power (right-associative)
    a^b a_b f(x) f' n!   scripts, calls, primes, factorial

LaTeX constructs are recognised here because recognising them is purely a
matter of notation: the many ways of writing a derivative, integrals that end
at their differential, sums, limits, matrices and ``cases``. What anything
*means* (is ``x^{a}`` a power or a vector component? is ``y = 2`` an equation
or an assignment?) is left to the compiler.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from enum import IntEnum
from typing import Literal, TypeVar, cast

from relativisticpy.errors import ParseError
from relativisticpy.language import syntax as ast
from relativisticpy.language import vocabulary as vocab
from relativisticpy.language.lexer import tokenize
from relativisticpy.language.source import Source, Span
from relativisticpy.language.tokens import Token, TokenKind

_T = TypeVar("_T")
K = TokenKind


class Power(IntEnum):
    """Binding powers of infix operators, loosest first."""

    NONE = 0
    SUBSTITUTION = 10
    EQUATION = 20
    OR = 30
    AND = 40
    NOT = 45
    COMPARISON = 50
    SUM = 60
    PRODUCT = 70
    SIGN = 80
    POWER = 90


_COMPARISONS: dict[TokenKind, Literal["<", ">", "<=", ">=", "==", "!="]] = {
    K.LESS: "<",
    K.GREATER: ">",
    K.LESS_EQ: "<=",
    K.GREATER_EQ: ">=",
    K.EQ_EQ: "==",
    K.NOT_EQ: "!=",
}
_SEPARATORS = frozenset({K.NEWLINE, K.SEMICOLON, K.ROW_SEP})
_KEYWORDS = frozenset({"and", "or", "not"})
_INTEGRALS = {"int": 1, "oint": 1, "iint": 2, "iiint": 3}
# Commands that end the argument of a function written without parentheses.
_ARGUMENT_STOPPERS = frozenset(vocab.FUNCTION_COMMANDS) | {
    "partial", "nabla", "int", "iint", "iiint", "oint", "sum", "prod", "lim",
}  # fmt: skip
_CASE_WORDS = frozenset({"if", "for", "when", "where"})
_OTHERWISE = frozenset({"otherwise", "else"})
# A name that reads as a differential inside an integral or a derivative: dx, dtheta, dx1.
_DIFFERENTIAL = re.compile(
    r"d(?:" + "|".join(sorted(vocab.GREEK, key=len, reverse=True)) + r"|[A-Za-z][0-9]*)"
)
_KNOWN_COMMANDS = sorted(
    {"frac", "dfrac", "tfrac", "cfrac", "sqrt", "sum", "prod", "lim", "partial", "nabla"}
    | {"pdv", "dv", "binom", "dbinom", "tbinom", "lfloor", "lceil", "begin", "end"}
    | set(_INTEGRALS)
    | set(vocab.FUNCTION_COMMANDS)
    | set(vocab.SYMBOL_COMMANDS)
    | set(vocab.OPERATOR_COMMANDS)
    | set(vocab.ACCENT_COMMANDS)
)


def parse(source: str | Source) -> ast.Program:
    """Parse a program: a sequence of statements separated by line breaks, ``;`` or ``\\\\``."""
    if isinstance(source, str):
        source = Source(source)
    return Parser(tokenize(source), source).parse_program()


def parse_expression(source: str | Source) -> ast.Expr:
    """Parse text that must consist of exactly one expression."""
    if isinstance(source, str):
        source = Source(source)
    parser = Parser(tokenize(source), source)
    parser.skip_separators()
    expr = parser.expression()
    parser.skip_separators()
    parser.expect_end()
    return expr


def split_index_labels(name: str) -> list[str]:
    """Split a run of letters into index labels the way LaTeX reads them.

    Every letter is its own index, except that Greek names stay whole::

        >>> split_index_labels("ab")
        ['a', 'b']
        >>> split_index_labels("munu")
        ['mu', 'nu']
        >>> split_index_labels("theta")
        ['theta']

    Names containing digits or underscores (``a1``, ``x_0``) are one label.
    """
    if not name.isalpha() or name in vocab.GREEK:
        return [name]
    labels: list[str] = []
    i = 0
    while i < len(name):
        for length in range(min(7, len(name) - i), 1, -1):
            if name[i : i + length] in vocab.GREEK:
                labels.append(name[i : i + length])
                i += length
                break
        else:
            labels.append(name[i])
            i += 1
    return labels


class Parser:
    """Parses one token list. Use :func:`parse` or :func:`parse_expression`."""

    def __init__(self, tokens: list[Token], source: Source) -> None:
        self.tokens = tokens
        self.source = source
        self.pos = 0
        self.previous = tokens[0]
        # Inside brackets and environments line breaks are insignificant.
        self._nesting = 0
        # Inside |...| a bar closes the absolute value.
        self._abs_depth = 0
        # Inside an integrand a differential (dx) ends the expression.
        self._integral_depth = 0

    # -- cursor ---------------------------------------------------------------

    def _significant(self, index: int) -> int:
        if self._nesting:
            while self.tokens[index].kind is K.NEWLINE:
                index += 1
        return index

    def peek(self, offset: int = 0) -> Token:
        index = self._significant(self.pos)
        for _ in range(offset):
            if self.tokens[index].kind is K.EOF:
                break
            index = self._significant(index + 1)
        return self.tokens[index]

    def advance(self) -> Token:
        self.pos = self._significant(self.pos)
        token = self.tokens[self.pos]
        if token.kind is not K.EOF:
            self.pos += 1
        self.previous = token
        return token

    def check(self, kind: TokenKind, value: str | None = None) -> bool:
        token = self.peek()
        return token.kind is kind and (value is None or token.value == value)

    def accept(self, kind: TokenKind, value: str | None = None) -> Token | None:
        return self.advance() if self.check(kind, value) else None

    def expect(self, kind: TokenKind, *, opener: Token | None = None, after: str = "") -> Token:
        token = self.peek()
        if token.kind is kind:
            return self.advance()
        expected = kind.value if kind in (K.EOF, K.NEWLINE) else f"'{kind.value}'"
        message = f"expected {expected}"
        if opener is not None:
            message += f" to close {opener.describe()} from line {opener.span.line}"
            message += f", column {opener.span.column}"
        if after:
            message += f" after {after}"
        raise ParseError(f"{message}, found {token.describe()}", span=token.span)

    def expect_end(self) -> None:
        token = self.peek()
        if token.kind is not K.EOF:
            raise ParseError(f"unexpected {token.describe()}", span=token.span)

    def skip_separators(self) -> None:
        while self.peek().kind in _SEPARATORS:
            self.advance()

    def _skip_newlines(self) -> None:
        while self.tokens[self.pos].kind is K.NEWLINE:
            self.pos += 1

    def _span(self, start: Token | Span) -> Span:
        """The span from ``start`` to the last token consumed."""
        span = start if isinstance(start, Span) else start.span
        return span.to(self.previous.span) if self.previous.span.end > span.start else span

    @contextmanager
    def _nested(self, nested: bool = True) -> Iterator[None]:
        """Enter brackets (line breaks insignificant) or, with ``nested=False``, a block."""
        saved = (self._nesting, self._abs_depth, self._integral_depth)
        self._nesting = saved[0] + 1 if nested else 0
        self._abs_depth = self._integral_depth = 0
        try:
            yield
        finally:
            self._nesting, self._abs_depth, self._integral_depth = saved

    def _attempt(self, parse: Callable[[], _T]) -> tuple[_T, None] | tuple[None, ParseError]:
        """Run ``parse``; if it fails, rewind and return the error instead."""
        saved = (self.pos, self.previous, self._nesting, self._abs_depth, self._integral_depth)
        try:
            return parse(), None
        except ParseError as error:
            self.pos, self.previous, self._nesting, self._abs_depth, self._integral_depth = saved
            return None, error

    # -- statements -----------------------------------------------------------

    def parse_program(self) -> ast.Program:
        first = self.peek()
        statements: list[ast.Statement] = []
        self.skip_separators()
        while not self.check(K.EOF):
            statements.extend(self.statement())
            if not self.check(K.EOF):
                self._expect_separator()
            self.skip_separators()
        span = first.span.to(self.peek().span)
        return ast.Program(tuple(statements), span=span)

    def _expect_separator(self) -> None:
        token = self.peek()
        if token.kind not in _SEPARATORS:
            raise ParseError(
                f"unexpected {token.describe()}",
                span=token.span,
                hint="put each statement on its own line, or separate statements with ';'",
            )

    def statement(self) -> list[ast.Statement]:
        token = self.peek()
        if token.kind is K.BEGIN and token.value in vocab.ALIGNMENT_ENVIRONMENTS:
            return self._alignment()
        start = self._significant(self.pos)
        if token.kind is K.IDENT and self.peek(1).kind is K.COLON and self.peek(2).kind is K.LPAREN:
            statement: ast.Statement = self._arrow_function()
        else:
            statement = self._expression_statement()
        latex = any(_is_latex(t) for t in self.tokens[start : self.pos])
        return [replace(statement, latex=latex)]

    def _expression_statement(self) -> ast.Statement:
        first = self.expression()
        targets = [first]
        while self.accept(K.COMMA):
            targets.append(self.expression())
        if define := self.accept(K.DEFINE):
            self._skip_newlines()
            value = self.expression()
            return self._definition(targets, value, define)
        if len(targets) > 1:
            raise ParseError(
                "expected ':=' after a list of names",
                span=self._span(first.span),
                hint="to define several names at once, write  x, y := ...",
            )
        return ast.Evaluate(first, span=first.span)

    def _definition(self, targets: list[ast.Expr], value: ast.Expr, define: Token) -> ast.Statement:
        span = targets[0].span.to(value.span)
        if len(targets) == 1 and isinstance(call := targets[0], ast.Call):
            if not isinstance(call.func, ast.Name):
                raise ParseError("cannot define this", span=call.func.span)
            params = []
            for arg in call.args:
                if not isinstance(arg, ast.Name):
                    raise ParseError("function parameters must be names", span=arg.span)
                params.append(arg.id)
            return ast.FunctionDefinition(call.func.id, tuple(params), value, span=span)
        for target in targets:
            if not isinstance(target, ast.Name | ast.Scripted):
                raise ParseError(
                    f"cannot define {_describe_node(target)}",
                    span=target.span,
                    hint="the left of := must be a name (x), a tensor (T^{a}_{b}) "
                    "or a function (f(x))",
                )
        return ast.Define(tuple(targets), value, span=span)

    def _arrow_function(self) -> ast.FunctionDefinition:
        name = self.advance()
        self.expect(K.COLON)
        opener = self.expect(K.LPAREN)
        params: list[str] = []
        with self._nested():
            if not self.check(K.RPAREN):
                params.append(self.expect(K.IDENT).value)
                while self.accept(K.COMMA):
                    params.append(self.expect(K.IDENT).value)
            self.expect(K.RPAREN, opener=opener)
        self.expect(K.ARROW, after="the parameters")
        body: ast.Expr | ast.Block = self._block() if self.check(K.LBRACE) else self.expression()
        return ast.FunctionDefinition(name.value, tuple(params), body, span=self._span(name))

    def _block(self) -> ast.Block:
        opener = self.expect(K.LBRACE)
        statements: list[ast.Statement] = []
        with self._nested(False):
            self.skip_separators()
            while not self.check(K.RBRACE):
                if self.check(K.EOF):
                    self.expect(K.RBRACE, opener=opener)
                statements.extend(self.statement())
                if self.check(K.EOF):
                    self.expect(K.RBRACE, opener=opener)
                if not self.check(K.RBRACE):
                    self._expect_separator()
                self.skip_separators()
            self.advance()
        if not statements:
            raise ParseError("a function body cannot be empty", span=self._span(opener))
        return ast.Block(tuple(statements), span=self._span(opener))

    def _alignment(self) -> list[ast.Statement]:
        """``\\begin{align} ... \\end{align}``: statements separated by ``\\\\``."""
        self.advance()
        statements: list[ast.Statement] = []
        with self._nested():
            while True:
                while self.peek().kind in (K.ROW_SEP, K.SEMICOLON):
                    self.advance()
                if self.check(K.END):
                    break
                statements.extend(self.statement())
                if not self.check(K.END):
                    self._expect_separator()
            self.advance()
        return statements

    # -- expressions ----------------------------------------------------------

    def expression(self, min_power: int = Power.NONE) -> ast.Expr:
        left = self._prefix()
        while True:
            token = self.peek()
            power = self._infix_power(token)
            if power <= min_power:
                return left
            left = self._infix(left, token, power)

    def _infix_power(self, token: Token) -> int:
        kind = token.kind
        if kind is K.BAR:
            if self._abs_depth:
                return Power.NONE  # it closes the absolute value
            if self.peek(1).kind is K.UNDERSCORE:
                return Power.SUBSTITUTION
            return Power.PRODUCT  # 2|x|
        if kind is K.EQUAL:
            return Power.EQUATION
        if kind is K.IDENT and token.value == "or":
            return Power.OR
        if kind is K.IDENT and token.value == "and":
            return Power.AND
        if kind in _COMPARISONS:
            return Power.COMPARISON
        if kind in (K.PLUS, K.MINUS):
            return Power.SUM
        if kind in (K.STAR, K.SLASH):
            return Power.PRODUCT
        if kind is K.DOUBLE_STAR:
            return Power.POWER
        if self._starts_operand(token):
            return Power.PRODUCT  # juxtaposition
        return Power.NONE

    def _infix(self, left: ast.Expr, token: Token, power: int) -> ast.Expr:
        kind = token.kind
        if kind is K.BAR and power == Power.SUBSTITUTION:
            return self._substitution(left)
        if power == Power.PRODUCT and kind not in (K.STAR, K.SLASH):
            right = self.expression(Power.PRODUCT)
            return ast.Binary("*", left, right, implicit=True, span=left.span.to(right.span))
        self.advance()
        self._skip_newlines()
        if kind is K.EQUAL:
            right = self.expression(Power.EQUATION)
            if self.check(K.EQUAL):
                raise ParseError(
                    "an equation can have only one '='",
                    span=self.peek().span,
                    hint="write each equation as its own statement",
                )
            return ast.Equation(left, right, span=left.span.to(right.span))
        if kind in _COMPARISONS:
            operands, ops = [left], [_COMPARISONS[kind]]
            operands.append(self.expression(Power.COMPARISON))
            while self.peek().kind in _COMPARISONS:
                ops.append(_COMPARISONS[self.advance().kind])
                self._skip_newlines()
                operands.append(self.expression(Power.COMPARISON))
            span = left.span.to(operands[-1].span)
            return ast.Comparison(tuple(operands), tuple(ops), span=span)
        if kind is K.DOUBLE_STAR:
            right = self.expression(Power.POWER - 1)  # right-associative
            return ast.Binary("**", left, right, span=left.span.to(right.span))
        right = self.expression(power)
        op = cast(Literal["+", "-", "*", "/", "and", "or"], token.value)
        if op == "/":
            return _quotient(left, right, left.span.to(right.span), leibniz=False)
        return ast.Binary(op, left, right, span=left.span.to(right.span))

    def _starts_operand(self, token: Token) -> bool:
        """Whether ``token`` can begin an operand, making juxtaposition a product."""
        kind = token.kind
        if self._integral_depth and self._at_differential():
            return False
        if kind in (K.NUMBER, K.LPAREN, K.LBRACKET, K.LBRACE):
            return True
        if kind is K.IDENT:
            return token.value not in _KEYWORDS
        if kind is K.COMMAND:
            return token.value not in ("rfloor", "rceil")
        if kind is K.BEGIN:
            return token.value in vocab.MATRIX_ENVIRONMENTS | vocab.CASES_ENVIRONMENTS
        if kind is K.BAR:
            return not self._abs_depth and self.peek(1).kind is not K.UNDERSCORE
        return False

    def _prefix(self) -> ast.Expr:
        token = self.peek()
        kind = token.kind
        if kind in (K.MINUS, K.PLUS):
            self.advance()
            operand = self.expression(Power.SIGN)
            op: Literal["-", "+"] = "-" if kind is K.MINUS else "+"
            return ast.Unary(op, operand, span=token.span.to(operand.span))
        if kind is K.IDENT and token.value == "not":
            self.advance()
            operand = self.expression(Power.NOT)
            return ast.Unary("not", operand, span=token.span.to(operand.span))
        if kind is K.STRING:
            self.advance()
            return ast.String(token.value, span=token.span)
        return self._postfix(self._atom(), grouped=kind in (K.LPAREN, K.LBRACE))

    def _atom(self) -> ast.Expr:
        token = self.peek()
        kind = token.kind
        if kind is K.NUMBER:
            self.advance()
            return ast.Number(token.value, span=token.span)
        if kind is K.IDENT:
            if token.value == "d" and self._at_plain_differential():
                return self._plain_differential()
            self.advance()
            return ast.Name(token.value, token.verbatim, span=token.span)
        if kind is K.COMMAND:
            return self._command()
        if kind is K.BEGIN:
            return self._environment()
        if kind is K.LPAREN:
            return self._parenthesized()
        if kind is K.LBRACKET:
            return self._list()
        if kind is K.LBRACE:
            return self._braced()
        if kind is K.BAR:
            return self._absolute_value()
        raise ParseError(f"expected an expression, found {token.describe()}", span=token.span)

    def _postfix(self, node: ast.Expr, *, grouped: bool = False) -> ast.Expr:
        """Scripts, calls, primes and factorials. A ``grouped`` node (from brackets)
        starts new scripts rather than extending its own: ``(x^2)^3`` is not ``x^{2^3}``."""
        while True:
            token = self.peek()
            kind = token.kind
            if kind in (K.CARET, K.UNDERSCORE):
                node = self._attach_script(node, extend=not grouped)
                grouped = False
            elif (
                kind is K.LBRACE
                and isinstance(node, ast.Scripted)
                and self.peek(1).kind is K.RBRACE
            ):
                self.advance()  # R^{a}{}_{bcd}: the empty group only spaces the indices
                self.advance()
            elif (
                kind is K.LPAREN
                and not token.space_before
                and isinstance(node, ast.Name | ast.Prime)
            ):
                node = self._call(node)
            elif kind is K.PRIME and isinstance(node, ast.Name):
                order = 0
                while self.accept(K.PRIME):
                    order += 1
                node = ast.Prime(node, order, span=self._span(node.span))
            elif kind is K.BANG:
                self.advance()
                node = ast.Factorial(node, span=self._span(node.span))
            else:
                return node

    def _call(self, func: ast.Expr) -> ast.Call:
        opener = self.advance()
        args: list[ast.Expr] = []
        with self._nested():
            if not self.check(K.RPAREN):
                args.append(self.expression())
                while self.accept(K.COMMA):
                    if self.check(K.RPAREN):
                        break
                    args.append(self.expression())
            self.expect(K.RPAREN, opener=opener)
        return ast.Call(func, tuple(args), span=self._span(func.span))

    def _parenthesized(self) -> ast.Expr:
        opener = self.advance()
        with self._nested():
            if self.accept(K.RPAREN):
                return ast.Tuple((), span=self._span(opener))
            first = self.expression()
            if not self.check(K.COMMA):
                self.expect(K.RPAREN, opener=opener)
                return first
            items = [first]
            while self.accept(K.COMMA):
                if self.check(K.RPAREN):
                    break
                items.append(self.expression())
            self.expect(K.RPAREN, opener=opener)
        return ast.Tuple(tuple(items), span=self._span(opener))

    def _list(self) -> ast.List:
        opener = self.advance()
        items: list[ast.Expr] = []
        with self._nested():
            if not self.check(K.RBRACKET):
                items.append(self.expression())
                while self.accept(K.COMMA):
                    if self.check(K.RBRACKET):
                        break
                    items.append(self.expression())
            self.expect(K.RBRACKET, opener=opener)
        return ast.List(tuple(items), span=self._span(opener))

    def _braced(self) -> ast.Expr:
        """``{expr}`` groups; ``{ x**2, x > 0; -x, otherwise }`` is a piecewise function."""
        opener = self.advance()
        with self._nested():
            first = self.expression()
            if not self.accept(K.COMMA):
                self.expect(K.RBRACE, opener=opener)
                return first
            pieces = [(first, self._condition())]
            while self.accept(K.SEMICOLON):
                if self.check(K.RBRACE):
                    break
                value = self.expression()
                self.expect(K.COMMA, after="a piece of a piecewise function")
                pieces.append((value, self._condition()))
            self.expect(K.RBRACE, opener=opener)
        return ast.Piecewise(tuple(pieces), span=self._span(opener))

    def _condition(self) -> ast.Expr | None:
        """A piecewise condition, optionally introduced by a word: ``\\text{if } x > 0``."""
        token = self.peek()
        if token.kind in (K.IDENT, K.STRING) and token.value.strip().lower() in _OTHERWISE:
            self.advance()
            return None
        if token.kind in (K.IDENT, K.STRING) and token.value.strip().lower() in _CASE_WORDS:
            self.advance()
        return self.expression(Power.SUBSTITUTION)

    def _absolute_value(self) -> ast.Abs:
        opener = self.advance()
        with self._nested():
            self._abs_depth = 1
            operand = self.expression()
            self.expect(K.BAR, opener=opener)
        return ast.Abs(operand, span=self._span(opener))

    def _substitution(self, expr: ast.Expr) -> ast.Substitution:
        self.advance()  # the bar
        self.expect(K.UNDERSCORE)
        opener = self.expect(K.LBRACE, after="the evaluation bar '|_'")
        bindings: list[tuple[ast.Expr, ast.Expr]] = []
        with self._nested():
            while True:
                binding = self.expression()
                if not isinstance(binding, ast.Equation):
                    raise ParseError("expected a substitution such as x = 2", span=binding.span)
                bindings.append((binding.lhs, binding.rhs))
                if not self.accept(K.COMMA):
                    break
            self.expect(K.RBRACE, opener=opener)
        return ast.Substitution(expr, tuple(bindings), span=self._span(expr.span))

    # -- scripts and indices ----------------------------------------------------

    def _attach_script(self, node: ast.Expr, *, extend: bool) -> ast.Scripted:
        marker = self.advance()
        kind: Literal["^", "_"] = "^" if marker.kind is K.CARET else "_"
        script = self._script(kind, marker)
        if extend and isinstance(node, ast.Scripted):
            base, scripts = node.base, (*node.scripts, script)
        else:
            base, scripts = node, (script,)
        return ast.Scripted(base, scripts, span=base.span.to(script.span))

    def _script(self, kind: Literal["^", "_"], marker: Token) -> ast.Script:
        token = self.peek()
        if token.kind is K.LBRACE:
            return self._braced_script(kind, marker)
        if token.kind is K.IDENT:
            self.advance()
            name = ast.Name(token.value, token.verbatim, span=token.span)
            labels = [token.value] if token.verbatim else split_index_labels(token.value)
            items = self._items_from_labels(labels, token)
            indices = ast.IndexList(items, span=token.span)
            return ast.Script(kind, name, indices, span=marker.span.to(token.span))
        if token.kind is K.NUMBER:
            self.advance()
            number = ast.Number(token.value, span=token.span)
            return ast.Script(
                kind, number, self._numeric_indices(token), span=marker.span.to(token.span)
            )
        if token.kind in (K.MINUS, K.PLUS):
            self.advance()
            operand = self._atom()
            op: Literal["-", "+"] = "-" if token.kind is K.MINUS else "+"
            signed = ast.Unary(op, operand, span=token.span.to(operand.span))
            return ast.Script(kind, signed, None, span=marker.span.to(signed.span))
        if token.kind in (K.LPAREN, K.COMMAND):
            atom = self._atom()
            return ast.Script(kind, atom, None, span=marker.span.to(atom.span))
        what = "superscript" if kind == "^" else "subscript"
        raise ParseError(
            f"expected a {what} after '{kind}', found {token.describe()}", span=token.span
        )

    def _braced_script(self, kind: Literal["^", "_"], marker: Token) -> ast.Script:
        """A braced script, read both as an index list and as an expression."""
        opener = self.advance()
        if self.check(K.RBRACE):
            self.advance()
            empty = ast.IndexList((), span=self._span(opener))
            return ast.Script(kind, None, empty, span=self._span(marker))
        start = (self.pos, self.previous)
        indices, index_error = self._attempt(lambda: self._index_list(opener))
        after_indices = (self.pos, self.previous)
        self.pos, self.previous = start

        def braced_expression() -> ast.Expr:
            with self._nested():
                body = self.expression()
                self.expect(K.RBRACE, opener=opener)
            return body

        body, body_error = self._attempt(braced_expression)
        if body is None:
            if indices is None:
                assert index_error is not None
                assert body_error is not None
                # A subscript is almost always an index list, so it reports the index
                # reading's error; a superscript reports whichever reading got further.
                index_end = index_error.span.start if index_error.span else -1
                body_end = body_error.span.start if body_error.span else -1
                raise body_error if kind == "^" and body_end > index_end else index_error
            self.pos, self.previous = after_indices
        return ast.Script(kind, body, indices, index_error, span=self._span(marker))

    def _index_list(self, opener: Token) -> ast.IndexList:
        items: list[ast.IndexItem] = []
        derivative: Literal[",", ";"] | None = None
        pending: Token | None = None  # an opening bracket waiting for its first index
        inside = False  # inside a bracket opened in this index list
        with self._nested():
            while not self.check(K.RBRACE):
                token = self.peek()
                if token.kind in (K.COMMA, K.SEMICOLON):
                    self.advance()  # T_{a,b}: partial derivative; T_{a;b}: covariant derivative
                    derivative = "," if token.kind is K.COMMA else ";"
                elif token.kind in (K.LPAREN, K.LBRACKET):
                    if pending is not None or inside:
                        message = "(anti)symmetrisation brackets cannot be nested"
                        raise ParseError(message, span=token.span)
                    pending = self.advance()
                elif token.kind in (K.RPAREN, K.RBRACKET):
                    closer = self.advance()
                    if pending is not None or not items or items[-1].closes is not None:
                        raise ParseError("brackets must enclose indices", span=closer.span)
                    closes: Literal[")", "]"] = ")" if closer.kind is K.RPAREN else "]"
                    items[-1] = replace(items[-1], closes=closes)
                    inside = False
                else:
                    new = self._index_items(derivative)
                    if pending is not None:
                        opens: Literal["(", "["] = "(" if pending.kind is K.LPAREN else "["
                        new[0] = replace(new[0], opens=opens)
                        pending, inside = None, True
                    items.extend(new)
            if pending is not None:
                raise ParseError("brackets must enclose indices", span=pending.span)
            self.expect(K.RBRACE, opener=opener)
        _check_brackets(items)
        return ast.IndexList(tuple(items), span=self._span(opener))

    def _index_items(self, derivative: Literal[",", ";"] | None) -> list[ast.IndexItem]:
        token = self.peek()
        if token.kind is K.NUMBER:
            numeric = self._numeric_indices(token)
            if numeric is None:
                raise ParseError("an index value must be a whole number", span=token.span)
            self.advance()
            return [replace(item, derivative=derivative) for item in numeric.items]
        if token.kind is not K.IDENT or token.value in _KEYWORDS:
            raise ParseError(f"expected an index, found {token.describe()}", span=token.span)
        self.advance()
        labels = [token.value] if token.verbatim else split_index_labels(token.value)
        items = list(self._items_from_labels(labels, token))
        if self.peek().kind in (K.EQUAL, K.COLON):
            self.advance()
            items[-1] = replace(items[-1], value=self._index_value())
        return [replace(item, derivative=derivative) for item in items]

    def _index_value(self) -> ast.Expr:
        token = self.peek()
        if token.kind is K.NUMBER and token.value.isdigit():
            self.advance()
            return ast.Number(token.value, span=token.span)
        if token.kind is K.IDENT:
            self.advance()
            return ast.Name(token.value, token.verbatim, span=token.span)
        raise ParseError(
            f"expected an index value (a whole number or a coordinate), found {token.describe()}",
            span=token.span,
        )

    def _items_from_labels(self, labels: list[str], token: Token) -> tuple[ast.IndexItem, ...]:
        if len(labels) == 1 or token.span.text != token.value:
            return tuple(ast.IndexItem(label, span=token.span) for label in labels)
        items, offset = [], token.span.start
        for label in labels:
            span = Span(self.source, offset, offset + len(label))
            items.append(ast.IndexItem(label, span=span))
            offset += len(label)
        return tuple(items)

    def _numeric_indices(self, token: Token) -> ast.IndexList | None:
        """``01`` as indices is the components 0 and 1, as in ``g_{01}``."""
        if not token.value.isdigit():
            return None
        start = token.span.start
        items = tuple(
            ast.IndexItem(
                None,
                ast.Number(digit, span=Span(self.source, start + i, start + i + 1)),
                span=Span(self.source, start + i, start + i + 1),
            )
            for i, digit in enumerate(token.value)
        )
        return ast.IndexList(items, span=token.span)

    # -- LaTeX commands ---------------------------------------------------------

    def _command(self) -> ast.Expr:
        token = self.peek()
        name = token.value
        if name in ("frac", "dfrac", "tfrac", "cfrac"):
            return self._fraction()
        if name == "sqrt":
            return self._root()
        if name in ("sum", "prod"):
            return self._big_operator()
        if name in _INTEGRALS:
            return self._integral()
        if name == "lim":
            return self._limit()
        if name in ("partial", "nabla"):
            self.advance()
            return ast.Operator(cast(Literal["partial", "nabla"], name), span=token.span)
        if name in ("pdv", "dv"):
            return self._physics_derivative()
        if name in ("binom", "dbinom", "tbinom"):
            self.advance()
            n = self._brace_argument(token)
            k = self._brace_argument(token)
            func = ast.Name("binomial", span=token.span)
            return ast.Call(func, (n, k), span=self._span(token))
        if name in ("lfloor", "lceil"):
            return self._floor_or_ceiling()
        if name in vocab.FUNCTION_COMMANDS:
            return self._function()
        if name in vocab.ACCENT_COMMANDS:
            return self._accent()
        close = difflib.get_close_matches(name, _KNOWN_COMMANDS, n=1)
        raise ParseError(
            f"unknown LaTeX command \\{name}",
            span=token.span,
            hint=f"did you mean \\{close[0]}?" if close else None,
        )

    def _argument_tokens(self, command: Token) -> list[Token]:
        """The raw tokens of a command argument: a braced group, a parenthesised
        group, or (as LaTeX reads it) one character: ``\\frac12`` is one half."""
        self._skip_newlines()
        token = self.tokens[self.pos]
        if token.kind is K.LBRACE:
            return self._raw_group(K.LBRACE, K.RBRACE, command)[1:-1]
        if token.kind is K.LPAREN:
            return self._raw_group(K.LPAREN, K.RPAREN, command)
        if token.kind in (K.NUMBER, K.IDENT) and len(token.value) > 1 and not _is_latex(token):
            first, rest = _split_first_character(token, self.source)
            self.tokens[self.pos] = rest
            self.previous = first
            return [first]
        if token.kind in (K.NUMBER, K.IDENT):
            return [self.advance()]
        raise ParseError(f"\\{command.value} is missing an argument", span=token.span)

    def _raw_group(self, open_kind: TokenKind, close_kind: TokenKind, owner: Token) -> list[Token]:
        """Consume a balanced group, returning its tokens including the brackets."""
        opener = self.tokens[self.pos]
        depth, collected = 0, []
        while True:
            token = self.tokens[self.pos]
            if token.kind is K.EOF:
                raise ParseError(
                    f"expected '{close_kind.value}' to close {opener.describe()} "
                    f"in the argument of \\{owner.value}",
                    span=opener.span,
                )
            self.pos += 1
            collected.append(token)
            if token.kind is open_kind:
                depth += 1
            elif token.kind is close_kind:
                depth -= 1
                if depth == 0:
                    self.previous = token
                    return collected

    def _subparse(self, tokens: list[Token], what: str, owner: Token) -> ast.Expr:
        """Parse a token list captured by :meth:`_argument_tokens` as one expression."""
        tokens = [t for t in tokens if t.kind is not K.NEWLINE]
        if not tokens:
            raise ParseError(f"the {what} of \\{owner.value} is empty", span=owner.span)
        end = Span(self.source, tokens[-1].span.end, tokens[-1].span.end)
        parser = Parser([*tokens, Token(K.EOF, "", end)], self.source)
        parser._nesting = 1
        expr = parser.expression()
        if not parser.check(K.EOF):
            token = parser.peek()
            raise ParseError(f"unexpected {token.describe()} in the {what}", span=token.span)
        return expr

    def _brace_argument(self, command: Token) -> ast.Expr:
        opener = self.expect(K.LBRACE, after=f"\\{command.value}")
        with self._nested():
            expr = self.expression()
            self.expect(K.RBRACE, opener=opener)
        return expr

    def _fraction(self) -> ast.Expr:
        command = self.advance()
        numerator = self._subparse(self._argument_tokens(command), "numerator", command)
        denominator = self._subparse(self._argument_tokens(command), "denominator", command)
        return _quotient(numerator, denominator, self._span(command), leibniz=True)

    def _accent(self) -> ast.Accent:
        command = self.advance()
        base = self._subparse(self._argument_tokens(command), "argument", command)
        return ast.Accent(vocab.ACCENT_COMMANDS[command.value], base, span=self._span(command))

    def _root(self) -> ast.Root:
        command = self.advance()
        degree = None
        if self.tokens[self.pos].kind is K.LBRACKET:
            group = self._raw_group(K.LBRACKET, K.RBRACKET, command)[1:-1]
            degree = self._subparse(group, "degree", command)
        radicand = self._subparse(self._argument_tokens(command), "radicand", command)
        return ast.Root(radicand, degree, span=self._span(command))

    def _limits(self, command: Token) -> tuple[list[Token] | None, list[Token] | None]:
        """The ``_lower`` and ``^upper`` of a big operator, in either order."""
        lower = upper = None
        for _ in range(2):
            if self.tokens[self.pos].kind is K.UNDERSCORE and lower is None:
                self.pos += 1
                lower = self._script_tokens(command)
            elif self.tokens[self.pos].kind is K.CARET and upper is None:
                self.pos += 1
                upper = self._script_tokens(command)
        return lower, upper

    def _script_tokens(self, command: Token) -> list[Token]:
        token = self.tokens[self.pos]
        if token.kind is K.LBRACE:
            return self._raw_group(K.LBRACE, K.RBRACE, command)[1:-1]
        if token.kind is K.MINUS:
            self.pos += 1
            return [token, self.advance()]
        if token.kind in (K.NUMBER, K.IDENT, K.COMMAND):
            return [self.advance()]
        raise ParseError(f"expected a limit after \\{command.value}", span=token.span)

    def _big_operator(self) -> ast.BigOperator:
        command = self.advance()
        op: Literal["sum", "prod"] = "sum" if command.value == "sum" else "prod"
        lower, upper = self._limits(command)
        if lower is None or upper is None:
            raise ParseError(
                f"\\{op} needs limits",
                span=command.span,
                hint=f"for example \\{op}_{{n=0}}^{{N}} a_n",
            )
        start = self._subparse(lower, "lower limit", command)
        if not isinstance(start, ast.Equation) or not isinstance(start.lhs, ast.Name):
            raise ParseError(
                f"the lower limit of \\{op} must name its variable, as in n=0",
                span=start.span,
            )
        end = self._subparse(upper, "upper limit", command)
        body = self.expression(Power.SUM)
        span = self._span(command)
        return ast.BigOperator(op, start.lhs, start.rhs, end, body, span=span)

    def _integral(self) -> ast.Integral:
        command = self.advance()
        count = _INTEGRALS[command.value]
        lower_tokens, upper_tokens = self._limits(command)
        lower = self._subparse(lower_tokens, "lower limit", command) if lower_tokens else None
        upper = self._subparse(upper_tokens, "upper limit", command) if upper_tokens else None
        self._integral_depth += 1
        try:
            variables: list[ast.Expr] = []
            if self._at_differential():  # \int dx f(x)
                while len(variables) < count and self._at_differential():
                    variables.append(self._differential())
                integrand = self.expression(Power.SUM)
            else:
                integrand = self.expression(Power.COMPARISON)
            while len(variables) < count and self._at_differential():
                variables.append(self._differential())
            if len(variables) < count:  # \int \frac{dr}{1 - 2M/r}: the differential is on top
                integrand, found = _take_differentials(integrand, count - len(variables))
                variables.extend(found)
            if len(variables) < count:
                token = self.peek()
                raise ParseError(
                    f"expected a differential such as dx to end the integral, "
                    f"found {token.describe()}",
                    span=token.span,
                )
        finally:
            self._integral_depth -= 1
        bounds = tuple(
            ast.Bound(variable, lower, upper, span=variable.span) for variable in variables
        )
        return ast.Integral(integrand, bounds, span=self._span(command))

    def _at_differential(self) -> bool:
        """Is the next token a differential: ``dx``, ``d x``, ``d\\theta``, ``\\mathrm{d}x``?"""
        token = self.peek()
        if token.kind is not K.IDENT:
            return False
        if token.value == "d":
            follower = self.peek(1)
            return follower.kind is K.IDENT or (
                follower.kind is K.LBRACE and not follower.space_before
            )
        return not token.verbatim and _DIFFERENTIAL.fullmatch(token.value) is not None

    def _differential(self) -> ast.Expr:
        token = self.advance()
        if token.value != "d":
            span = Span(self.source, token.span.start + 1, token.span.end)
            return ast.Name(token.value[1:], span=span)
        if self.check(K.LBRACE):
            opener = self.advance()
            with self._nested():
                variable = self.expression()
                self.expect(K.RBRACE, opener=opener)
            return variable
        name = self.advance()
        return ast.Name(name.value, name.verbatim, span=name.span)

    def _limit(self) -> ast.Limit:
        command = self.advance()
        if self.tokens[self.pos].kind is not K.UNDERSCORE:
            raise ParseError(
                "\\lim needs a subscript", span=command.span, hint="for example \\lim_{x \\to 0}"
            )
        self.pos += 1
        tokens = [t for t in self._script_tokens(command) if t.kind is not K.NEWLINE]
        arrows = [i for i, t in enumerate(tokens) if t.kind is K.ARROW]
        if len(arrows) != 1:
            raise ParseError("expected a limit such as x \\to 0", span=self._span(command))
        variable = self._subparse(tokens[: arrows[0]], "variable", command)
        point_tokens = tokens[arrows[0] + 1 :]
        direction: Literal["+", "-"] | None = None
        kinds = [t.kind for t in point_tokens]
        for suffix in ([K.CARET, K.PLUS], [K.CARET, K.MINUS]):
            for pattern in (suffix, [K.CARET, K.LBRACE, suffix[1], K.RBRACE]):
                if kinds[-len(pattern) :] == pattern and len(kinds) > len(pattern):
                    direction = "+" if suffix[1] is K.PLUS else "-"
                    point_tokens = point_tokens[: -len(pattern)]
        point = self._subparse(point_tokens, "limit point", command)
        body = self.expression(Power.SUM)
        return ast.Limit(variable, point, body, direction, span=self._span(command))

    def _physics_derivative(self) -> ast.Derivative:
        r"""The physics package: ``\pdv{f}{x}``, ``\pdv[2]{f}{x}``, ``\pdv{f}{x}{y}``,
        ``\dv{x}``."""
        command = self.advance()
        partial = command.value == "pdv"
        order = 1
        if self.tokens[self.pos].kind is K.LBRACKET:
            self.pos += 1
            order = int(self.expect(K.NUMBER, after="'['").value)
            self.expect(K.RBRACKET)
        first = self._brace_argument(command)
        variables: list[ast.Expr] = []
        while self.tokens[self.pos].kind is K.LBRACE:
            variables.append(self._brace_argument(command))
        if not variables:  # \pdv{x} alone is the operator ∂/∂x
            return ast.Derivative(None, ((first, order),), partial, span=self._span(command))
        if order != 1 and len(variables) > 1:
            raise ParseError(
                "a mixed derivative cannot also have an order", span=self._span(command)
            )
        pairs = tuple((variable, order) for variable in variables)
        return ast.Derivative(first, pairs, partial, span=self._span(command))

    def _floor_or_ceiling(self) -> ast.Call:
        opener = self.advance()
        closer = "rfloor" if opener.value == "lfloor" else "rceil"
        with self._nested():
            operand = self.expression()
            if not self.check(K.COMMAND, closer):
                token = self.peek()
                raise ParseError(
                    f"expected \\{closer} to close \\{opener.value}, found {token.describe()}",
                    span=token.span,
                )
            self.advance()
        func = ast.Name("floor" if closer == "rfloor" else "ceiling", span=opener.span)
        return ast.Call(func, (operand,), span=self._span(opener))

    def _function(self) -> ast.Expr:
        r"""``\sin x``, ``\sin(x)``, ``\sin^2\theta``, ``\sin^{-1} x``, ``\log_{2} x``."""
        command = self.advance()
        name = vocab.FUNCTION_COMMANDS[command.value]
        power: ast.Expr | None = None
        base: ast.Expr | None = None
        while self.peek().kind in (K.CARET, K.UNDERSCORE):
            marker = self.advance()
            script = self._script("^" if marker.kind is K.CARET else "_", marker)
            if script.body is None:
                raise ParseError("expected an expression", span=script.span)
            if marker.kind is K.CARET:
                power = script.body
            else:
                base = script.body
        if _is_minus_one(power) and command.value in vocab.INVERSE_FUNCTIONS:
            name, power = vocab.INVERSE_FUNCTIONS[command.value], None
        args = self._function_arguments()
        if base is not None:
            node: ast.Expr = ast.Call(
                ast.Name("log", span=command.span), (*args, base), span=self._span(command)
            )
        else:
            node = ast.Call(ast.Name(name, span=command.span), args, span=self._span(command))
        if power is not None:
            script = ast.Script("^", power, None, span=power.span)
            node = ast.Scripted(node, (script,), span=self._span(command))
        return node

    def _function_arguments(self) -> tuple[ast.Expr, ...]:
        """Arguments in parentheses, or (as typeset maths reads it) the factors that
        follow, up to explicit spacing, an operator, another function or a
        differential: ``\\sin\\omega t`` is sin(ωt) but ``\\sin\\theta\\, d\\phi`` is not."""
        token = self.peek()
        if token.kind is K.LPAREN:
            opener = self.advance()
            args: list[ast.Expr] = []
            with self._nested():
                args.append(self.expression())
                while self.accept(K.COMMA):
                    args.append(self.expression())
                self.expect(K.RPAREN, opener=opener)
            return tuple(args)
        if token.kind is K.LBRACE:
            opener = self.advance()
            with self._nested():
                arg = self.expression()
                self.expect(K.RBRACE, opener=opener)
            return (arg,)
        if not self._starts_operand(token):
            raise ParseError(f"expected an argument, found {token.describe()}", span=token.span)
        arg = self.expression(Power.PRODUCT)
        while True:
            token = self.peek()
            if token.explicit_space or not self._starts_operand(token):
                break
            if token.kind is K.COMMAND and token.value in _ARGUMENT_STOPPERS:
                break
            if self._at_differential() or (token.kind is K.IDENT and token.value == "d"):
                break
            right = self.expression(Power.PRODUCT)
            arg = ast.Binary("*", arg, right, implicit=True, span=arg.span.to(right.span))
        return (arg,)

    # -- derivatives in Leibniz notation ----------------------------------------

    def _at_plain_differential(self) -> bool:
        """``d{...}``, ``d^2{...}`` or ``d^{2}{...}`` directly after the ``d``."""
        follower = self.peek(1)
        if follower.kind is K.LBRACE and not follower.space_before:
            return True
        if follower.kind is not K.CARET:
            return False
        kinds = [self.peek(i).kind for i in range(2, 6)]
        return kinds[:2] == [K.NUMBER, K.LBRACE] or kinds == [
            K.LBRACE,
            K.NUMBER,
            K.RBRACE,
            K.LBRACE,
        ]

    def _plain_differential(self) -> ast.Differential:
        start = self.advance()
        order = 1
        if self.accept(K.CARET):
            if self.accept(K.LBRACE):
                order = int(self.expect(K.NUMBER).value)
                self.expect(K.RBRACE)
            else:
                order = int(self.expect(K.NUMBER).value)
        opener = self.expect(K.LBRACE)
        with self._nested():
            expr = self.expression()
            self.expect(K.RBRACE, opener=opener)
        return ast.Differential(expr, order, span=self._span(start))

    # -- environments -----------------------------------------------------------

    def _environment(self) -> ast.Expr:
        begin = self.peek()
        environment = begin.value
        if environment in vocab.MATRIX_ENVIRONMENTS:
            return self._matrix()
        if environment in vocab.CASES_ENVIRONMENTS:
            return self._cases()
        if environment in vocab.ALIGNMENT_ENVIRONMENTS:
            raise ParseError(
                f"\\begin{{{environment}}} holds whole equations, so it cannot appear "
                "inside an expression",
                span=begin.span,
            )
        raise ParseError(f"unsupported environment {environment!r}", span=begin.span)

    def _matrix(self) -> ast.Matrix:
        begin = self.advance()
        rows: list[tuple[ast.Expr, ...]] = []
        row: list[ast.Expr] = []
        with self._nested():
            while not self.check(K.END):
                row.append(self.expression())
                if self.accept(K.AMPERSAND):
                    continue
                if self.accept(K.ROW_SEP) or self.check(K.END):
                    rows.append(tuple(row))
                    row = []
                    continue
                token = self.peek()
                raise ParseError(
                    f"expected '&', '\\\\' or \\end{{{begin.value}}}, found {token.describe()}",
                    span=token.span,
                )
            self.advance()
        if not rows:
            raise ParseError("a matrix needs at least one entry", span=self._span(begin))
        for number, entries in enumerate(rows[1:], start=2):
            if len(entries) != len(rows[0]):
                raise ParseError(
                    f"row {number} of the matrix has {len(entries)} entries "
                    f"but row 1 has {len(rows[0])}",
                    span=entries[0].span.to(entries[-1].span),
                )
        return ast.Matrix(begin.value, tuple(rows), span=self._span(begin))

    def _cases(self) -> ast.Piecewise:
        begin = self.advance()
        pieces: list[tuple[ast.Expr, ast.Expr | None]] = []
        with self._nested():
            while not self.check(K.END):
                value = self.expression()
                self.accept(K.COMMA)
                self.expect(K.AMPERSAND, after="a case's value")
                pieces.append((value, self._condition()))
                if not self.accept(K.ROW_SEP) and not self.check(K.END):
                    token = self.peek()
                    raise ParseError(
                        f"expected '\\\\' or \\end{{{begin.value}}}, found {token.describe()}",
                        span=token.span,
                    )
            self.advance()
        if not pieces:
            raise ParseError("cases needs at least one case", span=self._span(begin))
        return ast.Piecewise(tuple(pieces), span=self._span(begin))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_latex(token: Token) -> bool:
    """Whether a token was written as a LaTeX command or in Unicode mathematics."""
    text = token.span.text
    return text.startswith("\\") or not text.isascii()


def _is_minus_one(node: ast.Expr | None) -> bool:
    return (
        isinstance(node, ast.Unary)
        and node.op == "-"
        and isinstance(node.operand, ast.Number)
        and node.operand.text == "1"
    )


def _split_first_character(token: Token, source: Source) -> tuple[Token, Token]:
    start = token.span.start
    first = Token(token.kind, token.value[0], Span(source, start, start + 1), token.space_before)
    rest_kind = K.NUMBER if token.value[1].isdigit() or token.value[1] == "." else K.IDENT
    rest = Token(rest_kind, token.value[1:], Span(source, start + 1, token.span.end))
    return first, rest


def _quotient(numerator: ast.Expr, denominator: ast.Expr, span: Span, *, leibniz: bool) -> ast.Expr:
    """A quotient, which may be a derivative written in Leibniz notation.

    With ``leibniz`` (a ``\\frac``) every Leibniz form is recognised: ``\\frac{df}{dx}``,
    ``\\frac{d^2 x^\\mu}{d\\tau^2}``, ``\\frac{\\partial L}{\\partial x^\\mu}``. Otherwise
    only the braced plain-text differential is: ``d{f}/d{x}``; ``dy/dx`` stays a ratio.
    """
    if leibniz or isinstance(numerator, ast.Differential):
        top = _leibniz_numerator(numerator)
        bottom = _leibniz_denominator(denominator) if top is not None else None
        if top is not None and bottom is not None:
            (top_partial, top_order, expr), (bottom_partial, variables) = top, bottom
            if top_partial != bottom_partial:
                raise ParseError("a derivative cannot mix d and \\partial", span=span)
            total = sum(order for _, order in variables)
            if (top_order or 1) != total:
                raise ParseError(
                    f"the derivative's order is {top_order or 1} in the numerator "
                    f"but {total} in the denominator",
                    span=span,
                )
            return ast.Derivative(expr, tuple(variables), top_partial, span=span)
    return ast.Binary("/", numerator, denominator, span=span)


def _factors(node: ast.Expr) -> list[ast.Expr]:
    """The factors of a juxtaposition: ``d^2 y`` is ``[d^2, y]``."""
    if isinstance(node, ast.Binary) and node.op == "*" and node.implicit:
        return _factors(node.left) + _factors(node.right)
    return [node]


def _product(factors: list[ast.Expr]) -> ast.Expr | None:
    if not factors:
        return None
    result = factors[0]
    for factor in factors[1:]:
        result = ast.Binary("*", result, factor, implicit=True, span=result.span.to(factor.span))
    return result


def _numeric_power(node: ast.Expr) -> tuple[ast.Expr, int] | None:
    """``x^2`` → ``(x, 2)``: split a trailing whole-number superscript off a node."""
    if not isinstance(node, ast.Scripted):
        return None
    last = node.scripts[-1]
    if last.kind != "^" or not isinstance(last.body, ast.Number) or not last.body.text.isdigit():
        return None
    rest = node.scripts[:-1]
    base = (
        ast.Scripted(node.base, rest, span=node.base.span.to(rest[-1].span)) if rest else node.base
    )
    return base, int(last.body.text)


def _differential_marker(node: ast.Expr) -> tuple[bool, int | None] | None:
    """``d``, ``d^n``, ``\\partial`` or ``\\partial^n`` → ``(partial, order)``."""
    if isinstance(node, ast.Name) and node.id == "d":
        return False, None
    if isinstance(node, ast.Operator) and node.kind == "partial":
        return True, None
    if (split := _numeric_power(node)) is not None:
        inner = _differential_marker(split[0])
        if inner is not None and inner[1] is None:
            return inner[0], split[1]
    return None


def _joined_differential(node: ast.Expr) -> ast.Expr | None:
    """A name that reads as a differential: ``dx`` → ``x``, ``dx^{\\mu}`` → ``x^{\\mu}``."""
    if isinstance(node, ast.Scripted) and isinstance(node.base, ast.Name):
        inner = _joined_differential(node.base)
        return None if inner is None else ast.Scripted(inner, node.scripts, span=node.span)
    if (
        isinstance(node, ast.Name)
        and not node.verbatim
        and not node.span.text.startswith("\\")
        and _DIFFERENTIAL.fullmatch(node.id)
    ):
        span = Span(node.span.source, node.span.start + 1, node.span.end)
        return ast.Name(node.id[1:], span=span)
    return None


def _leibniz_numerator(node: ast.Expr) -> tuple[bool, int | None, ast.Expr | None] | None:
    """``d``, ``d^2 y``, ``dy``, ``\\partial^2 f``, ``d^2{f}`` → ``(partial, order, f)``."""
    head, *rest = _factors(node)
    if isinstance(head, ast.Differential):
        return False, head.order, _product([head.expr, *rest])
    if (variable := _joined_differential(head)) is not None:
        return False, None, _product([variable, *rest])
    if (marker := _differential_marker(head)) is not None:
        return marker[0], marker[1], _product(rest)
    return None


def _leibniz_denominator(node: ast.Expr) -> tuple[bool, list[tuple[ast.Expr, int]]] | None:
    """``dx^2``, ``d\\tau^2``, ``d{x}^2`` or ``\\partial x \\partial y``
    → ``(partial, variables)``."""
    factors = _factors(node)
    variables: list[tuple[ast.Expr, int]] = []
    kinds: set[bool] = set()
    i = 0
    while i < len(factors):
        factor = factors[i]
        marker = _differential_marker(factor)
        if marker is not None and marker[1] is None and i + 1 < len(factors):
            kinds.add(marker[0])
            variable = factors[i + 1]
            variables.append(_numeric_power(variable) or (variable, 1))
            i += 2
            continue
        base, order = _numeric_power(factor) or (factor, 1)
        if isinstance(base, ast.Differential) and base.order == 1:
            kinds.add(False)
            variables.append((base.expr, order))
        elif (joined := _joined_differential(factor)) is not None:
            kinds.add(False)
            variables.append(_numeric_power(joined) or (joined, 1))
        else:
            return None
        i += 1
    if not variables or len(kinds) != 1:
        return None
    return kinds.pop(), variables


def _take_differentials(integrand: ast.Expr, count: int) -> tuple[ast.Expr, list[ast.Expr]]:
    """Take up to ``count`` differentials out of the numerator of a fraction:
    ``\\frac{r^2 dr}{f(r)}`` is ``r^2 / f(r)`` integrated over ``r``."""
    if not (isinstance(integrand, ast.Binary) and integrand.op == "/"):
        return integrand, []
    kept: list[ast.Expr] = []
    variables: list[ast.Expr] = []
    for factor in _factors(integrand.left):
        joined = _joined_differential(factor)
        if joined is not None and isinstance(joined, ast.Name) and len(variables) < count:
            variables.append(joined)
        else:
            kept.append(factor)
    if not variables:
        return integrand, []
    numerator = _product(kept) or ast.Number("1", span=integrand.left.span)
    return ast.Binary("/", numerator, integrand.right, span=integrand.span), variables


def _check_brackets(items: list[ast.IndexItem]) -> None:
    """Check the (anti)symmetrisation brackets that open and close within one index list.

    A bracket may also open in one index list and close in another, as in
    ``\\nabla_{[\\lambda} R_{\\rho\\sigma]\\mu\\nu}``; the compiler pairs those.
    """
    open_item: ast.IndexItem | None = None
    for item in items:
        if item.opens is not None:
            open_item = item
        if item.closes is not None and open_item is not None:
            if {"(": ")", "[": "]"}[open_item.opens or "("] != item.closes:
                raise ParseError(
                    "mismatched brackets: '(' must close with ')' and '[' with ']'",
                    span=open_item.span.to(item.span),
                )
            if open_item is item:
                raise ParseError(
                    "only two or more indices can be (anti)symmetrised", span=item.span
                )
            open_item = None


def _describe_node(node: ast.Expr) -> str:
    names = {
        ast.Number: "a number",
        ast.String: "a string",
        ast.Binary: "an expression",
        ast.Unary: "an expression",
        ast.List: "a list",
        ast.Tuple: "a tuple",
        ast.Equation: "an equation",
    }
    return names.get(type(node), "this")
