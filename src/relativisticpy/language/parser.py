r"""Parsing: tokens in, syntax tree out.

A Pratt (precedence-climbing) parser. Every infix operator has a binding power
in :class:`Power`, from loosest to tightest::

    expr |_{x=a}         substitution
    a = b                equation
    b \pmod{m}           the modulus of a congruence, of all of b
    a or b, a and b      logic
    a < b <= c           comparison (chainable)
    a + b, a - b         sum
    a * b, a / b, a b    product (juxtaposition is multiplication), a \bmod m
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
from dataclasses import fields, replace
from enum import IntEnum
from typing import Literal, NamedTuple, TypeVar, cast

from relativisticpy.errors import ParseError
from relativisticpy.language import dirac
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
    MODULUS = 25
    OR = 30
    AND = 40
    NOT = 45
    COMPARISON = 50
    EVALUATION = 55
    SUM = 60
    CONVOLUTION = 65
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
_SIGNS: dict[TokenKind, Literal["-", "+", "±", "∓"]] = {
    K.MINUS: "-",
    K.PLUS: "+",
    K.PLUS_MINUS: "±",
    K.MINUS_PLUS: "∓",
}
_SEPARATORS = frozenset({K.NEWLINE, K.SEMICOLON, K.ROW_SEP})

#: The counts of parameters of {}_pF_q: numbers, or a letter in its general form.
_COUNT = re.compile(r"[0-9]+|[A-Za-z]")
# What may follow a statement: a comma or full stop just before one of these is prose.
_STATEMENT_ENDS = frozenset({K.EOF, K.NEWLINE, K.ROW_SEP, K.END, K.SEMICOLON, K.RBRACE})
# Equality and the relations RelativisticPy reads but does not compute; they chain.
_RELATIONS = frozenset({K.EQUAL, K.RELATION, K.ARROW, K.COLON})
_OPERATORS = frozenset(
    {K.PLUS, K.MINUS, K.STAR, K.SLASH, K.DOUBLE_STAR, K.PLUS_MINUS, K.MINUS_PLUS, K.WEDGE}
)
# A line that starts with one of these continues the statement on the line before it.
_CONTINUING = _OPERATORS | _RELATIONS | frozenset(_COMPARISONS)
# Of these, the signs that may start a statement of its own, as -x may: a line that
# starts with one continues the line before only when it is indented further.
_SIGNS_FIRST = frozenset({K.MINUS, K.PLUS_MINUS, K.MINUS_PLUS})
_KEYWORDS = frozenset({"and", "or", "not"})
# Words of prose that join equations, when written as text: \text{and}, \text{where}.
_CONNECTIVES = frozenset({"and", "or", "where", "with", "for", "if", "when", "since", "so"})
_INTEGRALS = {"int": 1, "oint": 1, "iint": 2, "iiint": 3}
# Commands that end the argument of a function written without parentheses.
_ARGUMENT_STOPPERS = frozenset(vocab.FUNCTION_COMMANDS) | {
    "partial", "nabla", "int", "iint", "iiint", "oint", "sum", "prod", "lim",
    "frac", "dfrac", "tfrac",  # \sin\theta \frac{a}{b} is sin θ times a/b (decision 11)
}  # fmt: skip
_CASE_WORDS = frozenset({"if", "for", "when", "where"})
_OTHERWISE = frozenset({"otherwise", "else"})
# The remainder, a \bmod m, and the modulus of a congruence, a \equiv b \pmod{m} or \mod m.
_MODULI = frozenset({"bmod", "mod", "pmod"})
# The extremes taken over a variable, \max_v, the operator names of where they are,
# \operatorname*{arg\,max}_k, the bounds, \sup_x and \inf_x, and the names of all of them
# in plain text, as min_{x}.
_OPTIMA = frozenset({"max", "min"})
_ARG_OPTIMA = frozenset({"argmax", "argmin"})
_BOUNDS = frozenset({"sup", "inf"})
_NAMED_OPTIMA = _OPTIMA | _ARG_OPTIMA | _BOUNDS
# The words that put the constraints of an optimum after it, \text{subject to} and
# \text{s.t.}, as they are read; \mathrm{s.t.} is read as the letters and stops of "s.t.".
_SUBJECT_TO = frozenset({"subject to", "s.t.", "s.t", "st", "such that", "subject to:"})
# The commands that are a label in a script, as \max is in \lambda_{\max}.
_LABEL_COMMANDS = _OPTIMA
# A command whose argument is prose, as in \text{where}.
_TEXT_COMMAND = re.compile(r"\\(?:text[a-z]*|mbox|hbox)(?![A-Za-z])")
# The halves of old-style cases, \left\{ \begin{array} ... \end{array} \right.
_LEFT_BRACE = re.compile(r"\\left\s*(?:\\\{|\\lbrace)")
_RIGHT_DOT = re.compile(r"\\right\s*\.")
# Square brackets around what an evaluation bracket evaluates: [F], \left[F\right], \big[F\big],
# and \left. F \right].
_SQUARE = re.compile(
    r"(?:\\left\s*(?:\[|\\lbrack|\.)|\\[bB]igg?l?\s*(?:\[|\\lbrack)|\[|\\lbrack)"
    r".*(?:\]|\\rbrack)",
    re.DOTALL,
)
# A name that reads as a differential inside an integral or a derivative: dx, dtheta, dx1,
# and dx_1, of a variable numbered as x_1 is.
_DIFFERENTIAL = re.compile(
    r"d(?:" + "|".join(sorted(vocab.GREEK, key=len, reverse=True)) + r"|[A-Za-z][0-9]*)"
    r"(?:_[0-9]+)?"
)
# Where "did you mean" looks: every command, and every environment, the language reads.
# A suggestion must be this close, so that \top is not taken for \to.
_KNOWN_COMMANDS = sorted(vocab.COMMANDS)
_ENVIRONMENTS = sorted(set().union(*vocab.ENVIRONMENTS.values()))
_CLOSE = 0.75


# The commonest environments first, when they are listed.
_FIRST = (
    "matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix", "smallmatrix", "array",
    "equation", "align", "gather", "multline", "split", "eqnarray", "flalign", "alignat",
)  # fmt: skip


def _environment_list() -> str:
    """The environments a program may use, by kind, for a note."""
    kinds = []
    for kind, names in vocab.ENVIRONMENTS.items():
        plain = [name for name in names if not name.endswith("*")]
        plain.sort(key=lambda name: (_FIRST.index(name) if name in _FIRST else len(_FIRST), name))
        listed = ", ".join(plain[:-1]) + f" and {plain[-1]}"
        starred = ", with their starred forms" if any(name.endswith("*") for name in names) else ""
        kinds.append(f"{listed} for {kind}{starred}")
    return "the supported environments are " + "; ".join(kinds)


_ENVIRONMENT_LIST = _environment_list()


def parse(source: str | Source, *, recover: bool = False) -> ast.Program:
    r"""Parse a program: a sequence of statements separated by line breaks, ``;`` or ``\\``.

    The first statement that cannot be read raises its
    :class:`~relativisticpy.errors.ParseError`. With ``recover``, the parser goes on
    past each such statement instead, to the next one, so that every error can be
    reported and every statement that can be read can be run: the program holds the
    statements that parse, and in ``errors`` the error of each that does not, both in
    the order of the source. :meth:`~relativisticpy.language.syntax.Program.in_order`
    gives them together, as written::

        >>> from relativisticpy.language import parse, to_sexpr
        >>> program = parse("a := 1\nb := (2\nc := a + \\frca{1}{2}\nd := 4", recover=True)
        >>> [to_sexpr(statement) for statement in program.statements]
        ['(:= a 1)', '(:= d 4)']
        >>> [(error.message, error.span.line) for error in program.errors]
        [("unclosed '('", 2), ('unknown LaTeX command \\frca', 3)]
        >>> [type(item).__name__ for item in program.in_order()]
        ['Define', 'ParseError', 'ParseError', 'Define']

    A statement that cannot be read ends where the brackets and environments it opens
    close, at a line break, ``;`` or ``\\``; if they never close, at the end of the
    line its error is on. So an unclosed bracket costs its own line, and the statement on
    the next line is read.
    """
    if isinstance(source, str):
        source = Source(source)
    tokens = _generalised_fractions(tokenize(source, recover=recover))
    return Parser(tokens, source, recover=recover).parse_program()


def parse_expression(source: str | Source) -> ast.Expr:
    """Parse text that must consist of exactly one expression."""
    if isinstance(source, str):
        source = Source(source)
    parser = Parser(_generalised_fractions(tokenize(source)), source)
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

    def __init__(self, tokens: list[Token], source: Source, *, recover: bool = False) -> None:
        self.tokens = tokens
        self.source = source
        self.recover = recover  # go on past a statement that cannot be read
        self.pos = 0
        self.previous = tokens[0]
        # Inside brackets and environments line breaks are insignificant.
        self._nesting = 0
        # Inside |...| a bar closes the absolute value.
        self._abs_depth = 0
        # Inside an integrand a differential (dx) ends the expression.
        self._integral_depth = 0
        # The column the current statement starts at, for continuation lines.
        self._statement_column = 1
        # Inside the rows of an alignment, a row can continue the one before.
        self._rows = 0
        # Inside a script, ':' gives an index its value (T_{a:0}) rather than relating.
        self._in_script = False
        # How deeply the current statement is nested, as inside an alignment.
        self._statement_nesting = 0
        # Inside the unbracketed argument of \sin and friends, brackets end it.
        self._function_argument = False
        # Inside a display, such as \begin{equation}, every statement is LaTeX.
        self._display = 0
        # In the current statement, the last operator that ended a line, which the next
        # line continued, and the first token of that line.
        self._continued: tuple[Token, Token] | None = None
        # The last argument of a command read, braces and all, to mark it if it is empty.
        self._group: Span | None = None
        # The binding power the expression being read stops at: an optimum that begins a
        # sum takes the whole of it, and one in a product only its factor.
        self._power: int = Power.NONE
        # Inside what an optimum is of, "subject to" ends it, even written as letters.
        self._bodies = 0

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
        token = self.tokens[index]
        if token.error is not None and not offset:  # what the lexer could not read
            raise token.error
        return token

    def advance(self) -> Token:
        self.pos = self._significant(self.pos)
        token = self.tokens[self.pos]
        if token.error is not None:
            raise token.error
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
        if opener is not None and kind in _CLOSING:
            raise _unclosed(opener, token)
        expected = kind.value if kind in (K.EOF, K.NEWLINE) else f"'{kind.value}'"
        message = f"expected {expected}"
        if after:
            message += f" after {after}"
        raise ParseError(f"{message}, found {token.describe()}", span=token.span)

    def expect_end(self) -> None:
        token = self.peek()
        if token.kind is not K.EOF:
            raise ParseError(f"unexpected {token.describe()}", span=token.span)

    def skip_separators(self) -> None:
        while self.tokens[self._significant(self.pos)].kind in _SEPARATORS:
            self.advance()

    def _skip_newlines(self) -> None:
        while self.tokens[self.pos].kind is K.NEWLINE:
            self.pos += 1

    def _continue_after(self, operator: Token) -> None:
        """Go past the line breaks after an operator, which the expression continues over,
        remembering where, to say so if the next line turns out to be a statement."""
        if self.tokens[self.pos].kind is K.NEWLINE and not self._nesting:
            self._skip_newlines()
            self._continued = (operator, self.tokens[self.pos])
        self._skip_newlines()

    def _span(self, start: Token | Span) -> Span:
        """The span from ``start`` to the last token consumed."""
        span = start if isinstance(start, Span) else start.span
        return span.to(self.previous.span) if self.previous.span.end > span.start else span

    @contextmanager
    def _nested(self, nested: bool = True) -> Iterator[None]:
        """Enter brackets (line breaks insignificant) or, with ``nested=False``, a block."""
        saved = (self._nesting, self._abs_depth, self._integral_depth, self._function_argument)
        self._nesting = saved[0] + 1 if nested else 0
        self._abs_depth = self._integral_depth = 0
        self._function_argument = False
        try:
            yield
        finally:
            (self._nesting, self._abs_depth, self._integral_depth, self._function_argument) = saved

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
        statements: list[ast.Statement] = []
        errors: list[ParseError] = []
        self.skip_separators()
        while self.tokens[self._significant(self.pos)].kind is not K.EOF:
            start = self._significant(self.pos)
            try:
                made = self.statement()
                if not self.check(K.EOF):
                    self._expect_separator(made)
            except ParseError as error:
                if not self.recover:
                    raise
                errors.append(error)
                self.pos = self._resume(start, error)
                self.previous = self.tokens[self.pos - 1]
            else:
                statements.extend(made)
            self.skip_separators()
        span = self.tokens[0].span.to(self.tokens[-1].span)
        return ast.Program(tuple(statements), tuple(errors), span=span)

    def _resume(self, start: int, error: ParseError) -> int:
        """Where to go on after the statement from token ``start``, which cannot be read:
        after its end, a separator at which the brackets and environments it opened are
        closed, and which no continuation line follows; or, if they never close, after
        the line of its error."""
        tokens = self.tokens
        at = tokens[start].span.start
        if error.span is not None and error.span.source is self.source:
            at = max(at, error.span.start)
        depth, index = 0, start
        while tokens[index].kind is not K.EOF:
            token = tokens[index]
            if token.kind in _OPENERS or token.kind is K.BEGIN:
                depth += 1
            elif token.kind in _CLOSERS or token.kind is K.END:
                depth = max(0, depth - 1)
            elif (
                token.kind in _SEPARATORS
                and depth == 0
                and token.span.start >= at
                and not self._continued_after(index)
            ):
                return index + 1
            index += 1
        for index in range(start, len(tokens) - 1):  # the brackets never close
            if tokens[index].kind in _SEPARATORS and tokens[index].span.start >= at:
                return index + 1
        return len(tokens) - 1

    def _continued_after(self, index: int) -> bool:
        """Whether the line after the line break at ``index`` continues the statement
        before it, as a line starting with ``+`` does."""
        if self.tokens[index].kind is not K.NEWLINE:
            return False
        following = self.tokens[index + 1]
        return following.kind in _CONTINUING and following.kind not in _SIGNS_FIRST

    def _expect_separator(self, made: list[ast.Statement] | None = None) -> None:
        """Expect what ends a statement: a line break, ';' or '\\\\'. ``made`` are the
        statements just read, which say what was meant when something else follows."""
        token = self.peek()
        if token.kind in _SEPARATORS:
            return
        continued = self._continued
        # x +, then a line y := 2, which the + made part of the expression
        if (
            continued is not None
            and token.kind is K.DEFINE
            and token.span.start >= continued[1].span.start
        ):
            raise _dangling(continued[0], continued[1])
        raise _stray(token, self.previous, made or [])

    def statement(self) -> list[ast.Statement]:
        token = self.peek()
        if token.kind is K.BEGIN and token.value in vocab.ALIGNMENT_ENVIRONMENTS:
            return self._alignment()
        start = self._significant(self.pos)
        outer = self._statement_column, self._statement_nesting
        self._statement_column, self._statement_nesting = token.span.column, self._nesting
        self._continued = None
        try:
            return self._statement(token, start)
        finally:
            self._statement_column, self._statement_nesting = outer

    def _statement(self, token: Token, start: int) -> list[ast.Statement]:
        if token.kind is K.IDENT and self.peek(1).kind is K.COLON and self.peek(2).kind is K.LPAREN:
            statements: list[ast.Statement] = [self._arrow_function()]
        else:
            statements = self._expression_statements()
        # A statement in a LaTeX display is LaTeX however it is written: in
        # \begin{equation} E = mc^2 \end{equation}, mc is m times c. So is one with any
        # command, even one that leaves no token of its own, as \, and {\rm d} do.
        tokens = self.tokens[start : self.pos]
        begin = self.tokens[start - 1].span.end if start else 0  # with \mathrm before d
        written = self.source.text[begin : tokens[-1].span.end] if tokens else ""
        latex = (
            self._display > 0
            or self.source.latex
            or "\\" in written
            or any(_is_latex(t) for t in tokens)
        )
        if not latex:  # Python, written as a program would be, which reads as something else
            for statement in statements:
                _python_habits(statement)
        return [replace(statement, latex=latex) for statement in statements]

    def _expression_statements(self) -> list[ast.Statement]:
        """Expressions and definitions, such as ``x, y := ...``, separated by commas, wide
        spaces or words of prose such as ``\\text{and}``, as in ``a = b, \\quad c \\equiv d``.
        A comma or full stop that ends the statement is the punctuation of the sentence
        the equation belongs to: ``E = mc^2 \\ .``"""
        statements: list[ast.Statement] = []
        if self._at_connective(self.peek()) and self._starts_operand(self.peek(1)):
            self.advance()  # \text{where} \ E^* \equiv ...: the word is prose
        clause = [self.expression()]
        joined = [False]  # whether each expression follows the one before after a comma alone
        while True:
            if define := self.accept(K.DEFINE):
                count = _defined(clause, joined)
                statements.extend(_evaluations(clause[:-count], joined[:-count]))
                self._continue_after(define)
                value = self.expression()
                statements.append(self._definition(clause[-count:], value, define))
                # x \equiv r/M, \quad y \equiv t/M: another statement may follow on the line
                if not self._statement_follows(count) or self._separator() is None:
                    break
                clause, joined = [self.expression()], [False]
                continue
            separator = self._separator()
            if separator is None:
                statements.extend(_evaluations(clause, joined))
                break
            clause.append(self.expression())
            joined.append(separator == ",")
        self._skip_punctuation()
        return _ranged(statements)

    def _separator(self) -> Literal[",", " "] | None:
        """Read what separates two expressions of a statement: a comma alone (``,``), or
        wide spacing or a word, maybe after a comma (`` ``). ``None`` if the statement
        ends here, after a word of prose that ends it, if there is one."""
        if self._at_connective(self.peek()):
            self.advance()
            return " " if self._starts_operand(self.peek()) else None
        if self.check(K.COMMA) and not self._at_punctuation():
            self.advance()
            if self._at_connective(self.peek()):  # a = 1, \text{and} b = 2
                self.advance()
                return " "
            return " " if self.peek().wide_space else ","
        return " " if self._at_wide_space() else None

    def _statement_follows(self, targets: int) -> bool:
        """After the value of a definition of ``targets`` names, whether another statement
        follows: after wide spacing or a word, or after a comma when it is an equation or
        a definition. ``x, y := 1, 2`` is not two statements."""
        if self._at_connective(self.peek()) or self._at_wide_space():
            return True
        if not self.check(K.COMMA) or self._at_punctuation():
            return False
        if self.peek(1).wide_space or self._at_connective(self.peek(1)):
            return True
        return targets == 1 and self._relates(self._significant(self._significant(self.pos) + 1))

    def _relates(self, index: int) -> bool:
        """Whether the expression starting at token ``index`` relates or defines at its
        top level, as ``y \\equiv t/M`` does and ``2`` does not."""
        depth, start = 0, index
        while True:
            token = self.tokens[index]
            kind = token.kind
            if kind is K.EOF or (depth == 0 and token.wide_space and index > start):
                return False
            if kind in _OPENERS or kind is K.BEGIN:
                depth += 1
            elif kind in _CLOSERS or kind is K.END:
                depth -= 1
                if depth < 0:
                    return False
            elif depth == 0 and (kind in _SEPARATORS or kind is K.COMMA):
                if kind is not K.NEWLINE or not self._nesting:
                    return False
            elif depth == 0 and (kind in _RELATIONS or kind in _COMPARISONS or kind is K.DEFINE):
                return True
            index += 1

    def _at_connective(self, token: Token) -> bool:
        """A word of prose such as ``\\text{and}``, ``\\text{or}`` or ``\\text{where}`` at a
        statement's top level, which separates equations as wide spacing does:
        ``a = 1 \\quad \\text{and} \\quad b = 2``. Typed plainly, ``and`` and ``or`` are
        logic."""
        return (
            token.kind is K.IDENT
            and token.value in _CONNECTIVES
            and token.verbatim
            and self._nesting == self._statement_nesting
        )

    def _at_wide_space(self) -> bool:
        """Wide spacing at a statement's top level separates equations:
        ``\\Gamma^0_{00} = \\partial_0 \\alpha \\qquad \\Gamma^0_{01} = \\partial_1 \\alpha``."""
        token = self.peek()
        return (
            token.wide_space
            and self._nesting == self._statement_nesting
            and self._starts_operand(token)
        )

    def _at_punctuation(self) -> bool:
        return self.peek().kind in (K.COMMA, K.PERIOD) and self.peek(1).kind in _STATEMENT_ENDS

    def _skip_punctuation(self) -> None:
        if self._at_punctuation():
            self.advance()

    def _definition(self, targets: list[ast.Expr], value: ast.Expr, define: Token) -> ast.Statement:
        span = targets[0].span.to(value.span)
        equiv = _equivalence(define)
        if equiv and len(targets) == 1 and _modular(value):  # x \equiv 2 \pmod{3}: a congruence
            congruence = ast.Relation(targets[0], "≡", value, define.span.text, span=span)
            return ast.Evaluate(congruence, span=span)
        if len(targets) == 1 and (applied := _operator_applied(targets[0])) is not None:
            # \operatorname{sinc} x := ..., the argument written after the name as books
            # write it, defines sinc as sinc(x) := ... does
            name, param = applied
            return ast.FunctionDefinition(name, (param,), value, span=span)
        if len(targets) == 1 and not _definable(targets[0]):
            if equiv:  # A \equiv B, where A cannot be defined, states an identity
                identity_node = _identity(targets[0], value, span, define.span.text)
                return ast.Evaluate(identity_node, span=span)
            if "\\" in targets[0].span.text or _scripted_call(targets[0]) or _sampled(targets[0]):
                # := defines, as R_{nl}(r) := ... or x[n] := ...; the compiler says how, or
                # why it cannot
                return ast.Define(tuple(targets), value, span=span)
        if len(targets) == 1 and isinstance(call := targets[0], ast.Call):
            called = _parameter(call.func)  # H'(x) := ...: a prime is part of the name
            if called is None:
                raise ParseError("cannot define this", span=call.func.span)
            params = []
            for arg in call.args:
                parameter = _parameter(arg)  # r(t') := ...: so is a parameter's
                if parameter is None:
                    raise ParseError("function parameters must be names", span=arg.span)
                params.append(parameter)
            return ast.FunctionDefinition(called, tuple(params), value, span=span)
        for target in targets:
            if not (
                isinstance(target, ast.Name | ast.Scripted | dirac.DEFINABLE) or _decorated(target)
            ):
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
        self._rows += 1
        self._display += 1
        try:
            with self._nested():
                while True:
                    while self.peek().kind in (K.ROW_SEP, K.SEMICOLON):
                        self.advance()
                    if self.check(K.END):
                        break
                    statements.extend(self.statement())
                    # breqn's dgroup holds its dmath environments one after another
                    following = self.peek()
                    nested = following.kind is K.BEGIN and following.value in (
                        vocab.ALIGNMENT_ENVIRONMENTS
                    )
                    if not self.check(K.END) and not nested:
                        self._expect_separator()
                self.advance()
        finally:
            self._rows -= 1
            self._display -= 1
        return statements

    # -- expressions ----------------------------------------------------------

    def expression(self, min_power: int = Power.NONE) -> ast.Expr:
        outer, self._power = self._power, min_power
        try:
            return self._operations(self._prefix(), min_power)
        finally:
            self._power = outer

    def _operations(self, left: ast.Expr, min_power: int) -> ast.Expr:
        """The operations that follow ``left``, as far as ``min_power`` lets them go."""
        while True:
            if self._continues():
                self._skip_newlines()
            elif self._continues_row():
                self.pos = self._significant(self.pos) + 1  # past the row break
                if self.check(K.LBRACE) and self.peek(1).kind is K.RBRACE:
                    self.advance()  # and past an empty group, as in & & {} - c
                    self.advance()
            token = self.peek()
            power = self._infix_power(token)
            if power <= min_power:
                return left
            left = self._infix(left, token, power)

    def _continues(self) -> bool:
        """Whether a line break here is followed by a continuation line: the next line,
        when it starts with an operator that cannot start a statement, as ``+`` or ``=``
        cannot, or a line indented beyond the statement that starts with any operator,
        ``-`` included::

            ds^2 = -(1 - 2M/r) dt^2
            + dr^2 / (1 - 2M/r)
                - r^2 d\\Omega^2
        """
        if self._nesting or self.tokens[self.pos].kind is not K.NEWLINE:
            return False
        index = self.pos
        while self.tokens[index].kind is K.NEWLINE:
            index += 1
        token = self.tokens[index]
        if token.kind not in _CONTINUING:
            return False
        if token.span.column > self._statement_column:
            return True
        return index == self.pos + 1 and token.kind not in _SIGNS_FIRST

    def _continues_row(self) -> bool:
        """Whether an alignment's row break is followed by a row that continues this
        expression: one that starts with a relation, as in ``a &= b \\\\ &= c``, or
        with an operator after an alignment marker, as in ``a &= b \\\\ & + c``."""
        if not self._rows:
            return False
        index = self._significant(self.pos)
        if self.tokens[index].kind is not K.ROW_SEP:
            return False
        index = self._significant(index + 1)
        token = self.tokens[index]
        if token.kind is K.LBRACE and self.tokens[self._significant(index + 1)].kind is K.RBRACE:
            aligned = token.aligned  # & & {} - c: the empty group only makes '-' binary
            token = self.tokens[self._significant(self._significant(index + 1) + 1)]
        else:
            aligned = token.aligned
        relation = token.kind in _RELATIONS or token.kind in _COMPARISONS
        return relation or (aligned and token.kind in _OPERATORS)

    def _infix_power(self, token: Token) -> int:
        kind = token.kind
        if kind is K.COMMAND and token.value in _MODULI:  # a \bmod m, a \equiv b \pmod{m}
            return Power.PRODUCT if token.value == "bmod" else Power.MODULUS
        if kind is K.COMMAND and token.value == "circledast":  # x[n] \circledast h[n]
            return Power.CONVOLUTION
        if kind is K.DEFINE and self._nesting > self._statement_nesting and _equivalence(token):
            return Power.EQUATION  # solve(x \equiv 2 \pmod{3}, x): in brackets, a relation
        if kind is K.BAR:
            if self._abs_depth:
                return Power.NONE  # it closes the absolute value
            if self.peek(1).kind in (K.UNDERSCORE, K.CARET):  # x^3 \Big|_0^1: a side, or a sum
                return Power.EVALUATION if self._bar_with_limits() else Power.SUBSTITUTION
            if _given_bar(token):  # \Pr(A \mid B), X \mid N \sim \operatorname{Bin}(N, p)
                return Power.EQUATION
            return Power.PRODUCT  # 2|x|
        if kind in _RELATIONS and not (kind is K.COLON and self._in_script):
            return Power.EQUATION
        if self._at_connective(token):
            return Power.NONE  # \text{and} between equations ends the one before it
        if self._bodies and self._subject_to_word(self._significant(self.pos)):
            return Power.NONE  # \mathrm{s.t.} ends what an optimum is of
        if kind is K.IDENT and token.value == "or":
            return Power.OR
        if kind is K.IDENT and token.value == "and":
            return Power.AND
        if kind in _COMPARISONS:
            return Power.COMPARISON
        if kind in (K.PLUS, K.MINUS, K.PLUS_MINUS, K.MINUS_PLUS) or self._direct_sum(token):
            return Power.SUM
        if kind in (K.STAR, K.SLASH, K.WEDGE, K.COMPOSE) or self._hadamard(token):
            return Power.PRODUCT
        if kind is K.DOUBLE_STAR:
            return Power.POWER
        if self._starts_operand(token) and not self._at_wide_space():
            return Power.PRODUCT  # juxtaposition
        return Power.NONE

    def _infix(self, left: ast.Expr, token: Token, power: int) -> ast.Expr:
        kind = token.kind
        if kind is K.COMMAND and token.value in _MODULI:
            return self._modulus(left)
        if kind is K.COMMAND and token.value == "circledast":
            return self._circular_convolution(left)
        if kind is K.BAR and power in (Power.SUBSTITUTION, Power.EVALUATION):
            return self._bar(left)
        if self._hadamard(token):
            self.advance()
            right = self._elementwise_operand()
            return ast.Binary("⊙", left, right, span=left.span.to(right.span))
        if self._direct_sum(token):
            self.advance()
            right = self.expression(Power.SUM)
            return ast.Binary("⊕", left, right, span=left.span.to(right.span))
        if power == Power.PRODUCT and kind not in (K.STAR, K.SLASH, K.WEDGE, K.COMPOSE):
            self._numbers_apart(token)
            right = self.expression(Power.PRODUCT)
            return ast.Binary("*", left, right, implicit=True, span=left.span.to(right.span))
        self._continue_after(self.advance())
        if kind in _RELATIONS or kind is K.DEFINE or _given_bar(token):
            return self._relations(left, token)
        if kind in _COMPARISONS:
            operands, ops = [left], [_COMPARISONS[kind]]
            operands.append(self.expression(Power.COMPARISON))
            while self.peek().kind in _COMPARISONS:
                comparison = self.advance()
                ops.append(_COMPARISONS[comparison.kind])
                self._continue_after(comparison)
                operands.append(self.expression(Power.COMPARISON))
            span = left.span.to(operands[-1].span)
            return ast.Comparison(tuple(operands), tuple(ops), span=span)
        if kind is K.DOUBLE_STAR:
            right = self.expression(Power.POWER - 1)  # right-associative
            return ast.Binary("**", left, right, span=left.span.to(right.span))
        if kind is K.SLASH and (derivative := self._solidus_derivative(left)) is not None:
            return derivative  # d/dx, dy/dx, d(x^3)/dx: Leibniz's fraction in a line of text
        right = self.expression(power)
        op = cast(Literal["+", "-", "*", "/", "and", "or", "∧", "±", "∓", "∘"], token.value)
        if op == "/":
            return _quotient(left, right, left.span.to(right.span), leibniz=False)
        return ast.Binary(op, left, right, span=left.span.to(right.span))

    def _modulus(self, left: ast.Expr) -> ast.Binary:
        r"""``a \bmod m``, the remainder, which binds as a product does; or ``b \pmod{m}`` and
        ``b \mod m``, the modulus of a congruence, of all of b. Either is ``(mod b m)``."""
        command = self.advance()
        if command.value == "bmod":
            modulus = self.expression(Power.PRODUCT)
        elif self.check(K.LBRACE):
            modulus = self._brace_argument(command)
        else:
            modulus = self.expression(Power.SIGN)
        return ast.Binary("mod", left, modulus, span=left.span.to(self._span(command)))

    def _circular_convolution(self, left: ast.Expr) -> ast.CircularConvolution:
        """``x \\circledast h``: the circular convolution of what is on either side, which
        binds looser than a product, so that ``x[n] \\circledast h[n]`` convolves x[n] and
        h[n]; with its length as a subscript, ``x \\circledast_N h``."""
        command = self.advance()
        length = None
        if self.check(K.UNDERSCORE):
            marker = self.advance()
            length = self._script("_", marker).body
            if length is None:
                raise ParseError(
                    "the subscript of \\circledast is the length of the convolution",
                    span=self._span(marker),
                    hint="for example x \\circledast_N h, or x \\circledast_{4} h",
                )
        self._continue_after(command)
        right = self.expression(Power.CONVOLUTION)
        return ast.CircularConvolution(left, right, length, span=left.span.to(right.span))

    def _hadamard(self, token: Token) -> bool:
        """``A \\odot B`` or ``A ⊙ B``: the elementwise product, where ``\\odot`` stands
        between two operands, the second maybe signed. Elsewhere, as in ``M_\\odot``, it
        is the Sun."""
        if not (
            token.kind is K.IDENT
            and token.value == "odot"
            and _is_latex(token)
            and not self._in_script
        ):
            return False
        follower = self.peek(1)
        if follower.kind in _SIGNS:  # A \odot -B
            return self._starts_operand(self.peek(2))
        return self._starts_operand(follower)

    def _direct_sum(self, token: Token) -> bool:
        """``A \\oplus B`` or ``A ⊕ B``: the direct sum, where ``\\oplus`` stands between two
        operands, and binds as a sum does."""
        return (
            token.kind is K.IDENT
            and token.value == "oplus"
            and _is_latex(token)
            and not self._in_script
            and self._starts_operand(self.peek(1))
        )

    def _numbers_apart(self, token: Token) -> None:
        """Juxtaposed numbers, ``10 000`` or ``2\\,3``, are not a product: say so."""
        if token.kind is K.NUMBER and self.previous.kind is K.NUMBER:
            raise ParseError(
                f"two numbers side by side, {self.previous.value} and {token.value}",
                span=self.previous.span.to(token.span),
                hint="multiply with \\cdot, or group digits in threes, as in 10\\,000",
            )

    def _elementwise_operand(self) -> ast.Expr:
        """The right operand of ``\\odot``: the factors that follow it, maybe after a
        sign, as in ``w_a \\odot \\mathbb{E}_t[X_{ta}]`` or ``A \\odot -B``."""
        sign = self.peek()
        if sign.kind in _SIGNS:
            self.advance()
            operand = self._elementwise_operand()
            return ast.Unary(_SIGNS[sign.kind], operand, span=sign.span.to(operand.span))
        operand = self.expression(Power.PRODUCT)
        while (
            self._starts_operand(token := self.peek())
            and not self._at_wide_space()
            and not self._hadamard(token)
            and not self._direct_sum(token)
        ):
            self._numbers_apart(token)
            right = self.expression(Power.PRODUCT)
            operand = ast.Binary(
                "*", operand, right, implicit=True, span=operand.span.to(right.span)
            )
        return operand

    def _relations(self, left: ast.Expr, first: Token) -> ast.Expr:
        """``a = b``, ``a \\sim b``, or relations in a row: ``a = b = c``."""
        operands, relations, written = [left], [_relation(first)], [first.span.text]
        while True:
            operands.append(self.expression(Power.EQUATION))
            if self.peek().kind not in _RELATIONS and not _given_bar(self.peek()):
                break
            relation = self.advance()
            relations.append(_relation(relation))
            written.append(relation.span.text)
            self._continue_after(relation)
        span = left.span.to(operands[-1].span)
        if "∣" in relations:  # \Pr(X = 2 \mid N = 3), X \mid Y = 0 \sim \mathcal{N}(0, 1)
            return _given(operands, relations, written)
        operands, relations, written = _with_comparisons(operands, relations, written)
        if len(relations) > 1:
            return ast.Chain(tuple(operands), tuple(relations), tuple(written), span=span)
        if relations[0] == "=":
            return ast.Equation(left, operands[1], span=span)
        return ast.Relation(left, relations[0], operands[1], written[0], span=span)

    def _starts_operand(self, token: Token) -> bool:
        """Whether ``token`` can begin an operand, making juxtaposition a product."""
        kind = token.kind
        if self._integral_depth and self._at_differential():
            return False
        if kind in (K.NUMBER, K.LPAREN, K.LBRACKET, K.LBRACE, K.ELLIPSIS, K.LANGLE):
            return True
        if kind is K.IDENT:
            return token.value not in _KEYWORDS and not self._at_connective(token)
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
        if kind in (K.PLUS_MINUS, K.MINUS_PLUS):
            self.advance()
            operand = self.expression(Power.SIGN)
            sign: Literal["±", "∓"] = "±" if kind is K.PLUS_MINUS else "∓"
            return ast.Unary(sign, operand, span=token.span.to(operand.span))
        if kind is K.STAR and not self._starts_operand(self.peek(1)):
            self.advance()  # a lone \cdot, as in a column of dots
            return ast.Dots(span=token.span)
        if kind in (K.STAR, K.DOUBLE_STAR) and self._starts_operand(self.peek(1)):
            self.advance()  # the Hodge dual, *F; **A is the dual of the dual
            operand = self.expression(Power.SIGN)
            node = ast.Unary("*", operand, span=token.span.to(operand.span))
            return ast.Unary("*", node, span=node.span) if kind is K.DOUBLE_STAR else node
        if kind is K.STRING:
            self.advance()
            return ast.String(token.value, span=token.span)
        if kind is K.LBRACE and self.peek(1).kind is K.RBRACE:
            self.advance()  # {} -T: an empty group only spaces what follows
            self.advance()
            return self._prefix()
        if kind in (K.CARET, K.UNDERSCORE):
            return self._prescripted()
        grouped = kind in (K.LPAREN, K.LBRACE)
        return self._postfix(self._atom(), grouped=grouped, braced=kind is K.LBRACE)

    def _prescripted(self) -> ast.Expr:
        """A script before its symbol labels it: ``{}^{(3)}R`` is the curvature of space."""
        marker = self.advance()
        if self.check(K.LBRACE):
            group = self._raw_group(K.LBRACE, K.RBRACE, marker)
            label = self.source.text[group[0].span.end : group[-1].span.start].strip()
        else:
            label = self.advance().span.text
        token = self.peek()
        if token.kind is not K.IDENT:
            written = self._span(marker).text
            example = "x" + ("^2" if marker.kind is K.CARET else "_0")
            raise ParseError(
                f"'{written}' has nothing to attach to",
                span=self._span(marker),
                hint=f"write it after what it is on, as in {example}",
            )
        self.advance()
        if marker.kind is K.UNDERSCORE and _COUNT.fullmatch(label):
            order = self._hypergeometric_order(token)
            if order is not None:  # {}_2F_1(a, b; c; z)
                return self._postfix(self._hypergeometric(marker, label, order))
        name = ast.Name(f"{marker.value}{{{label}}}{token.value}", True, span=self._span(marker))
        return self._postfix(name)

    def _hypergeometric_order(self, name: Token) -> str | None:
        """The ``q`` of ``{}_pF_q`` after its ``F``, which the lexer may have joined to it, as
        it joins ``F_1``; ``None``, having read nothing, if ``name`` is no such ``F``."""
        if (joined := re.fullmatch(r"F_([0-9]+|[A-Za-z])", name.value)) is not None:
            return joined.group(1)
        if name.value != "F" or not self.check(K.UNDERSCORE):
            return None
        before = self.pos, self.previous
        marker = self.advance()
        script, _ = self._attempt(lambda: self._script("_", marker))
        order = script.body if script is not None else None
        if isinstance(order, ast.Number) and order.text.isdigit():
            return order.text
        if isinstance(order, ast.Name) and len(order.id) == 1:
            return order.id
        self.pos, self.previous = before
        return None

    def _hypergeometric(self, marker: Token, count: str, order: str) -> ast.Call:
        """``{}_pF_q(a_1, ..., a_p; b_1, ..., b_q; z)``, the generalised hypergeometric
        function, its parameter lists separated by semicolons: a call of ``_{p}F_q`` with an
        argument for each list, the list itself where it holds several or none."""
        written = f"{{}}_{count}F_{order}"
        if not self.check(K.LPAREN) or self.peek().space_before:
            raise ParseError(
                f"{written}, the hypergeometric function, needs its parameters in brackets",
                span=self._span(marker),
                hint=f"write them, separated by semicolons, as in {written}(a, b; c; z)",
            )
        name = ast.Name(f"_{{{count}}}F_{order}", True, span=self._span(marker))
        opener = self.advance()
        groups: list[ast.Expr] = []
        with self._nested():
            while True:
                start = self.peek()
                items: list[ast.Expr] = []
                if not self.check(K.SEMICOLON) and not self.check(K.RPAREN):
                    items.append(self.expression())
                    while self.accept(K.COMMA):
                        items.append(self.expression())
                if len(items) == 1:
                    groups.append(items[0])
                else:
                    span = (
                        self._span(start)
                        if items
                        else Span(start.span.source, start.span.start, start.span.start)
                    )
                    groups.append(ast.List(tuple(items), span=span))
                if not self.accept(K.SEMICOLON):
                    break
            self.expect(K.RPAREN, opener=opener)
        return ast.Call(name, tuple(groups), span=self._span(marker))

    def _atom(self) -> ast.Expr:
        token = self.peek()
        kind = token.kind
        if kind is K.ELLIPSIS:
            self.advance()
            return ast.Dots(span=token.span)
        if kind is K.NUMBER:
            self.advance()
            return ast.Number(token.value, span=token.span)
        if kind is K.IDENT:
            if token.value == "d" and self._at_plain_differential():
                return self._plain_differential()
            self.advance()
            if token.verbatim and token.value in _ARG_OPTIMA:  # \operatorname*{arg\,max}_k
                return self._optimum(token, token.value)
            following = self.peek()
            if (
                token.verbatim
                and token.value == "arg"
                and following.kind is K.IDENT
                and following.verbatim
                and following.value in _OPTIMA
            ):  # \mathrm{arg\,min}_w
                self.advance()
                return self._optimum(token, "arg" + following.value)
            if token.value in _NAMED_OPTIMA and following.kind is K.UNDERSCORE:
                return self._optimum(token, token.value)  # min_{x} in plain text
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
            return dirac.ket(self) if dirac.at_ket(self) else self._absolute_value()
        if kind is K.LANGLE:
            return dirac.angle(self)
        previous = self.previous
        after = ""
        if (
            previous is not token
            and previous.span.end <= token.span.start
            and previous.kind in _OPERATORS | _RELATIONS | {K.DEFINE, *_COMPARISONS}
        ):
            after = f" after '{previous.span.text}'"
        raise ParseError(
            f"expected an expression{after}, found {token.describe()}", span=token.span
        )

    def _postfix(self, node: ast.Expr, *, grouped: bool = False, braced: bool = False) -> ast.Expr:
        """Scripts, calls, primes and factorials. A ``grouped`` node (from brackets)
        starts new scripts rather than extending its own: ``(x^2)^3`` is not ``x^{2^3}``.
        A ``braced`` one staggers its indices, as MTW write ``{\\Gamma^r}_{tt}``: a script
        of a kind it does not have yet extends it."""
        while True:
            token = self.peek()
            kind = token.kind
            if kind in (K.CARET, K.UNDERSCORE) and (
                (evaluated := self._evaluation_bracket(node)) is not None
            ):  # \left[ x e^x \right]_0^1
                node, grouped = evaluated, False
            elif kind in (K.CARET, K.UNDERSCORE):
                staggered = braced and _without_script(node, "^" if kind is K.CARET else "_")
                node = self._attach_script(node, extend=not grouped or staggered)
                grouped = False
            elif kind is K.BAR and self._right_bar(token):  # \left. V \right|_{q = p/d}
                node, grouped = self._bar(node, enclosed=True), False
            elif (
                kind is K.LBRACE
                and not token.space_before  # z \, {}_2F_1: a prescript of what follows
                and self.peek(1).kind is K.RBRACE
                and (isinstance(node, ast.Scripted) or self.peek(2).kind in (K.CARET, K.UNDERSCORE))
            ):
                self.advance()  # R^{a}{}_{bcd}: the empty group only spaces the indices
                self.advance()
            elif (
                kind is K.LPAREN
                and not token.space_before
                and isinstance(node, ast.Name | ast.Prime)
                and (not self._function_argument or _plain_call(node, token))
            ):
                node = self._call(node)
            elif (
                kind is K.LPAREN
                and not token.space_before
                and isinstance(node, ast.Scripted)
                and (numbered := _numbered(node)) is not None
                and (not self._function_argument or _plain_call(numbered, token))
            ):  # \psi_1(x) and R_{21}(r) are functions, as x_0(t) is
                node = self._call(numbered)
            elif (
                kind is K.LPAREN
                and not token.space_before
                and isinstance(node, ast.Scripted)
                and _starred(node)
                and not self._function_argument
            ):  # \psi^*(x) is \psi(x)^*, the conjugate of the call
                call = self._call(node.base)
                node = ast.Scripted(call, node.scripts, span=self._span(node.span))
            elif kind is K.LBRACE and _applicable(node) and self._at_typeset_arguments():
                node = self._typeset_call(node)
            elif kind is K.PRIME and grouped and not braced:  # (x^3)': of all the brackets
                order = 0
                while self.accept(K.PRIME):
                    order += 1
                node, grouped = ast.Primed(node, order, span=self._span(node.span)), False
            elif kind is K.LPAREN and not token.space_before and isinstance(node, ast.Primed):
                node = self._call(node)  # (x^3)'(2), the derivative at 2
            elif kind is K.PRIME and isinstance(node, ast.Name):
                order = 0
                while self.accept(K.PRIME):
                    order += 1
                node = ast.Prime(node, order, span=self._span(node.span))
            elif kind is K.PRIME and isinstance(node, ast.Scripted | ast.Operator):
                order = 0
                while self.accept(K.PRIME):
                    order += 1
                if isinstance(node, ast.Scripted):
                    node = self._primed(node, order)
            elif kind is K.COMMA and [self.peek(i).kind for i in (1, 2, 3)] == [
                K.LBRACE,
                K.RBRACE,
                K.UNDERSCORE,
            ]:  # \phi,{}_\mu is \phi_{,\mu}
                for _ in range(3):
                    self.advance()
                script = self._script("_", self.advance())
                if script.indices is not None:
                    items = tuple(replace(i, derivative=",") for i in script.indices.items)
                    script = replace(script, indices=replace(script.indices, items=items))
                base, scripts = (
                    (node.base, node.scripts) if isinstance(node, ast.Scripted) else (node, ())
                )
                node = ast.Scripted(base, (*scripts, script), span=self._span(base.span))
            elif kind is K.BANG:
                self.advance()
                if self.check(K.BANG) and not self.peek().space_before:  # n!!
                    bangs = token.span.to(self.advance().span)
                    double = ast.Name("factorial2", span=bangs)
                    node = ast.Call(double, (node,), span=self._span(node.span))
                else:
                    node = ast.Factorial(node, span=self._span(node.span))
            else:
                return node

    def _primed(self, node: ast.Scripted, order: int) -> ast.Expr:
        """Primes after scripts. After an unbraced index label they prime the label:
        ``x^\\mu'`` is ``x^{\\mu'}``. A name with a number as its subscript is a name of
        its own, so ``\\phi_0'`` primes it whole, as ``phi_0'`` does in plain text.
        Otherwise they prime the symbol, as TeX sets ``R_{\\mu\\nu}'``: ``R'_{\\mu\\nu}``."""
        span = self._span(node.span)
        script = node.scripts[-1]
        if isinstance(script.body, ast.Name) and not script.span.text[1:].lstrip().startswith("{"):
            return _primed_script(node, "'" * order)
        base = node.base
        if not isinstance(base, ast.Name):
            raise ParseError(
                "a prime after these scripts is ambiguous",
                span=span,
                hint="write the prime on the symbol, as in x'_0, or in the script, as in x_{0'}",
            )
        if (digits := _numeric_subscripts(node)) is not None:
            name = ast.Name(f"{base.id}_{digits}", base.verbatim, span=node.span)
            return ast.Prime(name, order, span=span)
        return ast.Scripted(ast.Prime(base, order, span=span), node.scripts, span=span)

    def _at_typeset_arguments(self) -> bool:
        """Whether a braced group of arguments in ``\\left( ... \\right)`` follows, as
        SymPy typesets a function's: ``a{\\left(t \\right)}``."""
        index = self._significant(self.pos) + 1
        opener = self.tokens[index]
        if opener.kind is not K.LPAREN or not opener.span.text.startswith("\\left"):
            return False
        depth = 0
        for position in range(index, len(self.tokens)):
            kind = self.tokens[position].kind
            if kind is K.LPAREN:
                depth += 1
            elif kind is K.RPAREN:
                depth -= 1
                if depth == 0:
                    closer, after = self.tokens[position], self.tokens[position + 1]
                    return closer.span.text.startswith("\\right") and after.kind is K.RBRACE
            elif kind is K.EOF:
                return False
        return False

    def _typeset_call(self, func: ast.Expr) -> ast.Expr:
        """``a{\\left(t \\right)}``, a function applied as SymPy typesets it, so that what
        RelativisticPy prints reads back as it was: ``a^{2}{\\left(t \\right)}`` is
        ``a(t)^2``."""
        self.advance()  # the brace
        base: ast.Expr = func
        powers: tuple[ast.Script, ...] = ()
        if isinstance(func, ast.Scripted):  # \psi_{1}^{2}{\left(x \right)} is psi_1(x)^2
            subscripts = tuple(script for script in func.scripts if script.kind == "_")
            powers = tuple(script for script in func.scripts if script.kind == "^")
            base = func.base
            if subscripts:
                numbered = _numbered(replace(func, scripts=subscripts))
                assert numbered is not None
                base = numbered
        call = self._call(base)
        self.expect(K.RBRACE, after="the arguments")
        call = replace(call, span=self._span(func.span))
        if powers:
            return ast.Scripted(call, powers, span=call.span)
        return call

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
        if self._at_brace_cases(opener):  # \left\{ \begin{array} ... \end{array} \right.
            with self._nested():
                cases = self._cases()
                self.expect(K.RPAREN, opener=opener)
            return replace(cases, span=self._span(opener))
        with self._nested():
            if self.accept(K.RPAREN):
                return ast.Tuple((), span=self._span(opener))
            first = self.expression()
            if not self.check(K.COMMA):
                self.expect(K.RPAREN, opener=opener)
                # from the bracket on, so that what is built on it, (a + b)^2, keeps it
                return replace(first, span=self._span(opener))
            items = [first]
            while self.accept(K.COMMA):
                if self.check(K.RPAREN):
                    break
                items.append(self.expression())
            self.expect(K.RPAREN, opener=opener)
        if len(items) == 2 and opener.span.text.endswith(("\\{", "\\lbrace")):
            bracket = self._poisson_bracket(items[0], items[1], opener)
            if bracket is not None:
                return bracket
        return ast.Tuple(tuple(items), span=self._span(opener))

    def _poisson_bracket(self, f: ast.Expr, g: ast.Expr, opener: Token) -> ast.Expr | None:
        """``\\{f, g\\}_{q, p}``, the Poisson bracket with respect to the canonical pair
        ``(q, p)``, as the derivatives it stands for, ``∂f/∂q ∂g/∂p - ∂f/∂p ∂g/∂q``.
        ``None``, having read nothing, where no pair is written as its subscript: without
        one, ``\\{a, b\\}`` is a pair, or a set."""
        if not self.check(K.UNDERSCORE) or self.peek().space_before:
            return None

        def pair() -> tuple[ast.Expr, ast.Expr]:
            self.advance()  # _
            brace = self.expect(K.LBRACE)
            with self._nested():
                q = self.expression()
                self.expect(K.COMMA)
                p = self.expression()
                self.expect(K.RBRACE, opener=brace)
            return q, p

        variables, _ = self._attempt(pair)
        if variables is None:
            return None
        q, p = variables
        span = self._span(opener)

        def derivative(of: ast.Expr, by: ast.Expr) -> ast.Expr:
            return ast.Derivative(of, ((by, 1),), True, span=span)

        def product(left: ast.Expr, right: ast.Expr) -> ast.Expr:
            return ast.Binary("*", left, right, implicit=True, span=span)

        first = product(derivative(f, q), derivative(g, p))
        second = product(derivative(f, p), derivative(g, q))
        return ast.Binary("-", first, second, span=span)

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
                return replace(first, span=self._span(opener))
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
        """A piecewise condition, optionally introduced by a word: ``\\text{if } x > 0``.
        The word may carry the sentence's punctuation, as in ``\\text{otherwise.}``"""
        token = self.peek()
        word = token.value.strip().rstrip(".,;").strip().lower()
        if token.kind in (K.IDENT, K.STRING) and word in _OTHERWISE:
            self.advance()
            return None
        if token.kind in (K.IDENT, K.STRING) and word in _CASE_WORDS:
            self.advance()
        return self.expression(Power.SUBSTITUTION)

    def _at_brace_cases(self, opener: Token) -> bool:
        """Whether ``opener`` starts cases written the old way, as an array in a brace
        that only opens: ``\\left\\{ \\begin{array}{ll} ... \\end{array} \\right.``"""
        begin = self.peek()
        if not (
            _LEFT_BRACE.fullmatch(opener.span.text)
            and begin.kind is K.BEGIN
            and begin.value in ("array", "matrix")
        ):
            return False
        index, depth = self._significant(self.pos), 0
        while True:  # to the \end{array}
            kind = self.tokens[index].kind
            if kind is K.EOF:
                return False
            if kind is K.BEGIN:
                depth += 1
            elif kind is K.END:
                depth -= 1
            index += 1
            if depth == 0:
                break
        while self.tokens[index].kind is K.NEWLINE:
            index += 1
        closer = self.tokens[index]
        return closer.kind is K.RPAREN and _RIGHT_DOT.fullmatch(closer.span.text) is not None

    def _absolute_value(self) -> ast.Abs:
        opener = self.advance()
        with self._nested():
            self._abs_depth = 1
            operand = self.expression()
            self.expect(K.BAR, opener=opener)
        return ast.Abs(operand, span=self._span(opener))

    def _bar(self, expr: ast.Expr, *, enclosed: bool = False) -> ast.Expr:
        """The evaluation bar after ``expr``: substitutions, ``|_{x = 2, y = 1}``, made at
        once, or limits, ``|_0^1`` and ``|_{x=0}^{x=1}``. The bar ``\\right|`` writes,
        ``enclosed``, evaluates what it and ``\\left.`` enclose, and one substitution for a
        name is its value at that point, ``\\left. F \\right|_{x=a}``, as a point written
        alone is, ``\\left. F \\right|_a``."""
        self.advance()  # the bar
        lower, upper = self._bar_scripts()
        point, *others = lower
        if enclosed and upper is None and not others and not isinstance(point, ast.Equation):
            return ast.Evaluation(expr, point, None, None, span=self._span(expr.span))
        return self._evaluated(expr, lower, upper, enclosed=enclosed)

    def _evaluated(
        self,
        expr: ast.Expr,
        lower: tuple[ast.Expr, ...],
        upper: ast.Expr | None,
        *,
        enclosed: bool,
    ) -> ast.Substitution | ast.Evaluation:
        """``expr`` evaluated as the scripts of a bar or a bracket say: between limits, each
        written alone or with the variable, ``_{x=a}``; or, without a superscript, with
        substitutions, each written with what it substitutes for."""
        span = self._span(expr.span)
        if upper is None:
            bindings = []
            for binding in lower:
                if not isinstance(binding, ast.Equation):
                    raise ParseError("expected a substitution such as x = 2", span=binding.span)
                bindings.append((binding.lhs, binding.rhs))
            (target, point), *others = bindings
            if enclosed and not others and isinstance(target, ast.Name):
                return ast.Evaluation(expr, point, None, target, span=span)
            return ast.Substitution(expr, tuple(bindings), span=span)
        if len(lower) != 1:
            raise ParseError(
                "an evaluation between limits has one lower limit",
                span=lower[1].span,
                hint="write the limits alone, as in |_0^1, or with the variable: |_{x=0}^{x=1}",
            )
        bottom = lower[0]
        variable: ast.Expr | None = None
        if isinstance(bottom, ast.Equation):
            variable, bottom = bottom.lhs, bottom.rhs
        if isinstance(upper, ast.Equation):
            if variable is not None and variable != upper.lhs:
                raise ParseError(
                    f"the limits of an evaluation are of one variable, not of "
                    f"{variable.span.text} and {upper.lhs.span.text}",
                    span=upper.lhs.span,
                )
            variable, upper = upper.lhs, upper.rhs
        return ast.Evaluation(expr, bottom, upper, variable, span=span)

    def _bar_scripts(self) -> tuple[tuple[ast.Expr, ...], ast.Expr | None]:
        """The scripts of an evaluation bar or bracket, a subscript and maybe a superscript,
        in either order: what the subscript holds, which may be several substitutions, as in
        ``_{x = 1, y = 2}``, and what the superscript holds."""
        lower: tuple[ast.Expr, ...] | None = None
        upper: ast.Expr | None = None
        while (marker := self.peek()).kind in (K.UNDERSCORE, K.CARET):
            if marker.kind is K.UNDERSCORE and lower is None:
                lower = self._bar_script(marker)
            elif marker.kind is K.CARET and upper is None:
                held = self._bar_script(marker)
                if len(held) > 1:
                    raise ParseError("an upper limit is one value", span=held[1].span)
                upper = held[0]
            else:
                break
        if lower is None:
            raise ParseError(
                "an evaluation bar needs a subscript: its point, or its lower limit",
                span=self.previous.span,
                hint="for example |_{x=2}, or |_0^1",
            )
        return lower, upper

    def _bar_script(self, marker: Token) -> tuple[ast.Expr, ...]:
        """What a script of an evaluation bar holds: in braces, one item or several, separated
        by commas or by the rows of ``\\substack``; or what one token writes, as 0 of
        ``|_0^1``."""
        self.advance()  # the marker
        if not self.check(K.LBRACE):
            script = self._script("_" if marker.kind is K.UNDERSCORE else "^", marker)
            if script.body is None:
                raise ParseError("expected a value after the evaluation bar", span=script.span)
            return (script.body,)
        opener = self.advance()
        items: list[ast.Expr] = []
        with self._nested():
            while True:
                items.append(self.expression())
                if not (self.accept(K.COMMA) or self.accept(K.ROW_SEP)):
                    break
            self.expect(K.RBRACE, opener=opener)
        return tuple(items)

    def _bar_with_limits(self) -> bool:
        """Whether the bar at the cursor has limits, a superscript as well as a subscript, as
        ``|_0^1`` and ``|^{x=1}_{x=0}`` do, rather than a point or substitutions."""
        if self.peek(1).kind is K.CARET:
            return True
        offset = 2  # what the subscript holds
        if self.peek(offset).kind is K.LBRACE:
            depth = 0
            while True:
                kind = self.peek(offset).kind
                if kind is K.EOF:
                    return False
                depth += (kind is K.LBRACE) - (kind is K.RBRACE)
                offset += 1
                if depth == 0:
                    break
        else:
            offset += 1
        return self.peek(offset).kind is K.CARET

    def _right_bar(self, token: Token) -> bool:
        """Whether a bar is the one ``\\right|`` writes, after the bracket it closes, with a
        script after it: the evaluation bar of what ``\\left.`` and ``\\right|`` enclose."""
        closer = self.previous
        return (
            closer.kind is K.RPAREN
            and closer.span.start == token.span.start
            and self.peek(1).kind in (K.UNDERSCORE, K.CARET)
        )

    def _evaluation_bracket(self, node: ast.Expr) -> ast.Evaluation | None:
        """``\\left[F\\right]_a^b`` or ``[F]_{x=a}^{x=b}``: F(b) − F(a), where square
        brackets around one expression have a subscript and a superscript; and ``[F]_{x=a}``,
        F at a. ``None``, having read nothing, for any other script, and for brackets around
        a name alone, as chemistry writes a concentration, ``[\\mathrm{A}]_0``."""
        content = _square_bracketed(node)
        if content is None or isinstance(content, ast.Name):
            return None

        def bracket() -> ast.Evaluation:
            lower, upper = self._bar_scripts()
            evaluated = self._evaluated(content, lower, upper, enclosed=True)
            if not isinstance(evaluated, ast.Evaluation):
                raise ParseError("substitutions after brackets are their scripts", span=node.span)
            return replace(evaluated, span=self._span(node.span))

        evaluated, _ = self._attempt(bracket)
        return evaluated

    # -- scripts and indices ----------------------------------------------------

    def _attach_script(self, node: ast.Expr, *, extend: bool) -> ast.Expr:
        """A script on ``node``. In LaTeX an empty script is nothing, as TeX sets it:
        ``{\\partial}_{}`` is ``\\partial``, as a macro called with nothing makes it. In
        plain text, ``x_{}`` is an error rather than the symbol ``x_``."""
        marker = self.advance()
        kind: Literal["^", "_"] = "^" if marker.kind is K.CARET else "_"
        latex = self._display > 0 or self.source.latex or "\\" in node.span.text
        if latex and self.check(K.LBRACE) and self.peek(1).kind is K.RBRACE:
            self.advance()
            self.advance()
            return node
        if _a_number_set(node) and self.check(K.LBRACE) and self.peek(1).kind in _COMPARISONS:
            script = self._bound_script(node, kind, marker)  # \mathbb{Z}_{>0}
        else:
            script = self._script(kind, marker)
        if extend and isinstance(node, ast.Scripted):
            base, scripts = node.base, (*node.scripts, script)
        else:
            base, scripts = node, (script,)
        return ast.Scripted(base, scripts, span=node.span.to(script.span))

    def _script(self, kind: Literal["^", "_"], marker: Token) -> ast.Script:
        token = self.peek()
        if token.kind is K.LBRACE:
            return self._braced_script(kind, marker)
        if token.kind is K.STAR:  # r^*, \phi_*: a name decoration, not a power
            self.advance()
            star = ast.Name("*", verbatim=True, span=token.span)
            return ast.Script(kind, star, None, span=marker.span.to(token.span))
        if token.kind is K.IDENT or (token.kind is K.COMMAND and token.value in _LABEL_COMMANDS):
            self.advance()
            verbatim = token.verbatim or token.kind is K.COMMAND  # \lambda_\max: a label
            name = ast.Name(token.value, verbatim, span=token.span)
            labels = [token.value] if verbatim else split_index_labels(token.value)
            items = self._items_from_labels(labels, token)
            indices = ast.IndexList(items, span=token.span)
            return ast.Script(kind, name, indices, span=marker.span.to(token.span))
        if token.kind is K.NUMBER:
            self.advance()
            number = ast.Number(token.value, span=token.span)
            return ast.Script(
                kind, number, self._numeric_indices(token), span=marker.span.to(token.span)
            )
        if token.kind in (K.PLUS_MINUS, K.MINUS_PLUS) or (
            token.kind in (K.MINUS, K.PLUS) and not self._signs_operand(kind)
        ):  # r_+, i^- and h_+ \cos\omega t: a label, not a sign
            self.advance()
            sign = ast.Name(token.value, verbatim=True, span=token.span)
            return ast.Script(kind, sign, None, span=marker.span.to(token.span))
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

    def _bound_script(
        self, numbers: ast.Expr, kind: Literal["^", "_"], marker: Token
    ) -> ast.Script:
        """``\\mathbb{Z}_{>0}``: a set of numbers with a bound in its script, the numbers
        of the set that keep it, read as the comparison ``ℤ > 0``."""
        opener = self.advance()
        relation = self.advance()
        bound = self.expression(Power.COMPARISON)
        self.expect(K.RBRACE, opener=opener)
        span = relation.span.to(bound.span)
        comparison = ast.Comparison((numbers, bound), (_COMPARISONS[relation.kind],), span=span)
        return ast.Script(kind, comparison, None, span=self._span(marker))

    def _signs_operand(self, kind: Literal["^", "_"]) -> bool:
        """Whether the ``+`` or ``-`` after a script marker signs what follows it, as plain
        text writes ``x^-1`` and ``e^-x``, rather than being the whole script, a label, as
        TeX sets ``r_+``, ``h_+ \\cos\\omega t``, ``x^+ x^-`` and ``e^+e^-``. It signs a
        number, as in ``x^-1`` and ``a_-1``; in a superscript, it also signs an operand
        written against it, unless that is a name labelled with a sign of its own."""
        follower = self.peek(1)
        if not self._starts_operand(follower):
            return False
        if follower.kind is K.NUMBER:
            return True
        if kind == "_" or follower.space_before:
            return False
        marker, label = self.peek(2), self.peek(3)
        return not (
            follower.kind is K.IDENT
            and marker.kind in (K.CARET, K.UNDERSCORE)
            and not marker.space_before
            and label.kind in (K.PLUS, K.MINUS)
        )

    def _braced_label(self) -> bool:
        """Whether the braces of a script hold a label alone: a sign, as the + of the
        pseudoinverse ``A^{+}`` and of ``r_{+}``, or ``\\max`` or ``\\min``, as in
        ``\\lambda_{\\max}``."""
        token, closer = self.peek(), self.peek(1)
        signs = (K.PLUS, K.MINUS, K.STAR, K.PLUS_MINUS, K.MINUS_PLUS)
        label = token.kind in signs or (token.kind is K.COMMAND and token.value in _LABEL_COMMANDS)
        return label and closer.kind is K.RBRACE

    def _braced_script(self, kind: Literal["^", "_"], marker: Token) -> ast.Script:
        """A braced script, read both as an index list and as an expression."""
        what = "superscript" if kind == "^" else "subscript"
        base = _written_before(marker)
        opener = self.advance()
        if self.check(K.RBRACE):
            closer = self.advance()
            raise ParseError(
                f"empty {what}",
                span=opener.span.to(closer.span).labelled("nothing in the braces"),
                hint=f"write what the {what} holds, as in {base}{kind}{{0}}, or remove it",
            )
        if not self._closes(self.pos - 1):
            line = self.source.text[marker.span.start :].split("\n", 1)[0].rstrip()
            example = f", as in {base}{line}}}" if len(base + line) < 40 else ""
            raise ParseError(
                "unclosed '{'",
                span=opener.span.labelled("this '{' is never closed"),
                hint=f"add '}}' where the {what} ends{example}",
            )
        if self._braced_label():  # A^{+}, \lambda_{\max}: a label, as A^+ and \lambda_\max are
            label = self._script(kind, marker)
            closer = self.advance()
            return replace(label, span=marker.span.to(closer.span))
        start = (self.pos, self.previous)
        indices, index_error = self._attempt(lambda: self._index_list(opener))
        after_indices = (self.pos, self.previous)
        self.pos, self.previous = start

        def braced_expression() -> ast.Expr:
            in_script, self._in_script = self._in_script, True
            try:
                with self._nested():
                    body = self.expression()
                    self.expect(K.RBRACE, opener=opener)
            finally:
                self._in_script = in_script
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

    def _closes(self, index: int) -> bool:
        """Whether the brace at token ``index`` is closed before the input ends."""
        depth = 0
        for token in self.tokens[index:]:
            if token.kind is K.LBRACE:
                depth += 1
            elif token.kind is K.RBRACE:
                depth -= 1
                if depth == 0:
                    return True
        return False

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
                elif token.kind is K.LBRACE:  # R_{\rho\sigma{\mu\nu}}
                    group = self.advance()
                    while not self.check(K.RBRACE):
                        if self.check(K.EOF):
                            self.expect(K.RBRACE, opener=group)
                        items.extend(self._index_items(derivative))
                    self.advance()
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
        if token.kind is K.ELLIPSIS:  # T^{\mu_1 \cdots \mu_k}
            self.advance()
            return [ast.IndexItem("⋯", derivative=derivative, span=token.span)]
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
        if suffix := self._label_suffix():
            last = items[-1]
            items[-1] = replace(last, label=f"{last.label}{suffix}", span=self._span(last.span))
        if self.peek().kind in (K.EQUAL, K.COLON):
            self.advance()
            items[-1] = replace(items[-1], value=self._index_value())
        return [replace(item, derivative=derivative) for item in items]

    def _label_suffix(self) -> str:
        """A subscript or primes on an index label: ``\\mu_1``, ``\\nu'``, ``\\mu_{12}'``."""
        suffix = ""
        while True:
            token = self.tokens[self.pos]
            follower = self.tokens[self.pos + 1] if token.kind is not K.EOF else token
            if token.kind in (K.PRIME, K.STAR):  # \mu', \phi_{t*}
                suffix += "'" if token.kind is K.PRIME else "*"
                self.advance()
            elif token.kind is K.UNDERSCORE and follower.kind in (K.NUMBER, K.IDENT):
                self.advance()
                suffix += "_" + self.advance().value
            elif token.kind is K.UNDERSCORE and follower.kind is K.LBRACE:
                owner = self.advance()
                group = self._raw_group(K.LBRACE, K.RBRACE, owner)
                inner = self.source.text[group[0].span.end : group[-1].span.start].strip()
                suffix += "_{" + inner + "}"
            else:
                return suffix

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
        if name in ("Box", "square"):
            self.advance()
            return ast.Operator("box", span=token.span)
        if name == "pounds":
            self.advance()
            return ast.Operator("lie", span=token.span)
        if name in ("pdv", "dv"):
            return self._physics_derivative()
        if name in _BINOMIALS:  # \binom{n}{k}, and Karamata's {n \brace k} and {n \brack k}
            self.advance()
            n = self._brace_argument(token)
            k = self._brace_argument(token)
            func = ast.Name(_BINOMIALS[name], span=token.span)
            return ast.Call(func, (n, k), span=self._span(token))
        if name in ("lfloor", "lceil"):
            return self._floor_or_ceiling()
        if name in _MODULI:  # with nothing before it to take the remainder of
            raise ParseError(
                f"\\{name} needs what it divides before it",
                span=token.span,
                hint=r"for example 12! \bmod 13, or a \equiv b \pmod{m}",
            )
        if name in vocab.FUNCTION_COMMANDS:
            return self._function()
        if name in vocab.ACCENT_COMMANDS:
            return self._accent()
        if name in _BOUNDS:
            return self._optimum(self.advance(), name)  # \sup_{x \in S} f(x)
        if name == "underset" and (beneath := self._beneath()) is not None:
            return beneath  # \underset{x}{\min} f(x)
        if name in dirac.COMMANDS:  # \ket{\psi}, \braket{\phi|\psi}
            return dirac.command(self)
        notes = self._expanded_from(token)
        if (unsupported := vocab.UNSUPPORTED_COMMANDS.get(name)) is not None:
            message, hint = unsupported
            raise ParseError(message, span=token.span, hint=hint, notes=notes)
        close = difflib.get_close_matches(name, _KNOWN_COMMANDS, n=1, cutoff=_CLOSE)
        raise ParseError(
            f"unknown LaTeX command \\{name}",
            span=token.span,
            hint=f"did you mean \\{close[0]}?" if close else None,
            notes=notes,
        )

    def _expanded_from(self, token: Token) -> tuple[str, ...]:
        """A note, for a token that a document's macro expanded to, that says which
        macro, and where it is defined: ``\\Gam, defined on line 7, expands to \\Gama``."""
        written = self.source.expansion_at(token.span.start)
        macro = _MACRO_NAME.match(written) if written is not None else None
        if macro is None or macro.group() == token.span.text:
            return ()
        defined = dict(self.source.definitions).get(macro.group()[1:])
        where = f", defined on {defined}," if defined else ""
        return (f"{macro.group()}{where} expands to {token.span.text}",)

    def _argument_tokens(self, command: Token, what: str = "argument") -> list[Token]:
        """The raw tokens of a command argument: a braced group, a parenthesised
        group, or (as LaTeX reads it) one character: ``\\frac12`` is one half."""
        self._skip_newlines()
        token = self.tokens[self.pos]
        self._group = token.span
        if token.kind is K.LBRACE:
            return self._raw_group(K.LBRACE, K.RBRACE, command)[1:-1]
        if token.kind is K.LPAREN:
            return self._raw_group(K.LPAREN, K.RPAREN, command)
        sign = not command.span.text.startswith("\\")  # √16 is the root of 16, \sqrt16 not
        if (
            token.kind in (K.NUMBER, K.IDENT)
            and len(token.value) > 1
            and not _is_latex(token)
            and not (sign and token.kind is K.NUMBER)
        ):
            first, rest = _split_first_character(token, self.source)
            self.tokens[self.pos] = rest
            self.previous = first
            return [first]
        if token.kind in (K.NUMBER, K.IDENT):
            return [self.advance()]
        example = _ARGUMENT_EXAMPLES.get(command.value, f"\\{command.value}{{x}}")
        raise ParseError(
            f"\\{command.value} is missing its {what}",
            span=token.span,
            hint=f"write it in braces, as in {example}",
        )

    def _raw_group(self, open_kind: TokenKind, close_kind: TokenKind, owner: Token) -> list[Token]:
        """Consume a balanced group, returning its tokens including the brackets."""
        opener = self.tokens[self.pos]
        depth, collected = 0, []
        while True:
            token = self.tokens[self.pos]
            if token.kind is K.EOF:
                close = close_kind.value
                raise ParseError(
                    f"unclosed '{opener.span.text}' in the argument of \\{owner.value}",
                    span=opener.span.labelled(f"this '{opener.span.text}' is never closed"),
                    hint=f"add '{close}' where the argument ends",
                )
            self.pos += 1
            collected.append(token)
            if token.kind is open_kind:
                depth += 1
            elif token.kind is close_kind:
                depth -= 1
                if depth == 0:
                    self.previous = token
                    self._group = opener.span.to(token.span)
                    return collected

    def _subparse(self, tokens: list[Token], what: str, owner: Token) -> ast.Expr:
        """Parse a token list captured by :meth:`_argument_tokens` as one expression."""
        tokens = [t for t in tokens if t.kind is not K.NEWLINE]
        if not tokens:
            span = owner.span if self._group is None else self._group
            raise ParseError(
                f"the {what} of \\{owner.value} is empty", span=span.labelled(f"empty {what}")
            )
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
        numerator = self._subparse(
            self._argument_tokens(command, "numerator"), "numerator", command
        )
        denominator = self._subparse(
            self._argument_tokens(command, "denominator"), "denominator", command
        )
        quotient = _quotient(numerator, denominator, self._span(command), leibniz=True)
        if isinstance(quotient, ast.Derivative) and quotient.expr is None:
            # As SymPy prints a derivative, \frac{d}{d r} A{\left(r \right)}, it is of the
            # function after it alone, not of the rest of the term.
            operand = self._typeset_operand()
            if operand is not None:
                return replace(quotient, expr=operand, span=self._span(command))
        return quotient

    def _typeset_operand(self) -> ast.Expr | None:
        """A function applied as SymPy typesets it, maybe raised to a power, if one is
        next: ``A{\\left(r \\right)}`` or ``A^{2}{\\left(r \\right)}``."""
        if self.peek().kind is not K.IDENT:
            return None
        before = (self.pos, self.previous)
        node, _ = self._attempt(lambda: self._postfix(self._atom()))
        call = node.base if isinstance(node, ast.Scripted) else node
        if isinstance(call, ast.Call) and _TYPESET_ARGUMENTS.search(call.span.text):
            return node
        self.pos, self.previous = before  # something else, read as it always was
        return None

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
        """The tokens of a script of a big operator, a limit or an optimum: in braces, or
        in round brackets, as AsciiMath groups them, ``lim_(x->0)`` and ``sum_(n=1)^oo``;
        a signed token; or one token."""
        token = self.tokens[self.pos]
        self._group = token.span
        if token.kind is K.LBRACE:
            return self._raw_group(K.LBRACE, K.RBRACE, command)[1:-1]
        if token.kind is K.LPAREN:
            return self._raw_group(K.LPAREN, K.RPAREN, command)[1:-1]
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
        if lower is not None and upper is None and (given := self._index_set(lower, command)):
            body = self.expression(Power.SUM)  # \sum_{d \mid n}, over the values a condition gives
            return ast.BigOperator(op, given, None, None, body, span=self._span(command))
        if lower is not None and upper is None and not any(t.kind is K.EQUAL for t in lower):
            variable = self._subparse(lower, "index", command)  # \sum_b: over an index
            body = self.expression(Power.SUM)
            return ast.BigOperator(op, variable, None, None, body, span=self._span(command))
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

    def _index_set(self, tokens: list[Token], command: Token) -> ast.Expr | None:
        r"""The condition under a big operator that gives its index its values, where it is
        written with a bar or in rows (design/notation.md, 19): ``d \mid n``, the divisors
        of n, or ``\substack{1 \le k \le n \\ \gcd(k, n) = 1}``, a condition in each row,
        all of which hold. ``None`` for a script read otherwise, as an index or a limit."""
        rows = _rows(tokens)
        bars = [_divides_at(row) for row in rows]
        if len(rows) < 2 and bars[0] is None:
            return None
        conditions: list[ast.Expr] = []
        for row, bar in zip(rows, bars, strict=True):
            if bar is None:
                conditions.append(self._subparse(row, "condition", command))
                continue
            left = self._subparse(row[:bar], "condition", command)
            right = self._subparse(row[bar + 1 :], "condition", command)
            written = row[bar].span.text
            conditions.append(
                ast.Relation(left, "∣", right, written, span=left.span.to(right.span))
            )
        condition = conditions[0]
        for other in conditions[1:]:
            condition = ast.Binary("and", condition, other, span=condition.span.to(other.span))
        return condition

    def _integral(self) -> ast.Integral:
        command = self.advance()
        count = _INTEGRALS[command.value]
        lower_tokens, upper_tokens = self._limits(command)
        lower = self._subparse(lower_tokens, "lower limit", command) if lower_tokens else None
        upper = self._subparse(upper_tokens, "upper limit", command) if upper_tokens else None
        self._integral_depth += 1
        try:
            variables: list[tuple[ast.Expr, ast.Expr | None]] = []
            start = self.pos, self.previous
            # \int dx f(x); as physics writes it, \int dx\, dy\, f(x, y) integrates over both
            while self._at_differential():
                variables.append(self._integration_variable())
            if not variables:
                # \int x^2 + 1 \, dx runs to its differential, but with the differential in
                # a numerator, \int \frac{dr}{r^2} - \pi ends with its term
                integrand = self.expression(Power.SUM)
                if not self._at_differential() and not _take_differentials(integrand)[1]:
                    integrand = self._operations(integrand, Power.COMPARISON)
            elif self._starts_operand(self.peek()):
                integrand = self.expression(Power.SUM)
            elif count == 1 and lower is not None and upper is None:
                # \int_M d\omega, the integral of the form d\omega over a region
                (self.pos, self.previous), variables = start, []
                integrand = self.expression(Power.COMPARISON)
            else:  # \int_0^L dx: of 1
                differentials = self._span(self.tokens[self._significant(start[0])])
                integrand = ast.Number("1", span=differentials)
            while len(variables) < count and self._at_differential():
                variables.append(self._integration_variable())
            if len(variables) < count and (evaluated := self._evaluated_integrand(integrand)):
                integrand = evaluated  # \int \left. f \right|_{r = r_+} d\theta
                while len(variables) < count and self._at_differential():
                    variables.append(self._integration_variable())
            if len(variables) < count:  # \int \frac{dr}{1 - 2M/r}: the differential is on top
                integrand, found = _take_differentials(integrand)
                variables.extend(found)
            if len(variables) < count and lower is not None and upper is None:
                # \int_M \omega: a form over a region, with no differential, and
                # \iint_{x^2 + y^2 \le 1} f \, dA, over a region with its element of area
                region = tuple(
                    ast.Bound(variable, dimension=dimension, span=variable.span)
                    for variable, dimension in variables
                )
                return ast.Integral(integrand, region, lower, span=self._span(command))
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
            ast.Bound(variable, lower, upper, dimension, span=variable.span)
            for variable, dimension in variables
        )
        return ast.Integral(integrand, bounds, span=self._span(command))

    def _at_differential(self) -> bool:
        """Is the next token a differential: ``dx``, ``d x``, ``d\\theta``, ``\\mathrm{d}x``,
        ``{\\rm d}x``, or a volume element such as ``d^4x`` or ``d^{4}x``?"""
        token = self.peek()
        if token.kind is K.LBRACE:  # {\rm d}x
            d, closer, after = self.peek(1), self.peek(2), self.peek(3)
            return (
                d.kind is K.IDENT
                and d.value == "d"
                and closer.kind is K.RBRACE
                and after.kind is K.IDENT
            )
        if token.kind is not K.IDENT:
            return False
        if token.value == "d" and self._at_volume_element():
            return True
        if token.value == "d":
            follower = self.peek(1)
            return follower.kind is K.IDENT or (
                follower.kind is K.LBRACE and not follower.space_before
            )
        return not token.verbatim and _DIFFERENTIAL.fullmatch(token.value) is not None

    def _at_volume_element(self) -> bool:
        """``d^4x``, ``d^4 x``, ``d^{4}x``, ``d^nx``: a differential over several dimensions."""
        caret, dimension, after = self.peek(1), self.peek(2), self.peek(3)
        if caret.kind is not K.CARET:
            return False
        if dimension.kind is K.LBRACE:  # d^{4}x
            closer, variable = self.peek(4), self.peek(5)
            return (
                after.kind in (K.NUMBER, K.IDENT)
                and closer.kind is K.RBRACE
                and variable.kind is K.IDENT
            )
        if dimension.kind is K.NUMBER:
            return after.kind is K.IDENT
        return dimension.kind is K.IDENT and (
            len(dimension.value) == 2 or (len(dimension.value) == 1 and after.kind is K.IDENT)
        )

    def _evaluated_integrand(self, integrand: ast.Expr) -> ast.Expr | None:
        """An evaluation bar between the integrand and its differential, as in
        ``\\int \\left. \\sqrt{g} \\right|_{r = r_+} d\\theta``, evaluates the integrand: the
        integrand evaluated, when a differential follows the bar; otherwise ``None``, having
        read nothing."""
        if self._abs_depth or not (self.check(K.BAR) and self.peek(1).kind is K.UNDERSCORE):
            return None
        before = self.pos, self.previous
        evaluated, _ = self._attempt(lambda: self._bar(integrand))
        if evaluated is not None and self._at_differential():
            return evaluated
        self.pos, self.previous = before
        return None

    def _integration_variable(self) -> tuple[ast.Expr, ast.Expr | None]:
        """The variable of a differential, and its dimension if it is a volume element."""
        if self.peek().value != "d" or not self._at_volume_element():
            return self._differential(), None
        self.advance()  # d
        self.advance()  # ^
        braced = self.accept(K.LBRACE)
        token = self.advance()
        if braced:
            self.expect(K.RBRACE)
        dimension: ast.Expr
        if token.kind is K.NUMBER:
            dimension = ast.Number(token.value, span=token.span)
        elif braced:  # d^{n}x
            dimension = ast.Name(token.value, token.verbatim, span=token.span)
        else:
            first, rest = token.value[0], token.value[1:]
            dimension = ast.Name(first, span=token.span)
            if rest:  # d^nx
                primes = self._primes()
                return ast.Name(rest + primes, span=self._span(token)), dimension
        name = self.advance()
        primes = self._primes()  # d^3x', over the primed coordinates
        return ast.Name(name.value + primes, name.verbatim, span=self._span(name)), dimension

    def _differential(self) -> ast.Expr:
        if self.check(K.LBRACE):  # {\rm d}x
            self.advance()
            token = self.advance()
            self.advance()
        else:
            token = self.advance()
        if token.value != "d":  # dx, and dt' whose variable is t'
            primes = self._primes()
            span = Span(self.source, token.span.start + 1, self.previous.span.end)
            return self._scripted(ast.Name(token.value[1:] + primes, span=span))
        if self.check(K.LBRACE):
            opener = self.advance()
            with self._nested():
                variable = self.expression()
                self.expect(K.RBRACE, opener=opener)
            return variable
        name = self.advance()
        primes = self._primes()
        return self._scripted(ast.Name(name.value + primes, name.verbatim, span=self._span(name)))

    def _scripted(self, variable: ast.Name) -> ast.Expr:
        """A differential's variable with the scripts written against it: ``dx^\\mu`` is the
        differential of ``x^\\mu``, and ``dp^0`` of ``p^0``, as a line integral writes them."""
        node: ast.Expr = variable
        while self.peek().kind in (K.CARET, K.UNDERSCORE) and not self.peek().space_before:
            node = self._attach_script(node, extend=True)
        return node

    def _primes(self) -> str:
        """Primes directly after a name, as in ``dt'``."""
        primes = ""
        while self.tokens[self.pos].kind is K.PRIME:
            primes += "'"
            self.advance()
        return primes

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
            order = _order(self.expect(K.NUMBER, after="'['"))
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
        r"""``\sin x``, ``\sin(x)``, ``\sin^2\theta``, ``\sin^{-1} x``, ``\log_{2} x``.
        Only ``\log`` takes a subscript, its base, and ``\max`` and ``\min`` the variable
        they are over, ``\max_x``: ``\det_g`` is an error."""
        command = self.advance()
        name = vocab.FUNCTION_COMMANDS[command.value]
        if command.value == "arg" and self.check(K.COMMAND) and self.peek().value in _OPTIMA:
            self.advance()  # \arg\min_w: where the minimum is, not the argument of a number
            return self._optimum(command, "arg" + self.previous.value)
        power: ast.Expr | None = None
        base: ast.Expr | None = None
        while self.peek().kind in (K.CARET, K.UNDERSCORE):
            if self.check(K.UNDERSCORE) and command.value in _OPTIMA:
                return self._optimum(command, command.value)  # \max_v: over v
            marker = self.advance()
            script = self._script("^" if marker.kind is K.CARET else "_", marker)
            if script.body is None:
                raise ParseError("expected an expression", span=script.span)
            if marker.kind is K.CARET:
                power = script.body
            elif command.value == "log":
                base = script.body
            else:
                raise ParseError(
                    f"a subscript on \\{command.value} is not supported",
                    span=command.span.to(script.span),
                    hint="only \\log takes a subscript, its base, as in \\log_{2} x",
                )
        if _is_minus_one(power) and command.value in vocab.INVERSE_FUNCTIONS:
            name, power = vocab.INVERSE_FUNCTIONS[command.value], None
        if self.peek().kind not in (K.LPAREN, K.LBRACKET, K.LBRACE) and not (
            self._starts_operand(self.peek())
        ):
            written = self._span(command).text
            example = "\\log_{2} x" if base is not None else f"{written} x or {written}(x)"
            raise ParseError(
                f"{written} needs an argument",
                span=self.peek().span.labelled(f"expected what {written} acts on"),
                hint=f"write it after the function, as in {example}",
            )
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

    # -- optima -----------------------------------------------------------------

    def _optimum(self, start: Token, op: str) -> ast.Optimum:
        """An optimum, as books write it: ``\\max_{x} f(x)``, ``\\arg\\min_{w} f(w)``,
        ``\\operatorname*{arg\\,max}_k |X_k|``, ``\\sup_{0 \\le x < 1} x^n``: the variables
        beneath the operator, with the set they are in or the conditions they keep, what it
        is of after it, and the constraints after "subject to" (:meth:`_subject_to`)."""
        if not self.check(K.UNDERSCORE):
            raise self._without_variable(start, op)
        self.advance()
        token = self.tokens[self.pos]
        if token.kind not in (K.LBRACE, K.IDENT, K.COMMAND):
            raise self._without_variable(start, op)
        under = self._script_tokens(start)
        variables, constraints = self._beneath_optimum(under, start, op)
        return self._optimum_after(start, op, variables, constraints)

    def _without_variable(self, start: Token, op: str) -> ParseError:
        """The error for an optimum written without the variable it is over, as ``\\arg\\max f``."""
        written = self._span(start).text
        extreme = "maximum" if op.endswith(("max", "sup")) else "minimum"
        what = {
            "sup": "the supremum",
            "inf": "the infimum",
            "argmax": "where a maximum is",
            "argmin": "where a minimum is",
        }.get(op, f"a {extreme}")
        return ParseError(
            f"{written}, {what}, needs the variable it is over, beneath it",
            span=self._span(start),
            hint=f"write it as {written}_{{x}} f(x), or {written}_{{x \\in S}} f(x)",
        )

    def _beneath(self) -> ast.Optimum | None:
        """``\\underset{x \\in S}{\\min} f(x)``: an optimum whose variable ``\\underset`` sets
        beneath it; ``None``, having read nothing, where what is set beneath is no optimum."""
        before = self.pos, self.previous
        command = self.advance()
        under = self._argument_tokens(command, "subscript")
        operator = _optimum_operator(self._argument_tokens(command, "operator"))
        if operator is None:
            self.pos, self.previous = before
            return None
        variables, constraints = self._beneath_optimum(under, command, operator)
        return self._optimum_after(command, operator, variables, constraints)

    def _optimum_after(
        self, start: Token, op: str, variables: list[ast.Expr], constraints: list[ast.Expr]
    ) -> ast.Optimum:
        """The rest of an optimum, after its variables: what it is of, and its constraints.
        It is of the whole sum it begins, up to a relation or "subject to", as
        ``\\min_x \\; x^2 - 4x + 7`` is; where it is a factor, or has a sign before it, as in
        ``2\\max_x f(x) + 1``, of that factor alone."""
        whole = self._power < Power.SUM or self._subject_to_ahead()
        token = self.peek()
        if not (self._starts_operand(token) or token.kind in _SIGNS):
            written = self._span(start).text
            raise ParseError(
                f"{written} needs what it is the {'maximum' if 'max' in op else 'minimum'} "
                "of, after it",
                span=token.span.labelled("expected an expression"),
                hint=f"write it after the operator, as in {written} f(x)",
            )
        self._bodies += 1
        try:
            body = self.expression(Power.COMPARISON if whole else Power.SUM)
        finally:
            self._bodies -= 1
        if isinstance(body, ast.List) and len(body.items) == 1:
            body = body.items[0]  # \max_c \left[\ln c + \beta V(k - c)\right]: brackets group
        constraints = [*constraints, *self._subject_to()]
        kind = cast(Literal["max", "min", "argmax", "argmin", "sup", "inf"], op)
        return ast.Optimum(kind, tuple(variables), body, tuple(constraints), span=self._span(start))

    def _beneath_optimum(
        self, tokens: list[Token], owner: Token, op: str
    ) -> tuple[list[ast.Expr], list[ast.Expr]]:
        """The variables an optimum is over, and the conditions beneath it: ``x``, ``x, y``,
        ``x \\in S``, ``x, y \\in \\mathbb{R}``, the ``x`` of ``0 \\le x \\le 3`` and of
        ``x \\geq 0``; and, in further rows of ``\\substack``, conditions alone."""
        variables: list[ast.Expr] = []
        constraints: list[ast.Expr] = []
        written = owner.span.text
        for number, row in enumerate(_split_tokens(tokens, K.ROW_SEP)):
            pending: list[ast.Expr] = []
            for item_tokens in _split_tokens(row, K.COMMA):
                item = self._subparse(item_tokens, "variable", owner)
                if number == 0 and _written_variable(item):
                    pending.append(item)
                    continue
                if number > 0 and isinstance(item, ast.Comparison | ast.Equation | ast.Relation):
                    constraints.append(item)
                    continue
                variable = _variable_of(item)
                if variable is None:
                    raise ParseError(
                        f"beneath {written}, {item.span.text} is no variable: write the "
                        "variables an optimum is over, with the set they are in or the "
                        "conditions they keep",
                        span=item.span,
                        hint=f"as in {written}_{{x}}, {written}_{{x \\in S}} or "
                        f"{written}_{{0 \\le x \\le 1}}",
                    )
                variables.extend([*pending, variable])
                constraints.append(_on_each([*pending, variable], item))
                pending = []
            variables.extend(pending)
        if not variables:
            raise self._without_variable(owner, op)
        return variables, constraints

    def _subject_to(self) -> list[ast.Expr]:
        """The constraints of an optimum after "subject to" or "s.t.", on its line or the
        lines after it, as books set them, separated by commas, wide spaces and line
        breaks: ``\\text{subject to} \\quad x_1 \\leq 4, \\quad x_1, x_2 \\geq 0``. A name
        before a comma shares the relation after it, so that ``x_1, x_2 \\geq 0`` holds of
        both."""
        index = self._after_breaks(self.pos)
        size = self._subject_to_word(index)
        if not size:
            return []
        self.pos = index + size
        self.previous = self.tokens[self.pos - 1]
        constraints: list[ast.Expr] = []
        pending: list[ast.Expr] = []
        while True:
            item = self.expression()
            if _written_variable(item):
                pending.append(item)
            elif (
                pending and (first := _first_operand(item)) is not None and _written_variable(first)
            ):
                constraints.append(_on_each([*pending, first], item))
                pending = []
            else:
                if pending:
                    raise _no_constraint(pending[0])
                constraints.append(item)
            if self._next_constraint():
                continue
            if pending:
                raise _no_constraint(pending[0])
            return constraints

    def _next_constraint(self) -> bool:
        """Go to the next constraint of an optimum, if one follows: after a comma, a word
        such as ``\\text{and}``, or wide spacing before a relation; on the next row of an
        alignment, set under the one before it, or after "s.t." again."""
        if self.check(K.COMMA) and not self._at_punctuation():
            self.advance()
            if self._at_connective(self.peek()):  # x \geq 0, \text{and} y \geq 0
                self.advance()
            return True
        if self._at_connective(self.peek()):
            self.advance()
            return True
        if self._at_wide_space() and self._relates(self._significant(self.pos)):
            return True
        index = self._after_breaks(self.pos)
        if index == self.pos:
            return False
        if size := self._subject_to_word(index):
            self.pos = index + size
            self.previous = self.tokens[self.pos - 1]
            return True
        if self.tokens[index].aligned and self._relates(index):  # & h(x) = 0, beneath
            self.pos = index
            self.previous = self.tokens[index - 1]
            return True
        return False

    def _after_breaks(self, index: int) -> int:
        """The first token after the line breaks and row separators at ``index``."""
        while self.tokens[index].kind in (K.NEWLINE, K.ROW_SEP):
            index += 1
        return index

    def _subject_to_ahead(self) -> bool:
        """Whether "subject to" follows what an optimum is of, on its line or at the start
        of the next: with no relation, comma or wide spacing before it, at the level the
        optimum is written at."""
        depth, index = 0, self.pos
        while True:
            token = self.tokens[index]
            if token.kind is K.EOF:
                return False
            if depth == 0 and self._subject_to_word(index):
                return True
            if token.kind in _OPENERS or token.kind is K.BEGIN:
                depth += 1
            elif token.kind in _CLOSERS or token.kind is K.END:
                depth -= 1
                if depth < 0:
                    return False
            elif depth == 0 and token.kind in _SEPARATORS:
                return bool(self._subject_to_word(self._after_breaks(index)))
            elif depth == 0 and (
                token.kind in _RELATIONS
                or token.kind in _COMPARISONS
                or token.kind in (K.COMMA, K.DEFINE)
                or (token.wide_space and index > self.pos)
            ):
                return False
            index += 1

    def _subject_to_word(self, index: int) -> int:
        """How many tokens the words "subject to" take at ``index``: ``\\text{subject to}``,
        ``\\text{s.t.}``, ``\\text{such that}``, ``\\mathrm{s.t.}`` and, in plain text,
        ``subject to``; 0 where they are not there."""
        token = self.tokens[index]
        if token.kind is K.STRING and token.value.strip().lower() in _SUBJECT_TO:
            return 1
        if token.kind is K.IDENT and token.verbatim and token.value == "st":
            return 1
        following = self.tokens[min(index + 1, len(self.tokens) - 1)]
        if (token.kind, token.value, following.kind, following.value) == (
            K.IDENT,
            "subject",
            K.IDENT,
            "to",
        ) and not (token.verbatim or following.verbatim):
            return 2
        letters = [(t.kind, t.value) for t in self.tokens[index : index + 4]]
        if letters[:3] == [(K.IDENT, "s"), (K.PERIOD, "."), (K.IDENT, "t")] and token.verbatim:
            return 4 if len(letters) == 4 and letters[3][0] is K.PERIOD else 3
        return 0

    def _function_arguments(self) -> tuple[ast.Expr, ...]:
        """Arguments in parentheses or square brackets (``\\exp[-x^2]``), or (as typeset
        maths reads it) the factors that follow, up to explicit spacing, an operator,
        another function, a fraction or a differential: ``\\sin\\omega t`` is sin(ωt)
        but ``\\sin\\theta\\, d\\phi`` and ``\\sin\\theta \\frac{a}{b}`` are not."""
        token = self.peek()
        if token.kind in (K.LPAREN, K.LBRACKET):
            opener = self.advance()
            args: list[ast.Expr] = []
            with self._nested():
                args.append(self.expression())
                while self.accept(K.COMMA):
                    args.append(self.expression())
                self.expect(K.RPAREN if token.kind is K.LPAREN else K.RBRACKET, opener=opener)
            return tuple(args)
        if token.kind is K.LBRACE:
            opener = self.advance()
            with self._nested():
                arg = self.expression()
                self.expect(K.RBRACE, opener=opener)
            return (arg,)
        outer, self._function_argument = self._function_argument, True
        try:
            return (self._unbracketed_argument(),)
        finally:
            self._function_argument = outer

    def _unbracketed_argument(self) -> ast.Expr:
        arg = self.expression(Power.PRODUCT)
        while True:
            token = self.peek()
            if token.explicit_space or not self._starts_operand(token):
                break
            if token.kind is K.LPAREN:  # \cos\theta (\dot\phi)^2: brackets end the argument
                break
            if token.kind is K.COMMAND and token.value in _ARGUMENT_STOPPERS:
                break
            if self._at_differential() or (token.kind is K.IDENT and token.value == "d"):
                break
            if self._hadamard(token) or self._direct_sum(token):  # \sin x \odot y
                break
            self._numbers_apart(token)
            right = self.expression(Power.PRODUCT)
            arg = ast.Binary("*", arg, right, implicit=True, span=arg.span.to(right.span))
        return arg

    # -- derivatives in Leibniz notation ----------------------------------------

    def _solidus_derivative(self, left: ast.Expr) -> ast.Expr | None:
        """Leibniz's fraction set with a solidus, as a line of text sets it, read after the
        solidus as ``\\frac`` reads it (design/notation.md, decision 39): ``d/dx``, ``dy/dx``,
        ``d(x^3)/dx``, ``d^2y/dx^2``, ``∂f/∂x``. In a product it is the fraction of the
        factors from its numerator's differential on, as ``x dy/dx`` is ``x \\frac{dy}{dx}``.
        It keeps the quotient as written, which is what it means where the program defines
        a name it is written with, as dx. ``None``, having read nothing, where either side
        of the solidus is no differential."""
        tail = _leibniz_tail(left)
        if tail is None or not self._at_solidus_denominator():
            return None
        before = self.pos, self.previous
        denominator = self._solidus_denominator()
        span = tail.numerator.span.to(denominator.span)
        quotient = _quotient(tail.numerator, denominator, span, leibniz=True)
        if not isinstance(quotient, ast.Derivative):
            self.pos, self.previous = before
            return None
        ratio = ast.Binary("/", tail.numerator, denominator, span=span)
        written = ast.Written(ratio, _differential_names(ratio))
        result: ast.Expr = replace(quotient, written=written)
        if tail.sign is not None:
            sign = tail.sign.op
            assert sign in ("-", "+")
            result = ast.Unary(sign, result, span=tail.sign.span.to(span))
        if tail.rest is None:
            return result
        span = left.span.to(span)  # from what is written first, its bracket too
        return ast.Binary("*", tail.rest, result, implicit=tail.beside, span=span)

    def _at_solidus_denominator(self) -> bool:
        """Whether a differential follows a solidus: ``dx``, ``d x``, ``dθ``, ``d{x}``, ``∂x``,
        or differentials in brackets, as AsciiMath writes ``(dy)/(dx)``."""
        token, follower = self.peek(), self.peek(1)
        if token.kind is K.LPAREN:
            return _starts_a_differential(follower)
        if token.kind is K.LBRACE:  # {\rm d}x
            return self._at_differential()
        if token.kind is K.IDENT and token.value == "d":
            return follower.kind is K.IDENT or self._at_plain_differential()
        return _starts_a_differential(token) and (token.kind is K.IDENT or follower.kind is K.IDENT)

    def _solidus_denominator(self) -> ast.Expr:
        """The denominator of Leibniz's fraction after a solidus: a differential and its
        power, ``dx^2``, ``d\\theta``, ``∂x``, ``{\\rm d}x``, or differentials in brackets,
        ``(dx dy)``; never a call, so that ``d/dx(x^3)`` is the derivative of x^3."""
        token = self.peek()
        if token.kind is K.LPAREN:
            return self._postfix(self._parenthesized(), grouped=True)
        if token.kind is K.IDENT and token.value == "d" and self._at_plain_differential():
            return self._postfix(self._plain_differential())  # d{x}^2
        marker: ast.Expr
        if token.kind is K.LBRACE:  # {\rm d}x
            self.advance()
            d = self.advance()
            self.expect(K.RBRACE)
            marker = ast.Name("d", d.verbatim, span=self._span(token))
        elif token.kind is K.COMMAND:  # ∂x, \partial x
            self.advance()
            marker = ast.Operator("partial", span=token.span)
        elif token.value != "d":  # dx, dx^2, dx_0, dt'
            self.advance()
            joined: ast.Expr = ast.Name(token.value, token.verbatim, span=token.span)
            order = len(self._primes())
            if order:
                joined = ast.Prime(
                    ast.Name(token.value, span=token.span), order, span=self._span(token)
                )
            return self._differential_scripts(joined)
        else:  # d x, dθ, d\theta
            self.advance()
            marker = ast.Name("d", token.verbatim, span=token.span)
        name = self.advance()
        primed = ast.Name(name.value + self._primes(), name.verbatim, span=self._span(name))
        variable = self._differential_scripts(primed)
        return ast.Binary("*", marker, variable, implicit=True, span=self._span(token))

    def _differential_scripts(self, node: ast.Expr) -> ast.Expr:
        """A differential with the scripts written against it: ``dx^2``, ``dx_0``."""
        while self.peek().kind in (K.CARET, K.UNDERSCORE) and not self.peek().space_before:
            node = self._attach_script(node, extend=True)
        return node

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
                order = _order(self.expect(K.NUMBER))
                self.expect(K.RBRACE)
            else:
                order = _order(self.expect(K.NUMBER))
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
        close = difflib.get_close_matches(environment, _ENVIRONMENTS, n=1, cutoff=_CLOSE)
        raise ParseError(
            f"unsupported environment {environment}",
            span=begin.span,
            notes=(_ENVIRONMENT_LIST,),
            hint=f"did you mean {close[0]}?" if close else None,
        )

    def _matrix(self) -> ast.Matrix:
        begin = self.advance()
        rows: list[tuple[ast.Expr, ...]] = []
        row: list[ast.Expr] = []
        with self._nested():
            while not self.check(K.END):
                if self.peek().kind in (K.AMPERSAND, K.ROW_SEP):  # an empty cell
                    row.append(ast.Dots(span=self.peek().span))
                else:
                    row.append(self.expression())
                if self.peek().kind in (K.COMMA, K.PERIOD) and self.peek(1).kind in (
                    K.AMPERSAND,
                    K.ROW_SEP,
                    K.END,
                ):
                    self.advance()  # the punctuation of a sentence
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

    def _cases(self) -> ast.Expr:
        begin = self.advance()
        pieces: list[tuple[ast.Expr, ast.Expr | None]] = []
        conditions = False
        with self._nested():
            while not self.check(K.END):
                value = self.expression()
                if self.peek().kind in (K.COMMA, K.PERIOD):
                    self.advance()
                if self.peek().kind in (K.ROW_SEP, K.END):  # equations in a brace
                    pieces.append((value, None))
                    self.accept(K.ROW_SEP)
                    continue
                self.expect(K.AMPERSAND, after="a case's value")
                pieces.append((value, self._condition()))
                conditions = True
                if self.peek().kind in (K.COMMA, K.PERIOD, K.SEMICOLON) and self.peek(1).kind in (
                    K.ROW_SEP,
                    K.END,
                ):
                    self.advance()  # the punctuation of a sentence: x > 0, \\
                if not self.accept(K.ROW_SEP) and not self.check(K.END):
                    token = self.peek()
                    raise ParseError(
                        f"expected '\\\\' or \\end{{{begin.value}}}, found {token.describe()}",
                        span=token.span,
                    )
            self.advance()
        if not pieces:
            raise ParseError("cases needs at least one case", span=self._span(begin))
        if not conditions:  # a brace gathering equations: \cases{a = ... \cr t = ...}
            return ast.List(tuple(value for value, _ in pieces), span=self._span(begin))
        return ast.Piecewise(tuple(pieces), span=self._span(begin))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_CLOSING = frozenset({K.RPAREN, K.RBRACKET, K.RBRACE, K.BAR})
# The closers of openers as they are written.
_CLOSES = {
    "(": ")", "[": "]", "{": "}", "\\{": "\\}", "\\lbrace": "\\rbrace", "\\lbrack": "\\rbrack",
    "\\langle": "\\rangle", "|": "|",
}  # fmt: skip
# Tokens that cannot be inside brackets: a bracket still open at one of them was never closed.
_OUTSIDE = frozenset({K.EOF, K.DEFINE, K.END, K.ROW_SEP, K.SEMICOLON})


def _rows(tokens: list[Token]) -> list[list[Token]]:
    r"""The rows of a script, as ``\substack`` writes them: its tokens split at each row
    break outside brackets."""
    rows: list[list[Token]] = [[]]
    depth = 0
    for token in tokens:
        if token.kind in _OPENERS:
            depth += 1
        elif token.kind in _CLOSERS:
            depth -= 1
        if token.kind is K.ROW_SEP and depth == 0:
            rows.append([])
        elif token.kind is not K.NEWLINE:
            rows[-1].append(token)
    return [row for row in rows if row] or [[]]


def _divides_at(row: list[Token]) -> int | None:
    r"""Where a row says that its index divides a number, as ``d \mid n`` and ``d | n`` do:
    the place of its one bar outside brackets, between two operands; ``None`` where it has
    none, or two, as the absolute value of ``|k| \le n`` has."""
    depth, bars = 0, []
    for place, token in enumerate(row):
        if token.kind in _OPENERS:
            depth += 1
        elif token.kind in _CLOSERS:
            depth -= 1
        elif token.kind is K.BAR and depth == 0:
            bars.append(place)
    return bars[0] if len(bars) == 1 and 0 < bars[0] < len(row) - 1 else None


def _unclosed(opener: Token, found: Token) -> ParseError:
    """The error for a bracket that ``found`` does not close: unclosed, marked where it was
    opened, if what was found cannot be inside it, or else the wrong closer."""
    text = opener.span.text
    if text.startswith("\\left"):
        text = text[len("\\left") :].strip()
    close = _CLOSES.get(text, "}" if opener.kind is K.LBRACE else ")")
    unclosed = found.kind in _OUTSIDE or (
        found.kind not in _CLOSERS and found.span.line > opener.span.line
    )
    if not unclosed:
        span = found.span.labelled(f"expected '{close}' here", (opener.span, "opened here"))
        message = f"expected '{close}' to close '{text}', found {found.describe()}"
        return ParseError(message, span=span)
    hint = f"add '{close}' to close it"
    others = []
    if found.kind is not K.EOF:
        others.append((found.span, f"still inside the '{text}' here"))
        if found.span.line > opener.span.line:
            hint += f", at the end of line {found.span.line - 1}"
    span = opener.span.labelled(f"this '{text}' is never closed", *others)
    return ParseError(f"unclosed '{text}'", span=span, hint=hint)


# Python's keywords, which in plain text would read as symbols, each with what to write.
_PYTHON_KEYWORDS = {
    "import": "the functions are built in, as diff(f, x) and solve(x^2 = 4, x) are: "
    "there is nothing to import",
    "def": "define a function by its parameters, as in f(x) := x^2",
    "return": "a function's value is what follows its :=, as in f(x) := x^2",
    "while": r"write the terms out, or a sum, as in \sum_{i=0}^{2} i^2",
    "for": r"write the terms out, or a sum, as in \sum_{i=0}^{2} i^2",
}
_PYTHON_KEYWORDS["from"] = _PYTHON_KEYWORDS["import"]


def _children(node: ast.Node) -> Iterator[ast.Node]:
    """The nodes directly inside ``node``."""
    pending: list[object] = [getattr(node, f.name) for f in fields(node)]
    while pending:
        value = pending.pop()
        if isinstance(value, ast.Node):
            yield value
        elif isinstance(value, tuple):
            pending.extend(value)


def _python_habits(statement: ast.Statement) -> None:
    """Raise for Python written in a plain-text statement, which the language would read
    as something else: a keyword as a symbol, ``x[1]`` as a product and ``n=5`` in a call
    as an equation."""
    pending: list[ast.Node] = [statement]
    while pending:
        node = pending.pop()
        pending.extend(_children(node))
        if (
            isinstance(node, ast.Name)
            and node.id in _PYTHON_KEYWORDS
            and not node.verbatim
            and node.span.text == node.id
        ):
            raise _keyword(node, statement)
        if (
            isinstance(node, ast.Binary)
            and node.implicit
            and isinstance(node.right, ast.List)
            and not isinstance(node.left, ast.Number)
            and not isinstance(_factor_before(node), ast.Name)  # x[n]: the reader decides
            and node.left.span.end == node.right.span.start
            and node.right.span.text.startswith("[")
        ):
            raise python_indexing(f"{node.left.span.text}{node.right.span.text}", node.span)
        if isinstance(node, ast.Call):
            _keyword_arguments(node)


def python_indexing(written: str, span: Span) -> ParseError:
    """The error for brackets written right after what they would index, as Python indexes
    a list, ``x[1]``, which the language does not. After a name, the reader says it, as
    only it knows whether the name is a sequence (design/notation.md, 31)."""
    return ParseError(
        f"{written} indexes as Python does, which the language does not",
        span=span,
        hint="a component is written with indices, as in V^{0}; for a product, put a space "
        "before the bracket",
    )


def _factor_before(node: ast.Binary) -> ast.Expr:
    """The factor written right before the last of a product, h of ``x[k] h[n-k]``."""
    left = node.left
    while isinstance(left, ast.Binary) and left.op == "*":
        left = left.right
    return left


def _keyword(name: ast.Name, statement: ast.Statement) -> ParseError:
    """The error for one of Python's keywords, written as a statement would be."""
    if name.id == "for" and any(
        isinstance(node, ast.List) and node.span.start < name.span.start < node.span.end
        for node in _walk(statement)
    ):
        what = "a list comprehension"
        hint = r"write the list out, as in [0, 1, 4], or a sum, as in \sum_{i=0}^{2} i^2"
    else:
        what = f"Python's {'import' if name.id == 'from' else name.id}"
        hint = _PYTHON_KEYWORDS[name.id]
    return ParseError(f"{what} is not part of the language", span=statement.span, hint=hint)


def _walk(node: ast.Node) -> Iterator[ast.Node]:
    pending = [node]
    while pending:
        current = pending.pop()
        yield current
        pending.extend(_children(current))


def _keyword_arguments(call: ast.Call) -> None:
    """Raise for Python's keyword argument, ``n=5``: after the first argument, a name and a
    value with ``=`` between them and no space, where the name is in no other argument.
    An equation, as a substitution in ``subs(x^2, x=2)`` is, names what the other
    arguments hold, and a first argument, as in ``solve(x=2)``, is what is acted on."""
    for argument in call.args[1:]:
        if not (isinstance(argument, ast.Equation) and isinstance(argument.lhs, ast.Name)):
            continue
        name, value = argument.lhs, argument.rhs
        source = argument.span.source
        if Span(source, name.span.end, value.span.start).text != "=":
            continue
        others = [node for other in call.args if other is not argument for node in _walk(other)]
        if any(isinstance(node, ast.Name) and node.id == name.id for node in others):
            continue
        written = call.span.text
        offset = argument.span.start - call.span.start
        example = written[:offset] + value.span.text + written[offset + len(argument.span.text) :]
        raise ParseError(
            f"{argument.span.text} is a keyword argument, which the language does not have",
            span=argument.span,
            hint=f"give the value in its place, as in {example}",
        )


def _written_before(token: Token, limit: int = 24) -> str:
    """What is written just before a token, back to a space or an operator: the base of a
    script, as ``\\Gamma^{a}`` is before ``_{bc``, for an example in a message."""
    text = token.span.source.text
    start = token.span.start
    while start > 0 and token.span.start - start < limit and text[start - 1] not in " \t\n+-*/=,;(":
        start -= 1
    return text[start : token.span.start]


def _dangling(operator: Token, continuation: Token) -> ParseError:
    """The error for a line that ends with an operator, when the next line, which was read
    as its continuation, is a statement of its own."""
    op, line = operator.span.text, operator.span.line
    span = operator.span.labelled(
        f"line {line} ends with this '{op}'",
        (continuation.span, f"so line {continuation.span.line} was read as its continuation"),
    )
    return ParseError(
        f"the expression on line {line} ends with '{op}'",
        span=span,
        notes=("a line that ends with an operator continues on the next line",),
        hint=f"complete the expression on line {line}, or remove the '{op}'",
    )


def _stray(token: Token, previous: Token, made: list[ast.Statement]) -> ParseError:
    """The error for a token that cannot follow a complete statement."""
    kind, text = token.kind, token.span.text
    if kind in _CLOSERS:
        opener = {K.RPAREN: "(", K.RBRACKET: "[", K.RBRACE: "{"}[kind]
        return ParseError(
            f"unmatched '{text}'",
            span=token.span.labelled(f"no '{opener}' is open here"),
            hint=f"remove it, or add the '{opener}' it closes",
        )
    if kind is K.AMPERSAND:
        return ParseError(
            "'&' outside an environment",
            span=token.span,
            notes=("'&' separates the columns of an environment such as align or pmatrix",),
            hint="remove it, or write the equations in \\begin{align} ... \\end{align}",
        )
    if kind is K.PERIOD:
        return ParseError("'.' is not an operator", span=token.span, hint=_period_hint(previous))
    if kind is K.COMMA:
        last = made[-1] if made else None
        hint = "to make a list, write it in brackets, as in [a, b]"
        if isinstance(last, ast.Define) and len(last.targets) == 1:
            value = last.value.span
            if value.line == token.span.line:
                line = token.span.source.line_text(token.span.line)
                rest = line[value.column - 1 :].strip().rstrip(",;")
                example = f"{last.targets[0].span.text} := [{rest}]"
                hint = f"to define a list, write it in brackets: {example}"
        return ParseError("unexpected ','", span=token.span, hint=hint)
    separate = "put each statement on its own line, or separate statements with ';'"
    return ParseError(
        f"unexpected {token.describe()}",
        span=token.span,
        hint=separate if kind is K.DEFINE else None,
    )


def _period_hint(previous: Token) -> str:
    """What a full stop between operands was likely meant to be."""
    if previous.kind is K.NUMBER:
        return f"a decimal has digits after its point, as in {previous.span.text}.0"
    return "for a product, write the factors side by side, or with \\cdot between them"


def _is_latex(token: Token) -> bool:
    """Whether a token was written as a LaTeX command or in Unicode mathematics."""
    text = token.span.text
    return text.startswith("\\") or not text.isascii()


def _plain_call(node: ast.Name | ast.Prime, parenthesis: Token) -> bool:
    """Whether, inside the unbracketed argument of a function, ``node`` followed by
    ``parenthesis`` is a call: a name written plainly with a plain parenthesis, as in
    ``\\ln A(r)``, but not ``\\cos\\theta\\left(...\\right)``, a product."""
    name = node.func if isinstance(node, ast.Prime) else node
    return parenthesis.span.text == "(" and not (
        name.span.text.startswith("\\") or not name.span.text.isascii()
    )


def _applicable(node: ast.Expr) -> bool:
    """Whether arguments typeset after ``node`` apply it, as a function: a name, maybe
    numbered, or raised to a power, as SymPy prints the square of a(t),
    ``a^{2}{\\left(t\\right)}``, and \\psi_1(x), ``\\psi_{1}{\\left(x \\right)}``."""
    if isinstance(node, ast.Scripted):
        return isinstance(node.base, ast.Name) and all(
            isinstance(script.body, ast.Number)
            and (script.kind == "^" or script.body.text.isdigit())
            for script in node.scripts
        )
    return isinstance(node, ast.Name)


def _numbered(node: ast.Scripted) -> ast.Name | None:
    """The name a name with whole numbers as its subscripts makes, as plain text writes
    it: ``\\psi_1`` is ``psi_1`` and ``R_{21}`` is ``R_21``; ``None`` for other scripts."""
    digits = _numeric_subscripts(node)
    if digits is None or not isinstance(node.base, ast.Name):
        return None
    return ast.Name(f"{node.base.id}_{digits}", node.base.verbatim, span=node.span)


def _starred(node: ast.Scripted) -> bool:
    """Whether a name's one script is a star, as ``\\psi^*``'s is."""
    if not isinstance(node.base, ast.Name) or len(node.scripts) != 1:
        return False
    script = node.scripts[0]
    star = isinstance(script.body, ast.Name) and script.body.id == "*" and script.body.verbatim
    return script.kind == "^" and star


def _square_bracketed(node: ast.Expr) -> ast.Expr | None:
    """The one expression square brackets hold, as F of ``[F]``, ``\\left[F\\right]`` and
    ``\\big[F\\big]``; ``None`` for anything else, as a list or a pair."""
    if not _SQUARE.fullmatch(node.span.text.strip()):
        return None
    if isinstance(node, ast.List):
        return node.items[0] if len(node.items) == 1 else None
    return None if isinstance(node, ast.Tuple) else node


def _without_script(node: ast.Expr, kind: Literal["^", "_"]) -> bool:
    """Whether ``node`` is a name with scripts, none of them of this kind."""
    return (
        isinstance(node, ast.Scripted)
        and isinstance(node.base, ast.Name)
        and all(script.kind != kind for script in node.scripts)
    )


def _numeric_subscripts(node: ast.Scripted) -> str | None:
    """The digits of a name's scripts, when they are all whole-number subscripts:
    ``\\phi_0`` → ``0``, ``T_{01}`` → ``01``."""
    digits = []
    for script in node.scripts:
        body = script.body
        if script.kind != "_" or not isinstance(body, ast.Number) or not body.text.isdigit():
            return None
        digits.append(body.text)
    return "".join(digits)


def _order(token: Token) -> int:
    """The order of a derivative, as in ``\\pdv[2]`` or ``d^{2}{f}``."""
    if not token.value.isdigit():
        raise ParseError("the order of a derivative must be a whole number", span=token.span)
    return int(token.value)


def _split_tokens(tokens: list[Token], kind: TokenKind) -> list[list[Token]]:
    """``tokens`` split at each token of ``kind`` outside brackets and environments."""
    parts: list[list[Token]] = [[]]
    depth = 0
    for token in tokens:
        if token.kind in _OPENERS or token.kind is K.BEGIN:
            depth += 1
        elif token.kind in _CLOSERS or token.kind is K.END:
            depth -= 1
        if token.kind is kind and depth == 0:
            parts.append([])
        else:
            parts[-1].append(token)
    return parts


def _optimum_operator(tokens: list[Token]) -> str | None:
    """The optimum an operator set by ``\\underset`` writes, as ``\\min``, ``\\arg\\max``
    or ``\\operatorname{argmin}`` do; ``None`` for anything else."""
    written = [(t.kind, t.value) for t in tokens if t.kind is not K.NEWLINE]
    verbatim = all(t.verbatim for t in tokens if t.kind is K.IDENT)
    if len(written) == 1:
        kind, value = written[0]
        if kind is K.COMMAND and value in _OPTIMA | _BOUNDS:
            return value
        if kind is K.IDENT and verbatim and value in _NAMED_OPTIMA:
            return value
    commands = {kind for kind, _ in written} == {K.COMMAND}
    if len(written) == 2 and written[0][1] == "arg" and written[1][1] in _OPTIMA:
        return "arg" + written[1][1] if commands or verbatim else None
    return None


def _written_variable(node: ast.Expr) -> bool:
    """Whether a node is a variable as an optimum is written over it: a name, ``x``,
    ``\\mathbf{x}`` or ``x_1``, maybe with subscripts, as ``x_{1}``, or an accent."""
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Accent):
        return isinstance(node.base, ast.Name)
    if isinstance(node, ast.Scripted) and isinstance(node.base, ast.Name):
        return all(script.kind == "_" for script in node.scripts)
    return False


def _first_operand(node: ast.Expr) -> ast.Expr | None:
    """The left side of a relation, an equation or a comparison."""
    if isinstance(node, ast.Relation | ast.Equation):
        return node.lhs
    if isinstance(node, ast.Comparison):
        return node.operands[0]
    return None


def _variable_of(node: ast.Expr) -> ast.Expr | None:
    """The variable a condition beneath an optimum is on: the ``x`` of ``x \\in S``, of
    ``x \\geq 0`` and of ``0 \\le x \\le 3``; ``None`` for anything else."""
    if isinstance(node, ast.Relation) and node.op in ("∈", "∉"):
        return node.lhs if _written_variable(node.lhs) else None
    if not isinstance(node, ast.Comparison):
        return None
    if len(node.ops) == 2:
        return node.operands[1] if _written_variable(node.operands[1]) else None
    return next((side for side in node.operands if _written_variable(side)), None)


def _on_each(names: list[ast.Expr], condition: ast.Expr) -> ast.Expr:
    """A condition on the last of ``names``, as on each of them: ``x_1, x_2 \\geq 0`` has
    the tuple of both on its left."""
    if len(names) == 1 or _first_operand(condition) is not names[-1]:
        return condition
    group = ast.Tuple(tuple(names), span=names[0].span.to(names[-1].span))
    span = group.span.to(condition.span)
    if isinstance(condition, ast.Comparison):
        return replace(condition, operands=(group, *condition.operands[1:]), span=span)
    assert isinstance(condition, ast.Relation | ast.Equation)
    return replace(condition, lhs=group, span=span)


def _no_constraint(node: ast.Expr) -> ParseError:
    return ParseError(
        f'{node.span.text} is no constraint: after "subject to", write relations',
        span=node.span,
        hint="as in \\text{subject to} \\quad x \\geq 0, \\quad x + y = 1",
    )


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
    """The factors of a juxtaposition, in order: ``d^2 y`` is ``[d^2, y]``."""
    factors: list[ast.Expr] = []
    pending = [node]
    while pending:  # without recursion: a product may have thousands of factors
        current = pending.pop()
        if isinstance(current, ast.Binary) and current.op == "*" and current.implicit:
            pending.append(current.right)
            pending.append(current.left)
        else:
            factors.append(current)
    return factors


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
    """A name that reads as a differential: ``dx`` → ``x``, ``dx^{\\mu}`` → ``x^{\\mu}``,
    ``dx'`` → ``x'``."""
    if isinstance(node, ast.Scripted) and isinstance(node.base, ast.Name):
        inner = _joined_differential(node.base)
        return None if inner is None else ast.Scripted(inner, node.scripts, span=node.span)
    if isinstance(node, ast.Prime):
        inner = _joined_differential(node.func)
        if not isinstance(inner, ast.Name):
            return None
        return ast.Name(inner.id + "'" * node.order, span=inner.span.to(node.span))
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
    """``d``, ``d^2 y``, ``dy``, ``\\partial^2 f``, ``d^2{f}``, ``d(x^3)``, ``df(x)``
    → ``(partial, order, f)``; and ``d(\\sin^2\\theta)``, read as a call of d, →
    ``(False, None, \\sin^2\\theta)``."""
    head, *rest = _factors(node)
    if isinstance(head, ast.Differential):
        return False, head.order, _product([head.expr, *rest])
    if isinstance(head, ast.Call) and _is_d(head.func) and len(head.args) == 1:
        return False, None, _product([head.args[0], *rest])
    if (variable := _joined_differential(head)) is not None:
        return False, None, _product([variable, *rest])
    if (marker := _differential_marker(head)) is not None:
        return marker[0], marker[1], _product(rest)
    if (called := _called_differential(head)) is not None:
        return False, None, _product([called, *rest])
    return None


def _called_differential(node: ast.Expr) -> ast.Expr | None:
    """What a differential written as a call is of: ``d(mv)`` of mv, as Feynman writes
    ``F = d(mv)/dt``, and ``df(x)`` of f(x)."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return None
    if node.func.id == "d" and len(node.args) == 1:
        return node.args[0]
    if (function := _joined_differential(node.func)) is not None:
        return ast.Call(function, node.args, span=function.span.to(node.span))
    return None


def _differential_head(node: ast.Expr) -> bool:
    """Whether a factor starts the numerator of Leibniz's fraction: ``d``, ``d^2``, ``∂``,
    ``dy``, ``d(x^3)``, ``df(x)`` or ``d{f}``."""
    return (
        isinstance(node, ast.Differential)
        or _differential_marker(node) is not None
        or _joined_differential(node) is not None
        or _called_differential(node) is not None
    )


def _starts_a_differential(token: Token) -> bool:
    """Whether a token starts a differential: d, a name that reads as one, as dx, or ∂."""
    if token.kind is K.IDENT:
        return token.value == "d" or (
            not token.verbatim and _DIFFERENTIAL.fullmatch(token.value) is not None
        )
    return token.kind is K.COMMAND and token.value == "partial"


class _LeibnizTail(NamedTuple):
    """A product before a solidus, as Leibniz's fraction takes it: what comes before the
    numerator, if anything; whether the two are side by side, rather than joined by ``*``;
    the sign before the numerator, as in ``-d/dx``; and the numerator."""

    rest: ast.Expr | None
    beside: bool
    sign: ast.Unary | None
    numerator: ast.Expr


def _leibniz_tail(node: ast.Expr) -> _LeibnizTail | None:
    """A product before a solidus as Leibniz's fraction takes it: ``x dy`` is x, then the
    numerator dy, and ``-d`` the numerator d, negated. The numerator runs from a
    differential (:func:`_differential_head`) to the end, its factors side by side, as
    ``d^2 y`` writes them. ``None`` where there is no such numerator."""
    chain: list[tuple[ast.Expr, bool]] = []  # each factor, and whether it is beside the last
    current = node
    while isinstance(current, ast.Binary) and current.op == "*":
        chain.append((current.right, current.implicit))
        current = current.left
    chain.append((current, True))
    chain.reverse()
    for start, (factor, beside) in enumerate(chain):
        sign = None
        if isinstance(factor, ast.Unary) and factor.op in ("-", "+"):
            sign, factor = factor, factor.operand  # -d/dx, -d^2 y/dx^2
        if not _differential_head(factor) or not all(b for _, b in chain[start + 1 :]):
            continue
        numerator = _product([factor, *(each for each, _ in chain[start + 1 :])])
        assert numerator is not None
        if start == 0:  # the whole product, as written, brackets and all
            return _LeibnizTail(None, True, sign, node if sign is None else numerator)
        rest = chain[0][0]
        for each, joined in chain[1:start]:
            rest = ast.Binary("*", rest, each, implicit=joined, span=rest.span.to(each.span))
        return _LeibnizTail(rest, beside, sign, numerator)
    return None


def _differential_names(node: ast.Expr) -> tuple[str, ...]:
    """The names a quotient is written with that read as differentials, as dy and dx, and
    the d of a differential written as a call, as in d(x^3): where the program defines one
    of them, the quotient is theirs."""
    found: dict[str, None] = {}
    for current in _walk(node):
        if isinstance(current, ast.Name) and _joined_differential(current) is not None:
            found[current.id] = None
        elif isinstance(current, ast.Call) and _called_differential(current) is not None:
            assert isinstance(current.func, ast.Name)
            found[current.func.id] = None
    return tuple(found)


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


def _take_differentials(
    integrand: ast.Expr,
) -> tuple[ast.Expr, list[tuple[ast.Expr, ast.Expr | None]]]:
    """Take the differentials out of the numerators of the fractions an integrand is a
    product of, with the dimension of each that is a volume element: ``\\frac{r^2 dr}{f(r)}``
    is ``r^2 / f(r)`` integrated over ``r``, and ``\\frac{d^3 p}{(2\\pi)^3} \\frac{1}{2E}``
    is integrated over the three dimensions of ``p``."""
    factors: list[ast.Expr] = []
    variables: list[tuple[ast.Expr, ast.Expr | None]] = []
    for factor in _factors(integrand):
        if isinstance(factor, ast.Binary) and factor.op == "/":
            numerator, found = _numerator_differentials(factor.left)
            if found:
                variables.extend(found)
                numerator = numerator or ast.Number("1", span=factor.left.span)
                factor = ast.Binary("/", numerator, factor.right, span=factor.span)
        factors.append(factor)
    if not variables:
        return integrand, []
    product = _product(factors)
    assert product is not None
    return product, variables


def _numerator_differentials(
    numerator: ast.Expr,
) -> tuple[ast.Expr | None, list[tuple[ast.Expr, ast.Expr | None]]]:
    """The factors of a numerator that are not differentials, and the variable and
    dimension of each that is: ``dr``, ``d\\theta``, ``dp^0``, or ``d^3 p``."""
    factors = _factors(numerator)
    kept: list[ast.Expr] = []
    found: list[tuple[ast.Expr, ast.Expr | None]] = []
    index = 0
    while index < len(factors):
        factor = factors[index]
        following = factors[index + 1] if index + 1 < len(factors) else None
        joined = _joined_differential(factor)
        dimension = _volume_dimension(factor)
        if joined is not None:  # dr, dp^0
            found.append((joined, None))
        elif following is not None and _variable(following) and _is_d(factor):  # d\theta
            found.append((_unprimed_name(following), None))
            index += 1
        elif following is not None and _variable(following) and dimension is not None:
            found.append((_unprimed_name(following), dimension))  # d^3 p
            index += 1
        else:
            kept.append(factor)
        index += 1
    return _product(kept), found


def _unprimed_name(node: ast.Expr) -> ast.Expr:
    """A primed variable as the name it is, ``x'``, as a differential's variable is named."""
    if isinstance(node, ast.Prime):
        return ast.Name(node.func.id + "'" * node.order, node.func.verbatim, span=node.span)
    return node


def _is_d(node: ast.Expr) -> bool:
    """The ``d`` of a differential written apart from its variable: ``d\\theta``,
    ``\\mathrm{d}x``."""
    return isinstance(node, ast.Name) and node.id == "d"


def _volume_dimension(node: ast.Expr) -> ast.Expr | None:
    """The dimension of a volume element's ``d^n``, as in ``d^3 p`` or ``d^d \\ell``."""
    if not (isinstance(node, ast.Scripted) and _is_d(node.base) and len(node.scripts) == 1):
        return None
    script = node.scripts[0]
    dimension = script.body
    if script.kind != "^" or not isinstance(dimension, ast.Number | ast.Name):
        return None
    if isinstance(dimension, ast.Number) and not dimension.text.isdigit():
        return None
    return dimension


def _variable(node: ast.Expr) -> bool:
    """A name, maybe primed or with indices, that a differential can be of: ``x``,
    ``x'``, ``p_1``, ``p^0``."""
    if isinstance(node, ast.Scripted):
        return isinstance(node.base, ast.Name) and all(s.indices for s in node.scripts)
    return isinstance(node, ast.Name | ast.Prime)


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


def _defined(clause: list[ast.Expr], joined: list[bool]) -> int:
    """How many of the expressions before ``:=`` it defines: the last, and the names
    listed before it with commas, as in ``x, y := ...``."""
    count = 1
    while (
        count < len(clause)
        and joined[len(clause) - count]
        and isinstance(clause[-count], ast.Name | ast.Scripted)
        and isinstance(clause[-count - 1], ast.Name | ast.Scripted)
    ):
        count += 1
    return count


def _evaluations(clause: list[ast.Expr], joined: list[bool]) -> list[ast.Statement]:
    """The statements of expressions separated by commas and spaces."""
    return [
        ast.Evaluate(target, span=target.span) for target in _memberships(*_ranges(clause, joined))
    ]


def _ranges(clause: list[ast.Expr], joined: list[bool]) -> tuple[list[ast.Expr], list[bool]]:
    """``\\mu, \\nu = 0, 1, 2, 3``: names set equal to a list of values say what the names
    range over. The list is one relation, ``(\\mu, \\nu) \\in (0, 1, 2, 3)``, which is read
    but not computed, rather than a definition of ``\\nu`` between two numbers."""
    items: list[tuple[ast.Expr, bool]] = []
    index = 0
    while index < len(clause):
        item = clause[index]
        if not (isinstance(item, ast.Equation) and _member(item.lhs)):
            items.append((item, joined[index]))
            index += 1
            continue
        names, link = 0, joined[index]  # the members listed before it with commas
        while names < len(items) and link and _member(items[-1 - names][0]):
            link = items[-1 - names][1]
            names += 1
        end = index + 1  # and the values listed after it
        while end < len(clause) and joined[end] and _is_value(clause[end]):
            end += 1
        values = clause[index + 1 : end]
        if not names and not (values and _is_numeral(item.rhs) and _is_numeral(values[0])):
            items.append((item, joined[index]))
            index += 1
            continue
        members = [member for member, _ in items[len(items) - names :]]
        del items[len(items) - names :]
        lhs: ast.Expr = item.lhs
        if members:
            lhs = ast.Tuple((*members, item.lhs), span=members[0].span.to(item.lhs.span))
        last = values[-1] if values else item.rhs
        rhs = ast.Tuple((item.rhs, *values), span=item.rhs.span.to(last.span))
        items.append((ast.Relation(lhs, "∈", rhs, span=lhs.span.to(rhs.span)), link))
        index = end
    return [item for item, _ in items], [link for _, link in items]


def _ranged(statements: list[ast.Statement]) -> list[ast.Statement]:
    """A definition followed on its line by the range of its index, ``x_n := n h, \\quad
    n = 0, \\ldots, N-1``, as one statement: the range belongs to the definition, which
    is made for each value of n."""
    joined: list[ast.Statement] = []
    for statement in statements:
        last = joined[-1] if joined else None
        if (
            isinstance(last, ast.Define)
            and len(last.targets) == 1
            and isinstance(statement, ast.Evaluate)
            and isinstance(statement.expr, ast.Relation)
            and statement.expr.op == "∈"
            and not statement.expr.written
            and (members := _members(statement.expr.lhs)) is not None
            and members <= _labels_written(last.targets[0])
        ):
            span = last.span.to(statement.span)
            joined[-1] = replace(last, ranges=(*last.ranges, statement.expr), span=span)
            continue
        joined.append(statement)
    return joined


def _members(node: ast.Expr) -> set[str] | None:
    """The names a range is given for, ``n`` or ``(i, j)``; ``None`` for anything else."""
    items = node.items if isinstance(node, ast.Tuple) else (node,)
    if not all(isinstance(item, ast.Name) for item in items):
        return None
    return {item.id for item in items if isinstance(item, ast.Name)}


def _labels_written(target: ast.Expr) -> set[str]:
    """The index labels a definition's target is written with: n of ``x_n``, and a and b of
    ``T^{a}{}_{b}``."""
    if isinstance(target, ast.Name):
        base, underscore, tail = target.id.partition("_")
        return set(split_index_labels(tail)) if base and underscore and tail.isalpha() else set()
    if isinstance(target, ast.Scripted):
        return {
            str(item.label)
            for script in target.scripts
            if script.indices is not None
            for item in script.indices.items
            if item.label is not None
        }
    return set()


def _memberships(targets: list[ast.Expr], joined: list[bool]) -> list[ast.Expr]:
    """``a, b \\in \\mathcal{A}``: names listed with commas before a membership are
    members too, so the list is one statement, ``(a, b) \\in \\mathcal{A}``, not ``a``
    and ``b \\in ...``; and so are names listed before a comparison, as in ``a, b > 0``,
    which is ``(a, b) > 0``, and before ``\\sim``, with an ellipsis among them, as in
    ``X_1, \\ldots, X_n \\sim D``. Prose, as in ``\\text{where} \\quad x, y \\in X``, is not."""
    grouped: list[ast.Expr] = []
    names: list[ast.Expr] = []
    for target, link in zip(targets, joined, strict=True):
        if names and not link:
            grouped.extend(names)
            names = []
        if _member(target) or (isinstance(target, ast.Dots) and names):
            names.append(target)
            continue
        dotted = any(isinstance(name, ast.Dots) for name in names)
        if isinstance(target, ast.Relation) and target.op == "~" and names and _member(target.lhs):
            members = ast.Tuple((*names, target.lhs), span=names[0].span.to(target.lhs.span))
            span = names[0].span.to(target.span)
            target = ast.Relation(members, "~", target.rhs, target.written, span=span)
        elif dotted:
            grouped.extend(names)
        elif (
            isinstance(target, ast.Relation) and target.op == "∈" and names and _member(target.lhs)
        ):
            members = ast.Tuple((*names, target.lhs), span=names[0].span.to(target.lhs.span))
            target = ast.Relation(members, "∈", target.rhs, span=names[0].span.to(target.span))
        elif isinstance(target, ast.Comparison) and names and _member(target.operands[0]):
            first = target.operands[0]
            members = ast.Tuple((*names, first), span=names[0].span.to(first.span))
            operands = (members, *target.operands[1:])
            span = names[0].span.to(target.span)
            target = ast.Comparison(operands, target.ops, span=span)
        else:
            grouped.extend(names)
        names = []
        grouped.append(target)
    grouped.extend(names)
    return grouped


def _a_number_set(node: ast.Expr) -> bool:
    """``\\mathbb{R}``, ``\\mathbb{Z}`` and the other sets of numbers written in
    blackboard bold."""
    return isinstance(node, ast.Name) and node.id in ("ℂ", "ℕ", "ℚ", "ℝ", "ℤ")


def _member(node: ast.Expr) -> bool:
    """A name that can be one of a list of members: ``x``, or ``p_{1}``, but not a word
    of prose such as ``\\text{where}``."""
    if isinstance(node, ast.Scripted):
        return isinstance(node.base, ast.Name) and _member(node.base)
    return isinstance(node, ast.Name) and not (
        node.verbatim and _TEXT_COMMAND.match(node.span.text) is not None
    )


def _is_value(node: ast.Expr) -> bool:
    """An expression that is not itself a relation, which can be one of a list of values."""
    return not isinstance(node, ast.Equation | ast.Relation | ast.Chain | ast.Comparison)


def _is_numeral(node: ast.Expr) -> bool:
    """A number, perhaps signed, or an ellipsis: ``0``, ``-1``, ``\\ldots``."""
    if isinstance(node, ast.Unary) and node.op in ("-", "+"):
        node = node.operand
    return isinstance(node, ast.Number | ast.Dots)


def _with_comparisons(
    operands: list[ast.Expr], relations: list[str], written: list[str]
) -> tuple[list[ast.Expr], list[str], list[str]]:
    """Equalities in a row with comparisons among them, as ``a = b \\geq c`` and
    ``a \\leq b = c``: a comparison binds tighter than =, so that ``b \\geq c`` was one
    operand, and ``a`` equal to a truth value. Written without brackets of its own, each
    comparison is links of the chain, b \\geq c beside a = b. ``(x \\geq 0)`` in brackets
    stays one operand."""
    if not set(relations) <= {"=", "≈"}:
        return operands, relations, written
    spliced: list[ast.Expr] = []
    linked: list[str] = []
    spelled: list[str] = []
    for position, operand in enumerate(operands):
        if position:
            linked.append(relations[position - 1])
            spelled.append(written[position - 1])
        if not (
            isinstance(operand, ast.Comparison)
            and operand.span.start == operand.operands[0].span.start  # not in brackets
        ):
            spliced.append(operand)
            continue
        spliced.append(operand.operands[0])
        for op, before, after in zip(
            operand.ops, operand.operands, operand.operands[1:], strict=False
        ):
            between = Span(after.span.source, before.span.end, after.span.start)
            linked.append(op)
            spelled.append(between.text.strip())
            spliced.append(after)
    return spliced, linked, spelled


def _relation(token: Token) -> str:
    """The symbol of a relation token: ``=``, ``→``, ``≡``, ``∣`` of the conditional bar, or
    the relation's own symbol."""
    if _given_bar(token):
        return "∣"
    if token.kind is K.EQUAL:
        return "="
    if token.kind is K.DEFINE:
        return "≡"
    if token.kind is K.ARROW:
        return "→"
    if token.kind is K.COLON:
        return ":"
    return token.value


def _given(operands: list[ast.Expr], relations: list[str], written: list[str]) -> ast.Expr:
    r"""Relations in a row with the conditional bar among them, as one relation: what is
    conditioned, the bar and what it is given, each side one expression, as
    ``X = 2 \mid N = 3``; before ``\sim``, a law given another, ``(X \mid Y = 0) \sim D``."""
    span = operands[0].span.to(operands[-1].span)
    at = relations.index("∣")
    if relations[-1] == "~" and at < len(relations) - 1:
        law = _given(operands[:-1], relations[:-1], written[:-1])
        return ast.Relation(law, "~", operands[-1], written[-1], span=span)
    before = _linked(operands[: at + 1], relations[:at], written[:at])
    after = _linked(operands[at + 1 :], relations[at + 1 :], written[at + 1 :])
    return ast.Relation(before, "∣", after, written[at], span=span)


def _linked(operands: list[ast.Expr], relations: list[str], written: list[str]) -> ast.Expr:
    """Operands and the relations between them as one expression: an operand alone, an
    equation, a relation, or a chain."""
    if not relations:
        return operands[0]
    span = operands[0].span.to(operands[-1].span)
    if len(relations) == 1 and relations[0] == "=":
        return ast.Equation(operands[0], operands[1], span=span)
    if len(relations) == 1:
        return ast.Relation(operands[0], relations[0], operands[1], written[0], span=span)
    return ast.Chain(tuple(operands), tuple(relations), tuple(written), span=span)


def _given_bar(token: Token) -> bool:
    r"""Whether a bar is the conditional bar, ``\mid``, outside an absolute value and the
    brackets of Dirac's notation, where a bar is theirs: ``\Pr(A \mid B)``."""
    return token.kind is K.BAR and token.span.text.strip() == "\\mid"


# The name of a macro, at the start of a call of it as written: \Gam in \Gam^t_{tr}.
_MACRO_NAME = re.compile(r"\\[A-Za-z]+")
# Arguments as SymPy typesets them, in braces and \left( \right): a{\left(t \right)}.
_TYPESET_ARGUMENTS = re.compile(r"\{\s*\\left\s*\(")

# How each command that takes arguments is written, for the hint of a missing one.
_ARGUMENT_EXAMPLES = {
    **{name: f"\\{name}{{1}}{{2}}" for name in ("frac", "dfrac", "tfrac", "cfrac")},
    "sqrt": "\\sqrt{x}",
}

# TeX's generalised fractions, and the LaTeX commands they become.
_FRACTION_COMMANDS = {"over": "frac", "choose": "binom", "brace": "brace", "brack": "brack"}
# The commands written as a binomial coefficient is, and the functions they call: the
# Stirling numbers in Karamata's brackets, {n \brace k} and {n \brack k}, as Knuth writes them.
_BINOMIALS = {
    "binom": "binomial",
    "dbinom": "binomial",
    "tbinom": "binomial",
    "brace": "stirling2",
    "brack": "stirling1",
}
_OPENERS = frozenset({K.LBRACE, K.LPAREN, K.LBRACKET})
_CLOSERS = frozenset({K.RBRACE, K.RPAREN, K.RBRACKET})
_GROUP_LIMITS = frozenset({K.ROW_SEP, K.SEMICOLON, K.BEGIN, K.END, K.EOF})


def _generalised_fractions(tokens: list[Token]) -> list[Token]:
    r"""Rewrite TeX's ``{a \over b}`` as ``\frac{a}{b}``, ``\choose`` as ``\binom``, and
    ``\brace`` and ``\brack`` as the brackets of the Stirling numbers.

    As in TeX, ``\over`` divides everything in the group it is in: the braces or
    brackets around it, or else its alignment cell or line. One pass over the
    tokens, keeping a stack of the groups open, rewrites every group as it closes.
    """
    if not any(t.kind is K.COMMAND and t.value in _FRACTION_COMMANDS for t in tokens):
        return tokens
    groups = [_Group()]
    for token in tokens:
        group = groups[-1]
        if token.kind in _GROUP_LIMITS:
            group.end_segment()
            group.tokens.append(token)
            group.start = len(group.tokens)
            continue
        if token.aligned:  # an alignment cell starts here
            group.end_segment()
        if token.kind is K.COMMAND and token.value in _FRACTION_COMMANDS:
            group.overs.append((len(group.tokens), token))
        elif token.kind in _OPENERS:
            group.tokens.append(token)
            groups.append(_Group())
        elif token.kind in _CLOSERS and len(groups) > 1:
            inner = groups.pop()
            groups[-1].tokens.extend(inner.finish())
            groups[-1].tokens.append(token)
        else:
            group.tokens.append(token)
    while len(groups) > 1:  # unclosed brackets: the parser will report them
        inner = groups.pop()
        groups[-1].tokens.extend(inner.finish())
    return groups[0].finish()


class _Group:
    """A group of tokens being read, and the ``\\over`` commands in its current segment."""

    def __init__(self) -> None:
        self.tokens: list[Token] = []
        self.start = 0  # where the current segment (cell or line) begins
        self.overs: list[tuple[int, Token]] = []  # (position in tokens, \over)

    def end_segment(self) -> None:
        if self.overs:
            segment = self.tokens[self.start :]
            overs = [(position - self.start, over) for position, over in self.overs]
            self.tokens[self.start :] = _fraction(segment, overs)
            self.overs = []
        self.start = len(self.tokens)

    def finish(self) -> list[Token]:
        self.end_segment()
        return self.tokens


def _fraction(tokens: list[Token], overs: list[tuple[int, Token]]) -> list[Token]:
    """``a \\over b``, given the tokens of a segment without its ``\\over``s, and where
    they were. A second ``\\over``, which TeX would reject, divides the denominator."""
    if not overs:
        return tokens
    (position, over), rest = overs[0], overs[1:]
    numerator = tokens[:position]
    denominator = _fraction(tokens[position:], [(p - position, o) for p, o in rest])
    # The new tokens span the whole fraction, so that the fraction's span does too.
    first = numerator[0].span if numerator else over.span
    last = denominator[-1].span if denominator else over.span
    space_before = numerator[0].space_before if numerator else over.space_before
    return [
        Token(K.COMMAND, _FRACTION_COMMANDS[over.value], first.to(last), space_before),
        Token(K.LBRACE, "{", Span(first.source, first.start, first.start)),
        *numerator,
        Token(K.RBRACE, "}", over.span),
        Token(K.LBRACE, "{", over.span),
        *denominator,
        Token(K.RBRACE, "}", Span(last.source, last.end, last.end)),
    ]


def _equivalence(token: Token) -> bool:
    r"""Whether a token is ``\equiv`` or ``≡``, which defines at a statement's top level."""
    return token.span.text in ("\\equiv", "≡")


def _modular(value: ast.Expr) -> bool:
    r"""Whether the side of ``\equiv`` ends in a modulus, as ``2 \pmod{3}`` and
    ``b + c \bmod m`` do, which makes it a congruence (design/notation.md, 30)."""
    if isinstance(value, ast.Binary) and value.op == "mod":
        return True
    return isinstance(value, ast.Binary) and value.op in ("+", "-") and _modular(value.right)


def _definable(target: ast.Expr) -> bool:
    """Whether ``target :=`` can define it: a name, maybe decorated, a tensor, a function of
    names, maybe primed (decision 7 of design/notation.md), or a ket, a bracket or angle
    brackets, as ``|0\\rangle``."""
    if isinstance(target, ast.Call):
        return _parameter(target.func) is not None and all(
            _parameter(arg) is not None for arg in target.args
        )
    return isinstance(target, ast.Name | ast.Scripted | dirac.DEFINABLE) or _decorated(target)


def _parameter(node: ast.Expr) -> str | None:
    """The name a function or a parameter is defined with: ``x``, or ``t'``, as a prime on
    the left of := is part of the name, or ``x_0`` of ``x_{0}`` and of Unicode's ``x₀``, as
    a whole number subscript is; ``None`` for anything else."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Prime):
        return node.func.id + "'" * node.order
    if isinstance(node, ast.Scripted) and (numbered := _numbered(node)) is not None:
        return numbered.id
    return None


def _sampled(target: ast.Expr) -> bool:
    """``x[n]``: a name and, right after it, a point in square brackets, which the parser
    reads as a product and the compiler as a sequence at its label (design/notation.md,
    31)."""
    return (
        isinstance(target, ast.Binary)
        and target.op == "*"
        and isinstance(target.left, ast.Name)
        and isinstance(target.right, ast.List)
        and len(target.right.items) == 1
        and target.right.span.text.startswith("[")
        and target.right.span.start == target.left.span.end
    )


def _scripted_call(target: ast.Expr) -> bool:
    """``R_{nl}(r)``: a name with scripts and, right after it, names in brackets, which
    the parser reads as a product and the compiler as a function's name and parameters."""
    if not (isinstance(target, ast.Binary) and target.op == "*"):
        return False
    func, group = target.left, target.right
    items = group.items if isinstance(group, ast.Tuple) else (group,)
    return (
        isinstance(func, ast.Scripted)
        and isinstance(func.base, ast.Name)
        and group.span.text.startswith("(")
        and group.span.start == func.span.end
        and all(isinstance(item, ast.Name) for item in items)
    )


def _operator_applied(target: ast.Expr) -> tuple[str, str] | None:
    """``\\operatorname{sinc} x``: a name written as an operator, followed by one name, its
    argument, as books write a function they define (decision 11 of design/notation.md);
    the name and the argument's name. ``None`` for anything else."""
    if not (isinstance(target, ast.Binary) and target.op == "*"):
        return None
    func, argument = target.left, target.right
    if not (isinstance(func, ast.Name) and isinstance(argument, ast.Name)):
        return None
    if not func.span.text.startswith(_OPERATOR_NAMES) or len(func.id) < 2:
        return None  # \mathrm{d} x is a differential
    return func.id, argument.id


#: The commands that write a name as an operator's, as \operatorname{sinc} and \mathrm{sinc}.
_OPERATOR_NAMES = ("\\operatorname", "\\mathrm")


def _decorated(target: ast.Expr) -> bool:
    """A decorated name, which names a symbol of its own: primed, as the coordinate ``x'``
    is, or under an accent that is not a time derivative, as ``\\hat{p}`` and ``\\bar{x}``
    are."""
    if isinstance(target, ast.Prime):
        return True
    return (
        isinstance(target, ast.Accent)
        and target.accent not in ("dot", "ddot")
        and isinstance(target.base, ast.Name)
    )


def _identity(target: ast.Expr, value: ast.Expr, span: Span, equiv: str) -> ast.Expr:
    r"""``A \equiv B``, where ``A`` cannot be defined, states that A is identically B.
    ``equiv`` is how the identity was written: ``\equiv`` or ``≡``."""
    operands, relations, written = [target], ["≡"], [equiv]
    if isinstance(value, ast.Chain):
        operands += value.operands
        relations += value.relations
        written += value.written or [""] * len(value.relations)
    elif isinstance(value, ast.Equation):
        operands += [value.lhs, value.rhs]
        relations.append("=")
        written.append("=")
    elif isinstance(value, ast.Relation):
        operands += [value.lhs, value.rhs]
        relations.append(value.op)
        written.append(value.written)
    else:
        return ast.Relation(target, "≡", value, equiv, span=span)
    return ast.Chain(tuple(operands), tuple(relations), tuple(written), span=span)


def _primed_script(node: ast.Scripted, primes: str) -> ast.Scripted:
    """``x^\\mu'``: the primes belong to the last script's name and index."""
    script = node.scripts[-1]
    body = script.body
    if isinstance(body, ast.Name):
        body = replace(body, id=body.id + primes)
    indices = script.indices
    last = indices.items[-1] if indices is not None and indices.items else None
    if indices is not None and last is not None and last.label is not None:
        items = (*indices.items[:-1], replace(last, label=last.label + primes))
        indices = replace(indices, items=items)
    return replace(node, scripts=(*node.scripts[:-1], replace(script, body=body, indices=indices)))
