r"""Lexical analysis: source text in, tokens out.

The lexer is where RelativisticPy's two dialects become one. Plain-text maths
(``g_{mu nu} := [[...]]``, ``x**2``) and LaTeX (``\frac{1}{2}``,
``\Gamma^{\mu}_{\nu\rho}``) produce the same stream of tokens, so the parser
never needs to know which one it is reading.

The lexer also absorbs everything in LaTeX that is about typesetting rather
than mathematics: spacing and sizing commands, ``\label{...}``, math-mode
delimiters (``$``, ``\[``), alignment markers (``&`` inside ``align``) and
style commands (``\mathrm{d}`` reads as ``d``).

Normalisations, so later stages see one spelling per idea:

* Greek letters in every form (``\theta``, ``\vartheta``, ``theta``, ``θ``)
  become the identifier ``theta``.
* Operator commands become operator tokens: ``\cdot`` is ``*``, ``\leq`` is
  ``<=``, ``\to`` is ``->`` and ``\coloneqq``/``\equiv`` are ``:=``.
* ``\left`` and ``\right`` are paired. Display brackets such as
  ``\left[ ... \right]`` become grouping parentheses, ``\left| ... \right|``
  an absolute value, and ``\left. ... \right|`` a parenthesised expression
  followed by an evaluation bar.
* Unicode superscripts are spelled out: ``r²`` lexes exactly like ``r^{2}``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from relativisticpy.errors import ParseError
from relativisticpy.language import vocabulary as vocab
from relativisticpy.language.source import Source, Span
from relativisticpy.language.tokens import Token, TokenKind

_NUMBER = re.compile(r"(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?")
# An underscore belongs to a name only when a letter or digit follows it:
# ``x_0`` and ``abs_func`` are names, while ``g_{mu}`` is ``g`` with a subscript.
_IDENT = re.compile(r"[A-Za-z](?:[A-Za-z0-9]|_(?=[A-Za-z0-9]))*")
# What may follow an escaped underscore in a name, as in ``geodesic\_equations``.
_NAME_TAIL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9]|_(?=[A-Za-z0-9]))*")
_COMMAND_NAME = re.compile(r"[A-Za-z]+")
# The argument of \mathbb when it is one letter or digit: {R}, { R }, or R itself.
_BLACKBOARD_ARGUMENT = re.compile(r"\s*(?:\{\s*([A-Za-z0-9])\s*\}|([A-Za-z0-9]))")
_TEXT_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
# The optional spacing argument of a row break, as in ``\\[2pt]``.
_ROW_SPACING = re.compile(
    r"[ \t]*\[\s*-?(?:\d+(?:\.\d*)?|\.\d+)\s*(?:pt|em|ex|mm|cm|in|bp|pc|sp|mu)\s*\]"
)

_SPACE = frozenset(" \t\r\f\v~")
_ARGUMENT_SPACE = _SPACE | {"\n"}  # whitespace allowed between a command and its argument

_SINGLE_CHARACTER: dict[str, TokenKind] = {
    "+": TokenKind.PLUS,
    "-": TokenKind.MINUS,
    "*": TokenKind.STAR,
    "/": TokenKind.SLASH,
    "^": TokenKind.CARET,
    "_": TokenKind.UNDERSCORE,
    "!": TokenKind.BANG,
    "=": TokenKind.EQUAL,
    "<": TokenKind.LESS,
    ">": TokenKind.GREATER,
    "(": TokenKind.LPAREN,
    ")": TokenKind.RPAREN,
    "[": TokenKind.LBRACKET,
    "]": TokenKind.RBRACKET,
    "{": TokenKind.LBRACE,
    "}": TokenKind.RBRACE,
    "|": TokenKind.BAR,
    ",": TokenKind.COMMA,
    ":": TokenKind.COLON,
    ";": TokenKind.SEMICOLON,
    "&": TokenKind.AMPERSAND,
}

_TWO_CHARACTER: dict[str, TokenKind] = {
    "**": TokenKind.DOUBLE_STAR,
    ":=": TokenKind.DEFINE,
    "==": TokenKind.EQ_EQ,
    "!=": TokenKind.NOT_EQ,
    "<=": TokenKind.LESS_EQ,
    ">=": TokenKind.GREATER_EQ,
    "->": TokenKind.ARROW,
}

# A quote directly after one of these is a prime, as in f'(x); otherwise it starts a string.
_PRIMEABLE = frozenset(
    {
        TokenKind.IDENT, TokenKind.NUMBER, TokenKind.COMMAND, TokenKind.RPAREN,
        TokenKind.RBRACKET, TokenKind.RBRACE, TokenKind.PRIME,
    }
)  # fmt: skip

# Delimiters that may follow \left, \right and the sizing commands, by role.
_OPENING = frozenset({"(", "[", r"\{", r"\lbrace", r"\lbrack", r"\langle"})
_CLOSING = frozenset({")", "]", r"\}", r"\rbrace", r"\rbrack", r"\rangle"})
_BARS = frozenset({"|", r"\|", r"\vert", r"\Vert", r"\lvert", r"\rvert", r"\lVert", r"\rVert"})
_FLOORS = {r"\lfloor": "lfloor", r"\rfloor": "rfloor", r"\lceil": "lceil", r"\rceil": "rceil"}


def tokenize(source: str | Source) -> list[Token]:
    """Split source text into tokens. The last token is always ``EOF``."""
    if isinstance(source, str):
        source = Source(source)
    return Lexer(source).tokenize()


class Lexer:
    """Converts one :class:`Source` into a list of tokens. Use :func:`tokenize`."""

    def __init__(self, source: Source) -> None:
        self.source = source
        self.text = source.text
        self.pos = 0
        self.tokens: list[Token] = []
        self.space_before = False
        self.explicit_space = False
        self.wide_space = False  # \quad or wider precedes the next token
        self.aligned = False  # an alignment marker '&' precedes the next token
        self._open_left: list[tuple[str, Span]] = []  # pending \left delimiters
        self._environments: list[tuple[str, Span]] = []  # open \begin{...}
        self._transparent_braces: set[int] = set()  # offsets of '}' closing a style command
        self._font_ends: list[int] = []  # where the groups of font switches such as \rm end
        self._closers: dict[int, str] = {}  # offsets of '}' closing \matrix{...} and friends
        self._display_ends: dict[int, int] = {}  # offsets of the \] or $$ closing a display

    # -- driver ---------------------------------------------------------------

    def tokenize(self) -> list[Token]:
        while self.pos < len(self.text):
            self._scan()
        if self._open_left:
            raise ParseError(r"'\left' has no matching '\right'", span=self._open_left[-1][1])
        if self._environments:
            name, span = self._environments[-1]
            raise ParseError(f"\\begin{{{name}}} has no matching \\end{{{name}}}", span=span)
        self._emit(TokenKind.EOF, "", self.pos, self.pos)
        return self.tokens

    def _scan(self) -> None:
        char = self.text[self.pos]
        if self.pos in self._display_ends:
            self._environments.pop()
            end = self.pos + self._display_ends.pop(self.pos)
            self._emit(TokenKind.END, "displaymath", self.pos, end)
        elif self.text.startswith("$$", self.pos):
            self._display("$$", "$$")
        elif char in _SPACE or char == "$":
            self._skip(1, explicit=char == "~")
        elif char == "\n":
            self._emit(TokenKind.NEWLINE, "\n", self.pos, self.pos + 1)
        elif char in "#%":
            self._skip_comment()
        elif (char.isascii() and char.isdigit()) or (char == "." and self._peek(1).isdigit()):
            self._match(_NUMBER, TokenKind.NUMBER)
        elif char == ".":
            if self.text.startswith("...", self.pos):
                self._emit(TokenKind.ELLIPSIS, TokenKind.ELLIPSIS.value, self.pos, self.pos + 3)
            else:
                self._emit(TokenKind.PERIOD, ".", self.pos, self.pos + 1)
        elif char.isascii() and char.isalpha():
            self._identifier()
        elif char == "\\":
            self._backslash()
        elif char in "'\"":
            self._quote(char)
        elif char == "&" and self._in_alignment():
            self._skip(1)
            self.aligned = True
        elif char == "}" and self.pos in self._closers:
            environment = self._closers.pop(self.pos)
            self._environments.pop()
            self._emit(TokenKind.END, environment, self.pos, self.pos + 1)
        elif char == "}" and self.pos in self._transparent_braces:
            self._transparent_braces.discard(self.pos)
            self.pos += 1
        elif (kind := _TWO_CHARACTER.get(self.text[self.pos : self.pos + 2])) is not None:
            self._emit(kind, kind.value, self.pos, self.pos + 2)
        elif (kind := _SINGLE_CHARACTER.get(char)) is not None:
            self._emit(kind, char, self.pos, self.pos + 1)
        else:
            self._unicode(char)

    # -- primitives -----------------------------------------------------------

    def _emit(
        self, kind: TokenKind, value: str, start: int, end: int, *, verbatim: bool = False
    ) -> None:
        span = Span(self.source, start, end)
        while self._font_ends and start >= self._font_ends[-1]:
            self._font_ends.pop()
        # Names in a style command's braces (\mathrm{mass}) or after a font switch
        # ({\rm Tr}) are verbatim.
        styled = bool(self._transparent_braces or self._font_ends)
        verbatim = verbatim or (kind is TokenKind.IDENT and styled)
        token = Token(
            kind,
            value,
            span,
            self.space_before,
            self.explicit_space,
            verbatim,
            self.aligned,
            self.wide_space,
        )
        self.tokens.append(token)
        if kind is TokenKind.NEWLINE:  # spacing before a line break belongs to what follows
            self.space_before = True
        else:
            self.space_before = self.explicit_space = self.aligned = self.wide_space = False
        self.pos = end

    def _skip(self, count: int, *, explicit: bool = False) -> None:
        self.pos += count
        self.space_before = True
        self.explicit_space = self.explicit_space or explicit

    def _peek(self, offset: int = 0) -> str:
        return self.text[self.pos + offset : self.pos + offset + 1]

    def _error(
        self, message: str, start: int, end: int | None = None, hint: str | None = None
    ) -> ParseError:
        end = start + 1 if end is None else end
        return ParseError(message, span=Span(self.source, start, end), hint=hint)

    def _match(self, pattern: re.Pattern[str], kind: TokenKind) -> None:
        match = pattern.match(self.text, self.pos)
        assert match is not None  # guaranteed by the caller's first-character check
        self._emit(kind, match.group(), self.pos, match.end())

    def _identifier(self) -> None:
        """A name; LaTeX's escaped underscore joins its parts: ``geodesic\\_equations``."""
        start = self.pos
        match = _IDENT.match(self.text, start)
        assert match is not None
        end = match.end()
        while self.text.startswith("\\_", end):
            tail = _NAME_TAIL.match(self.text, end + 2)
            if tail is None:
                break
            end = tail.end()
        self._emit(TokenKind.IDENT, self.text[start:end].replace("\\_", "_"), start, end)

    def _skip_comment(self) -> None:
        end = self.text.find("\n", self.pos)
        self.pos = len(self.text) if end == -1 else end
        self.space_before = True

    def _skip_whitespace(self) -> None:
        while self._peek() in _ARGUMENT_SPACE:
            self.pos += 1

    def _in_alignment(self) -> bool:
        if not self._environments:
            return False
        return self._environments[-1][0] in vocab.ALIGNMENT_ENVIRONMENTS

    def _quote(self, char: str) -> None:
        previous = self.tokens[-1] if self.tokens else None
        if char == "'" and not self.space_before and previous and previous.kind in _PRIMEABLE:
            self._emit(TokenKind.PRIME, "'", self.pos, self.pos + 1)
            return
        end = self.text.find(char, self.pos + 1)
        newline = self.text.find("\n", self.pos + 1)
        if end == -1 or -1 < newline < end:
            raise self._error("unterminated string", self.pos)
        self._emit(TokenKind.STRING, self.text[self.pos + 1 : end], self.pos, end + 1)

    def _unicode(self, char: str) -> None:
        start = self.pos
        if (name := vocab.UNICODE_LETTERS.get(char)) is not None:
            self._emit(TokenKind.IDENT, name, start, start + 1)
        elif (name := vocab.DOUBLE_STRUCK.get(char)) is not None:
            self._emit(TokenKind.IDENT, name, start, start + 1, verbatim=True)
        elif (relation := vocab.UNICODE_RELATIONS.get(char)) is not None:
            self._emit(TokenKind.RELATION, relation, start, start + 1)
        elif (command := vocab.UNICODE_COMMANDS.get(char)) is not None:
            self._emit(TokenKind.COMMAND, command, start, start + 1)
        elif (kind := vocab.UNICODE_OPERATORS.get(char)) is not None:
            self._emit(kind, kind.value, start, start + 1)
        elif char in vocab.UNICODE_SUPERSCRIPTS:
            self._script_run(vocab.UNICODE_SUPERSCRIPTS, TokenKind.CARET)
        elif char in vocab.UNICODE_SUBSCRIPTS:
            self._script_run(vocab.UNICODE_SUBSCRIPTS, TokenKind.UNDERSCORE)
        else:
            raise self._error(f"unexpected character {char!r}", start)

    def _script_run(self, table: Mapping[str, str], kind: TokenKind) -> None:
        """Spell out a run of Unicode super/subscripts: ``x⁻¹`` lexes as ``x^{-1}``."""
        start = end = self.pos
        plain = ""
        while end < len(self.text) and self.text[end] in table:
            plain += table[self.text[end]]
            end += 1
        self._emit(kind, kind.value, start, end)
        self._emit(TokenKind.LBRACE, "{", start, end)
        for piece in re.findall(r"\d+|\D", plain):
            piece_kind = (
                TokenKind.NUMBER
                if piece.isdigit()
                else _SINGLE_CHARACTER.get(piece, TokenKind.IDENT)
            )
            self._emit(piece_kind, piece, start, end)
        self._emit(TokenKind.RBRACE, "}", start, end)

    # -- backslash sequences ----------------------------------------------------

    def _backslash(self) -> None:
        start = self.pos
        follower = self._peek(1)
        if follower == "\\":
            self._emit(TokenKind.ROW_SEP, "\\\\", start, start + 2)
            if spacing := _ROW_SPACING.match(self.text, self.pos):
                self.pos = spacing.end()
        elif follower.isascii() and follower.isalpha():
            match = _COMMAND_NAME.match(self.text, start + 1)
            assert match is not None
            self.pos = match.end()
            self._command(match.group(), start)
        elif follower in (" ", "\t", "\r", "\n", ",", ";", ":", "!", ">", "~"):
            self._skip(2, explicit=True)  # spacing, or a backslash-newline line continuation
        elif follower == "[":
            self._display("\\[", "\\]")
        elif follower in ("]", "(", ")"):
            self._skip(2)  # math delimiters \( \), and an unmatched \]
        elif follower == "{":
            self._emit(TokenKind.LPAREN, "(", start, start + 2)
        elif follower == "}":
            self._emit(TokenKind.RPAREN, ")", start, start + 2)
        elif follower == "|":
            self._emit(TokenKind.BAR, "|", start, start + 2)
        elif follower == "":
            raise self._error("a backslash must start a LaTeX command", start)
        else:
            raise self._error(f"unsupported LaTeX escape '\\{follower}'", start, start + 2)

    def _command(self, name: str, start: int) -> None:
        if name in ("left", "right"):
            self._left_right(name, start)
        elif name in vocab.SIZE_COMMANDS:
            self._sized_delimiter(start)
        elif name in ("begin", "end"):
            self._environment(name, start)
        elif name in vocab.TEXT_COMMANDS:
            self._text_command(start)
        elif name == "mathbb" and (letter := self._blackboard_letter()) is not None:
            self._emit(TokenKind.IDENT, letter, start, self.pos, verbatim=True)
        elif name in vocab.STYLE_COMMANDS:
            self._style_command()
        elif name in vocab.FONT_DECLARATIONS:
            self._font_declaration()
        elif name in vocab.PLAIN_ENVIRONMENTS and self._next_non_space() == "{":
            self._plain_environment(name, start)
        elif name in ("bgroup", "egroup"):  # plain TeX's braces
            brace = TokenKind.LBRACE if name == "bgroup" else TokenKind.RBRACE
            self._emit(brace, brace.value, start, self.pos)
        elif name == "cr":
            self._emit(TokenKind.ROW_SEP, "\\\\", start, self.pos)
        elif name in vocab.ELLIPSIS_COMMANDS:
            self._emit(TokenKind.ELLIPSIS, TokenKind.ELLIPSIS.value, start, self.pos)
        elif (relation := vocab.RELATION_COMMANDS.get(name)) is not None:
            self._emit(TokenKind.RELATION, relation, start, self.pos)
        elif name in vocab.CONNECTIVE_COMMANDS:
            self._emit(TokenKind.SEMICOLON, ";", start, self.pos)
        elif name in vocab.LAYOUT_COMMANDS:
            self.space_before = True
            self.explicit_space = self.explicit_space or name in vocab.SPACING_COMMANDS
            self.wide_space = self.wide_space or name in vocab.WIDE_SPACING_COMMANDS
        elif name in vocab.LAYOUT_COMMANDS_WITH_ARGUMENT:
            if self._peek() == "*":
                self.pos += 1
            self._read_group(start, name)
            self.space_before = True
        elif name in vocab.LINE_BREAK_COMMANDS:
            self._skip_optional_argument()
            self._emit(TokenKind.NEWLINE, "\n", start, self.pos)
        elif (kind := vocab.OPERATOR_COMMANDS.get(name)) is not None:
            self._emit(kind, kind.value, start, self.pos)
        elif (symbol := vocab.SYMBOL_COMMANDS.get(name)) is not None:
            # \hbar is one symbol, whatever letters spell its name
            self._emit(TokenKind.IDENT, symbol, start, self.pos, verbatim=True)
        else:
            self._emit(TokenKind.COMMAND, name, start, self.pos)

    def _read_group(self, start: int, name: str) -> str:
        """Read a balanced ``{...}`` argument and return its raw contents."""
        self._skip_whitespace()
        if self._peek() != "{":
            raise self._error(f"\\{name} must be followed by a braced argument", start, self.pos)
        depth, cursor = 0, self.pos
        while cursor < len(self.text):
            char = self.text[cursor]
            if char == "\\":
                cursor += 2
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    contents = self.text[self.pos + 1 : cursor]
                    self.pos = cursor + 1
                    return contents
            cursor += 1
        raise self._error(f"unclosed argument of \\{name}", self.pos)

    def _skip_optional_argument(self) -> None:
        """Skip an optional ``[...]`` argument, as in ``\\linebreak[4]``."""
        if self._peek() == "[":
            end = self.text.find("]", self.pos)
            if end != -1:
                self.pos = end + 1

    def _text_command(self, start: int) -> None:
        r"""``\text{otherwise}`` is the name ``otherwise``; ``\text{for all}`` a string."""
        if self._peek() == "*":  # \operatorname*{...}
            self.pos += 1
        contents = self._read_group(start, "text").strip()
        if not contents:
            self.space_before = True
        elif _TEXT_NAME.fullmatch(contents):
            self._emit(TokenKind.IDENT, contents, start, self.pos, verbatim=True)
        else:
            self._emit(TokenKind.STRING, contents, start, self.pos)

    def _blackboard_letter(self) -> str | None:
        r"""``\mathbb{R}``: the double-struck letter, ℝ, read up to the closing brace;
        ``None``, reading nothing, if the argument is not a single letter or digit."""
        match = _BLACKBOARD_ARGUMENT.match(self.text, self.pos)
        if match is None:
            return None
        self.pos = match.end()
        return vocab.double_struck(match.group(1) or match.group(2))

    def _style_command(self) -> None:
        r"""Drop the command and its braces, but lex the argument: ``\mathrm{d}`` is ``d``."""
        self._skip_whitespace()
        if self._peek() != "{":
            return  # \mathbf x: the next token is the argument
        depth, cursor = 0, self.pos
        while cursor < len(self.text):
            char = self.text[cursor]
            if char == "\\":
                cursor += 2
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    self._transparent_braces.add(cursor)
                    self.pos += 1
                    return
            cursor += 1
        raise self._error("unclosed argument of a style command", self.pos)

    def _display(self, opener: str, closer: str) -> None:
        r"""``\[ ... \]`` and ``$$ ... $$`` hold one formula, however its lines break, so
        they read as a ``displaymath`` environment. An unmatched delimiter is skipped."""
        end = self.text.find(closer, self.pos + len(opener))
        if end == -1:
            self._skip(len(opener))
            return
        self._display_ends[end] = len(closer)
        span = Span(self.source, self.pos, self.pos + len(opener))
        self._environments.append(("displaymath", span))
        self._emit(TokenKind.BEGIN, "displaymath", self.pos, self.pos + len(opener))

    def _font_declaration(self) -> None:
        r"""``{\rm d}``: the rest of the group is in the font, and its names are verbatim."""
        depth, cursor = 0, self.pos
        while cursor < len(self.text):
            char = self.text[cursor]
            if char == "\\":
                cursor += 2
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                if depth == 0:
                    break
                depth -= 1
            cursor += 1
        self._font_ends.append(cursor)
        self.space_before = True

    def _next_non_space(self) -> str:
        cursor = self.pos
        while cursor < len(self.text) and self.text[cursor] in _ARGUMENT_SPACE:
            cursor += 1
        return self.text[cursor : cursor + 1]

    def _plain_environment(self, name: str, start: int) -> None:
        r"""``\matrix{a & b \cr c & d}`` lexes like ``\begin{matrix} ... \end{matrix}``."""
        self._skip_whitespace()
        depth, cursor = 0, self.pos
        while cursor < len(self.text):
            char = self.text[cursor]
            if char == "\\":
                cursor += 2
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    break
            cursor += 1
        else:
            raise self._error(f"unclosed argument of \\{name}", self.pos)
        self._closers[cursor] = name
        self._environments.append((name, Span(self.source, start, self.pos + 1)))
        self._emit(TokenKind.BEGIN, name, start, self.pos + 1)

    def _environment(self, name: str, start: int) -> None:
        environment = self._read_group(start, name).strip()
        span = Span(self.source, start, self.pos)
        if name == "begin":
            self._environments.append((environment, span))
            self._emit(TokenKind.BEGIN, environment, start, self.pos)
            if environment in vocab.ENVIRONMENTS_WITH_ARGUMENT:
                self._read_group(start, f"begin{{{environment}}}")
            return
        if not self._environments:
            raise ParseError(f"\\end{{{environment}}} without a matching \\begin", span=span)
        opened, opened_span = self._environments.pop()
        if opened != environment:
            raise ParseError(
                f"\\end{{{environment}}} does not match \\begin{{{opened}}}",
                span=span,
                hint=f"\\begin{{{opened}}} was opened at line {opened_span.line}",
            )
        self._emit(TokenKind.END, environment, start, self.pos)

    def _read_delimiter(self) -> str | None:
        """Read the delimiter after ``\\left``, ``\\right`` or ``\\big``, if there is one."""
        self._skip_whitespace()
        char = self._peek()
        if char in ("(", ")", "[", "]", "|", ".", "/"):
            self.pos += 1
            return char
        if char == "\\":
            follower = self._peek(1)
            if follower in ("{", "}", "|"):
                self.pos += 2
                return "\\" + follower
            match = _COMMAND_NAME.match(self.text, self.pos + 1)
            if match and "\\" + match.group() in (_OPENING | _CLOSING | _BARS | _FLOORS.keys()):
                self.pos = match.end()
                return "\\" + match.group()
        return None

    def _left_right(self, name: str, start: int) -> None:
        delimiter = self._read_delimiter()
        if delimiter is None:
            raise self._error(
                f"\\{name} must be followed by a delimiter",
                start,
                self.pos,
                hint=r"for example \left( ... \right), or \left. ... \right| for an evaluation bar",
            )
        span = Span(self.source, start, self.pos)
        if name == "left":
            role = self._delimiter_role(delimiter)
            self._open_left.append((role, span))
            if role == "bar":
                self._emit(TokenKind.BAR, "|", start, self.pos)
            elif role in ("lfloor", "lceil"):
                self._emit(TokenKind.COMMAND, role, start, self.pos)
            else:  # brackets of every shape group, and '.' is an invisible opener
                self._emit(TokenKind.LPAREN, "(", start, self.pos)
            return
        if not self._open_left:
            raise ParseError(r"'\right' has no matching '\left'", span=span)
        opened, _ = self._open_left.pop()
        if opened == "bar":
            self._emit(TokenKind.BAR, "|", start, self.pos)
        elif opened in ("lfloor", "lceil"):
            self._emit(TokenKind.COMMAND, "r" + opened[1:], start, self.pos)
        else:
            self._emit(TokenKind.RPAREN, ")", start, self.pos)
            if delimiter in _BARS:  # \left. f \right|_{x=0} is an evaluation bar
                self._emit(TokenKind.BAR, "|", start, self.pos)

    def _sized_delimiter(self, start: int) -> None:
        r"""``\big(`` and friends: a delimiter with an explicit size, otherwise ordinary."""
        delimiter = self._read_delimiter()
        if delimiter is None or delimiter == ".":
            self.space_before = True
        elif delimiter in _OPENING:
            self._emit(TokenKind.LPAREN, "(", start, self.pos)
        elif delimiter in _CLOSING:
            self._emit(TokenKind.RPAREN, ")", start, self.pos)
        elif delimiter in _BARS:
            self._emit(TokenKind.BAR, "|", start, self.pos)
        elif delimiter in _FLOORS:
            self._emit(TokenKind.COMMAND, _FLOORS[delimiter], start, self.pos)
        else:
            raise self._error(f"unsupported delimiter {delimiter!r}", start, self.pos)

    @staticmethod
    def _delimiter_role(delimiter: str) -> str:
        if delimiter in _BARS:
            return "bar"
        if delimiter in (r"\lfloor", r"\rfloor"):
            return "lfloor"
        if delimiter in (r"\lceil", r"\rceil"):
            return "lceil"
        return "group"
