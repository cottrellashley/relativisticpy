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
import unicodedata
from collections.abc import Mapping
from dataclasses import replace

from relativisticpy.errors import ParseError
from relativisticpy.language import vocabulary as vocab
from relativisticpy.language.source import Source, Span
from relativisticpy.language.tokens import Token, TokenKind

_NUMBER = re.compile(r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_DIGITS = frozenset("0123456789")
# A group of three digits after a thin space, \, or ~, continuing a number as SI
# groups digits: 10\,000, and 3.141\,592, where the last group of decimals may be
# shorter: 6.022\,140\,76.
_GROUP_SPACE = r"(?:\\,|\\thinspace(?![A-Za-z])[ \t]*|~|[\u00a0\u2009\u202f])"
_DIGIT_GROUP = re.compile(_GROUP_SPACE + r"(?P<digits>[0-9]{3})(?![0-9])")
_LAST_DIGIT_GROUP = re.compile(_GROUP_SPACE + r"(?P<digits>[0-9]{1,2})(?![0-9])")
_FRACTION = re.compile(r"\.[0-9]+")
# An underscore belongs to a name only when a letter or digit follows it:
# ``x_0`` and ``abs_func`` are names, while ``g_{mu}`` is ``g`` with a subscript.
_IDENT = re.compile(r"[A-Za-z](?:[A-Za-z0-9]|_(?=[A-Za-z0-9]))*")
# What may follow an escaped underscore in a name, as in ``geodesic\_equations``.
_NAME_TAIL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9]|_(?=[A-Za-z0-9]))*")
_COMMAND_NAME = re.compile(r"[A-Za-z]+")
# The argument of \mathbb when it is one letter or digit: {R}, { R }, or R itself.
_BLACKBOARD_ARGUMENT = re.compile(r"\s*(?:\{\s*([A-Za-z0-9])\s*\}|([A-Za-z0-9]))")
_TEXT_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
# The spacing in an operator's name, as in \operatorname{arg\,max}, which is one name.
_OPERATOR_SPACING = re.compile(r"\\[,;:! ]|\s|~")
# What may follow a row break: a star, which forbids a page break there (``\\*``), and
# an optional spacing argument, as in ``\\[2pt]``.
_ROW_SPACING = re.compile(
    r"\*?(?:[ \t]*\[\s*-?(?:\d+(?:\.\d*)?|\.\d+)\s*(?:pt|em|ex|mm|cm|in|bp|pc|sp|mu)\s*\])?"
)

# What may stand between the halves of a bracket broken over two rows, as in
# \left( a \right. \\ & \left. + b \right): the row break, alignment and spacing.
_LAYOUT = (
    r"\s|%[^\n]*|\\(?:nonumber|notag|quad|qquad|displaystyle)(?![A-Za-z])"
    r"|\\(?:label|tag|hspace|vspace|phantom|hphantom|vphantom)\*?\{[^{}]*\}|\\[,;:!> ]"
)
_BROKEN_BRACKET = re.compile(
    rf"(?:{_LAYOUT})*\\\\\*?(?:[ \t]*\[[^\]\n]*\])?(?P<cells>(?:{_LAYOUT}|&|\{{\}})*)\\left\s*\."
)

# Unicode's spaces, which typeset as LaTeX's deliberate spacing does: the no-break space
# is ~, the thin space (U+2009) \, and so on.
_UNICODE_SPACE = frozenset("\u00a0\u202f\u205f\u3000") | {chr(c) for c in range(0x2000, 0x200B)}
# Characters that take no room at all, such as the byte order mark that starts some files.
_INVISIBLE = frozenset("\ufeff\u200b\u200c\u200d\u2060")
_SPACE = frozenset(" \t\r~") | _UNICODE_SPACE
_ARGUMENT_SPACE = _SPACE | {"\n"}  # whitespace allowed between a command and its argument

# What Python makes of an escape in a string that is not raw, and what the escape begins
# in LaTeX: "\frac{1}{2}" arrives as a form feed and "rac{1}{2}". The first four are
# never meant; a tab, a line break or a carriage return is, unless the rest of a command
# follows it, as "heta" follows the tab of "\theta".
_PYTHON_ESCAPES = {
    "\a": ("a", "an alert"),
    "\b": ("b", "a backspace"),
    "\f": ("f", "a form feed"),
    "\v": ("v", "a vertical tab"),
    "\t": ("t", "a tab"),
    "\n": ("n", "a line break"),
    "\r": ("r", "a carriage return"),
}
_NEVER_MEANT = frozenset("\a\b\f\v")
# The fewest letters after a tab, line break or carriage return that make it the escape
# of a command: "heta", but not "u", after which a line break may well be meant.
_LEAST_REST = {"\t": 2, "\r": 2, "\n": 3}
_STYLE_EXAMPLES = {"mathcal": "A", "mathscr": "L", "mathfrak": "g", "mathrm": "d", "mathbf": "v"}
_ARGUMENT_EXAMPLES = {"label": "eq:1", "tag": "1", "vspace": "1em", "hspace": "1em"}
# Python's remainder, x % 2: after an operand, and followed only by a number, maybe
# compared, as in x % 2 == 0, or by an operand compared, as in n % k == 0. A word alone,
# as in M := 2e30 % kg, is a comment.
_OPERAND_ENDS = frozenset(
    {TokenKind.IDENT, TokenKind.NUMBER, TokenKind.RPAREN, TokenKind.RBRACKET, TokenKind.RBRACE}
)
_NUMBER_OPERAND = r"\s*\(?\s*[0-9][0-9.]*\s*\)?"
_OPERAND = r"\s*\(?\s*[A-Za-z0-9_.]+\s*\)?"
_COMPARED = rf"\s*(?:==|!=|<=|>=|<|>){_OPERAND}"
_REMAINDER = re.compile(rf"(?:{_NUMBER_OPERAND}(?:{_COMPARED})?|{_OPERAND}{_COMPARED})\s*")
_FRACTION_SLASH = "\N{FRACTION SLASH}"
# The alphabets of Unicode's mathematical letters, and the commands that set them.
_ALPHABETS = {
    "DOUBLE-STRUCK": "mathbb",
    "SCRIPT": "mathcal",
    "FRAKTUR": "mathfrak",
    "SANS-SERIF": "mathsf",
    "MONOSPACE": "mathtt",
    "BOLD": "mathbf",
}

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


_LETTERS = re.compile(r"[A-Za-z]+")
_LONGEST_GREEK = max(map(len, vocab.GREEK))


def _script_letters(text: str, start: int) -> int:
    """How many letters at ``start`` a script without braces takes: one, or a Greek
    letter's name, the longest that the letters begin with."""
    match = _LETTERS.match(text, start)
    letters = match.group() if match else text[start : start + 1]
    for length in range(min(_LONGEST_GREEK, len(letters)), 1, -1):
        if letters[:length] in vocab.GREEK:
            return length
    return 1


def tokenize(source: str | Source, *, recover: bool = False) -> list[Token]:
    """Split source text into tokens. The last token is always ``EOF``.

    What cannot be read raises a :class:`~relativisticpy.errors.ParseError`; with
    ``recover``, it becomes an ``ERROR`` token carrying the error instead, and the
    lexer goes on after it."""
    if isinstance(source, str):
        source = Source(source)
    return Lexer(source, recover=recover).tokenize()


class Lexer:
    """Converts one :class:`Source` into a list of tokens. Use :func:`tokenize`."""

    def __init__(self, source: Source, *, recover: bool = False) -> None:
        self.source = source
        self.text = source.text
        self.recover = recover  # go on past errors, as ERROR tokens
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
        self._ampersands: dict[int, int] = {}  # alignment markers so far in each open row
        # The style commands whose braces are open, by the offset of their closing brace:
        # where the command starts, and how many tokens there were before it, so that the
        # first token in the braces starts at the command, as \mathrm{d} is written.
        self._style_groups: dict[int, tuple[int, int]] = {}

    # -- driver ---------------------------------------------------------------

    def tokenize(self) -> list[Token]:
        if self.source.unreadable is not None:  # as a document's macros that do not expand
            message, start, end = self.source.unreadable
            self._fail(ParseError(message, span=Span(self.source, start, end)))
            self.pos = len(self.text)
        while self.pos < len(self.text):
            if not self.recover:
                self._scan()
                continue
            before = self.pos
            try:
                self._scan()
            except ParseError as error:
                self._fail(error, before)
        for _, span in reversed(self._open_left):  # the innermost first
            self._fail(ParseError(r"'\left' has no matching '\right'", span=span))
        for name, span in reversed(self._environments):
            message = f"\\begin{{{name}}} has no matching \\end{{{name}}}"
            self._fail(ParseError(message, span=span))
        # The input ends just after its last token, which is where what is missing goes,
        # not on a blank line after it.
        end = next(
            (t.span.end for t in reversed(self.tokens) if t.kind is not TokenKind.NEWLINE), 0
        )
        self._emit(TokenKind.EOF, "", end, end)
        return self.tokens

    def _fail(self, error: ParseError, before: int | None = None) -> None:
        """Raise ``error`` or, going on past errors, put an ``ERROR`` token where it is
        and go on after it: after what was being read from ``before``, or, for an
        opener that is never closed, in the place of the opener's token."""
        if not self.recover:
            raise error
        span = error.span if error.span is not None else Span(self.source, self.pos, self.pos)
        if before is None:
            for index, token in enumerate(self.tokens):
                if token.span.start == span.start and token.kind is not TokenKind.ERROR:
                    self.tokens[index] = replace(token, kind=TokenKind.ERROR, error=error)
                    return
        self.tokens.append(Token(TokenKind.ERROR, "", span, error=error))
        if before is not None:
            self.pos = max(self.pos if self.pos > before else before + 1, span.start + 1)
            if self.text[span.start : span.start + 1] == "\n":  # the line still ends there
                newline = Span(self.source, span.start, span.start + 1)
                self.tokens.append(Token(TokenKind.NEWLINE, "\n", newline))

    def _scan(self) -> None:
        char = self.text[self.pos]
        if char in _PYTHON_ESCAPES and self._python_escape(char):
            raise self._raw_string_error(char)
        if self.pos in self._display_ends:
            self._environments.pop()
            end = self.pos + self._display_ends.pop(self.pos)
            self._emit(TokenKind.END, "displaymath", self.pos, end)
        elif self.text.startswith("$$", self.pos):
            self._display("$$", "$$")
        elif char in _SPACE or char == "$":
            self._skip(1, explicit=char == "~" or char in _UNICODE_SPACE)
        elif char in _INVISIBLE:
            self.pos += 1
        elif char == "\n":
            self._emit(TokenKind.NEWLINE, "\n", self.pos, self.pos + 1)
        elif char == "%" and self._remainder():
            raise self._remainder_error()
        elif char in "#%":
            self._skip_comment()
        elif char in _DIGITS or (char == "." and self._peek(1) in _DIGITS):
            self._number()
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
            self._column_marker()
        elif char == "}" and self.pos in self._closers:
            environment = self._closers.pop(self.pos)
            self._environments.pop()
            self._emit(TokenKind.END, environment, self.pos, self.pos + 1)
        elif char == "}" and self.pos in self._transparent_braces:
            self._transparent_braces.discard(self.pos)
            _, before = self._style_groups.pop(self.pos, (0, len(self.tokens)))
            if len(self.tokens) > before:  # the last token in the braces ends with them
                last = self.tokens[-1]
                self.tokens[-1] = replace(
                    last, span=Span(self.source, last.span.start, self.pos + 1)
                )
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
        written = start
        for command, before in self._style_groups.values():
            if before == len(self.tokens):  # the first token in a style command's braces
                written = min(written, command)
        span = Span(self.source, written, end)
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
        if kind in (TokenKind.ROW_SEP, TokenKind.BEGIN):  # a row starts
            self._ampersands[len(self._environments)] = 0
        self.pos = end

    def _skip(self, count: int, *, explicit: bool = False) -> None:
        self.pos += count
        self.space_before = True
        self.explicit_space = self.explicit_space or explicit

    def _peek(self, offset: int = 0) -> str:
        return self.text[self.pos + offset : self.pos + offset + 1]

    def _error(
        self,
        message: str,
        start: int,
        end: int | None = None,
        hint: str | None = None,
        *,
        label: str = "",
        notes: tuple[str, ...] = (),
    ) -> ParseError:
        end = start + 1 if end is None else end
        span = Span(self.source, start, end).labelled(label)
        return ParseError(message, span=span, hint=hint, notes=notes)

    def _python_escape(self, char: str) -> bool:
        """Whether ``char`` is what Python made of an escape that begins a LaTeX command,
        in a string that was not raw: always for an alert, a backspace, a form feed or a
        vertical tab, and for a tab, line break or carriage return that the rest of a
        command follows, as ``heta`` follows the tab of ``"\\theta"``."""
        if char in _NEVER_MEANT:
            return True
        if char == "\r" and self._peek(1) == "\n":
            return False  # a line ends, as on Windows
        command = self._escaped_command(char)
        least = _LEAST_REST[char]
        if command is None or len(command) - 1 < least:
            return False
        rest = command[1:]
        return rest not in vocab.GREEK and rest not in vocab.FUNCTION_COMMANDS

    def _escaped_command(self, char: str) -> str | None:
        """The command an escape begins, if the letters after ``char`` complete one:
        ``theta`` after a tab and ``heta``, but not after a tab and ``hetas``."""
        letter = _PYTHON_ESCAPES[char][0]
        match = _LETTERS.match(self.text, self.pos + 1)
        name = letter + (match.group() if match else "")
        return name if name in vocab.COMMANDS or name in vocab.UNSUPPORTED_COMMANDS else None

    def _raw_string_error(self, char: str) -> ParseError:
        """The error for an escape Python turned into a control character."""
        letter, what = _PYTHON_ESCAPES[char]
        command = self._escaped_command(char)
        where = f" in \\{command}" if command else ""
        start = self.text.rfind("\n", 0, self.pos) + 1
        end = self.text.find("\n", self.pos + 1)
        line = self.text[start : len(self.text) if end == -1 else end]
        escapes = {ord(c): "\\" + e for c, (e, _) in _PYTHON_ESCAPES.items()}
        written = line.translate(escapes).strip()
        if len(written) > 50:
            written = f"\\{command or letter}"
        return self._error(
            f"Python turned \\{letter}{where} into {what}",
            self.pos,
            label=f"{what}, not \\{letter}",
            notes=(
                "in a Python string that is not raw, \\a, \\b, \\f, \\n, \\r, \\t and \\v "
                "are control characters",
            ),
            hint=f'write the string as a raw string, with r before its quote: r"{written}"',
        )

    def _number(self) -> None:
        """A number. Digits grouped in threes by thin spaces are one number, as SI writes
        them: ``299\\,792\\,458`` is 299792458, not a product."""
        start = self.pos
        match = _NUMBER.match(self.text, start)
        assert match is not None  # guaranteed by the caller's first-character check
        digits, end = match.group(), match.end()
        _, point, decimals = digits.partition(".")
        if "e" not in digits.lower() and (not point or len(decimals) == 3):
            while group := _DIGIT_GROUP.match(self.text, end):
                digits, end = digits + group.group("digits"), group.end()
            grouped = end > match.end()
            if grouped and point and (last := _LAST_DIGIT_GROUP.match(self.text, end)):
                digits, end = digits + last.group("digits"), last.end()  # 6.022\,140\,76
            if grouped and not point and (fraction := _FRACTION.match(self.text, end)):
                digits, end = digits + fraction.group(), fraction.end()  # 1\,000.5
        self._emit(TokenKind.NUMBER, digits, start, end)

    def _identifier(self) -> None:
        """A name; LaTeX's escaped underscore joins its parts: ``geodesic\\_equations``.

        A name that is a script without braces is one letter, as TeX reads it, or a
        Greek letter spelled out, as plain text writes ``\\mu``: ``R^a_b`` is
        ``R^{a}_{b}``, ``x^ab`` is ``x^{a} b`` and ``x^mu`` is ``x^{mu}``."""
        start = self.pos
        if self._at_script():
            end = start + _script_letters(self.text, start)
            self._emit(TokenKind.IDENT, self.text[start:end], start, end)
            return
        match = _IDENT.match(self.text, start)
        assert match is not None
        end = match.end()
        while self.text.startswith("\\_", end):
            tail = _NAME_TAIL.match(self.text, end + 2)
            if tail is None:
                break
            end = tail.end()
        self._emit(TokenKind.IDENT, self.text[start:end].replace("\\_", "_"), start, end)

    def _at_script(self) -> bool:
        """Whether what comes next is a script without braces: after ``^`` or ``_``, and
        not in a style command's braces, where ``x_\\mathrm{max}`` keeps its name whole."""
        previous = self.tokens[-1] if self.tokens else None
        if previous is None or previous.kind not in (TokenKind.CARET, TokenKind.UNDERSCORE):
            return False
        while self._font_ends and self.pos >= self._font_ends[-1]:
            self._font_ends.pop()
        return not (self._transparent_braces or self._font_ends)

    def _remainder(self) -> bool:
        """Whether a ``%`` is Python's remainder, as in ``x % 2``, rather than a comment:
        in plain text, after an operand on its line, and followed by nothing but a short
        expression, which a comment would not be."""
        if self.source.latex or self._environments or not self.tokens:
            return False
        if self.tokens[-1].kind not in _OPERAND_ENDS:
            return False
        line_start = self.text.rfind("\n", 0, self.pos) + 1
        if "\\" in self.text[line_start : self.pos]:
            return False  # LaTeX, in which % is always a comment
        end = self.text.find("\n", self.pos)
        rest = self.text[self.pos + 1 : len(self.text) if end == -1 else end]
        return _REMAINDER.fullmatch(rest) is not None

    def _remainder_error(self) -> ParseError:
        previous = self.tokens[-1]
        end = self.text.find("\n", self.pos)
        divisor = self.text[self.pos + 1 : len(self.text) if end == -1 else end].strip()
        dividend = previous.span.text
        return self._error(
            f"'%' starts a comment, so {dividend} % {divisor} is read as {dividend}",
            self.pos,
            label="the rest of the line is a comment",
            notes=("as in LaTeX, '%' starts a comment: there is no remainder operator",),
            hint=f"for the remainder of {dividend} divided by {divisor}, write "
            f"{dividend} - {divisor} \\lfloor {dividend}/{divisor} \\rfloor; for a comment, "
            "write #",
        )

    def _skip_comment(self) -> None:
        end = self.text.find("\n", self.pos)
        self.pos = len(self.text) if end == -1 else end
        self.space_before = True

    def _skip_whitespace(self) -> None:
        while self._peek() in _ARGUMENT_SPACE:
            self.pos += 1

    def _in_alignment(self, environments: frozenset[str] = vocab.ALIGNMENT_ENVIRONMENTS) -> bool:
        """Whether the innermost open environment is one of ``environments``."""
        if not self._environments:
            return False
        return self._environments[-1][0] in environments

    def _column_marker(self, count: int = 1) -> None:
        """Count alignment markers in the row. In ``align``, columns come in pairs, each
        pair an equation, so a second marker separates equations as wide spacing does:
        ``a &= 1 & b &= 2``."""
        depth = len(self._environments)
        self._ampersands[depth] = self._ampersands.get(depth, 0) + count
        if self._ampersands[depth] % 2 == 0 and self._in_alignment(vocab.COLUMN_PAIR_ENVIRONMENTS):
            self.wide_space = True

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
        if (name := vocab.UNICODE_LETTERS.get(char) or vocab.DOUBLE_STRUCK.get(char)) is not None:
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
            raise self._unexpected(char, start)

    def _unexpected(self, char: str, start: int) -> ParseError:
        """The error for a character the language has no use for: named by its code
        point, since it may be invisible or an accent that would sit on a quote mark,
        and with what to write instead when that is clear."""
        code = f"U+{ord(char):04X}"
        name = unicodedata.name(char, "")
        described = f"{code} {name}".strip()
        hint = None
        if (accent := vocab.COMBINING_ACCENTS.get(char)) is not None or unicodedata.combining(char):
            before = self.text[start - 1] if start else ""
            letter = vocab.UNICODE_LETTERS.get(before)
            base = f"\\{letter}" if letter in vocab.GREEK else (letter or before)
            if accent is not None and base:
                hint = f"write \\{accent}{{{base}}}"
            message = f"unexpected character {described}, an accent on {before}" if before else ""
            return self._error(message or f"unexpected character {described}", start, hint=hint)
        plain = unicodedata.normalize("NFKC", char)
        if _FRACTION_SLASH in plain:  # a vulgar fraction: ½ is 1, the slash, then 2
            numerator, _, denominator = plain.partition(_FRACTION_SLASH)
            hint = f"write \\frac{{{numerator}}}{{{denominator}}}"
        elif plain != char and plain.isascii() and plain.isprintable():
            alphabet = next((c for word, c in _ALPHABETS.items() if word in name.split()), None)
            if alphabet is not None and "MATHEMATICAL" in name:
                hint = f"write \\{alphabet}{{{plain}}}"
            else:
                hint = f"write {plain}"
        if char.isascii() and char.isprintable():
            return self._error(f"unexpected character '{char}'", start)
        shown = f"'{char}' " if char.isprintable() and not char.isspace() else ""
        return self._error(f"unexpected character {shown}({described})", start, hint=hint)

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
            if self._in_alignment(vocab.SINGLE_EQUATION_ENVIRONMENTS):
                self._emit(TokenKind.NEWLINE, "\n", start, start + 2)  # the equation goes on
            else:
                self._emit(TokenKind.ROW_SEP, "\\\\", start, start + 2)
            if spacing := _ROW_SPACING.match(self.text, self.pos):
                self.pos = spacing.end()  # \\*[2pt]: neither is part of the next row
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
            self._text_command(name, start)
        elif name == "mathbb" and (letter := self._blackboard_letter()) is not None:
            self._emit(TokenKind.IDENT, letter, start, self.pos, verbatim=True)
        elif name in vocab.STYLE_COMMANDS:
            self._style_command(name, start)
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
            self._read_group(start, name, _ARGUMENT_EXAMPLES.get(name, "x"))
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

    def _read_group(self, start: int, name: str, example: str = "...") -> str:
        """Read a balanced ``{...}`` argument and return its raw contents."""
        self._skip_whitespace()
        if self._peek() != "{":
            raise self._error(
                f"\\{name} must be followed by a braced argument, as in \\{name}{{{example}}}",
                start,
                self.pos,
            )
        cursor = self._closing_brace(name)
        contents = self.text[self.pos + 1 : cursor]
        self.pos = cursor + 1
        return contents

    def _closing_brace(self, name: str) -> int:
        """Where the braced argument of ``\\name`` at the cursor closes."""
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
                    return cursor
            cursor += 1
        written = self.text[self.pos : self.pos + 30].rstrip()
        example = f", as in \\{name}{written}}}" if "\n" not in written and written else ""
        raise self._error(
            f"unclosed argument of \\{name}",
            self.pos,
            label="this '{' is never closed",
            hint=f"add '}}' where the argument ends{example}",
        )

    def _skip_optional_argument(self) -> None:
        """Skip an optional ``[...]`` argument, as in ``\\linebreak[4]``."""
        if self._peek() == "[":
            end = self.text.find("]", self.pos)
            if end != -1:
                self.pos = end + 1

    def _text_command(self, name: str, start: int) -> None:
        r"""``\text{otherwise}`` is the name ``otherwise``; ``\text{for all}`` a string. An
        operator's name is one name, however it is spaced: ``\operatorname{arg\,max}`` is
        ``argmax``."""
        if self._peek() == "*":  # \operatorname*{...}
            self.pos += 1
        example = "tr" if name == "operatorname" else "word"
        contents = self._read_group(start, name, example).strip()
        if name == "operatorname":
            contents = _OPERATOR_SPACING.sub("", contents)
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

    def _style_command(self, name: str, start: int) -> None:
        r"""Drop the command and its braces, but lex the argument: ``\mathrm{d}`` is ``d``,
        written as ``\mathrm{d}``, so that its span is the whole command."""
        self._skip_whitespace()
        if self._peek() != "{":
            if self.pos == len(self.text) or self._peek() in "})]&":
                example = _STYLE_EXAMPLES.get(name, "x")
                raise self._error(
                    f"\\{name} needs an argument, as in \\{name}{{{example}}}", start, self.pos
                )
            return  # \mathbf x: the next token is the argument
        cursor = self._closing_brace(name)
        self._transparent_braces.add(cursor)
        self._style_groups[cursor] = (start, len(self.tokens))
        self.pos += 1

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
        cursor = self._closing_brace(name)
        self._closers[cursor] = name
        self._environments.append((name, Span(self.source, start, self.pos + 1)))
        self._emit(TokenKind.BEGIN, name, start, self.pos + 1)

    def _environment(self, name: str, start: int) -> None:
        environment = self._read_group(start, name, "equation").strip()
        span = Span(self.source, start, self.pos)
        if name == "begin":
            self._environments.append((environment, span))
            self._emit(TokenKind.BEGIN, environment, start, self.pos)
            if environment in vocab.ENVIRONMENTS_WITH_ARGUMENT:
                self._read_group(start, f"begin{{{environment}}}", "cc")
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
        opened, opened_span = self._open_left.pop()
        if delimiter == "." and opened != "invisible":
            broken = _BROKEN_BRACKET.match(self.text, self.pos)
            if broken is not None:  # \left( a \right. \\ & \left. + b \right): one bracket
                self._open_left.append((opened, opened_span))
                self.pos = broken.end()
                self.space_before = True
                self._ampersands[len(self._environments)] = 0
                self._column_marker(broken.group("cells").count("&"))
                return
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
        elif delimiter == "/":  # a \big/ is a division, only taller
            self._emit(TokenKind.SLASH, "/", start, self.pos)
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
        return "invisible" if delimiter == "." else "group"
