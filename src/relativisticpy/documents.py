r"""LaTeX documents: their display equations, with the author's macros expanded.

An equation in a paper sits inside a document. It is set off by
``\begin{equation}`` and ``\end{equation}``, ``$$``, ``\[`` or an alignment
environment, often through shorthands the author defined (``\be`` and
``\ee``), and it is written with the author's macros. A :class:`Document`
finds every display equation and expands the document's macros in it. Each
equation becomes a :class:`~relativisticpy.language.Source` that knows where
it sits in the file, so that errors point at the right line::

    >>> document = Document(r'''
    ... \newcommand{\be}{\begin{equation}}
    ... \newcommand{\ee}{\end{equation}}
    ... \def\half{\frac{1}{2}}
    ... The kinetic energy is
    ... \be E = \half m v^2 \ . \ee
    ... ''')
    >>> [(equation.source.origin[0], equation.source.text) for equation in document.equations()]
    [(6, '\\begin{equation} E = \\frac{1}{2} m v^2 \\ . \\end{equation}')]
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from collections.abc import Iterator, Mapping
from dataclasses import InitVar, dataclass, field, replace
from pathlib import Path

from relativisticpy.language import Source
from relativisticpy.language.source import Piece

#: Environments that display equations.
DISPLAY_ENVIRONMENTS = frozenset(
    {
        "equation", "equation*", "eqnarray", "eqnarray*", "align", "align*",
        "gather", "gather*", "multline", "multline*", "flalign", "flalign*",
        "displaymath",
    }
)  # fmt: skip

_DEFINITION = re.compile(
    r"\\(?:(?:re|provide)?newcommand\*?|DeclareRobustCommand\*?|DeclareMathOperator\*?"
    r"|[gex]?def|let)(?![A-Za-z])"
)
_ENVIRONMENT_DEFINITION = re.compile(r"\\(?:re)?newenvironment\*?(?![A-Za-z])")
_CONTROL_SEQUENCE = re.compile(r"\\([A-Za-z]+|.)", re.DOTALL)
_PARAMETERS = re.compile(r"(?:#\d)*")
_PARAMETER = re.compile(r"#(\d)")
_ARGUMENT_COUNT = re.compile(r"\s*\[\s*(\d)\s*\]")
_MAX_EXPANSIONS = 10_000
_SECTION = re.compile(r"\\(?:chapter|section)\*?\s*[{\[]")
# A control word that ends a text, which a letter after it would join: \alpha.
_TRAILING_WORD = re.compile(r"\\[A-Za-z]+$")
# What hides text from TeX: \iffalse up to its \else or \fi, and the comment environment.
_HIDING = re.compile(r"\\(?:if[A-Za-z]*|else|fi)(?![A-Za-z])|\\(?:begin|end)\s*\{comment\}")
# TeX's conditionals, which a skipped \iffalse counts to find its own \fi; a document
# declares more with \newif. (\iff is a relation, and \ifthenelse a macro.)
_CONDITIONALS = frozenset(
    {
        "if", "ifcat", "ifnum", "ifdim", "ifodd", "ifvmode", "ifhmode", "ifmmode",
        "ifinner", "ifvoid", "ifhbox", "ifvbox", "ifx", "ifeof", "iftrue", "iffalse",
        "ifcase", "ifdefined", "ifcsname", "iffontchar", "ifincsname", "ifpdf",
    }
)  # fmt: skip
_NEWIF = re.compile(r"\\newif\s*\\(if[A-Za-z]+)")
_DISPLAY_CHARACTER = re.compile(r"[\\$]")
_ENVIRONMENT = re.compile(r"\s*\{([^}]*)\}")
_BEGIN_OR_END = re.compile(r"\\(begin|end)\s*\{\s*([^}]*?)\s*\}")
_INLINE_END = re.compile(r"(?<!\\)\$")
# Display maths in Markdown, $$...$$, outside code and HTML comments. Code blocks, code
# spans and comments are matched only so that what they contain is skipped, as when
# rendered: fenced code; code indented by four spaces after a blank line; and code
# spans, which are closed by as many backticks as open them and cannot cross a blank
# line, so that a stray backtick hides nothing.
_MARKDOWN = re.compile(
    r"^ {0,3}(?P<fence>`{3,}|~{3,}).*?^ {0,3}(?P=fence)[ \t]*$"
    r"|<!--.*?-->"
    r"|(?:\A|^[ \t]*\n)(?P<indented>(?: {4}|\t)[^\n]*(?:\n(?:[ \t]*\n)*(?: {4}|\t)[^\n]*)*)"
    r"|(?<!`)(?P<ticks>`+)(?!`)(?:(?!\n[ \t]*\n).)*?(?<!`)(?P=ticks)(?!`)"
    r"|\$\$(?P<math>.+?)\$\$",
    re.DOTALL | re.MULTILINE,
)
_LIST_ITEM = re.compile(r" {0,3}(?:[-+*]|\d{1,9}[.)])(?:[ \t]|$)")


@dataclass(frozen=True)
class Macro:
    """A macro the document defines: ``\\name`` with ``arity`` arguments.

    When ``default`` is not ``None`` the first argument is optional, in brackets.
    An ``alias``, made by ``\\let\\name\\command`` of a command the document does not
    define, stands for that command, which is never expanded again.
    """

    name: str
    arity: int
    body: str
    default: str | None = None
    alias: bool = False


@dataclass(frozen=True)
class Equation:
    """A display equation of a document, as RelativisticPy will read it, and the
    number of the section (or chapter) it is in."""

    source: Source
    environment: str
    section: int = 0

    @property
    def line(self) -> int:
        return self.source.origin[0]


@dataclass
class Document:
    """A LaTeX document, read for its equations. ``known`` are macros defined before
    the document's own, as those of its packages are."""

    text: str
    name: str = "<document>"
    known: InitVar[Mapping[str, Macro] | None] = None
    macros: dict[str, Macro] = field(init=False)

    def __post_init__(self, known: Mapping[str, Macro] | None) -> None:
        self.text = strip_comments(self.text)
        # TeX skips what \iffalse and the comment environment hide, definitions included.
        visible = _blank(self.text, _hidden(self.text))
        self.macros, definitions = _read_definitions(visible, known)
        # Definitions such as \newcommand{\be}{\begin{equation}} are not displays.
        self._body = _blank(visible, definitions)
        self._lines = [0, *(match.end() for match in re.finditer("\n", self.text))]
        self._sections = [match.start() for match in _SECTION.finditer(self._body)]

    @classmethod
    def read(cls, path: str | Path) -> Document:
        """Read a document from a file, with the macros of its local packages
        (``\\usepackage{macros}`` with a ``macros.sty`` beside it) and of the files it
        includes with ``\\input`` and ``\\include``. Its own definitions come first."""
        path = Path(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        packages: dict[str, Macro] = {}
        for extra in _local_files(strip_comments(text), path.parent):
            found = find_macros(strip_comments(extra.read_text(encoding="utf-8", errors="replace")))
            packages = {**found, **packages}
        return cls(text, str(path), packages)

    def equations(self) -> list[Equation]:
        """Every display equation, in order, with the document's macros expanded. An
        equation whose macros cannot be expanded is given as written, and reading it
        raises a :class:`~relativisticpy.errors.ParseError` that says why."""
        delimiters = {
            name: _BEGIN_OR_END.sub(r"\\\1{\2}", macro.body.strip())
            for name, macro in self.macros.items()
            if macro.arity == 0 and _is_delimiter(macro.body.strip())
        }
        found = []
        for start, end, environment in _displays(self._body, delimiters):
            line = bisect_right(self._lines, start)
            origin = (line, start - self._lines[line - 1] + 1)
            section = bisect_left(self._sections, start)
            found.append(Equation(self._source(start, end, origin), environment, section))
        return found

    def _source(self, start: int, end: int, origin: tuple[int, int]) -> Source:
        """The source of the display at ``[start, end)``, its macros expanded."""
        written = self.text[start:end]
        try:
            text, pieces = _expand(written, self.macros)
        except _Unexpandable as error:
            unreadable = (str(error), error.start, error.end)
            return Source(written, self.name, origin, latex=True, unreadable=unreadable)
        if text == written:
            return Source(text, self.name, origin, latex=True)
        return Source(text, self.name, origin, latex=True, written=written, pieces=pieces)


def _local_files(text: str, folder: Path) -> list[Path]:
    """The local packages and included files a document names, where they exist."""
    found = []
    for command, names in re.findall(
        r"\\(usepackage|input|include)(?:\[[^\]]*\])?\{([^}]*)\}", text
    ):
        suffix = ".sty" if command == "usepackage" else ".tex"
        for name in names.split(","):
            for candidate in (folder / name.strip(), folder / f"{name.strip()}{suffix}"):
                if candidate.is_file() and candidate not in found:
                    found.append(candidate)
                    break
    return found


def strip_comments(text: str) -> str:
    """Remove ``%`` comments, keeping line breaks so that lines keep their numbers."""
    return re.sub(r"(?<!\\)((?:\\\\)*)%[^\n]*", r"\1", text)


def find_macros(text: str) -> dict[str, Macro]:
    """The macros a document defines with ``\\newcommand``, ``\\def`` and friends."""
    return _read_definitions(text)[0]


def _read_definitions(
    text: str, known: Mapping[str, Macro] | None = None
) -> tuple[dict[str, Macro], list[tuple[int, int]]]:
    """The macros a document defines, after the ``known`` ones, and where their
    definitions, and those of environments, are."""
    macros = dict(known or {})
    spans = []
    for match in _DEFINITION.finditer(text):
        try:
            macro, end = _definition(text, match, macros)
        except ValueError:
            continue  # a definition we cannot read is one we do not expand
        spans.append((match.start(), end))
        if not (macro.name in macros and match.group().startswith("\\provide")):
            macros[macro.name] = macro
    for match in _ENVIRONMENT_DEFINITION.finditer(text):
        try:
            spans.append((match.start(), _environment_definition_end(text, match.end())))
        except ValueError:
            continue
    return macros, spans


def expand(text: str, macros: Mapping[str, Macro]) -> str:
    """Expand the given macros wherever they occur in ``text``, recursively."""
    return _expand(text, macros)[0]


class _Unexpandable(ValueError):
    """Macros that cannot be expanded, and the extent of the first of them in the text."""

    def __init__(self, message: str, start: int, end: int) -> None:
        super().__init__(message)
        self.start, self.end = start, end


# A run of the text being expanded: its length, and the characters of the written text
# it came from, copied as they are or, if not, expanded from them.
_Run = tuple[int, int, int, bool]


def _expand(text: str, macros: Mapping[str, Macro]) -> tuple[str, tuple[Piece, ...]]:
    """Expand the macros in ``text``, and say where each piece of the result came from.

    What is still to be read is ``rest``, and ``runs`` says where its characters came
    from, its first run last. A macro's expansion is put back in front of what follows
    it, to be read again, unless the macro is an alias of a command: ``\\let\\old\\vec``
    makes ``\\old`` the command ``\\vec``, even once ``\\vec`` is redefined."""
    rest = text
    runs: list[_Run] = [(len(text), 0, len(text), True)] if text else []
    done: list[str] = []
    done_runs: list[_Run] = []
    position = expansions = 0
    while (start := rest.find("\\", position)) != -1:
        match = _CONTROL_SEQUENCE.match(rest, start)
        macro = macros.get(match.group(1)) if match else None
        if match is None or macro is None:
            position = match.end() if match else len(rest)
            continue
        before = rest[max(0, start - 64) : start] or (done[-1] if done else "")
        if start:
            done.append(rest[:start])
            done_runs.extend(_take(runs, start))
        head = _take(runs, match.end() - start)
        where = min(run[1] for run in head), max(run[2] for run in head)
        expansions += 1
        if expansions > _MAX_EXPANSIONS:
            raise _Unexpandable(f"the macro \\{macro.name} expands without end", *where)
        try:
            arguments, end = _arguments(rest, match.end(), macro)
        except ValueError as error:
            raise _Unexpandable(f"cannot expand \\{macro.name}: {error}", *where) from None
        invocation = head + _take(runs, end - match.end())
        low, high = min(run[1] for run in invocation), max(run[2] for run in invocation)
        body = macro.body if macro.alias else _substitute(macro.body, arguments)
        # Keep the expansion from joining a command before it or a letter after it:
        # \alpha\R, with \R defined as R, is \alpha R, not \alphaR.
        if body[:1].isalpha() and _TRAILING_WORD.search(before):
            body = " " + body
        if _TRAILING_WORD.search(body) and rest[end : end + 1].isalpha():
            body += " "
        if macro.alias:
            if body:
                done.append(body)
                done_runs.append((len(body), low, high, False))
            rest = rest[end:]
        else:
            rest = body + rest[end:]
            if body:
                runs.append((len(body), low, high, False))
        position = 0
    done.append(rest)
    done_runs.extend(_take(runs, len(rest)))
    return "".join(done), _pieces(done_runs)


def _take(runs: list[_Run], length: int) -> list[_Run]:
    """Take the runs of the first ``length`` characters off ``runs``."""
    taken = []
    while length > 0:
        size, start, end, copied = runs.pop()
        if size > length:  # the run goes on beyond them
            split = start + length if copied else start
            taken.append((length, start, split if copied else end, copied))
            runs.append((size - length, split, end, copied))
            break
        taken.append((size, start, end, copied))
        length -= size
    return taken


def _pieces(runs: list[_Run]) -> tuple[Piece, ...]:
    """The pieces of an expansion, from its runs, neighbours that continue each other
    being one piece."""
    pieces: list[Piece] = []
    offset = 0
    for size, start, end, copied in runs:
        if size == 0:
            continue
        last = pieces[-1] if pieces else None
        if last is not None and last.copied and copied and last.written_end == start:
            pieces[-1] = last._replace(written_end=end)
        elif last is not None and not (last.copied or copied) and last[1:3] == (start, end):
            pass  # more of the same expansion
        else:
            pieces.append(Piece(offset, start, end, copied))
        offset += size
    return tuple(pieces)


def _substitute(body: str, arguments: list[str]) -> str:
    r"""A macro's body with its arguments in place of ``#1``, ``#2`` and so on, each kept
    apart from a command beside it, as TeX keeps them: ``\lvert#1\rvert`` with ``x`` is
    ``\lvert x\rvert``, not ``\lvertx\rvert``."""

    result, done = "", 0
    for match in _PARAMETER.finditer(body):
        number = int(match.group(1))
        if not 1 <= number <= len(arguments):
            continue
        result += body[done : match.start()]
        value = arguments[number - 1]
        if value[:1].isalpha() and _TRAILING_WORD.search(result[-64:]):
            value = " " + value
        if _TRAILING_WORD.search(value) and body[match.end() : match.end() + 1].isalpha():
            value += " "
        result += value
        done = match.end()
    return result + body[done:]


# -- reading definitions ----------------------------------------------------------


def _definition(text: str, match: re.Match[str], macros: Mapping[str, Macro]) -> tuple[Macro, int]:
    command = match.group()
    position = _skip_space(text, match.end())
    if command.startswith("\\let"):
        name, position = _defined_name(text, position)
        position = _skip_space(text, position)
        if text.startswith("=", position):
            position = _skip_space(text, position + 1)
        target = _CONTROL_SEQUENCE.match(text, position)
        if target is None:
            raise ValueError("\\let without a target")
        # \let copies what the target means now, not its name: after \let\old\vec,
        # \renewcommand{\vec}[1]{\old{...}} does not make \old expand to itself
        if (meaning := macros.get(target.group(1))) is not None:
            return replace(meaning, name=name), target.end()
        return Macro(name, 0, target.group(), alias=True), target.end()
    if command.startswith(("\\def", "\\gdef", "\\edef", "\\xdef")):
        name, position = _defined_name(text, position)
        parameters = _PARAMETERS.match(text, position)
        assert parameters is not None
        if not text.startswith("{", parameters.end()):
            raise ValueError("delimited parameters are not supported")
        body, end = _group(text, parameters.end())
        return Macro(name, parameters.group().count("#"), body), end
    name, position = _defined_name(text, position)
    if command.startswith("\\DeclareMathOperator"):
        body, end = _group(text, _skip_space(text, position))
        return Macro(name, 0, f"\\operatorname{{{body}}}"), end
    arity, default = 0, None
    if count := _ARGUMENT_COUNT.match(text, position):
        arity, position = int(count.group(1)), count.end()
        position = _skip_space(text, position)
        if text.startswith("[", position):
            end = _closing_bracket(text, position)
            default, position = text[position + 1 : end], end + 1
    body, end = _group(text, _skip_space(text, position))
    return Macro(name, arity, body, default), end


def _environment_definition_end(text: str, position: int) -> int:
    """Where ``\\newenvironment{name}[n][default]{begin}{end}`` ends, from after its
    command."""
    _, position = _group(text, _skip_space(text, position))
    if count := _ARGUMENT_COUNT.match(text, position):
        position = _skip_space(text, count.end())
        if text.startswith("[", position):
            position = _closing_bracket(text, position) + 1
    _, position = _group(text, _skip_space(text, position))
    _, position = _group(text, _skip_space(text, position))
    return position


def _defined_name(text: str, position: int) -> tuple[str, int]:
    """The name after ``\\newcommand`` or ``\\def``: ``\\name`` or ``{\\name}``."""
    braced = text.startswith("{", position)
    if braced:
        position = _skip_space(text, position + 1)
    match = _CONTROL_SEQUENCE.match(text, position)
    if match is None:
        raise ValueError("expected a command name")
    position = match.end()
    if braced:
        position = _skip_space(text, position)
        if not text.startswith("}", position):
            raise ValueError("expected '}' after the command name")
        position += 1
    return match.group(1), position


def _arguments(text: str, position: int, macro: Macro) -> tuple[list[str], int]:
    arguments = []
    for number in range(macro.arity):
        if number == 0 and macro.default is not None:
            cursor = _skip_space(text, position)
            if text.startswith("[", cursor):
                end = _closing_bracket(text, cursor)
                arguments.append(text[cursor + 1 : end])
                position = end + 1
            else:
                arguments.append(macro.default)
            continue
        position = _skip_space(text, position)
        if text.startswith("{", position):
            argument, position = _group(text, position)
        elif (command := _CONTROL_SEQUENCE.match(text, position)) is not None:
            argument, position = command.group(), command.end()
        elif position < len(text):
            argument, position = text[position], position + 1
        else:
            raise ValueError(f"\\{macro.name} is missing an argument")
        arguments.append(argument)
    return arguments, position


def _group(text: str, position: int) -> tuple[str, int]:
    """The contents of the braced group starting at ``position``, and where it ends."""
    if not text.startswith("{", position):
        raise ValueError("expected '{'")
    depth = 0
    cursor = position
    while cursor < len(text):
        char = text[cursor]
        if char == "\\":
            cursor += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[position + 1 : cursor], cursor + 1
        cursor += 1
    raise ValueError("unclosed '{'")


def _closing_bracket(text: str, position: int) -> int:
    depth = 0
    for cursor in range(position, len(text)):
        if text[cursor] == "{":
            depth += 1
        elif text[cursor] == "}":
            depth -= 1
        elif text[cursor] == "]" and depth == 0:
            return cursor
    raise ValueError("unclosed '['")


def _skip_space(text: str, position: int) -> int:
    while position < len(text) and text[position] in " \t\n\r":
        position += 1
    return position


# -- finding displays --------------------------------------------------------------


def _hidden(text: str) -> list[tuple[int, int]]:
    """Where TeX skips text: from ``\\iffalse`` to its ``\\else`` or ``\\fi``, and in the
    ``comment`` environment."""
    conditionals = _CONDITIONALS | set(_NEWIF.findall(text))
    spans = []
    start: int | None = None
    depth = 0  # of conditionals inside \iffalse; 0 in a comment environment
    for match in _HIDING.finditer(text):
        word = match.group()
        if start is None:
            if word == "\\iffalse":
                start, depth = match.start(), 1
            elif word.startswith("\\begin"):
                start, depth = match.start(), 0
        elif depth == 0:
            if word.startswith("\\end"):
                spans.append((start, match.end()))
                start = None
        elif word == "\\else" and depth == 1:  # \iffalse ... \else: what follows is read
            spans.append((start, match.end()))
            start = None
        elif word == "\\fi":
            depth -= 1
            if depth == 0:
                spans.append((start, match.end()))
                start = None
        elif word[1:] in conditionals:
            depth += 1
    if start is not None:
        spans.append((start, len(text)))
    return spans


def _blank(text: str, spans: list[tuple[int, int]]) -> str:
    """``text`` with ``spans`` blanked out, keeping its line breaks and every offset."""
    pieces = []
    done = 0
    for start, end in sorted(spans):
        start = max(start, done)
        if end <= start:
            continue
        pieces.append(text[done:start])
        pieces.append(re.sub(r"[^\n]", " ", text[start:end]))
        done = end
    pieces.append(text[done:])
    return "".join(pieces)


def _is_delimiter(body: str) -> bool:
    if body in ("$$", "\\[", "\\]"):
        return True
    match = re.fullmatch(r"\\(begin|end)\s*\{([^}]*)\}", body)
    return match is not None and match.group(2).strip() in DISPLAY_ENVIRONMENTS


def _displays(text: str, delimiters: dict[str, str]) -> Iterator[tuple[int, int, str]]:
    """The extent and environment of each display, skipping inline maths."""
    openers = {name: body for name, body in delimiters.items() if not _closes(body)}
    closers = {name: body for name, body in delimiters.items() if _closes(body)}
    position = 0
    while (found := _DISPLAY_CHARACTER.search(text, position)) is not None:
        position = found.start()
        if text[position] == "\\":
            match = _CONTROL_SEQUENCE.match(text, position)
            if match is None:  # a backslash that ends the text
                return
            name = match.group(1)
            opener = openers.get(name, f"\\{name}" if name in ("[", "begin") else None)
            if opener is None:
                position = match.end()
                continue
            if opener == "\\begin":
                environment = _ENVIRONMENT.match(text, match.end())
                if environment is None or environment.group(1).strip() not in DISPLAY_ENVIRONMENTS:
                    position = match.end()
                    continue
                opener = f"\\begin{{{environment.group(1).strip()}}}"
            end = _display_end(text, match.end(), opener, closers)
            if end is None:
                position = match.end()
                continue
            yield position, end, _environment_name(opener)
            position = end
        elif text.startswith("$$", position):
            end = text.find("$$", position + 2)
            if end == -1:  # an unclosed $$: what follows it is still read
                position += 2
                continue
            yield position, end + 2, "$$"
            position = end + 2
        else:  # inline maths: skip to its closing dollar
            closing = _INLINE_END.search(text, position + 1)
            position = closing.end() if closing else len(text)


def _closes(body: str) -> bool:
    return body.startswith("\\end") or body == "\\]"


def _environment_name(opener: str) -> str:
    if opener in ("\\[", "$$"):
        return "displaymath" if opener == "\\[" else "$$"
    return opener[len("\\begin{") : -1]


def _display_end(text: str, position: int, opener: str, closers: dict[str, str]) -> int | None:
    """Where the display opened by ``opener`` ends, after its closing delimiter."""
    if opener in ("\\[", "$$"):
        closing = "\\]" if opener == "\\[" else "$$"
        written = re.escape(closing)
    else:
        environment = _environment_name(opener)
        closing = f"\\end{{{environment}}}"
        written = rf"\\end\s*\{{{re.escape(environment)}\}}"  # \end {equation} too
    names = [written] + [
        rf"\\{re.escape(name)}(?![A-Za-z])" for name, body in closers.items() if body == closing
    ]
    match = re.compile("|".join(names)).search(text, position)
    return match.end() if match else None


# -- Markdown notebooks ---------------------------------------------------------


def math_blocks(text: str, name: str = "<markdown>") -> list[Source]:
    """The ``$$...$$`` blocks of a Markdown document, outside code, each a source that
    knows where it is in the document, so that errors point at the right line. A block
    is a LaTeX display, so every statement in it is read as LaTeX."""
    return [block for block, _ in find_math_blocks(text, name)]


def find_math_blocks(text: str, name: str) -> Iterator[tuple[Source, int]]:
    """Each block, and the offset just after its closing ``$$``."""
    line, counted = 1, 0  # the line at offset counted
    position = 0
    while (match := _MARKDOWN.search(text, position)) is not None:
        position = match.end()
        indented = match.start("indented")
        if indented != -1 and _continues_list(text, indented):
            position = max(indented, match.start() + 1)  # a list item's text, not code
            continue
        if match.group("math") is not None:
            start = match.start("math")
            line += text.count("\n", counted, start)
            counted = start
            origin = (line, start - text.rfind("\n", 0, start))
            yield Source(match.group("math"), name, origin, latex=True), match.end()


def _continues_list(text: str, offset: int) -> bool:
    """Whether the indented line at ``offset`` goes on with a list item rather than
    starting a block of code: the last line with text before it is part of a list."""
    end = offset
    while end > 0 and text[end - 1] in " \t\n":
        end -= 1
    if end == 0:
        return False
    line = text[text.rfind("\n", 0, end) + 1 : end]
    return _LIST_ITEM.match(line) is not None or line[:1] in (" ", "\t")
