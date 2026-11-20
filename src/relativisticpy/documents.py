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
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from relativisticpy.language import Source

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
_CONTROL_SEQUENCE = re.compile(r"\\([A-Za-z]+|.)", re.DOTALL)
_PARAMETERS = re.compile(r"(?:#\d)*")
_ARGUMENT_COUNT = re.compile(r"\s*\[\s*(\d)\s*\]")
_MAX_EXPANSIONS = 10_000
_SECTION = re.compile(r"\\(?:chapter|section)\*?\s*[{\[]")
# Display maths in Markdown, $$...$$, outside code. Fenced code blocks and code
# spans are matched only so that what they contain is skipped, as when rendered.
_MARKDOWN = re.compile(
    r"^ {0,3}(?P<fence>`{3,}|~{3,}).*?^ {0,3}(?P=fence)[ \t]*$"
    r"|(?P<ticks>`+).+?(?P=ticks)"
    r"|\$\$(?P<math>.+?)\$\$",
    re.DOTALL | re.MULTILINE,
)


@dataclass(frozen=True)
class Macro:
    """A macro the document defines: ``\\name`` with ``arity`` arguments.

    When ``default`` is not ``None`` the first argument is optional, in brackets.
    """

    name: str
    arity: int
    body: str
    default: str | None = None


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
    """A LaTeX document, read for its equations."""

    text: str
    name: str = "<document>"
    macros: dict[str, Macro] = field(init=False)

    def __post_init__(self) -> None:
        self.text = strip_comments(self.text)
        self.macros, spans = _read_definitions(self.text)
        # Definitions such as \newcommand{\be}{\begin{equation}} are not displays.
        self._body = self.text
        for start, end in spans:
            blank = re.sub(r"[^\n]", " ", self._body[start:end])
            self._body = self._body[:start] + blank + self._body[end:]

    @classmethod
    def read(cls, path: str | Path) -> Document:
        """Read a document from a file, with the macros of its local packages
        (``\\usepackage{macros}`` with a ``macros.sty`` beside it) and of the files it
        includes with ``\\input`` and ``\\include``. Its own definitions come first."""
        path = Path(path)
        document = cls(path.read_text(encoding="utf-8", errors="replace"), str(path))
        for extra in _local_files(document.text, path.parent):
            text = strip_comments(extra.read_text(encoding="utf-8", errors="replace"))
            document.macros = {**find_macros(text), **document.macros}
        return document

    def equations(self) -> list[Equation]:
        """Every display equation, in order, with the document's macros expanded."""
        delimiters = {
            name: macro.body.strip()
            for name, macro in self.macros.items()
            if macro.arity == 0 and _is_delimiter(macro.body.strip())
        }
        found = []
        for start, end, environment in _displays(self._body, delimiters):
            raw = self.text[start:end]
            for name, body in delimiters.items():
                raw = re.sub(rf"\\{name}(?![A-Za-z])", _constant(body), raw)
            text = expand(raw, self.macros)
            line = self.text.count("\n", 0, start) + 1
            column = start - self.text.rfind("\n", 0, start)
            section = len(_SECTION.findall(self._body, 0, start))
            source = Source(text, self.name, (line, column))
            found.append(Equation(source, environment, section))
        return found


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


def _read_definitions(text: str) -> tuple[dict[str, Macro], list[tuple[int, int]]]:
    """The macros a document defines, and where their definitions are."""
    macros: dict[str, Macro] = {}
    spans = []
    for match in _DEFINITION.finditer(text):
        try:
            macro, end = _definition(text, match)
        except ValueError:
            continue  # a definition we cannot read is one we do not expand
        spans.append((match.start(), end))
        if not (macro.name in macros and match.group().startswith("\\provide")):
            macros[macro.name] = macro
    return macros, spans


def expand(text: str, macros: dict[str, Macro]) -> str:
    """Expand the given macros wherever they occur in ``text``, recursively."""
    done: list[str] = []
    expansions = 0
    while (start := text.find("\\")) != -1:
        match = _CONTROL_SEQUENCE.match(text, start)
        macro = macros.get(match.group(1)) if match else None
        if match is None or macro is None:
            end = match.end() if match else len(text)
            done.append(text[:end])
            text = text[end:]
            continue
        expansions += 1
        if expansions > _MAX_EXPANSIONS:
            raise ValueError(f"the macro \\{macro.name} expands without end")
        arguments, end = _arguments(text, match.end(), macro)
        body = macro.body
        for number, argument in enumerate(arguments, start=1):
            body = body.replace(f"#{number}", argument)
        # Keep a following letter from joining the expansion's last command.
        if re.search(r"\\[A-Za-z]+$", body) and text[end : end + 1].isalpha():
            body += " "
        done.append(text[:start])
        text = body + text[end:]
    done.append(text)
    return "".join(done)


# -- reading definitions ----------------------------------------------------------


def _definition(text: str, match: re.Match[str]) -> tuple[Macro, int]:
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
        return Macro(name, 0, target.group()), target.end()
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


def _is_delimiter(body: str) -> bool:
    if body in ("$$", "\\[", "\\]"):
        return True
    match = re.fullmatch(r"\\(begin|end)\{([^}]*)\}", body)
    return match is not None and match.group(2) in DISPLAY_ENVIRONMENTS


def _displays(text: str, delimiters: dict[str, str]) -> Iterator[tuple[int, int, str]]:
    """The extent and environment of each display, skipping inline maths."""
    openers = {name: body for name, body in delimiters.items() if not _closes(body)}
    closers = {name: body for name, body in delimiters.items() if _closes(body)}
    position = 0
    while position < len(text):
        char = text[position]
        if char == "\\":
            match = _CONTROL_SEQUENCE.match(text, position)
            assert match is not None
            name = match.group(1)
            opener = openers.get(name, f"\\{name}" if name in ("[", "begin") else None)
            if opener is None:
                position = match.end()
                continue
            if opener == "\\begin":
                environment = re.match(r"\s*\{([^}]*)\}", text[match.end() :])
                if environment is None or environment.group(1) not in DISPLAY_ENVIRONMENTS:
                    position = match.end()
                    continue
                opener = f"\\begin{{{environment.group(1)}}}"
            end = _display_end(text, match.end(), opener, closers)
            if end is None:
                position = match.end()
                continue
            yield position, end, _environment_name(opener)
            position = end
        elif char == "$":
            if text.startswith("$$", position):
                end = text.find("$$", position + 2)
                if end == -1:
                    return
                yield position, end + 2, "$$"
                position = end + 2
            else:  # inline maths: skip to its closing dollar
                closing = re.search(r"(?<!\\)\$", text[position + 1 :])
                position = position + 1 + (closing.end() if closing else len(text))
        else:
            position += 1


def _constant(text: str) -> Callable[[re.Match[str]], str]:
    """A replacement for :func:`re.sub` that inserts ``text`` as it is."""

    def replacement(_: re.Match[str]) -> str:
        return text

    return replacement


def _closes(body: str) -> bool:
    return body.startswith("\\end") or body == "\\]"


def _environment_name(opener: str) -> str:
    if opener in ("\\[", "$$"):
        return "displaymath" if opener == "\\[" else "$$"
    return opener[len("\\begin{") : -1]


def _display_end(text: str, position: int, opener: str, closers: dict[str, str]) -> int | None:
    """Where the display opened by ``opener`` ends, after its closing delimiter."""
    closing = {"\\[": "\\]", "$$": "$$"}.get(opener) or opener.replace("\\begin", "\\end")
    names = [re.escape(closing)] + [
        rf"\\{re.escape(name)}(?![A-Za-z])" for name, body in closers.items() if body == closing
    ]
    match = re.compile("|".join(names)).search(text, position)
    return match.end() if match else None


# -- Markdown notebooks ---------------------------------------------------------


def math_blocks(text: str, name: str = "<markdown>") -> list[Source]:
    """The ``$$...$$`` blocks of a Markdown document, outside code, each a source that
    knows where it is in the document, so that errors point at the right line."""
    return [block for block, _ in find_math_blocks(text, name)]


def find_math_blocks(text: str, name: str) -> Iterator[tuple[Source, int]]:
    """Each block, and the offset just after its closing ``$$``."""
    for match in _MARKDOWN.finditer(text):
        if match.group("math") is not None:
            start = match.start("math")
            origin = (text.count("\n", 0, start) + 1, start - text.rfind("\n", 0, start))
            yield Source(match.group("math"), name, origin), match.end()
