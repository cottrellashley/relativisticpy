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
from collections.abc import Callable, Iterator, Mapping
from dataclasses import InitVar, dataclass, field, replace
from pathlib import Path

from relativisticpy.language import Source
from relativisticpy.language.source import Piece

#: Environments that display equations, among them breqn's ``dmath`` and ``dgroup`` and
#: IEEEtrantools' ``IEEEeqnarray``, and ``math``, which sets its equation in the text.
DISPLAY_ENVIRONMENTS = frozenset(
    {
        "equation", "equation*", "eqnarray", "eqnarray*", "align", "align*",
        "gather", "gather*", "multline", "multline*", "flalign", "flalign*",
        "alignat", "alignat*", "displaymath", "math", "dmath", "dmath*", "dgroup",
        "dgroup*", "IEEEeqnarray", "IEEEeqnarray*",
    }
)  # fmt: skip

#: Commands that read another file into a document, where they stand.
INCLUDE_COMMANDS = ("input", "include", "subfile")

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
# What follows a macro where an argument is missing, rather than being one: \pd{f} = 0.
_NOT_ARGUMENTS = frozenset("=+-*/,;:.)]}&|<>^_")
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
# Display maths in Markdown, outside code and HTML comments: $$...$$, \[...\], a display
# environment such as \begin{equation}...\end{equation}, and a fenced block of maths,
# ```math, as GitHub writes one; and fenced blocks of RelativisticPy, ```relativisticpy.
# Other code blocks, code spans and comments are matched only so that what they contain
# is skipped, as when rendered: fenced code; code indented by four spaces after a blank
# line; and code spans, which are closed by as many backticks as open them and cannot
# cross a blank line, so that a stray backtick hides nothing.
_MARKDOWN = re.compile(
    r"^ {0,3}(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>[^\n]*)\n(?P<code>.*?)^ {0,3}(?P=fence)[ \t]*$"
    r"|<!--.*?-->"
    r"|(?:\A|^[ \t]*\n)(?P<indented>(?: {4}|\t)[^\n]*(?:\n(?:[ \t]*\n)*(?: {4}|\t)[^\n]*)*)"
    r"|(?<!`)(?P<ticks>`+)(?!`)(?:(?!\n[ \t]*\n).)*?(?<!`)(?P=ticks)(?!`)"
    r"|\$\$(?P<math>.+?)\$\$"
    r"|\\\[(?P<bracket>.+?)\\\]"
    r"|(?P<environment>\\begin\s*\{(?P<name>"
    + "|".join(re.escape(name) for name in sorted(DISPLAY_ENVIRONMENTS, key=len, reverse=True))
    + r")\}.*?\\end\s*\{(?P=name)\})",
    re.DOTALL | re.MULTILINE,
)
# The fenced blocks that are run: maths, as GitHub fences it, and programs.
_MATH_FENCES = frozenset({"math"})
_PROGRAM_FENCES = frozenset({"relativisticpy", "relpy", "rpy"})
_LIST_ITEM = re.compile(r" {0,3}(?:[-+*]|\d{1,9}[.)])(?:[ \t]|$)")


@dataclass(frozen=True)
class Macro:
    """A macro the document defines: ``\\name`` with ``arity`` arguments.

    When ``default`` is not ``None`` the first argument is optional, in brackets.
    An ``alias``, made by ``\\let\\name\\command`` of a command the document does not
    define, stands for that command, which is never expanded again. ``where`` says
    where it is defined, as ``line 7`` or ``macros.sty, line 3``, for messages.
    """

    name: str
    arity: int
    body: str
    default: str | None = None
    alias: bool = False
    where: str = field(default="", compare=False)


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
    the document's own, as those of its packages are.

    ``broken`` says, for each macro whose definition cannot be read, why, so that an
    equation using it says so rather than that the macro is unknown. A document read
    from a file with :meth:`read` has the equations of the files it reads with
    ``\\input``, ``\\include`` and ``\\subfile`` in their places, and ``missing`` names
    the files it reads that could not be found, with the line of each."""

    text: str
    name: str = "<document>"
    known: InitVar[Mapping[str, Macro] | None] = None
    macros: dict[str, Macro] = field(init=False)
    broken: dict[str, str] = field(init=False)
    missing: list[tuple[str, int]] = field(init=False, default_factory=list)

    def __post_init__(self, known: Mapping[str, Macro] | None) -> None:
        self.text = strip_comments(self.text)
        # TeX skips what \iffalse and the comment environment hide, definitions included.
        visible = _blank(self.text, _hidden(self.text))
        self._lines = [0, *(match.end() for match in re.finditer("\n", self.text))]
        self.macros, definitions, broken = _read_definitions(visible, known, self._where)
        self.broken = {name: why for name, why in broken.items() if name not in self.macros}
        # Definitions such as \newcommand{\be}{\begin{equation}} are not displays.
        self._body = _blank(visible, definitions)
        self._sections = [match.start() for match in _SECTION.finditer(self._body)]
        self._included: list[tuple[int, Document]] = []  # where each file is read in

    def _where(self, offset: int) -> str:
        """Where a character is, in words: ``line 7``, or ``macros.sty, line 7`` when the
        document is another file than the one being read."""
        return f"line {bisect_right(self._lines, offset)}"

    def __repr__(self) -> str:
        """``Document('paper.tex': 120 lines, 14 display equations, 3 macros)``, counting
        the equations of the files it includes."""
        lines = self.text.count("\n") + 1
        displays = self._display_count()
        macros = len(self.macros)
        return (
            f"Document({self.name!r}: {lines} line{'s' * (lines != 1)}, {displays} display "
            f"equation{'s' * (displays != 1)}, {macros} macro{'s' * (macros != 1)})"
        )

    @classmethod
    def read(cls, path: str | Path) -> Document:
        """Read a document from a file, with the macros of its local packages
        (``\\usepackage{macros}`` with a ``macros.sty`` beside it) and of the files it
        includes with ``\\input`` and ``\\include``. Its own definitions come first.
        The equations of the files it includes are its equations, where it includes
        them, each with its own file's name and positions."""
        path = Path(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        packages: dict[str, Macro] = {}
        for extra in _local_files(strip_comments(text), path.parent):
            found = find_macros(strip_comments(extra.read_text(encoding="utf-8", errors="replace")))
            for macro in found.values():
                packages.setdefault(
                    macro.name, replace(macro, where=f"{extra.name}, {macro.where}")
                )
        document = cls(text, str(path), packages)
        document._include(path.parent, {path.resolve()})
        return document

    def _include(self, folder: Path, reading: set[Path]) -> None:
        """Read the files this document includes, and theirs in turn, as TeX would: a
        file is found as named, or with ``.tex`` after its name, beside the document."""
        for match in _INCLUDE.finditer(self._body):
            name = (match.group("braced") or match.group("bare") or "").strip()
            path = next(
                (
                    candidate
                    for candidate in (folder / name, folder / f"{name}.tex")
                    if candidate.is_file()
                ),
                None,
            )
            if path is None:
                self.missing.append((name, bisect_right(self._lines, match.start())))
                continue
            if path.resolve() in reading or len(reading) > _MAX_INCLUDE_DEPTH:
                continue  # a file that includes itself, as TeX would read it, without end
            text = path.read_text(encoding="utf-8", errors="replace")
            included = Document(text, str(path), self.macros)
            included._include(folder, reading | {path.resolve()})
            self.missing.extend(included.missing)
            self._included.append((match.start(), included))

    def equations(self) -> list[Equation]:
        """Every display equation, in order, with the document's macros expanded, and
        those of the files it includes where it includes them. An equation whose macros
        cannot be expanded is given as written, and reading it raises a
        :class:`~relativisticpy.errors.ParseError` that says why."""
        found: list[Equation] = []
        included = iter(self._included)
        offset = 0  # the sections of the files included so far
        nxt = next(included, None)
        for start, end, environment in self._display_extents():
            while nxt is not None and nxt[0] < start:
                offset = self._read_in(nxt, offset, found)
                nxt = next(included, None)
            line = bisect_right(self._lines, start)
            origin = (line, start - self._lines[line - 1] + 1)
            section = bisect_left(self._sections, start) + offset
            found.append(Equation(self._source(start, end, origin), environment, section))
        while nxt is not None:
            offset = self._read_in(nxt, offset, found)
            nxt = next(included, None)
        return found

    def _read_in(self, included: tuple[int, Document], offset: int, found: list[Equation]) -> int:
        """Add the equations of a file included at an offset, numbering its sections after
        those before it, and return the number of sections included so far."""
        at, document = included
        before = bisect_left(self._sections, at) + offset
        for equation in document.equations():
            found.append(replace(equation, section=before + equation.section))
        return offset + document._section_count()

    def _display_count(self) -> int:
        """How many displays the document has, with those of the files it includes."""
        own = len(self._display_extents())
        return own + sum(document._display_count() for _, document in self._included)

    def _section_count(self) -> int:
        """How many sections the document has, with those of the files it includes."""
        return len(self._sections) + sum(doc._section_count() for _, doc in self._included)

    def _display_extents(self) -> list[tuple[int, int, str]]:
        delimiters = {
            name: _BEGIN_OR_END.sub(r"\\\1{\2}", macro.body.strip())
            for name, macro in self.macros.items()
            if macro.arity == 0 and _is_delimiter(macro.body.strip())
        }
        return list(_displays(self._body, delimiters))

    def _source(self, start: int, end: int, origin: tuple[int, int]) -> Source:
        """The source of the display at ``[start, end)``, its macros expanded."""
        written = self.text[start:end]
        try:
            text, pieces = _expand(written, self.macros, self.broken)
        except _Unexpandable as error:
            unreadable = (str(error), error.start, error.end)
            return Source(written, self.name, origin, latex=True, unreadable=unreadable)
        if text == written:
            return Source(text, self.name, origin, latex=True)
        definitions = tuple(
            (name, macro.where) for name, macro in self.macros.items() if macro.where
        )
        return Source(
            text,
            self.name,
            origin,
            latex=True,
            written=written,
            pieces=pieces,
            definitions=definitions,
        )


# \input{file}, \include{file}, \subfile{file}, and \input file, as plain TeX writes it.
_INCLUDE = re.compile(
    r"\\(?:" + "|".join(INCLUDE_COMMANDS) + r")(?![A-Za-z])"
    r"\s*(?:\{(?P<braced>[^{}]*)\}|(?P<bare>[^\s{}\\]+))"
)
_MAX_INCLUDE_DEPTH = 16


def _local_files(text: str, folder: Path) -> list[Path]:
    """The local packages and included files a document names, where they exist."""
    found = []
    for command, names in re.findall(
        r"\\(usepackage|input|include|subfile)(?:\[[^\]]*\])?\{([^}]*)\}", text
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
    lines = [0, *(match.end() for match in re.finditer("\n", text))]
    return _read_definitions(text, None, lambda at: f"line {bisect_right(lines, at)}")[0]


def _read_definitions(
    text: str, known: Mapping[str, Macro] | None, where: Callable[[int], str]
) -> tuple[dict[str, Macro], list[tuple[int, int]], dict[str, str]]:
    """The macros a document defines, after the ``known`` ones; where their definitions,
    and those of environments, are; and why each definition that cannot be read cannot,
    for the macro it would define. ``where`` says where an offset is, in words."""
    macros = dict(known or {})
    spans = []
    broken: dict[str, str] = {}
    for match in _DEFINITION.finditer(text):
        try:
            macro, end = _definition(text, match, macros)
        except _Unreadable as error:  # a definition we cannot read is one we do not expand
            if error.name:
                broken[error.name] = str(error).replace("{where}", where(match.start()))
            continue
        except ValueError:
            continue
        spans.append((match.start(), end))
        if not (macro.name in macros and match.group().startswith("\\provide")):
            macros[macro.name] = replace(macro, where=where(match.start()))
            broken.pop(macro.name, None)
    for match in _ENVIRONMENT_DEFINITION.finditer(text):
        try:
            spans.append((match.start(), _environment_definition_end(text, match.end())))
        except ValueError:
            continue
    return macros, spans, broken


def expand(text: str, macros: Mapping[str, Macro]) -> str:
    """Expand the given macros wherever they occur in ``text``, recursively."""
    return _expand(text, macros)[0]


class _Unreadable(ValueError):
    """A definition that cannot be read, and the name of the macro it would define."""

    def __init__(self, message: str, name: str = "") -> None:
        super().__init__(message)
        self.name = name


class _MissingArgument(ValueError):
    """A macro called with fewer arguments than it takes; ``end`` is where the
    arguments given end."""

    def __init__(self, message: str, end: int) -> None:
        super().__init__(message)
        self.end = end


class _Unexpandable(ValueError):
    """Macros that cannot be expanded, and the extent of the first of them in the text."""

    def __init__(self, message: str, start: int, end: int) -> None:
        super().__init__(message)
        self.start, self.end = start, end


# A run of the text being expanded: its length, and the characters of the written text
# it came from, copied as they are or, if not, expanded from them.
_Run = tuple[int, int, int, bool]


def _expand(
    text: str, macros: Mapping[str, Macro], broken: Mapping[str, str] | None = None
) -> tuple[str, tuple[Piece, ...]]:
    """Expand the macros in ``text``, and say where each piece of the result came from.
    A macro in ``broken``, whose definition cannot be read, cannot be expanded.

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
        if match is not None and macro is None and broken and match.group(1) in broken:
            where = _written(runs, start, match.end())
            raise _Unexpandable(broken[match.group(1)], *where)
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
        except _MissingArgument as error:  # \pd{f} = 0, where \pd takes two
            given = _written(runs, 0, error.end - match.end())
            raise _Unexpandable(str(error), where[0], max(where[1], given[1])) from None
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


def _written(runs: list[_Run], start: int, end: int) -> tuple[int, int]:
    """Where the characters ``[start, end)`` of the text still to be read, which ``runs``
    describe, were written, without taking them off ``runs``."""
    low: int | None = None
    high: int | None = None
    offset = 0
    for size, written_start, written_end, copied in reversed(runs):
        run_start, run_end = offset, offset + size
        if run_end > start and run_start < max(end, start + 1):
            if copied:
                first = written_start + max(start, run_start) - run_start
                last = written_start + min(max(end, start + 1), run_end) - run_start
            else:
                first, last = written_start, written_end
            low = first if low is None else min(low, first)
            high = last if high is None else max(high, last)
        offset = run_end
        if offset >= end:
            break
    return (low or 0, high if high is not None else (low or 0))


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
            brace = text.find("{", parameters.end())
            written = text[position : brace if brace != -1 else position + 10].strip()
            raise _Unreadable(
                f"\\{name} is defined on {{where}} with delimited parameters, {written}, "
                "which are not supported",
                name,
            )
        body, end = _body(text, parameters.end(), name)
        return Macro(name, parameters.group().count("#"), body), end
    name, position = _defined_name(text, position)
    if command.startswith("\\DeclareMathOperator"):
        body, end = _body(text, _skip_space(text, position), name)
        return Macro(name, 0, f"\\operatorname{{{body}}}"), end
    arity, default = 0, None
    if count := _ARGUMENT_COUNT.match(text, position):
        arity, position = int(count.group(1)), count.end()
        position = _skip_space(text, position)
        if text.startswith("[", position):
            end = _closing_bracket(text, position)
            default, position = text[position + 1 : end], end + 1
    body, end = _body(text, _skip_space(text, position), name)
    return Macro(name, arity, body, default), end


def _body(text: str, position: int, name: str) -> tuple[str, int]:
    """The body of the definition of ``\\name``, braced at ``position``, and its end."""
    try:
        return _group(text, position)
    except ValueError as error:
        raise _Unreadable(
            f"the definition of \\{name} on {{where}} cannot be read: {error}", name
        ) from None


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
    """The arguments of a macro called at ``position``, and where they end. As TeX does,
    an argument without braces is one character or one command; but one that cannot
    be meant as an argument, such as ``=`` in ``\\pd{f} = 0``, or the end of the
    equation, means that an argument is missing."""
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
        cursor = _skip_space(text, position)
        if text.startswith("{", cursor):
            argument, position = _group(text, cursor)
        elif (
            text.startswith(("\\\\", "\\end"), cursor)
            or cursor == len(text)
            or (text[cursor] in _NOT_ARGUMENTS)
        ):
            given = number - (macro.default is not None)
            takes = macro.arity - (macro.default is not None)
            arguments_ = f"{takes} argument{'s' * (takes != 1)}"
            verb = "is" if given == 1 else "are"
            message = f"\\{macro.name} takes {arguments_}, but {given} {verb} given here"
            raise _MissingArgument(message, position)
        elif (command := _CONTROL_SEQUENCE.match(text, cursor)) is not None:
            argument, position = command.group(), command.end()
        else:
            argument, position = text[cursor], cursor + 1
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
    """The blocks of maths a Markdown document runs, outside code, each a source that
    knows where it is in the document, so that errors point at the right line: displays
    in ``$$...$$`` or ``\\[...\\]``, display environments such as
    ``\\begin{equation}``, and fenced blocks of maths, ```` ```math ````. Each is a LaTeX
    display, so every statement in it is read as LaTeX. Fenced blocks of RelativisticPy,
    ```` ```relativisticpy ```` (or ``relpy``, or ``rpy``), are programs, run as written."""
    return [block for block, _ in find_math_blocks(text, name)]


def find_math_blocks(text: str, name: str) -> Iterator[tuple[Source, int]]:
    """Each block, and the offset just after it, where its results go."""
    line, counted = 1, 0  # the line at offset counted
    position = 0
    while (match := _MARKDOWN.search(text, position)) is not None:
        position = match.end()
        indented = match.start("indented")
        if indented != -1 and _continues_list(text, indented):
            position = max(indented, match.start() + 1)  # a list item's text, not code
            continue
        group, latex = _block(match)
        if group is None:
            continue
        start = match.start(group)
        line += text.count("\n", counted, start)
        counted = start
        origin = (line, start - text.rfind("\n", 0, start))
        yield Source(match.group(group), name, origin, latex=latex), match.end()


def _block(match: re.Match[str]) -> tuple[str | None, bool]:
    """Which group of a match of the Markdown is a block to run, and whether it is LaTeX;
    ``None`` for what is skipped, such as code."""
    for group in ("math", "bracket", "environment"):
        if match.group(group) is not None:
            return group, True
    if match.group("fence") is not None:
        info = (match.group("info").split() or [""])[0].strip("{}.").lower()
        if info in _MATH_FENCES:
            return "code", True
        if info in _PROGRAM_FENCES:
            return "code", False
    return None, False


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
