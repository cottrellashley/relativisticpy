"""The notation matrix: the same mathematics written every way a book writes it, each way
with an answer known without RelativisticPy.

A construct of the core, such as a fraction, a derivative, a determinant or a contraction,
names its forms, its operands and its oracle. A form is a template in LaTeX, plain text or
Unicode, with the book or convention that writes it so. An operand is what the templates
are filled with, written in each notation and in Python. The oracle is Python that
computes the answer from an operand alone, with SymPy or NumPy: never from RelativisticPy,
and never from the form. A cell is one form filled with one operand, so that every form of
a construct must give the answer its oracle gives, and so the same answer as every other.

    uv run python benchmarks/notations.py                           # the scoreboard
    uv run python benchmarks/notations.py --construct calculus      # some constructs
    uv run python benchmarks/notations.py --failures                # why each cell fails
    uv run python benchmarks/notations.py --update                  # record what passes now
    uv run python benchmarks/notations.py --show CELL               # a cell, as a case
    uv run python benchmarks/notations.py --oracles                 # each oracle, by two routes
    uv run python benchmarks/notations.py --seeds 0,1,2             # what the hash seed changes

A cell is run as ``benchmarks/curriculum.py`` runs a case, in a fresh workbook, within the
same time limit, its answer compared as strictly. Each runs in a process of its own,
started from this one before it has computed anything, so that no cell's outcome depends
on which cells ran before it, or beside it: several run at a time. A run fixes Python's
hash seed as the curriculum's does.

``curriculum/notations/passing.txt`` records the cells that pass; a cell may join it,
never leave it (``tests/test_notations.py`` holds them to it). The format of a
specification is described in ``curriculum/README.md``.
"""

from __future__ import annotations

import argparse
import ast
import collections
import functools
import importlib.util
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
import time
import zlib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait
from pathlib import Path
from types import ModuleType
from typing import Any

from relativisticpy import errors
from relativisticpy.__main__ import fix_hash_seed


def _beside(name: str) -> ModuleType:
    """The benchmark called ``name``, which is beside this one, imported once."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


#: The curriculum's harness, whose comparison of answers and time limits these are.
curriculum = _beside("curriculum")

ROOT = curriculum.ROOT / "notations"
PASSING = ROOT / "passing.txt"
TIME_LIMIT: float = curriculum.TIME_LIMIT  # seconds for each cell
#: The notations a form is written in, by the mark of its fenced block.
NOTATIONS = {"latex": "LaTeX", "text": "plain text", "unicode": "Unicode"}
PYTHON = "python"  # how an operand is written for the oracle
#: How long past its time limit a cell's process may run before it is stopped, in seconds:
#: a computation that no signal interrupts is stopped here.
GRACE = 15.0
#: The time an operand's oracle and its second route may take together, in seconds.
ORACLE_LIMIT = 120.0

_SLOT = re.compile(r"«([^«»\n]+)»")
_FENCE = re.compile(
    r"^```(?P<info>[\w-]*)[ \t]*\n(?P<body>.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL
)
_FORM = re.compile(
    r"^\*\*Form\*\*\s*·\s*`(?P<name>[^`]+)`(?:\s*·\s*(?P<source>.+?))?\s*$", re.MULTILINE
)
_REFUSED = re.compile(
    r"^\*\*Refused\*\*\s*·\s*`(?P<kind>[^`]+)`(?:\s*·\s*`(?P<message>[^`]+)`)?"
    r"(?:\s*·\s*(?P<why>.+?))?\s*$",
    re.MULTILINE,
)
_OF = re.compile(
    r"^\*\*(?P<which>Only of|Not of)\*\*\s*·\s*(?P<names>`[^`]+`(?:\s*,\s*`[^`]+`)*)"
    r"(?:\s*·\s*(?P<why>.+?))?\s*$",
    re.MULTILINE,
)
_CODE = re.compile(r"`([^`]+)`")
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
_NAME = re.compile(r"[^\s/`]+")  # of a form or an operand: a part of a cell's identifier
_SECTIONS = ("Setup", "Names", "Operands", "Oracle", "Second route", "Forms")


# -- what a specification holds ---------------------------------------------------------


@dataclass(frozen=True)
class Form:
    """One way of writing a construct: a template in one notation, and who writes it so."""

    name: str
    title: str
    source: str
    notation: str  # latex, text or unicode
    template: str
    line: int
    refused: str = ""  # the kind of error it must be refused with, where it must be
    message: str = ""  # a part of that error's text
    only_of: tuple[str, ...] = ()  # the operands it is written of, where not all
    not_of: tuple[str, ...] = ()  # the operands it is never written of


@dataclass(frozen=True)
class Operand:
    """What the templates are filled with: for each notation, and for Python, how each
    slot is written."""

    name: str
    written: Mapping[str, Mapping[str, str]]
    line: int


@dataclass(frozen=True)
class Construct:
    """One construct: its forms, its operands, and the oracle that knows its answers."""

    id: str
    title: str
    path: Path
    slots: tuple[str, ...] = ()
    setup: Mapping[str, str] = field(default_factory=dict)  # by notation; "python-scope" too
    names: str = ""  # Python: the names its operands and its oracle are written with
    oracle: str = ""
    second_route: str = ""
    operands: tuple[Operand, ...] = ()
    forms: tuple[Form, ...] = ()
    faults: tuple[str, ...] = ()  # what is wrong with how the specification is written

    @property
    def where(self) -> str:
        """Its file, from the root of the repository where it is in it."""
        root = curriculum.ROOT.parent
        return str(self.path.relative_to(root) if root in self.path.parents else self.path)

    def cells(self) -> list[Cell]:
        """Its cells, form by form: each form filled with each operand it is written of,
        where the operand is written in the form's notation."""
        found = []
        for form in self.forms:
            for operand in self.operands:
                cell = Cell(self, form, operand)
                if cell.exists:
                    found.append(cell)
        return found

    def python(self, operand: Operand) -> str:
        """The oracle's first lines for an operand: the names, then its slots bound, all
        at once, so that none is written with another's value."""
        slots = operand.written.get(PYTHON, {})
        if not slots:
            return self.names
        values = ", ".join(f"({value})" for value in slots.values())
        return f"{self.names}{', '.join(slots)} = {values}\n"


@dataclass(frozen=True)
class Cell:
    """One form of a construct filled with one operand: a program, and the oracle that
    knows its answer."""

    construct: Construct
    form: Form
    operand: Operand

    @property
    def id(self) -> str:
        return f"{self.construct.id}/{self.form.name}/{self.operand.name}"

    @property
    def where(self) -> str:
        return f"{self.construct.where}:{self.form.line}"

    @property
    def exists(self) -> bool:
        """Whether the form is written of the operand, and the operand in its notation."""
        form, name = self.form, self.operand.name
        if (form.only_of and name not in form.only_of) or name in form.not_of:
            return False
        return _filled(self._template, self._spellings) is not None and self.scope is not None

    @property
    def _template(self) -> str:
        setup = self.construct.setup.get(self.form.notation, "")
        return "\n".join(part.strip("\n") for part in (setup, self.form.template) if part.strip())

    @property
    def _spellings(self) -> Mapping[str, str]:
        return self.operand.written.get(self.form.notation, {})

    @property
    def program(self) -> str:
        """The form's template, after the construct's setup, each slot filled."""
        return _filled(self._template, self._spellings) or ""

    @property
    def scope(self) -> str | None:
        """The Python run first, whose names the program may read, each slot filled as
        Python writes it; ``None`` where the operand does not write one of them."""
        scope = self.construct.setup.get("python-scope", "")
        return _filled(scope, self.operand.written.get(PYTHON, {}))

    @property
    def oracle(self) -> str:
        """Python that says what the program must give: the oracle's answer for the
        operand, or, of a form that must be refused, the kind of the error."""
        if not self.form.refused:
            return self.construct.python(self.operand) + self.construct.oracle
        message = f"message = {self.form.message!r}\n" if self.form.message else ""
        return f"error = {self.form.refused!r}\n{message}"

    def case(self) -> Any:
        """The cell as a case of the curriculum, which runs it and compares its answer. A
        program in LaTeX is given as LaTeX, which every statement of it then is, whatever
        it is written with (``design/notation.md``, decision 10); one in plain text or in
        Unicode is given as it is typed."""
        return curriculum.Case(
            id=self.id,
            title=self.form.title,
            domain=self.construct.path.parent.name,
            topic=self.construct.path.stem,
            level=1,
            source=self.form.source,
            program=self.program,
            oracle=self.oracle,
            path=self.construct.path,
            line=self.form.line,
            scope=self.scope or "",
            latex=self.form.notation == "latex",
        )

    def shown(self) -> str:
        """The cell as a case of the curriculum is written: its program and its oracle."""
        scope = f"```python-scope\n{self.scope}```\n\n" if self.scope else ""
        return (
            f"## {self.form.title}, of {self.operand.name}\n\n"
            f"**Level 1** · `{self.id}` · {self.form.source}\n\n"
            f"{scope}```{self.form.notation}\n{self.program}\n```\n\n"
            f"```python\n{self.oracle}```\n\n{self.where}"
        )


def _filled(template: str, spellings: Mapping[str, str]) -> str | None:
    """``template`` with each slot, ``«f»``, replaced by how it is written, or ``None``
    where one of its slots is not written."""
    if any(slot not in spellings for slot in _SLOT.findall(template)):
        return None
    return _SLOT.sub(lambda match: spellings[match[1]], template)


# -- reading the specifications ---------------------------------------------------------


def load(root: Path = ROOT) -> list[Construct]:
    """Every construct under ``root``, in the order of its areas and files."""
    return [read(path) for path in sorted(root.glob("*/*.md"))]


def read(path: Path) -> Construct:
    """The construct a file specifies, named after its folder and the file:
    ``calculus/derivative.md`` is ``calculus.derivative``."""
    text = path.read_text(encoding="utf-8")
    title = re.match(r"# (.+?)\s*$", text, re.MULTILINE)
    faults = [] if title else ["it does not start with a title, as in '# Fractions'"]
    sections = {name: (body, line) for name, body, line in _sections(text, 2)}
    faults += [f"## {name} is no section of a construct" for name in sections.keys() - _SECTIONS]
    setup = {
        fence["info"]: fence["body"]
        for fence in _FENCE.finditer(sections.get("Setup", ("", 0))[0])
        if fence["info"] in (*NOTATIONS, "python-scope")
    }
    slots, operands, table_faults = _operands(*sections.get("Operands", ("", 0)))
    forms, form_faults = _forms(*sections.get("Forms", ("", 0)))
    return Construct(
        id=f"{path.parent.name}.{path.stem}",
        title=title[1] if title else path.stem,
        path=path,
        slots=slots,
        setup=setup,
        names=_python(sections.get("Names", ("", 0))[0]),
        oracle=_python(sections.get("Oracle", ("", 0))[0]),
        second_route=_python(sections.get("Second route", ("", 0))[0]),
        operands=operands,
        forms=forms,
        faults=(*faults, *table_faults, *form_faults),
    )


def _blanked(text: str) -> str:
    """``text`` with its fenced blocks made blank, every offset kept, so that a line of a
    program that starts with ``#`` is not taken for a heading."""
    return _FENCE.sub(lambda fence: re.sub(r"[^\n]", " ", fence[0]), text)


def _sections(text: str, level: int, line: int = 1) -> list[tuple[str, str, int]]:
    """The sections of ``text`` whose headings have ``level`` number signs: the title of
    each, its body, and the line of its heading, where ``text`` starts on ``line``."""
    heading = re.compile(rf"^{'#' * level} (?P<title>.+?)[ \t]*$", re.MULTILINE)
    found = list(heading.finditer(_blanked(text)))
    sections = []
    for number, match in enumerate(found):
        end = found[number + 1].start() if number + 1 < len(found) else len(text)
        at = line + text.count("\n", 0, match.start())
        sections.append((match["title"], text[match.end() : end], at))
    return sections


def _python(body: str) -> str:
    """The Python of a section: what its blocks marked ``python`` hold, in order."""
    return "".join(fence["body"] for fence in _FENCE.finditer(body) if fence["info"] == PYTHON)


def _rows(body: str, line: int) -> list[tuple[int, list[str]]]:
    """The rows of the table in ``body``, each with its line and its cells, as GitHub
    reads them: a bar written ``\\|`` is a bar in its cell, and does not end it."""
    rows = []
    for number, written in enumerate(body.splitlines()):
        written = written.strip()
        if written.startswith("|"):
            cells = re.split(r"(?<!\\)\|", written)[1:]
            cells = cells[:-1] if cells and not cells[-1].strip() else cells
            rows.append((line + number, [cell.replace("\\|", "|").strip() for cell in cells]))
    return rows


def _operands(body: str, line: int) -> tuple[tuple[str, ...], tuple[Operand, ...], list[str]]:
    """The slots, the operands and the faults of an ``## Operands`` table. Its first
    column names an operand, on the first of its rows; its second says which notations a
    row is written in; and each column headed by a slot's name, as `` `f` ``, says how
    that slot is written in them. Any other column is commentary."""
    rows = [row for row in _rows(body, line) if not all(set(cell) <= set(":-") for cell in row[1])]
    if not rows:
        return (), (), ["## Operands has no table"]
    header = rows[0][1]
    columns = {
        number: code[1]
        for number, cell in enumerate(header)
        if number >= 2 and (code := _CODE.fullmatch(cell))
    }
    faults, operands = [], []
    name, at, written = "", 0, {}
    for row_line, cells in rows[1:]:
        cells += [""] * (len(header) - len(cells))
        if len(cells) > len(header):
            faults.append(
                f"line {row_line}: a row of {len(cells)} cells under {len(header)} headings; "
                "in a table, write a bar as \\|"
            )
            continue
        if cells[0]:
            if name:
                operands.append(Operand(name, written, at))
            code = _CODE.fullmatch(cells[0])
            name, at, written = (code[1] if code else cells[0]), row_line, {}
        notations = [notation.strip() for notation in cells[1].split(",") if notation.strip()]
        unknown = [n for n in notations if n not in (*NOTATIONS, PYTHON)]
        if unknown or not notations or not name:
            said = ", ".join(unknown) or "nothing"
            kinds = ", ".join((*NOTATIONS, PYTHON))
            faults.append(f"line {row_line}: written in {said}, which is none of {kinds}")
            continue
        for number, slot in columns.items():
            if not cells[number]:
                continue
            code = _CODE.fullmatch(cells[number])
            if code is None:
                faults.append(f"line {row_line}: {slot} is written {cells[number]}, not as code")
                continue
            for notation in notations:
                if slot in written.setdefault(notation, {}):
                    faults.append(f"line {row_line}: {name} writes {slot} twice in {notation}")
                written[notation][slot] = code[1]
    if name:
        operands.append(Operand(name, written, at))
    return tuple(columns.values()), tuple(operands), faults


def _forms(body: str, line: int) -> tuple[tuple[Form, ...], list[str]]:
    """The forms of a ``## Forms`` section, each a ``###`` section, and their faults."""
    forms, faults = [], []
    for title, section, at in _sections(body, 3, line):
        named = _FORM.search(section)
        if named is None:
            faults.append(f"line {at}: {title} has no line '**Form** · `name` · source'")
            continue
        templates = [f for f in _FENCE.finditer(section) if f["info"] in NOTATIONS]
        if len(templates) != 1:
            kinds = ", ".join(NOTATIONS)
            faults.append(f"line {at}: {named['name']} needs one template, marked {kinds}")
            continue
        refused = _REFUSED.search(section)
        of = {match["which"]: match["names"] for match in _OF.finditer(section)}
        forms.append(
            Form(
                name=named["name"],
                title=title,
                source=(named["source"] or "").strip(),
                notation=templates[0]["info"],
                template=templates[0]["body"],
                line=at,
                refused=refused["kind"] if refused else "",
                message=(refused["message"] or "") if refused else "",
                only_of=tuple(_CODE.findall(of.get("Only of", ""))),
                not_of=tuple(_CODE.findall(of.get("Not of", ""))),
            )
        )
    return tuple(forms), faults


def passing(path: Path = PASSING) -> set[str]:
    """The cells recorded as passing."""
    found: set[str] = curriculum.passing(path)
    return found


def problems(constructs: Sequence[Construct]) -> list[str]:
    """What is wrong with the specifications themselves, as far as reading them tells:
    a part missing, a name used twice, a slot nothing writes, a form with no cell. What
    running an oracle tells is :func:`check`'s to say."""
    found = []
    for construct in constructs:
        said: list[str] = list(construct.faults)
        if not construct.oracle.strip():
            said.append("it has no ## Oracle with a ```python block")
        if not construct.forms:
            said.append("it has no forms")
        for kind, names in (
            ("form", [form.name for form in construct.forms]),
            ("operand", [operand.name for operand in construct.operands]),
        ):
            counts = collections.Counter(names)
            said += [f"the {kind} {name} is there {n} times" for name, n in counts.items() if n > 1]
            said += [
                f"the {kind} {name} is no part of an identifier: it has a space, a / or a `"
                for name in counts
                if not _NAME.fullmatch(name)
            ]
        bound = {slot for operand in construct.operands for slot in operand.written.get(PYTHON, {})}
        said += [
            f"the slot {slot} is written in Python, but is not a name Python can bind"
            for slot in construct.slots
            if slot in bound and not _IDENTIFIER.fullmatch(slot)
        ]
        said += [
            f"the slot {slot} is also a name of ## Names, which it would replace"
            for slot in construct.slots
            if slot in bound and slot in _assigned(construct.names)
        ]
        said += _setup_problems(construct)
        for form in construct.forms:
            said += _form_problems(construct, form)
        found += [f"{construct.where}: {construct.id}: {each}" for each in said]
    return found


def _assigned(python: str) -> set[str]:
    """The names a block of Python binds at its top level, as far as reading it tells."""
    try:
        tree = ast.parse(python)
    except SyntaxError:
        return set()
    targets = [
        target for node in tree.body if isinstance(node, ast.Assign) for target in node.targets
    ]
    targets += [
        node.target for node in tree.body if isinstance(node, ast.AnnAssign | ast.AugAssign)
    ]
    named = [node for target in targets for node in ast.walk(target) if isinstance(node, ast.Name)]
    defined = [node for node in tree.body if isinstance(node, ast.FunctionDef | ast.ClassDef)]
    return {node.id for node in named} | {node.name for node in defined}


def _setup_problems(construct: Construct) -> list[str]:
    said = []
    for notation, lines in construct.setup.items():
        for slot in dict.fromkeys(_SLOT.findall(lines)):
            if slot not in construct.slots:
                said.append(f"its setup in {notation} has «{slot}», which no operand writes")
    return said


def _form_problems(construct: Construct, form: Form) -> list[str]:
    said = []
    if not form.source:
        said.append(f"the form {form.name} names no source")
    for slot in dict.fromkeys(_SLOT.findall(form.template)):
        if slot not in construct.slots:
            said.append(f"the form {form.name} has «{slot}», which no operand writes")
    known = {operand.name for operand in construct.operands}
    for name in (*form.only_of, *form.not_of):
        if name not in known:
            said.append(f"the form {form.name} names the operand {name}, which is not there")
    if form.refused and not _is_an_error(form.refused):
        said.append(f"the form {form.name} must be refused with {form.refused}, which is no error")
    if not any(Cell(construct, form, operand).exists for operand in construct.operands):
        said.append(f"the form {form.name} has no cell: no operand is written for it")
    return said


def _is_an_error(kind: str) -> bool:
    found = getattr(errors, kind, None)
    return isinstance(found, type) and issubclass(found, errors.RelativisticPyError)


# -- running a cell -----------------------------------------------------------------------


@dataclass(frozen=True)
class Outcome:
    """How a cell went, in the curriculum's words: ``passed``, ``wrong`` (a value given as
    the answer, but not the answer, or an answer where it had to be refused), ``kept`` (a
    value kept as written, as a note of the program says), ``refused``, ``timeout``,
    ``crashed`` or ``broken`` (the specification is at fault)."""

    cell: Cell
    outcome: str
    reason: str = ""
    detail: str = ""
    seconds: float = 0.0

    @property
    def passed(self) -> bool:
        return self.outcome == "passed"

    @property
    def answer(self) -> str | None:
        """The answer the program gave, as the curriculum shows it, where it gave one that
        is not the oracle's; ``None`` where it passed, or gave none."""
        if self.outcome != "wrong":
            return None
        given = [line for line in self.detail.splitlines() if _GOT.match(line)]
        return "; ".join(_GOT.sub("", line) for line in given) if given else self.detail

    @property
    def expected(self) -> str | None:
        """The answer the oracle gives, as the curriculum shows it beside a wrong one."""
        wanted = [line for line in self.detail.splitlines() if _EXPECTED.match(line)]
        return "; ".join(_EXPECTED.sub("", line) for line in wanted) if wanted else None

    def describe(self) -> str:
        """The cell, how it went and why, with the program it ran."""
        head = f"{self.cell.id} [{self.outcome}] {self.reason}".rstrip()
        program = self.cell.program.splitlines()
        lines = [head, f"  {self.cell.where}"]
        lines += [
            f"  {'program ' if not number else '        '} {line}"
            for number, line in enumerate(program)
        ]
        lines += [f"  {line}" for line in self.detail.splitlines()]
        return "\n".join(lines)


_GOT = re.compile(r"(?:output \d+: )?got\s+")
_EXPECTED = re.compile(r"(?:output \d+: )?expected ")
#: What a process says of the work it was given: an outcome, a reason, a detail, seconds.
_Said = tuple[str, str, str, float]


def run(cell: Cell, time_limit: float | None = TIME_LIMIT) -> Outcome:
    """Run a cell as the curriculum runs a case: its program in a fresh workbook, within
    the time limit, and what it gives compared with its oracle's answer as strictly."""
    return Outcome(cell, *_ran(cell, time_limit))


def _ran(cell: Cell, time_limit: float | None) -> _Said:
    result = curriculum.run(cell.case(), time_limit)
    return result.outcome, result.reason, result.detail, result.seconds


def run_all(
    cells: Sequence[Cell], jobs: int | None = None, time_limit: float | None = TIME_LIMIT
) -> list[Outcome]:
    """Run each cell in a process of its own, ``jobs`` of them at a time, and say how each
    went, in the order of the cells."""
    work = [functools.partial(_ran, cell, time_limit) for cell in cells]
    stop = None if time_limit is None else time_limit + GRACE
    return [
        Outcome(cell, *said)
        for cell, said in zip(cells, _each_alone(work, jobs, stop), strict=True)
    ]


def _each_alone(
    work: Sequence[Callable[[], _Said]], jobs: int | None, stop: float | None
) -> list[_Said]:
    """What each piece of work says, each done in a process of its own, forked from this
    one, so that all start from the same state, which none of them changes: ``jobs`` at a
    time, by default one for each processor. A process still running after ``stop``
    seconds is stopped. Where a process cannot be forked, the work is done here, in turn."""
    if "fork" not in multiprocessing.get_all_start_methods():
        return [each() for each in work]
    context = multiprocessing.get_context("fork")
    jobs = max(1, jobs or os.cpu_count() or 1)
    said: list[_Said | None] = [None] * len(work)
    waiting = collections.deque(enumerate(work))
    running: dict[Connection, tuple[int, Any, float]] = {}
    while waiting or running:
        while waiting and len(running) < jobs:
            number, each = waiting.popleft()
            receiver, sender = context.Pipe(duplex=False)
            process = context.Process(target=_say, args=(each, sender))
            process.start()
            sender.close()
            running[receiver] = (number, process, time.monotonic())
        for receiver in wait(list(running), timeout=1.0):
            assert isinstance(receiver, Connection)
            number, process, started = running.pop(receiver)
            try:
                said[number] = receiver.recv()
            except EOFError:  # it ended without a word: killed, or out of memory
                process.join()
                seconds = time.monotonic() - started
                reason = f"its process ended with status {process.exitcode}"
                said[number] = ("crashed", reason, "", seconds)
            receiver.close()
            process.join()
        for receiver, (number, process, started) in list(running.items()):
            seconds = time.monotonic() - started
            if stop is not None and seconds > stop:
                process.kill()
                process.join()
                receiver.close()
                del running[receiver]
                said[number] = ("timeout", f"stopped after {seconds:.0f} s", "", seconds)
    return [each for each in said if each is not None]


def _say(work: Callable[[], _Said], sender: Connection) -> None:
    """Do a piece of work in this process, and send what it says to the one that waits."""
    try:
        said = work()
    except BaseException as error:  # the harness's own failure, said rather than lost
        said = ("crashed", f"{type(error).__name__}: {error}".splitlines()[0], "", 0.0)
    sender.send(said)
    sender.close()


# -- checking an oracle ---------------------------------------------------------------------


@dataclass(frozen=True)
class Checked:
    """Whether an operand's oracle gives an answer, and its second route the same."""

    construct: Construct
    operand: Operand
    outcome: str  # passed or broken
    reason: str = ""

    @property
    def id(self) -> str:
        return f"{self.construct.id}//{self.operand.name}"

    @property
    def passed(self) -> bool:
        return self.outcome == "passed"

    def describe(self) -> str:
        where = f"{self.construct.where}:{self.operand.line}"
        return f"{self.id} [{self.outcome}] {self.reason}".rstrip() + f"\n  {where}"


def check(construct: Construct, operand: Operand) -> Checked:
    """Run an operand's oracle, then the construct's second route after it, which computes
    the answer another way and asserts that the two agree."""
    return Checked(construct, operand, *_checked(construct, operand)[:2])


def _checked(construct: Construct, operand: Operand) -> _Said:
    start = time.perf_counter()
    source = construct.python(operand) + construct.oracle
    case = curriculum.Case(
        id=f"{construct.id}//{operand.name}",
        title=construct.title,
        domain=construct.path.parent.name,
        topic=construct.path.stem,
        level=1,
        source="",
        program="",
        oracle=source + construct.second_route,
        path=construct.path,
        line=operand.line,
    )
    try:
        with curriculum._time_limit(ORACLE_LIMIT):
            expectation = curriculum.oracle(case)
    except TimeoutError:
        return "broken", f"the oracle takes over {ORACLE_LIMIT:g} s", "", ORACLE_LIMIT
    except AssertionError as error:
        said = f": {error}" if str(error) else ""
        return "broken", f"the second route disagrees{said}", "", time.perf_counter() - start
    except Exception as error:
        reason = f"the oracle fails: {type(error).__name__}: {error}".splitlines()[0]
        return "broken", reason, "", time.perf_counter() - start
    seconds = time.perf_counter() - start
    if not any(name in expectation for name in ("expected", "outputs")):
        return "broken", "the oracle sets neither expected nor outputs", "", seconds
    return "passed", "", "", seconds


def check_all(constructs: Sequence[Construct], jobs: int | None = None) -> list[Checked]:
    """Check the oracle of every operand of every construct, each in a process of its own."""
    pairs = [(construct, operand) for construct in constructs for operand in construct.operands]
    work = [functools.partial(_checked, construct, operand) for construct, operand in pairs]
    said = _each_alone(work, jobs, ORACLE_LIMIT + GRACE)
    return [
        Checked(construct, operand, outcome, reason)
        for (construct, operand), (outcome, reason, _, _) in zip(pairs, said, strict=True)
    ]


# -- forms that disagree ----------------------------------------------------------------------


@dataclass(frozen=True)
class Disagreement:
    """The forms of a construct that give different answers of one operand: each answer,
    with the forms that give it. ``None`` is the oracle's answer."""

    construct: Construct
    operand: Operand
    answers: Mapping[str | None, tuple[str, ...]]
    expected: str | None = None  # the oracle's answer, as shown beside a wrong one

    def describe(self) -> str:
        head = f"{self.construct.id}, of {self.operand.name}: {len(self.answers)} answers"
        lines = [head]
        for answer, forms in self.answers.items():
            if answer is None:
                answer = "the answer" + (f", {self.expected}" if self.expected else "")
            lines.append(f"  {answer}\n      {', '.join(forms)}")
        return "\n".join(lines)


def disagreements(outcomes: Iterable[Outcome]) -> list[Disagreement]:
    """For each operand of each construct, the forms that disagree with each other: those
    that gave an answer, grouped by it, where there is more than one. A form that passed
    gave the oracle's answer, and one that did not, the answer the curriculum shows for
    it. A form that must be refused, or that gave no answer, takes no part."""
    answers: dict[tuple[str, str], dict[str | None, list[str]]] = {}
    expected: dict[tuple[str, str], str] = {}
    cells: dict[tuple[str, str], Cell] = {}
    for outcome in outcomes:
        cell = outcome.cell
        if cell.form.refused or outcome.outcome not in ("passed", "wrong"):
            continue
        key = (cell.construct.id, cell.operand.name)
        cells.setdefault(key, cell)
        answers.setdefault(key, {}).setdefault(outcome.answer, []).append(cell.form.name)
        if outcome.expected is not None:
            expected.setdefault(key, outcome.expected)
    return [
        Disagreement(
            cells[key].construct,
            cells[key].operand,
            {answer: tuple(forms) for answer, forms in sorted(given.items(), key=_oracle_first)},
            expected.get(key),
        )
        for key, given in answers.items()
        if len(given) > 1
    ]


def _oracle_first(item: tuple[str | None, list[str]]) -> tuple[bool, int]:
    answer, forms = item
    return answer is not None, -len(forms)


# -- the scoreboard ---------------------------------------------------------------------------


_KINDS = ("passed", "wrong", "kept", "refused")


@dataclass
class Board:
    outcomes: list[Outcome] = field(default_factory=list)

    def by_construct(self) -> str:
        header = f"{'construct':38}{'forms':>6}{'operands':>9}{'cells':>7}"
        lines = [header + "".join(f"{kind:>8}" for kind in (*_KINDS, "other"))]
        constructs = list(dict.fromkeys(o.cell.construct.id for o in self.outcomes))
        for construct in [*constructs, None]:
            row = [o for o in self.outcomes if construct in (None, o.cell.construct.id)]
            forms = len({(o.cell.construct.id, o.cell.form.name) for o in row})
            operands = len({(o.cell.construct.id, o.cell.operand.name) for o in row})
            name = construct if construct is not None else "all"
            lines.append(f"{name:38}{forms:>6}{operands:>9}{len(row):>7}" + _counts(row))
        return "\n".join(lines)

    def by_notation(self) -> str:
        lines = [f"{'notation':38}{'forms':>6}{'':9}{'cells':>7}"]
        for notation, name in NOTATIONS.items():
            row = [o for o in self.outcomes if o.cell.form.notation == notation]
            forms = len({(o.cell.construct.id, o.cell.form.name) for o in row})
            lines.append(f"{name:38}{forms:>6}{'':9}{len(row):>7}" + _counts(row))
        return "\n".join(lines)

    def by_form(self) -> str:
        lines = []
        grouped: dict[tuple[str, str], list[Outcome]] = {}
        for outcome in self.outcomes:
            key = (outcome.cell.construct.id, outcome.cell.form.name)
            grouped.setdefault(key, []).append(outcome)
        last = None
        for (construct, form), row in grouped.items():
            if construct != last:
                lines.append(construct)
                last = construct
            notation = NOTATIONS[row[0].cell.form.notation]
            must = "  must be refused" if row[0].cell.form.refused else ""
            lines.append(f"  {form:34}{notation:>12}{len(row):>7}" + _counts(row) + must)
        return "\n".join(lines)

    def refused_most(self, top: int = 15) -> str:
        """What is refused most, by reason: the next things to build."""
        reasons: dict[str, list[str]] = collections.defaultdict(list)
        for outcome in self.outcomes:
            if outcome.outcome == "refused":
                reasons[curriculum._generalised(outcome.reason)].append(outcome.cell.id)
        lines = []
        for reason, ids in sorted(reasons.items(), key=lambda item: -len(item[1]))[:top]:
            more = f" (+{len(ids) - 3} more)" if len(ids) > 3 else ""
            lines.append(f"{len(ids):4d}  {reason}\n        {', '.join(ids[:3])}{more}")
        return "\n".join(lines)

    def forms_refused_most(self, top: int = 15) -> str:
        """The forms refused of the most operands, with the reason given most often."""
        refused: dict[str, list[Outcome]] = collections.defaultdict(list)
        for outcome in self.outcomes:
            if outcome.outcome == "refused":
                refused[f"{outcome.cell.construct.id}/{outcome.cell.form.name}"].append(outcome)
        lines = []
        for form, row in sorted(refused.items(), key=lambda item: -len(item[1]))[:top]:
            reasons = collections.Counter(outcome.reason for outcome in row)
            lines.append(f"{len(row):4d}  {form}\n        {reasons.most_common(1)[0][0]}")
        return "\n".join(lines)


def _counts(row: Sequence[Outcome]) -> str:
    counts = collections.Counter(outcome.outcome for outcome in row)
    other = len(row) - sum(counts[kind] for kind in _KINDS)
    return "".join(f"{counts[kind]:>8}" for kind in _KINDS) + f"{other:>8}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--construct",
        action="append",
        help="only this construct, as calculus.derivative, or area, as calculus (repeatable)",
    )
    parser.add_argument("--form", action="append", help="only forms of this name (repeatable)")
    parser.add_argument("--operand", action="append", help="only this operand (repeatable)")
    parser.add_argument("--notation", action="append", choices=list(NOTATIONS))
    parser.add_argument("--cell", action="append", help="only this cell (repeatable)")
    parser.add_argument("--recorded", action="store_true", help="only cells recorded as passing")
    parser.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="one cell in N, chosen by its identifier: the same cells every time",
    )
    parser.add_argument("--failures", action="store_true", help="say why each cell fails")
    parser.add_argument("--update", action="store_true", help="record the cells that pass")
    parser.add_argument("--list", action="store_true", help="name the cells, and run none")
    parser.add_argument("--show", metavar="CELL", help="a cell's program and oracle, as a case")
    parser.add_argument(
        "--oracles",
        action="store_true",
        help="run no cell: check that each oracle gives an answer, and its second route the same",
    )
    parser.add_argument("--jobs", type=int, help="cells run at a time (one for each processor)")
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT, help="seconds per cell")
    parser.add_argument(
        "--seeds",
        type=curriculum._seeds,
        help="run the cells once under each of these hash seeds, as 0,1,2, and list those "
        "whose outcome differs: findings, which do not fail the run",
    )
    parser.add_argument(
        "--outcomes",
        type=Path,
        metavar="FILE",
        help="write the outcome of each cell to FILE: a line each, with its reason",
    )
    if argv is None:
        fix_hash_seed()
    args = parser.parse_args(argv)
    if args.seeds:
        return _under_seeds(args)

    constructs = load()
    broken = problems(constructs)
    if broken:
        print("The specifications have problems of their own:", file=sys.stderr)
        for problem in broken:
            print(f"  {problem}", file=sys.stderr)
        return 1
    if args.oracles:
        return _oracles(_chosen_constructs(constructs, args), args)
    every = [cell for construct in constructs for cell in construct.cells()]
    chosen = _chosen(every, args)
    if args.show:
        shown = [cell for cell in every if cell.id == args.show]
        print(shown[0].shown() if shown else f"no cell is called {args.show}")
        return 0 if shown else 1
    if args.list:
        print("\n".join(cell.id for cell in chosen))
        return 0

    start = time.monotonic()
    board = Board(run_all(chosen, args.jobs, args.time_limit))
    seconds = time.monotonic() - start
    if args.outcomes:
        _write(args.outcomes, board.outcomes)
    _report(board, seconds, args)
    if args.update:
        _record(every, board.outcomes)
    failed = {outcome.cell.id for outcome in board.outcomes if not outcome.passed}
    regressions = sorted(passing() & failed)
    if regressions:
        print(f"\n{len(regressions)} cells recorded as passing now fail: {', '.join(regressions)}")
        return 1
    return 0


def _report(board: Board, seconds: float, args: argparse.Namespace) -> None:
    outcomes = board.outcomes
    passed = sum(outcome.passed for outcome in outcomes)
    share = passed / len(outcomes) if outcomes else 0
    constructs = len({outcome.cell.construct.id for outcome in outcomes})
    print(
        f"{len(outcomes)} cells of {constructs} constructs, {passed} pass ({share:.0%}), "
        f"in {seconds:.0f} s\n"
    )
    print(board.by_construct())
    print("\n" + board.by_notation())
    print("\nBy form:\n" + board.by_form())
    counts = collections.Counter(outcome.outcome for outcome in outcomes if not outcome.passed)
    if counts:
        print("\n" + ", ".join(f"{n} {outcome}" for outcome, n in counts.most_common()))
    wrong = [outcome for outcome in outcomes if outcome.outcome in ("wrong", "crashed", "broken")]
    if wrong:
        print("\nWrong answers and crashes, the most serious:")
        for outcome in wrong:
            print(outcome.describe())
    differing = disagreements(outcomes)
    if differing:
        print("\nForms that disagree with each other, by operand:")
        for disagreement in differing:
            print(disagreement.describe())
    if board.refused_most():
        print("\nWhat is refused most, the next things to build:")
        print(board.refused_most())
        print("\nThe forms refused of the most operands:")
        print(board.forms_refused_most())
    if args.failures:
        print("\nEvery failure:")
        for outcome in outcomes:
            if not outcome.passed:
                print(outcome.describe())


def _chosen_constructs(
    constructs: Sequence[Construct], args: argparse.Namespace
) -> list[Construct]:
    """The constructs a command line chooses, by name or by area."""
    wanted = args.construct
    return [
        construct
        for construct in constructs
        if not wanted or construct.id in wanted or construct.id.partition(".")[0] in wanted
    ]


def _chosen(cells: Sequence[Cell], args: argparse.Namespace) -> list[Cell]:
    """The cells a command line chooses, in order."""
    constructs = {construct.id for construct in _chosen_constructs(_constructs(cells), args)}
    recorded = passing() if args.recorded else None
    chosen = [
        cell
        for cell in cells
        if cell.construct.id in constructs
        and (not args.form or cell.form.name in args.form)
        and (not args.operand or cell.operand.name in args.operand)
        and (not args.notation or cell.form.notation in args.notation)
        and (not args.cell or cell.id in args.cell)
        and (recorded is None or cell.id in recorded)
    ]
    return sample(chosen, args.sample) if args.sample else chosen


def _constructs(cells: Iterable[Cell]) -> list[Construct]:
    return list({cell.construct.id: cell.construct for cell in cells}.values())


def sample(cells: Sequence[Cell], one_in: int) -> list[Cell]:
    """One cell in ``one_in``, chosen by its identifier alone, so that the sample is the
    same every time, whatever else is in the matrix."""
    return [cell for cell in cells if zlib.crc32(cell.id.encode("utf-8")) % one_in == 0]


def _oracles(constructs: Sequence[Construct], args: argparse.Namespace) -> int:
    checked = check_all(constructs, args.jobs)
    if args.outcomes:
        _write(args.outcomes, checked)
    failed = [each for each in checked if not each.passed]
    for each in failed:
        print(each.describe())
    routes = sum(bool(construct.second_route.strip()) for construct in constructs)
    print(
        f"{len(checked)} oracles of {len(constructs)} constructs, {routes} with a second "
        f"route: {len(checked) - len(failed)} agree"
    )
    return 1 if failed else 0


def _write(path: Path, outcomes: Iterable[Outcome | Checked]) -> None:
    """Write the outcome of each cell, or of each oracle, a line each, as the curriculum
    writes those of its cases."""
    lines = []
    for outcome in outcomes:
        identifier = outcome.id if isinstance(outcome, Checked) else outcome.cell.id
        lines.append(f"{identifier}\t{outcome.outcome}\t{' '.join(outcome.reason.split())}\n")
    path.write_text("".join(lines), encoding="utf-8")


def _record(cells: Iterable[Cell], outcomes: Iterable[Outcome]) -> None:
    """Record the cells that pass, keeping those not run this time."""
    ran = {outcome.cell.id: outcome.passed for outcome in outcomes}
    known = {cell.id for cell in cells}
    kept = {cell_id for cell_id in passing() if cell_id in known and cell_id not in ran}
    now = kept | {cell_id for cell_id, ok in ran.items() if ok}
    header = "# Cells that pass. Written by benchmarks/notations.py --update.\n"
    PASSING.write_text(
        header + "".join(f"{cell_id}\n" for cell_id in sorted(now)), encoding="utf-8"
    )
    print(f"\nrecorded {len(now)} passing cells in {PASSING.relative_to(curriculum.ROOT.parent)}")


# -- under several hash seeds -------------------------------------------------------------


def _under_seeds(args: argparse.Namespace) -> int:
    """Run the chosen cells once under each hash seed, each run in a process of its own,
    and list the cells whose outcome differs between them: findings, which do not fail
    the run."""
    options = ["--time-limit", str(args.time_limit)]
    for option, values in (
        ("--construct", args.construct),
        ("--form", args.form),
        ("--operand", args.operand),
        ("--notation", args.notation),
        ("--cell", args.cell),
    ):
        options += [part for value in values or [] for part in (option, value)]
    options += ["--recorded"] if args.recorded else []
    options += ["--sample", str(args.sample)] if args.sample else []
    options += ["--jobs", str(args.jobs)] if args.jobs else []
    outcomes: dict[str, dict[str, tuple[str, str]]] = {}
    with tempfile.TemporaryDirectory() as folder:
        for seed in args.seeds:
            written = Path(folder) / f"{seed}.tsv"
            command = [sys.executable, str(Path(__file__).resolve()), *options]
            ran = subprocess.run(
                [*command, "--outcomes", str(written)],
                env={**os.environ, "PYTHONHASHSEED": seed},
                capture_output=True,
                text=True,
                check=False,
            )
            if not written.is_file():
                said = ran.stdout + ran.stderr
                print(f"the matrix did not run under seed {seed}:\n{said}", file=sys.stderr)
                return 1
            outcomes[seed] = curriculum._outcomes(written)
            print(f"under seed {seed}: {(ran.stdout.splitlines() or [''])[0]}")
    found = curriculum.differing(outcomes)
    if not found:
        print(f"\nEvery cell has the same outcome under seeds {', '.join(args.seeds)}.")
        return 0
    print(f"\nThe outcome of {len(found)} cells depends on the hash seed:")
    for cell_id, seeds_of in found.items():
        said = "; ".join(
            f"{outcome} under {', '.join(seeds)}" for outcome, seeds in seeds_of.items()
        )
        print(f"{cell_id}: {said}")
        for seed in args.seeds:
            outcome, reason = outcomes[seed][cell_id]
            if reason:
                print(f"  under {seed}, {outcome}: {reason}")
    return 0


def recorded_cells(one_in: int | None = None) -> Iterator[Cell]:
    """The cells recorded as passing, or one in ``one_in`` of them, as :func:`sample`
    chooses."""
    recorded = passing()
    cells = [cell for construct in load() for cell in construct.cells() if cell.id in recorded]
    yield from (sample(cells, one_in) if one_in else cells)


if __name__ == "__main__":
    sys.exit(main())
