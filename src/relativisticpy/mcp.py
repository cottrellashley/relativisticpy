r"""The MCP server: RelativisticPy as a tool for AI agents.

``relativisticpy mcp`` serves RelativisticPy over the Model Context Protocol, on standard
input and output, to any MCP client: Claude Code, Claude Desktop, an IDE, or an agent of
one's own. An agent writes mathematics as it would in a paper, and RelativisticPy computes
it exactly, or says where it went wrong.

Tools:

- ``compute`` runs a program and gives, for each statement, its values, what it defines,
  notes on how it was read, and its error, if it failed; a named ``session`` keeps
  definitions from one call to the next;
- ``verify`` checks one claimed equality or step of a derivation, ``lhs = rhs`` or a chain
  ``a = b = c``, with definitions such as a metric as its context: it holds, differs,
  relates, is not decided, or is an error, with the difference of its sides in LaTeX;
- ``check`` checks every display equation of a LaTeX or Markdown document;
- ``help`` gives the documentation, offline; ``list_sessions`` and ``reset_session``
  manage the sessions.

The language reference and every page of the documentation are resources,
``relativisticpy://docs/{topic}``, and two prompts, ``verify_derivation`` and
``check_paper``, start the work they are named after.

Each call runs in a worker process of its own, which the server stops when the call runs
over its time limit; a statement that runs over is stopped first, softly, so that a
session keeps what it defined. Workbooks are given no Python scope; documents are read
from files only in the folders the server is started with (``--allow-files``); results
are cut to a size, with a note. The server needs the ``mcp`` extra:
``pip install "relativisticpy[mcp]"``. See ``relativisticpy help mcp``.
"""

# No ``from __future__ import annotations`` here: the SDK reads the annotations of the
# tools defined in create_server, which name what is imported there, at run time.

import argparse
import contextlib
import json
import math
import os
import re
import signal
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import IO, TYPE_CHECKING, Annotated, Any, Literal, cast
from unittest import mock

from relativisticpy import __version__, checking, documents
from relativisticpy import help as documentation
from relativisticpy.checking import Report, Verdict
from relativisticpy.documents import Document, Equation
from relativisticpy.errors import (
    FileError,
    NotComputedError,
    RelativisticPyError,
    UsageError,
    internal_error,
    place,
)
from relativisticpy.language import Source, Span, parse
from relativisticpy.steps import Snapshot, as_data, defined_between, snapshot, statements_of
from relativisticpy.workbook import Step, Workbook, notebook_blocks

if sys.version_info >= (3, 12):
    from typing import TypedDict
else:  # pydantic reads the results' schemas, and below 3.12 only from typing_extensions'
    try:
        from typing_extensions import TypedDict
    except ImportError:  # without the mcp extra, which brings it, there is no server to run
        from typing import TypedDict

if TYPE_CHECKING:
    from anyio.abc import Process
    from anyio.streams.buffered import BufferedByteReceiveStream
    from mcp.server import MCPServer

__all__ = ["Settings", "create_server", "main"]

#: Where the documentation is online.
DOCS = "https://relativisticpy.org"

#: The longest a call may run, in seconds, unless the server is started with another.
TIME_LIMIT = 60.0
#: The most characters a result may have, as JSON, unless the server is started with another.
MAX_OUTPUT = 50_000
#: The most named sessions at once, unless the server is started with another number.
MAX_SESSIONS = 8
#: The memory each worker process may use, in MiB, where the system enforces it (Linux).
MEMORY_LIMIT = 4096
#: The time a check gives each statement, as ``relativisticpy check`` does, in seconds.
STATEMENT_TIME_LIMIT = 30.0
#: The most characters a program, a claim or a document may have.
MAX_INPUT = 1_000_000

#: How long a worker has, after its call's time limit, to answer before it is stopped.
GRACE = 5.0
#: How long a new worker has to start: the first, after an installation, compiles.
STARTUP = 120.0
#: How many workers compute for calls without a session at once.
SCRATCH_WORKERS = 4

#: The names a program, a claim and its context are located by.
INPUT, CLAIM, CONTEXT = "<input>", "<claim>", "<context>"

#: What a session may be called.
_SESSION_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}")

#: The verdicts and conditions a check may fail on, as ``--fail-on`` takes them.
FailOn = Literal[
    "holds",
    "differs",
    "relates",
    "ill-formed",
    "defines",
    "computes",
    "not computed",
    "not understood",
    "not parsed",
    "time limit",
    "empty",
]

#: What ``verify`` says of a claim, from the verdict a check gives it.
VerifyVerdict = Literal["holds", "differs", "relates", "not decided", "error"]

_VERIFIED: dict[Verdict, str] = {
    Verdict.HOLDS: "holds",
    Verdict.DIFFERS: "differs",
    Verdict.RELATES: "relates",
    Verdict.NOT_COMPUTED: "not decided",
    Verdict.NOT_UNDERSTOOD: "error",
    Verdict.ILL_FORMED: "error",
    Verdict.NOT_PARSED: "error",
    Verdict.COMPUTES: "error",
    Verdict.DEFINES: "error",
}
#: Which verdict of a chain's links decides the chain's: the first that one link has.
_PRECEDENCE = ("error", "differs", "relates", "not decided", "holds")


@dataclass(frozen=True)
class Settings:
    """How a server is bounded, as ``relativisticpy mcp`` is started:

    - ``allowed``, the folders ``check`` may read documents from; none, by default, so
      that it takes only text;
    - ``time_limit``, the longest a call may run, in seconds; a call may ask for less;
    - ``max_output``, the most characters a result may have, as JSON;
    - ``max_sessions``, the most named sessions at once;
    - ``memory_limit``, the memory each worker process may use, in MiB, where the system
      enforces it; ``None`` for no limit.
    """

    allowed: tuple[Path, ...] = ()
    time_limit: float = TIME_LIMIT
    max_output: int = MAX_OUTPUT
    max_sessions: int = MAX_SESSIONS
    memory_limit: int | None = MEMORY_LIMIT

    def __post_init__(self) -> None:
        if not (self.time_limit > 0 and math.isfinite(self.time_limit)):
            raise UsageError(f"a time limit is a positive number of seconds, not {self.time_limit}")
        if self.max_output < 1000:
            raise UsageError(
                f"a result needs room: max_output is 1000 or more, not {self.max_output}"
            )
        if self.max_sessions < 0:
            raise UsageError(f"max_sessions is a number of sessions, not {self.max_sessions}")
        if self.memory_limit is not None and self.memory_limit < 256:
            raise UsageError(f"a worker needs 256 MiB or more, not {self.memory_limit}")

    def limit(self, asked: float | None) -> float:
        """The time a call has: what it asks for, up to the server's limit."""
        return self.time_limit if asked is None else min(asked, self.time_limit)


# -- the results, as their schemas say -----------------------------------------------------


class Defined(TypedDict):
    """Something a statement defined: its name, what kind of thing it is (``value``,
    ``function``, ``tensor``, ``metric``, ``coordinate`` or ``space``) and how it is
    written."""

    name: str
    kind: str
    written: str


class Statement(TypedDict):
    """One statement of a program and what came of it, as ``relativisticpy --format json``
    gives it: where it is, the statement as written, its values (each as
    ``relativisticpy.display.data`` describes it, with its ``text``, ``unicode`` and
    ``latex`` forms, and whether it is ``complete``, the ``reasons`` it is not, and
    whether it is ``approximate``), what it defined, notes on how it was read, what its
    values are given under that books assume and the program did not say, as ``a > 0``,
    and its error, if it failed (``RelativisticPyError.to_dict()``: type, message, place,
    notes and hint)."""

    file: str | None
    line: int | None
    column: int | None
    end_line: int | None
    end_column: int | None
    statement: str | None
    values: list[dict[str, Any]]
    defines: list[Defined]
    notes: list[str]
    assuming: list[str]
    error: dict[str, Any] | None


class Computed(TypedDict):
    """What a program gave: one entry for each statement, in order; whether every
    statement ran and none failed (``ok``), and whether besides every value is a complete
    answer, with nothing kept as written (``complete``); and notes, such as a time limit
    reached or a result cut short."""

    session: str | None
    statements: list[Statement]
    ok: bool
    complete: bool
    seconds: float
    notes: list[str]
    truncated: bool


class Judgement(TypedDict):
    """The verdict on one statement of a claim, or of its context: where it is, the
    statement, the verdict, the verdict ``relativisticpy check`` gives it, why, and the
    left side minus the right, where it differs or relates, in each form of
    ``relativisticpy.display``."""

    verdict: str
    check: str
    file: str
    line: int | None
    column: int | None
    end_line: int | None
    end_column: int | None
    statement: str
    detail: str
    reason: str | None
    hint: str | None
    notes: list[str]
    difference: dict[str, Any] | None


class Verified(TypedDict):
    """The verdict on a claim, and on each of its links; the statements of the context
    that did not simply define something, as an error; and notes."""

    verdict: VerifyVerdict
    links: list[Judgement]
    context: list[Judgement]
    seconds: float
    notes: list[str]
    truncated: bool


class Checked(TypedDict):
    """The report on a document, as ``Report.to_dict()`` gives it, with the policy it was
    checked against, whether it failed it, which findings are listed, and notes."""

    file: str
    equations: int
    statements: int
    seconds: float
    lines: list[int | None] | None
    context: int
    counts: dict[str, int]
    reasons: dict[str, int]
    findings: list[dict[str, Any]]
    warnings: list[str]
    ok: bool
    policy: list[str]
    failed: bool
    listed: Literal["failing", "all"]
    notes: list[str]
    truncated: bool


class SessionInfo(TypedDict):
    """A named session: the statements it has run, what it defines, whether a call is
    running in it, and how long ago it was last used, in seconds."""

    session: str
    statements: int
    defined: list[dict[str, Any]]
    busy: bool
    idle_seconds: float


class Sessions(TypedDict):
    """The named sessions, and the limits of this server."""

    sessions: list[SessionInfo]
    limits: dict[str, Any]


class Reset(TypedDict):
    """Whether a session was there to forget."""

    session: str
    reset: bool
    notes: list[str]


# == the worker: one process, one request at a time =========================================
#
# A worker reads requests from its standard input, one line of JSON each, and answers each
# with one line of JSON on its standard output: {"result": ...}, {"error": ...} or
# {"interrupted": true}. It computes in its main thread, so that a time limit can stop a
# statement with SIGALRM, as relativisticpy check stops one, and SIGINT can interrupt it.


def _work() -> None:
    """Serve requests until the server closes the pipe: the worker's main function."""
    settings: dict[str, Any] = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    wire = _claim_standard_output()
    _watch_parent()
    _limit_memory(settings.get("memory_limit"))
    engine = _Engine(settings)
    _answer(wire, {"ready": True, "pid": os.getpid()})
    while True:
        try:
            line = sys.stdin.buffer.readline()
        except KeyboardInterrupt:  # an interruption that came after its request was answered
            continue
        if not line:
            return  # the server is done
        try:
            reply = engine.handle(json.loads(line))
        except KeyboardInterrupt:
            reply = {"interrupted": True}
        except _Expired:  # an alarm that came as a request ended
            reply = {"error": _error_data(_over_time(float(engine.last_limit), None))}
        except Exception as error:  # a failure of the worker's own: reported, not fatal
            reply = {"error": _error_data(internal_error(error, during="serving this request"))}
        _answer(wire, reply)


def _claim_standard_output() -> IO[bytes]:
    """The pipe to the server, kept for answers alone: whatever else writes to standard
    output, as a library's print, goes to standard error instead."""
    wire = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return wire


def _answer(wire: IO[bytes], reply: dict[str, Any]) -> None:
    wire.write(json.dumps(reply, ensure_ascii=True, default=str).encode("ascii") + b"\n")
    wire.flush()


def _watch_parent() -> None:
    """End the worker if the server ends without stopping it, as when it is killed: its
    parent is then another process."""
    if sys.platform == "win32":
        return
    parent = os.getppid()

    def watch() -> None:
        while os.getppid() == parent:
            time.sleep(1.0)
        os._exit(0)

    threading.Thread(target=watch, name="relativisticpy-watch", daemon=True).start()


def _limit_memory(mebibytes: int | None) -> None:
    """Bound the worker's address space, where the system enforces it: an allocation
    beyond it fails with MemoryError, which a statement reports, rather than exhausting
    the machine."""
    if mebibytes is None or sys.platform == "win32":
        return
    import resource

    wanted = int(mebibytes) * 1024 * 1024
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        wanted = min(wanted, hard)
    with contextlib.suppress(ValueError, OSError):  # not every system lets it be set
        resource.setrlimit(resource.RLIMIT_AS, (wanted, hard))


class _Expired(BaseException):
    """The alarm of a call's time limit. Like ``KeyboardInterrupt`` it is not an
    ``Exception``, so that no ``except Exception`` in SymPy takes it for a failure and
    computes on."""


@contextlib.contextmanager
def _alarm(seconds: float) -> Iterator[None]:
    """Raise :class:`_Expired` in the block once ``seconds`` have passed: with SIGALRM, in
    the main thread, where there is one; elsewhere the server's stop is the only limit."""
    usable = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
    if not usable:
        yield
        return

    def expire(signum: int, frame: Any) -> None:
        raise _Expired

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, max(seconds, 0.001))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _shorten_alarm(deadline: float) -> None:
    """Bring a check's own time limit on the statement in hand forward to the call's
    deadline, if that comes first. Only a limit that is running is changed, so that the
    alarm always has its handler."""
    if (
        not hasattr(signal, "setitimer")
        or threading.current_thread() is not threading.main_thread()
    ):
        return
    remaining = deadline - time.monotonic()
    current, again = signal.getitimer(signal.ITIMER_REAL)
    if 0 < remaining < current:
        signal.setitimer(signal.ITIMER_REAL, remaining, again)


class _Engine:
    """What a worker does: compute a program, verify a claim or check a document, in a
    workbook of its own or in the session it holds."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.max_output = int(settings.get("max_output", MAX_OUTPUT))
        self.roots = [Path(folder) for folder in settings.get("allowed", [])]
        self.session: Workbook | None = None
        self.inputs = 0  # the programs the session has run, which name its inputs
        self.last_limit = TIME_LIMIT

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        """The answer to one request."""
        operation = request.get("op")
        arguments: dict[str, Any] = request.get("args") or {}
        limit = self.last_limit = float(request.get("limit") or TIME_LIMIT)
        operations: dict[str, Callable[..., dict[str, Any]]] = {
            "compute": self.compute,
            "verify": self.verify,
            "check": self.check,
        }
        if operation == "ping":
            return {"result": {}}
        if operation not in operations:
            raise ValueError(f"no operation {operation!r}")
        try:
            result = operations[str(operation)](limit=limit, **arguments)
        except RelativisticPyError as error:
            return {"error": _error_data(error)}
        return {"result": _capped(result, self.max_output)}

    # -- compute ----------------------------------------------------------------------------

    def compute(self, *, source: str, session: bool, limit: float) -> dict[str, Any]:
        """Run a program, one statement at a time, in the session or a new workbook,
        until it ends or its time runs out."""
        started = time.monotonic()
        deadline = started + limit
        if session:
            if self.session is None:
                self.session = Workbook()
            workbook = self.session
            self.inputs += 1
            name = f"<input {self.inputs}>"
        else:
            workbook = Workbook()
            name = INPUT
        statements: list[dict[str, Any]] = []
        notes: list[str] = []
        complete = True
        done = 0  # where the last statement that ran ends
        try:
            steps = workbook.each(source, name=name, recover=True)
            before = snapshot(workbook)
            while True:
                if time.monotonic() >= deadline:  # the time ran out between two statements
                    complete = False
                    notes += _not_run(source, name, done, limit)
                    break
                finished = False  # whether the statement in hand has been run and shown
                try:
                    with _alarm(deadline - time.monotonic()):
                        step = next(steps, None)
                        if step is None:
                            break
                        after = snapshot(workbook)
                        statements.append(_entry(step, before, after))
                        before, done, finished = after, step.span.end, True
                except _Expired:
                    complete = False
                    running = None if finished else _running(source, name, done)
                    if running is not None:
                        statements.append(_failed(_over_time(limit, running, session=session)))
                        done = running.end
                    notes += _not_run(source, name, done, limit)
                    break
        except RelativisticPyError as error:  # the program could not be run at all
            statements.append(_failed(error))
        finally:
            if not session:
                workbook.close()
        if not source.strip():
            notes.append("the program is empty: there was nothing to run")
        ok = complete and all(entry["error"] is None for entry in statements)
        result: dict[str, Any] = {
            "session": None,
            "statements": statements,
            "ok": ok,
            "complete": ok and _answered(statements),
            "seconds": round(time.monotonic() - started, 3),
            "notes": notes,
            "truncated": False,
        }
        if session:
            result["defined"] = _definitions(workbook)
        return result

    # -- verify -----------------------------------------------------------------------------

    def verify(self, *, claim: str, context: str | None, limit: float) -> dict[str, Any]:
        """Judge a claim, after its context, as a check judges the equations of a paper."""
        started = time.monotonic()
        equations = []
        if context and context.strip():
            equations.append(Equation(Source(context, CONTEXT), "$$"))
        equations.append(Equation(Source(claim, CLAIM), "$$"))
        cut: list[Source] = []
        with _bounded(started + limit, cut)() as workbook:
            report = checking.check_equations(workbook, equations, CLAIM, time_limit=limit)
        links = [f for f in report.findings if f.file == CLAIM and f.verdict is not Verdict.DEFINES]
        defining = [f for f in report.findings if f.file == CLAIM and f.verdict is Verdict.DEFINES]
        judged = [_judgement(finding) for finding in links]
        notes: list[str] = []
        if not judged:
            judged = [_nothing_to_verify(finding) for finding in defining[:1]]
            if not judged:
                notes.append("nothing to verify: the claim has no statement")
        problems = [
            _judgement(finding)
            for finding in report.findings
            if finding.file == CONTEXT and finding.verdict not in _QUIET
        ]
        if any(problem["verdict"] == "error" for problem in problems):
            notes.append(
                "the context did not run as written (see context): the claim was judged "
                "without what its failing statements would define"
            )
        if cut:
            notes.append(f"the time limit of {limit:g} s was reached before the claim was judged")
        verdicts = [judgement["verdict"] for judgement in judged] or ["error"]
        if cut and not judged:
            verdicts = ["not decided"]
        verdict = next(v for v in _PRECEDENCE if v in verdicts)
        return {
            "verdict": verdict,
            "links": judged,
            "context": problems,
            "seconds": round(time.monotonic() - started, 3),
            "notes": notes,
            "truncated": False,
        }

    # -- check ------------------------------------------------------------------------------

    def check(
        self,
        *,
        document: str | None,
        path: str | None,
        markdown: bool,
        name: str,
        lines: str | None,
        policy: list[str],
        strict: bool,
        everything: bool,
        statement_time_limit: float,
        limit: float,
    ) -> dict[str, Any]:
        """Check a document, given as text or as the path of a file in the folders the
        server may read, against a policy."""
        started = time.monotonic()
        failing: set[str] = set(checking.STRICT) if strict else set()
        failing |= set(policy)
        Report("", 1).failed(failing)  # each name a verdict, or refused before the work
        span = checking.line_range(lines) if lines else None
        each = min(statement_time_limit, limit)
        cut: list[Source] = []
        with _confined(self.roots), _bounded(started + limit, cut)() as workbook:
            if path is not None:
                report = self._check_file(workbook, Path(path), markdown, span, each)
            elif markdown:
                blocks = [Equation(block, "$$") for block in notebook_blocks(document or "", name)]
                report = checking.check_equations(workbook, blocks, name, span, each)
            else:
                report = workbook.check(Document(document or "", name), lines=span, time_limit=each)
        data = report.to_dict()
        data["policy"] = sorted(failing)
        data["failed"] = bool(failing) and report.failed(failing)
        data["listed"] = "all" if everything else "failing"
        notes: list[str] = []
        if not everything:
            listed = report.listing(False)
            data["findings"] = [finding.to_dict() for finding in listed]
            hidden = len(report) - len(listed)
            if hidden:
                notes.append(
                    f"findings lists what fails --strict and what ran over its time limit; "
                    f"the other {hidden} statements are counted in counts: all=true lists "
                    "every statement"
                )
        if cut:
            first = cut[0]
            notes.append(
                f"the time limit of {limit:g} s was reached: the equations from "
                f"{first.name}:{first.origin[0]} on were not checked; give it longer, or "
                "check fewer lines at a time, with lines"
            )
        data["notes"] = notes
        data["truncated"] = False
        return data

    @staticmethod
    def _check_file(
        workbook: Workbook,
        path: Path,
        markdown: bool,
        lines: tuple[int | None, int | None] | None,
        each: float,
    ) -> Report:
        if markdown or path.suffix.lower() in (".md", ".markdown"):
            text = documents.read_text(path)
            blocks = [Equation(block, "$$") for block in notebook_blocks(text, str(path))]
            return checking.check_equations(workbook, blocks, str(path), lines, each)
        return workbook.check(path, lines=lines, time_limit=each)


#: The verdicts on a context that need no mention: it defined, computed or held.
_QUIET = frozenset({Verdict.DEFINES, Verdict.COMPUTES, Verdict.HOLDS})


def _definitions(workbook: Workbook) -> list[dict[str, Any]]:
    """Everything a session defines, as data."""
    try:
        return [{"name": n, **d.to_dict()} for n, d in workbook.names().items()]
    except Exception:  # a list of what is defined is never a reason to fail
        return []


def _entry(step: Step, before: Snapshot, after: Snapshot) -> dict[str, Any]:
    """A statement and what came of it, as a line of ``relativisticpy --format json``,
    with what it defined, told by what the workbook defined ``before`` it and ``after``."""
    defined = defined_between(before, after)
    try:
        return as_data(step, defined)
    except Exception as error:  # a failure of our own, while describing it: reported
        failure = internal_error(error, span=step.span, during="describing this result")
        return as_data(replace(step, values=None, error=failure), defined)


def _answered(statements: Sequence[dict[str, Any]]) -> bool:
    """Whether every value of the statements is a complete answer."""
    return all(value.get("complete", True) for entry in statements for value in entry["values"])


def _failed(error: RelativisticPyError) -> dict[str, Any]:
    """The entry of a program that could not be run, as one statement that failed."""
    where = place(error.span)
    return {
        "file": where["file"],
        "line": where["line"],
        "column": where["column"],
        "end_line": where["end_line"],
        "end_column": where["end_column"],
        "statement": where["text"],
        "values": [],
        "defines": [],
        "notes": [],
        "assuming": [],
        "error": error.to_dict(),
    }


def _over_time(limit: float, span: Span | None, *, session: bool = False) -> NotComputedError:
    """The error of a statement that ran over its call's time limit."""
    notes = ["the definitions before it are kept"] if session else []
    return NotComputedError(
        f"over the time limit of {limit:g} s",
        span=span,
        notes=notes,
        hint="give it longer with time_limit, up to the server's limit, or compute it in "
        "steps, in a session, defining its parts first",
    )


def _statements(source: str, name: str) -> list[Span]:
    """Where each statement of a program is, in order, and where reading each that cannot
    be read failed."""
    try:
        program = parse(Source(source, name), recover=True)
    except RelativisticPyError:
        return []
    found = (planned.statement.span for planned in statements_of(program))
    return [span for span in found if span is not None]


def _running(source: str, name: str, done: int) -> Span | None:
    """The statement that was running when the time ran out, the last that ran having
    ended at ``done``: the first that ends after it, as the next link of a chain does."""
    return next((span for span in _statements(source, name) if span.end > done), None)


def _not_run(source: str, name: str, done: int, limit: float) -> list[str]:
    """The note on the statements a time limit left unrun: those after ``done``."""
    later = [span for span in _statements(source, name) if span.start >= done]
    if not later:
        return []
    first = later[0]
    count = f"{len(later)} statement{'s' if len(later) != 1 else ''}"
    return [
        f"the time limit of {limit:g} s was reached: {count}, from {first.source.name}:"
        f"{first.line}:{first.column} on, did not run"
    ]


def _judgement(finding: checking.Finding) -> dict[str, Any]:
    """A finding as ``verify`` gives it: its own verdict, and the check's."""
    data = finding.to_dict()
    judged: dict[str, Any] = {"verdict": _VERIFIED[finding.verdict], "check": data["verdict"]}
    judged.update((key, value) for key, value in data.items() if key != "verdict")
    if finding.verdict is Verdict.COMPUTES:
        judged["detail"] = "this is an expression, not a claim: write an equation, lhs = rhs"
        judged["notes"] = [*judged["notes"], f"its value is {finding.detail}"]
        judged["hint"] = judged["hint"] or "to claim it has a value, write = and the value"
    return judged


def _nothing_to_verify(finding: checking.Finding) -> dict[str, Any]:
    """The judgement of a claim that only defines: ``=`` defines what has no meaning yet."""
    judged = _judgement(finding)
    judged["detail"] = (
        "nothing to verify: this defines its left side, which had no meaning yet, as = "
        "defines a new name"
    )
    judged["hint"] = (
        "to claim something about it, define it first, in the context; or claim an "
        "equation between quantities the context defines"
    )
    return judged


def _bounded(deadline: float, cut: list[Source]) -> type[Workbook]:
    """A kind of workbook that runs no statement once a call's deadline has passed, and
    records the sources it did not finish. A check runs its equations in a new workbook
    of the kind it is given, so each of them comes through here."""

    class Bounded(Workbook):
        def each(
            self, source: str | Source, *, name: str = INPUT, recover: bool = False
        ) -> Iterator[Step]:
            text = source if isinstance(source, Source) else Source(source, name)
            if time.monotonic() >= deadline:
                cut.append(text)
                return iter(())
            return _until(super().each(source, name=name, recover=recover), deadline, cut, text)

    return Bounded


def _until(
    steps: Iterator[Step], deadline: float, cut: list[Source], source: Source
) -> Iterator[Step]:
    """The steps of a source until the deadline, each stopped by it if it runs past."""
    while True:
        _shorten_alarm(deadline)
        step = next(steps, None)
        if step is None:
            return
        yield step
        if time.monotonic() >= deadline:
            cut.append(source)
            return


@contextlib.contextmanager
def _confined(roots: Sequence[Path]) -> Iterator[None]:
    """Let a check read files only in ``roots``: the document, and every file it reads
    with \\input, \\include or \\usepackage, wherever it points. Every file a check
    reads, it reads through ``read_text``."""
    read = documents.read_text

    def guarded(path: Path) -> str:
        if not any(_inside(Path(path).resolve(), root) for root in roots):
            raise FileError(
                f"{path} is outside the folders this server may read",
                hint="a document, and the files it includes, must be in a folder the "
                "server was started with, by --allow-files",
            )
        return read(path)

    with (
        mock.patch.object(documents, "read_text", guarded),
        mock.patch.object(checking, "read_text", guarded),
    ):
        yield


def _inside(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def _error_data(error: RelativisticPyError) -> dict[str, Any]:
    """An error as the server is told it: as data, and as the command line prints it."""
    return {**error.to_dict(), "rendered": error.render()}


# -- cutting results to size ----------------------------------------------------------------


def _size(value: Any) -> int:
    """How many characters a result has as the client is sent it, as indented JSON."""
    return len(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _shortened(value: Any, width: int, items: int | None) -> Any:
    """``value`` with every string cut to ``width`` characters and every list to
    ``items`` items, keeping its shape."""
    if isinstance(value, str):
        if len(value) <= width:
            return value
        return value[:width] + f"… [cut: {len(value):,} characters]"
    if isinstance(value, list):
        kept = value if items is None else value[:items]
        return [_shortened(item, width, items) for item in kept]
    if isinstance(value, dict):
        return {key: _shortened(item, width, items) for key, item in value.items()}
    return value


#: The lists of a result whose length a note gives when they are cut.
_MAIN_LISTS = ("statements", "findings", "links")


def _capped(result: dict[str, Any], limit: int) -> dict[str, Any]:
    """A result cut to ``limit`` characters if it is longer, the note that says what was
    cut included: strings first, then lists."""
    if _size(result) <= limit:
        return result
    for width in (4000, 1000, 300, 100):
        cut = _cut(
            _shortened(result, width, None),
            f"the result was cut to fit {limit:,} characters: text longer than {width:,} "
            "characters is cut short, marked …",
        )
        if _size(cut) <= limit:
            return cut
    longest = max((len(v) for v in _lists(result)), default=0)
    low, high = 0, longest
    while low < high:  # the most items each list may keep
        middle = (low + high + 1) // 2
        if _size(_kept(result, limit, middle)) <= limit:
            low = middle
        else:
            high = middle - 1
    return _kept(result, limit, low)


def _kept(result: dict[str, Any], limit: int, items: int) -> dict[str, Any]:
    """A result with its text cut to 100 characters and its lists to ``items`` items, with
    the note that says so."""
    shortened = _shortened(result, 100, items)
    lost = [
        f"{key}: {len(shortened.get(key, []))} of {len(result[key])} kept"
        for key in _MAIN_LISTS
        if isinstance(result.get(key), list) and len(result[key]) > items
    ]
    detail = f" ({'; '.join(lost)})" if lost else ""
    return _cut(
        shortened,
        f"the result was cut to fit {limit:,} characters: text longer than 100 characters "
        f"is cut short, marked …, and lists keep their first {items} items{detail}; ask for "
        "less at a time",
    )


def _lists(value: Any) -> Iterator[list[Any]]:
    if isinstance(value, list):
        yield value
        for item in value:
            yield from _lists(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _lists(item)


def _cut(result: dict[str, Any], note: str) -> dict[str, Any]:
    result["truncated"] = True
    result["notes"] = [*result.get("notes", []), note]
    return result


# == the server's side of a worker ===========================================================


class _CallError(Exception):
    """A call that failed as a whole, as the agent is told it, as a diagnostic is: what,
    then why, in notes, then how to go on, in a hint."""

    def __init__(self, message: str, hint: str | None = None, notes: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.message, self.hint, self.notes = message, hint, list(notes)

    def __str__(self) -> str:
        lines = [self.message, *(f"note: {note}" for note in self.notes)]
        if self.hint:
            lines.append(f"hint: {self.hint}")
        return "\n".join(lines)


class _Lost(_CallError):
    """A worker that stopped, or was stopped, while a call ran in it."""


class _Worker:
    """A worker process, as the server sees it: a request goes to its standard input, and
    its answer comes back on its standard output."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.lock: Any = None  # an anyio.Lock, made in the event loop
        self._process: Process | None = None
        self._reader: BufferedByteReceiveStream | None = None

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.returncode is None

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    async def start(self) -> None:
        import anyio
        from anyio.streams.buffered import BufferedByteReceiveStream

        self.lock = anyio.Lock()
        configuration = {
            "max_output": self.settings.max_output,
            "allowed": [str(folder) for folder in self.settings.allowed],
            "memory_limit": self.settings.memory_limit,
        }
        code = "from relativisticpy.mcp import _work; _work()"
        self._process = await anyio.open_process(
            [sys.executable, "-c", code, json.dumps(configuration)],
            stdin=-1,
            stdout=-1,
            stderr=None,  # the server's own, which a client keeps as its log
            cwd=str(_quiet_folder()),
            env=_environment(),
            start_new_session=sys.platform != "win32",  # a terminal's Ctrl-C is the server's
        )
        assert self._process.stdout is not None
        self._reader = BufferedByteReceiveStream(self._process.stdout)
        try:
            with anyio.fail_after(STARTUP):
                ready = await self._receive()
        except Exception as error:
            await self.stop()
            raise _Lost(
                "RelativisticPy's worker process did not start",
                hint=f"run python -c 'import relativisticpy' to see why ({error!r})",
            ) from None
        if not ready.get("ready"):
            await self.stop()
            raise _Lost("RelativisticPy's worker process did not start")

    async def request(self, op: str, args: dict[str, Any], *, limit: float, stop: float) -> Any:
        """Ask the worker for something, and its answer: stopped after ``stop`` seconds,
        and interrupted if the call is cancelled, as when the client gives up on it."""
        import anyio

        async with self.lock:
            process = self._process
            if process is None or process.returncode is not None or process.stdin is None:
                raise _Lost("RelativisticPy's worker process has stopped")
            line = json.dumps({"op": op, "args": args, "limit": limit}).encode() + b"\n"
            try:
                await process.stdin.send(line)
                with anyio.fail_after(stop):
                    reply = await self._receive()
            except TimeoutError:
                await self.stop()
                raise _Lost(
                    f"the computation took longer than its time limit of {limit:g} s, and "
                    "did not stop when asked to, so it was stopped",
                    hint="give it longer with time_limit, up to the server's limit, or "
                    "compute it in steps, in a session, defining its parts first",
                ) from None
            except anyio.get_cancelled_exc_class():
                with anyio.CancelScope(shield=True):
                    await self._interrupt()
                raise
            except Exception:
                code = process.returncode
                await self.stop()
                raise _Lost(
                    "RelativisticPy's worker process stopped while computing this"
                    + (f" (exit status {code})" if code is not None else ""),
                    hint="it may have run out of memory: compute it in smaller steps",
                ) from None
        if "error" in reply:
            error = reply["error"]
            raise _CallError(str(error.get("rendered") or error.get("message")))
        if reply.get("interrupted"):
            raise _CallError("the computation was interrupted")
        return reply.get("result", {})

    async def _receive(self) -> dict[str, Any]:
        assert self._reader is not None
        line = await self._reader.receive_until(b"\n", 64 * 1024 * 1024)
        answer = json.loads(line)
        if not isinstance(answer, dict):
            raise ValueError("an answer is an object")
        return answer

    async def _interrupt(self) -> None:
        """Interrupt the computation in hand, as Ctrl-C would, and read the answer it gives,
        so that the next request gets its own; stop the worker if it does not answer."""
        import anyio

        process = self._process
        if process is not None and process.returncode is None and sys.platform != "win32":
            with contextlib.suppress(ProcessLookupError):
                process.send_signal(signal.SIGINT)
            with anyio.move_on_after(GRACE):
                with contextlib.suppress(Exception):
                    await self._receive()
                    return
        await self.stop()

    async def stop(self) -> None:
        """Stop the worker at once, and wait for it to end."""
        import anyio

        process, self._process = self._process, None
        if process is None:
            return
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        with anyio.CancelScope(shield=True), contextlib.suppress(Exception):
            await process.aclose()


def _quiet_folder() -> Path:
    """Where a worker starts: the folder RelativisticPy is installed in, so that no module
    of the server's working directory can stand in for one it imports."""
    import relativisticpy

    return Path(relativisticpy.__file__).resolve().parent.parent


def _environment() -> dict[str, str]:
    """A worker's environment: the server's, with Python's safe path, so that the working
    directory is not searched for modules, and one thread for linear algebra, which
    symbolic work does not need and which would take memory in every worker."""
    environment = dict(os.environ)
    environment["PYTHONSAFEPATH"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        environment.setdefault(name, "1")
    return environment


@dataclass
class _Session:
    """A named session: the worker that holds its workbook, and what it has done."""

    name: str
    worker: _Worker
    statements: int = 0
    defined: list[dict[str, Any]] = field(default_factory=list)
    used: float = field(default_factory=time.monotonic)
    calls: int = 0  # running now


class _Pool:
    """The workers of a server: one for each named session, and a few for calls without
    one, each of which computes in a new workbook."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.sessions: dict[str, _Session] = {}
        self.idle: list[_Worker] = []
        self.workers: set[_Worker] = set()
        self._slots: Any = None  # an anyio.Semaphore, made in the event loop
        self._opening: Any = None

    def _ready(self) -> None:
        import anyio

        if self._slots is None:
            self._slots = anyio.Semaphore(SCRATCH_WORKERS)
            self._opening = anyio.Lock()

    async def _new(self) -> _Worker:
        worker = _Worker(self.settings)
        self.workers.add(worker)
        try:
            await worker.start()
        except BaseException:  # cancelled as well as failed: no process is left behind
            await self._drop(worker)
            raise
        return worker

    async def _take(self) -> _Worker:
        while self.idle:
            worker = self.idle.pop()
            if worker.alive:
                return worker
            await self._drop(worker)
        return await self._new()

    async def _drop(self, worker: _Worker) -> None:
        self.workers.discard(worker)
        await worker.stop()

    async def warm(self) -> None:
        """Start a worker before it is asked for, so that the first call need not wait."""
        self._ready()
        if not self.idle:
            with contextlib.suppress(_CallError):
                self.idle.append(await self._new())

    @contextlib.asynccontextmanager
    async def scratch(self) -> AsyncIterator[_Worker]:
        """A worker for a call without a session, given back afterwards."""
        self._ready()
        async with self._slots:
            worker = await self._take()
            try:
                yield worker
            finally:
                if worker.alive and len(self.idle) < SCRATCH_WORKERS:
                    self.idle.append(worker)
                else:
                    await self._drop(worker)

    async def session(self, name: str) -> _Session:
        """The session called ``name``, started if it is new."""
        self._ready()
        if not _SESSION_NAME.fullmatch(name):
            raise _CallError(
                f"{name!r} cannot name a session",
                hint="a session's name is up to 64 letters, digits and the signs _ . : -, "
                "starting with a letter or digit, as in schwarzschild or paper-2",
            )
        async with self._opening:
            session = self.sessions.get(name)
            if session is not None:
                return session
            if len(self.sessions) >= self.settings.max_sessions:
                named = ", ".join(sorted(self.sessions)) or "none"
                raise _CallError(
                    f"this server keeps at most {self.settings.max_sessions} sessions at "
                    f"once, and has {len(self.sessions)}: {named}",
                    hint="reset_session one you no longer need, or compute without a session",
                )
            session = _Session(name, await self._take())
            self.sessions[name] = session
            return session

    async def forget(self, name: str) -> bool:
        """Forget a session and stop its worker; whether there was one."""
        session = self.sessions.pop(name, None)
        if session is None:
            return False
        await self._drop(session.worker)
        return True

    async def close(self) -> None:
        """Stop every worker."""
        for worker in list(self.workers):
            await self._drop(worker)
        self.idle.clear()
        self.sessions.clear()


# == the server ==============================================================================

_INSTRUCTIONS = """\
RelativisticPy reads mathematics as it is written, in LaTeX, plain text or Unicode, and \
computes it exactly, with SymPy: General Relativity (metrics, connections, curvature), \
index notation, calculus and algebra, and the equations of LaTeX papers, which it checks. \
Use it rather than doing long algebra by hand, and to check the algebra you did.

- compute runs statements and gives each one's values, in text, Unicode and LaTeX. Give \
it a session name to keep definitions from one call to the next.
- verify checks one claimed equality or step of a derivation, lhs = rhs or a chain \
a = b = c, after its context, such as a metric: holds, differs, relates, not decided or \
error, with the left side minus the right in LaTeX.
- check checks every display equation of a LaTeX or Markdown document.
- help gives the documentation; help with topic reference is the whole language on one \
page, and with the name of a function, as besselj, that function in full. The pages are \
also resources, relativisticpy://docs/{{topic}}.

The language in brief:
- := defines. = defines only what has no meaning yet; otherwise it is a claim, which is \
checked, or an equation.
- In LaTeX a run of letters is a product, as typeset: \\frac{{2GM}}{{r}} is 2 G M / r. \
Write \\mathrm{{name}} for a name of several letters.
- A script without braces takes one letter: write \\Gamma^{{r}}_{{tt}}, not \\Gamma^r_tt.
- A name right before a bracket calls it, as A(r); to multiply, leave a space.
- g_{{\\mu\\nu}} := Schwarzschild(M) defines a metric (also Kerr, ReissnerNordstrom, \
FLRW, DeSitter and others), and so does a line element, ds^2 = ...; then \
\\Gamma^{{a}}_{{bc}}, R^{{a}}_{{bcd}}, R_{{ab}}, R, G_{{ab}} and R_{{abcd}} R^{{abcd}} \
are computed.

Each call may run for {time_limit:g} s at most. {files}"""

_COMPUTE = """\
Run a RelativisticPy program and return what each statement gave.

A program is mathematics as written: LaTeX (\\frac{{1}}{{2}}, \\Gamma^{{r}}_{{tt}}), \
plain text (x**2, g_{{mu nu}}) or Unicode (θ, ∂), mixed freely, with statements on \
lines of their own or separated by ;. := defines; = defines only what has no meaning \
yet, and is otherwise a claim, which is checked, or an equation.

Returns one entry for each statement, in order, as relativisticpy --format json gives \
it: where it is, the statement, its values (each with its kind and its text, unicode \
and latex forms; a tensor with its components), what it defines, notes on how it was \
read, and its error, if it failed, with the error's type, message, place, notes and \
hint. A statement that fails does not stop the others.

Without session, the program runs in a new, empty workbook. With session, it runs in \
the workbook of that name, made on first use, so that definitions persist from one call \
to the next; reset_session forgets them.

A call may run for {time_limit:g} s at most; time_limit asks for less. A statement \
still running then fails, over the time limit, and the definitions before it are kept.

Examples: "g_{{\\mu\\nu}} := Schwarzschild(M)" then "R_{{abcd}} R^{{abcd}}" in one \
session; "\\int_0^\\infty e^{{-x^2}} \\, dx"; "f(x) := x^2 e^{{-x}}; f'(x)"."""

_VERIFY = """\
Check one claimed equality, or one step of a derivation, and say whether it holds.

claim is an equation, lhs = rhs, or a chain, a = b = c, each link of which is judged; \
context holds what the claim relies on, run first: definitions, a metric \
(g_{{\\mu\\nu}} := Schwarzschild(M), or a line element), coordinates, or symbols \
declared as variables (Reals := [x]).

The verdict, on the claim and on each link:
- holds: proven equal.
- differs: every quantity is known and the sides are not equal: a mistake. The left \
side minus the right is given, in LaTeX among its forms.
- relates: the sides are not equal, but involve quantities the context does not define, \
so the claim may be an equation to solve rather than an identity; the difference is \
given. Declare the symbols in the context (Reals := [x]) to have it judged as differs.
- not decided: not proven equal, though numbers do not show the sides differ; or a \
relation such as \\approx; or over the time limit.
- error: the claim could not be read or computed, or is not a claim; the detail, place, \
notes and hint say why.

Statements of the context that fail are listed in context. A call may run for \
{time_limit:g} s at most; time_limit asks for less."""

_CHECK = """\
Check the equations of a LaTeX document or Markdown notebook, as relativisticpy check \
does: run its display equations in order, as a reader would, and give each statement a \
verdict: holds, differs, relates, ill-formed, defines, computes, not computed, not \
understood or not parsed.

Give the document as text, in document, or as a path to a file. {files} A Markdown \
notebook (format markdown, or a .md file) runs its $$...$$ blocks.

Returns the report: the counts of each verdict, the reasons statements were not \
understood, warnings, and the findings, each with its verdict, place, statement, \
detail, hint and the difference of its sides. By default findings lists what fails \
--strict (differs, ill-formed, not parsed) and what ran over its time limit; all lists \
every statement. strict, or fail_on, gives a policy: failed says whether the document \
fails it. lines checks part of a document, as 120:340, after running what comes before \
it for its definitions.

Each statement may run for statement_time_limit seconds (default {statement:g}); the \
whole check for {time_limit:g} s at most, or time_limit. A check that runs out of time \
reports what it checked, and says where it stopped."""

_HELP = """\
The documentation of RelativisticPy, offline, as Markdown. Without a topic: what the \
language looks like, and the topics. topic reference is the whole language on one \
page: every statement, command, environment, constant, verdict and error, and every \
function by name. Other topics include getting-started, language, latex, tensors, \
notation, checking, python and troubleshooting. The name of a function, as besselj, \
gives that function in full: its forms, its arguments, its examples with their answers, \
and what computes it; latex() is the function where a topic has its name."""

_LIST_SESSIONS = """\
List the named sessions: for each, how many statements it has run, what it defines, \
whether a call is running in it, and how long since it was last used; and the limits of \
this server."""

_RESET_SESSION = """\
Forget a named session, with every definition in it, and stop its worker. A later \
compute with the same name starts it anew, empty."""

_FILES_ALLOWED = "check may read documents in {folders}."
_FILES_REFUSED = "check takes documents as text: this server reads no files."
_PATH_ALLOWED = "A path is read only in the folders this server may read: {folders}."
_PATH_REFUSED = "This server reads no files, so give the document as text."


def create_server(settings: Settings | None = None) -> "MCPServer[Any]":
    """The MCP server, with its tools, resources and prompts, bounded by ``settings``.
    ``server.run()`` serves it over standard input and output; in tests,
    ``mcp.Client(server)`` connects to it in memory."""
    return _build(settings or Settings())[0]


def _build(settings: Settings) -> "tuple[MCPServer[Any], _Pool]":
    """The server, and the pool of workers its tools compute in."""
    import anyio
    from mcp.server import MCPServer
    from mcp.server.mcpserver.exceptions import ResourceNotFoundError, ToolError
    from mcp.server.mcpserver.resources import FunctionResource
    from mcp.types import ToolAnnotations
    from pydantic import Field

    pool = _Pool(settings)
    folders = ", ".join(str(folder) for folder in settings.allowed)
    files = _FILES_ALLOWED.format(folders=folders) if folders else _FILES_REFUSED
    paths = _PATH_ALLOWED.format(folders=folders) if folders else _PATH_REFUSED

    @contextlib.asynccontextmanager
    async def lifespan(server: "MCPServer[Any]") -> AsyncIterator[dict[str, Any]]:
        async with anyio.create_task_group() as group:
            group.start_soon(pool.warm)
            try:
                yield {}
            finally:
                group.cancel_scope.cancel()
                with anyio.CancelScope(shield=True):
                    await pool.close()

    server = MCPServer(
        "relativisticpy",
        title="RelativisticPy",
        description="Computes mathematics written as in a paper, in LaTeX, plain text or "
        "Unicode, exactly: General Relativity, index notation, calculus and algebra; and "
        "checks the equations of papers.",
        instructions=_INSTRUCTIONS.format(time_limit=settings.time_limit, files=files),
        website_url=DOCS,
        version=__version__,
        lifespan=lifespan,
        log_level="WARNING",
    )

    def refuse(error: _CallError) -> ToolError:
        return ToolError(str(error))

    @server.tool(
        title="Compute mathematics",
        description=_COMPUTE.format(time_limit=settings.time_limit),
        annotations=ToolAnnotations(destructive_hint=False, open_world_hint=False),
    )
    async def compute(
        source: Annotated[
            str,
            Field(
                description="The program: statements in LaTeX, plain text or Unicode, on "
                "lines of their own or separated by ;"
            ),
        ],
        session: Annotated[
            str | None,
            Field(
                description="A session to run in, whose definitions persist from one call "
                "to the next, as schwarzschild; made on first use. Without it, the program "
                "runs in a new, empty workbook"
            ),
        ] = None,
        time_limit: Annotated[
            float | None,
            Field(gt=0, description="Seconds the call may run, up to the server's limit"),
        ] = None,
    ) -> Computed:
        _size_of(source, "the program")
        limit = settings.limit(time_limit)
        try:
            if session is None:
                async with pool.scratch() as worker:
                    reply = await worker.request(
                        "compute",
                        {"source": source, "session": False},
                        limit=limit,
                        stop=limit + GRACE,
                    )
                return cast(Computed, reply)
            record = await pool.session(session)
            record.calls += 1
            try:
                reply = await record.worker.request(
                    "compute", {"source": source, "session": True}, limit=limit, stop=limit + GRACE
                )
            except _Lost as lost:
                await pool.forget(session)
                stopped = f"the session {session} was stopped with it, and what it defined is lost"
                again = "run its definitions again" + (f"; {lost.hint}" if lost.hint else "")
                raise _CallError(lost.message, hint=again, notes=[*lost.notes, stopped]) from None
            finally:
                record.calls -= 1
                record.used = time.monotonic()
            record.defined = reply.pop("defined", [])
            record.statements += len(reply["statements"])
            reply["session"] = session
            return cast(Computed, reply)
        except _CallError as error:
            raise refuse(error) from None

    @server.tool(
        title="Verify a claim",
        description=_VERIFY.format(time_limit=settings.time_limit),
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    async def verify(
        claim: Annotated[
            str,
            Field(
                description="The claim: an equation, lhs = rhs, or a chain, a = b = c, as "
                "\\Gamma^{r}_{\\theta\\theta} = 2M - r"
            ),
        ],
        context: Annotated[
            str | None,
            Field(
                description="What the claim relies on, run first: definitions, a metric, "
                "coordinates, symbols declared as variables, as g_{\\mu\\nu} := "
                "Schwarzschild(M)"
            ),
        ] = None,
        time_limit: Annotated[
            float | None,
            Field(gt=0, description="Seconds the call may run, up to the server's limit"),
        ] = None,
    ) -> Verified:
        _size_of(claim, "the claim")
        _size_of(context or "", "the context")
        limit = settings.limit(time_limit)
        try:
            async with pool.scratch() as worker:
                reply = await worker.request(
                    "verify",
                    {"claim": claim, "context": context},
                    limit=limit,
                    stop=limit + GRACE,
                )
        except _CallError as error:
            raise refuse(error) from None
        return cast(Verified, reply)

    @server.tool(
        title="Check a document",
        description=_CHECK.format(
            files=paths, statement=STATEMENT_TIME_LIMIT, time_limit=settings.time_limit
        ),
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    async def check(
        document: Annotated[
            str | None,
            Field(description="The text of a LaTeX document, or of a Markdown notebook"),
        ] = None,
        path: Annotated[
            str | None,
            Field(
                description="The path of a document to read instead, in a folder this "
                "server may read; relative to the server's working directory or to one of "
                "those folders"
            ),
        ] = None,
        format: Annotated[
            Literal["latex", "markdown"] | None,
            Field(
                description="How to read the document: latex (the default for text) or "
                "markdown; a path goes by its suffix, .md being markdown"
            ),
        ] = None,
        name: Annotated[
            str,
            Field(description="The name findings in text are located by, as paper.tex"),
        ] = "<document>",
        lines: Annotated[
            str | None,
            Field(description="Only these lines: 120:340, 13, 10: or :10"),
        ] = None,
        strict: Annotated[
            bool,
            Field(
                description="Fail on what --strict fails on: differs, ill-formed, not "
                "parsed, a time limit, or no equations"
            ),
        ] = False,
        fail_on: Annotated[
            list[FailOn] | None,
            Field(description="Fail on these verdicts and conditions, as --fail-on"),
        ] = None,
        all: Annotated[
            bool,
            Field(description="List every statement's finding, not only those that fail"),
        ] = False,
        statement_time_limit: Annotated[
            float | None,
            Field(gt=0, description="Seconds each statement may run (default 30)"),
        ] = None,
        time_limit: Annotated[
            float | None,
            Field(gt=0, description="Seconds the whole check may run, up to the server's limit"),
        ] = None,
    ) -> Checked:
        if (document is None) == (path is None):
            raise ToolError("give the document as text, in document, or as a path: one of them")
        try:
            file = None if path is None else str(_readable(path, settings.allowed))
        except _CallError as error:
            raise refuse(error) from None
        _size_of(document or "", "the document")
        limit = settings.limit(time_limit)
        each = min(statement_time_limit or STATEMENT_TIME_LIMIT, limit)
        arguments = {
            "document": document,
            "path": file,
            "markdown": format == "markdown",
            "name": name,
            "lines": lines,
            "policy": list(fail_on or []),
            "strict": strict,
            "everything": all,
            "statement_time_limit": each,
        }
        try:
            async with pool.scratch() as worker:
                reply = await worker.request(
                    "check", arguments, limit=limit, stop=limit + each + GRACE
                )
        except _CallError as error:
            raise refuse(error) from None
        return cast(Checked, reply)

    @server.tool(
        name="help",
        title="Read the documentation",
        description=_HELP,
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
        structured_output=False,
    )
    def help_tool(
        topic: Annotated[
            str | None,
            Field(
                description="A topic, as reference, language, latex or tensors, or the name "
                "of a function, as besselj; none lists the topics"
            ),
        ] = None,
    ) -> str:
        if topic is None or not topic.strip():
            return _index()
        try:
            return documentation.page(topic)
        except RelativisticPyError as error:
            raise ToolError(_rendered(error)) from None

    @server.tool(
        title="List the sessions",
        description=_LIST_SESSIONS,
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    async def list_sessions() -> Sessions:
        now = time.monotonic()
        sessions = [
            SessionInfo(
                session=record.name,
                statements=record.statements,
                defined=record.defined,
                busy=record.calls > 0,
                idle_seconds=0.0 if record.calls else round(now - record.used, 1),
            )
            for record in pool.sessions.values()
        ]
        return Sessions(sessions=sessions, limits=_limits(settings))

    @server.tool(
        title="Reset a session",
        description=_RESET_SESSION,
        annotations=ToolAnnotations(
            destructive_hint=True, idempotent_hint=True, open_world_hint=False
        ),
    )
    async def reset_session(
        session: Annotated[str, Field(description="The session to forget")],
    ) -> Reset:
        forgotten = await pool.forget(session)
        notes = [] if forgotten else [f"there was no session {session}: nothing to forget"]
        return Reset(session=session, reset=forgotten, notes=notes)

    for topic, title in documentation.topics():
        server.add_resource(
            FunctionResource.from_function(
                _page_reader(topic),
                uri=f"relativisticpy://docs/{topic}",
                name=topic,
                title=title,
                description=_page_description(topic, title),
                mime_type="text/markdown",
            )
        )

    @server.resource(
        "relativisticpy://docs/{topic}",
        name="page",
        title="A page of the documentation",
        description="A page of RelativisticPy's documentation, by its topic or another "
        "name for it, or a function of the language in full, by its name, as Markdown",
        mime_type="text/markdown",
    )
    def page(topic: str) -> str:
        try:
            return documentation.page(topic)
        except RelativisticPyError as error:
            raise ResourceNotFoundError(_rendered(error)) from None

    @server.prompt(title="Verify a derivation")
    def verify_derivation(
        derivation: Annotated[
            str, Field(description="The derivation: its steps, in LaTeX or plain text")
        ],
        context: Annotated[
            str,
            Field(description="What it relies on: definitions, a metric, coordinates"),
        ] = "",
    ) -> str:
        """Check a derivation step by step with RelativisticPy's verify tool."""
        given = f"\n\nIts context, to give verify as context:\n\n{context}" if context else ""
        return (
            "Check this derivation with RelativisticPy, step by step, rather than by hand.\n\n"
            "For each step that claims two expressions are equal, call the verify tool with "
            "the step as claim (lhs = rhs, or a chain a = b = c) and what the step relies "
            "on as context: definitions, a metric, coordinates, and symbols declared as "
            "variables with Reals := [x]. Report each step's verdict:\n"
            "- holds: say so;\n"
            "- differs: give the difference of the sides it shows, in LaTeX, say what the "
            "step should have been, and find it with compute if it helps;\n"
            "- relates: the step is not an identity unless its symbols are declared; "
            "declare them and verify again;\n"
            "- not decided, or error: give the detail and the hint, fix the notation, and "
            "verify again.\n"
            "If you are unsure how to write something, read the help tool's reference "
            f"topic.\n\nThe derivation:\n\n{derivation}{given}"
        )

    @server.prompt(title="Check a paper")
    def check_paper(
        document: Annotated[
            str,
            Field(description="The document: the path of a file the server may read, or its LaTeX"),
        ],
    ) -> str:
        """Check the equations of a LaTeX paper with RelativisticPy's check tool, and
        explain what does not hold."""
        return (
            "Check the equations of this document with RelativisticPy's check tool: pass "
            "it as path if it is the path of a file, or as document if it is LaTeX. Then "
            "summarise the counts of each verdict, and for each finding that differs, is "
            "ill-formed or is not parsed, say where it is, what is wrong, using the "
            "difference of its sides and its hint, and how to fix it. For a finding that "
            "relates quantities nothing defines, say what would have to be defined for it "
            "to be checked. Verify a corrected equation with the verify tool before you "
            f"propose it.\n\nThe document:\n\n{document}"
        )

    return server, pool


def _size_of(text: str, what: str) -> None:
    """Refuse input too long to be a program, a claim or a document."""
    from mcp.server.mcpserver.exceptions import ToolError

    if len(text) > MAX_INPUT:
        raise ToolError(
            f"{what} has {len(text):,} characters, more than the {MAX_INPUT:,} this server takes"
        )


def _readable(text: str, allowed: Sequence[Path]) -> Path:
    """The file a path names, if the server may read it: in one of the folders it was
    started with, after symbolic links and .. are followed."""
    if not allowed:
        raise _CallError(
            "this server reads no files: it was started without --allow-files",
            hint="give the document's text, in document; or start the server with "
            "--allow-files FOLDER",
        )
    given = Path(text).expanduser()
    candidates = [given] if given.is_absolute() else [Path.cwd() / given]
    if not given.is_absolute():
        candidates += [folder / given for folder in allowed]
    folders = ", ".join(str(folder) for folder in allowed)
    inside = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if any(_inside(resolved, folder) for folder in allowed):
            if resolved.is_file():
                return resolved
            inside.append(resolved)
    if not inside:
        raise _CallError(
            f"{text} is not in a folder this server may read",
            hint=f"the folders it may read are {folders}",
        )
    if inside[0].is_dir():
        raise _CallError(f"{text} is a folder, not a file", hint="give the path of a document")
    raise _CallError(f"no file {text}", hint=f"the folders this server may read are {folders}")


def _limits(settings: Settings) -> dict[str, Any]:
    return {
        "time_limit": settings.time_limit,
        "max_output": settings.max_output,
        "max_sessions": settings.max_sessions,
        "memory_limit_mib": settings.memory_limit,
        "files": [str(folder) for folder in settings.allowed],
    }


def _rendered(error: RelativisticPyError) -> str:
    """An error as text: its message, then its notes and hint."""
    lines = [error.message, *(f"note: {note}" for note in error.notes)]
    if error.hint:
        lines.append(f"hint: {error.hint}")
    return "\n".join(lines)


def _page_reader(topic: str) -> Callable[[], str]:
    def read() -> str:
        return documentation.page(topic)

    read.__name__ = topic.replace("-", "_")
    return read


def _page_description(topic: str, title: str) -> str:
    if topic == "reference":
        return (
            "The whole language on one page: every statement, command, environment, "
            "constant, verdict and error, and every function by name"
        )
    return f"{title}: a page of RelativisticPy's documentation"


def _index() -> str:
    """What the help tool says without a topic: what the language looks like, and the
    topics of the documentation."""
    from relativisticpy.cli import CHEAT_SHEET

    lines = [
        "# RelativisticPy",
        "",
        "RelativisticPy reads mathematics as written, in LaTeX, plain text or Unicode, and "
        "computes it:",
        "",
        "```",
    ]
    for code, meaning in CHEAT_SHEET:
        lines.append(f"{code:<36}  {meaning}" if len(code) <= 36 else f"{code}\n{'':38}{meaning}")
    lines += [
        "```",
        "",
        "Topics, for help with a topic, or the resource relativisticpy://docs/TOPIC; the "
        "topic reference is the whole language on one page:",
        "",
    ]
    lines += [f"- {topic}: {title}" for topic, title in documentation.topics()]
    lines += [
        "",
        "help with the name of a function, as besselj, or the resource "
        "relativisticpy://docs/besselj, gives that function in full: its forms, its arguments, "
        "its examples with their answers, and what computes it.",
        "",
        f"Online: {DOCS}",
    ]
    return "\n".join(lines)


# == the command =============================================================================

_DESCRIPTION = """\
Serve RelativisticPy to AI agents over the Model Context Protocol (MCP), on standard
input and output, as tools: compute, verify, check, help, list_sessions and
reset_session; with the documentation as resources, and prompts.

An MCP client starts this command and talks to it; run by hand, it waits for one.
Each call runs in a worker process that is stopped when the call runs over its time
limit; workbooks are given no Python scope, and documents are read from files only in
the folders given with --allow-files."""

_EPILOG = f"""\
Connect a client:
  claude mcp add relativisticpy -- relativisticpy mcp
  Claude Desktop, in claude_desktop_config.json:
    {{"mcpServers": {{"relativisticpy": {{"command": "relativisticpy", "args": ["mcp"]}}}}}}

Examples:
  relativisticpy mcp
  relativisticpy mcp --allow-files ~/papers --time-limit 300

It needs the mcp extra: pip install "relativisticpy[mcp]".

Learn more: relativisticpy help mcp, or {DOCS}/mcp/"""


def parser() -> argparse.ArgumentParser:
    """The command line of ``relativisticpy mcp``."""
    arguments = argparse.ArgumentParser(
        prog="relativisticpy mcp",
        usage="relativisticpy mcp [options]",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arguments.add_argument(
        "--allow-files",
        nargs="?",
        const=".",
        action="append",
        metavar="FOLDER",
        help="let check read documents in FOLDER, and the files they include there; "
        "without FOLDER, the working directory. Give it again for more folders. Without "
        "it, check takes documents as text only",
    )
    arguments.add_argument(
        "--time-limit",
        type=_positive(float, "seconds, as in --time-limit 120"),
        default=TIME_LIMIT,
        metavar="SECONDS",
        help=f"the longest any call may run; a call may ask for less (default: {TIME_LIMIT:g})",
    )
    arguments.add_argument(
        "--max-output",
        type=_positive(int, "characters, as in --max-output 100000"),
        default=MAX_OUTPUT,
        metavar="CHARACTERS",
        help="the most characters a result may have, as JSON; a longer one is cut, with a "
        f"note (default: {MAX_OUTPUT})",
    )
    arguments.add_argument(
        "--max-sessions",
        type=_positive(int, "sessions, as in --max-sessions 4", zero=True),
        default=MAX_SESSIONS,
        metavar="N",
        help=f"the most named sessions at once (default: {MAX_SESSIONS})",
    )
    arguments.add_argument(
        "--memory-limit",
        type=_positive(int, "MiB, as in --memory-limit 8192, or 0 for none", zero=True),
        default=MEMORY_LIMIT,
        metavar="MIB",
        help="the memory each worker process may use, where the system enforces it, as "
        f"Linux does; 0 for no limit (default: {MEMORY_LIMIT})",
    )
    return arguments


def _positive(kind: Callable[[str], Any], what: str, *, zero: bool = False) -> Callable[[str], Any]:
    def read(text: str) -> Any:
        try:
            value = kind(text)
        except ValueError:
            value = math.nan
        if not (value >= 0 if zero else value > 0) or not math.isfinite(value):
            raise argparse.ArgumentTypeError(f"takes a positive number of {what}, not {text}")
        return value

    return read


def settings_from(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> Settings:
    """The settings a command line asks for; a folder that is not one is refused."""
    folders = []
    for given in arguments.allow_files or []:
        folder = Path(given).expanduser().resolve()
        if not folder.is_dir():
            parser.error(f"--allow-files takes a folder, and {given} is not one")
        if folder not in folders:
            folders.append(folder)
    try:
        return Settings(
            allowed=tuple(folders),
            time_limit=arguments.time_limit,
            max_output=arguments.max_output,
            max_sessions=arguments.max_sessions,
            memory_limit=arguments.memory_limit or None,
        )
    except UsageError as error:
        parser.error(error.message)


def main(argv: Sequence[str] | None = None) -> int:
    """``relativisticpy mcp``: serve RelativisticPy over MCP on standard input and output
    until the client closes them; the exit status."""
    command = parser()
    arguments = command.parse_args(list(sys.argv[1:] if argv is None else argv))
    settings = settings_from(arguments, command)
    import importlib.util

    if importlib.util.find_spec("mcp") is None:
        print(
            "relativisticpy mcp: error: the MCP server needs the mcp package: pip install "
            '"relativisticpy[mcp]"',
            file=sys.stderr,
        )
        return 1
    server = create_server(settings)
    terminal = False
    with contextlib.suppress(ValueError, OSError, AttributeError):  # closed, or no stream
        terminal = sys.stdin.isatty()
    if terminal:
        print(
            "relativisticpy mcp: serving RelativisticPy over MCP on standard input and "
            "output, for an MCP client to start; Ctrl-C stops it (relativisticpy help mcp)",
            file=sys.stderr,
        )
    try:
        server.run("stdio")
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
