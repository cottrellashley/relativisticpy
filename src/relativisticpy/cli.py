r"""The ``relativisticpy`` command: run programs and notebooks, check papers, read the
documentation offline, or compute at a prompt.

::

    relativisticpy schwarzschild.relpy      run a program, printing each result as it comes
    relativisticpy notes.md                 run the blocks of maths of a Markdown notebook
    relativisticpy notes.md --render -o out.md   the notebook with its results typeset
    relativisticpy -c "R_{abcd} R^{abcd}"   run code given on the command line
    relativisticpy -                        run standard input as a program
    relativisticpy check paper.tex          check the equations of a LaTeX document
    relativisticpy help [TOPIC]             print the documentation, offline
    relativisticpy mcp                      serve RelativisticPy to AI agents, over MCP
    relativisticpy                          start the prompt, in a terminal

Each result is printed as soon as it is computed, labelled with the statement it came
from. A run stops at the first error, with every result before it printed, unless it is
given ``--keep-going``. ``--format json`` prints one JSON object per statement.

The exit status is 0 when everything succeeded; 1 when a statement failed, or a check
failed by its policy (``--strict``, ``--fail-on``); 2 when the command line was wrong,
as an unknown option or a file that cannot be read; 130 after Ctrl-C; and 143 when
the command was stopped with SIGTERM.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import glob
import io
import json
import math
import multiprocessing
import os
import re
import select
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn, TextIO

from relativisticpy import __version__, checking
from relativisticpy import display as shown
from relativisticpy import help as documentation
from relativisticpy.checking import Report, Verdict
from relativisticpy.documents import Document, Equation, read_text
from relativisticpy.errors import (
    FileError,
    InternalError,
    ParseError,
    RelativisticPyError,
    UsageError,
    debugging,
    internal_error,
    place,
)
from relativisticpy.language import Source, Span, parse
from relativisticpy.language import syntax as ast
from relativisticpy.workbook import (
    Step,
    Workbook,
    _steps_of,
    notebook_blocks,
    read_program,
    render_notebook,
)

# The exit statuses, as the shell sees them.
OK, FAILED, USAGE, INTERRUPTED, TERMINATED = 0, 1, 2, 130, 143

#: Where the documentation is online.
DOCS = "https://relativisticpy.org"

#: The forms results are printed in.
FORMS = ("unicode", "text", "latex", "json")
#: The forms a report of a check is printed in.
CHECK_FORMS = ("unicode", "text", "json", "github")

#: The prompt, and the prompt for a line that goes on with the input above it.
PROMPT, CONTINUATION = "rp> ", "... "

#: How long a statement or a check runs before a progress line appears, in seconds.
PROGRESS_AFTER = 1.0

_MARKDOWN = (".md", ".markdown")
_COMMANDS = ("check", "help", "mcp")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command with ``argv`` (by default ``sys.argv[1:]``) and return its exit
    status. A command line that is wrong raises ``SystemExit(2)``, as argparse does."""
    for stream in (sys.stdout, sys.stderr):
        _replace_what_cannot_print(stream)
    arguments = list(sys.argv[1:] if argv is None else argv)
    with _sigterm():
        try:
            return _main(arguments)
        except BrokenPipeError:  # the reader stopped reading, as `| head` does
            _stop_workers()
            with contextlib.suppress(OSError, ValueError):
                os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
            return FAILED
        except KeyboardInterrupt:  # Ctrl-C outside a run: stop, as the shell expects
            _stop_workers()
            _flush(sys.stdout)
            print(file=sys.stderr)
            return INTERRUPTED
        except _Terminated:
            _stop_workers()
            _flush(sys.stdout)
            print("relativisticpy: terminated", file=sys.stderr)
            return TERMINATED
        except Exception as error:  # a failure of the command's own: said to be one
            _stop_workers()
            failure = internal_error(error, during="running the command")
            _Console(sys.stdout, sys.stderr).diagnostic(failure)
            return FAILED


def _main(arguments: list[str]) -> int:
    command, rest = _command(arguments)
    if command == "check":
        return check(rest)
    if command == "help":
        return help_command(rest)
    if command == "mcp":
        return mcp_command(rest)
    return run(rest)


#: The options that take a value, of every command: what follows them is never a command.
_TAKES_VALUE = frozenset(
    {"-c", "--code", "--format", "-j", "--workers", "-o", "--output", "--color"}
    | {"--lines", "--time-limit", "--fail-on"}
    | {"--max-output", "--max-sessions", "--memory-limit"}
)


def _command(arguments: list[str]) -> tuple[str, list[str]]:
    """The command a command line names, ``check`` or ``help``, wherever it is among the
    options (``relativisticpy -j 2 check paper.tex``), and the rest; ``run`` otherwise.
    A file named check is written ./check."""
    position = 0
    while position < len(arguments):
        argument = arguments[position]
        if argument == "--":
            break
        if argument in _TAKES_VALUE:
            position += 2
        elif argument.startswith("-") and argument != "-":
            position += 1
        else:
            if argument in _COMMANDS:
                return argument, arguments[:position] + arguments[position + 1 :]
            break
    return "run", arguments


# -- running programs ------------------------------------------------------------------


def run(arguments: Sequence[str]) -> int:
    """``relativisticpy [FILE | -c CODE | -]``: run a program or a notebook, render a
    notebook, print the IR, or start the prompt."""
    parser = _run_parser()
    args = parser.parse_args(_code_verbatim(list(arguments)))
    plan = _plan(parser, args)
    console = _Console(sys.stdout, sys.stderr, args.color)
    workbook = Workbook(workers=args.workers)
    try:
        status = OK
        if plan.mode == "render":
            status = _render_command(workbook, plan, console)
        elif plan.mode == "ir":
            status = _ir_command(workbook, plan, console)
        elif plan.mode == "run":
            printer = _Printer(plan.form, console, _width(sys.stdout))
            runner = _Runner(workbook, printer, console, keep_going=plan.keep_going)
            status = runner.run(plan.sources)
        if status == USAGE or not plan.interactive:
            return status
        return interact(
            workbook,
            form=plan.form,
            banner=_is_terminal(sys.stdin) and plan.mode == "prompt",
            color=args.color,
        )
    except _Stopped as stopped:
        return stopped.status
    except (KeyboardInterrupt, _Terminated):
        _stop_workers(workbook)
        raise
    finally:
        workbook.close()


@dataclass
class _Plan:
    """What a command line asks a run to do."""

    mode: str  # run, render, ir or prompt
    sources: list[Source] = field(default_factory=list)  # what runs, in order
    name: str = ""  # the file, <stdin> or <command line>
    text: str = ""  # the document, for --render
    form: str = "unicode"
    interactive: bool = False
    keep_going: bool = False
    output: str | None = None


def _plan(parser: argparse.ArgumentParser, args: argparse.Namespace) -> _Plan:
    """What the arguments ask for, once checked: flags that do not go together, and files
    that cannot be read, are refused as usage errors."""
    if args.latex and args.format not in (None, "latex"):
        parser.error(
            f"--latex is short for --format latex, so it cannot go with --format {args.format}"
        )
    form = "latex" if args.latex else args.format or _default_form(sys.stdout)
    files: list[str] = args.files
    if len(files) > 1:
        parser.error(_several_files(files))
    if files and args.code is not None:
        parser.error(f"give a FILE or -c CODE, not both (got {files[0]} and -c)")
    piped = not files and args.code is None and not _is_terminal(sys.stdin)
    stdin = files == ["-"] or (piped and not args.interactive)
    mode = "render" if args.render else "ir" if args.ir else "run"
    if not files and args.code is None and not stdin:
        if args.render or args.ir:
            flag = "--render" if args.render else "--ir"
            parser.error(f"{flag} needs a program: give a FILE, -c CODE, or - for standard input")
        mode = "prompt"
    _refuse_combinations(parser, args, mode, form)
    plan = _Plan(mode, form=form, interactive=args.interactive or mode == "prompt")
    plan.keep_going = args.keep_going
    plan.output = args.output
    if mode == "prompt":
        return plan
    if args.code is not None:
        plan.name = "<command line>"
        plan.text = "\n".join(args.code)
        if args.render:
            parser.error("--render needs a Markdown notebook (.md), not -c CODE")
        plan.sources = [Source(plan.text, plan.name)]
        return plan
    if stdin:
        plan.name = "<stdin>"
        try:
            plan.text = _stdin_text()
        except FileError as error:
            _file_error(parser, error, "-", args.color)
        if args.render:
            return plan
        plan.sources = [Source(plan.text, plan.name)]
        return plan
    path = Path(files[0])
    plan.name = str(path)
    if args.render:
        if path.suffix.lower() not in _MARKDOWN:
            parser.error(f"--render needs a Markdown notebook (.md), and {path} is not one")
        try:
            plan.text = read_text(path)
        except FileError as error:
            _file_error(parser, error, path, args.color)
        return plan
    try:
        plan.sources = read_program(path)
    except FileError as error:
        _file_error(parser, error, path, args.color)
    return plan


def _refuse_combinations(
    parser: argparse.ArgumentParser, args: argparse.Namespace, mode: str, form: str
) -> None:
    """Refuse flags that do nothing where they are given, or contradict each other."""
    chosen = args.latex or args.format is not None
    if mode == "render":
        if args.ir:
            parser.error("--render and --ir do not go together: give one")
        if chosen and form != "latex":
            parser.error(f"--render typesets results in LaTeX, so --format {form} does nothing")
        if args.interactive:
            parser.error("--render writes a document, and -i starts the prompt: give one")
        if args.keep_going:
            parser.error("--render always goes on past errors, typesetting each: drop -k")
    elif args.output is not None:
        parser.error("-o FILE writes the document of --render; to save results, redirect them")
    if mode == "ir":
        if chosen:
            parser.error("--ir prints the IR, which has one form: drop --format and --latex")
        if args.interactive or args.keep_going:
            parser.error(
                f"--ir only compiles, so {'-i' if args.interactive else '-k'} does nothing"
            )
    if mode == "prompt" and args.keep_going:
        parser.error("--keep-going is for running a program; at the prompt, errors never stop")
    if args.interactive and form == "json":
        parser.error("--format json is for running a program, not for the prompt (-i)")


def _several_files(files: list[str]) -> str:
    """What to say when a run is given several files: a mistyped command, or a run of
    one file at a time."""
    first = files[0]
    close = difflib.get_close_matches(first, _COMMANDS, n=1, cutoff=0.6)
    if close and not Path(first).exists():
        return f"unknown command {first}; did you mean {close[0]}? (relativisticpy {close[0]} ...)"
    shown_files = ", ".join(files[:3]) + (", ..." if len(files) > 3 else "")
    return (
        f"a run takes one FILE, but {len(files)} were given ({shown_files}); "
        "to check several documents: relativisticpy check FILE..."
    )


def _file_error(
    parser: argparse.ArgumentParser, error: FileError, path: Path | str, color: str
) -> NoReturn:
    """Refuse a file that cannot be run, with what to do instead: exit status 2, as for a
    command line that is wrong, which it is."""
    hint = error.hint
    path = Path(path)
    if not path.exists() and not path.suffix:
        close = difflib.get_close_matches(str(path), _COMMANDS, n=1, cutoff=0.6)
        if close:
            hint = f"to run the command {close[0]}: relativisticpy {close[0]} ..."
    if path.is_dir():
        hint = f"to check the documents in it: relativisticpy check {path}"
    text = f"relativisticpy: error: {error.message}" + (f"\nhint: {hint}" if hint else "")
    parser.exit(USAGE, _Style(sys.stderr, color).diagnostic(text) + "\n")


def _stdin_text() -> str:
    """Standard input, as UTF-8 text, to its end; a
    :class:`~relativisticpy.errors.FileError` if it is not UTF-8. If nothing comes for a
    while, as when a harness leaves it open, say what is awaited, rather than wait in
    silence."""
    buffer = getattr(sys.stdin, "buffer", None)
    if buffer is None:  # a stream of text already, as in a test
        return str(sys.stdin.read())
    if not _ready(sys.stdin, STDIN_SILENCE, otherwise=True):
        sys.stderr.write(
            "relativisticpy: reading from standard input, to its end; to run a file or code "
            "instead: relativisticpy FILE, or -c CODE\n"
        )
        sys.stderr.flush()
    data = buffer.read()
    try:
        return str(data.decode("utf-8-sig"))
    except UnicodeDecodeError as error:
        raise FileError(
            f"standard input is not UTF-8 text: the byte 0x{data[error.start]:02X} at "
            f"offset {error.start} is not"
        ) from None


#: How long standard input may be silent, in seconds, before a run says it is waiting.
STDIN_SILENCE = 2.0


def _ready(stream: TextIO, seconds: float, *, otherwise: bool) -> bool:
    """Whether a stream has something to read, or its end, within ``seconds``;
    ``otherwise`` where that cannot be told, as of a pipe on Windows."""
    try:
        ready, _, _ = select.select([stream], [], [], seconds)
    except (OSError, ValueError, TypeError, AttributeError):
        return otherwise
    return bool(ready)


def _code_verbatim(arguments: list[str]) -> list[str]:
    """``-c`` takes what follows it as code, whatever it is, as Python's ``-c`` does:
    ``-c '-x'`` runs -x."""
    found = []
    position = 0
    while position < len(arguments):
        argument = arguments[position]
        if argument == "--":
            found.extend(arguments[position:])
            break
        if argument in ("-c", "--code") and position + 1 < len(arguments):
            found.append(f"--code={arguments[position + 1]}")
            position += 2
            continue
        found.append(argument)
        position += 1
    return found


class _Runner:
    """Runs programs statement by statement, as ``relativisticpy FILE`` does: each result
    printed as soon as it is computed, each error reported in order, and a stop at the
    first error unless it keeps going."""

    def __init__(
        self, workbook: Workbook, printer: _Printer, console: _Console, *, keep_going: bool
    ) -> None:
        self.workbook = workbook
        self.printer = printer
        self.console = console
        self.keep_going = keep_going

    def run(self, sources: Sequence[Source]) -> int:
        """Run the sources in order, as one program; the exit status."""
        planned = [(source, step) for source in sources for step in _planned(source)]
        self.printer.labelled = len(planned) > 1
        failures: list[Span] = []
        progress = _Progress(self.console, "computing")
        started = time.monotonic()  # when the statement being computed started
        current: Span | None = None
        running: Iterator[Step] | None = None
        source: Source | None = None
        try:
            for position, (block, (_, span)) in enumerate(planned):
                if block is not source:  # the next block of a notebook
                    source, running = block, self.workbook.each(block, recover=True)
                assert running is not None
                current, started = span, time.monotonic()
                before = _snapshot(self.workbook)
                progress.begin(f"{_where(span, short=True)} {_shortened(span.text, 50)}")
                try:
                    step = next(running, None)
                finally:
                    progress.end()
                current = None
                if step is None:  # never so: each gives the steps it was planned to
                    continue
                step = self._show(step, before)
                if step.error is None:
                    continue
                failures.append(step.span)
                if not self.keep_going:
                    self._stop(step, [item for _, item in planned[position + 1 :]])
                    return FAILED
        except KeyboardInterrupt:
            self._interrupted("interrupted", current, started)
            raise _Stopped(INTERRUPTED) from None
        except _Terminated:
            self._interrupted("terminated", current, started)
            raise _Stopped(TERMINATED) from None
        if failures:
            self.console.note(_failures_summary(failures, len(planned)))
            return FAILED
        return OK

    def _show(self, step: Step, before: _Snapshot) -> Step:
        """Print one step: its values and what it defines, or its error; the step, with
        the internal error that printing it raised, if it did."""
        try:
            defined = _defined(before, _snapshot(self.workbook)) if step.error is None else []
            self.printer.step(step, defined)
        except (KeyboardInterrupt, _Terminated, BrokenPipeError):
            raise
        except Exception as error:  # a failure of our own, while printing: reported
            failure = internal_error(error, span=step.span, during="printing this result")
            step = Step(step.span, step.statement, error=failure)
            self.printer.step(step, [])
        return step

    def _stop(self, step: Step, later: list[tuple[ast.Statement | ParseError, Span]]) -> None:
        """After the first error: the errors of the statements after it that cannot be
        read, which are known without running anything, and what did not run."""
        unread = [(item, span) for item, span in later if isinstance(item, ParseError)]
        for error, span in unread:
            self.printer.step(Step(span, ast.Statement(span=span), error=error), [])
        if not later:
            return
        if len(later) == 1:
            rest = "the statement after it did not run" + (
                ", and cannot be read, as said above" if unread else ""
            )
        else:
            rest = f"the {len(later)} statements after it did not run" + (
                f", and {len(unread)} of them cannot be read, as said above" if unread else ""
            )
        self.console.note(
            f"stopped at the first error, at {_where(step.span)}: {rest}; "
            f"--keep-going (-k) runs {'it' if len(later) == 1 else 'them'}"
        )

    def _interrupted(self, what: str, current: Span | None, started: float) -> None:
        """Say where a run stopped, once its workers are stopped."""
        _stop_workers(self.workbook)
        if current is not None:
            seconds = _duration(time.monotonic() - started)
            where = f"after {seconds} computing {_where(current)}, {_shortened(current.text, 40)}"
        else:
            where = "between two statements"
        self.console.note(f"{what} {where}; the results above are complete")


class _Stopped(Exception):
    """A run stopped by Ctrl-C or SIGTERM, after saying where: the exit status it ends
    with."""

    def __init__(self, status: int) -> None:
        super().__init__(status)
        self.status = status


def _planned(source: Source) -> list[tuple[ast.Statement | ParseError, Span]]:
    """The steps ``each(source, recover=True)`` gives, in order, before any runs: each
    statement, each link of a chain, and each statement that cannot be read."""
    return _steps_of(parse(source, recover=True))


def _failures_summary(failures: list[Span], total: int) -> str:
    places = ", ".join(_where(span) for span in failures[:5])
    more = f" and {len(failures) - 5} more" if len(failures) > 5 else ""
    return f"{len(failures)} of {_plural(total, 'statement')} failed: {places}{more}"


# -- what a statement defined --------------------------------------------------------


@dataclass(frozen=True)
class _Snapshot:
    """What a workbook defines, to tell what a statement defined."""

    coordinates: tuple[str, ...] | None
    names: dict[str, tuple[str, str, Span | None]]


@dataclass(frozen=True)
class _Defined:
    """Something a statement defined: its name, its kind and how it is written."""

    name: str
    kind: str
    written: str
    again: bool  # defined before, and defined anew


def _snapshot(workbook: Workbook) -> _Snapshot:
    """What a workbook defines now; nothing, if that cannot be told, which only says less
    of what a statement defined."""
    try:
        coordinates = workbook.coordinates
        names = {
            name: (definition.kind, definition.written, definition.span)
            for name, definition in workbook.names().items()
            if definition.kind != "coordinate"
        }
    except Exception:  # what a statement defined is said, never a reason to stop
        return _Snapshot(None, {})
    return _Snapshot(tuple(str(c) for c in coordinates) if coordinates else None, names)


def _defined(before: _Snapshot, after: _Snapshot) -> list[_Defined]:
    """What changed from one snapshot to the next: the coordinates, if they did, and each
    name defined or defined anew, in the order the workbook lists them."""
    found: list[_Defined] = []
    if after.coordinates and after.coordinates != before.coordinates:
        found.extend(
            _Defined(name, "coordinate", _coordinate(name), bool(before.coordinates))
            for name in after.coordinates
        )
    for name, (kind, written, span) in after.names.items():
        old = before.names.get(name)
        if old is None or old[:2] != (kind, written) or old[2] != span:
            found.append(_Defined(name, kind, written, old is not None))
    return found


def _coordinate(name: str) -> str:
    from relativisticpy.tensors.indices import latex_label

    return str(latex_label(name))


def _defines_text(defined: list[_Defined], form: str) -> str:
    """``defines x``, ``defines the metric g_{μν} and the coordinates t, r, θ, φ``."""
    coordinates = [_written(d.written, form) for d in defined if d.kind == "coordinate"]
    phrases = []
    for definition in defined:
        written = _written(definition.written, form)
        if definition.kind == "metric":
            phrases.append(f"the metric {written}")
        elif definition.kind == "space":
            phrases.append(f"the space {written}")
        elif definition.kind != "coordinate":
            phrases.append(written)
    if coordinates:
        phrases.append("the coordinates " + ", ".join(coordinates))
    verb = "redefines" if all(d.again for d in defined) else "defines"
    return f"{verb} {_listed(phrases)}"


def _written(written: str, form: str) -> str:
    """A name as written, ``g_{\\mu\\nu}``, in a form: ``g_{μν}`` in Unicode."""
    if form != "unicode":
        return written
    try:
        return str(shown._source(written, [], "unicode"))
    except Exception:  # how a name is shown never stops a run
        return written


# -- printing results ------------------------------------------------------------------


class _Printer:
    """Prints the steps of a run in a form: each labelled with its statement, unless
    the program has one statement only, whose result is printed alone."""

    def __init__(self, form: str, console: _Console, width: int | None) -> None:
        self.form = form
        self.console = console
        self.width = width
        self.labelled = True

    def step(self, step: Step, defined: list[_Defined]) -> None:
        """Print what came of a statement: a line of JSON, or its label, its values, what
        it defined and its notes on standard output, or its error on standard error."""
        if self.form == "json":
            self.console.out(_json(_record(step, defined), self.console.stdout))
            return
        if step.error is not None:
            self.error(step.error)
            return
        lines = []
        indent = "  " if self.labelled and self.form != "latex" else ""
        if self.labelled:
            lines.append(self._label(step.span))
        for value in step.values or []:
            text = _shown(value, self.form)
            lines.extend(indent + line for line in text.split("\n"))
        if defined:
            lines.append(self._comment(indent + _defines_text(defined, self.form)))
        for note in step.notes:
            lines.append(self._comment(f"{indent}note: {note}"))
        if lines:
            self.console.out("\n".join(lines))

    def error(self, error: RelativisticPyError) -> None:
        self.console.diagnostic(error)

    def _label(self, span: Span) -> str:
        """``plain.relpy:2:1: x^2``: where a statement is, and the statement."""
        label = f"{_where(span)}: {' '.join(span.text.split())}"
        if self.form == "latex":
            return "% " + label
        if self.width is not None and len(label) > self.width:
            label = label[: max(1, self.width - 1)] + "…"
        return self.console.style.dim(label)

    def _comment(self, text: str) -> str:
        if self.form == "latex":
            return "% " + text.strip()
        return self.console.style.dim(text)


def _shown(value: Any, form: str) -> str:
    """A value in a form, as :mod:`relativisticpy.display` shows it."""
    if form == "latex":
        return shown.latex(value)
    if form == "text":
        return shown.text(value)
    return shown.unicode(value)


def display(value: Any, *, latex_output: bool = False) -> str:
    """How a result prints, as :mod:`relativisticpy.display` shows it: mathematics for a
    terminal, or LaTeX on request."""
    return shown.latex(value) if latex_output else shown.unicode(value)


def _record(step: Step, defined: list[_Defined]) -> dict[str, Any]:
    """A step as data: where its statement is, what it is, and what came of it, with
    each value as :func:`relativisticpy.display.data` describes it."""
    where = place(step.span)
    return {
        "file": where["file"],
        "line": where["line"],
        "column": where["column"],
        "end_line": where["end_line"],
        "end_column": where["end_column"],
        "statement": where["text"],
        "values": [shown.data(value) for value in step.values or []],
        "defines": [{"name": d.name, "kind": d.kind, "written": d.written} for d in defined],
        "notes": list(step.notes),
        "assuming": list(step.assuming),
        "error": step.error.to_dict() if step.error is not None else None,
    }


def _json(record: dict[str, Any], stream: TextIO) -> str:
    """A record as a line of JSON, in ASCII where the stream cannot show Unicode, so that
    no character of it is ever replaced."""
    plain = not _can_encode(stream, "θ²−✓")
    return json.dumps(record, ensure_ascii=plain, default=str)


# -- --render and --ir -----------------------------------------------------------------


def _render_command(workbook: Workbook, plan: _Plan, console: _Console) -> int:
    """``--render``: the notebook with its results, to standard output or ``-o FILE``;
    the errors are typeset in it, and reported on standard error too."""
    progress = _Progress(console, "rendering")
    progress.begin(plan.name)
    try:
        document, errors = render_notebook(workbook, plan.text, plan.name)
    except KeyboardInterrupt:
        _stop_workers(workbook)
        console.note(f"interrupted while rendering {plan.name}; nothing was written")
        raise _Stopped(INTERRUPTED) from None
    except _Terminated:
        _stop_workers(workbook)
        console.note(f"terminated while rendering {plan.name}; nothing was written")
        raise _Stopped(TERMINATED) from None
    finally:
        progress.end()
    if plan.output is not None and plan.output != "-":
        try:
            _write_atomically(Path(plan.output), document)
        except OSError as failure:
            reason = failure.strerror or failure
            console.note(f"relativisticpy: error: cannot write {plan.output}: {reason}")
            return USAGE
    else:
        console.out(document.rstrip("\n"))
    for error in errors:
        console.diagnostic(error)
    if errors:
        places = ", ".join(_where(error.span) for error in errors)
        console.note(
            f"{_plural(len(errors), 'error')}, each typeset in the document where it "
            f"happened: {places}"
        )
        return FAILED
    return OK


def _write_atomically(path: Path, text: str) -> None:
    """Write a file whole or not at all: to a new file beside it, then renamed."""
    folder = path.parent if str(path.parent) else Path(".")
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=folder)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as file:
            file.write(text)
        if path.exists():
            with contextlib.suppress(OSError):
                os.chmod(temporary, path.stat().st_mode & 0o7777)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def _ir_command(workbook: Workbook, plan: _Plan, console: _Console) -> int:
    """``--ir``: what a program, or the blocks of a notebook in order, compile to."""
    notebook = Path(plan.name).suffix.lower() in _MARKDOWN
    try:
        listing = workbook.ir(plan.sources if notebook else plan.sources[0])
    except RelativisticPyError as error:
        console.diagnostic(error)
        return FAILED
    except Exception as error:  # a failure of our own: reported as one
        console.diagnostic(internal_error(error, during="compiling this"))
        return FAILED
    console.out(listing)
    return OK


# -- checking papers -------------------------------------------------------------------


def check(argv: Sequence[str]) -> int:
    """``relativisticpy check``: what became of each equation of some documents."""
    parser = _check_parser()
    args = parser.parse_args(list(argv))
    if args.latex:
        parser.error("--latex is for runs; a check reports in unicode, text, json or github")
    lines = None
    if args.lines is not None:
        if len(args.files) > 1 or any(_is_pattern(f) or Path(f).is_dir() for f in args.files):
            parser.error("--lines picks lines of one document: give one FILE with it")
        try:
            lines = checking.line_range(args.lines)
        except UsageError as error:
            parser.error(error.message)
    policy: set[str] = set(checking.STRICT) if args.strict else set()
    for names in args.fail_on or []:
        policy |= names
    named = _policy_name(args.strict, args.fail_on)
    form = args.format or _default_form(sys.stdout)
    console = _Console(sys.stdout, sys.stderr, args.color)
    documents, status = _documents(args.files, console)
    reports: list[Report] = []
    width = _width(sys.stdout) if form in ("unicode", "text") else None
    progress = _Progress(console, "checking")
    for document in documents:
        name = "<stdin>" if document == "-" else str(document)
        progress.begin(name, total=_line_count(document))
        try:
            with _watched(progress)(workers=args.workers) as workbook:
                report = _check_one(workbook, document, lines, args.time_limit)
        except FileError as error:  # not UTF-8, or a Jupyter notebook: said, and skipped
            hint = f"\nhint: {error.hint}" if error.hint else ""
            console.diagnostic_text(f"relativisticpy check: error: {error.message}{hint}")
            status = max(status, USAGE)
            continue
        except (KeyboardInterrupt, _Terminated) as stop:
            _stop_workers()
            what = "interrupted" if isinstance(stop, KeyboardInterrupt) else "terminated"
            done = "the reports above are complete" if reports else "nothing was reported"
            console.note(f"{what} while checking {name}; {done}")
            return INTERRUPTED if isinstance(stop, KeyboardInterrupt) else TERMINATED
        finally:
            progress.end()
        reports.append(report)
        failed = bool(policy) and report.failed(policy)
        if failed:
            status = max(status, FAILED)
        text = _report_text(report, form, args.all, width, policy, named, failed, console)
        console.out(text if len(reports) == 1 or form in ("json", "github") else "\n" + text)
    if len(reports) > 1 and form in ("unicode", "text"):
        console.out("\n" + _total(reports, policy, named))
    return status


def _policy_name(strict: bool, fail_on: list[set[str]] | None) -> str:
    """What the policy of a check was asked as: ``--strict``, ``--fail-on differs``."""
    names = []
    if strict:
        names.append("--strict")
    if fail_on:
        names.append("--fail-on " + ",".join(sorted(set().union(*fail_on))))
    return " and ".join(names)


def _check_one(
    workbook: Workbook,
    document: Path | str,
    lines: tuple[int | None, int | None] | None,
    time_limit: float,
) -> Report:
    """The report on one document: a LaTeX document, a Markdown notebook (without the
    results ``--render`` typeset in it), or standard input, read as LaTeX."""
    if document == "-":
        text = _stdin_text()
        return workbook.check(Document(text, "<stdin>"), lines=lines, time_limit=time_limit)
    path = Path(document)
    if path.suffix.lower() in _MARKDOWN:
        text = read_text(path)
        blocks = [Equation(source, "$$") for source in notebook_blocks(text, str(path))]
        return checking.check_equations(workbook, blocks, str(path), lines, time_limit)
    return workbook.check(path, lines=lines, time_limit=time_limit)


def _report_text(
    report: Report,
    form: str,
    everything: bool,
    width: int | None,
    policy: set[str],
    named: str,
    failed: bool,
    console: _Console,
) -> str:
    """A report in a form: text for a person, with what failed the policy said last; one
    line of JSON; or GitHub Actions annotations after its summary."""
    if form == "json":
        data = report.to_dict()
        data["policy"] = sorted(policy)
        data["failed"] = failed
        return _json(data, console.stdout)
    if form == "github":
        return "\n".join([report.summary(), *_annotations(report, everything)])
    text = report.describe(all=everything, width=width)
    if form == "text" or not _can_encode(console.stdout, "✓✗↔→"):
        text = _ascii_marks(text)
    if failed:
        text += "\n\n" + console.style.red(f"failed by {named}") + ": " + _failing(report, policy)
    return console.style.marks(text)


def _failing(report: Report, policy: set[str]) -> str:
    """What made a report fail its policy: ``1 differs (paper.tex:13:3), 1 not parsed``."""
    failing = report.failures(policy)
    parts = []
    for verdict in Verdict:
        found = [f for f in failing if f.verdict is verdict and f.reason != checking.TIME_LIMIT]
        if found:
            places = ", ".join(finding.location for finding in found[:3])
            more = f" and {len(found) - 3} more" if len(found) > 3 else ""
            parts.append(f"{len(found)} {verdict} ({places}{more})")
    late = [f for f in failing if f.reason == checking.TIME_LIMIT]
    if late:
        places = ", ".join(finding.location for finding in late[:3])
        parts.append(f"{len(late)} over the time limit ({places})")
    if not failing and checking.EMPTY in policy and report.equations == 0:
        parts.append("no equation was found")
    return "; ".join(parts)


def _annotations(report: Report, everything: bool) -> list[str]:
    """The findings of a report as GitHub Actions workflow commands, which annotate the
    lines they are about: errors for what differs, is ill-formed or is not parsed,
    warnings for what is not computed or not understood, and notices for the rest."""
    lines = []
    for finding in report.listing(everything):
        if finding.verdict in (Verdict.DIFFERS, Verdict.ILL_FORMED, Verdict.NOT_PARSED):
            level = "error"
        elif finding.verdict in (Verdict.NOT_COMPUTED, Verdict.NOT_UNDERSTOOD):
            level = "warning"
        else:
            level = "notice"
        where = place(finding.at or finding.span)
        properties = {
            "file": finding.file,
            "line": where["line"],
            "col": where["column"],
            "endLine": where["end_line"],
            "endColumn": where["end_column"],
            "title": f"RelativisticPy: {finding.verdict}",
        }
        message = " ".join(finding.text.split())
        if finding.detail:
            message += "\n" + finding.detail
        message += "".join(f"\nnote: {note}" for note in finding.notes)
        if finding.hint:
            message += f"\nhint: {finding.hint}"
        listed = ",".join(
            f"{key}={_escape(str(value), True)}"
            for key, value in properties.items()
            if value is not None
        )
        lines.append(f"::{level} {listed}::{_escape(message, False)}")
    return lines


def _escape(text: str, property_: bool) -> str:
    """Text as a workflow command takes it."""
    text = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    if property_:
        text = text.replace(":", "%3A").replace(",", "%2C")
    return text


def _total(reports: list[Report], policy: set[str], named: str) -> str:
    """The last line of a check of several documents."""
    equations = sum(report.equations for report in reports)
    statements = sum(len(report) for report in reports)
    seconds = sum(report.seconds for report in reports)
    line = (
        f"{len(reports)} documents checked in {seconds:.1f} s: "
        f"{_plural(equations, 'equation')}, {_plural(statements, 'statement')}"
    )
    if policy:
        failing = [report.name for report in reports if report.failed(policy)]
        if failing:
            line += f"; {len(failing)} failed by {named}: {', '.join(failing)}"
        else:
            line += f"; none failed by {named}"
    return line


def _ascii_marks(text: str) -> str:
    """A report with marks a console without Unicode can show, each distinct."""
    marks = {"✓": "+", "✗": "x", "↔": "<>", "→": "->"}
    for verdict in Verdict:
        mark = verdict.mark
        if mark in marks:
            text = text.replace(f"{mark} {verdict.value}", f"{marks[mark]} {verdict.value}")
    return text


def _documents(arguments: list[str], console: _Console) -> tuple[list[Path | str], int]:
    """The documents ``check`` is given: files, ``-``, patterns such as ``*.tex``, and
    folders, in which it finds the LaTeX documents that no other includes and the
    Markdown notebooks with maths in them. A file that is missing is reported, and the
    others are checked."""
    found: list[Path | str] = []
    status = OK
    seen: set[str] = set()

    def add(document: Path | str) -> None:
        key = document if isinstance(document, str) else str(document.resolve())
        if key not in seen:
            seen.add(key)
            found.append(document)

    for argument in arguments:
        if argument == "-":
            add("-")
            continue
        path = Path(argument)
        if path.is_dir():
            inside = _documents_in(path)
            if not inside:
                console.diagnostic_text(
                    f"relativisticpy check: error: {path} has no LaTeX documents (.tex), and "
                    "no Markdown notebooks (.md) with maths in them"
                )
                status = USAGE
            for document in inside:
                add(document)
        elif not path.exists() and _is_pattern(argument):
            matches = sorted(glob.glob(argument, recursive=True))
            if not matches:
                console.diagnostic_text(f"relativisticpy check: error: no file matches {argument}")
                status = USAGE
            for match in matches:
                if Path(match).is_file():
                    add(Path(match))
        elif not path.exists():
            console.diagnostic_text(f"relativisticpy check: error: no file {path}")
            status = USAGE
        else:
            add(path)  # a file that cannot be read otherwise is reported when it is checked
    return found, status


def _is_pattern(argument: str) -> bool:
    return any(character in argument for character in "*?[")


_SKIPPED_FOLDERS = frozenset(
    {"node_modules", "site", "_build", "build", "dist", "venv", "__pycache__"}
)
_INCLUDE = re.compile(r"\\(?:input|include|subfile)\s*\{([^}]+)\}")


def _documents_in(folder: Path) -> list[Path]:
    """The documents in a folder and the folders in it, but not hidden ones: the LaTeX
    documents that no other includes, and the Markdown notebooks that have maths."""
    files: list[Path] = []
    for root, folders, names in os.walk(folder):
        folders[:] = sorted(
            f for f in folders if not f.startswith(".") and f not in _SKIPPED_FOLDERS
        )
        files.extend(Path(root) / name for name in sorted(names))
    latex = [f for f in files if f.suffix.lower() == ".tex"]
    included: set[Path] = set()
    for document in latex:
        with contextlib.suppress(OSError, UnicodeDecodeError):
            for match in _INCLUDE.finditer(document.read_text(encoding="utf-8")):
                target = document.parent / match.group(1).strip()
                for candidate in (target, target.with_name(target.name + ".tex")):
                    included.add(candidate.resolve())
    found = []
    for file in files:
        suffix = file.suffix.lower()
        if suffix == ".tex" and file.resolve() not in included:
            found.append(file)
        elif suffix in _MARKDOWN:
            with contextlib.suppress(OSError, UnicodeDecodeError, RelativisticPyError):
                if notebook_blocks(file.read_text(encoding="utf-8"), str(file)):
                    found.append(file)
    return found


def _line_count(document: Path | str) -> int | None:
    if document == "-":
        return None
    try:
        with open(document, "rb") as file:
            return sum(1 for _ in file)
    except OSError:
        return None


def _watched(progress: _Progress) -> type[Workbook]:
    """A kind of workbook that tells a progress line which equation a check is at. A
    check runs its equations in a new workbook of the kind it is given, so each of them
    comes through :meth:`~relativisticpy.Workbook.each` here first."""

    class Watched(Workbook):
        def each(
            self, source: str | Source, *, name: str = "<input>", recover: bool = False
        ) -> Iterator[Step]:
            if isinstance(source, Source):
                progress.at(source.name, source.origin[0])
            return super().each(source, name=name, recover=recover)

    return Watched


# -- reading the documentation -----------------------------------------------------------


def help_command(argv: Sequence[str]) -> int:
    """``relativisticpy help [TOPIC]``: the documentation, offline, or one function of the
    language in full."""
    parser = argparse.ArgumentParser(
        prog="relativisticpy help",
        description="Print the documentation, offline: the list of topics, one topic, or "
        "one function of the language in full.",
        epilog=f"Examples:\n  relativisticpy help\n  relativisticpy help language\n"
        f"  relativisticpy help besselj\n\nThe same pages are online at {DOCS}.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "topic",
        nargs="?",
        help="a topic, as the list of topics names it, or the name of a function, as besselj",
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="colour in a terminal (default: auto)",
    )
    args = parser.parse_args(list(argv))
    console = _Console(sys.stdout, sys.stderr, args.color)
    if args.topic is None:
        console.out(help_text(prompt=False))
        return OK
    try:
        text = documentation.page(args.topic)
    except UsageError as error:
        console.diagnostic_text(
            f"relativisticpy help: error: {error.message}"
            + (f"\nhint: {error.hint}" if error.hint else "")
        )
        return USAGE
    console.out(text.rstrip("\n"))
    return OK


# -- serving agents ------------------------------------------------------------------------------


def mcp_command(argv: Sequence[str]) -> int:
    """``relativisticpy mcp``: serve RelativisticPy to AI agents over the Model Context
    Protocol, on standard input and output; :mod:`relativisticpy.mcp` is the server."""
    from relativisticpy import mcp

    return mcp.main(argv)


#: What the language looks like, as ``help`` shows it first: statements that run, in
#: order, each with what it does.
CHEAT_SHEET = (
    ("x := 3", "define a name: := always defines"),
    ("f(y) := y^2 \\sin(y)", "define a function"),
    ("ds^2 = -(1 - 2M/r) dt^2 + dr^2/(1 - 2M/r) + r^2 d\\Omega^2", "a metric, by its line element"),
    ("g_{\\mu\\nu} := Schwarzschild(M)", "or a known spacetime"),
    ("\\Gamma^{r}_{tt}", "a component of the connection"),
    ("R_{abcd} R^{abcd}", "contract indices"),
    ("R_{\\mu\\nu} = 0", "check an equation: does it hold?"),
    ("\\int_0^1 x^2 \\, dx", "LaTeX works as written"),
)


def help_text(*, prompt: bool) -> str:
    """The first page of help: what the language looks like, the topics of the
    documentation, and the commands of the command line or of the prompt."""
    lines = [
        "RelativisticPy reads mathematics as written, in LaTeX, plain text or Unicode, and",
        "computes it:",
        "",
    ]
    for code, meaning in CHEAT_SHEET:
        if len(code) > 34:  # a long statement on a line of its own
            lines += [f"  {code}", f"  {'':36}{meaning}"]
        else:
            lines.append(f"  {code:<36}{meaning}")
    topics = documentation.topics()
    lines.append("")
    if topics:
        asked = "help TOPIC" if prompt else "relativisticpy help TOPIC"
        lines.append(f"The documentation, offline: {asked}, where TOPIC is one of")
        size = max(len(topic) for topic, _ in topics) + 2
        lines.extend(f"  {topic:<{size}}{title}" for topic, title in topics)
        lines += [
            "or the name of a function, as besselj, which it describes in full: its forms,",
            "its arguments, its examples with their answers, and what computes it.",
        ]
    else:
        lines.append(f"The documentation is not installed with this copy; it is at {DOCS}.")
    lines.append("")
    if prompt:
        lines += [
            "At the prompt:",
            "  help [TOPIC]     this help, a topic, or a function in full",
            "  who              what is defined",
            "  reset            forget every definition",
            "  exit, quit       leave, as Ctrl-D does",
            "  Ctrl-C           clear the line, or stop a computation; definitions are kept",
        ]
    else:
        lines += [
            "Commands:",
            "  relativisticpy FILE            run a program, or a Markdown notebook (.md)",
            "  relativisticpy -c CODE         run CODE",
            "  relativisticpy                 start the prompt",
            "  relativisticpy check FILE...   check the equations of LaTeX documents",
            "  relativisticpy help TOPIC      print a topic, or a function in full",
            "  relativisticpy mcp             serve RelativisticPy to AI agents, over MCP",
            "  relativisticpy --help          every option",
        ]
    lines += ["", f"Online: {DOCS}"]
    return "\n".join(lines)


# -- the prompt ---------------------------------------------------------------------------


def interact(
    workbook: Workbook,
    *,
    read: Callable[[str], str] | None = None,
    out: TextIO | None = None,
    err: TextIO | None = None,
    form: str = "unicode",
    banner: bool = True,
    color: str = "auto",
) -> int:
    r"""The prompt: read an input, run it, print what came of it, and go on; the exit
    status, 0, when the session ends.

    Definitions persist from one input to the next. An error, even a failure of
    RelativisticPy's own, never ends the session. Ctrl-C clears the input, or stops a
    computation, and keeps every definition; Ctrl-D leaves from an empty prompt. An input
    goes on over several lines while a bracket, ``\begin{...}``, ``$$`` or ``\[`` is
    open, or while a line ends with an operator; lines pasted at once are one input.
    Whole lines that are commands: ``help [TOPIC]``, ``who``, ``reset``, ``exit`` and
    ``quit``.

    ``read`` reads a line after showing a prompt, as ``input`` does, which it is by
    default, with readline's editing, history and completion in a terminal."""
    out = out or sys.stdout
    err = err or sys.stderr
    console = _Console(out, err, color)
    editing = read is None and _is_terminal(sys.stdin)
    if editing:
        _line_editor(workbook)
    prompt = _Prompt(workbook, read or input, console, form, pasted=editing)
    if banner:
        console.out(
            f"RelativisticPy {__version__}: type help for help, who for what is defined, "
            "exit or Ctrl-D to leave."
        )
    try:
        return prompt.loop()
    finally:
        if editing:
            _save_history()


class _Prompt:
    """The state of a session at the prompt."""

    def __init__(
        self,
        workbook: Workbook,
        read: Callable[[str], str],
        console: _Console,
        form: str,
        *,
        pasted: bool,
    ) -> None:
        self.workbook = workbook
        self.read = read
        self.console = console
        self.form = form
        self.pasted = pasted  # whether lines pasted at once can be told apart
        self.inputs = 0  # how many inputs ran, which names them: <input 3>

    def loop(self) -> int:
        while True:
            try:
                text = self._input()
            except EOFError:  # Ctrl-D at an empty prompt
                self.console.out("")
                return OK
            if text is None or not text.strip():
                continue
            command = text.strip()
            if command in ("exit", "quit", "exit()", "quit()"):
                return OK
            if self._command(command):
                continue
            python = _python_hint(text)
            if python is not None:
                self.console.diagnostic_text(f"hint: {python}")
                continue
            self.inputs += 1
            self._run(Source(text, f"<input {self.inputs}>"))

    def _input(self) -> str | None:
        """One input, over as many lines as it takes; ``None`` if Ctrl-C cleared it."""
        lines: list[str] = []
        while True:
            try:
                line = self.read(CONTINUATION if lines else PROMPT)
            except KeyboardInterrupt:  # Ctrl-C: the input is dropped, the session goes on
                self.console.out("")
                return None
            except EOFError:
                if lines:  # Ctrl-D on an unfinished input: run what there is
                    self.console.out("")
                    return "\n".join(lines)
                raise
            lines.append(line)
            text = "\n".join(lines)
            if self.pasted and _pending(sys.stdin):  # the rest of a paste: the same input
                continue
            if not _unfinished(text):
                return text

    def _command(self, command: str) -> bool:
        """Do a whole-line command, if the input is one."""
        words = command.split()
        if words[0] in ("help", "?", "help()") and len(words) <= 2:
            if len(words) == 1:
                self.console.out(help_text(prompt=True))
                return True
            try:
                self.console.out(documentation.page(words[1]).rstrip("\n"))
            except UsageError as error:
                self.console.diagnostic(error)
            return True
        if command == "who":
            self.console.out(_who(self.workbook, self.form))
            return True
        if command == "reset":
            self._reset()
            return True
        return False

    def _reset(self) -> None:
        """Forget every definition, once the person says so."""
        names = [_written(d.written, self.form) for d in self.workbook.names().values()]
        if not names:
            self.console.out("nothing is defined")
            return
        listed = ", ".join(names[:5]) + (f" and {len(names) - 5} more" if len(names) > 5 else "")
        try:
            answer = self.read(f"forget {listed}? [y/N] ")
        except (EOFError, KeyboardInterrupt):
            self.console.out("")
            answer = ""
        if answer.strip().lower() in ("y", "yes"):
            self.workbook.reset()
            self.console.out("every definition is forgotten")
        else:
            self.console.out("nothing is forgotten")

    def _run(self, source: Source) -> None:
        """Run an input, a statement at a time, printing what came of each."""
        printer = _Printer(self.form, self.console, None)
        printer.labelled = False
        progress = _Progress(self.console, "computing")
        started = time.monotonic()
        current: Span | None = None
        try:
            planned = _planned(source)
            steps = self.workbook.each(source, recover=True)
            for _, span in planned:
                current, started = span, time.monotonic()
                before = _snapshot(self.workbook)
                progress.begin(_shortened(span.text, 60))
                try:
                    step = next(steps, None)
                finally:
                    progress.end()
                if step is None:  # never so: each gives the steps it was planned to
                    continue
                try:
                    failed = step.error is not None
                    defined = [] if failed else _defined(before, _snapshot(self.workbook))
                    printer.step(step, defined)
                except (KeyboardInterrupt, _Terminated):
                    raise
                except Exception as error:  # a failure of our own, while printing
                    during = "printing this result"
                    printer.error(internal_error(error, span=step.span, during=during))
        except KeyboardInterrupt:
            _stop_workers(self.workbook)
            what = f" {_shortened(current.text, 40)}" if current is not None else ""
            self.console.note(
                f"interrupted: stopped computing{what} after "
                f"{_duration(time.monotonic() - started)}; every definition is kept"
            )
        except _Terminated:
            raise
        except Exception as error:  # a failure of our own: reported, and the session goes on
            self.console.diagnostic(internal_error(error, span=current, during="running this"))


def _who(workbook: Workbook, form: str) -> str:
    """What a workbook defines, by kind, each in the order it was defined."""
    kinds = {
        "coordinate": "coordinates",
        "metric": "metric",
        "function": "functions",
        "tensor": "tensors",
        "value": "values",
        "space": "spaces",
    }
    grouped: dict[str, list[str]] = {}
    for definition in workbook.names().values():
        grouped.setdefault(definition.kind, []).append(_written(definition.written, form))
    if not grouped:
        return "nothing is defined yet"
    size = max(len(kinds.get(kind, kind)) for kind in grouped) + 2
    return "\n".join(
        f"{kinds.get(kind, kind):<{size}}{', '.join(grouped[kind])}"
        for kind in [*kinds, *sorted(set(grouped) - set(kinds))]
        if kind in grouped
    )


#: Input that is Python, which the prompt would otherwise read as mathematics: a line of a
#: Python session, pasted, and calls of Python's or of RelativisticPy's Python API.
_PYTHON = re.compile(
    r"^\s*(?:>>>|\.\.\.)(?:\s|$)"
    r"|\b(?:Workbook|print)\s*\("
    r"|\b(?:wb|np|sympy|sp|relativisticpy)\.[A-Za-z_]",
    re.MULTILINE,
)


def _python_hint(text: str) -> str | None:
    """What to say to input that is Python rather than the language, if it looks so."""
    if not _PYTHON.search(text):
        return None
    return (
        "this prompt reads the RelativisticPy language, not Python: write the mathematics "
        "itself, as in R_{abcd} R^{abcd}; in Python, give it to a workbook, as in "
        'Workbook().expr(r"R_{abcd} R^{abcd}")'
    )


#: A backslash and what follows it, as far as brackets are concerned: an environment's
#: \begin or \end, \left or \right with its delimiter, an escaped bracket, a command, or
#: an escaped character.
_BACKSLASH = re.compile(
    r"\\(?P<environment>begin|end)\s*\{[^}]*\}"
    r"|\\(?P<size>left|right)\s*(?:\\[{}|]|\\[A-Za-z]+|.)?"
    r"|\\(?P<escaped>[\[\]{}])"
    r"|\\[A-Za-z]+|\\.",
    re.DOTALL,
)
_OPENS = frozenset({"begin", "left", "[", "{"})


def _unfinished(source: str) -> bool:
    r"""Whether an input goes on over the next line: a bracket, ``\left``,
    ``\begin{...}``, ``$$`` or ``\[`` is open, or it ends with an operator or a single
    backslash (but not with a LaTeX row break, ``\\``). Comments and text in quotes count
    for nothing."""
    depth = displays = position = 0
    kept = []  # the input without its comments, to see how it ends
    while position < len(source):
        character = source[position]
        if character == "%":  # a comment, to the end of its line
            newline = source.find("\n", position)
            position = len(source) if newline == -1 else newline
            continue
        if character == '"':  # text in quotes
            closing = source.find('"', position + 1)
            end = len(source) if closing == -1 else closing + 1
            kept.append(source[position:end])
            position = end
            continue
        if source.startswith("$$", position):
            displays += 1
            kept.append("$$")
            position += 2
            continue
        if character == "\\" and position + 1 < len(source):
            match = _BACKSLASH.match(source, position)
            assert match is not None  # \\. matches a backslash with anything after it
            kind = match.group("environment") or match.group("size") or match.group("escaped")
            if kind is not None:
                depth += 1 if kind in _OPENS else -1
            kept.append(match.group(0))
            position = match.end()
            continue
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        kept.append(character)
        position += 1
    if depth > 0 or displays % 2:
        return True
    written = "".join(kept).rstrip()
    if written.endswith("\\\\"):
        return False
    return written.endswith(("\\", "+", "-", "*", "/", "=", ":=", "\\cdot", "\\times"))


#: How soon, in seconds, the next line of a paste follows the one before it.
PASTE = 0.03


def _pending(stream: TextIO) -> bool:
    """Whether more input is waiting already, as when several lines are pasted at once: a
    terminal gives a line at a time, and the rest of a paste follows at once."""
    return _ready(stream, PASTE, otherwise=False)


# -- the line editor ------------------------------------------------------------------------


def _history_path() -> Path | None:
    """Where the prompt keeps its history: ``RELATIVISTICPY_HISTORY`` if it is set (to
    nothing, for none), or the user's state directory."""
    configured = os.environ.get("RELATIVISTICPY_HISTORY")
    if configured is not None:
        return Path(configured) if configured else None
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    return base / "relativisticpy" / "history"


def _line_editor(workbook: Workbook) -> bool:
    """Give ``input`` readline's line editing, its history, kept in :func:`_history_path`
    from one session to the next, and completion; whether readline is there. Without it,
    the prompt works as well, without editing."""
    try:
        import readline
    except ImportError:  # as on some builds of Python: the prompt works without it
        return False
    path = _history_path()
    if path is not None:
        with contextlib.suppress(OSError):
            readline.read_history_file(path)
    readline.set_history_length(2000)
    readline.set_completer_delims(" \t\n()[]{},;+-*/^_=&|<>'\"")
    readline.set_completer(_completer(workbook))
    if "libedit" in (readline.__doc__ or ""):  # the editline of macOS and of some builds
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")
    return True


def _save_history() -> None:
    """Keep the prompt's history for the next session, if readline kept one."""
    path = _history_path()
    readline = sys.modules.get("readline")
    if path is None or readline is None:
        return
    with contextlib.suppress(OSError):
        path.parent.mkdir(parents=True, exist_ok=True)
        readline.write_history_file(str(path))


def _completer(workbook: Workbook) -> Callable[[str, int], str | None]:
    """Completion of the built-in functions, spacetimes and constants, of LaTeX commands,
    of the prompt's commands and of what the workbook defines."""
    matches: list[str] = []

    def complete(text: str, state: int) -> str | None:
        if state == 0:
            matches[:] = _completions(workbook, text)
        return matches[state] if state < len(matches) else None

    return complete


def _completions(workbook: Workbook, text: str) -> list[str]:
    """What ``text`` may be the start of: a LaTeX command after a backslash; otherwise a
    built-in function, spacetime or constant, a name the workbook defines, or a command
    of the prompt."""
    from relativisticpy.compiler.builtins import BUILTINS, CONSTANTS
    from relativisticpy.language import vocabulary

    if text.startswith("\\"):
        commands = set(vocabulary.COMMANDS) | set(vocabulary.GREEK)
        commands |= set(vocabulary.PARSER_COMMANDS)
        return sorted("\\" + name for name in commands if name.startswith(text[1:]))
    names = set(BUILTINS) | set(CONSTANTS) | set(workbook.names())
    names |= {"help", "who", "reset", "exit", "quit", "Coordinates"}
    return sorted(name for name in names if name.startswith(text))


# -- the terminal ------------------------------------------------------------------------------


class _Style:
    """Colour, used sparingly, where the stream is a terminal that shows it: never with
    ``NO_COLOR`` set, ``TERM=dumb`` or ``--color never``."""

    def __init__(self, stream: TextIO, choice: str = "auto") -> None:
        self.enabled = _colour(stream, choice)

    def _paint(self, text: str, code: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if self.enabled and text else text

    def dim(self, text: str) -> str:
        return self._paint(text, "2")

    def red(self, text: str) -> str:
        return self._paint(text, "1;31")

    def marks(self, text: str) -> str:
        """A report, the mark and name of each verdict in colour: green for what holds,
        red for what differs or is ill-formed, yellow for what was not checked."""
        if not self.enabled:
            return text
        colours = {"✓": "32", "+": "32", "✗": "31", "x": "31", "?": "33", "!": "33", "~": "33"}

        def paint(match: re.Match[str]) -> str:
            colour = colours.get(match.group(1))
            return self._paint(match.group(0), colour) if colour else match.group(0)

        names = "|".join(re.escape(v.value) for v in sorted(Verdict, key=lambda v: -len(v.value)))
        return re.sub(rf"(?<!\S)(\S{{1,2}}) (?:{names})(?=\s|$)", paint, text, flags=re.MULTILINE)

    def diagnostic(self, text: str) -> str:
        """A diagnostic with its headline, its gutter and carets, and ``note:`` and
        ``hint:``, in colour."""
        if not self.enabled:
            return text
        lines = text.split("\n")
        headline = re.match(r"^[^:\n]+:", lines[0])
        if headline is not None:
            lines[0] = self._paint(headline.group(0), "1;31") + lines[0][headline.end() :]
        for number, line in enumerate(lines[1:], start=1):
            if line.startswith(("note:", "hint:")):
                lines[number] = self._paint(line[:5], "1" if line[0] == "n" else "1;36") + line[5:]
            elif re.match(r"\s*(?:-->|:::) ", line):
                lines[number] = self._paint(line, "34")
            elif (gutter := re.match(r"(\s*\d*\s*\|)(.*)$", line)) is not None:
                rest = gutter.group(2)
                if not gutter.group(1)[:-1].strip():  # a row of marks under a line
                    rest = re.sub(r"\^+", lambda m: self._paint(m.group(0), "1;31"), rest)
                lines[number] = self._paint(gutter.group(1), "34") + rest
        return "\n".join(lines)


def _colour(stream: TextIO, choice: str) -> bool:
    if choice == "never":
        return False
    if choice == "always":
        return True
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR", "") not in ("", "0"):
        return True
    if os.environ.get("TERM") == "dumb":
        return False
    return _is_terminal(stream)


def _is_terminal(stream: Any) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def _width(stream: TextIO) -> int | None:
    """The width of the terminal a stream writes to; ``None`` for a pipe or a file, where
    nothing is cut short or wrapped."""
    if not _is_terminal(stream):
        return None
    return max(20, shutil.get_terminal_size().columns)


def _can_encode(stream: TextIO, text: str) -> bool:
    """Whether a stream can write ``text``: a stream of text in memory, which has no
    encoding, takes anything."""
    encoding = getattr(stream, "encoding", None)
    if encoding is None:
        return isinstance(stream, io.StringIO)
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError, TypeError):
        return False
    return True


def _default_form(stream: TextIO) -> str:
    """Unicode, where the stream can show it; plain text otherwise."""
    return "unicode" if _can_encode(stream, "✓²θ→−") else "text"


def _replace_what_cannot_print(stream: TextIO) -> None:
    """Let a stream that cannot encode some character, such as output redirected on
    Windows, print ``?`` for it rather than fail."""
    if not _can_encode(stream, "✓✗→θ²"):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


def _flush(stream: TextIO) -> None:
    with contextlib.suppress(OSError, ValueError):
        stream.flush()


class _Console:
    """Where a command writes: results to standard output, as soon as they exist, and
    diagnostics to standard error, after flushing what came before them, so that a log
    reads in order. Each write first clears a progress line, if one is shown."""

    def __init__(self, stdout: TextIO, stderr: TextIO, color: str = "auto") -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.style = _Style(stdout, color)
        self.error_style = _Style(stderr, color)
        self.lock = threading.Lock()
        self.progress_shown = False

    def clear(self) -> None:
        """Take the progress line away, if one is shown (with the lock held)."""
        if self.progress_shown:
            with contextlib.suppress(OSError, ValueError):
                self.stderr.write("\r\x1b[K")
                self.stderr.flush()
            self.progress_shown = False

    def out(self, text: str) -> None:
        """A line of output, written at once."""
        with self.lock:
            self.clear()
            self.stdout.write(text + "\n")
            self.stdout.flush()

    def note(self, text: str) -> None:
        """A message to the person, on standard error, after the output before it."""
        with self.lock:
            self.clear()
            _flush(self.stdout)
            self.stderr.write(text + "\n")
            self.stderr.flush()

    def diagnostic(self, error: RelativisticPyError) -> None:
        """An error, as :meth:`~relativisticpy.errors.RelativisticPyError.render` gives
        it, with the Python traceback of an internal error if ``RELATIVISTICPY_DEBUG`` is
        set."""
        text = error.render()
        cause = error.cause
        if debugging() and isinstance(error, InternalError) and cause is not None:
            frames = traceback.format_exception(type(cause), cause, cause.__traceback__)
            text += "\n" + "".join(frames).rstrip()
        self.diagnostic_text(text)

    def diagnostic_text(self, text: str) -> None:
        self.note(self.error_style.diagnostic(text))


class _Progress:
    """A line on standard error that says what is being computed, and for how long, once
    it has taken :data:`PROGRESS_AFTER` seconds: only where standard error is a terminal,
    and taken away as soon as anything else is written."""

    def __init__(self, console: _Console, verb: str) -> None:
        self.console = console
        self.verb = verb
        self.enabled = _is_terminal(console.stderr) and os.environ.get("TERM") != "dumb"
        self.what = ""
        self.where: tuple[str, int] | None = None  # the file and line a check is at
        self.total: int | None = None  # the lines of the file checked
        self.started = 0.0
        self.thread: threading.Thread | None = None
        self.stopping = threading.Event()
        self.spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏" if _can_encode(console.stderr, "⠋·") else "|/-\\"
        self.dot = "·" if _can_encode(console.stderr, "·") else "-"

    def begin(self, what: str, *, total: int | None = None) -> None:
        """Start to say what is being done, after a while."""
        if not self.enabled:
            return
        self.what, self.total, self.where = what, total, None
        self.started = time.monotonic()
        self.stopping.clear()
        self.thread = threading.Thread(target=self._run, name="relativisticpy-progress")
        self.thread.daemon = True
        self.thread.start()

    def at(self, name: str, line: int) -> None:
        """Say which line, of which file, is being done."""
        self.where = (name, line)

    def end(self) -> None:
        """Stop saying it, and take the line away."""
        if self.thread is None:
            return
        self.stopping.set()
        self.thread.join()
        self.thread = None
        with self.console.lock:
            self.console.clear()

    def line(self, turn: int) -> str:
        """The progress line: ``⠼ checking paper.tex, line 9102 of 12011 · 1 min 12 s``."""
        what = _short_name(self.what) if self.verb != "computing" else self.what
        text = f"{self.spinner[turn % len(self.spinner)]} {self.verb} {what}"
        if self.where is not None:
            name, line = self.where
            if name != self.what:
                text += f", {_short_name(name)}:{line}"
            else:
                text += f", line {line}" + (f" of {self.total}" if self.total else "")
        text += f" {self.dot} {_duration(time.monotonic() - self.started)}"
        width = shutil.get_terminal_size().columns
        return text if len(text) < width else text[: max(1, width - 2)] + "…"

    def _run(self) -> None:
        if self.stopping.wait(PROGRESS_AFTER):
            return
        turn = 0
        while not self.stopping.is_set():
            text = self.line(turn)
            with self.console.lock, contextlib.suppress(OSError, ValueError):
                self.console.stderr.write("\r\x1b[K" + text)
                self.console.stderr.flush()
                self.console.progress_shown = True
            turn += 1
            self.stopping.wait(0.1)


# -- signals and workers --------------------------------------------------------------------


class _Terminated(BaseException):
    """SIGTERM, as a harness stops a command that takes too long: raised where the
    command is, so that it says where it stopped, flushes what it printed, and stops its
    workers before it ends with status 143."""


@contextlib.contextmanager
def _sigterm() -> Iterator[None]:
    usable = hasattr(signal, "SIGTERM") and threading.current_thread() is threading.main_thread()
    if not usable:
        yield
        return

    def stop(signum: int, frame: Any) -> None:
        raise _Terminated

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _stop_workers(workbook: Workbook | None = None) -> None:
    """Stop the worker processes at once, without waiting for what they compute: after
    Ctrl-C, or when the command is terminated. A workbook starts new ones when it needs
    them again, and keeps every definition."""
    for child in multiprocessing.active_children():
        with contextlib.suppress(Exception):
            child.kill()  # a worker took this command's own SIGTERM handler with it
    if workbook is not None:
        with contextlib.suppress(Exception):
            workbook.close()


# -- the command line's parsers --------------------------------------------------------------


class _Version(argparse.Action):
    """``--version``: RelativisticPy's version, and Python's, SymPy's and NumPy's, which a
    report of a bug needs."""

    def __init__(self, option_strings: Sequence[str], dest: str, **kwargs: Any) -> None:
        super().__init__(option_strings, dest, nargs=0, help=kwargs.get("help"))

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: Any,
        values: Any,
        option: str | None = None,
    ) -> None:
        print(version_text())
        parser.exit()


def version_text() -> str:
    """``relativisticpy 2.0.0 (Python 3.13.1, SymPy 1.14.0, NumPy 2.3.1)``."""
    from importlib import metadata

    parts = [f"Python {sys.version.split()[0]}"]
    for package, name in (("sympy", "SymPy"), ("numpy", "NumPy")):
        with contextlib.suppress(metadata.PackageNotFoundError):
            parts.append(f"{name} {metadata.version(package)}")
    return f"relativisticpy {__version__} ({', '.join(parts)})"


_RUN_DESCRIPTION = """\
Compute mathematics written as in a paper: LaTeX, plain text or Unicode.

relativisticpy FILE runs a program, or the blocks of maths of a Markdown notebook (.md),
one statement at a time, printing each result as soon as it is computed, labelled with
its statement. It stops at the first error, with every result before it printed, unless
given --keep-going. With no FILE, it starts the prompt in a terminal, and reads standard
input as a program from a pipe.

Files: a program is any file of statements (.relpy by convention); a notebook is a .md
or .markdown file, whose $$...$$, \\[...\\], equation environments and ```math blocks
run in order. A LaTeX document (.tex) is checked, not run: relativisticpy check FILE."""

_RUN_EPILOG = f"""\
Examples:
  relativisticpy -c '\\int_0^1 x^2 \\, dx'                          1/3
  relativisticpy -c 'g_{{\\mu\\nu}} := Schwarzschild(M); R_{{abcd}} R^{{abcd}}'
  relativisticpy schwarzschild.relpy -k              every statement, every error
  relativisticpy notes.md --render -o notes.md       typeset each block's results
  printf 'x := 2\\nx^3\\n' | relativisticpy --format json
  relativisticpy check paper.tex --strict

Exit status: 0 done, and everything succeeded; 1 a statement failed; 2 the command line
was wrong (an unknown option, or a file that cannot be read); 130 interrupted (Ctrl-C);
143 terminated (SIGTERM).

Environment: NO_COLOR turns colour off; RELATIVISTICPY_DEBUG=1 shows the Python
traceback of an internal error; RELATIVISTICPY_HISTORY is the prompt's history file.

Learn more: relativisticpy help, relativisticpy help cli, or {DOCS}"""

_USAGE = (
    "relativisticpy [options] [FILE | -c CODE | -]\n"
    "       relativisticpy check [options] FILE...\n"
    "       relativisticpy help [TOPIC]\n"
    "       relativisticpy mcp [options]"
)


def _run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relativisticpy",
        usage=_USAGE,
        description=_RUN_DESCRIPTION,
        epilog=_RUN_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "files",
        nargs="*",
        metavar="FILE",
        help="a program, a Markdown notebook (.md), or - for standard input",
    )
    parser.add_argument(
        "-c",
        "--code",
        action="append",
        metavar="CODE",
        help="run CODE instead of a file: statements separated by line breaks or ';'. "
        "Given more than once, the pieces run in order, as lines of one program",
    )
    parser.add_argument(
        "-k",
        "--keep-going",
        action="store_true",
        help="run every statement, reporting every error in order, rather than stopping "
        "at the first; the status is still 1 if one failed",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="start the prompt after running FILE or CODE, keeping what it defined; "
        "or start it when standard input is not a terminal",
    )
    parser.add_argument(
        "--format",
        choices=FORMS,
        help="how results are printed: unicode (the default, where it can be shown), "
        "text (ASCII, in the language's own syntax, which reads back), latex, or json "
        "(JSON Lines: one object per statement, with its values or its error)",
    )
    parser.add_argument("--latex", action="store_true", help="the same as --format latex")
    parser.add_argument(
        "--render",
        action="store_true",
        help="print a Markdown notebook with each block's results typeset after it, and "
        "each error where it happened; rendering again replaces them",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="with --render, write the document to FILE (which may be the notebook itself)",
    )
    parser.add_argument(
        "--ir",
        action="store_true",
        help="print the IR, the computation the program compiles to, instead of computing it",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=_workers,
        default=1,
        metavar="N",
        help="processes that share component-wise work, such as simplifying the "
        "components of a tensor: a number, or 0 or auto for one per CPU (default: 1)",
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="colour in a terminal (default: auto, which NO_COLOR and TERM=dumb turn off)",
    )
    parser.add_argument(
        "-V",
        "--version",
        action=_Version,
        help="print the versions of RelativisticPy, Python and SymPy",
    )
    return parser


#: What each verdict means, as ``check --help`` lists them.
VERDICTS = {
    Verdict.HOLDS: "proven equal",
    Verdict.DIFFERS: "every quantity is known, and the sides are not equal",
    Verdict.RELATES: "the sides involve quantities nothing defines: a relation to solve",
    Verdict.ILL_FORMED: "the notation breaks the rules of index notation",
    Verdict.DEFINES: "a definition",
    Verdict.COMPUTES: "an expression, computed",
    Verdict.NOT_COMPUTED: "a relation such as \\sim, a condition, or over the time limit",
    Verdict.NOT_UNDERSTOOD: "something RelativisticPy cannot do yet, with the reason",
    Verdict.NOT_PARSED: "not read, with the reason",
}

_CHECK_DESCRIPTION = (
    "Check the equations of LaTeX documents and Markdown notebooks: run them in order, as "
    "a\nreader would, and say what became of each.\n\n"
    + "\n".join(f"  {v.mark} {v.value:<15} {VERDICTS[v]}" for v in Verdict)
    + """

The report lists what fails --strict, and counts the rest; --all lists everything.

Files: a .md or .markdown file is a notebook; anything else is LaTeX, with the macros of
its local packages and of the files it includes (\\input, \\include). A folder is searched
for the LaTeX documents no other includes, and the notebooks with maths; - is standard
input, read as LaTeX."""
)

_CHECK_EPILOG = f"""\
Verdicts for --fail-on: differs, relates, ill-formed, defines, computes, not-computed,
not-understood, not-parsed, and the conditions time-limit (a statement over its time
limit) and empty (a document with no equations). --strict is
--fail-on differs,ill-formed,not-parsed,time-limit,empty.

Examples:
  relativisticpy check paper.tex
  relativisticpy check paper.tex --lines 120:340 --all
  relativisticpy check chapters/ --strict
  relativisticpy check paper.tex --format github --fail-on differs

Exit status: 0 the check passed its policy (always, without --strict or --fail-on); 1 it
failed it; 2 the command line was wrong, or a file cannot be read; 130 interrupted.

Learn more: relativisticpy help checking, or {DOCS}/checking/"""


def _check_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relativisticpy check",
        usage="relativisticpy check [options] FILE...",
        description=_CHECK_DESCRIPTION,
        epilog=_CHECK_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="LaTeX documents, Markdown notebooks, folders, patterns such as '*.tex', or -",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="list every statement, with its reason, not only those that fail",
    )
    parser.add_argument(
        "--lines",
        metavar="RANGE",
        help="check only these lines of one document: FIRST:LAST, a line, FIRST: or :LAST. "
        "What comes before them still runs, for its definitions",
    )
    parser.add_argument(
        "--time-limit",
        type=_seconds,
        default=30.0,
        metavar="SECONDS",
        help="seconds allowed for each statement, or inf for no limit (default: 30)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit with status 1 if an equation differs or is ill-formed, a statement is "
        "not parsed or over its time limit, or a document has no equations",
    )
    parser.add_argument(
        "--fail-on",
        type=_verdicts,
        action="append",
        metavar="VERDICTS",
        help="exit with status 1 if a statement meets one of these verdicts, separated by "
        "commas, as in --fail-on differs,not-parsed (see below)",
    )
    parser.add_argument(
        "--format",
        choices=CHECK_FORMS,
        help="unicode (the default, where it can be shown), text (ASCII marks), json (one "
        "object per document), or github (annotations for GitHub Actions)",
    )
    parser.add_argument("--latex", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "-j",
        "--workers",
        type=_workers,
        default=1,
        metavar="N",
        help="processes that share component-wise work: a number, or 0 or auto for one "
        "per CPU (default: 1)",
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="colour in a terminal (default: auto, which NO_COLOR and TERM=dumb turn off)",
    )
    return parser


def _seconds(text: str) -> float:
    """The value of ``--time-limit``: a positive number of seconds, or ``inf``."""
    try:
        seconds = float(text)
    except ValueError:
        seconds = math.nan
    if not seconds > 0:  # nor nan
        raise argparse.ArgumentTypeError(
            f"takes a positive number of seconds, as in --time-limit 30, or inf for no "
            f"limit, not {text}"
        )
    return seconds


def _workers(text: str) -> int | None:
    """The value of ``-j``: a number of processes, or ``auto`` or ``0`` for one per CPU."""
    if text.strip().lower() in ("auto", "0"):
        return None
    try:
        number = int(text)
    except ValueError:
        number = -1
    if number < 0:
        raise argparse.ArgumentTypeError(
            f"takes a number of processes, as in -j 4, or auto (or 0) for one per CPU, not {text}"
        )
    return number


def _verdicts(text: str) -> set[str]:
    """The value of ``--fail-on``: verdicts and conditions, separated by commas."""
    names = [name.strip() for name in text.split(",") if name.strip()]
    if not names:
        raise argparse.ArgumentTypeError("takes verdicts, as in --fail-on differs,not-parsed")
    try:
        Report("", 1).failed(names)
    except UsageError as error:
        raise argparse.ArgumentTypeError(
            f"{error.message}; and the conditions time-limit and empty"
        ) from None
    return set(names)


# -- small helpers ------------------------------------------------------------------------------


def _where(span: Span | None, *, short: bool = False) -> str:
    """Where a span starts, as editors and compilers write it: ``notes.md:12:5``; with
    ``short``, the file by its name alone, as a progress line has room for."""
    if span is None:
        return "?"
    name = _short_name(span.source.name) if short else span.source.name
    return f"{name}:{span.line}:{span.column}"


def _short_name(name: str) -> str:
    """A file by its name alone: ``paper.tex`` for ``/home/a/papers/paper.tex``."""
    return name if name.startswith("<") else Path(name).name or name


def _shortened(text: str, width: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def _duration(seconds: float) -> str:
    """``0.4 s``, ``12 s``, ``3 min 12 s``."""
    if seconds < 10:
        return f"{seconds:.1f} s"
    if seconds < 60:
        return f"{seconds:.0f} s"
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes} min {rest} s"


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _listed(names: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]
