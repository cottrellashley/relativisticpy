"""The ``relativisticpy`` command.

::

    relativisticpy schwarzschild.relpy      run a program and print each result
    relativisticpy notes.md                 run the $$...$$ blocks of a Markdown file
    relativisticpy notes.md --render        print the document with results typeset
    relativisticpy -c "R_{abcd} R^{abcd}"   run code given on the command line
    relativisticpy --ir program.relpy       print the IR instead of computing
    relativisticpy check paper.tex          check the equations of a LaTeX document
    relativisticpy                          start an interactive session
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO

from relativisticpy import __version__
from relativisticpy.checking import Verdict
from relativisticpy.documents import math_blocks
from relativisticpy.errors import RelativisticPyError
from relativisticpy.language import Source
from relativisticpy.tensors import Tensor
from relativisticpy.workbook import Workbook, latex

_OPENING, _CLOSING = "([{", ")]}"


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main(list(sys.argv[1:] if argv is None else argv))
    except BrokenPipeError:  # the reader stopped reading, as `| head` does
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 1


def _main(arguments: list[str]) -> int:
    if arguments[:1] == ["check"]:
        return check(arguments[1:])
    parser = argparse.ArgumentParser(
        prog="relativisticpy",
        description="General Relativity for physicists in a hurry.",
    )
    parser.add_argument("file", nargs="?", help="a program, or a Markdown file with $$ blocks")
    parser.add_argument("-c", "--code", help="run this code instead of a file")
    parser.add_argument("--latex", action="store_true", help="print results as LaTeX")
    parser.add_argument(
        "--render", action="store_true", help="print a Markdown file with its results typeset"
    )
    parser.add_argument("--ir", action="store_true", help="print the IR instead of computing")
    parser.add_argument(
        "-j", "--workers", type=int, default=1, help="processes for component-wise work"
    )
    parser.add_argument("--version", action="version", version=f"relativisticpy {__version__}")
    args = parser.parse_args(arguments)

    with Workbook(workers=args.workers) as workbook:
        if args.code is None and args.file is None:
            return interact(workbook, latex_output=args.latex)
        try:
            if args.code is not None:
                text, name, notebook = args.code, "<command line>", False
            else:
                path = Path(args.file)
                text, name = path.read_text(encoding="utf-8"), str(path)
                notebook = path.suffix.lower() in (".md", ".markdown")
            if notebook and args.render:
                print(workbook.markdown(text, name=name))
                return 0
            programs = math_blocks(text, name) if notebook else [Source(text, name)]
            if args.ir:
                print("\n".join(workbook.ir(program) for program in programs))
                return 0
            for program in programs:
                for result in workbook.exe(program):
                    print(display(result, latex_output=args.latex))
        except RelativisticPyError as error:
            print(f"{type(error).__name__}: {error}", file=sys.stderr)
            return 1
        except OSError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
    return 0


def check(argv: Sequence[str]) -> int:
    """``relativisticpy check``: what became of each equation of a document."""
    parser = argparse.ArgumentParser(
        prog="relativisticpy check",
        description="Run the equations of a LaTeX document or Markdown notebook in order, "
        "and report which hold, which differ, and which could not be read.",
    )
    parser.add_argument("file", help="a LaTeX document, or a Markdown notebook")
    parser.add_argument("--lines", help="check only these lines, as FIRST:LAST")
    parser.add_argument(
        "--all",
        action="store_true",
        help="list every statement, not only those that hold or differ",
    )
    parser.add_argument(
        "--time-limit",
        type=_seconds,
        default=30.0,
        help="seconds allowed for each statement, or inf for no limit (default: 30)",
    )
    args = parser.parse_args(argv)
    lines = None
    if args.lines:
        first, _, last = args.lines.partition(":")
        if not (first.isdigit() and last.isdigit()):
            parser.error("--lines takes two line numbers, as in --lines 120:340")
        lines = (int(first), int(last))
    with Workbook() as workbook:
        try:
            report = workbook.check(args.file, lines=lines, time_limit=args.time_limit)
        except OSError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
    print(report.summary())
    shown = {Verdict.HOLDS, Verdict.DIFFERS}
    findings = [f for f in report.findings if args.all or f.verdict in shown]
    if findings:
        print()
    for finding in findings:
        print(finding.describe())
    return 0


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


def display(value: Any, *, latex_output: bool = False) -> str:
    """How a result prints: tensors as their nonzero components, LaTeX on request."""
    if latex_output:
        return latex(value)
    if isinstance(value, Tensor) and value.rank > 0:
        head = (value.name or "T") + str(value.indices)
        nonzero = value.nonzero()
        if not nonzero:
            return f"{head} = 0"
        lines = [f"{head}:"]
        for labels, component in nonzero.items():
            lines.append(f"  [{', '.join(labels)}] = {component}")
        return "\n".join(lines)
    return str(value)


def interact(
    workbook: Workbook,
    *,
    latex_output: bool = False,
    read: Callable[[str], str] = input,
    out: TextIO | None = None,
) -> int:
    """A read-evaluate-print loop. Definitions persist; unclosed brackets continue the input."""
    out = out or sys.stdout
    with contextlib.suppress(ImportError):  # history and line editing, where available
        import readline  # noqa: F401
    print(f"RelativisticPy {__version__}: definitions persist; press Ctrl-D to leave.", file=out)
    while True:
        try:
            source = _read_statement(read)
        except (EOFError, KeyboardInterrupt):
            print(file=out)
            return 0
        if not source.strip():
            continue
        try:
            for result in workbook.exe(source):
                print(display(result, latex_output=latex_output), file=out)
        except RelativisticPyError as error:
            print(f"{type(error).__name__}: {error}", file=out)


def _read_statement(read: Callable[[str], str]) -> str:
    lines = [read(">>> ")]
    while _unfinished("\n".join(lines)):
        lines.append(read("... "))
    return "\n".join(lines)


def _unfinished(source: str) -> bool:
    """Whether the input continues on the next line: a bracket is open, or the line
    ends with an operator or a backslash (but not with a LaTeX row break, ``\\\\``)."""
    depth = sum(source.count(c) for c in _OPENING) - sum(source.count(c) for c in _CLOSING)
    if depth > 0:
        return True
    stripped = source.rstrip()
    if stripped.endswith("\\\\"):
        return False
    return stripped.endswith(("\\", "+", "-", "*", "/", "=", ":="))
