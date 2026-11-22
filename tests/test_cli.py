"""The relativisticpy command: runs, checks, help, and its command line."""

import contextlib
import io
import json
import os
import re
import signal
import subprocess
import sys
import textwrap
import time
from importlib import metadata

import pytest

from relativisticpy import Tensor, Workbook
from relativisticpy import cli as command
from relativisticpy.cli import display, main
from relativisticpy.workbook import RESULT_END, RESULT_START

POSIX = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals and pipes")


def run(capsys, *arguments, stdin=None, monkeypatch=None):
    """The exit status, standard output and standard error of a command line."""
    status = main(list(arguments))
    captured = capsys.readouterr()
    return status, captured.out, captured.err


@pytest.fixture
def program(tmp_path):
    def write(text, name="program.relpy"):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return path

    return write


@pytest.fixture
def piped(monkeypatch):
    """Standard input from a pipe, with this text."""

    def pipe(text):
        monkeypatch.setattr(sys, "stdin", io.StringIO(text))

    return pipe


# -- running a program ----------------------------------------------------------------------


def test_one_statement_prints_its_value_alone(capsys):
    assert run(capsys, "-c", "x^2") == (0, "x²\n", "")
    assert run(capsys, "-c", r"\int_0^1 x^2 \, dx") == (0, "1/3\n", "")


def test_each_result_is_labelled_with_its_statement(capsys):
    # CLI-05, DOC-40: which statement each result came from
    status, out, err = run(
        capsys, "-c", "g_{\\mu\\nu} := Schwarzschild(M)\n\\Gamma^{r}_{tt}\nR_{\\mu\\nu} = 0"
    )
    assert status == 0
    assert err == ""
    assert out.splitlines() == [
        "<command line>:1:1: g_{\\mu\\nu} := Schwarzschild(M)",
        "  defines the metric g_{μν} and the coordinates t, r, θ, φ",
        "<command line>:2:1: \\Gamma^{r}_{tt}",
        "  M (r − 2M)/r³",
        "<command line>:3:1: R_{\\mu\\nu} = 0",
        "  R_{μν} = 0: holds in all 10 components",
    ]


def test_statements_on_one_line_are_told_apart_by_column(capsys):
    _, out, _ = run(capsys, "-c", "x := 2; x^2; 1 = 2 = 1")
    labels = [line for line in out.splitlines() if not line.startswith(" ")]
    assert labels == [
        "<command line>:1:1: x := 2",
        "<command line>:1:9: x^2",
        "<command line>:1:14: 1 = 2",
        "<command line>:1:18: 2 = 1",
    ]


def test_a_definition_says_what_it_defines(capsys):
    # CLI-46: definitions are not silent
    _, out, _ = run(capsys, "-c", "x := 2\nx := 3\nf(y) := y^2\nCoordinates := [t, r]")
    assert "  defines x\n" in out
    assert "  redefines x\n" in out
    assert "  defines f(y)\n" in out
    assert "  defines the coordinates t, r\n" in out


def test_a_run_stops_at_the_first_error_with_every_result_before_it(capsys, program):
    # CLI-01, AGT-10, DOC-18: one failing statement hides nothing before it
    path = program("x := 3\nx^2\nx^3\ndiff(sin(x), x)\nx^4\n")
    status, out, err = run(capsys, str(path))
    assert status == 1
    assert f"{path}:2:1: x^2\n  9\n{path}:3:1: x^3\n  27\n" in out
    assert "x^4" not in out
    assert err.startswith("EvaluationError: cannot differentiate with respect to a number\n")
    assert f" --> {path}:4:1" in err
    assert err.endswith(
        f"stopped at the first error, at {path}:4:1: the statement after it did not run; "
        "--keep-going (-k) runs it\n"
    )


def test_keep_going_reports_every_error_in_order(capsys, program):
    # CLI-12: every error, in the order of the source, parse errors among the others
    path = program("a := 1\nT_{ab} u^a\nb := 2\n\\frca{1}{2}\nS_{ab} v^b\na + b\n")
    status, out, err = run(capsys, str(path), "-k")
    assert status == 1
    assert out.endswith(f"{path}:6:1: a + b\n  3\n")
    headlines = [line for line in err.splitlines() if "Error:" in line]
    assert headlines == [
        "CompileError: T is not a tensor",
        "ParseError: unknown LaTeX command \\frca",
        "CompileError: S is not a tensor",
    ]
    assert err.endswith(f"3 of 6 statements failed: {path}:2:1, {path}:4:1, {path}:5:1\n")


def test_after_a_stop_every_statement_that_cannot_be_read_is_reported(capsys, program):
    path = program("x := 1\nT_{ab} u^a\ny := 2\n\\frca{1}{2}\n")
    status, _, err = run(capsys, str(path))
    assert status == 1
    assert "T is not a tensor" in err
    assert "unknown LaTeX command \\frca" in err  # known without running anything
    assert "the 2 statements after it did not run, and 1 of them cannot be read" in err


def test_a_notebook_runs_its_blocks_in_order(capsys, program):
    notes = program("Prose $$ y := 2 $$ and\n\n$$\ny + 1\n$$\n", "notes.md")
    assert run(capsys, str(notes)) == (
        0,
        f"{notes}:1:10: y := 2\n  defines y\n{notes}:4:1: y + 1\n  3\n",
        "",
    )


def test_notebook_errors_point_at_the_file(capsys, program):
    notes = program("Prose.\n\n$$\ny := 2\n$$\n\nMore.\n\n$$\n\\farc{1}{2}\n$$\n", "notes.md")
    status, _, err = run(capsys, str(notes))
    assert status == 1
    assert f"--> {notes}:10:1" in err


def test_a_file_saved_with_a_byte_order_mark(capsys, tmp_path):
    path = tmp_path / "square.relpy"
    path.write_text("x := 3\nx^2\n", encoding="utf-8-sig")
    status, out, _ = run(capsys, str(path))
    assert status == 0
    assert out.endswith("x^2\n  9\n")


def test_display():
    tensor = Tensor([[1, 0], [0, 2]], "_{ab}", name="h")
    assert display(tensor) == "h_{ab}: 2 of 4 components non-zero\n  h_{0 0} = 1\n  h_{1 1} = 2"
    assert display(Tensor([[0, 0], [0, 0]], "_{ab}")) == "_{ab} = 0"  # never named T
    assert display(3) == "3"


# -- forms of output ------------------------------------------------------------------------


def test_json_lines_one_object_per_statement(capsys, program):
    # CLI-16, AGT-08: everything a person can read, a program can read
    path = program("x := 2\nx^3\nT_{ab} u^a\n")
    status, out, err = run(capsys, str(path), "--format", "json", "-k")
    assert status == 1
    records = [json.loads(line) for line in out.splitlines()]
    assert [record["line"] for record in records] == [1, 2, 3]
    assert records[0]["defines"] == [{"name": "x", "kind": "value", "written": "x"}]
    [value] = records[1]["values"]
    assert (value["kind"], value["value"], value["latex"]) == ("number", 8, "8")
    assert records[1]["statement"] == "x^3"
    assert records[1]["error"] is None
    error = records[2]["error"]
    assert (error["type"], error["message"]) == ("CompileError", "T is not a tensor")
    assert (error["line"], error["column"]) == (3, 1)
    assert error["hint"] == "define it first, for example T_{ab} := [...]"
    assert "T is not a tensor" not in err  # the error is in the data, not repeated
    assert err.startswith("1 of 3 statements failed")


def test_latex_labels_are_comments(capsys):
    assert run(capsys, "-c", "\\frac{1}{x}", "--latex") == (0, "\\frac{1}{x}\n", "")
    _, out, _ = run(capsys, "-c", "y := 2\n\\frac{1}{y}", "--format", "latex")
    assert out.splitlines() == [
        "% <command line>:1:1: y := 2",
        "% defines y",
        "% <command line>:2:1: \\frac{1}{y}",
        "\\frac{1}{2}",
    ]


def test_text_reads_back(capsys):
    _, out, _ = run(capsys, "-c", "\\theta^2 + 1", "--format", "text")
    assert out == "\\theta^2 + 1\n"
    assert Workbook().expr(out.strip()) == Workbook().expr("\\theta^2 + 1")


def test_a_console_that_cannot_show_unicode_gets_text(monkeypatch):
    # CLI-26: never a ? for a character the console cannot show
    output = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", output)
    assert main(["-c", "\\theta^2"]) == 0
    output.flush()
    assert output.buffer.getvalue().decode("cp1252") == "\\theta^2\n"


def test_json_in_ascii_where_the_console_cannot_show_unicode(monkeypatch):
    output = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", output)
    assert main(["-c", "\\theta^2", "--format", "json"]) == 0
    output.flush()
    record = json.loads(output.buffer.getvalue().decode("ascii"))
    assert record["values"][0]["unicode"] == "θ²"


# -- standard input -------------------------------------------------------------------------


def test_standard_input_from_a_pipe_runs_as_a_program(capsys, piped):
    # CLI-06, AGT-09: no banner, no prompts, errors on standard error, status 1
    piped("T_{ab} u^a\nx := 2\nx^3\n")
    status, out, err = run(capsys)
    assert status == 1
    assert out == ""
    assert "rp>" not in err
    assert "RelativisticPy" not in err
    assert " --> <stdin>:1:1" in err
    piped("x := 2\nx^3\n")
    assert run(capsys) == (0, "<stdin>:1:1: x := 2\n  defines x\n<stdin>:2:1: x^3\n  8\n", "")


def test_a_failure_of_the_command_itself_is_called_that(capsys, monkeypatch):
    # CLI-04: never a traceback, at the top either
    def broken(arguments):
        raise RuntimeError("broken")

    monkeypatch.setattr(command, "run", broken)
    status, _, err = run(capsys, "-c", "x")
    assert status == 1
    assert err.startswith("internal error: RelativisticPy failed while running the command")
    assert "Traceback" not in err


def test_standard_input_that_is_not_utf8(capsys, monkeypatch):
    stdin = io.TextIOWrapper(io.BytesIO("x + \\text{é}\n".encode("latin-1")))
    monkeypatch.setattr(sys, "stdin", stdin)
    with pytest.raises(SystemExit) as stop:
        main(["-"])
    assert stop.value.code == 2
    assert "standard input is not UTF-8 text: the byte 0xE9" in capsys.readouterr().err
    stdin.seek(0)
    assert main(["check", "-"]) == 2
    assert "relativisticpy check: error: standard input is not UTF-8" in capsys.readouterr().err


def test_input_left_unfinished_is_an_error(capsys, piped):
    piped("x := 2\n(x + 1\n")
    status, _, err = run(capsys)
    assert status == 1
    assert "this '(' is never closed" in err


def test_a_dash_is_standard_input(capsys, piped):
    # CLI-36
    piped("1 + 1\n")
    assert run(capsys, "-") == (0, "2\n", "")


def test_the_prompt_after_a_program(capsys, piped):
    # -i: run, then the prompt, with what the program defined
    piped("x^2\nexit\n")
    status, out, _ = run(capsys, "-c", "x := 3", "-i")
    assert status == 0
    assert "defines x" in out
    assert "9" in out.splitlines()[-1] or "9\n" in out


@POSIX
def test_standard_input_left_open_says_what_it_waits_for():
    # AGT-09: a harness that leaves standard input open hears why nothing happens
    script = textwrap.dedent(
        """
        import sys
        from relativisticpy import cli
        cli.STDIN_SILENCE = 0.2
        sys.exit(cli.main([]))
        """
    )
    run_ = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(2)
    out, err = run_.communicate("1 + 1\n", timeout=60)
    assert "reading from standard input, to its end" in err
    assert out == "2\n"
    assert run_.returncode == 0


# -- files and the command line ---------------------------------------------------------------


def test_files_that_cannot_be_run(capsys, tmp_path):
    # CLI-23, CLI-34, ERR-42, DOC-40: what is wrong, in words, and what to do instead
    cases = {
        tmp_path / "missing.relpy": "no file",
        tmp_path: "is a folder, not a file",
    }
    latin = tmp_path / "latin.relpy"
    latin.write_bytes("\\text{é}\n".encode("latin-1"))
    cases[latin] = "is not UTF-8 text: the byte 0xE9 at line 1, column 7 is not"
    paper = tmp_path / "paper.tex"
    paper.write_text("\\documentclass{article}\n")
    cases[paper] = "is a LaTeX document, not a program"
    notebook = tmp_path / "notes.ipynb"
    notebook.write_text("{}")
    cases[notebook] = "is a Jupyter notebook"
    for path, message in cases.items():
        with pytest.raises(SystemExit) as stop:
            main([str(path)])
        assert stop.value.code == 2  # the command line was wrong
        err = capsys.readouterr().err
        assert err.startswith("relativisticpy: error: ")
        assert str(path) in err.splitlines()[0]
        assert message in err
        assert "Errno" not in err
        assert "usage:" not in err
    main_hint = {
        paper: f"relativisticpy check {paper}",
        tmp_path: f"relativisticpy check {tmp_path}",
    }
    for path, hint in main_hint.items():
        with pytest.raises(SystemExit):
            main([str(path)])
        assert hint in capsys.readouterr().err


def test_a_mistyped_command(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as stop:
        main(["chekc", "paper.tex"])
    assert stop.value.code == 2
    assert "unknown command chekc; did you mean check?" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["chek"])
    assert "relativisticpy check" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["-c", "x", "file.relpy"], "give a FILE or -c CODE, not both"),
        (["a.relpy", "b.relpy"], "a run takes one FILE, but 2 were given"),
        (["--render", "-c", "x"], "--render needs a Markdown notebook (.md), not -c CODE"),
        (["--render", "--ir", "notes.md"], "--render and --ir do not go together"),
        (["--render", "notes.md", "--format", "json"], "--render typesets results in LaTeX"),
        (["--render", "notes.md", "-k"], "--render always goes on past errors"),
        (["--ir", "-c", "x", "--latex"], "--ir prints the IR, which has one form"),
        (["--ir", "-c", "x", "-k"], "--ir only compiles, so -k does nothing"),
        (["-o", "out.md", "-c", "x"], "-o FILE writes the document of --render"),
        (["--latex", "--format", "text", "-c", "x"], "--latex is short for --format latex"),
        (["--format", "json", "-i", "-c", "x"], "--format json is for running a program"),
        (["-j", "-1", "-c", "x"], "takes a number of processes, as in -j 4, or auto"),
        (["--format", "xml", "-c", "x"], "invalid choice: 'xml'"),
    ],
)
def test_flags_that_do_not_go_together(capsys, arguments, message):
    # CLI-22, CLI-38, AGT-25: refused, never ignored
    with pytest.raises(SystemExit) as stop:
        main(arguments)
    assert stop.value.code == 2
    assert message in capsys.readouterr().err


def test_render_and_ir_need_a_program(capsys, monkeypatch):
    monkeypatch.setattr(command, "_is_terminal", lambda stream: True)  # at a terminal
    for flag in ("--render", "--ir"):
        with pytest.raises(SystemExit):
            main([flag])
        assert f"{flag} needs a program" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["-k"])
    assert "--keep-going is for running a program" in capsys.readouterr().err


def test_code_that_starts_with_a_minus_and_code_given_twice(capsys):
    # CLI-43, CLI-22
    assert run(capsys, "-c", "-x") == (0, "−x\n", "")
    _, out, _ = run(capsys, "-c", "x := 2", "-c", "x^2")
    assert out.endswith("<command line>:2:1: x^2\n  4\n")


def test_workers(capsys, monkeypatch):
    # CLI-38: 0 or auto is a process for each CPU
    made = []
    real = command.Workbook

    def workbook(*, workers=1):
        made.append(workers)
        return real(workers=workers)

    monkeypatch.setattr(command, "Workbook", workbook)
    for value, workers in (("0", None), ("auto", None), ("3", 3)):
        assert main(["-j", value, "-c", "x"]) == 0
        assert made.pop() == workers
    capsys.readouterr()


def test_version(capsys):
    # CLI-44
    for flag in ("--version", "-V"):
        with pytest.raises(SystemExit) as stop:
            main([flag])
        assert stop.value.code == 0
        out = capsys.readouterr().out
        assert out.startswith(f"relativisticpy {metadata.version('relativisticpy')} (Python ")
        assert f"SymPy {metadata.version('sympy')}" in out


def test_help_teaches_the_command(capsys):
    # CLI-14, AGT-06, AGT-20: examples, files, exit codes, where to learn more
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert out.startswith("usage: relativisticpy [options] [FILE | -c CODE | -]\n")
    assert "relativisticpy check [options] FILE..." in out.split("\n\n")[0]
    for needed in ("Examples:", "Exit status: 0", "130 interrupted", "Learn more:", ".md", "-k"):
        assert needed in out
    with pytest.raises(SystemExit):
        main(["check", "--help"])
    out = capsys.readouterr().out
    for needed in ("✓ holds", "! not parsed", "--fail-on", "Exit status:", "Examples:"):
        assert needed in out


def test_ctrl_c_stops_a_run_and_says_where(capsys, monkeypatch):
    # CLI-13: what was computed is printed; where it stopped is said; status 130
    real = Workbook.each

    def interrupted(self, source, **options):
        steps = real(self, source, **options)
        yield next(steps)
        raise KeyboardInterrupt

    monkeypatch.setattr(Workbook, "each", interrupted)
    status, out, err = run(capsys, "-c", "x := 2\nx^2")
    assert status == 130
    assert "defines x" in out
    assert err.startswith("interrupted after ")
    assert "computing <command line>:2:1, x^2; the results above are complete" in err
    assert "Traceback" not in err


def test_an_internal_error_says_it_is_one_and_the_run_goes_on(capsys, monkeypatch):
    # CLI-04, CLI-32: a failure of our own is never blamed on the input
    real = command._shown

    def broken(value, form):
        if value == 4:
            raise ValueError("cannot show this")
        return real(value, form)

    monkeypatch.setattr(command, "_shown", broken)
    status, out, err = run(capsys, "-c", "1 + 3\n2 + 3", "-k")
    assert status == 1
    assert out.endswith("<command line>:2:1: 2 + 3\n  5\n")
    assert err.startswith("internal error: RelativisticPy failed while printing this result")
    assert "please report it" in err


@POSIX
def test_output_and_errors_keep_their_order_in_a_pipe(tmp_path):
    # CLI-27: what was printed before an error comes before it in a log
    program = tmp_path / "program.relpy"
    program.write_text("x := 3\nx^2\nT_{ab} u^a\n")
    done = subprocess.run(
        [sys.executable, "-m", "relativisticpy", str(program)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
    )
    assert done.returncode == 1
    assert done.stdout.index("  9") < done.stdout.index("CompileError")


@POSIX
def test_sigterm_stops_a_run_and_its_workers(tmp_path):
    # AGT-18: as a harness stops a command that takes too long
    pids = tmp_path / "pids"
    script = tmp_path / "slow.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import os, sys, time
            import sympy
            from relativisticpy import workbook
            from relativisticpy.cli import main

            def slow(item):
                with open({str(pids)!r}, "a") as file:
                    file.write(f"{{os.getpid()}}\\n")
                time.sleep(60)
                return item

            present = workbook.Workbook._present

            def patched(self, output):
                value = present(self, output)
                if value == sympy.Symbol("slow"):
                    self._executor.map(slow, list(range(8)))
                return value

            if __name__ == "__main__":
                workbook.Workbook._present = patched
                sys.exit(main(sys.argv[1:]))
            """
        )
    )
    program = tmp_path / "program.relpy"
    program.write_text("x := 2\nx^2\nslow\nx^3\n")
    process = subprocess.Popen(
        [sys.executable, str(script), "-j", "2", str(program)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and len(_pids(pids)) < 2:
            time.sleep(0.05)
        workers = _pids(pids)
        assert len(workers) == 2
        started = time.monotonic()
        process.send_signal(signal.SIGTERM)
        out, err = process.communicate(timeout=30)
        assert time.monotonic() - started < 10  # without waiting for the workers
        assert process.returncode == 143
        assert "  4\n" in out
        assert "x^3" not in out
        assert "terminated after" in err
        assert f"{program}:3:1, slow" in err
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(_alive(pid) for pid in workers):
            time.sleep(0.05)
        assert not [pid for pid in workers if _alive(pid)]
    finally:
        process.kill()
        for pid in _pids(pids):
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)


def _pids(path):
    return [int(pid) for pid in path.read_text().split()] if path.exists() else []


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    status = f"/proc/{pid}/status"
    try:
        with open(status) as file:
            return "\nState:\tZ" not in file.read()
    except OSError:
        return True


# -- --render and --ir ----------------------------------------------------------------------

NOTEBOOK = textwrap.dedent(
    r"""
    # A notebook

    $$
    a := 2 \\
    a^2
    $$

    A block with a mistake:

    $$
    \begin{aligned}
    b &:= a + 1 \\
    b^3 \\
    \mathrm{diff}(\sin(a), a)
    \end{aligned}
    $$

    $$
    b + 1
    $$
    """
).lstrip()


def test_render_typesets_every_block_and_each_error_in_place(capsys, program):
    # CLI-01, CLI-45, DOC-18, ERR-43, AGT-17: a failing block is typeset as its error
    notes = program(NOTEBOOK, "notes.md")
    status, out, err = run(capsys, str(notes), "--render")
    assert status == 1
    blocks = out.split(RESULT_START)
    assert len(blocks) == 4
    assert "$$\n4\n$$" in blocks[1]
    assert "$$\n27\n$$" in blocks[2]
    assert "EvaluationError: cannot differentiate with respect to a number" in blocks[2]
    assert "$$\n4\n$$" in blocks[3]  # the block after the error ran
    assert out.count(RESULT_END) == 3
    assert f"--> {notes}:14:1" in err
    assert err.endswith(f"1 error, each typeset in the document where it happened: {notes}:14:1\n")


def test_rendering_twice_gives_the_same_document(capsys, program, tmp_path):
    # DOC-23, AGT-17: results are marked, and replaced rather than run again
    notes = program(NOTEBOOK, "notes.md")
    once, twice = tmp_path / "once.md", tmp_path / "twice.md"
    assert main([str(notes), "--render", "-o", str(once)]) == 1
    assert main([str(once), "--render", "-o", str(twice)]) == 1
    assert once.read_text() == twice.read_text()
    assert main([str(notes), "--render", "-o", str(notes)]) == 1  # in place
    assert main([str(notes), "--render", "-o", str(notes)]) == 1
    assert notes.read_text() == once.read_text()
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []
    capsys.readouterr()
    status, out, _ = run(capsys, str(once), "-k")  # a rendered notebook runs as it was
    assert status == 1
    assert out.count("defines") == 2
    assert out.count("  4\n") == 2


def test_render_without_errors_writes_the_document(capsys, program):
    notes = program("Prose.\n\n$$ x := 2; x^2 $$\n\nEnd.\n", "notes.md")
    status, out, err = run(capsys, str(notes), "--render")
    assert (status, err) == (0, "")
    assert (
        out == f"Prose.\n\n$$ x := 2; x^2 $$\n\n{RESULT_START}\n$$\n4\n$$\n{RESULT_END}\n\nEnd.\n"
    )


def test_render_reads_standard_input(capsys, piped):
    piped("$$ 1 + 1 $$\n")
    status, out, _ = run(capsys, "--render", "-")
    assert status == 0
    assert "$$\n2\n$$" in out


def test_ir_follows_the_blocks_of_a_notebook(capsys, program):
    # CLI-19: definitions carry from block to block, and nothing is computed
    notes = program(
        "$$ Coordinates := [t, r] $$\n\n$$ g_{ab} := diag(-1, r^2) $$\n\n$$ \\Gamma^{t}_{tt} $$\n",
        "notes.md",
    )
    status, out, err = run(capsys, str(notes), "--ir")
    assert (status, err) == (0, "")
    assert out.startswith(f"# {notes}:1\n# {notes}:3\n# {notes}:5\n")
    assert "christoffel" in out


# -- checking papers --------------------------------------------------------------------------

PAPER = textwrap.dedent(
    r"""
    \documentclass{article}
    \begin{document}
    \begin{equation} ds^2 = dr^2 + r^2 d\theta^2 \end{equation}
    \begin{equation} \Gamma^r_{\theta\theta} = -r \end{equation}
    \begin{equation} \Gamma^\theta_{r\theta} = r \end{equation}
    \end{document}
    """
).lstrip()


@pytest.fixture
def paper(tmp_path):
    path = tmp_path / "paper.tex"
    path.write_text(PAPER)
    return path


def test_check_says_what_fails_a_policy(capsys, paper):
    # CLI-07: the build fails, and says why
    status, out, _ = run(capsys, "check", str(paper), "--strict")
    assert status == 1
    assert f"{paper}:5:18: ✗ differs" in out
    assert out.rstrip().endswith(f"failed by --strict: 1 differs ({paper}:5:18)")
    assert run(capsys, "check", str(paper))[0] == 0  # no policy, no failure


def test_fail_on_takes_verdicts(capsys, paper):
    status, out, _ = run(capsys, "check", str(paper), "--fail-on", "differs,not-parsed")
    assert status == 1
    assert "failed by --fail-on differs,not-parsed: 1 differs" in out
    assert run(capsys, "check", str(paper), "--fail-on", "not-parsed")[0] == 0
    with pytest.raises(SystemExit) as stop:
        main(["check", str(paper), "--fail-on", "wrong"])
    assert stop.value.code == 2
    assert "'wrong' is not a verdict" in capsys.readouterr().err


def test_strict_fails_on_a_document_without_equations_and_on_time_limits(capsys, tmp_path):
    # CLI-20
    empty = tmp_path / "empty.tex"
    empty.write_text("\\documentclass{article}\nNo maths.\n")
    status, out, _ = run(capsys, "check", str(empty), "--strict")
    assert status == 1
    assert "failed by --strict: no equation was found" in out


def test_check_json_is_the_report_as_data(capsys, paper):
    # CLI-16, AGT-08
    status, out, _ = run(capsys, "check", str(paper), "--format", "json", "--strict")
    assert status == 1
    data = json.loads(out)
    assert data["file"] == str(paper)
    assert data["counts"] == {"holds": 1, "differs": 1, "defines": 1}
    assert data["policy"] == sorted(["differs", "ill-formed", "not parsed", "time limit", "empty"])
    assert data["failed"] is True
    assert [finding["verdict"] for finding in data["findings"]] == ["defines", "holds", "differs"]


def test_check_github_annotations(capsys, paper):
    status, out, _ = run(capsys, "check", str(paper), "--format", "github")
    assert status == 0
    [annotation] = [line for line in out.splitlines() if line.startswith("::")]
    where = f"file={paper},line=5,col=18,endLine=5,endColumn=45"
    assert annotation.startswith(f"::error {where},title=RelativisticPy%3A differs::")
    assert "%0A" in annotation  # the detail, on a line of its own


def test_check_several_documents_patterns_and_folders(capsys, tmp_path, paper):
    # CLI-35, AGT-25
    chapters = tmp_path / "chapters"
    chapters.mkdir()
    (chapters / "main.tex").write_text("\\input{one}\n$$ x := 1 $$\n")
    (chapters / "one.tex").write_text("$$ 1 + 1 = 2 $$\n")  # included: checked through main
    (chapters / "notes.md").write_text("$$ 2 + 2 = 4 $$\n")
    (chapters / "README.md").write_text("No maths here.\n")
    status, out, _ = run(capsys, "check", str(paper), str(chapters), "--strict")
    assert status == 1
    names = [
        line.split(":")[0] for line in out.splitlines() if re.match(r"\S+: \d+ equation", line)
    ]
    assert names == [str(paper), str(chapters / "main.tex"), str(chapters / "notes.md")]
    assert out.rstrip().endswith(f"3 documents checked in {out.rstrip().split(' in ')[-1]}")
    pattern = str(tmp_path / "*.tex")
    status, out, _ = run(capsys, "check", pattern)
    assert status == 0
    assert out.startswith(f"{paper}: 3 equations")


def test_check_reports_what_it_cannot_read_and_checks_the_rest(capsys, tmp_path, paper):
    # ERR-42, CLI-33
    latin = tmp_path / "latin.tex"
    latin.write_bytes(b"\\begin{equation}\n\\text{caf\xe9} = 1\n\\end{equation}\n")
    status, out, err = run(capsys, "check", str(tmp_path / "missing.tex"), str(latin), str(paper))
    assert status == 2
    assert out.startswith(f"{paper}: 3 equations")
    assert f"relativisticpy check: error: no file {tmp_path / 'missing.tex'}" in err
    assert f"relativisticpy check: error: {latin} is not UTF-8 text" in err


def test_check_reads_standard_input(capsys, piped):
    # CLI-36
    piped("\\begin{equation} 2 + 2 = 4 \\end{equation}\n")
    status, out, _ = run(capsys, "check", "-", "--all")
    assert status == 0
    assert out.startswith("<stdin>: 1 equation, 1 statement")
    assert "<stdin>:1:18: ✓ holds" in out


def test_options_before_the_command(capsys, paper):
    # AGT-25: relativisticpy -j 2 check paper.tex
    assert run(capsys, "-j", "2", "check", str(paper))[0] == 0


def test_check_lines_reads_what_comes_before_them(capsys, paper):
    # CLI-18, CLI-41: one line, with the line element above it known
    status, out, _ = run(capsys, "check", str(paper), "--lines", "5")
    assert status == 0
    assert out.startswith(f"{paper}: 1 equation, 1 statement, in line 5")
    assert f"{paper}:5:18: ✗ differs" in out
    with pytest.raises(SystemExit):
        main(["check", str(paper), str(paper), "--lines", "5"])
    assert "--lines picks lines of one document" in capsys.readouterr().err


def test_check_in_a_terminal_wraps_and_in_a_pipe_never_cuts(capsys, tmp_path, monkeypatch):
    # CLI-08, CLI-25
    long = tmp_path / "long.tex"
    terms = " + ".join(f"x_{{{n}}}" for n in range(40))
    long.write_text(f"\\begin{{equation}} {terms} = 1 + {terms} \\end{{equation}}\n")
    _, out, _ = run(capsys, "check", str(long))
    assert terms in out  # whole, in a pipe
    monkeypatch.setattr(command, "_width", lambda stream: 60)
    _, out, _ = run(capsys, "check", str(long))
    assert max(len(line) for line in out.splitlines() if "|" not in line) <= 60


def test_check_skips_the_results_render_typeset(capsys, program):
    notes = program("$$ x := 2 $$\n\n$$ x^2 $$\n", "notes.md")
    assert main([str(notes), "--render", "-o", str(notes)]) == 0
    status, out, _ = run(capsys, "check", str(notes), "--all")
    assert status == 0
    assert out.startswith(f"{notes}: 2 equations, 2 statements")


def test_the_progress_of_a_check(monkeypatch):
    # CLI-13: on a terminal, what is being checked, where, and for how long
    unicode = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    console = command._Console(unicode, unicode)
    progress = command._Progress(console, "checking")
    progress.what, progress.total, progress.started = "paper.tex", 120, time.monotonic() - 72
    progress.at("paper.tex", 91)
    assert re.fullmatch(r". checking paper\.tex, line 91 of 120 · 1 min 1[23] s", progress.line(0))
    progress.at("chapter.tex", 3)
    assert ", chapter.tex:3 · " in progress.line(1)


def test_a_progress_line_shows_in_a_terminal_and_is_cleared(monkeypatch):
    class Terminal(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr(command, "PROGRESS_AFTER", 0.0)
    out, err = Terminal(), Terminal()
    console = command._Console(out, err, "never")
    progress = command._Progress(console, "computing")
    progress.begin("x^2")
    time.sleep(0.3)
    progress.end()
    console.out("4")
    assert "computing x^2" in err.getvalue()
    assert err.getvalue().endswith("\r\x1b[K")  # cleared before anything else is written
    assert out.getvalue() == "4\n"


# -- colour ------------------------------------------------------------------------------------


def test_colour_only_where_it_is_wanted(monkeypatch):
    # CLI-26: sparingly, in a terminal, never with NO_COLOR or TERM=dumb
    class Terminal(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm")
    assert command._Style(Terminal()).enabled
    assert not command._Style(io.StringIO()).enabled  # a pipe
    assert command._Style(io.StringIO(), "always").enabled
    assert not command._Style(Terminal(), "never").enabled
    monkeypatch.setenv("NO_COLOR", "1")
    assert not command._Style(Terminal()).enabled
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("TERM", "dumb")
    assert not command._Style(Terminal()).enabled


def test_a_diagnostic_in_colour():
    style = command._Style(io.StringIO(), "always")
    text = (
        "CompileError: T is not a tensor\n --> f:1:1\n  |\n1 | T_{ab}\n  | ^^^^^^\nhint: define it"
    )
    painted = style.diagnostic(text)
    assert painted.startswith("\x1b[1;31mCompileError:\x1b[0m T is not a tensor")
    assert "\x1b[1;31m^^^^^^\x1b[0m" in painted
    assert "\x1b[1;36mhint:\x1b[0m define it" in painted
    assert re.sub(r"\x1b\[[\d;]*m", "", painted) == text


# -- help --------------------------------------------------------------------------------------


def test_help_lists_the_topics_and_shows_the_language(capsys):
    # AGT-06, CLI-15: the documentation, offline
    status, out, _ = run(capsys, "help")
    assert status == 0
    assert out.startswith("RelativisticPy reads mathematics as written")
    assert "  checking         Checking papers" in out
    assert "relativisticpy check FILE..." in out


def test_the_examples_help_shows_run():
    wb = Workbook()
    for code, _ in command.CHEAT_SHEET:
        assert all(step.error is None for step in wb.each(code)), code


def test_help_on_a_topic(capsys):
    status, out, _ = run(capsys, "help", "checking")
    assert status == 0
    assert out.startswith("# Checking papers")
    assert run(capsys, "help", "check")[1] == out  # another name for it
    status, _, err = run(capsys, "help", "tensers")
    assert status == 2
    assert "there is no help on 'tensers'" in err
    assert "did you mean tensors?" in err


def test_the_package_requires_the_sympy_it_needs():
    # Index notation needs SymPy 1.14: with 1.12 or 1.13, a dozen tests fail.
    assert "sympy>=1.14" in (metadata.requires("relativisticpy") or [])
