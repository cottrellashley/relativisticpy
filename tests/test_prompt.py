"""The prompt: driven by injected input, and in a pseudo-terminal as a person has it."""

import io
import os
import re
import select
import subprocess
import sys
import textwrap
import time

import pytest
import sympy

from relativisticpy import Workbook
from relativisticpy import cli as command
from relativisticpy.cli import interact


class Session:
    """A session at the prompt, its input given line by line: a string is typed, an
    exception is raised by the read, as Ctrl-C and Ctrl-D raise theirs."""

    def __init__(self, *lines, workbook=None, **options):
        self.lines = list(lines)
        self.prompts = []
        self.out = io.StringIO()
        self.status = interact(
            workbook or Workbook(),
            read=self.read,
            out=self.out,
            err=self.out,
            banner=False,
            **options,
        )
        self.printed = self.out.getvalue()

    def read(self, prompt):
        self.prompts.append(prompt)
        if not self.lines:
            raise EOFError
        line = self.lines.pop(0)
        if isinstance(line, BaseException | type):
            raise line
        return line


def test_definitions_persist_and_brackets_continue_the_input():
    session = Session("y := [1,", "2]", "y", "x )", "Coordinates := [t, x]", "", "\\delta^{a}_{a}")
    assert session.status == 0
    assert "[1, 2]" in session.printed
    assert "ParseError" in session.printed
    assert session.printed.rstrip().endswith("2")
    assert session.prompts[:2] == ["rp> ", "... "]


def test_an_error_never_ends_the_session_and_says_where_it_was_typed():
    # CLI-28: each input has a name of its own
    session = Session("x := 5", "T_{ab} u^a", "x")
    assert " --> <input 2>:1:1" in session.printed
    assert session.printed.rstrip().endswith("5")


def test_a_failure_of_our_own_is_called_that_and_the_session_goes_on(monkeypatch):
    # CLI-04, ERR-01: an internal error, never a traceback that ends the session
    real = command._shown

    def broken(value, form):
        if value == 4:
            raise ValueError("cannot show this")
        return real(value, form)

    monkeypatch.setattr(command, "_shown", broken)
    session = Session("y := 3", "1 + 3", "y")
    assert "internal error: RelativisticPy failed while printing this result" in session.printed
    assert "Traceback" not in session.printed
    assert session.printed.rstrip().endswith("3")


def test_a_failure_outside_any_statement_is_reported_too(monkeypatch):
    def broken(source):
        raise RuntimeError("broken")

    monkeypatch.setattr(command, "_planned", broken)
    session = Session("1 + 1", "exit")
    assert "internal error: RelativisticPy failed while running this" in session.printed
    assert session.status == 0


def test_ctrl_c_clears_the_input_and_keeps_every_definition():
    # CLI-02: Ctrl-C at the prompt, even in the middle of an input, drops the input
    session = Session("x := 5", "(1 +", KeyboardInterrupt, "x^2", KeyboardInterrupt, "x")
    assert session.status == 0
    assert "25" in session.printed
    assert session.printed.rstrip().endswith("5")
    assert "ParseError" not in session.printed


def test_ctrl_c_stops_a_computation_and_keeps_every_definition(monkeypatch):
    # CLI-02, CLI-13
    present = Workbook._present

    def slow(self, output):
        value = present(self, output)
        if value == sympy.Symbol("slow"):
            raise KeyboardInterrupt
        return value

    monkeypatch.setattr(Workbook, "_present", slow)
    session = Session("y := 7", "slow", "y")
    assert re.search(
        r"interrupted: stopped computing slow after \d\.\d s; every definition is kept",
        session.printed,
    )
    assert session.printed.rstrip().endswith("7")


def test_ctrl_d_leaves_only_from_an_empty_prompt():
    # CLI-02: on an unfinished input it runs what there is, and the session goes on
    session = Session("x := 2", "(x +", EOFError, "x^2")
    assert "ParseError" in session.printed
    assert session.printed.rstrip().endswith("4")
    assert session.status == 0


def test_an_environment_keeps_the_input_open():
    # CLI-03: an aligned block is one input, read whole
    session = Session("\\begin{aligned}", "a &:= 1 \\\\", "b &:= 2", "\\end{aligned}", "a + b")
    assert session.prompts[:4] == ["rp> ", "... ", "... ", "... "]
    assert session.printed.rstrip().endswith("3")
    assert "Error" not in session.printed


@pytest.mark.parametrize(
    ("text", "unfinished"),
    [
        ("x := (1 +", True),
        ("f(x", True),
        ("\\begin{aligned} a &:= 1", True),
        ("\\begin{aligned} a &:= 1 \\end{aligned}", False),
        ("\\left( x \\right.", False),
        ("\\left. x \\right)", False),
        ("\\left\\{ x", True),
        ("$$ x", True),
        ("$$ x $$", False),
        ("\\[ x", True),
        ("\\[ x \\]", False),
        ("ds^2 = -dt^2 +", True),
        ("x :=", True),
        ("a \\\\", False),  # a row break ends a row
        ("x \\", True),  # a backslash at the end goes on, as in Python
        ("x + 1 % (", False),  # a comment counts for nothing
        ("x + % comment", True),
        ('"(" + x', False),  # nor does text in quotes
        ("[0, 1)", False),  # a half-open interval closes what it opens
        ("x)", False),
        ("x \\cdot", True),
    ],
)
def test_what_keeps_an_input_open(text, unfinished):
    assert command._unfinished(text) is unfinished


def test_help_at_the_prompt():
    # CLI-15, DOC-19
    session = Session("help", "help checking", "help nothing", "?")
    assert session.printed.count("At the prompt:") == 2
    assert "# Checking papers" in session.printed
    assert "there is no help on 'nothing'" in session.printed


def test_exit_and_quit_leave():
    # DOC-19: never read as the symbols exit and quit
    for word in ("exit", "quit", "exit()", "quit()"):
        session = Session("x := 2", word, "x")
        assert session.status == 0
        assert not session.printed.rstrip().endswith("2")


def test_who_lists_what_is_defined():
    # CLI-28
    session = Session("who", "x := 2", "f(y) := y^2", "g_{\\mu\\nu} := Schwarzschild(M)", "who")
    assert "nothing is defined yet" in session.printed
    listing = session.printed.split("defines the metric")[-1]
    assert "coordinates  t, r, θ, φ" in listing
    assert "metric       g_{μν}" in listing
    assert "functions    f(y)" in listing
    assert "values       x" in listing


def test_reset_asks_first():
    session = Session("x := 2", "reset", "n", "x", "reset", "yes", "x")
    assert "forget x? [y/N] " in session.prompts
    assert "nothing is forgotten" in session.printed
    assert "every definition is forgotten" in session.printed
    lines = session.printed.rstrip().splitlines()
    assert "2" in lines  # x was 2,
    assert lines[-1] == "x"  # then is the symbol x


def test_python_gets_a_hint_rather_than_a_result():
    # CLI-15: Python typed at the prompt is not read as mathematics
    for line in ("wb = Workbook()", ">>> x = 1", 'wb.expr("x^2")', "print(x)"):
        session = Session(line)
        assert "hint: this prompt reads the RelativisticPy language, not Python" in session.printed


def test_definitions_are_acknowledged():
    # CLI-46, DOC-19
    session = Session("x := 2", "ds^2 = -dt^2 + dx^2")
    assert "defines x" in session.printed
    assert "defines the metric g_{μν} and the coordinates t, x" in session.printed


def test_latex_and_text_at_the_prompt():
    assert Session("\\theta^2", form="latex").printed.strip() == "\\theta^{2}"
    assert Session("\\theta^2", form="text").printed.strip() == "\\theta^2"


def test_the_banner_says_how_to_get_help(monkeypatch):
    out = io.StringIO()
    interact(Workbook(), read=lambda prompt: (_ for _ in ()).throw(EOFError), out=out, err=out)
    assert "type help for help" in out.getvalue()


def test_completion():
    # CLI-28
    wb = Workbook()
    wb.exe("Schwarzy := 2")
    assert command._completions(wb, "Schw") == ["Schwarzschild", "Schwarzy"]
    assert "\\Gamma" in command._completions(wb, "\\Gam")
    assert command._completions(wb, "wh") == ["who"]


def test_where_history_is_kept(monkeypatch, tmp_path):
    monkeypatch.setenv("RELATIVISTICPY_HISTORY", str(tmp_path / "history"))
    assert command._history_path() == tmp_path / "history"
    monkeypatch.setenv("RELATIVISTICPY_HISTORY", "")
    assert command._history_path() is None
    monkeypatch.delenv("RELATIVISTICPY_HISTORY")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    if sys.platform != "win32":
        assert command._history_path() == tmp_path / "relativisticpy" / "history"


# -- in a pseudo-terminal -----------------------------------------------------------------------

pty = pytest.importorskip("pty") if sys.platform != "win32" else None
TERMINAL = pytest.mark.skipif(sys.platform == "win32", reason="a pseudo-terminal is POSIX")

#: What a shell does for the commands it starts: their terminal is their own, so that
#: Ctrl-C there interrupts them.
OWN_TERMINAL = "import fcntl, termios\nfcntl.ioctl(0, termios.TIOCSCTTY, 0)\n"

#: The command, with a statement ``slow`` that takes a minute, to be interrupted.
SCRIPT = textwrap.dedent(
    """
    import sys, time
    import sympy
    from relativisticpy import workbook
    from relativisticpy.cli import main

    present = workbook.Workbook._present

    def patched(self, output):
        value = present(self, output)
        if value == sympy.Symbol("slow"):
            time.sleep(60)
        return value

    workbook.Workbook._present = patched
    sys.exit(main(sys.argv[1:]))
    """
)


class Terminal:
    """relativisticpy in a pseudo-terminal, as a person at a keyboard has it."""

    def __init__(self, *arguments, history, colour=False):
        environment = dict(os.environ)
        environment.update(RELATIVISTICPY_HISTORY=str(history), TERM="xterm", COLUMNS="100")
        environment.pop("FORCE_COLOR", None)
        if colour:
            environment.pop("NO_COLOR", None)
        else:
            environment["NO_COLOR"] = "1"
        # Not pty.fork: a fork of a process with JAX's threads in it is refused. The
        # command makes the terminal its own, as a shell's child has it, for Ctrl-C.
        self.fd, terminal = pty.openpty()
        self.process = subprocess.Popen(
            [sys.executable, "-c", OWN_TERMINAL + SCRIPT, *arguments],
            stdin=terminal,
            stdout=terminal,
            stderr=terminal,
            env=environment,
            start_new_session=True,
        )
        os.close(terminal)
        self.output = b""
        self.seen = 0  # how much of the output was matched already

    def send(self, text):
        os.write(self.fd, text.encode())

    @property
    def text(self):
        return self.output.decode("utf-8", "replace").replace("\r\n", "\n")

    def _read(self, seconds):
        ready, _, _ = select.select([self.fd], [], [], seconds)
        if not ready:
            return True
        try:
            chunk = os.read(self.fd, 65536)
        except OSError:
            chunk = b""
        self.output += chunk
        return bool(chunk)

    def expect(self, pattern, timeout=60):
        """Wait until the output matches ``pattern``, after what was matched before."""
        deadline = time.monotonic() + timeout
        while (found := re.compile(pattern).search(self.text, self.seen)) is None:
            remaining = deadline - time.monotonic()
            nothing = remaining <= 0 or not self._read(min(remaining, 0.5))
            if nothing and re.compile(pattern).search(self.text, self.seen) is None:
                raise AssertionError(f"{pattern!r} never came in {self.text[self.seen :]!r}")
        self.seen = found.end()
        return self.text

    def wait(self, timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.process.poll()
            if status is not None:
                self._read(0.1)
                return status
            self._read(0.1)
        raise AssertionError(f"still running: {self.text!r}")

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        os.close(self.fd)


@pytest.fixture
def terminal(tmp_path):
    opened = []

    def start(*arguments, colour=False):
        session = Terminal(*arguments, history=tmp_path / "history", colour=colour)
        opened.append(session)
        return session

    yield start
    for session in opened:
        session.close()


@TERMINAL
def test_in_a_terminal_ctrl_c_clears_the_line_and_stops_a_computation(terminal):
    # CLI-02: as Python, IPython, Julia, R and ghci do
    session = terminal()
    session.expect("type help for help")
    session.send("x := 5\r")
    session.expect("defines x")
    session.expect("rp> ")
    session.send("x^2 + ")
    session.expect(r"x\^2 \+ ")
    session.send("\x03")  # half typed
    session.expect("rp> ")
    session.send("(1 +\r")
    session.expect(r"\.\.\. ")
    session.send("\x03")  # at a continuation
    session.expect("rp> ")
    session.send("slow\r")
    session.expect("computing slow")  # the progress line, after a second
    session.send("\x03")
    session.expect("interrupted: stopped computing slow after")
    session.expect("rp> ")
    session.send("x\r")
    session.expect(r"\n5\n")
    session.send("\x04")
    assert session.wait() == 0
    assert "ParseError" not in session.text


@TERMINAL
def test_in_a_terminal_ctrl_d_leaves_only_from_an_empty_prompt(terminal):
    session = terminal()
    session.expect("rp> ")
    session.send("(1 +\r")
    session.expect(r"\.\.\. ")
    session.send("\x04")  # on an unfinished input: run what there is
    session.expect("ParseError")
    session.send("2 + 2\r")
    session.expect(r"\n4\n")
    session.send("\x04")
    assert session.wait() == 0


@TERMINAL
def test_in_a_terminal_lines_pasted_at_once_are_one_input(terminal):
    # CLI-03: the README's line element, pasted over two lines
    session = terminal()
    session.expect("rp> ")
    session.send("Coordinates := [t, r, \\theta, \\phi]\r")
    session.expect("defines the coordinates")
    session.send(
        "ds^2 = -\\left(1 - \\frac{2M}{r}\\right) dt^2\n"
        "       + \\frac{dr^2}{1 - \\frac{2M}{r}} + r^2 d\\Omega^2\n"
    )
    session.expect("defines the metric")
    session.send("R_{abcd} R^{abcd}\r")
    session.expect("48M²/r⁶")
    session.send("exit\r")
    assert session.wait() == 0
    assert "Error" not in session.text


@TERMINAL
def test_in_a_terminal_history_is_kept_and_names_complete(terminal):
    # CLI-28
    first = terminal()
    first.expect("rp> ")
    first.send("x := 41\r")
    first.expect("defines x")
    first.expect("rp> ")
    first.send("\x04")
    assert first.wait() == 0
    second = terminal()
    second.expect("rp> ")
    second.send("\x1b[A\r")  # the line before, from the last session
    second.expect("defines x")
    second.expect("rp> ")
    second.send("g_{\\mu\\nu} := Schwarz\t")  # completed as it is typed
    second.expect("schild")
    second.send("(M)\r")
    second.expect("defines the metric g_{μν}")
    second.send("exit\r")
    assert second.wait() == 0


@TERMINAL
def test_in_a_terminal_colour_and_no_colour(terminal):
    # CLI-26
    session = terminal("-c", "T_{ab} u^a", colour=True)
    assert session.wait() == 1
    assert "\x1b[1;31mCompileError:\x1b[0m" in session.text
    plain = terminal("-c", "T_{ab} u^a")
    assert plain.wait() == 1
    assert "\x1b[" not in plain.text


@TERMINAL
def test_in_a_terminal_a_run_shows_its_progress_and_ctrl_c_says_where_it_stopped(
    terminal, tmp_path
):
    # CLI-13
    program = tmp_path / "program.relpy"
    program.write_text("x := 3\nx^2\nslow\nx^3\n")
    session = terminal(str(program))
    session.expect("computing program.relpy:3:1 slow")  # the file by its name alone
    session.send("\x03")
    assert session.wait() == 130
    assert "  9\n" in session.text
    assert re.search(rf"interrupted after \d\.\d s computing {program}:3:1, slow;", session.text)
    assert "x^3" not in session.text
