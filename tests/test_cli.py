"""The relativisticpy command."""

import io

import pytest

from relativisticpy import Tensor, Workbook
from relativisticpy.cli import display, interact, main


def test_code(capsys):
    assert main(["-c", "x := 3\nx^2"]) == 0
    assert capsys.readouterr().out == "9\n"


def test_file(tmp_path, capsys):
    program = tmp_path / "flat.relpy"
    program.write_text(r"g_{\mu\nu} := Minkowski()" "\n" r"R_{\mu\nu}" "\n")
    assert main([str(program)]) == 0
    assert capsys.readouterr().out == "R_{mu}_{nu} = 0\n"


def test_markdown(tmp_path, capsys):
    notes = tmp_path / "notes.md"
    notes.write_text("Prose $$ y := 2 $$ and $$ y + 1 $$.\n")
    assert main([str(notes)]) == 0
    assert capsys.readouterr().out == "3\n"
    assert main([str(notes), "--render"]) == 0
    assert "$$\n3\n$$" in capsys.readouterr().out


def test_notebook_errors_point_at_the_file(tmp_path, capsys):
    notes = tmp_path / "notes.md"
    notes.write_text("Prose.\n\n$$\ny := 2\n$$\n\nMore.\n\n$$\n\\farc{1}{2}\n$$\n")
    assert main([str(notes)]) == 1
    assert f"--> {notes}:10:1" in capsys.readouterr().err


def test_latex_and_ir(capsys):
    assert main(["-c", r"\frac{1}{x}", "--latex"]) == 0
    assert capsys.readouterr().out == "\\frac{1}{x}\n"
    assert main(["-c", "x^2", "--ir"]) == 0
    assert "pow" in capsys.readouterr().out


def test_errors_exit_with_one(capsys, tmp_path):
    assert main(["-c", "x +"]) == 1
    assert "ParseError" in capsys.readouterr().err
    assert main([str(tmp_path / "missing.relpy")]) == 1
    assert "error:" in capsys.readouterr().err


def test_a_file_saved_with_a_byte_order_mark(tmp_path, capsys):
    program = tmp_path / "square.relpy"
    program.write_text("x := 3\nx^2\n", encoding="utf-8-sig")
    assert main([str(program)]) == 0
    assert capsys.readouterr().out == "9\n"


def test_a_file_that_is_not_utf8(tmp_path, capsys):
    program = tmp_path / "latin.relpy"
    program.write_bytes("\\text{é}\n".encode("latin-1"))
    assert main([str(program)]) == 1
    assert f"error: {program} is not UTF-8 text" in capsys.readouterr().err


def test_help_mentions_check(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "relativisticpy check FILE" in capsys.readouterr().out


def test_ctrl_c_stops_a_run_quietly(monkeypatch, capsys):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(Workbook, "exe", interrupted)
    assert main(["-c", "x^2"]) == 130
    assert "Traceback" not in capsys.readouterr().err


def test_ctrl_c_leaves_the_interactive_session(monkeypatch):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    out = io.StringIO()
    assert interact(Workbook(), read=interrupted, out=out) == 130  # at the prompt
    monkeypatch.setattr("sys.stdin", io.StringIO("x^2\n"))
    monkeypatch.setattr(Workbook, "exe", interrupted)
    assert main([]) == 130  # while computing


def test_display():
    tensor = Tensor([[1, 0], [0, 2]], "_{ab}", name="h")
    assert display(tensor) == "h_{a}_{b}:\n  [0, 0] = 1\n  [1, 1] = 2"
    assert display(Tensor([[0, 0], [0, 0]], "_{ab}")) == "T_{a}_{b} = 0"
    assert display(3) == "3"


def test_interactive_session():
    lines = iter(["y := [1,", "2]", "y", "x )", "Coordinates := [t, x]", "", "\\delta^{a}_{a}"])

    def read(prompt):
        try:
            return next(lines)
        except StopIteration:
            raise EOFError from None

    out = io.StringIO()
    assert interact(Workbook(), read=read, out=out) == 0
    printed = out.getvalue()
    assert "[1, 2]" in printed
    assert "ParseError" in printed
    assert printed.rstrip().endswith("2")


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert capsys.readouterr().out.startswith("relativisticpy ")
