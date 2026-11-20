"""The Workbook: the session API."""

import re

import pytest
from sympy import Array, Symbol

from relativisticpy import Metric, Tensor, Workbook, latex
from relativisticpy.errors import ParseError
from relativisticpy.language.tokens import TokenKind

x = Symbol("x")


def test_expr_returns_the_last_value_and_exe_every_value():
    with Workbook() as wb:
        assert wb.exe("1\n2\n3") == [1, 2, 3]
        assert wb.expr("1\n2\n3") == 3
        assert wb.expr("y := 1") is None
        assert wb.run("4") == [4]


def test_definitions_persist_until_reset():
    wb = Workbook()
    wb.expr("y := 2")
    assert wb.expr("y") == 2
    wb.reset()
    assert wb.expr("y") == Symbol("y")


def test_errors_are_raised_not_returned():
    with pytest.raises(ParseError):
        Workbook().expr("x +")


def test_names_appear_in_errors():
    with pytest.raises(ParseError, match=r"notes\.relpy:1:"):
        Workbook().expr("x +", name="notes.relpy")


def test_looking_inside():
    wb = Workbook()
    assert wb.tokens("x + 1")[1].kind is TokenKind.PLUS
    assert len(wb.parse("x\ny").statements) == 2
    assert "out" in wb.ir("x^2")


def test_getitem():
    wb = Workbook()
    wb.exe("y := 3\nCoordinates := [x, z]\nT_{a} := [1, x]")
    assert wb["y"] == 3
    assert wb["T"].components == Array([1, x])
    with pytest.raises(KeyError):
        wb["nothing"]


def test_metric_and_coordinates():
    wb = Workbook()
    assert wb.metric is None
    assert wb.coordinates is None
    wb.exe(r"g_{\mu\nu} := Schwarzschild(M)")
    assert isinstance(wb.metric, Metric)
    assert wb.metric.ricci_scalar == 0
    assert [c.name for c in wb.coordinates] == ["t", "r", "theta", "phi"]


def test_tensor_results():
    wb = Workbook()
    result = wb.expr("Coordinates := [x, y]\nT_{ab} := [[x, 0], [0, 1]]\nT_{ab}")
    assert isinstance(result, Tensor)
    assert result.name == "T"
    assert result.coordinates == (x, Symbol("y"))


def test_run_file(tmp_path):
    program = tmp_path / "program.relpy"
    program.write_text("y := 2\ny^2\n")
    assert Workbook().run_file(program) == [4]
    notes = tmp_path / "notes.md"
    notes.write_text("Some prose.\n\n$$ z := 3 $$\n\nMore.\n\n$$ z^2 $$\n")
    assert Workbook().run_file(notes) == [9]


def test_notebook_errors_point_at_the_file(tmp_path):
    notes = tmp_path / "notes.md"
    notes.write_text("Prose.\n\n$$\ny := 2\n$$\n\nMore.\n\n$$\n\\farc{1}{2}\n$$\n")
    with pytest.raises(ParseError, match=re.escape(f"--> {notes}:10:1")):
        Workbook().run_file(notes)


def test_maths_in_code_is_not_run(tmp_path):
    document = "The `$$...$$` blocks run.\n\n```\n$$ 1/0 $$\n```\n\n$$ 1 + 1 $$\n"
    assert Workbook().markdown(document) == document[:-1] + "\n\n$$\n2\n$$\n"
    notes = tmp_path / "notes.md"
    notes.write_text(document)
    assert Workbook().run_file(notes) == [2]


def test_markdown_typesets_results_after_each_block():
    document = "Prose.\n\n$$\n\\frac{d}{dx} x^3\n$$\n\nThe end."
    rendered = Workbook().markdown(document)
    assert rendered.startswith(document.split("The end.")[0].rstrip())
    assert "$$\n3 x^{2}\n$$" in rendered
    assert rendered.endswith("The end.")


def test_latex():
    assert latex(x**2) == "x^{2}"
    tensor = Tensor([[1, 0], [0, 1]], "_{ab}", name="delta")
    assert latex(tensor).startswith(r"\delta_{a b}")
