"""The Workbook: the session API."""

import contextlib
import os
import re
import traceback
from pathlib import Path

import pytest
from sympy import Array, Symbol

from relativisticpy import Metric, Tensor, Workbook, display, latex
from relativisticpy.errors import (
    DataError,
    EvaluationError,
    FileError,
    InternalError,
    MetricError,
    ParseError,
    RelativisticPyError,
    UnknownNameError,
    UsageError,
)
from relativisticpy.language.tokens import TokenKind
from relativisticpy.workbook import RESULT_END, RESULT_START

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


def test_a_value_that_cannot_be_shown_is_its_statements_error():
    wb = Workbook()
    first, second = wb.each("[1, 2] < [3, 4]\n1 + 1")
    assert isinstance(first.error, EvaluationError)
    assert first.error.span.text == "[1, 2] < [3, 4]"
    assert second.values == [2]  # the next statement runs
    with pytest.raises(EvaluationError, match="not defined for these values"):
        wb.exe("[1, 2] < [3, 4]")


def test_a_failure_while_showing_a_value_is_an_internal_error(monkeypatch):
    # A bug of RelativisticPy's own is called that, never blamed on the input.
    def broken(value):
        raise TypeError("cannot tidy this")

    monkeypatch.setattr("relativisticpy.workbook.tidy", broken)
    wb = Workbook()
    (step,) = wb.each("x + 1")
    assert isinstance(step.error, InternalError)
    assert step.error.span.text == "x + 1"
    assert step.error.notes == ["Python says: TypeError: cannot tidy this"]
    assert "please report it" in step.error.hint
    with pytest.raises(InternalError, match="RelativisticPy failed") as raised:
        wb.exe("x + 1")
    assert isinstance(raised.value.cause, TypeError)  # kept, for the report


def test_an_equation_whose_sides_cannot_be_computed_keeps_its_verdict(monkeypatch):
    wb = Workbook()
    evaluate, calls = wb._executor.evaluate, []

    def flaky(node):  # computes the equation, then fails on its sides
        calls.append(node)
        if len(calls) > 1:
            raise ValueError("broken")
        return evaluate(node)

    monkeypatch.setattr(wb._executor, "evaluate", flaky)
    (step,) = wb.each("x^2 = 1")
    assert step.error is None
    assert step.sides is None


def test_names_appear_in_errors():
    with pytest.raises(ParseError, match=r"notes\.relpy:1:"):
        Workbook().expr("x +", name="notes.relpy")


def test_looking_inside():
    wb = Workbook()
    assert wb.tokens("x + 1")[1].kind is TokenKind.PLUS
    assert len(wb.parse("x\ny").statements) == 2
    assert "out" in wb.ir("x^2")


def test_looking_at_the_ir_changes_nothing():
    wb = Workbook()
    assert "out" in wb.ir("y := 2\ny^2")
    assert wb.expr("y") == Symbol("y")  # the definition was not made
    wb.exe("Coordinates := [t, x]")
    wb.ir("Coordinates := [r]")
    assert [c.name for c in wb.coordinates] == ["t", "x"]


def test_getitem():
    wb = Workbook()
    wb.exe("y := 3\nCoordinates := [x, z]\nT_{a} := [1, x]")
    assert wb["y"] == 3
    assert wb["T"].components == Array([1, x])
    with pytest.raises(KeyError):
        wb["nothing"]


class TestInspecting:
    """API-16, AGT-27, DOC-41: a workbook says what it knows."""

    @pytest.fixture
    def wb(self):
        wb = Workbook()
        wb.exe(r"""
            Coordinates := [t, r, \theta, \phi]
            g_{\mu\nu} := Schwarzschild(M)
            u^{\mu} := [1/\sqrt{1 - 2M/r}, 0, 0, 0]
            \Sigma := 3
            f(y) := y^2
        """)
        return wb

    def test_in_asks_without_failing(self, wb):
        assert "Sigma" in wb
        assert r"\Sigma" in wb
        assert "u^a" in wb
        assert "nothing" not in wb
        assert 0 not in wb  # not KeyError: 0, as the old sequence protocol gave

    def test_names_list_what_is_defined(self, wb):
        names = wb.names()
        assert list(wb) == list(names)
        assert {name: definition.kind for name, definition in names.items()} == {
            "t": "coordinate",
            "r": "coordinate",
            "theta": "coordinate",
            "phi": "coordinate",
            "g": "metric",
            "u": "tensor",
            "Sigma": "value",
            "f": "function",
        }
        assert names["u"].written == r"u^{\mu}"
        assert names["f"].written == "f(y)"
        assert names["u"].to_dict()["line"] == 4

    def test_the_repr_says_what_is_defined(self, wb):
        text = repr(wb)
        assert text.startswith("<Workbook: coordinates ")
        assert r"metric g_{\mu\nu} := Schwarzschild(M)" in text
        assert r"defined u^{\mu}, \Sigma, f(y)>" in text
        assert repr(Workbook()) == "<Workbook: nothing defined>"

    def test_what_was_written_reads_back(self, wb):
        assert wb[r"\Sigma"] == wb["Σ"] == wb["Sigma"] == 3
        data = Workbook()
        data["w^a"] = [1.0, 2.0]
        assert data["w^a"].to_numpy().tolist() == [1.0, 2.0]
        assert data["w"].to_numpy().tolist() == [1.0, 2.0]

    def test_g_is_the_metric(self, wb):
        assert wb["g"] is wb.metric

    def test_a_function_can_be_called(self, wb):
        assert wb["f"](3) == 9

    def test_coordinates_are_their_symbols(self, wb):
        assert wb["theta"] == Symbol("theta")

    def test_a_missing_name_is_a_key_error_that_says_what_is_defined(self, wb):
        with pytest.raises(UnknownNameError, match="T is not defined") as error:
            wb["T"]
        assert isinstance(error.value, KeyError)
        assert isinstance(error.value, RelativisticPyError)
        assert str(error.value).startswith("T is not defined")  # no quotes, as KeyError's
        assert "u^{\\mu}" in error.value.notes[0]
        with pytest.raises(UnknownNameError, match="x \\+ 1 is not a name"):
            wb["x + 1"]

    @pytest.mark.parametrize(
        ("name", "hint"),
        [
            (r"\Gamma", r"a built-in quantity is computed by a program, as in wb.expr(r'\Gamma')"),
            ("R", "a built-in quantity is computed by a program, as in wb.expr(r'R')"),
            ("M", "M is a symbol the metric is written with, and has no value"),
            ("T", "define it first, as in wb['T'] = 1 or wb.expr(r'T := 1')"),
        ],
    )
    def test_a_missing_name_says_what_it_may_be(self, wb, name, hint):
        with pytest.raises(UnknownNameError) as error:
            wb[name]
        assert error.value.hint == hint


class TestMetric:
    def test_the_metric_is_kept_with_what_it_computed(self):
        # API-17: one Metric until the metric is redefined, so its curvature is cached
        wb = Workbook()
        wb.exe(r"g_{\mu\nu} := Schwarzschild(M)")
        assert wb.metric is wb.metric
        assert wb.metric.christoffel is wb.metric.christoffel
        assert wb.metric.components == wb.expr(r"g_{\mu\nu}").components
        wb.exe(r"g_{\mu\nu} := ReissnerNordstrom(M, Q)")
        assert wb.metric.components[0, 0].has(Symbol("Q"))
        wb.reset()
        assert wb.metric is None

    def test_a_degenerate_metric_is_a_metric_error(self):
        # the same kind of error from the language as from Python: a ValueError too
        wb = Workbook()
        with pytest.raises(MetricError, match="degenerate") as error:
            wb.expr("Coordinates := [x, y]\ng_{ab} := [[1, 1], [1, 1]]\nR")
        assert isinstance(error.value, ValueError)
        assert isinstance(error.value, EvaluationError)


class TestErrorsFromPython:
    """API-20, ERR-31: every error a user can cause from Python is a RelativisticPyError,
    and also the exception Python's own habits expect."""

    def test_a_program_that_is_not_text(self):
        with pytest.raises(UsageError, match="exe takes the program as a str, not int") as error:
            Workbook().exe(123)
        assert isinstance(error.value, TypeError)

    def test_a_backend_that_does_not_exist(self):
        with pytest.raises(UsageError, match="the backend is numpy or jax, not 'torch'") as error:
            Workbook(backend="torch")
        assert isinstance(error.value, ValueError)

    def test_a_file_that_does_not_exist(self, tmp_path):
        missing = tmp_path / "missing.relpy"
        with pytest.raises(FileError, match="no file") as error:
            Workbook().run_file(missing)
        assert isinstance(error.value, OSError)

    def test_a_gradient_with_respect_to_what_is_not_data(self):
        pytest.importorskip("jax")
        wb = Workbook()
        wb["w^a"] = [1.0, 2.0]
        with pytest.raises(UnknownNameError, match="v is not bound to data") as error:
            wb.grad("w^a w_a", "v")
        assert error.value.notes == ["the tensors bound to data are w"]

    def test_a_user_error_shows_the_callers_line_not_relativisticpys_frames(self):
        # ERR-32: the diagnostic, without the internal traceback or a chained one
        with pytest.raises(EvaluationError) as error:
            Workbook().expr("Coordinates := [x, y]\ng_{ab} := [[1, 1], [1, 1]]\nR")
        assert error.value.__cause__ is None
        assert error.value.__suppress_context__
        frames = traceback.extract_tb(error.value.__traceback__)
        assert [frame.name for frame in frames] == [
            "test_a_user_error_shows_the_callers_line_not_relativisticpys_frames",
            "call",  # the boundary of the workbook, which raises the error
        ]


class TestBindingText:
    """A string bound from Python is read as the language reads it: never as Python."""

    def test_text_is_read_as_the_language(self):
        wb = Workbook()
        wb["J"] = "x^2 + 1"
        assert wb.expr("J") == x**2 + 1

    @pytest.mark.parametrize(
        "text", ["__import__('os').getcwd()", "__import__('os').getcwd", "eval('1')"]
    )
    def test_python_in_text_runs_nothing(self, text, monkeypatch):
        calls = []
        monkeypatch.setattr(os, "getcwd", lambda: calls.append(1) or "/")
        wb = Workbook()
        with contextlib.suppress(RelativisticPyError):
            wb["J"] = text
            wb.exe("J + 1")
        assert calls == []

    def test_text_that_is_not_an_expression_says_where(self):
        with pytest.raises(DataError, match="the text bound to J is not an expression") as error:
            Workbook()["J"] = "x +"
        assert error.value.span is not None
        assert error.value.span.source.name == "<the value of J>"

    def test_what_is_neither_a_number_nor_text(self):
        with pytest.raises(DataError, match="J can be bound to a number"):
            Workbook()["J"] = {"a": 1}


def test_each_is_lazy():
    # API-22: nothing runs until a step is asked for
    wb = Workbook()
    steps = wb.each("x := 2")
    assert "x" not in wb
    assert [step.values for step in steps] == [[]]
    assert wb["x"] == 2


def test_a_step_shows_its_statement_and_what_became_of_it():
    first, second = Workbook().each("R_{ab}\nx + 1")
    assert repr(first) == "Step(line 1: R_{ab} → CompileError: the Ricci tensor needs coordinates)"
    assert repr(second) == f"Step(line 2: x + 1 → [{display.unicode(x + 1)}])"


def test_check_reads_latex_text_as_a_document():
    # API-15, DOC-41: the text of a document is not a path
    report = Workbook().check(r"\begin{equation} 1 + 1 = 2 \end{equation}")
    assert [finding.verdict for finding in report] == ["holds"]
    report = Workbook().check("$$ x := 2 $$\n$$ x^2 = 4 $$")
    assert [finding.verdict for finding in report] == ["defines", "holds"]
    with pytest.raises(FileError, match=r"no file notes\.tex") as error:
        Workbook().check("notes.tex")
    assert "pass the text, or a Document" in error.value.hint


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
    program.write_text("y := 2\ny^3\n", encoding="utf-8-sig")  # as some editors save it
    assert Workbook().run_file(program) == [8]
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
    result = f"\n\n{RESULT_START}\n$$\n2\n$$\n{RESULT_END}\n"
    assert Workbook().markdown(document) == document[:-1] + result
    notes = tmp_path / "notes.md"
    notes.write_text(document)
    assert Workbook().run_file(notes) == [2]


def test_markdown_typesets_results_after_each_block():
    document = "Prose.\n\n$$\n\\frac{d}{dx} x^3\n$$\n\nThe end."
    rendered = Workbook().markdown(document)
    assert rendered.startswith(document.split("The end.")[0].rstrip())
    assert "$$\n3 x^{2}\n$$" in rendered
    assert rendered.endswith("The end.")


def test_markdown_renders_as_the_command_line_does():
    document = "$$\n\\farc{1}{2}\n$$\n\nThen:\n\n$$\nx := 3\nx^2\n$$\n"
    rendered = Workbook().markdown(document)
    assert "ParseError: unknown LaTeX command \\farc" in rendered  # typeset, and on it goes
    assert "$$\n9\n$$" in rendered
    assert Workbook().markdown(rendered) == rendered  # its results replaced, not run
    with pytest.raises(UsageError, match="takes the document as a str"):
        Workbook().markdown(Path("notes.md"))


def test_latex():
    assert latex(x**2) == "x^{2}"
    tensor = Tensor([[1, 0], [0, 1]], "_{ab}", name="delta")
    lines = [r"\begin{aligned}", r"\delta_{0\,0} &= 1 \\", r"\delta_{1\,1} &= 1", r"\end{aligned}"]
    assert latex(tensor) == "\n".join(lines)


def test_a_copy_defines_apart_from_its_workbook():
    wb = Workbook()
    wb.exe(r"f(x) := x + a; a := 1; g_{\mu\nu} := Schwarzschild(M)")
    trial = wb.copy()
    assert trial.exe("a := 5; f(0)") == [5]  # a function sees the names of its own copy
    assert wb.expr("f(0)") == 1
    trial.expr(r"g_{\mu\nu} := Minkowski()")
    assert trial.expr(r"R_{abcd} R^{abcd}") == 0
    assert str(wb.expr(r"R_{abcd} R^{abcd}")) == "48*M**2/r**6"
    assert "a" in trial
    assert wb["a"] == 1


def test_a_workbook_counts_what_may_change_it():
    wb = Workbook()
    revisions = [wb._revision]
    changes = (
        lambda: wb.expr("x := 2"),
        lambda: wb.__setitem__("y", 3),
        lambda: wb.space("assets", 2),
        wb.reset,
    )
    for change in changes:
        change()
        revisions.append(wb._revision)
    assert revisions == sorted(set(revisions))  # every change counted, and never undone
