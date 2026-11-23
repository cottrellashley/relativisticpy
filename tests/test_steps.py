"""The statements of a program and its steps, listed before any runs, and what came of each
step: what it defined, in words, and the step as data."""

import signal
import time

import pytest
import sympy

from relativisticpy import Workbook
from relativisticpy.errors import ParseError
from relativisticpy.language import Source, parse
from relativisticpy.language import syntax as ast
from relativisticpy.steps import (
    LEFT_UNDONE,
    REASONS,
    Defined,
    Snapshot,
    approximate,
    as_data,
    defined_between,
    in_words,
    incomplete,
    left_undone,
    snapshot,
    statements_of,
    steps_of,
    written_in,
)
from relativisticpy.tensors import Tensor


def program(text):
    return parse(Source(text, "notes.relpy"), recover=True)


class TestBeforeAProgramRuns:
    def test_a_chain_is_one_statement_of_a_step_for_each_link(self):
        first, chain = statements_of(program("x := 2\na = b = c"))
        assert first.span.text == "x := 2"
        assert [span.text for _, span in first.steps] == ["x := 2"]
        assert chain.span.text == "a = b = c"
        assert [span.text for _, span in chain.steps] == ["a = b", "b = c"]
        assert all(isinstance(step, ast.Evaluate) for step, _ in chain.steps)

    def test_a_statement_that_cannot_be_read_is_one_step_its_error(self):
        _, unread, last = statements_of(program("x := 2\n\\frca{1}{2} + y\nx + 1"))
        assert isinstance(unread.statement, ParseError)
        assert unread.span.text == "\\frca{1}{2} + y"
        assert unread.steps == ((unread.statement, unread.span),)
        assert last.span.text == "x + 1"

    def test_the_steps_are_those_each_gives(self):
        text = "x := 2\n\\frca{1}{2}\na = b = c"
        planned = [span.text for _, span in steps_of(program(text))]
        ran = [step.span.text for step in Workbook().each(text, recover=True)]
        assert planned == ran == ["x := 2", "\\frca{1}{2}", "a = b", "b = c"]


class TestWhatAStepDefined:
    def test_the_coordinates_then_each_name_in_the_order_the_workbook_lists_them(self):
        wb = Workbook()
        before = snapshot(wb)
        wb.exe(r"g_{\mu\nu} := Schwarzschild(M)")
        defined = defined_between(before, snapshot(wb))
        assert [(d.name, d.kind) for d in defined] == [
            ("t", "coordinate"),
            ("r", "coordinate"),
            ("theta", "coordinate"),
            ("phi", "coordinate"),
            ("g", "metric"),
        ]
        assert in_words(defined) == "defines the metric g_{μν} and the coordinates t, r, θ, φ"

    def test_a_name_defined_again_is_redefined(self):
        wb = Workbook()
        wb.exe("x := 1")
        before = snapshot(wb)
        wb.exe("x := 2")
        assert defined_between(before, snapshot(wb)) == [Defined("x", "value", "x", True)]
        assert in_words(defined_between(before, snapshot(wb))) == "redefines x"

    def test_nothing_is_defined_by_an_expression(self):
        wb = Workbook()
        wb.exe("x := 1")
        before = snapshot(wb)
        wb.exe("x + 1")
        assert defined_between(before, snapshot(wb)) == []

    def test_a_workbook_that_cannot_say_what_it_defines_says_nothing(self):
        class Broken:
            coordinates = None

            def names(self):
                raise ValueError("cannot list")

        assert snapshot(Broken()) == Snapshot(None, {})

    def test_spaces_and_names_in_words_in_each_form(self):
        defined = [
            Defined("A", "space", "assets", False),
            Defined("Sigma", "tensor", r"\Sigma_{ab}", False),
        ]
        assert in_words(defined) == "defines the space assets and Σ_{ab}"
        assert in_words(defined, "latex") == r"defines the space assets and \Sigma_{ab}"
        assert written_in(r"g_{\mu\nu}", "unicode") == "g_{μν}"
        assert written_in(r"g_{\mu\nu}", "text") == r"g_{\mu\nu}"


class TestAStepAsData:
    def test_where_it_is_what_it_is_and_what_came_of_it(self):
        _, step = Workbook().each(Source("x := 2\nx^2", "notes.relpy"))
        data = as_data(step)
        assert (data["file"], data["line"], data["column"]) == ("notes.relpy", 2, 1)
        assert data["statement"] == "x^2"
        assert [value["text"] for value in data["values"]] == ["4"]
        assert (data["defines"], data["notes"], data["error"]) == ([], [], None)

    def test_what_it_defined_and_its_error(self):
        defined = [Defined("x", "value", "x", False)]
        [step] = Workbook().each("R_{ab}")
        data = as_data(step, defined)
        assert data["defines"] == [{"name": "x", "kind": "value", "written": "x"}]
        assert data["error"]["message"] == "the Ricci tensor needs coordinates"
        assert data["values"] == []

    @pytest.mark.parametrize("form", ["unicode", "latex"])
    def test_a_name_that_cannot_be_shown_is_written_as_it_is(self, form):
        assert written_in("\\frac{", form) == "\\frac{"


def last(source):
    """The values of a program's last step, as data, and its notes."""
    *_, step = Workbook().each(source)
    assert step.error is None, step.error
    return as_data(step)["values"], step.notes


class TestWhetherAValueIsAnAnswer:
    """Each value says whether it is a complete answer, why not, and whether it is
    approximate, so that a program reading it, as an agent, can tell an answer from a
    value kept as written."""

    @pytest.mark.parametrize(
        ("source", "reason"),
        [
            (r"\int_0^1 \sin(\sin x) \, dx", "integral"),
            (r"\sum_{n=1}^{\infty} \frac{1}{n^n}", "sum"),
            (r"\prod_{n=2}^{\infty} \left(1 + \frac{1}{n^n}\right)", "product"),
        ],
    )
    def test_what_sympy_left_undone_is_kept_as_written_and_says_so(self, source, reason):
        [value], notes = last(source)
        assert (value["complete"], value["reasons"]) == (False, [reason])
        assert notes == [LEFT_UNDONE[reason]]

    def test_each_kind_of_operation_left_undone_has_its_reason(self):
        x = sympy.Symbol("x")
        f = sympy.Function("f")
        assert left_undone(sympy.Limit(sympy.sin(x) / x, x, 0)) == ["limit"]
        assert left_undone(sympy.Derivative(x**2, x)) == ["derivative"]
        assert left_undone(sympy.Derivative(f(x), x)) == []  # f'(x) is the answer
        integral = sympy.Integral(sympy.sin(sympy.sin(x)), (x, 0, 1))
        assert left_undone([1, sympy.Sum(1 / x**x, (x, 1, sympy.oo)) + integral]) == [
            "integral",
            "sum",
        ]
        assert set(REASONS) == {*LEFT_UNDONE, "unknown function", "time budget"}

    def test_a_tensor_with_an_integral_in_a_component_is_incomplete(self):
        x = sympy.Symbol("x")
        components = sympy.Array([sympy.Integral(sympy.sin(sympy.sin(x)), (x, 0, 1)), 0])
        assert incomplete(Tensor(components, "^a", (x, sympy.Symbol("y")))) == ["integral"]

    def test_a_call_of_a_function_the_program_does_not_know(self):
        [value], notes = last("foo(x)")
        assert value["reasons"] == ["unknown function"]
        assert notes == [
            "foo is not a function RelativisticPy knows, so foo(x) is an unknown function"
        ]
        [value], _ = last("f(x)")  # a function of the program's own, as books write one
        assert value["complete"]

    @pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="no SIGALRM here")
    def test_a_closed_form_whose_time_ran_out(self, monkeypatch):
        from relativisticpy.runtime import core

        def searching(*args, **kwargs):
            time.sleep(2)
            raise AssertionError("the budget should have stopped this")

        monkeypatch.setattr(core, "_CLOSED_FORM_BUDGET", 0.05)
        monkeypatch.setattr(core.assumptions, "summed", searching)
        [value], notes = last(r"\sum_{n=1}^{\infty} \frac{1}{n^n}")
        assert value["reasons"] == ["sum", "time budget"]
        assert "SymPy found no closed form of the sum in the 0.05 s it is given" in notes

    def test_an_answer_is_complete(self):
        [value], notes = last(r"\int_0^1 x^2 \, dx")
        assert (value["text"], value["complete"], value["reasons"]) == ("1/3", True, [])
        assert not value["approximate"]
        assert notes == []

    @pytest.mark.parametrize(
        ("source", "approximate"),
        [
            ("x := 0.5\nx^2", True),
            ("[1, 2.5]", True),
            (r"\begin{pmatrix} 1.5 & 2 \\ 3 & 4 \end{pmatrix}", True),
            ("0.5 x", True),
            (r"\frac{1}{3}", False),
            (r"\sqrt{2}", False),
            ("[1, 2]", False),
            ("solve([x + y = 1.5, x - y = 0], [x, y])", True),  # by unknown
            ("solve([x + y = 1, x - y = 0], [x, y])", False),
        ],
    )
    def test_a_value_with_a_decimal_in_it_is_approximate(self, source, approximate):
        [value], _ = last(source)
        assert value["approximate"] is approximate
        assert value["complete"]

    def test_numbers_in_arrays_are_approximate_by_their_type(self):
        numpy = pytest.importorskip("numpy")
        assert approximate(numpy.array([1.0, 2.0]))
        assert approximate(numpy.float64(1.0))
        assert approximate(1 + 2j)
        assert not approximate(numpy.array([1, 2]))
        assert approximate(numpy.array([1, 2.5], dtype=object))  # Python's numbers, held
        assert not approximate(True)
        assert not approximate("text")

    def test_a_table_or_a_series_is_approximate_by_its_numbers(self):
        pandas = pytest.importorskip("pandas")
        assert approximate(pandas.Series([1.5, 2.0]))
        assert not approximate(pandas.Series([1, 2]))
        assert approximate(pandas.DataFrame({"n": [1, 2], "x": [0.5, 1.5]}))
        assert not approximate(pandas.DataFrame({"n": [1, 2]}))
        [step] = Workbook(scope={"s": pandas.Series([1.5, 2.0])}).each(r"\texttt{s}")
        [value] = as_data(step)["values"]
        assert (value["complete"], value["approximate"]) == (True, True)
