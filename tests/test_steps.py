"""The statements of a program and its steps, listed before any runs, and what came of each
step: what it defined, in words, and the step as data."""

import pytest

from relativisticpy import Workbook
from relativisticpy.errors import ParseError
from relativisticpy.language import Source, parse
from relativisticpy.language import syntax as ast
from relativisticpy.steps import (
    Defined,
    Snapshot,
    as_data,
    defined_between,
    in_words,
    snapshot,
    statements_of,
    steps_of,
    written_in,
)


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
