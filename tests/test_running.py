"""Running programs a statement at a time, as the command line does: every error in
order."""

import pytest

from relativisticpy import Workbook
from relativisticpy.errors import CompileError, ParseError


class TestEveryErrorInOrder:
    """CLI-12: each error is reported where it is, parse errors among the others."""

    PROGRAM = "a := 1\nT_{ab} u^a\nb := 2\n\\frca{1}{2}\nS_{ab} v^b\na + b"

    def test_a_statement_that_cannot_be_read_is_a_step_in_its_place(self):
        steps = list(Workbook().each(self.PROGRAM, recover=True))
        kinds = [type(step.error).__name__ if step.error else step.values for step in steps]
        assert kinds == [[], "CompileError", [], "ParseError", "CompileError", [3]]
        assert [step.span.line for step in steps] == [1, 2, 3, 4, 5, 6]
        unread = steps[3]
        assert unread.span.text == "\\frca{1}{2}"  # the statement, for its label
        assert unread.error.span.text == "\\frca"  # the error, where it is
        assert unread.error.hint == "did you mean \\frac?"

    def test_without_recover_a_parse_error_is_raised_before_anything_runs(self):
        wb = Workbook()
        with pytest.raises(ParseError, match="frca"):
            list(wb.each(self.PROGRAM))
        assert "a" not in wb

    @pytest.mark.parametrize(
        ("program", "unread"),
        [
            ("a := (1 + 2\nb := 3", ["a := (1 + 2"]),
            ("x := 3; y +; z := 2", ["y +"]),
            ("x := 3\ny +\n", ["y +"]),
            (
                "a := 1\nb := (2\nc := a + \\frca{1}{2}\nd := 4",
                ["b := (2", "c := a + \\frca{1}{2}"],
            ),
            ("x := 1\n% a comment\ny +", ["y +"]),
            ("\\begin{aligned}\na &:= 1 \\\\\nb &:= \\frca{2}\n\\end{aligned}\nc := 3", None),
        ],
    )
    def test_the_span_of_a_statement_that_cannot_be_read(self, program, unread):
        steps = list(Workbook().each(program, recover=True))
        found = [step.span.text for step in steps if isinstance(step.error, ParseError)]
        if unread is None:  # an environment: all of it, from \begin to \end
            assert found == [program.rsplit("\n", 1)[0]]
        else:
            assert found == unread
        assert all(step.error is None for step in steps if step.span.text not in found)

    def test_the_steps_of_a_chain_are_its_links(self):
        steps = list(Workbook().each("1 = 2 = 1\n\\frca", recover=True))
        assert [step.span.text for step in steps] == ["1 = 2", "2 = 1", "\\frca"]

    def test_an_error_in_a_definition_still_lets_the_rest_run(self):
        steps = list(Workbook().each("T_{ab} u^a\nx := 2\nx^2", recover=True))
        assert isinstance(steps[0].error, CompileError)
        assert steps[2].values == [4]
