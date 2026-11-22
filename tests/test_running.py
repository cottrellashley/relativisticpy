"""Running programs a statement at a time, as the command line does: every error in
order, and the IR of a notebook's blocks together."""

import pytest
from sympy import Symbol

from relativisticpy import Workbook
from relativisticpy.errors import CompileError, ParseError
from relativisticpy.language import Source


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


class TestTheIrOfSeveralBlocks:
    """CLI-19: a notebook's blocks compile in order, in one copy of the workbook."""

    def test_a_block_sees_what_the_blocks_before_it_define(self):
        blocks = [
            Source("Coordinates := [t, r]", "notes.md", (3, 1)),
            Source("g_{ab} := diag(-1, r^2)", "notes.md", (7, 1)),
            Source("\\Gamma^{t}_{tt}", "notes.md", (11, 1)),
        ]
        listing = Workbook().ir(blocks)
        assert listing.splitlines()[:2] == ["# notes.md:3", "# notes.md:7"]
        assert "# notes.md:11" in listing
        assert "christoffel" in listing

    def test_each_node_is_listed_once(self):
        listing = Workbook().ir(["y := x^2", "y + 1", "y + 2"])
        nodes = [line.split(" ", 1)[0] for line in listing.splitlines() if line.startswith("%")]
        assert len(nodes) == len(set(nodes))
        assert listing.count("out ") == 2

    def test_nothing_is_defined(self):
        wb = Workbook()
        wb.ir(["z := 2", "z^2"])
        assert wb.expr("z") == Symbol("z")

    def test_one_program_is_as_before(self):
        assert not Workbook().ir("x^2").startswith("#")
