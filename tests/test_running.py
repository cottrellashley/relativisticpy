"""Running programs a statement at a time, as the command line does: every error in
order, the IR of a notebook's blocks together, and the files a run reads."""

import pytest
from sympy import Symbol

from relativisticpy import Workbook
from relativisticpy.errors import CompileError, FileError, ParseError
from relativisticpy.language import Source
from relativisticpy.workbook import (
    RESULT_END,
    RESULT_START,
    notebook_blocks,
    read_program,
    rendered_results,
)


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


class TestTheFilesARunReads:
    def test_a_program(self, tmp_path):
        program = tmp_path / "square.relpy"
        program.write_text("x := 3\nx^2\n", encoding="utf-8-sig")
        [source] = read_program(program)
        assert source.name == str(program)
        assert Workbook().run_file(program) == [9]

    def test_a_latex_document_is_checked_not_run(self, tmp_path):
        # CLI-34, DOC-40: a LaTeX document belongs to check
        for name, text in (("paper.tex", "$$ x = 1 $$"), ("notes.txt", "\\documentclass{article}")):
            document = tmp_path / name
            document.write_text(text)
            with pytest.raises(FileError, match="is a LaTeX document, not a program") as error:
                Workbook().run_file(document)
            assert f"relativisticpy check {document}" in error.value.hint

    def test_a_jupyter_notebook_is_refused(self, tmp_path):
        notebook = tmp_path / "notes.ipynb"
        notebook.write_text("{}")
        with pytest.raises(FileError, match="Jupyter notebook") as error:
            read_program(notebook)
        assert "jupyter nbconvert --to markdown" in error.value.hint

    def test_files_that_cannot_be_read(self, tmp_path):
        with pytest.raises(FileError, match="no file"):
            read_program(tmp_path / "missing.relpy")
        with pytest.raises(FileError, match="is a folder, not a file"):
            read_program(tmp_path)
        latin = tmp_path / "latin.relpy"
        latin.write_bytes("\\text{é}\n".encode("latin-1"))
        with pytest.raises(FileError, match="not UTF-8") as error:
            read_program(latin)
        assert "  " not in error.value.hint  # CLI-48: one space, as every hint has


class TestRenderedResults:
    """DOC-23: the results --render typesets are marked, and never run as input."""

    RENDERED = (
        "Prose.\n\n$$\nx := 2 \\\\\nx^2\n$$\n\n"
        f"{RESULT_START}\n$$\n4\n$$\n{RESULT_END}\n\nMore.\n\n$$\nx + 1\n$$\n"
    )

    def test_where_they_are(self):
        [(start, end)] = rendered_results(self.RENDERED)
        assert self.RENDERED[start:end].startswith(f"\n\n{RESULT_START}")
        assert self.RENDERED[start:end].endswith(RESULT_END)

    def test_they_are_not_run_and_lines_stay_the_files(self, tmp_path):
        blocks = notebook_blocks(self.RENDERED, "notes.md")
        assert [block.text.strip() for block in blocks] == ["x := 2 \\\\\nx^2", "x + 1"]
        [step] = Workbook().each(blocks[1])
        assert step.span.line == 17  # as the file has it, results and all
        notes = tmp_path / "notes.md"
        notes.write_text(self.RENDERED)
        assert Workbook().run_file(notes) == [4, 3]

    def test_markers_in_code_are_text(self):
        shown = f"```markdown\n$$ y $$\n\n{RESULT_START}\n$$ 1 $$\n{RESULT_END}\n```\n"
        assert rendered_results(shown) == []
