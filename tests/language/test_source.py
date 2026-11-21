"""Sources and spans: where program text came from, and how errors show it."""

from relativisticpy.language import Source, Span
from relativisticpy.language.source import Piece


def test_positions_are_one_based():
    source = Source("ab\ncd")
    assert source.location(0) == (1, 1)
    assert source.location(4) == (2, 2)
    assert source.line_text(2) == "cd"


def test_a_source_from_inside_a_file_reports_the_file_s_positions():
    # " x +\n  y" as it appears from line 3, column 3 of a file: after "$$"
    source = Source(" x +\n  y", "notes.md", origin=(3, 3))
    assert source.location(1) == (3, 4)
    assert source.location(7) == (4, 3)
    assert source.line_text(3) == "   x +"
    assert Span(source, 7, 8).render() == " --> notes.md:4:3\n  |\n4 |   y\n  |   ^"


def test_carets_stay_under_their_text_after_tabs():
    rendered = Span(Source("\tx :=\t1 @ 2"), 8, 9).render()
    line, caret = rendered.splitlines()[2:4]
    assert line.index("@") == caret.index("^")
    assert "\t" not in line


def test_a_source_made_by_expansion_reports_the_file_s_positions():
    # "\half x" as written, read as "\frac{1}{2} x"
    pieces = (Piece(0, 0, 5, copied=False), Piece(11, 5, 7, copied=True))
    source = Source(r"\frac{1}{2} x", written=r"\half x", pieces=pieces)
    assert source.location(3) == (1, 1)  # inside the expansion: where \half is
    assert source.location(12) == (1, 7)  # x
    assert source.written_range(0, 11) == (0, 5)
    assert Span(source, 6, 13).render().splitlines()[2:] == [r"1 | \half x", "  | ^^^^^^^"]
