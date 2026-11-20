"""Sources and spans: where program text came from, and how errors show it."""

from relativisticpy.language import Source, Span


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
