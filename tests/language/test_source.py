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


def excerpt(span: Span) -> list[str]:
    """The lines of an excerpt after its header."""
    return span.render().splitlines()[2:]


def test_a_span_over_several_lines_shows_every_line():
    # G_{\mu\nu} = 8\pi T_{\mu\nu\rho} broken over two lines: the right side is shown too
    source = Source(
        "\\begin{equation}\n  G_{\\mu\\nu} =\n    8 \\pi T_{\\mu\\nu\\rho}\n\\end{equation}"
    )
    start, end = source.text.index("G_"), source.text.index("\\end") - 1
    assert excerpt(Span(source, start, end)) == [
        "2 |   G_{\\mu\\nu} =",
        "  |   ^^^^^^^^^^^^",
        "3 |     8 \\pi T_{\\mu\\nu\\rho}",
        "  |     ^^^^^^^^^^^^^^^^^^^^",
    ]


def test_a_long_span_shows_its_ends():
    source = Source("x := " + "\n".join(f"  a_{i} +" for i in range(20)) + "\n  b")
    lines = excerpt(Span(source, 5, len(source.text)))
    assert "..." in lines
    assert lines[0].startswith(" 1 | x :=   a_0 +")
    assert lines[-2:] == ["21 |   b", "   |   ^"]
    assert len(lines) == 13  # three lines at each end, each with its carets, and '...'


def test_carets_stay_under_their_text_after_wide_characters():
    # the CJK characters take two columns each, as a terminal shows them
    source = Source("\\text{日本語} + \\frac{1}{")
    start = source.text.rindex("{")
    _, carets = excerpt(Span(source, start, start + 1))
    assert carets == "  | " + " " * (len("\\text{}") + 2 * 3 + len(" + \\frac{1}")) + "^"


def test_a_combining_accent_takes_no_column():
    source = Source("ψ\u0304 + x")  # ψ with a macron above it
    _, carets = excerpt(Span(source, 5, 6))
    assert carets == "  |     ^"  # under the x, four columns along


def test_control_characters_are_shown_and_take_one_column():
    # what Python makes of "\beta" in a string that is not raw: a backspace, then eta
    line, carets = excerpt(Span(Source("x + \x08eta"), 4, 5))
    assert line == "1 | x + \u2408eta"
    assert carets == "  |     ^"


def test_labels_mark_the_error_and_related_places():
    source = Source("Coordinates := [t, r, \\theta, \\phi\ng_{\\mu\\nu} := Schwarzschild(M)")
    opener, define = source.text.index("["), source.text.rindex(":=")
    span = Span(source, opener, opener + 1).labelled(
        "this '[' is never closed", (Span(source, define, define + 2), "the next statement")
    )
    assert span.render().splitlines() == [
        " --> <input>:1:16",
        "  |",
        "1 | Coordinates := [t, r, \\theta, \\phi",
        "  |                ^ this '[' is never closed",
        "2 | g_{\\mu\\nu} := Schwarzschild(M)",
        "  |            -- the next statement",
    ]


def test_lines_between_distant_places_are_elided():
    source = Source("(\n" + "x\n" * 5 + ":=")
    span = Span(source, 0, 1).labelled(
        "opened", (Span(source, len(source.text) - 2, len(source.text)), "here")
    )
    assert excerpt(span) == ["1 | (", "  | ^ opened", "...", "7 | :=", "  | -- here"]


def test_a_related_place_in_another_source_has_its_own_header():
    definition = Source("f(x) := x^2", "a.relpy")
    use = Source("f(1, 2)", "b.relpy")
    span = Span(use, 0, 7).labelled("", (Span(definition, 0, 4), "f is defined here"))
    assert span.render().splitlines()[-4:] == [
        " ::: a.relpy:1:1",
        "  |",
        "1 | f(x) := x^2",
        "  | ---- f is defined here",
    ]


def test_labels_take_no_part_in_comparing_spans():
    source = Source("a + b")
    assert Span(source, 0, 1).labelled("here") == Span(source, 0, 1)
    assert hash(Span(source, 0, 1).labelled("here")) == hash(Span(source, 0, 1))


def test_spans_and_sources_say_where_and_what():
    source = Source("R_{ab} + g_{ab}", "notes.relpy")
    assert repr(Span(source, 9, 15)) == "Span(notes.relpy:1:10, 'g_{ab}')"
    assert repr(source) == "Source('notes.relpy', 'R_{ab} + g_{ab}')"
    assert repr(Source("x" * 100)) == f"Source('<input>', '{'x' * 39}…')"
