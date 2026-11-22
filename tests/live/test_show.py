r"""show: a program shown in a notebook, each statement typeset with its results, its
notes and its errors in place, as relativisticpy --render shows them, but as HTML."""

import html
import re
import sys
import types

import pytest

from relativisticpy import Workbook, display
from relativisticpy.errors import ParseError, UsageError
from relativisticpy.language import Source
from relativisticpy.live import Shown, show

PROGRAM = r"""g_{\mu\nu} := Schwarzschild(M)
R_{abcd} R^{abcd}
\frca{1}{2}
K = 3
K = 4
\Gamma^{r}_{\theta\theta}"""


@pytest.fixture(scope="module")
def shown():
    return show(PROGRAM)


def test_each_statement_is_a_step_with_its_values(shown):
    assert isinstance(shown, Shown)
    assert len(shown) == 6
    assert [step.span.text for step in shown][:2] == [
        r"g_{\mu\nu} := Schwarzschild(M)",
        "R_{abcd} R^{abcd}",
    ]
    assert [display.text(value) for value in shown.values] == ["48*M^2/r^6", "False", "2*M - r"]


def test_an_error_is_in_its_place_and_later_statements_still_run(shown):
    (error,) = shown.errors
    assert isinstance(error, ParseError)
    assert shown.steps[2].error is error
    assert shown.steps[5].values
    assert not shown.ok


def test_the_html_typesets_each_statement_and_its_results(shown):
    page = shown.html()
    assert page == shown._repr_html_()
    statements = re.findall(r"<div class='rp-written'>(.*?)</div>", page)
    assert html.unescape(statements[0]) == r"\[g_{\mu\nu} := Schwarzschild(M)\]"
    values = [html.unescape(v) for v in re.findall(r"<div class='rp-value'>(.*?)</div>", page)]
    assert values == ["\\[" + display.latex(value) + "\\]" for value in shown.values]
    assert page.index("48 M^{2}") < page.index("frca") < page.index("2 M - r")


def test_what_a_statement_defines_and_its_notes_are_said(shown):
    page = html.unescape(shown.html())
    assert "defines the metric g_{μν} and the coordinates t, r, θ, φ" in page
    assert "defines K" in page
    assert "note: K already has a value, so = compares it" in page


def test_an_error_shows_as_render_gives_it_and_is_not_typeset(shown):
    page = shown.html()
    (error,) = shown.errors
    assert html.escape(error.render()) in page
    assert "<div class='rp-code'>\\frca{1}{2}</div>" in page
    assert "rp-step rp-failed" in page


def test_repr_is_the_same_as_text(shown):
    text = repr(shown)
    assert text.startswith("g_{\\mu\\nu} := Schwarzschild(M)\n  defines the metric")
    assert "R_{abcd} R^{abcd}\n  48M²/r⁶" in text
    assert "ParseError: unknown LaTeX command \\frca" in text
    assert "K = 4\n  False\n  note: K already has a value" in text


def test_a_workbook_keeps_its_definitions_from_one_call_to_the_next():
    wb = Workbook()
    show(r"g_{\mu\nu} := Schwarzschild(M)", workbook=wb)
    later = show(r"R_{abcd} R^{abcd}", workbook=wb)
    assert later.workbook is wb
    assert display.text(later.values[0]) == "48*M^2/r^6"
    alone = show(r"x := 2")
    assert "x" in alone.workbook
    assert "x" not in show("y := 3").workbook


def test_a_row_of_an_alignment_is_typeset_in_one():
    page = html.unescape(show(r"\begin{aligned} u &= 1 \\ v &= u + 1 \end{aligned}").html())
    assert r"\[\begin{aligned}u &= 1\end{aligned}\]" in page


def test_named_programs_say_where_their_errors_are():
    (error,) = show("x := (", name="cell.relpy").errors
    assert " --> cell.relpy:1:" in error.render()
    source = Source("y := )", "notes.md", (12, 1))
    assert "notes.md:12:" in show(source).errors[0].render()


def test_python_objects_are_for_a_later_version():
    with pytest.raises(UsageError, match="does not read Python objects yet") as raised:
        show(r"\hat{x} := \mathcal{F}[\texttt{signal}]", signal=[1, 2, 3])
    assert "wb['x_t'] = signal" in raised.value.hint


def test_what_it_is_given_is_checked():
    with pytest.raises(UsageError, match="show takes the program as a str, not int"):
        show(3)
    with pytest.raises(UsageError, match="workbook is a Workbook, not dict"):
        show("x", workbook={})


def test_numbers_show_as_a_table():
    np = pytest.importorskip("numpy")
    wb = Workbook()
    wb["w^a"] = np.array([0.5, 0.25])
    page = show("w^a", workbook=wb).html()
    assert "<table>" in page
    assert "rp-value" in page


def test_marimo_typesets_its_formulas(shown, monkeypatch):
    marimo = types.ModuleType("marimo")
    marimo.md = lambda text: types.SimpleNamespace(text=f"<marimo-tex>{text}</marimo-tex>")
    marimo.Html = lambda text: ("Html", text)
    monkeypatch.setitem(sys.modules, "marimo", marimo)
    kind, page = shown._display_()
    assert kind == "Html"
    assert "<marimo-tex>\\[R_{abcd} R^{abcd}\\]</marimo-tex>" in page
    assert "\\[" + display.latex(shown.values[0]) + "\\]" in page


def test_a_real_marimo_typesets_it(shown):
    pytest.importorskip("marimo", reason="marimo is a tool of the notebook tests, not a dependency")
    from marimo._output.formatting import try_format

    formatted = try_format(shown)
    assert formatted.mimetype == "text/html"
    assert formatted.data.count("marimo-tex") >= 2 * 7
