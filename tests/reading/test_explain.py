r"""Workbook.explain: how each statement is read, and why, as the reader decided it.

Each statement's reading says how each name was resolved, each ambiguous form decided,
with why, and the notes; nothing is computed, and the workbook is left as it was.
"""

import pytest

from relativisticpy.reading.readings import Explanation, Reading
from relativisticpy.runtime import Executor


def decisions(reading: Reading, kind: str) -> list[tuple[str, str, str]]:
    return [(d.written, d.reading, d.why) for d in reading.decisions if d.kind == kind]


def names(reading: Reading) -> dict[str, str]:
    return {name.written: name.meaning for name in reading.names}


class TestWhatItReturns:
    def test_a_reading_for_each_statement(self, wb):
        explanation = wb.explain("x := 2\ny\nz = 3")
        assert isinstance(explanation, Explanation)
        assert [reading.span.text for reading in explanation] == ["x := 2", "y", "z = 3"]
        assert len(explanation) == 3
        assert explanation[1].names[0].text() == "y: a symbol"

    def test_it_computes_nothing_and_defines_nothing(self, wb, monkeypatch):
        def refused(*args, **kwargs):
            raise AssertionError("explain computed a value")

        monkeypatch.setattr(Executor, "evaluate", refused)
        explanation = wb.explain("Coordinates := [x, y]\ng_{ab} := diag(0, 1)\nc := 3\nc + R")
        assert explanation[3].error is None  # the metric's check is not run
        assert wb.coordinates is None
        assert "c" not in wb

    def test_an_error_is_part_of_the_reading_and_the_rest_is_read(self, wb):
        explanation = wb.explain("T_{ab} + v_a\nq")
        assert explanation[0].error is not None
        assert explanation[0].error.message == "T is not a tensor"
        assert names(explanation[1]) == {"q": "a symbol"}

    def test_the_text_shows_values_as_display_shows_them(self, wb):
        text = str(wb.explain(r"x := \pi" "\n" r"x + \pi"))
        assert r"\pi: a constant, π" in text
        assert "x: a value, defined on line 1" in text


class TestDefinitionOrClaim:
    def test_a_new_name_is_defined(self, wb):
        (reading,) = wb.explain("K = 2")
        assert decisions(reading, "definition or claim") == [
            ("K = ...", "a definition", "K has no meaning yet")
        ]

    def test_a_name_with_a_value_is_claimed_with_the_note(self, wb):
        wb.exe("K := 2")
        (reading,) = wb.explain("K = 3")
        (decision,) = [d for d in reading.decisions if d.kind == "definition or claim"]
        assert (decision.reading, decision.why) == ("a claim", "K already has a value")
        assert decision.note == (
            "K already has a value, so = compares it; to give K a new value, write K := ..."
        )

    def test_a_function_and_a_built_in(self, wb):
        explanation = wb.explain("f(x) = x^2\nCoordinates := [t, r]\ng_{ab} := diag(-1, 1)\nR = 0")
        assert decisions(explanation[0], "definition or claim") == [
            ("f(x) = ...", "a definition of a function", "f has no meaning yet: a function of x")
        ]
        assert decisions(explanation[3], "definition or claim") == [
            ("R = ...", "a claim", "R is the Ricci scalar")
        ]


class TestTheAmbiguities:
    def test_notation_or_components(self, wb):
        (reading,) = wb.explain(r"G_{\mu\nu} = 8\pi G T_{\mu\nu}")
        assert decisions(reading, "notation or components") == [
            (r"G_{\mu\nu} = 8\pi G T_{\mu\nu}", "index notation", "T has no components")
        ]

    @pytest.mark.parametrize(
        ("source", "reading", "why"),
        [
            ("e(x + 1)", "a product", "e is a constant, never a function"),
            ("sin(x)", "a call of the builtin sin", "sin is a builtin function"),
            ("h(x)", "a call of the unknown function h", "nothing defines h"),
        ],
    )
    def test_a_product_or_a_call(self, wb, source, reading, why):
        (read,) = wb.explain(source)
        assert decisions(read, "product or call") == [(source, reading, why)]

    def test_an_index_or_a_power(self, wb):
        wb.exe("Coordinates := [t, r]\ng_{ab} := diag(-1, 1)")
        (reading,) = wb.explain(r"R_{ab} + y_{s}^2")
        assert decisions(reading, "index or power") == [
            ("R_{ab}", "a tensor with its indices", "R is the Ricci tensor"),
            ("y_{s}^2", "the symbol y_s", "y is not a tensor, so its subscript names a symbol"),
            ("y_{s}^2", "a power", "y is not a tensor"),
        ]
        assert names(reading)["y_s"] == "a symbol"

    def test_a_run_of_letters(self, wb):
        (reading,) = wb.explain(r"\frac{2GM}{r}")
        assert decisions(reading, "letters") == [
            ("GM", "G M", "in LaTeX a run of letters that names nothing is a product")
        ]
        assert names(reading) == {"G": "a symbol", "M": "a symbol", "r": "a symbol"}

    def test_a_constant_or_a_symbol(self, wb):
        (reading,) = wb.explain(r"e^{i \pi}")
        assert [d.written for d in reading.decisions if d.kind == "constant or symbol"] == [
            "e",
            "i",
            r"\pi",
        ]

    def test_a_function_named_alone_with_its_note(self, wb):
        (reading,) = wb.explain("simplify + 1")
        assert decisions(reading, "function named alone") == [
            ("simplify", "a symbol", "simplify is a function, named without its argument")
        ]
        assert reading.notes == [
            "simplify is a function, so alone it is read as a symbol; to apply it, write "
            "simplify(...)"
        ]


class TestNamesResolved:
    def test_against_the_coordinates_the_metric_and_the_scope(self, wb):
        wb.exe(r"Coordinates := [t, r]" "\n" r"g_{ab} := diag(-(1 - 2M/r), 1)" "\n" "c := 3")
        (reading,) = wb.explain("c t + M + Ric")
        assert names(reading) == {
            "c": "a value, defined on line 3",
            "t": "a coordinate",
            "M": "a symbol the metric is written with",
            "Ric": "the Ricci scalar of the metric in force",
        }
        assert reading.names[0].value == 3
