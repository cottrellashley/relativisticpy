r"""A relation on names nothing defines, standing alone, is an assumption for the rest of the
program (design/notation.md, decision 2): ``a > 0``, ``0 < x < 1``, ``a, b > 0`` and
``x \in \mathbb{R}``. What SymPy keeps on a symbol is put on it; the bounds it cannot keep are
kept in the session; and the reading says so, in a note.
"""

import pytest
import sympy

from relativisticpy import Document, Verdict, Workbook
from relativisticpy.errors import CompileError
from relativisticpy.language import parse, to_sexpr

a, b, x, y, r = sympy.symbols("a b x y r")


def reading(wb, source):
    """The one decision of kind "assumption or claim" a statement is read with, and its
    notes."""
    (read,) = wb.explain(source)
    decisions = [d for d in read.decisions if d.kind == "assumption or claim"]
    return decisions, read.notes


def facts(wb, name):
    """What the session assumes of a name, as SymPy was given it."""
    return wb._session.assumptions[name]._assumptions_orig


class TestWhatIsAnAssumption:
    def test_a_name_bounded_by_a_constant(self, wb):
        decisions, notes = reading(wb, "a > 0")
        assert [(d.written, d.reading) for d in decisions] == [("a > 0", "an assumption")]
        assert notes == ["a > 0 is an assumption: from here on, a is positive"]
        assert wb.exe("a > 0") == []  # it shows nothing
        assert wb.expr(r"\sqrt{a^2}") == sympy.Symbol("a", positive=True)

    def test_each_form_says_what_the_name_is(self, wb):
        said = {
            r"x \geq 0": {"nonnegative": True, "real": True},
            r"y < 0": {"negative": True, "real": True},
            r"z \neq 0": {"nonzero": True},
            r"t \in \mathbb{R}": {"real": True},
            r"n \in \mathbb{Z}": {"integer": True},
            r"m \in \mathbb{N}": {"integer": True, "nonnegative": True},
            r"k \in \mathbb{Z}_{>0}": {"integer": True, "positive": True, "real": True},
            r"l \in \mathbb{N}_0": {"integer": True, "nonnegative": True},
            r"q \in \mathbb{Q}": {"rational": True},
            r"w \in \mathbb{C}": {"complex": True},
            r"u \in \mathbb{R}^+": {"real": True, "positive": True},
            r"v \in (0, 1)": {"positive": True, "real": True},
        }
        for source, expected in said.items():
            wb.exe(source)
            name = source.split()[0]
            assert facts(wb, name) == expected, source

    def test_a_list_of_names_shares_the_relation(self, wb):
        assert to_sexpr(parse("a, b > 0").statements[0]) == "(> (tuple a b) 0)"
        _, notes = reading(wb, r"M, \rho_0, k_\mathrm{B} > 0")
        assert notes == [
            r"M, \rho_0, k_\mathrm{B} > 0 is an assumption: from here on, M, ρ₀ and k_B are "
            "positive"
        ]
        wb.exe(r"\theta, \phi \in \mathbb{R}")
        assert facts(wb, "theta") == facts(wb, "phi") == {"real": True}

    def test_a_bound_sympy_cannot_keep_is_kept_in_the_session(self, wb):
        _, notes = reading(wb, "0 < x < 1")
        assert notes == ["0 < x < 1 is an assumption: from here on, x is positive and x < 1"]
        wb.exe("0 < x < 1")
        assert facts(wb, "x") == {"positive": True, "real": True}
        assert [bound for bound, _ in wb._session.bounds] == [x < 1]

    def test_a_chain_bounds_each_name_by_the_nearest_constant(self, wb):
        wb.exe("a > r > 0")
        assert facts(wb, "a") == facts(wb, "r") == {"positive": True, "real": True}
        assert [bound for bound, _ in wb._session.bounds] == [a > r]
        wb.exe(r"-\pi < \omega < \pi")
        omega = sympy.Symbol("omega")
        assert facts(wb, "omega") == {"real": True}
        assert [bound for bound, _ in wb._session.bounds][1:] == [
            omega > -sympy.pi,
            omega < sympy.pi,
        ]

    def test_what_is_no_assumption_is_read_as_before(self, wb):
        t_h, t_c = sympy.symbols("T_h T_c")
        assert wb.expr("T_h > T_c") == (t_h > t_c)  # names alone: a condition
        assert wb.expr(r"k_B T > h c B") is not None
        wb.exe("y := 2")
        assert wb.expr("y > 0") is sympy.true  # a name with a value: a claim, decided
        assert reading(wb, "x^2 > 0")[0] == []  # more than a name
        assert "x" not in wb._session.assumptions


class TestMemberships:
    def test_a_name_in_a_set_of_numbers_is_no_space(self, wb):
        wb.exe(r"\omega \in \mathbb{R}")
        assert wb.expr(r"|e^{i\omega}|") == 1

    def test_index_labels_still_range_over_a_declared_space(self, wb):
        wb.exe(r"\mathcal{A} := \mathbb{R}^{3}" "\n" r"a, b \in \mathcal{A}")
        assert "a" in wb._session.labels
        assert "a" not in wb._session.assumptions

    def test_a_name_in_a_space_of_vectors_is_read_as_before(self, wb):
        with pytest.raises(CompileError, match=r"says what x is, which is read but not computed"):
            wb.exe(r"x \in \mathbb{R}^3")

    def test_what_is_not_a_name_says_so(self, wb):
        with pytest.raises(CompileError) as raised:
            wb.exe(r"\omega(V) + b \in \mathbb{R}")
        assert raised.value.message == (
            r"\omega(V) + b \in \mathbb{R} says what \omega(V) + b is, which is read but not "
            "computed"
        )
        assert raised.value.hint == (
            r"of a name nothing defines, as in x \in \mathbb{R}, it is an assumption about the "
            "name from then on"
        )


class TestFromThenOn:
    def test_what_was_defined_takes_the_assumption_and_what_was_shown_stays(self, wb):
        shown = wb.exe(r"y := a^2" "\n" r"\sqrt{y}")
        assert shown == [sympy.sqrt(a**2)]
        _, notes = reading(wb, "a > 0")
        assert notes[1] == (
            "a was used before: what was defined with it takes the assumption from here on, "
            "and what was shown stays as it was"
        )
        assert wb.exe("a > 0\n\\sqrt{y}\ny - a^2") == [sympy.Symbol("a", positive=True), 0]

    def test_an_integral_defined_before_is_computed_under_it(self, wb):
        wb.exe(r"F := \int_0^\infty e^{-a x} dx" "\n" "a > 0")
        assert wb.expr("F - 1/a") == 0

    def test_the_metric_takes_it(self, wb):
        wb.exe(r"Coordinates := [t, x]" "\n" r"g_{\mu\nu} := diag(-a^2, 1)")
        assert wb.expr(r"\sqrt{-g_{tt}}") == sympy.sqrt(a**2)
        wb.exe("a > 0")
        assert wb.expr(r"\sqrt{-g_{tt}}") == sympy.Symbol("a", positive=True)

    def test_a_coordinate_keeps_its_symbol_and_the_bound_is_kept(self, wb):
        wb.exe("Coordinates := [t, r]")
        _, notes = reading(wb, r"0 \leq r < +\infty")
        assert notes == [
            r"0 \leq r < +\infty is an assumption: from here on, r is zero or positive",
            "r is a coordinate, whose symbol stays as it is: what is assumed of it is used "
            "where it decides something, as by solve",
        ]
        wb.exe(r"0 \leq r < +\infty")
        assert wb.coordinates == sympy.symbols("t r")
        assert [bound for bound, _ in wb._session.bounds] == [r >= 0]

    def test_an_assumption_before_the_coordinates_is_theirs(self, wb):
        wb.exe("r > 0\nCoordinates := [t, r]")
        assert wb.coordinates[1] == sympy.Symbol("r", positive=True)

    def test_a_contradiction_is_refused(self, wb):
        wb.exe("a > 0")
        with pytest.raises(CompileError, match="a < 0 contradicts what was assumed before"):
            wb.exe("a < 0")
        assert facts(wb, "a") == {"positive": True, "real": True}

    def test_assumptions_add_up(self, wb):
        wb.exe(r"n \in \mathbb{Z}" "\n" r"n \geq 0")
        assert facts(wb, "n") == {"integer": True, "nonnegative": True, "real": True}
        assert wb.expr("(-1)^{2n}") == 1

    def test_explain_leaves_the_workbook_as_it_was(self, wb):
        wb.explain("a > 0")
        assert "a" not in wb._session.assumptions


class TestInAPaper:
    def test_a_range_of_coordinates_defines(self):
        document = Document(
            r"\begin{equation} ds^2 = -dt^2 + dr^2 \end{equation}"
            r"\begin{equation} 0 \leq r < +\infty \end{equation}"
        )
        report = Workbook().check(document)
        assert [finding.verdict for finding in report.findings] == [Verdict.DEFINES] * 2

    def test_a_list_of_defined_names_is_a_claim_about_each(self, wb):
        wb.exe("a := 1\nb := 2")
        assert wb.expr("a, b > 0") is sympy.true
