r"""The rules of index notation and spaces, as the typer words them: in the user's notation,
one phrasing for each rule, and, where it can tell, which index is wrong and how to fix
it."""

import pytest

from relativisticpy.errors import CompileError, IndexNotationError

SCHWARZSCHILD = "Coordinates := [t, r, \\theta, \\phi]\ng_{\\mu\\nu} := Schwarzschild(M)"


def error(wb, source, kind=CompileError):
    with pytest.raises(kind) as raised:
        wb.exe(source)
    return raised.value


def notes(wb, source):
    """The notes of the one step of a one-statement program."""
    (step,) = list(wb.each(source))
    assert step.error is None, step.error
    return step.notes


class TestTheUsersNotation:
    @pytest.mark.parametrize(
        ("source", "label"),
        [
            (r"R_{\mu\mu}", r"index \mu appears"),
            ("R_{μμ}", "index μ appears"),
            ("R_{mu mu}", "index mu appears"),
        ],
    )
    def test_a_label_is_written_as_the_statement_writes_it(self, wb, source, label):
        wb.exe(SCHWARZSCHILD)
        assert error(wb, source, IndexNotationError).message.startswith(label)

    def test_names_are_written_as_the_statement_writes_them(self, wb):
        wb.exe(SCHWARZSCHILD)
        assert error(wb, r"\Gamma^a_{bc}^d").message.startswith(r"\Gamma has 3 indices")
        found = error(wb, r"\eta_{\mu\nu}")
        assert found.message == r"\eta is not a tensor"
        assert found.hint == r"define it first, for example \eta_{\mu\nu} := [...]"

    def test_a_hint_writes_the_indices_where_the_user_did(self, wb):
        wb.exe(SCHWARZSCHILD)
        assert "T^{\\mu\\nu} := [...]" in error(wb, r"T^{\mu\nu}_{\ \ ;\nu}").hint
        assert (
            error(wb, r"\nabla_\mu u^\mu").hint == r"define it first, for example u^{\mu} := [...]"
        )


class TestOneRuleOnePhrasing:
    def test_an_equation_has_two_sides(self, wb):
        wb.exe(SCHWARZSCHILD + "\nV^{a} := [1, 0, 0, 0]\nW^{b} := [0, 1, 0, 0]")
        found = error(wb, "V^a = W^b", IndexNotationError)
        assert found.message == (
            "the two sides have different free indices: ^{a} on the left, ^{b} on the right"
        )

    def test_a_definition_has_two_sides(self, wb):
        wb.exe(SCHWARZSCHILD + "\nV^{a} := [1, 0, 0, 0]")
        found = error(wb, "W_{b} := V_{a}", IndexNotationError)
        assert "_{b} on the left, _{a} on the right" in found.message

    def test_an_index_three_times_is_one_rule_as_components_and_as_notation(self, wb):
        notation = error(wb, r"T_{\mu\nu} u^\nu u_\nu = 0", IndexNotationError).message
        wb.exe(SCHWARZSCHILD)
        components = error(wb, r"R_{\mu\nu} R^{\mu\nu} g_{\mu\nu}", IndexNotationError).message
        rule = "appears 3 times in one term; an index appears once, or twice to be summed"
        assert notation.endswith(rule)
        assert components.endswith(rule)

    def test_a_scalar_added_to_a_tensor_is_the_rule_of_sums(self, wb):
        wb.exe(SCHWARZSCHILD)
        found = error(wb, r"R_{\mu\nu} + 1", IndexNotationError)
        assert found.message == r"the free indices of a sum must agree: _{\mu\nu} and none"


class TestWhichIndexIsWrong:
    def test_the_one_term_whose_indices_differ(self, wb):
        # ERR-13: the Christoffel formula with \rho for \nu in its last term
        wb.exe(SCHWARZSCHILD)
        found = error(
            wb,
            r"\Gamma^{\lambda}_{\mu\nu} = \frac{1}{2} g^{\lambda\sigma}(\partial_\mu g_{\sigma\nu}"
            r" + \partial_\nu g_{\sigma\mu} - \partial_\sigma g_{\mu\rho})",
            IndexNotationError,
        )
        assert found.message == r"\rho is free in this term but in no other"
        assert found.span.text == r"\partial_\sigma g_{\mu\rho}"
        assert found.notes == [r"the other terms have _{\mu\nu\sigma} free", r"\nu is missing here"]
        assert found.hint == r"did you mean \partial_\sigma g_{\mu\nu}?"

    def test_as_notation_too(self, wb):
        found = error(wb, r"A_{\mu\nu} + B_{\mu\nu} - C_{\mu\rho} = 0", IndexNotationError)
        assert found.message == r"\rho is free in this term but in no other"

    def test_an_index_in_the_other_position(self, wb):
        wb.exe(SCHWARZSCHILD)
        found = error(wb, r"R_{\mu\nu} + G_{\mu\nu} + R_{\mu}{}^{\nu}", IndexNotationError)
        assert found.message == r"\nu is up in this term, but down in the others"

    def test_a_name_that_is_no_coordinate(self, wb):
        wb.exe(SCHWARZSCHILD)
        found = error(wb, "R_{zz}", IndexNotationError)
        assert found.notes == [
            r"z is not a coordinate, so it is an index; the coordinates are t, r, \theta and \phi"
        ]


class TestWhatAMetricIsNeededFor:
    def test_lowering_an_index_without_a_metric(self, wb):
        wb.exe("Coordinates := [t, x]\nV^{a} := [1, 2]")
        assert error(wb, "V_a").message == (
            "V is defined as V^{a}; writing V_a lowers an index, which needs a metric"
        )

    def test_the_hints_for_coordinates_and_a_metric_are_examples_that_work(self, wb):
        found = error(wb, "R_{tt}")
        assert found.message == "the Ricci tensor needs coordinates"
        assert found.hint == r"declare them first, for example Coordinates := [t, r, \theta, \phi]"
        wb.exe(r"Coordinates := [t, r, \theta, \phi]")  # the hint works
        found = error(wb, "R_{tt}")
        assert found.message == "the Ricci tensor needs a metric"
        assert found.hint == r"define one first, for example g_{\mu\nu} := diag(-1, 1, 1, 1)"
        wb.exe(r"g_{\mu\nu} := diag(-1, 1, 1, 1)")  # and so does this one
        assert wb.expr("R_{tt}") == 0

    def test_an_index_with_nowhere_to_range_says_where_it_can(self, wb):
        found = error(wb, r"U^\mu := [1, 0, 0, 0]")
        assert found.message == r"a tensor needs coordinates, or a space for the index \mu"
        assert found.notes == [
            "an index ranges over spacetime, over a space declared for it, or over the axis "
            "of data it is written on in the same statement"
        ]
        assert found.hint == r"declare coordinates first, as in Coordinates := [t, r, \theta, \phi]"
        wb.exe(r"Coordinates := [t, r, \theta, \phi]")  # the hint works
        wb.exe(r"U^\mu := [1, 0, 0, 0]")
        assert wb.expr("U^t") == 1

    def test_beside_data_an_index_is_told_to_range_over_a_space(self, wb):
        wb["w^a"] = [1, 1, 1]
        found = error(wb, "V^b := [1, 2]")
        assert found.hint == (
            r"name a space for it, as in \mathcal{A} := \mathbb{R}^{3}, and write b \in \mathcal{A}"
        )
        wb.exe("\\mathcal{A} := \\mathbb{R}^{2}\nb \\in \\mathcal{A}")  # the hint works
        wb.exe("V^b := [1, 2]")
        assert wb.expr("V^b").spaces[0].dim == 2

    def test_a_metric_dropped_with_its_coordinates(self, wb):
        wb.exe(SCHWARZSCHILD + "\nCoordinates := [t, x, y, z]")
        assert notes(wb, "R") == [
            "R is a symbol here: the metric was dropped when the coordinates changed on line 3"
        ]

    def test_the_weyl_tensor_in_two_dimensions_is_refused_where_it_is_asked_for(self, wb):
        wb.exe("Coordinates := [x, y]\ng_{ab} := [[1, 0], [0, 1]]")
        found = error(wb, "C_{abcd}")
        assert found.message == "the Weyl tensor is defined in three dimensions or more"
        assert found.span.text == "C_{abcd}"
        assert "the metric defined on line 2 is 2-dimensional" in found.notes
