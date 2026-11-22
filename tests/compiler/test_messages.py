r"""What the compiler says when notation breaks a rule: in the user's notation, one
phrasing for each rule, and, where it can tell, which part is wrong and how to fix it."""

import re

import pytest

from relativisticpy.errors import CompileError, IndexNotationError

SCHWARZSCHILD = "Coordinates := [t, r, \\theta, \\phi]\ng_{\\mu\\nu} := Schwarzschild(M)"


def error(wb, source, kind=CompileError):
    with pytest.raises(kind) as raised:
        wb.exe(source)
    return raised.value


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
        assert found.message == "the free indices of a sum must agree: _{mu}_{nu} and none"


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


class TestBuiltInNames:
    def test_a_name_shared_with_ones_own_says_what_it_is(self, wb):
        wb.exe(SCHWARZSCHILD)
        found = error(wb, r"A_{\mu} + B_{\nu} = C_{\mu}", IndexNotationError)
        assert found.message == "C has 3 or 4 indices, but 1 is given"
        assert found.notes == [
            "C is a built-in name: the Christoffel connection C^{a}_{bc} or the Weyl tensor "
            "C_{abcd}"
        ]
        assert found.hint == r"to use C for a tensor of your own, define it first: C_{\mu} := [...]"
        wb.exe(r"C_{\mu} := [1, 2, 3, 4]")  # the hint works
        assert wb.expr("C_{t}") == 1

    def test_a_built_in_over_data_says_how_to_define_ones_own(self, wb):
        numpy = pytest.importorskip("numpy")
        wb["X_{tk}"] = numpy.ones((5, 2))
        found = error(wb, "G_{kl} = X_{tk} X_{tl}")
        assert "G_{kl} is the Einstein tensor, which is built in" in found.notes
        assert found.hint == "to use G for a tensor of your own, define it: G_{kl} := ..."
        wb.exe("G_{kl} := X_{tk} X_{tl}")  # the hint works
        assert wb.expr("G_{kl}").to_numpy().tolist() == [[5.0, 5.0], [5.0, 5.0]]

    def test_brackets_do_not_hide_the_number_of_indices(self, wb):
        wb.exe(SCHWARZSCHILD)
        assert error(wb, r"R_{(\mu\nu\rho)}", IndexNotationError).message == (
            "R has 2 or 4 indices, but 3 are given"
        )


class TestWhatIsNotATensor:
    def test_a_word_as_a_subscript_names_a_symbol(self, wb):
        found = error(wb, "T_{eff}")
        assert found.notes == ["in braces each letter is an index: e, f and f"]
        assert found.hint == r"to name a symbol, write T_{\rm eff} or T_{\text{eff}}"
        assert str(wb.expr(r"T_{\rm eff}")) == "T_eff"  # the hint works

    def test_a_tensor_defined_without_indices_is_not_transposed_by_a_power(self, wb):
        wb.exe(r"A := \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}")
        found = error(wb, "A^T")
        assert found.message == "A^T is read as A to the power T"
        assert re.search(r"transpose\(A\)", found.hint)
        assert wb.expr("transpose(A)").tolist() == [[1, 3], [2, 4]]  # the hint works

    @pytest.mark.parametrize(
        ("source", "message"), [("x_{}", "subscript"), ("x^{}", "superscript")]
    )
    def test_an_empty_script(self, wb, source, message):
        assert error(wb, source).message == f"empty {message}"
