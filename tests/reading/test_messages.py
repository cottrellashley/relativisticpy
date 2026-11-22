r"""What the reader says of names and forms it cannot read as written, or reads in the less
likely way: built-in names, what is not a tensor, operators and functions, powers of
tensors and names never defined."""

import re

import pytest

from relativisticpy.errors import CompileError, IndexNotationError, ParseError

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


class TestBuiltInNames:
    SPACETIMES = (
        "the spacetimes are Minkowski, Schwarzschild, ReissnerNordstrom, Kerr, KerrNewman, "
        "FLRW, DeSitter and AntiDeSitter"
    )

    @pytest.mark.parametrize(
        ("written", "hint", "meant"),
        [
            ("Schwarzchild(M)", "did you mean Schwarzschild(M)?", "Schwarzschild(M)"),
            (
                "schwarzschild(M)",
                "in the language, spacetimes are capitalised: Schwarzschild(M)",
                "Schwarzschild(M)",
            ),
            ("kerr(M, a)", "in the language, spacetimes are capitalised: Kerr(M, a)", "Kerr(M, a)"),
        ],
    )
    def test_a_misspelt_spacetime_is_refused_with_the_closest(self, wb, written, hint, meant):
        found = error(wb, rf"g_{{\mu\nu}} := {written}")
        assert found.message == f"unknown spacetime {written.split('(')[0]}"
        assert found.notes == [self.SPACETIMES]
        assert found.hint == hint
        assert meant in hint
        wb.exe(rf"g_{{\mu\nu}} := {meant}")  # the hint works
        assert wb.expr("g_{tt}") != 0

    def test_a_spacetime_named_without_being_called(self, wb):
        found = error(wb, r"g_{\mu\nu} := Schwarzschild")
        assert found.message == "Schwarzschild is a spacetime: call it, as in Schwarzschild()"
        wb.exe(r"g_{\mu\nu} := Schwarzschild()")  # the example works
        assert wb.expr("Kretschmann") == 48 * __import__("sympy").Symbol("M") ** 2 / (
            __import__("sympy").Symbol("r") ** 6
        )

    def test_a_name_shared_with_ones_own_says_what_it_is(self, wb):
        wb.exe(SCHWARZSCHILD)
        found = error(wb, r"A_{\mu} + B_{\nu} = C_{\mu}", IndexNotationError)
        assert found.message == "C has 3 or 4 indices, but 1 is given"
        assert found.notes[0] == (
            "C is a built-in name: the Christoffel connection C^{a}_{bc} or the Weyl tensor "
            "C_{abcd}"
        )
        assert r"A is not defined, so A_{\mu} is one symbol" in found.notes[1]  # read before
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
        # in plain text an empty script is a mistake, said as it is read
        assert error(wb, source, ParseError).message == f"empty {message}"

    def test_an_empty_script_in_latex_is_nothing_as_in_tex(self, wb):
        # TeX sets \partial_{} as \partial, so the derivative still needs its variable
        assert error(wb, r"\partial_{} f").message == r"\partial needs an index or a variable"


class TestOperatorsAndFunctions:
    def test_a_function_named_alone(self, wb):
        wb.exe("f(x) := x^2")
        assert error(wb, "f + 1").message == "f is a function, so it needs an argument, as in f(x)"
        assert error(wb, "2 f").message == "f is a function, so it needs an argument, as in f(x)"

    def test_a_derivative_operator_alone(self, wb):
        wb.exe(SCHWARZSCHILD)
        assert error(wb, r"\partial_\mu").message == (
            r"\partial_\mu needs something to act on, as in \partial_\mu f"
        )

    @pytest.mark.parametrize(("source", "star"), [("x + * y", "*"), ("2 * * 3", "*")])
    def test_an_operator_written_twice_is_no_hodge_dual(self, wb, source, star):
        found = error(wb, source, ParseError)
        assert found.message == f"'{star}' has nothing on its left"
        assert found.hint == "remove it, or write what it multiplies"
        assert error(wb, r"\cdot x", ParseError).message == r"'\cdot' has nothing on its left"
        assert error(wb, "*F").message == "the Hodge dual is not supported yet"

    def test_what_is_written_as_written(self, wb):
        assert error(wb, r"a + \cdots + b").message.startswith(r"\cdots stands for terms")
        assert error(wb, r"\square f").message == r"\square needs coordinates"
        wrong = error(wb, r"\mathcal{A} := \mathbb{R}^{3.5}").message
        assert wrong == "the dimension of a space is a whole number, not 3.5"

    def test_a_statistic_of_two_expressions_suggests_a_covariance(self, wb):
        numpy = pytest.importorskip("numpy")
        wb["X_{ta}"] = numpy.ones((4, 2))
        found = error(wb, r"\mathbb{E}_t[X_{ta}, X_{tb}]")
        assert found.message == r"\mathbb{E}_t takes one expression, but 2 are given"
        assert found.hint == r"for a covariance, write \operatorname{Cov}_t(X_{ta}, X_{tb})"
        assert wb.expr(r"\operatorname{Cov}_t(X_{ta}, X_{tb})")  # the hint works

    def test_a_misspelt_statistic(self, wb):
        found = error(wb, r"\operatorname{E}_t(x)")
        assert found.hint == r"did you mean \mathbb{E}_t?"
        assert error(wb, r"\operatorname{Covv}_t(x)").hint == r"did you mean \operatorname{Cov}_t?"
        assert "is read as a symbol, Corr_t" in notes(wb, r"\operatorname{Corr}_t(x)")[0]

    def test_membership_of_an_undeclared_space(self, wb):
        found = error(wb, r"a \in \mathcal{B}")
        assert found.message == (
            r"a \in \mathcal{B} is read but not computed: \mathcal{B} is not a declared space"
        )
        assert found.hint == (
            r"to have a range over a space, declare it first, as in \mathcal{B} := \mathbb{R}^{3}"
        )
        wb.exe(r"\mathcal{B} := \mathbb{R}^{3}")  # the hint works
        wb.exe(r"a \in \mathcal{B}")
        assert error(wb, r"x \in \mathbb{R}^3").message == (
            r"x \in \mathbb{R}^3 says what x is, which is read but not computed"
        )


class TestPowersOfTensors:
    def test_the_inverse_metric(self, wb):
        wb.exe(SCHWARZSCHILD)
        for source in (r"g_{\mu\nu}^{-1}", r"(g_{\mu\nu})^{-1}"):
            found = error(wb, source, IndexNotationError)
            assert found.message == "cannot raise a tensor to the power -1"
            assert found.hint == r"the inverse metric is g^{\mu\nu}"
        assert wb.expr(r"g^{\mu\nu} g_{\nu\rho}")  # the hint works

    def test_a_vector_with_a_bare_number_above_is_a_component_or_a_power(self, wb):
        wb.exe(SCHWARZSCHILD)
        wb.exe(r"L_\mu := [0, 0, 0, r^2 \sin^2\theta]")
        found = error(wb, "L^3", IndexNotationError)
        assert found.message == (
            "L is a tensor, so L^3 could be its component with upper index 3 or a power of it"
        )
        assert found.hint == (
            "for the component, write L^{3}; to multiply L by itself, contract it, as in "
            "L_{a} L^{a}"
        )
        assert wb.expr("L^{3}") == 1  # the hint works, both ways
        r, theta = __import__("sympy").symbols("r theta")
        assert wb.expr("L_{a} L^{a}") == r**2 * __import__("sympy").sin(theta) ** 2
        assert wb.expr("L^0") == 0  # never a power
        assert error(wb, "L^4", IndexNotationError).message == "cannot raise a tensor to a power"

    def test_a_script_on_a_run_of_letters_binds_to_its_last_letter_a_tensor_too(self, wb):
        wb.exe(SCHWARZSCHILD)
        wb.exe(r"L_\mu := [0, 0, 0, r^2 \sin^2\theta]")
        assert wb.expr(r"3GML^{3}\gamma") == wb.expr(r"3 G M \gamma")
        assert (
            error(wb, r"3GML^3\gamma", IndexNotationError).message
            == error(wb, r"3 G M L^3 \gamma", IndexNotationError).message
        )

    def test_the_determinant_to_a_power_says_so(self, wb):
        wb.exe(SCHWARZSCHILD)
        assert notes(wb, "g^{-1}") == [
            r"g alone is the determinant of the metric, so g^{-1} is a power of it; the inverse "
            r"metric is g^{\mu\nu}"
        ]


class TestNamesNeverDefined:
    def test_a_built_in_spelled_out(self, wb):
        wb.exe(SCHWARZSCHILD)
        assert notes(wb, "RicciScalar") == [
            "RicciScalar is not defined, so it is a symbol; did you mean R or Ric, the Ricci "
            "scalar?"
        ]
        assert error(wb, "RicciScalar(g)").hint == "did you mean R or Ric, the Ricci scalar?"

    def test_a_misspelt_function(self, wb):
        found = error(wb, "simplfy(x^2 + 2x + 1)")
        assert found.message == "unknown function simplfy"
        assert found.hint == "did you mean simplify(...)?"

    def test_print_is_not_needed(self, wb):
        assert error(wb, "print(x)").hint == "write x"

    def test_an_unknown_function_with_a_word_for_a_name_says_so(self, wb):
        assert notes(wb, "foo(x)") == [
            "foo is not a function RelativisticPy knows, so foo(x) is an unknown function"
        ]
        assert notes(wb, "f(x)") == []  # a letter: the usual unknown function
        assert notes(wb, r"\gamma(x)") == []

    def test_a_builtin_named_alone(self, wb):
        assert notes(wb, "simplify + 1") == [
            "simplify is a function, so alone it is read as a symbol; to apply it, write "
            "simplify(...)"
        ]
        assert notes(wb, "N/2") == []  # the lapse

    def test_r_as_the_metrics_symbol(self, wb):
        wb.exe("Coordinates := [t, \\theta, \\phi]")
        wb.exe(r"ds^2 = -dt^2 + R^2 d\theta^2 + R^2 \sin^2\theta d\phi^2")
        assert notes(wb, "R") == [
            "the metric is written with R, so R is that symbol; the Ricci scalar is Ric"
        ]
