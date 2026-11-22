r"""What the compiler says when notation breaks a rule: in the user's notation, one
phrasing for each rule, and, where it can tell, which part is wrong and how to fix it."""

import re

import pytest

from relativisticpy.errors import CompileError, EvaluationError, IndexNotationError, ParseError

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
        assert error(wb, source).message == f"empty {message}"


def notes(wb, source):
    """The notes of the one step of a one-statement program."""
    (step,) = list(wb.each(source))
    assert step.error is None, step.error
    return step.notes


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


class TestArity:
    def test_a_spacetime_names_its_parameters(self, wb):
        assert error(wb, r"g_{\mu\nu} := Schwarzschild(M, G, c, d)").message == (
            "Schwarzschild(M, G, c) takes at most 3 arguments, not 4"
        )

    def test_one_argument_is_singular(self, wb):
        assert error(wb, "solve()").message == "solve() takes at least 1 argument, not 0"
        assert error(wb, "log(1, 2, 3)").message == "log() takes 1 or 2 arguments, not 3"

    @pytest.mark.parametrize("source", ["eig()", "eig(1, 2, 3)", "spec(x)"])
    def test_linear_algebra_takes_a_tensor(self, wb, source):
        assert "takes a tensor written with its indices" in error(wb, source).message

    def test_an_argument_by_name(self, wb):
        found = error(wb, "series(sin(x), x, 0, n=5)")
        assert (
            found.message == "series() takes its arguments in order, not by name, as n=5 gives one"
        )
        assert found.hint == "write series(sin(x), x, 0, 5)"
        assert wb.expr("series(sin(x), x, 0, 5)")  # the hint works


class TestBuiltinsSayWhatTheyTake:
    def test_rhs_of_what_is_no_equation(self, wb):
        assert error(wb, "RHS(x)", EvaluationError).message == (
            "RHS takes an equation, as in RHS(x^2 = 4), but x is not one"
        )

    def test_subs_takes_rules(self, wb):
        x = __import__("sympy").Symbol("x")
        assert wb.expr("subs(x^2, x = 3)") == 9
        assert wb.expr("subs(x^2, x, 3)") == 9
        found = error(wb, "subs(x^2, x)", EvaluationError).message
        assert "subs takes an expression and what to put for what" in found
        assert wb.expr("symbols(x)") == x
        assert "symbols takes names in quotes" in error(wb, "symbols(2)", EvaluationError).message

    def test_a_list_is_not_indexed(self, wb):
        found = error(wb, "[1, 2][0]")
        assert found.message == "a list is not indexed with [ ]"
        assert (
            found.hint
            == "to name the elements of a list, write them on the left of :=, as in a, b := [1, 2]"
        )
        wb.exe("a, b := [1, 2]")  # the hint works
        assert wb.expr("a") == 1
        wb.exe(SCHWARZSCHILD)
        assert (
            error(wb, r"geodesic_equations(\lambda)[1]").message == "a list is not indexed with [ ]"
        )

    def test_geodesics_are_integrated_from_python(self, wb):
        wb.exe(SCHWARZSCHILD)
        found = error(wb, "geodesic(g, [0, 10, pi/2, 0])")
        assert found.message == (
            "geodesics are integrated from Python, with relativisticpy.numeric.geodesic"
        )
        assert len(wb.expr(r"geodesic_equations(\tau)")) == 4  # the hint works


class TestPowersOfTensors:
    def test_the_inverse_metric(self, wb):
        wb.exe(SCHWARZSCHILD)
        for source in (r"g_{\mu\nu}^{-1}", r"(g_{\mu\nu})^{-1}"):
            found = error(wb, source, IndexNotationError)
            assert found.message == "cannot raise a tensor to the power -1"
            assert found.hint == r"the inverse metric is g^{\mu\nu}"
        assert wb.expr(r"g^{\mu\nu} g_{\nu\rho}")  # the hint works

    def test_the_determinant_to_a_power_says_so(self, wb):
        wb.exe(SCHWARZSCHILD)
        assert notes(wb, "g^{-1}") == [
            r"g alone is the determinant of the metric, so g^{-1} is a power of it; the inverse "
            r"metric is g^{\mu\nu}"
        ]


class TestWhatAMetricIsNeededFor:
    def test_lowering_an_index_without_a_metric(self, wb):
        wb.exe("Coordinates := [t, x]\nV^{a} := [1, 2]")
        assert error(wb, "V_a").message == (
            "V is defined as V^{a}; writing V_a lowers an index, which needs a metric"
        )

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


class TestSilentReadingsAreSaid:
    def test_greek_superscripts_on_a_name_that_means_nothing(self, wb):
        assert notes(wb, r"T^{\mu\nu}") == [
            r"T is not defined, so T^{\mu\nu} is read as T to a power; to use a tensor, define "
            r"it first: T^{\mu\nu} := [...]"
        ]
        assert notes(wb, r"\frac{d x^\mu}{d\tau}")[0].startswith(
            r"x is not defined, so x^\mu is read as x to a power"
        )
        assert notes(wb, "x^n y^n") == []  # a Latin exponent is a power, as written

    def test_a_label_written_up_and_down_on_a_name_that_means_nothing(self, wb):
        assert notes(wb, "u^a u_a = -1") == [
            "u is not defined, so u^a is a power and u_a one symbol; to use a tensor, define "
            "it first: u^{a} := [...]"
        ]
        assert notes(wb, r"u_\alpha u_\alpha = -1")[0].startswith(
            r"u is not defined, so u_\alpha is one symbol, not a component"
        )

    def test_a_derivative_with_an_index_and_no_coordinates(self, wb):
        assert notes(wb, r"\nabla_\mu \nabla^\mu \phi = 0") == [
            r"there are no coordinates, so a derivative with the index \mu is one with respect "
            r"to a variable named \mu; declare coordinates to make \mu an index"
        ]

    def test_an_undefined_name_added_to_a_tensor(self, wb):
        numpy = pytest.importorskip("numpy")
        wb["x_t"] = numpy.ones(3)
        found = error(wb, "x_t + y_t", IndexNotationError)
        assert found.message == "the free indices of a sum must agree: _{t} and none"
        assert found.notes == ["y is not defined, so y_t is read as one symbol, y_t"]


class TestDerivativesOfFunctions:
    def test_a_function_written_alone_is_differentiated_at_its_argument(self, wb):
        x = __import__("sympy").Symbol("x")
        wb.exe("f(x) := x^3")
        assert wb.expr(r"\frac{d^2 f}{dx^2}") == 6 * x
        assert wb.expr(r"\frac{d^2 f(x)}{dx^2}") == 6 * x
        assert wb.expr("diff(f(x), x, 2)") == 6 * x

    def test_a_value_written_with_what_it_depends_on_is_that_value(self, wb):
        x = __import__("sympy").Symbol("x")
        wb.exe("f := x^3")
        assert wb.expr(r"\frac{d^2 f(x)}{dx^2}") == 6 * x  # f(x) is f, not f times x

    def test_a_name_that_means_nothing_is_constant_and_says_so(self, wb):
        assert wb.expr(r"\frac{d^2 f}{dx^2}") == 0
        assert notes(wb, r"\frac{d^2 f}{dx^2}") == [
            "f does not depend on x, so its derivative is 0; for a function, write f(x)"
        ]


class TestHintsThatWork:
    def test_a_sum_over_an_index_without_limits_is_a_contraction(self, wb):
        numpy = pytest.importorskip("numpy")
        wb["w^a"] = numpy.array([1.0, 2.0])
        found = error(wb, r"\sum_a w^a")
        assert found.message == r"\sum needs limits"
        assert found.hint == (
            "to sum the components of w^a, contract it with the covector of ones: w^a 1_a"
        )
        assert wb.expr("w^a 1_a") == 3.0  # the hint works

    def test_solving_a_trace_for_the_ricci_scalar(self, wb):
        wb.exe(r"EFE := G_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G T_{\mu\nu}")
        assert str(wb.expr("solve(trace(EFE), R)")) == r"R = 4 \Lambda - 8 \pi G T"
