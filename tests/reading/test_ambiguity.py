r"""How the reader reads notation that the parser leaves ambiguous."""

import pytest
import sympy
from sympy import Array, Function, Symbol, symbols

from relativisticpy import Tensor
from relativisticpy.errors import CompileError, IndexNotationError, RelativisticPyError

x, y, r, theta, a, b, n = symbols("x y r theta a b n")


@pytest.fixture
def plane(wb):
    wb.exe(r"Coordinates := [x, y]" "\n" r"g_{ab} := [[1, 0], [0, x^2]]" "\n" r"V^{a} := [y, x]")
    return wb


class TestScripts:
    def test_a_superscript_on_a_symbol_is_a_power(self, wb):
        assert wb.expr("x^{a}") == x**a
        assert wb.expr(r"x^{n+1}") == x ** (n + 1)
        assert wb.expr(r"e^{i\pi}") == -1

    def test_a_superscript_on_a_tensor_is_an_index(self, plane):
        result = plane.expr("V^{a}")
        assert isinstance(result, Tensor)
        assert str(result.indices) == "^{a}"

    def test_a_subscript_on_a_symbol_names_a_symbol(self, wb):
        assert wb.expr("r_{s}") == Symbol("r_s")
        assert wb.expr("x_{0}^2") == Symbol("x_0") ** 2
        assert wb.expr(r"T_{\text{eff}}") == Symbol("T_eff")

    def test_a_subscripted_name_and_its_scripted_spelling_agree(self, wb):
        wb.exe("x_0 := 5")
        assert wb.expr("x_{0}") == 5

    def test_a_tower_of_superscripts_associates_to_the_right(self, wb):
        assert wb.expr("2^3^2") == 2**9

    def test_several_indices_on_an_unknown_name(self, wb):
        wb.exe("Coordinates := [x, y]")
        with pytest.raises(CompileError, match="T is not a tensor") as error:
            wb.expr("T_{ab}")
        assert "define it first, for example T_{ab} := [...]" in str(error.value)

    def test_index_only_superscript_on_a_symbol(self, wb):
        with pytest.raises(CompileError, match="x is not a tensor"):
            wb.expr("x^{a=0}")


class TestBuiltInTensorNames:
    @pytest.fixture
    def flat(self, wb):
        wb.exe(r"g_{\mu\nu} := Minkowski()")
        return wb

    @pytest.mark.parametrize(
        ("source", "indices"),
        [
            ("C^{a}_{bc}", "^{a}_{bc}"),
            (r"\Gamma^{a}_{bc}", "^{a}_{bc}"),
            ("R^{a}_{bcd}", "^{a}_{bcd}"),
            ("R_{ab}", "_{ab}"),
            ("Ric_{ab}", "_{ab}"),
            ("G_{ab}", "_{ab}"),
            ("W_{abcd}", "_{abcd}"),
            ("C_{abcd}", "_{abcd}"),
            (r"\delta^{a}_{b}", "^{a}_{b}"),
        ],
    )
    def test_names_and_ranks(self, flat, source, indices):
        assert str(flat.expr(source).indices) == indices

    def test_a_built_in_with_the_wrong_number_of_indices(self, flat):
        with pytest.raises(IndexNotationError, match="Ric has 2 indices, but 3 are given"):
            flat.expr("Ric_{abc}")

    def test_renaming(self, wb):
        wb.exe("MetricSymbol := G\nCoordinates := [x, y]\nG_{ab} := [[1, 0], [0, 1]]")
        assert wb.expr("G_{ab}").components == Array([[1, 0], [0, 1]])

    def test_without_a_metric(self, wb):
        wb.exe("Coordinates := [x, y]")
        with pytest.raises(CompileError, match="the Christoffel connection needs a metric"):
            wb.expr("C^{a}_{bc}")

    def test_without_coordinates(self, wb):
        with pytest.raises(CompileError, match="needs coordinates") as error:
            wb.expr("T_{a} := [1, 2]")
        assert "Coordinates := [t, r" in str(error.value)

    def test_r_is_a_symbol_when_the_metric_uses_it(self, wb):
        wb.exe(r"Coordinates := [\theta, \phi]" "\n" r"g_{ab} := [[R^2, 0], [0, R^2 \sin^2\theta]]")
        assert wb.expr("R") == Symbol("R")
        assert wb.expr("Ric") == 2 / Symbol("R") ** 2

    def test_a_user_tensor_shadows_a_built_in(self, wb):
        wb.exe("Coordinates := [x, y]\nGamma^{a}_{bc} := 0")
        assert wb.expr("Gamma^{a}_{bc}").components == Array.zeros(2, 2, 2)

    @pytest.mark.parametrize("spacetime", ["FLRW(k)", "ReissnerNordstrom(M, Q)"])
    def test_r_is_the_ricci_scalar_whatever_the_spacetime_is_called(self, wb, spacetime):
        # The names of spacetimes are not runs of letters: FLRW holds no R.
        wb.exe(rf"g_{{\mu\nu}} := {spacetime}")
        assert wb.expr("R") == wb.expr("Ric")
        assert wb.expr("R") != Symbol("R")

    def test_g_is_a_symbol_when_the_metric_uses_it(self, wb):
        wb.exe(r"Coordinates := [t, x]" "\n" r"g_{ab} := diag(-1, g^2 t^2)")
        g, t = Symbol("g"), Symbol("t")
        assert wb.expr("g") == g
        assert wb.expr(r"\det(g_{ab})") == -(g**2) * t**2
        wb.exe(r"g_{ab} := diag(-1, t^2)")
        assert wb.expr("g") == -(t**2)  # the determinant again

    @pytest.mark.parametrize(
        ("claim", "name"),
        [("R = 1", "R"), ("Ric = 1", "Ric"), ("g = 5", "g"), ("Kretschmann = 0", "Kretschmann")],
    )
    def test_with_a_metric_a_built_in_scalar_is_claimed_not_defined(self, wb, claim, name):
        wb.exe(r"g_{\mu\nu} := Schwarzschild(M)")
        before = wb.expr(name)
        assert wb.expr(claim) is not sympy.true  # a claim that fails
        assert wb.expr(name) == before  # and the built-in is as it was

    def test_a_built_in_defined_by_equiv_is_an_identity(self, wb):
        definition = r"R_{\mu\nu} \equiv R^{\alpha}{}_{\mu\alpha\nu}"
        assert wb.expr(definition).holds()  # without a metric, an identity of notation
        wb.exe(r"g_{\mu\nu} := Schwarzschild(M)")
        assert wb.expr(definition)  # with one, a claim that holds component by component
        assert wb.expr(r"R \equiv g^{\mu\nu} R_{\mu\nu}") is sympy.true
        M = Symbol("M")
        assert wb.expr("R_{abcd} R^{abcd}") == 48 * M**2 / r**6  # R is still built in
        assert wb.expr("R") == 0
        wb.exe(r"x \equiv \frac{r}{M}")  # a new name is still defined
        assert wb.expr("x") == r / M


class TestDefinitionsAndEquations:
    def test_an_existing_tensor_is_compared_not_overwritten(self, plane):
        assert plane.expr("V^{a} = [y, x]")  # an equation that holds
        assert plane.expr("V^{a}").components == Array([y, x])  # V is unchanged
        plane.exe("V^{a} := [1, 2]")  # := redefines
        assert plane.expr("V^{a}").components == Array([1, 2])

    def test_a_bare_name_is_assigned(self, wb):
        assert wb.exe("y = 3") == []
        assert wb.expr("y + 1") == 4

    def test_a_new_function_on_the_left_is_defined(self, wb):
        # f(x) = x^2 defines f, as x = 2 defines x; at a value, or again, it is a claim
        assert wb.exe("f(x) = x^2") == []
        assert wb.expr("f(2)") == 4
        assert wb.expr("f(2) = 4") is sympy.true
        assert wb.expr("g(2) = 4") == sympy.Eq(Function("g")(2), 4)  # g has no meaning
        assert wb.expr("f(x) = x^3") is not sympy.true
        assert wb.expr("f(2)") == 4  # unchanged

    def test_an_expression_on_the_left_is_an_equation(self, wb):
        assert wb.expr("x^2 + y^2 = 1") == sympy.Eq(x**2 + y**2, 1)

    def test_a_built_in_tensor_on_the_left_is_an_equation(self, plane):
        equation = plane.expr("R_{ab} = 0")
        assert len(equation.equations()) == 0  # flat... in polar-like coordinates
        assert bool(equation)

    def test_a_component_of_the_metric_is_a_claim_about_it(self, wb):
        wb.exe(r"g_{\mu\nu} := Schwarzschild(M)")
        assert wb.expr(r"g_{tt} = -(1 - \frac{2M}{r})") is sympy.true
        assert wb.expr(r"g_{rr} = 1") is not sympy.true
        M = Symbol("M")
        assert wb.expr("Kretschmann") == 48 * M**2 / r**6  # the metric is unchanged

    def test_a_component_of_a_new_tensor_defines_nothing(self, plane):
        with pytest.raises(CompileError, match="T is not a tensor") as error:
            plane.exe("T_{xy} = 0")
        assert r"T_{\mu\nu} := [...]" in str(error.value)
        with pytest.raises(CompileError, match="T is not a tensor"):
            plane.expr("T_{ab}")  # T_{xy} = 0 did not define all of T

    def test_a_component_equation_is_not_notation(self, wb):
        # Before T has components, G_{tt} = 8 pi T_{tt} cannot be decided; it is not
        # index notation, which would sum over the repeated t.
        wb.exe(r"g_{\mu\nu} := Schwarzschild(M)")
        with pytest.raises(CompileError, match="T is not a tensor"):
            wb.expr(r"G_{tt} = 8\pi T_{tt}")

    def test_the_identity_of_a_space_of_data_has_components(self, wb):
        # \delta_{ab} over the space of w is its identity matrix, not index notation
        numpy = pytest.importorskip("numpy")
        wb["w^a"] = numpy.array([1.0, 2.0])
        assert wb.expr(r"w^a \delta_{ab} w^b = 5") is sympy.true
        assert wb.expr(r"w^a \delta_{ab} w^b = 4") is not sympy.true
        assert wb.expr(r"w_a = \delta_{ab} w^b")

    def test_data_named_without_braces_is_compared_not_overwritten(self, wb):
        numpy = pytest.importorskip("numpy")
        wb["w^a"] = numpy.array([1.0, 2.0])
        wb["S_{ab}"] = numpy.array([[2.0, 0.0], [0.0, 1.0]])
        equation = wb.expr("w_a = S_{ab} w^b")  # w_a is w_{a}, a tensor that exists
        assert not equation
        assert wb.expr("w^a").to_numpy().tolist() == [1.0, 2.0]

    def test_definitions_check_their_free_indices(self, plane):
        sides = (
            r"the two sides have different free indices: _\{a\} on the left, _\{b\} on the right"
        )
        with pytest.raises(IndexNotationError, match=sides):
            plane.expr("W_{a} := V_{b}")

    def test_definitions_reorder_to_the_left(self, plane):
        plane.exe("W^{a} := [1, 2]\nT^{ba} := V^{a} W^{b}")
        # T's first slot is b: T^{ab} reads V^{b} W^{a}
        assert plane.expr("T^{ab}").components == Array([[y, x], [2 * y, 2 * x]])

    def test_a_tensor_needs_components(self, plane):
        with pytest.raises(CompileError, match="the components of T must be a list, a matrix or 0"):
            plane.expr("T_{ab} := 5")
        assert plane.expr("T^{a} := 0\nT^{a}").components == Array([0, 0])

    def test_list_shape_is_checked_as_written(self, plane):
        with pytest.raises(
            CompileError, match="expected 2×2 components for 2 coordinates, but found 2×3"
        ):
            plane.expr("T_{ab} := [[1, 2, 3], [4, 5, 6]]")

    def test_a_definition_takes_labels_only(self, plane):
        with pytest.raises(IndexNotationError, match="index labels only"):
            plane.expr("T_{a=0} := [1, 2]")

    def test_metric_positions(self, wb):
        wb.exe("Coordinates := [x, y]")
        with pytest.raises(IndexNotationError, match="both indices down"):
            wb.expr("g^{a}_{b} := [[1, 0], [0, 1]]")
        wb.exe("g^{ab} := [[1, 0], [0, 4]]")
        assert wb.expr("g_{ab}").components == Array([[1, 0], [0, sympy.Rational(1, 4)]])

    def test_a_degenerate_metric_is_refused_where_it_is_written(self, wb):
        from relativisticpy.errors import EvaluationError

        wb.exe("Coordinates := [x, y]")
        with pytest.raises(EvaluationError, match="the metric is degenerate") as error:
            wb.exe("g_{ab} := [[1, 1], [1, 1]]")
        assert "g_{ab} := [[1, 1], [1, 1]]" in str(error.value)
        assert "no curvature can be computed" in str(error.value)
        with pytest.raises(CompileError, match="there is no metric yet"):
            wb.expr("g^{ab}")  # the definition that failed defined nothing

    def test_a_metric_missing_a_coordinate_says_which(self, wb):
        from relativisticpy.errors import EvaluationError

        wb.exe("Coordinates := [t, r]")
        with pytest.raises(EvaluationError, match="the metric is degenerate") as error:
            wb.exe("ds^2 = -dt^2")
        assert "note: no term of it has dr" in str(error.value)
        assert "hint: give it a term with dr" in str(error.value)

    def test_a_metric_of_the_wrong_size_is_refused_where_it_is_written(self, wb):
        from relativisticpy.errors import EvaluationError

        wb.exe("Coordinates := [t, x, y, z]")
        with pytest.raises(EvaluationError, match="has 3×3 components, but there are 4") as error:
            wb.exe("g_{\\mu\\nu} := diag(-1, 1, 1)")
        assert "the coordinates are t, x, y and z" in str(error.value)

    def test_a_metric_that_is_not_symmetric_is_refused_by_component(self, wb):
        from relativisticpy.errors import EvaluationError

        wb.exe("Coordinates := [x, y]")
        with pytest.raises(EvaluationError, match=r"g_\{xy\} = 2 but g_\{yx\} = 3"):
            wb.exe("g_{ab} := [[1, 2], [3, 4]]")

    def test_a_failed_metric_leaves_the_one_in_force(self, wb):
        wb.exe(r"g_{\mu\nu} := Schwarzschild(M)")
        with pytest.raises(RelativisticPyError):
            wb.exe(r"ds^2 = -dt^2 + dr^2")  # degenerate in four coordinates
        assert wb.expr("Kretschmann") == 48 * Symbol("M") ** 2 / r**6

    def test_coordinates_list_names(self, wb):
        with pytest.raises(CompileError, match="expected a name"):
            wb.expr("Coordinates := [t, 2]")
        with pytest.raises(CompileError, match="must be distinct"):
            wb.expr("Coordinates := [t, t]")

    def test_assumptions_come_before_coordinates(self, wb):
        wb.exe("Coordinates := [t, r]")
        with pytest.raises(CompileError, match="r is already a coordinate"):
            wb.expr("Positive := [r]")

    def test_a_subscripted_symbol_can_be_defined(self, wb):
        wb.exe("r_{s} := 2 G M")
        assert wb.expr("r_s") == 2 * Symbol("G") * Symbol("M")


class TestIndexNotationErrors:
    @pytest.mark.parametrize(
        ("source", "message"),
        [
            ("V^{a} + V_{a}", r"the free indices of a sum must agree: \^\{a\} and _\{a\}"),
            ("V^{a} V^{a}", "index a appears twice as an upper index"),
            ("V^{a} V_{a} V^{a}", "index a appears 3 times"),
            ("V^{a} + 1", r"the free indices of a sum must agree: \^\{a\} and none"),
            ("1 / V^{a}", "cannot divide by a tensor"),
            ("V^{a}^2", "x is not a tensor|cannot raise a tensor to a power|V has 1 index"),
            ("(V^{a} V_{a})^{V^{b}}", "cannot raise to the power of a tensor"),
            ("R_{ab} = 1", r"two sides have different free indices: _\{ab\} on the left, none"),
            (
                "V^{a} = V_{a}",
                r"two sides have different free indices: \^\{a\} on the left, _\{a\}",
            ),
            ("V^{a=2}", "index value 2 is out of range for 2 coordinates"),
            ("V^{a=z}", "z is not a coordinate"),
        ],
    )
    def test_messages(self, plane, source, message):
        with pytest.raises(CompileError, match=message):
            plane.expr(source)

    def test_errors_point_at_the_source(self, plane):
        with pytest.raises(IndexNotationError) as error:
            plane.expr("V^{a} + V_{b}")
        assert error.value.span is not None
        assert error.value.span.text == "V_{b}"


class TestFunctionsAndOperators:
    def test_recursion_is_reported(self, wb):
        with pytest.raises(CompileError, match="f calls itself"):
            wb.expr("f(x) := f(x - 1)\nf(3)")

    def test_arity(self, wb):
        with pytest.raises(CompileError, match=r"f\(\) takes 1 argument, not 2"):
            wb.expr("f(x) := x\nf(1, 2)")
        with pytest.raises(CompileError, match=r"sin\(\) takes 1 argument, not 2"):
            wb.expr("sin(1, 2)")

    def test_a_value_is_not_a_function(self, wb):
        with pytest.raises(CompileError, match="y is not a function"):
            wb.expr("y := 2\ny(3)")

    def test_a_function_body_must_end_with_a_value(self, wb):
        with pytest.raises(CompileError, match="f does not return a value"):
            wb.expr("f : (x) -> {\n y := x\n}\nf(1)")

    def test_an_operator_needs_an_operand(self, wb):
        with pytest.raises(CompileError, match="needs something to act on"):
            wb.expr(r"\frac{d}{dx}")
        with pytest.raises(CompileError, match="needs an index or a variable"):
            wb.expr(r"\partial")

    def test_a_derivative_without_coordinates_is_calculus(self, wb):
        assert wb.expr(r"\partial_x (x^2 y)") == 2 * x * y

    def test_derivative_operators_take_the_next_factor(self, plane):
        result = plane.expr(r"\partial_a V^{a}")  # a divergence: a trace
        assert result == 0

    def test_a_number_on_a_derivative_is_its_order(self, wb):
        wb.exe("Coordinates := [t, x, y, z]\ng_{ab} := diag(-1, 1, 1, 1)")
        t = Symbol("t")
        assert wb.expr(r"\partial^2_t (t^3)") == 6 * t  # never \partial^y
        assert wb.expr(r"\partial^{2}_{t} (t^3)") == 6 * t
        assert wb.expr(r"\partial_t^2 (t^3)") == 6 * t
        assert wb.expr(r"\partial^2_{tx} (t^2 x)") == 2 * t
        assert wb.expr(r"\partial^2 (t^3)") == -6 * t  # \partial^\mu \partial_\mu
        assert wb.expr(r"\nabla^2 (x^3 y^2)") == wb.expr(r"\Box (x^3 y^2)")

    def test_a_lone_number_on_a_derivative_is_not_a_component(self, wb):
        wb.exe("Coordinates := [t, x, y, z]\ng_{ab} := diag(-1, 1, 1, 1)")
        with pytest.raises(CompileError, match=r"\\partial\^1 has no meaning") as error:
            wb.expr(r"\partial^1 x")
        assert r"\partial^t" in str(error.value)

    def test_comma_and_semicolon_derivatives_of_a_scalar(self, wb):
        wb.exe(r"g_{\mu\nu} := Schwarzschild(M)" "\n" r"\phi := r^2 \cos\theta")
        gradient = [0, 2 * r * sympy.cos(theta), -(r**2) * sympy.sin(theta), 0]
        assert wb.expr(r"\phi_{,\mu}").components == Array(gradient)
        assert wb.expr(r"\phi_{;\mu}").components == Array(gradient)
        assert wb.expr(r"\phi_{,r}") == 2 * r * sympy.cos(theta)
        assert sympy.simplify(wb.expr(r"\phi_{;\mu}{}^{;\mu}") - wb.expr(r"\Box \phi")) == 0

    def test_the_gradient_of_a_built_in_scalar_or_a_function(self, wb):
        wb.exe(r"ds^2 = -e^{2\alpha(r)} dt^2 + e^{2\beta(r)} dr^2 + r^2 d\Omega^2")
        assert wb.expr(r"\alpha_{,r}") == sympy.diff(Function("alpha")(r), r)
        wb.exe(r"g_{\mu\nu} := Schwarzschild(M)")
        assert wb.expr(r"R_{;\mu}").components == Array([0, 0, 0, 0])

    @pytest.mark.parametrize(
        ("source", "message", "hint"),
        [
            (r"\psi_{,\mu}", "psi is not defined, so it has no derivative", r"\psi := ..."),
            (r"F_{\mu\nu,\lambda}", "F is not a tensor", r"F_{\mu\nu} := [...]"),
        ],
    )
    def test_a_derivative_of_what_is_not_defined(self, wb, source, message, hint):
        wb.exe(r"g_{\mu\nu} := Schwarzschild(M)")
        with pytest.raises(CompileError, match=message) as error:
            wb.expr(source)  # never a symbol named psi_mu
        assert hint in str(error.value)

    def test_divergence_with_the_covariant_derivative(self, wb):
        wb.exe(r"Coordinates := [r, \theta]" "\n" r"g_{ab} := diag(1, r^2)" "\n" "V^{a} := [r, 0]")
        assert wb.expr(r"\nabla_a V^{a}") == 2  # ∇·(r e_r) = 2 in the plane

    def test_function_names_apply_to_what_follows(self, wb):
        assert wb.expr("sin x") == sympy.sin(x)
        assert wb.expr("f(t) := t^2\nf 3") == 9

    def test_latex_letter_products(self, wb):
        assert wb.expr(r"\frac{abc}{2}") == a * b * Symbol("c") / 2
        wb.exe("mass := 5")
        assert wb.expr(r"\frac{mass}{5}") == 1  # a defined name stays whole

    def test_signs_let_operators_act(self, wb):
        assert wb.expr(r"-\frac{d}{dx} x^2") == -2 * x
        wb.exe("Coordinates := [t, x]")
        wb.exe("ds^2 = -d t^2 + d x^2")
        assert wb.expr("g_{ab}").components == Array([[-1, 0], [0, 1]])

    def test_a_line_element_needs_coordinates_after_d(self, wb):
        wb.exe("Coordinates := [t, x]")
        with pytest.raises(CompileError, match="dy is not the differential of a coordinate"):
            wb.expr(r"ds^2 = -d t^2 + d y^2")
        with pytest.raises(CompileError, match="dy and dz are not differentials") as error:
            wb.expr(r"ds^2 = -dt^2 + dy^2 + dz^2")
        assert "the coordinates are t and x, as declared" in str(error.value)
        assert "Coordinates := [t, y, z]" in str(error.value)  # the hint works:
        wb.exe("Coordinates := [t, y, z]\nds^2 = -dt^2 + dy^2 + dz^2")
        assert wb.expr("g_{zz}") == 1

    def test_solid_angle_needs_angles(self, wb):
        wb.exe("Coordinates := [t, x]")
        with pytest.raises(CompileError, match=r"d\\Omega needs coordinates named"):
            wb.expr(r"ds^2 = -dt^2 + d\Omega^2")

    def test_dot_needs_one_variable(self, wb):
        from relativisticpy.errors import EvaluationError

        with pytest.raises(EvaluationError, match="cannot tell what"):
            wb.expr(r"\dot{f(x, y)}")
