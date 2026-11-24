r"""Vectors and vector calculus, read by their operands (design/notation.md, 8, 9, 16 and
25): lists, tuples and columns are vectors, with their dot and cross products and their
lengths; unit vectors where nothing defines them; derivatives, integrals and bars of each
component; ∇ in the frame of the coordinates in force, from a line element, from the
Cartesian coordinates written in them or from their names; equations between vectors,
decided component by component; and integrals over a disk, a ball or a box."""

from __future__ import annotations

import pytest
import sympy

from relativisticpy import Workbook
from relativisticpy.errors import CompileError, EvaluationError
from relativisticpy.runtime import vectors

x, y, z, t, r, theta, phi, rho, omega = sympy.symbols("x y z t r theta phi rho omega")
FIELD = r"\mathbf{v} := xy\,\hat{\mathbf{x}} + 2yz\,\hat{\mathbf{y}} + 3zx\,\hat{\mathbf{z}}"
SPHERICAL = r"ds^2 = dr^2 + r^2\, d\theta^2 + r^2 \sin^2\theta\, d\phi^2"
COLUMN = r"\begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}"
BASIS = "\n".join(
    rf"\mathbf{{{name}}} := {value}"
    for name, value in (("a", "(1, 0, 0)"), ("b", "(0, 1, 0)"), ("c", "(0, 0, 1)"))
)


def readings(wb: Workbook, source: str) -> list[str]:
    return [
        decision.text()
        for reading in wb.explain(source)
        for decision in reading.decisions
        if decision.kind == "vector or number"
    ]


def notes(wb: Workbook, source: str) -> list[str]:
    return [line for step in wb.each(source) for note in step.notes for line in note.split("\n")]


def error(wb: Workbook, source: str) -> CompileError:
    with pytest.raises(CompileError) as raised:
        wb.exe(source)
    return raised.value


def components(value: object) -> list[sympy.Expr]:
    return [sympy.simplify(each) for each in sympy.Array(value).reshape(len(value))]  # type: ignore[arg-type]


def same(found: object, expected: object) -> bool:
    """Whether two values are equal, their symbols compared by name."""
    value = sympy.sympify(found)
    plain = value.xreplace({s: sympy.Symbol(s.name) for s in value.free_symbols})
    return sympy.simplify(plain - sympy.sympify(expected)) == 0


class TestAlgebra:
    def test_a_dot_and_a_cross_product_of_lists(self, wb):
        wb.exe("v := [1, 2, 3]\nw := [4, 5, 6]")
        assert wb.expr(r"v \cdot w") == 32
        assert list(wb.expr(r"v \times w")) == [-3, 6, -3]

    @pytest.mark.parametrize("written", ["(1, 2, 3)", COLUMN])
    def test_a_tuple_or_a_column_is_a_vector(self, wb, written):
        wb.exe(rf"\mathbf{{v}} := {written}")
        assert wb.expr(r"\mathbf{v} \cdot \mathbf{v}") == 14
        assert wb.expr(r"|\mathbf{v}|") == sympy.sqrt(14)
        assert wb.expr(r"\|\mathbf{v}\|") == sympy.sqrt(14)
        assert wb.expr(r"\mathbf{v}^2") == 14  # its dot product with itself

    def test_the_length_of_a_complex_vector_takes_the_conjugate(self, wb):
        # the review's programs, each 0 when the length was the root of the squares
        wb.exe(r"|\psi\rangle := \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ i \end{pmatrix}")
        assert wb.expr(r"\| |\psi\rangle \|") == 1
        wb.exe(r"\mathbf{u} := (1, i)" "\n" r"\mathbf{E} := (E_0, i E_0, 0)")
        assert wb.expr(r"|\mathbf{u}|") == sympy.sqrt(2)
        assert same(wb.expr(r"|\mathbf{E}|^2"), 2 * sympy.Symbol("E_0") ** 2)

    def test_a_length_says_which_names_it_takes_as_real(self, wb):
        # |(E₀, iE₀, 0)|² is 2E₀² only if E₀ is real: written with i, it may not be
        wb.exe(r"\mathbf{E} := (E_0, i E_0, 0)")
        (step,) = wb.each(r"|\mathbf{E}|^2")
        assert any("takes E₀ as real" in note for note in step.notes)
        wb.exe(r"\mathbf{r} := (x, y, z)")  # nothing in it is complex: nothing to say
        (step,) = wb.each(r"|\mathbf{r}|")
        assert not any("as real" in note for note in step.notes)

    def test_a_row_and_a_column_have_one_length(self, wb):
        row = r"\left\| \begin{pmatrix} 1 & i \end{pmatrix} \right\|"
        column = r"\left\| \begin{pmatrix} 1 \\ i \end{pmatrix} \right\|"
        assert wb.expr(row) == wb.expr(column) == sympy.sqrt(2)
        wb.exe(r"\mathbf{r} := (x, y, z)")  # its names are real, as books take them
        assert same(wb.expr(r"|\mathbf{r}|"), sympy.sqrt(x**2 + y**2 + z**2))

    def test_a_name_declared_complex_is_conjugated_in_a_length(self, wb):
        wb.exe("Complex := [E_0]\n" r"\mathbf{E} := (E_0, i E_0, 0)")
        e = sympy.Symbol("E_0")
        assert same(wb.expr(r"|\mathbf{E}|^2"), 2 * e * sympy.conjugate(e))

    def test_unicode_and_arrows_write_them_too(self, wb):
        wb.exe("v := [1, 2, 3]\nw := [4, 5, 6]")
        assert wb.expr("v · w") == 32
        assert list(wb.expr("v × w")) == [-3, 6, -3]
        assert wb.expr("∇ · [x y, y z, z x]") == x + y + z
        wb.exe(r"\vec{a} := (1, 2, 3)" "\n" r"\vec{b} := (0, 0, 1)")
        assert list(wb.expr(r"\vec{a} \times \vec{b}")) == [2, -1, 0]

    def test_between_numbers_the_signs_multiply(self, wb):
        assert wb.expr(r"3 \times 4") == 12
        assert wb.expr(r"3 \cdot 4") == 12
        assert wb.expr(r"2 \times 10^{-5}") == sympy.Rational(1, 50000)

    def test_beside_a_number_a_vector_is_scaled_and_said_so(self, wb):
        wb.exe(r"\mathbf{v} := (1, 2, 3)")
        assert list(wb.expr(r"2 \cdot \mathbf{v}")) == [2, 4, 6]
        assert notes(wb, r"2 \cdot \mathbf{v}") == [
            r"\cdot beside a number is read as multiplying the vector by it"
        ]

    def test_the_cross_product_binds_before_the_dot_product(self, wb):
        wb.exe(BASIS)
        assert wb.expr(r"\mathbf{a} \cdot \mathbf{b} \times \mathbf{c}") == 1
        found = error(wb, r"\mathbf{a} \times \mathbf{b} \times \mathbf{c}")
        assert "two cross products without brackets" in found.message

    def test_a_cross_product_is_of_three_components(self, wb):
        found = error(wb, r"(1, 2) \times (3, 4)")
        assert found.message.endswith("these have 2 and 2")

    def test_the_readings_are_recorded(self, wb):
        wb.exe("v := [1, 2, 3]\nw := [4, 5, 6]")
        assert readings(wb, r"v \times w") == [
            r"vector or number: v \times w is their cross product, since \times is written "
            "between two vectors"
        ]
        assert notes(wb, r"v \times w") == [
            r"\times between vectors is read as their cross product, and between numbers as "
            "multiplication"
        ]


class TestUnitVectors:
    def test_hats_on_cartesian_coordinates(self, wb):
        wb.exe(FIELD)
        assert list(wb.expr(r"\mathbf{v}")) == [x * y, 2 * y * z, 3 * z * x]

    @pytest.mark.parametrize("written", [r"\hat{\imath}", r"\mathbf{i}", r"\hat{x}"])
    def test_the_unit_vector_along_x(self, wb, written):
        assert list(wb.expr(written)) == [1, 0, 0]

    def test_a_plain_hat_is_said_to_be_read_so(self, wb):
        assert notes(wb, r"v_x \hat{x} + v_y \hat{y}") == [
            r"\hat{x} is read as the unit vector along x; to name an operator or an estimate, "
            r"define \hat{x}",
            r"\hat{y} is read as the unit vector along y; to name an operator or an estimate, "
            r"define \hat{y}",
        ]

    def test_bold_i_j_and_k_are_said_to_be_read_so(self, wb):
        assert wb.expr(r"\mathbf{k} \cdot \mathbf{k}") == 1
        assert notes(wb, r"\mathbf{k}") == [
            r"\mathbf{k} is read as a unit vector of Cartesian coordinates; to name a vector k, "
            "define it"
        ]

    def test_a_definition_of_the_programs_own_wins(self, wb):
        wb.exe(r"\hat{x} := 5" "\n" r"\mathbf{k} := (0, 2, 0)")
        assert wb.expr(r"\hat{x}") == 5
        assert list(wb.expr(r"\mathbf{k}")) == [0, 2, 0]

    def test_a_hat_beside_an_operator_is_an_operators(self, wb):
        wb.exe(r"[\hat{x}, \hat{p}] = i\hbar")  # a rule, of the quantum algebra
        assert wb.expr(r"\hat{\varphi}") == sympy.Symbol("phihat")  # no coordinate: a name

    def test_the_unit_vectors_of_the_coordinates_in_force(self, wb):
        wb.exe(SPHERICAL)
        assert list(wb.expr(r"\hat{\mathbf{r}}")) == [1, 0, 0]
        assert list(wb.expr(r"\mathbf{e}_\theta")) == [0, 1, 0]
        assert list(wb.expr(r"\hat{\boldsymbol{\phi}}")) == [0, 0, 1]

    def test_a_cartesian_unit_vector_in_curvilinear_coordinates_is_refused(self, wb):
        wb.exe(SPHERICAL)
        found = error(wb, r"\hat{\mathbf{x}}")
        assert found.message == (
            r"\hat{\mathbf{x}} is a unit vector of Cartesian coordinates, and the coordinates "
            r"in force are (r, \theta, \phi)"
        )


class TestCalculus:
    def test_a_derivative_and_an_integral_of_each_component(self, wb):
        wb.exe(r"\mathbf{r} := (\cos t, \sin t, t)")
        assert list(wb.expr(r"\frac{d\mathbf{r}}{dt}")) == [-sympy.sin(t), sympy.cos(t), 1]
        assert list(wb.expr(r"\int_0^1 \mathbf{r}\,dt")) == [
            sympy.sin(1),
            1 - sympy.cos(1),
            sympy.Rational(1, 2),
        ]

    def test_the_derivative_of_a_function_whose_value_is_a_vector(self, wb):
        wb.exe(r"\mathbf{r}(t) := (t, t^2, 0)")
        assert list(wb.expr(r"\mathbf{r}'(t)")) == [1, 2 * t, 0]
        assert list(wb.expr(r"\mathbf{r}''(t)")) == [0, 2, 0]

    def test_a_partial_derivative_by_a_subscript(self, wb):
        v = sympy.Symbol("v")
        wb.exe(r"\mathbf{x}(u, v) := (u, v, u v)")
        assert list(wb.expr(r"\mathbf{x}_u")) == [1, 0, v]
        assert list(wb.expr(r"\mathbf{x}_{uv}")) == [0, 0, 1]

    def test_an_evaluation_bar_of_each_component(self, wb):
        wb.exe(FIELD)
        assert list(wb.expr(r"\left. \mathbf{v} \right|_{x=1}")) == [y, 2 * y * z, 3 * z]

    def test_a_field_at_a_point_of_a_path(self, wb):
        wb.exe(
            r"\mathbf{v} := y^2\,\hat{\mathbf{x}} + 2x(y + 1)\,\hat{\mathbf{y}}"
            "\n"
            r"\mathbf{r}(t) := (t, t, 0)"
        )
        assert wb.expr(r"\int_1^2 \mathbf{v}(\mathbf{r}(t)) \cdot \mathbf{r}'(t) \, dt") == 10

    def test_by_a_coordinate_whose_unit_vectors_turn_it_is_refused(self, wb):
        wb.exe(SPHERICAL)
        found = error(wb, r"\frac{\partial \hat{\mathbf{r}}}{\partial \theta}")
        assert "change with \\theta" in found.message


class TestOperators:
    def test_in_cartesian_coordinates(self, wb):
        wb.exe(FIELD)
        assert wb.expr(r"\nabla \cdot \mathbf{v}") == 3 * x + y + 2 * z
        assert list(wb.expr(r"\nabla \times \mathbf{v}")) == [-2 * y, -3 * z, -x]
        assert list(wb.expr(r"\nabla (x^2 y)")) == [2 * x * y, x**2, 0]
        assert wb.expr(r"\nabla^2 (x^2 y + z^3)") == 2 * y + 6 * z

    def test_identities_hold(self, wb):
        assert wb.expr(r"\nabla \times (\nabla f(x, y, z)) = \mathbf{0}")
        assert wb.expr(r"\nabla \cdot (\nabla \times (x y, y z, z x)) = 0")

    def test_in_spherical_coordinates_from_a_line_element(self, wb):
        wb.exe(SPHERICAL)
        assert wb.expr(r"\nabla \cdot \frac{\hat{\mathbf{r}}}{r^2}") == 0
        assert wb.expr(r"\nabla \cdot (r \hat{\mathbf{r}})") == 3
        f = sympy.Function("f")(r, theta, phi)
        gradient = wb.expr(r"\nabla f(r, \theta, \phi)")
        expected = [f.diff(r), f.diff(theta) / r, f.diff(phi) / (r * sympy.sin(theta))]
        assert all(same(a, b) for a, b in zip(gradient, expected, strict=True))

    def test_in_cylindrical_coordinates_a_rigid_rotation(self, wb):
        wb.exe(r"ds^2 = d\rho^2 + \rho^2\, d\varphi^2 + dz^2")
        wb.exe(r"\mathbf{v} := \omega \rho\, \hat{\boldsymbol{\varphi}}")
        assert list(wb.expr(r"\nabla \times \mathbf{v}")) == [0, 0, 2 * omega]

    def test_coordinates_known_by_their_names(self, wb):
        wb.exe(r"Coordinates := [r, \theta, \phi]")
        assert same(wb.expr(r"\nabla^2 (r^2 \cos\theta)"), 4 * sympy.cos(theta))
        assert notes(wb, r"\nabla^2 r") == [
            r"(r, \theta, \phi) are read as spherical coordinates, x, y, z = r sin θ cos φ, "
            "r sin θ sin φ, r cos θ"
        ]

    def test_coordinates_the_program_writes_x_y_and_z_in(self, wb):
        wb.exe(
            r"x := u \cos w" "\n" r"y := u \sin w" "\n" r"z := s" "\n" r"Coordinates := [u, w, s]"
        )
        assert same(wb.expr(r"\nabla^2 u^2"), 4)

    def test_the_derivative_along_a_vector(self, wb):
        k = sympy.Symbol("k")
        wb.exe(r"ds^2 = dx^2 + dy^2" "\n" r"\mathbf{u} := (k x, -k y)")
        assert list(wb.expr(r"(\mathbf{u} \cdot \nabla)\mathbf{u}")) == [k**2 * x, k**2 * y]

    def test_the_laplacian_written_with_delta(self, wb):
        wb.exe("f(x, y) := x^2 y + y^3")
        assert wb.expr(r"\Delta f") == 8 * y
        wb.exe(r"\Delta x := 0.1")
        assert wb.expr(r"\Delta x") == sympy.Float(0.1)  # the change in x, as before

    def test_the_gradient_of_a_name_that_names_nothing_is_refused(self, wb):
        found = error(wb, r"\nabla \Phi")
        assert found.hint == r"for a function of position, write \Phi(x, y, z)"

    def test_the_divergence_of_what_is_no_vector_is_refused(self, wb):
        found = error(wb, r"\nabla \cdot \mathbf{E} = 4\pi\rho")
        assert found.notes == [r"\mathbf{E} names nothing, so it is a number, not a vector"]
        wb.exe(r"\mathbf{E} := 5")
        assert error(wb, r"\nabla \cdot \mathbf{E}").notes == []  # a number, defined so

    def test_with_a_metric_of_spacetime_nabla_squared_is_the_dalembertian(self, wb):
        wb.exe(r"ds^2 = -dt^2 + dx^2 + dy^2 + dz^2")
        assert wb.expr(r"\nabla^2 (x^2 - t^2)") == 4


class TestDecision25:
    def test_at_a_point_the_hessian_and_the_gradient(self, wb):
        wb.exe(r"f(x, y) := x^2 + x y + y^2 - 3x")
        assert wb.expr(r"\nabla^2 f(0, 0)").tolist() == [[2, 1], [1, 2]]
        assert wb.expr(r"\nabla f(0, 0)").tolist() == [[-3], [0]]  # a column
        newton = r"\left[\nabla^2 f(0, 0)\right]^{-1} \nabla f(0, 0)"
        origin = r"\begin{pmatrix} 0 \\ 0 \end{pmatrix}"
        assert wb.expr(f"{origin} - {newton}").tolist() == [[2], [-1]]

    def test_it_is_said_in_a_note(self, wb):
        wb.exe(r"f(x, y) := x^2 + y^2")
        assert notes(wb, r"\nabla^2 f(1, 1)") == [
            r"\nabla^2 f(1, 1) is read as the Hessian of f at (1, 1): f is a function of the "
            r"vector of its 2 variables; \Delta f is the Laplacian"
        ]

    def test_at_its_variables_it_is_the_laplacian(self, wb):
        wb.exe(r"f(x, y) := x^2 + y^2")
        assert wb.expr(r"\nabla^2 f(x, y)") == 4

    def test_a_function_of_a_vector(self, wb):
        wb.exe(r"f(x_1, x_2) := x_1^2 + 10 x_2^2" "\n" r"\mathbf{x} := (1, 1)")
        assert wb.expr(r"f(\mathbf{x})") == 11


class TestEquations:
    def test_decided_component_by_component(self, wb):
        wb.exe(r"\mathbf{E} := (0, \cos(z - t), 0)" "\n" r"\mathbf{B} := (-\cos(z - t), 0, 0)")
        assert wb.expr(r"\nabla \times \mathbf{E} = -\partial_t \mathbf{B}")
        assert wb.expr(r"\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}")

    def test_a_component_that_differs_is_said(self, wb):
        equation = wb.expr("(1, 2) = (1, 3)")
        assert equation.holds() is False

    def test_by_a_hyperbolic_identity(self, wb):
        assert wb.expr(r"(\cosh^2 v - \sinh^2 v, 0) = (1, 0)")

    def test_a_pair_beside_a_number_is_refused_as_before(self, wb):
        found = error(wb, "[a, b] = 0")
        assert found.hint == "for the commutator, write a b - b a"

    def test_a_vector_beside_a_number_is_refused(self, wb):
        found = error(wb, r"(a, b, c) = 1")
        assert found.message == "(a, b, c) = 1 compares the list (a, b, c) with a scalar"
        wb.exe(FIELD)
        found = error(wb, r"\mathbf{v} = 1")
        assert found.message == r"\mathbf{v} = 1 equates a vector with 1, a number"

    def test_bold_and_plain_letters_are_two_names_where_the_program_defines_both(self, wb):
        # \mathbf{r} = (x, y, z), \quad r = |\mathbf{r}| was refused, as r and \mathbf{r}
        # were one name: r = |\mathbf{r}| equated the vector with its length (decision 9)
        wb.exe(r"\mathbf{r} = (x, y, z), \quad r = |\mathbf{r}|")
        assert str(wb.expr("r")) == "sqrt(x**2 + y**2 + z**2)"
        assert str(wb.expr(r"\mathbf{r} / r")) == str(wb.expr(r"\mathbf{r} / |\mathbf{r}|"))
        found = error(wb, r"\mathbf{r} = r")
        assert found.notes[-1] == (
            r"\mathbf{r} and r are two names, as the program defines both: a typeface makes "
            "a name of its own"
        )


class TestRegions:
    def test_over_a_disk(self, wb):
        integral = wb.expr(r"\iint_{x^2 + y^2 \le 1} e^{-(x^2 + y^2)} \, dA")
        assert same(integral, sympy.pi * (1 - sympy.exp(-1)))

    def test_over_a_ball(self, wb):
        R = sympy.Symbol("R")
        assert same(wb.expr(r"\iiint_{x^2 + y^2 + z^2 \le R^2} 1 \, dV"), 4 * sympy.pi * R**3 / 3)

    @pytest.mark.parametrize(
        ("written", "value"),
        [
            (r"\iiint_{[0,1]^3} x y z \, dV", sympy.Rational(1, 8)),
            (r"\iint_{[0,1] \times [0,2]} x y \, dA", 1),
        ],
    )
    def test_over_a_box(self, wb, written, value):
        assert wb.expr(written) == value

    def test_any_other_region_is_refused_as_before(self, wb):
        found = error(wb, r"\iint_D f \, dA")
        assert "integrals over regions" in found.message


class TestCurves:
    CIRCLE = r"C(t) := (\cos t, \sin t, 0), \quad 0 \le t \le 2\pi"

    def test_a_field_around_a_closed_curve(self, wb):
        wb.exe(self.CIRCLE + "\n" r"\mathbf{F} := -y\,\hat{\mathbf{x}} + x\,\hat{\mathbf{y}}")
        assert wb.expr(r"\oint_C \mathbf{F} \cdot d\mathbf{r}") == 2 * sympy.pi

    def test_along_a_curve_named_by_its_point(self, wb):
        wb.exe(r"\mathbf{r}(t) := (t, t^2, 0), \quad 0 \le t \le 1")
        line = r"\int_{\mathbf{r}} (x\,\hat{\mathbf{x}} + y\,\hat{\mathbf{y}}) \cdot d\mathbf{r}"
        assert wb.expr(line) == 1

    def test_numerically_where_there_is_no_closed_form(self, wb):
        wb.exe(r"C(t) := (t, 0, 0), \quad 0 \le t \le 1")
        line = r"\int_C e^{\cos^3 x} \hat{\mathbf{x}} \cdot d\mathbf{r}"
        assert abs(wb.expr(line) - 1.9727712499952477) < 1e-12
        assert notes(wb, line)[-1] == (
            "the integral along the curve is computed numerically, as SymPy found no closed "
            "form of it in 5 s"
        )

    def test_one_inside_another_numerically(self, wb):
        wb.exe(
            r"\mathbf{r}_1(s) := (\cos s, \sin s, 0), \quad 0 \le s \le 2\pi"
            "\n"
            r"\mathbf{r}_2(t) := (1 + \cos t, 0, \sin t), \quad 0 \le t \le 2\pi"
        )
        linking = (
            r"\oint_{\mathbf{r}_1} \oint_{\mathbf{r}_2} \frac{(\mathbf{r}_1 - \mathbf{r}_2) "
            r"\cdot (d\mathbf{r}_1 \times d\mathbf{r}_2)}{|\mathbf{r}_1 - \mathbf{r}_2|^3}"
        )
        assert abs(wb.expr(linking) + 4 * sympy.pi) < 1e-9  # Gauss: linked once
        assert notes(wb, linking)[-1] == (
            "the integrals along the curves, one inside the other, are computed numerically"
        )

    def test_a_curve_without_a_range_is_refused(self, wb):
        wb.exe(r"C(t) := (\cos t, \sin t, 0)")
        found = error(wb, r"\oint_C \mathbf{F} \cdot d\mathbf{r}")
        assert found.hint == r"state it, as 0 \le t \le 2\pi"

    def test_an_integrand_without_its_differential_is_refused(self, wb):
        wb.exe(self.CIRCLE)
        found = error(wb, r"\oint_C x^2 + y^2")
        assert "has no differential along it" in found.message


class TestRefusals:
    def test_the_three_dimensional_delta(self, wb):
        wb.exe(SPHERICAL)
        found = error(wb, r"\nabla \cdot \frac{\hat{\mathbf{r}}}{r^2} = 4\pi \delta^3(\mathbf{r})")
        assert found.message == (
            r"\delta^3(\mathbf{r}) is the Dirac delta in three dimensions, a distribution, "
            "which is not supported yet"
        )

    def test_the_gradient_of_a_vector(self, wb):
        wb.exe(r"\mathbf{u} := (x, y, z)")
        found = error(wb, r"\nabla \mathbf{u}")
        assert "the gradient of a vector" in found.message

    def test_vector_calculus_in_spacetime(self, wb):
        wb.exe(r"ds^2 = -dt^2 + dx^2 + dy^2 + dz^2")
        found = error(wb, r"\nabla \cdot (x, y, z)")
        assert found.message.startswith(r"\nabla \cdot (x, y, z) is of vector calculus")


class TestCartesianIndices:
    CARTESIAN = "ds^2 = dx^2 + dy^2 + dz^2\n"
    FIELD = r"A_{i} := [u(x, y, z), v(x, y, z), w(x, y, z)]"

    def test_repeated_lower_indices_are_summed(self, wb):
        wb.exe(self.CARTESIAN + self.FIELD)
        u, v, w = (sympy.Function(name)(x, y, z) for name in "uvw")
        assert wb.expr(r"\partial_i A_i") == u.diff(x) + v.diff(y) + w.diff(z)
        assert wb.expr(r"\delta_{ii}") == 3

    def test_the_levi_civita_symbol(self, wb):
        wb.exe(self.CARTESIAN + self.FIELD)
        curl_curl = r"\epsilon_{ijk} \partial_j (\epsilon_{klm} \partial_l A_m)"
        assert wb.expr(rf"{curl_curl} = \partial_i \partial_j A_j - \partial_j \partial_j A_i")
        assert readings(wb, r"\epsilon_{ijk}") == [
            r"vector or number: \epsilon_{ijk} is the Levi-Civita symbol, since nothing defines "
            r"\epsilon, and its 3 indices range over a space of 3 dimensions whose metric is "
            "the identity"
        ]

    def test_in_a_space_declared_as_the_reals_cubed(self, wb):
        wb.exe(r"V := \mathbb{R}^{3}" "\n" r"i, j, k, m, n \in V")
        identity = (
            r"\epsilon_{ijk} \epsilon_{imn} = \delta_{jm} \delta_{kn} - \delta_{jn} \delta_{km}"
        )
        assert wb.expr(identity)

    def test_in_spacetime_an_index_is_still_summed_once_up_once_down(self, wb):
        wb.exe(r"ds^2 = -dt^2 + dx^2" "\n" r"B_{i} := [f(t, x), g(t, x)]")
        found = error(wb, r"\partial_i B_i")
        assert "appears twice as a lower index" in found.message


class TestFrame:
    def test_scale_factors_are_the_positive_roots_books_write(self):
        r_, theta_ = sympy.symbols("r theta", real=True)
        assert vectors.positive_root(r_**2 * sympy.sin(theta_) ** 2) == r_ * sympy.sin(theta_)
        assert vectors.positive_root(1 + 4 * r_**2) == sympy.sqrt(1 + 4 * r_**2)

    def test_coordinates_that_are_not_orthogonal_are_refused(self):
        g = sympy.ImmutableMatrix([[1, x], [x, 1]])
        with pytest.raises(EvaluationError, match="not orthogonal"):
            vectors.scale_factors(g, (x, y))
