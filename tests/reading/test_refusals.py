r"""What the reader refuses rather than misreads: the guards of wave 0 of
``curriculum/ROADMAP.md``. A wrong answer given confidently is the worst thing the language
can do, so where a form would be read with another meaning, it is refused, saying what was
read, why, and what to write instead; and the hint works."""

import pytest
import sympy

from relativisticpy.errors import CompileError, EvaluationError, IndexNotationError

x, theta = sympy.symbols("x theta")


def error(wb, source, kind=CompileError):
    with pytest.raises(kind) as raised:
        wb.exe(source)
    return raised.value


def decisions(wb, source, kind):
    """The decisions of ``kind`` taken in reading the last statement of ``source``."""
    reading = wb.explain(source)[-1]
    return [(d.written, d.reading, d.why) for d in reading.decisions if d.kind == kind]


class TestColonEqualsDefinesOrRefuses:
    """Guard 1: ``:=`` defines, or is refused saying how its left side was read; it never
    makes an equation, as ``\\psi_n(x) := ...`` made Eq(psi_n x, ...) and defined nothing."""

    def test_a_subscripted_function_is_refused_with_its_family(self, wb):
        found = error(wb, r"\psi_n(x) := \sin(n x)")
        assert found.message == r"cannot define \psi_n(x), which is read as \psi_n times x"
        assert found.notes[0] == (
            r"a name with a subscript, a superscript or an accent, as \psi_n, is not yet the "
            "name of a function, nor of a family of them"
        )
        assert found.hint == (
            r"define a function of n too, as in \psi(n, x) := ..., and write \psi(1, x) for "
            r"\psi_1(x)"
        )
        assert wb.expr(r"\psi(n, x) := \sin(n x)" "\n" r"\psi(1, x)") == sympy.sin(x)

    def test_a_function_with_a_superscript(self, wb):
        found = error(wb, r"Y_2^1(\theta, \phi) := \sin\theta")
        assert found.message == (
            r"cannot define Y_2^1(\theta, \phi), which is read as Y_2^1 times (\theta, \phi)"
        )
        hint = r"give the function a plain name, as in Y(\theta, \phi) := ..."
        assert found.hint == hint
        assert wb.expr(r"Y(\theta, \phi) := \sin\theta" "\n" r"Y(\theta, 0)") == sympy.sin(theta)

    def test_a_product_of_names(self, wb):
        found = error(wb, r"\Delta\phi := 2")
        assert found.message == (
            r"cannot define \Delta\phi, which is read as the product \Delta times \phi"
        )
        assert found.notes == [
            ":= gives a value to a name, a tensor or a function, not to a product"
        ]
        assert found.hint == r"to name one quantity, write one name, as in \Delta_{\phi} := ..."
        assert wb.expr(r"\Delta_{\phi} := 2" "\n" r"3 \Delta_{\phi}") == 6

    def test_a_product_of_subscripted_names(self, wb):
        found = error(wb, r"\omega_e x_e := 52.8")
        assert found.message == (
            r"cannot define \omega_e x_e, which is read as the product \omega_e times x_e"
        )
        assert found.hint == (
            "to name one quantity, give it one name, and write that name where it is used"
        )

    def test_a_time_derivative(self, wb):
        found = error(wb, r"\dot{x} := x(\alpha - \beta y)")
        assert found.message == (
            r"cannot define \dot{x}: it is the time derivative of x, which takes its value "
            "from x"
        )
        assert found.hint.startswith("define x as a function of time, as in x(t) := ...")

    def test_a_function_of_what_is_not_a_name(self, wb):
        found = error(wb, r"H(e^{j\omega}) := 1 + 2e^{-j\omega}")
        assert found.message == (
            r"cannot define H(e^{j\omega}): the parameters of a function are names, and "
            r"e^{j\omega} is not"
        )
        assert found.hint == (
            r"define it with names as its parameters, as in H(z) := ..., and then write "
            r"H(e^{j\omega})"
        )
        wb.exe("H(z) := 1 + 2/z")
        assert wb.expr(r"H(e^{i\omega})") == 1 + 2 * sympy.exp(-sympy.I * sympy.Symbol("omega"))

    def test_nothing_is_defined_and_the_reading_says_how_it_was_read(self, wb):
        source = r"\psi_n(x) := \sin(n x)"
        assert decisions(wb, source, "product or call") == [
            (
                r"\psi_n(x)",
                r"\psi_n times x",
                r"\psi_n is not a function: a name with scripts is a symbol",
            )
        ]
        assert wb.explain(source)[0].error is not None
        assert next(iter(wb.each(source))).error is not None
        assert wb.names() == {}

    def test_equiv_still_states_an_identity(self, wb):
        # Carroll's chart map, \phi_1(x^1, x^2, x^3) \equiv (y^1, y^2): a relation, not refused
        (step,) = list(wb.each(r"\Delta\phi \equiv \phi_2 - \phi_1"))
        assert step.error is None

    def test_a_function_written_as_books_write_one_is_defined(self, wb):
        # Bracewell's sinc x := sin(πx)/(πx): the argument after the name, without brackets
        wb.exe(r"\operatorname{sinc} x := \frac{\sin \pi x}{\pi x}")
        assert wb.expr(r"\operatorname{sinc}\left(\frac{1}{2}\right)") == 2 / sympy.pi
        assert decisions(wb, r"\mathrm{sinc}\, x := 1", "product or call") == [
            (
                r"\mathrm{sinc}\, x",
                "sinc(x)",
                "a name written as an operator, with one name after it, is a function of it",
            )
        ]

    def test_an_operator_name_before_more_than_one_name_is_refused(self, wb):
        found = error(wb, r"\operatorname{sinc} x y := 2")
        assert found.message.startswith(r"cannot define \operatorname{sinc} x y")

    def test_brackets_and_numbers_say_how_they_were_read(self, wb):
        found = error(wb, r"\langle p, q \rangle := \int_{-1}^{1} p(x) q(x) \, dx")
        assert found.message == (
            r"cannot define \langle p, q \rangle, which is read as the pair (p, q)"
        )
        assert found.hint == "for a function of p, q, define one, as in f(p, q) := ..."
        found = error(wb, r"\mathbf{1} := \begin{pmatrix} 1 \\ 1 \end{pmatrix}")
        assert found.message == r"cannot define \mathbf{1}, which is read as the number 1"

    def test_a_primed_function_name(self, wb):
        found = error(wb, r"\rho'(x, t) := R e^{i(k x - \omega t)}")
        assert found.message == (
            r"cannot define \rho'(x, t): a function is named with a plain name, and \rho' is "
            "not one"
        )
        assert found.notes == [r"a prime on a name, as on \rho', is a derivative"]
        assert found.hint == "give the function a plain name, as in f(x, t) := ..."


class TestEqualsDefinesAFunctionOrRefuses:
    """Guard 1: ``=`` whose left side is a name with arguments defines that function, or is
    refused; as a product of a symbol and its argument, U_{eff}(r) = ... was an equation,
    and the minimum of U_{eff} was [], from dU_{eff}/dr = 0."""

    def test_a_subscripted_name_with_arguments(self, wb):
        found = error(wb, r"U_{\text{eff}}(r) = -\frac{k}{r} + \frac{\ell^2}{2 \mu r^2}")
        assert found.message == (
            r"U_{\text{eff}}(r) is read as U_{\text{eff}} times r, so = neither defines nor "
            "compares it"
        )
        assert found.hint == "give the function a plain name, as in U(r) := ..."
        wb.exe(r"U(r) := -\frac{k}{r} + \frac{\ell^2}{2 \mu r^2}")
        k, ell, mu = sympy.symbols("k ell mu")
        assert wb.expr(r"solve\left(\frac{d U(r)}{d r} = 0, r\right)") == [ell**2 / (k * mu)]

    def test_a_function_that_is_defined_is_compared(self, wb):
        wb.exe("f(x) := x^2")
        assert next(iter(wb.each("f(x) = x^3"))).error is None


class TestDotsKeepTheirPowers:
    """``\\dot{a}^2`` is (da/dt)^2: a power stays outside the time derivative, and only an
    index or a subscript moves inside, as in ``\\dot{x}^\\mu``. It was d(a^2)/dt."""

    def test_a_power_of_a_dotted_function(self, wb):
        t = sympy.Symbol("t")
        assert wb.expr("a(t) := t^3\n" r"\dot{a}^2") == 9 * t**4  # never 6 t^5
        assert wb.expr(r"\ddot{a}") == 6 * t
        assert (r"\dot{a}^2", r"a power of \dot{a}", "2 is no index of a") in decisions(
            wb, r"\dot{a}^2", "index or power"
        )

    def test_a_power_of_a_dotted_call(self, wb):
        a, t = sympy.Function("a"), sympy.Symbol("t")
        assert wb.expr(r"\dot{a(t)}^2") == sympy.Derivative(a(t), t) ** 2

    def test_a_dotted_symbol_nothing_defines_is_refused_as_before(self, wb):
        # \dot\theta alone is a symbol of its own, which d/dt would take for a constant: its
        # square stays refused until a Lagrangian's velocities are read, never computed wrong
        error(wb, r"\dot\theta^2", EvaluationError)


class TestDIsAVariableWithoutAnIndex:
    """Decision 18 of design/notation.md: d, the partial derivative's name, is a variable
    where it has no index and no differential below it, as the diameter in
    ``\\frac{\\pi d^2}{4}``. It was \\partial^2."""

    def test_a_diameter_squared(self, wb):
        d = sympy.Symbol("d")
        assert wb.expr(r"\frac{\pi d^2}{4}") == sympy.pi * d**2 / 4
        moment = wb.expr(r"\left. I \frac{\pi d^2}{4} \right|_{I = 2,\ d = 0.1}")
        assert abs(float(moment) - 0.005 * float(sympy.pi)) < 1e-15
        assert ("d^2", "d to a power", "d is a derivative only with an index") in decisions(
            wb, r"\frac{\pi d^2}{4}", "index or power"
        )

    def test_an_index_or_a_differential_below_makes_it_the_derivative(self, wb):
        t, x = sympy.symbols("t x")
        wb.exe("Coordinates := [t, x]")
        assert wb.expr("d_{a} (t^2 x)").components == sympy.Array([2 * t * x, t**2])
        assert decisions(wb, r"\frac{D^2}{d\tau^2} S", "index or power") == []


class TestPairsAreNotScalars:
    """Guard 4, in part: brackets of two items are read as a pair, so a pair equated to a
    scalar, a function of a pair and ``\\langle u, v \\rangle`` are refused, saying what
    the brackets often are in books. ``\\{L_x, L_y\\} = L_z`` was a silent False."""

    ANGULAR = r"L_x = y p_z - z p_y, \quad L_y = z p_x - x p_z, \quad L_z = x p_y - y p_x"

    def test_a_poisson_bracket_read_as_a_pair(self, wb):
        wb.exe(self.ANGULAR)
        found = error(wb, r"\{L_x, L_y\} = L_z")
        assert found.message == r"\{L_x, L_y\} = L_z compares the pair (L_x, L_y) with a scalar"
        assert found.notes == [
            "L_z is a scalar, and the pair (L_x, L_y) never equals one",
            "in these brackets, two items are often a Poisson bracket, which is not supported yet",
        ]
        assert found.hint == (
            r"for the Poisson bracket, write it out, as in \frac{\partial L_x}{\partial q} "
            r"\frac{\partial L_y}{\partial p} - \frac{\partial L_x}{\partial p} "
            r"\frac{\partial L_y}{\partial q}, summed over the canonical pairs (q, p)"
        )
        written_out = " + ".join(
            rf"\frac{{\partial L_x}}{{\partial {q}}} \frac{{\partial L_y}}{{\partial {p}}} "
            rf"- \frac{{\partial L_x}}{{\partial {p}}} \frac{{\partial L_y}}{{\partial {q}}}"
            for q, p in (("x", "p_x"), ("y", "p_y"), ("z", "p_z"))
        )
        assert wb.expr(f"{written_out} = L_z") is sympy.true  # the hint works

    def test_a_commutator_read_as_a_list(self, wb):
        wb.exe("a := 2\nb := 3")
        assert error(wb, "[a, b] = 0").hint == "for the commutator, write a b - b a"

    def test_a_chart_map_is_not_refused(self, wb):
        # Carroll's \phi_1(x^1, x^2, x^3) \equiv (y^1, y^2): a function may take its values
        # in pairs, so the equation stays, and a check says it relates
        (step,) = list(wb.each(r"\phi_1(x^1,x^2,x^3) \equiv (y^1,y^2)"))
        assert step.error is None

    def test_a_function_of_a_pair(self, wb):
        found = error(wb, r"\sin((1, 2))")
        assert found.message == "sin is applied to (1, 2), which is read as the pair (1, 2)"
        assert found.notes == ["sin applies to a number or an expression, not to a pair or a list"]
        assert found.hint == "apply it to each item, as in sin(1)"

    def test_an_inner_product_in_angle_brackets(self, wb):
        for written in (r"\langle (1, 2), (3, 4) \rangle", r"\sqrt{\langle p, p \rangle}"):
            assert "not as an inner product" in error(wb, written).message
        found = error(wb, r"\sqrt{\langle p, p \rangle}")
        assert found.message == (
            r"\langle p, p \rangle is read as the pair (p, p), not as an inner product"
        )
        assert found.hint == (
            r"for the inner product of vectors, write them with an index over a space and "
            r"contract it: \mathcal{V} := \mathbb{R}^{2}, a \in \mathcal{V}, then p^a q_a"
        )
        space = "\\mathcal{V} := \\mathbb{R}^{2}\na \\in \\mathcal{V}\n"
        assert wb.expr(space + "p^a := [1, 2]\nq^a := [3, 4]\np^a q_a") == 11  # the hint works

    def test_dirac_notation(self, wb):
        found = error(wb, r"\langle 0 | T\{\phi(x_1)\phi(x_2)\} | 0 \rangle = 2")
        assert found.message == (
            r"\langle 0 | T\{\phi(x_1)\phi(x_2)\} | 0 \rangle is read as a product with absolute "
            "values, not as Dirac's notation"
        )


class TestScriptedNamesWhoseBaseIsDefined:
    """Guard 3: a subscripted name nothing defines, whose base is defined so that it means a
    member or an element, is refused, where it was a symbol of its own: c_1 after c_n, x_i
    of a list summed over i, a recurrence, f^{(4)} of a function, k_{1\\mu} beside k_1^\\mu."""

    def test_a_member_of_a_family(self, wb):
        wb.exe("c_n := 1/n^2")
        found = error(wb, "c_1")
        assert (
            found.message
            == "c_1 names nothing, and c_n is a family in n, which is not supported yet"
        )
        assert found.notes == [
            "read as written, c_1 would be a symbol of its own, not c_n at n = 1"
        ]
        assert (
            found.hint == "define c as a function of n, as in c(n) := ..., and write c(1) for c_1"
        )
        assert wb.expr("c(n) := 1/n^2\nc(1)") == 1  # the hint works

    def test_members_written_in_latex(self, wb):
        wb.exe(r"E_k := \alpha + 2\beta \cos\frac{2\pi k}{6}")
        found = error(wb, r"E_\pi := 2E_0 + 2E_1 + 2E_{-1}")
        assert found.message.startswith("E_0 names nothing, and E_k is a family in k")
        wb.exe("a_n := n^2")
        found = error(wb, r"\frac{a_{n+1}}{a_n}")
        assert found.hint == (
            "define a as a function of n, as in a(n) := ..., and write a(n+1) for a_{n+1}"
        )

    def test_names_that_are_no_members_are_symbols(self, wb):
        x_0, v, y = sympy.symbols("x_0 v x")
        assert wb.expr("x_n := x_0 + n d\nx_0 + 1") == x_0 + 1  # a parameter of the family
        assert wb.expr("v_x := v \\cos x\nv_y := 2\nv_x + v_y") == v * sympy.cos(y) + 2
        assert wb.expr("p := (p_x, p_y)\np_x") == sympy.Symbol("p_x")  # an item of the pair

    def test_an_element_of_a_list(self, wb):
        wb.exe("x := [2, 4, 4, 4, 5, 5, 7, 9]\nn := 8")
        found = error(wb, r"\frac{1}{n} \sum_{i=1}^{n} x_i")
        assert found.message == (
            "x_i names nothing, and x is a list, whose elements are not written with a "
            "subscript yet"
        )
        assert found.hint == (
            r"give the list an index over a space: \mathcal{N} := \mathbb{R}^{8}, i \in "
            r"\mathcal{N} and x^i := [...]; then x^i 1_i is the sum of its elements"
        )
        space = "\\mathcal{N} := \\mathbb{R}^{8}\ni \\in \\mathcal{N}\n"
        assert wb.expr(space + "x^i := [2, 4, 4, 4, 5, 5, 7, 9]\nx^i 1_i") == 40

    @pytest.mark.parametrize(
        ("source", "message"),
        [
            (
                "T_0 = 0\nT_n = 2 T_{n-1} + 1, \\quad n > 0",
                "T_n is defined by a recurrence, through T_{n-1}, which is not supported yet",
            ),
            (
                "p_0 := \\frac{\\pi}{4}\np_n := p_{n-1} - 1",
                "p_n is defined by a recurrence, through p_{n-1}, which is not supported yet",
            ),
            (
                "C_0 = 1\nC_{n+1} = \\sum_{k=0}^{n} C_k C_{n-k}",
                "C_{n+1} is defined by a recurrence, through C_{n-k}, which is not supported yet",
            ),
        ],
    )
    def test_a_recurrence(self, wb, source, message):
        found = error(wb, source)
        assert found.message == message
        assert found.hint.startswith("give it in closed form, as a function of n:")

    def test_an_order_on_a_function(self, wb):
        wb.exe(r"f(x) := \sec x")
        found = error(wb, r"\frac{f^{(4)}(0)}{4!}")
        assert found.message == (
            "f^{(4)} is read as a name of its own, not as a derivative of the function f"
        )
        assert found.hint == "for its derivative of order 4, write f''''(x)"
        assert wb.expr(r"\frac{f''''(0)}{4!}") == sympy.Rational(5, 24)  # the hint works
        assert wb.expr("E^{(2)} := 3\nE^{(2)} + 1") == 4  # an order on a value is a label

    def test_an_index_on_a_numbered_tensor(self, wb):
        wb.exe("ds^2 = -dt^2 + dx^2 + dy^2 + dz^2\nk_1^\\mu := (E_1, E_1, 0, 0)")
        found = error(wb, r"k_1^\mu k_{1\mu}")
        assert found.message == (
            r"k_{1\mu} is read as one symbol, not as the tensor k_1 with its index \mu"
        )
        assert found.hint == r"write k_1{}_{\mu}"
        assert wb.expr(r"k_1^\mu k_1{}_{\mu}") == 0  # the hint works: a null vector


class TestOperatorsReadWithAnotherMeaning:
    """Guard 4: an operator the lexer reads as a product, a measure, a derivative or an
    exponential that would be computed with another meaning is refused until it is built:
    \\otimes of matrices, \\times of unit vectors nothing defines, d\\Omega, the derivative of
    a list, a Jacobian, the exponential of a matrix, and the bracket of two matrices."""

    SIGMA_X = r"\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}"
    SIGMA_Y = r"\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}"

    def test_a_kronecker_product(self, wb):
        found = error(wb, rf"{self.SIGMA_X} \otimes {self.SIGMA_Y}")
        assert found.message.endswith(
            "is read as their matrix product, not as their Kronecker product"
        )
        assert found.notes == ["the Kronecker product of matrices is not supported yet"]
        wb.exe("Coordinates := [t, x]\nA_{ab} := [[1, 2], [3, 4]]")
        assert wb.expr(r"A_{ab} \otimes A_{cd}").components.shape == (2, 2, 2, 2)  # with indices

    def test_a_cross_product_of_unit_vectors(self, wb):
        wb.exe(r"\mathbf{E} := E_0 \hat{\mathbf{x}}" "\n" r"\mathbf{B} := B_0 \hat{\mathbf{y}}")
        found = error(wb, r"\mathbf{E} \times \mathbf{B}")
        assert found.message == (
            r"\mathbf{E} \times \mathbf{B} is read as a product of symbols, not as a cross "
            r"product: \hat{x} and \hat{y} name nothing"
        )
        assert wb.expr(r"2 \times 10^{-5}") == sympy.Rational(1, 50000)  # numbers multiply

    def test_a_solid_angle(self, wb):
        found = error(wb, r"\int d\Omega \, \frac{\alpha^2}{4s} \left(1 + \cos^2\theta\right)")
        assert found.message == (
            r"d\Omega is read as the differential of a variable \Omega the integrand does not "
            r"mention, so the integral would be \Omega times it"
        )
        alpha, s = sympy.symbols("alpha s")
        over_angles = (
            r"\int_0^{2\pi} \int_0^{\pi} \frac{\alpha^2}{4s} \left(1 + \cos^2\theta\right) "
            r"\sin\theta \, d\theta \, d\phi"
        )
        assert wb.expr(over_angles) == 4 * sympy.pi * alpha**2 / (3 * s)  # the hint works

    def test_the_derivative_of_a_list(self, wb):
        wb.exe(r"\mathbf{x} := [r \cos\theta, r \sin\theta, z]")
        found = error(wb, r"\partial_\theta \mathbf{x}")
        assert found.message == r"the derivative of a list by \theta is not taken yet"
        assert found.hint == (
            r"give the list an index over a space, as in x^i with i \in \mathcal{E} := "
            r"\mathbb{R}^{3}, and differentiate that: \frac{\partial}{\partial \theta} x^i"
        )
        space = "\\mathcal{E} := \\mathbb{R}^{3}\ni \\in \\mathcal{E}\n"
        derivative = wb.expr(
            space + r"x^i := [r \cos\theta, r \sin\theta, z]"
            "\n"
            r"\frac{\partial}{\partial \theta} x^i"
        )
        r, theta = sympy.symbols("r theta")
        assert derivative.components == sympy.Array(
            [-r * sympy.sin(theta), r * sympy.cos(theta), 0]
        )

    def test_a_jacobian(self, wb):
        wb.exe(r"x := r \cos\theta" "\n" r"y := r \sin\theta")
        found = error(wb, r"\frac{\partial(x, y)}{\partial(r, \theta)}")
        assert found.message.endswith(
            "is read as the derivative of a list by a list, not as a Jacobian"
        )
        assert wb.expr(found.hint.removeprefix("write the Jacobian as a determinant: ")) == (
            sympy.Symbol("r")
        )

    def test_the_exponential_of_a_matrix(self, wb):
        wb.exe(r"A := \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}")
        for written in (r"e^{A t}", r"\exp(2 A)"):
            assert error(wb, written).message == (
                f"{written} is the exponential of a matrix, which is not computed yet"
            )

    def test_the_bracket_of_two_matrices(self, wb):
        wb.exe(f"K_1 := {self.SIGMA_X}\nK_2 := {self.SIGMA_Y}")
        found = error(wb, "[K_1, K_2]")
        assert (
            found.message == "[K_1, K_2] is read as a list of two matrices, not as their commutator"
        )
        assert found.hint == "for the commutator, write K_1 K_2 - K_2 K_1"
        assert error(wb, r"\{K_1, K_2\}").hint == "for the anticommutator, write K_1 K_2 + K_2 K_1"
        commutator = wb.expr("K_1 K_2 - K_2 K_1")  # the hint works
        assert sympy.Matrix(commutator) == sympy.Matrix([[2 * sympy.I, 0], [0, -2 * sympy.I]])


class TestConstrainedSymbols:
    """Guard 6: the derivative of a name nothing defines by one an earlier equation ties it
    to is refused, where it was the derivative of a constant, 0; and so is \\oint with no
    path, which was the indefinite integral."""

    def test_an_implicit_derivative(self, wb):
        (step,) = list(wb.each("x^3 + y^3 = 6xy"))
        assert step.error is None
        found = error(wb, r"\frac{dy}{dx}")
        assert found.message == (
            r"\frac{dy}{dx} is not 0: y is tied to x by the equation on line 1, x^3 + y^3 = 6xy"
        )
        assert found.hint == (
            r"write y(x) for y in the equation, and solve its derivative, as in "
            r"solve(\frac{d}{dx}(...) = 0, y'(x))"
        )
        x, y = sympy.symbols("x y")
        (slope,) = wb.expr(r"solve(\frac{d}{dx}(x^3 + y(x)^3 - 6 x y(x)) = 0, y'(x))")
        expected = (2 * y - x**2) / (y**2 - 2 * x)
        assert sympy.simplify(slope.subs(sympy.Function("y")(x), y) - expected) == 0

    def test_a_derivative_at_a_point(self, wb):
        wb.exe(r"m = \tanh\left(\frac{z J m + h}{k T}\right)")
        found = error(wb, r"\left. \frac{\partial m}{\partial h} \right|_{h = 0, m = 0}")
        assert found.message.startswith(
            r"\frac{\partial m}{\partial h} is not 0: m is tied to h by the equation on line 1"
        )

    def test_a_name_no_equation_ties_is_a_constant(self, wb):
        wb.exe("x^2 + y^2 = 1")
        assert wb.expr(r"\frac{dz}{dx}") == 0

    def test_its_own_equation_does_not_tie_a_derivative(self, wb):
        # a check computes the sides of d^2 t/d\tau^2 = 0 after the equation is read
        (step,) = list(wb.each(r"\frac{d^2 t}{d\tau^2} = 0"))
        assert step.error is None
        assert step.zero

    def test_a_contour_integral_with_no_path(self, wb):
        found = error(wb, r"J = \oint p\, dq")
        assert (
            found.message == r"\oint p\, dq is an integral around a closed path, which is not given"
        )
        assert wb.expr(r"\int_0^1 p \, dq") == sympy.Symbol("p")


class TestDependencesAreKept:
    """Misreadings like the guards', each with a clear remedy: a derivative by an expression
    whose d's cancelled, a derivative by \\bar z, a definition that relabelled its free
    index, and a function of a coordinate whose tensors kept the coordinate."""

    def test_a_derivative_by_an_expression(self, wb):
        wb.exe("p := p_0 e^{-B/T}")
        found = error(wb, r"-R \frac{d \ln p}{d(1/T)}")
        assert found.message == (
            r"\frac{d \ln p}{d(1/T)} is read as a fraction whose d's cancel, not as the "
            "derivative by 1/T"
        )
        assert found.hint == (
            r"write it by the chain rule, over dT: \frac{\frac{d}{dT} \ln p}"
            r"{\frac{d}{dT}\left(1/T\right)}"
        )
        written = found.hint.removeprefix("write it by the chain rule, over dT: ")
        assert wb.expr(f"-R {written}") == sympy.Symbol("B") * sympy.Symbol("R")  # it works

    def test_a_derivative_by_a_conjugate(self, wb):
        found = error(wb, r"\frac{\partial}{\partial \bar{z}} |z|^2")
        assert found.message == (
            r"\frac{\partial}{\partial \bar{z}} |z|^2 is read as a derivative by a symbol of its "
            r"own, \bar{z}, which is 0, not as the derivative by the conjugate of z"
        )
        wb.exe("Reals := [x, y]\nz := x + i y")
        x, y = sympy.symbols("x y", real=True)
        wirtinger = (
            r"\frac{1}{2}\left(\frac{\partial |z|^2}{\partial x} + i \frac{\partial |z|^2}"
            r"{\partial y}\right)"
        )
        assert wb.expr(wirtinger) == x + sympy.I * y  # the hint works: z

    def test_a_definition_keeps_its_free_indices(self, wb):
        wb.exe(
            "\\mathcal{E} := \\mathbb{R}^{3}\ni, j \\in \\mathcal{E}\n"
            "I_{ij} := [[8, -3, -3], [-3, 8, -3], [-3, -3, 8]]\n\\omega_j := (1, 0, 0)"
        )
        found = error(wb, r"L_i = I_{ij} \omega_i", IndexNotationError)
        assert found.message == (
            "the two sides have different free indices: _{i} on the left, _{j} on the right"
        )

    def test_a_function_of_a_coordinate_is_taken_at_its_argument(self, wb):
        wb.exe(
            "g_{\\mu\\nu} := Schwarzschild(M)\n"
            "u^\\mu := \\left[\\left(1 - \\frac{2M}{r}\\right)^{-1/2}, 0, 0, 0\\right]\n"
            "k_\\mu := \\left[-E, \\frac{E}{1 - 2M/r}, 0, 0\\right]\n"
            "\\omega(r) := -k_\\mu u^\\mu"
        )
        E, M, r_e = sympy.symbols("E M r_e")
        assert sympy.simplify(wb.expr(r"\omega(r_e)") - E / sympy.sqrt(1 - 2 * M / r_e)) == 0

    def test_a_coefficient_read_as_a_factor(self, wb):
        catalan = r"\frac{1 - \sqrt{1 - 4x}}{2x}"
        found = error(wb, rf"[x^{{10}}] {catalan}")
        assert found.message == (
            "[x^{10}] is read as the factor x^{10}, not as the coefficient of x^{10} in what "
            "follows"
        )
        assert found.hint == "read it from the series, series(f, x, 0, 11)"
        x = sympy.Symbol("x")
        assert wb.expr(f"series({catalan}, x, 0, 11)").coeff(x, 10) == 16796  # the hint works
        assert wb.expr("[x + 1] y") == (x + 1) * sympy.Symbol("y")  # brackets that group
