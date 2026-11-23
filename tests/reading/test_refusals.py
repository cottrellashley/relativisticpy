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

    def test_a_function_with_scripts_is_defined(self, wb):
        # decisions 5 and 6: the scripts are part of the function's name, which wins over
        # the spherical harmonic books write so
        wb.exe(r"Y_2^1(\theta, \phi) := \sin\theta")
        assert wb.expr(r"Y_2^1(\theta, 0)") == sympy.sin(theta)

    def test_a_product_of_names(self, wb):
        found = error(wb, r"\alpha\phi := 2")
        assert found.message == (
            r"cannot define \alpha\phi, which is read as the product \alpha times \phi"
        )
        assert found.notes == [
            ":= gives a value to a name, a tensor or a function, not to a product"
        ]
        assert found.hint == r"to name one quantity, write one name, as in \alpha_{\phi} := ..."
        assert wb.expr(r"\alpha_{\phi} := 2" "\n" r"3 \alpha_{\phi}") == 6
        # \Delta\phi, the change in \phi, is one name (tests/reading/test_labels.py)

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
        found = error(wb, r"H(\sqrt{\omega}) := 1 + 2\omega")
        assert found.message == (
            r"cannot define H(\sqrt{\omega}): the parameters of a function are names, and "
            r"\sqrt{\omega} is not"
        )
        assert found.hint == (
            r"define it with names as its parameters, as in H(z) := ..., and then write "
            r"H(\sqrt{\omega})"
        )
        wb.exe("H(z) := 1 + 2/z")
        assert wb.expr(r"H(e^{i\omega})") == 1 + 2 * sympy.exp(-sympy.I * sympy.Symbol("omega"))

    def test_nothing_is_defined_and_the_reading_says_how_it_was_read(self, wb):
        source = r"\omega_e x_e := 52.8"
        assert decisions(wb, source, "product or call") == [
            (
                r"\omega_e x_e",
                r"\omega_e times x_e",
                "in LaTeX names written side by side multiply",
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
        found = error(wb, r"\left( p, q \right) := \int_{-1}^{1} p(x) q(x) \, dx")
        assert (
            found.message == r"cannot define \left( p, q \right), which is read as the pair (p, q)"
        )
        assert found.hint == "for a function of p, q, define one, as in f(p, q) := ..."
        found = error(wb, r"\mathbf{2} := \begin{pmatrix} 2 \\ 2 \end{pmatrix}")
        assert found.message == r"cannot define \mathbf{2}, which is read as the number 2"

    def test_a_primed_function_name_is_defined(self, wb):
        # decision 7: on the left of :=, a prime is part of the name
        wb.exe(r"\rho'(x, t) := R e^{i(k x - \omega t)}")
        assert wb.expr(r"\rho'(0, 0)") == sympy.Symbol("R")

    def test_angle_brackets_define_a_bracket(self, wb):
        # \langle p, q \rangle := ... is no pair: the program's own bracket (decision 13)
        wb.exe(r"\langle p, q \rangle := \int_{-1}^{1} p(x) q(x) \, dx")
        assert wb.expr(r"\langle 1, x^2 \rangle") == sympy.Rational(2, 3)


class TestEqualsDefinesAFunctionOrRefuses:
    """Guard 1: ``=`` whose left side is a name with arguments defines that function, or is
    refused; as a product of a symbol and its argument, U_{eff}(r) = ... was an equation,
    and the minimum of U_{eff} was [], from dU_{eff}/dr = 0."""

    def test_a_subscripted_name_with_arguments(self, wb):
        wb.exe(r"U_{\text{eff}}(r) = -\frac{k}{r} + \frac{\ell^2}{2 \mu r^2}")
        k, ell, mu = sympy.symbols("k ell mu")
        found = wb.expr(r"solve\left(\frac{d U_{\text{eff}}(r)}{d r} = 0, r\right)")
        assert found == [ell**2 / (k * mu)]

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
    """Guard 4, in part: brackets of two items that are neither operators nor matrices are
    read as a pair, so a pair equated to a scalar and a function of a pair are refused, as
    are angle brackets around what is no vector, saying what the brackets often are in
    books. ``\\{L_x, L_y\\} = L_z`` was a silent False."""

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

    def test_a_relation_given_to_a_function_of_numbers(self, wb):
        # the review's programs, each an internal error: P(X < 0) of the program's own P
        wb.exe("P(x) := x^2 + 1\n" r"X \sim \mathcal{N}(0,1)")
        found = error(wb, "P(X < 0)")
        assert found.message == "P(X < 0) gives P the relation X < 0, and P takes a number"
        assert found.notes == [
            "P is the function defined on line 1",
            "a relation, as X < 0 is, is true or false, not a number",
        ]
        assert found.hint == r"for the probability of the event, write \Pr(X < 0)"
        assert wb.expr(found.hint.removeprefix("for the probability of the event, write ")) == (
            sympy.Rational(1, 2)
        )  # the hint works
        wb.exe("f(x) := x^2")
        assert error(wb, "f(1 < 2)").message == (
            "f(1 < 2) gives f the relation 1 < 2, and f takes a number"
        )
        assert error(wb, r"\sin(1 < 2)").hint is None

    def test_a_relation_a_function_hands_on_is_as_it_was(self, wb):
        # a function of an equation is none of numbers: it computes nothing with it
        wb.exe("S(e) := solve(e, x)")
        assert wb.expr("S(x^2 = 4)") == [-2, 2]

    def test_angle_brackets_around_what_is_no_vector(self, wb):
        assert wb.expr(r"\langle (1, 2), (3, 4) \rangle") == 11  # an inner product, now read
        found = error(wb, r"\sqrt{\langle p, p \rangle}")
        assert found.message == (
            r"\langle p, p \rangle is angle brackets around p, p, which are not two vectors, "
            "and no bracket the program defines"
        )
        assert found.hint == (
            r"define the bracket, as in \langle p, q \rangle := \int_{-1}^{1} p(x) q(x) \, dx"
        )
        wb.exe(found.hint.removeprefix("define the bracket, as in "))
        assert wb.expr(r"\sqrt{\langle x, x \rangle}") == sympy.sqrt(6) / 3  # the hint works

    def test_a_time_ordered_product_between_a_bra_and_a_ket(self, wb):
        found = error(wb, r"\langle 0 | T\{\phi(x_1)\phi(x_2)\} | 0 \rangle = 2")
        assert found.message == (
            r"T\{\phi(x_1)\phi(x_2)\}, between a bra and a ket, is read as a time-ordered "
            "product of fields, which is not supported yet"
        )


class TestScriptedNamesWhoseBaseIsDefined:
    """Guard 3: a subscripted name nothing defines, whose base is defined so that it means
    an element, is refused, where it was a symbol of its own: x_i of a list summed over i,
    f^{(2)} of a function of several variables, and a recurrence in a label that means
    something already. A member of a family, a recurrence, f^{(4)} of a function of one
    variable and k_{1\\mu} beside k_1^\\mu are read (tests/reading/test_families.py,
    test_functions.py and test_labels.py)."""

    def test_names_that_are_no_members_are_symbols(self, wb):
        x_0, v, y = sympy.symbols("x_0 v x")
        assert wb.expr("x_n := x_0 + n d\nx_0 + 1") == x_0 + 1  # a parameter of the family
        assert wb.expr("v_x := v \\cos x\nv_y := 2\nv_x + v_y") == v * sympy.cos(y) + 2
        assert wb.expr("p := (p_x, p_y)\np_x") == sympy.Symbol("p_x")  # an item of the pair

    def test_an_element_of_a_list(self, wb):
        wb.exe("x := [2, 4, 4, 4, 5, 5, 7, 9]\nn := 8")
        # a sum over the whole list counts its elements from its lower limit (decision 17)
        assert wb.expr(r"\frac{1}{n} \sum_{i=1}^{n} x_i") == 5
        found = error(wb, r"\sum_{i=1}^{3} x_i")
        assert found.message == (
            "x has 8 elements, but \\sum runs over 3: a sum over a part of a list is not "
            "supported yet"
        )
        found = error(wb, r"x_3 + 1")
        assert found.message == (
            "x_3 names nothing, and x is a list, whose elements are not written with a "
            "subscript yet"
        )
        assert found.hint == (
            r"give the list an index over a space: \mathcal{N} := \mathbb{R}^{8}, i \in "
            r"\mathcal{N} and x^i := [...]; then x^i 1_i is the sum of its elements"
        )
        space = "\\mathcal{N} := \\mathbb{R}^{8}\ni \\in \\mathcal{N}\n"
        assert wb.expr(space + "x^i := [2, 4, 4, 4, 5, 5, 7, 9]\nx^i 1_i") == 40

    def test_a_recurrence_in_a_label_that_means_something(self, wb):
        found = error(wb, "n := 3\np_0 := 1\np_n := p_{n-1} - 1")
        assert found.message == (
            "p_n is written with its own members, as a recurrence in n is, but n already "
            "means something here"
        )
        assert (
            found.hint
            == "write the recurrence with a label that means nothing yet, as in p_k := ..."
        )
        assert wb.expr("p_k := p_{k-1} - 1\np_{5}") == -4  # the hint works

    def test_an_order_on_a_function_of_several_variables(self, wb):
        # of a function of one variable it is the derivative (tests/reading/test_functions.py)
        wb.exe("g(x, y) := x^2 y^2")
        found = error(wb, r"g^{(2)}(0)")
        assert found.message == (
            "g^{(2)} is read as a name of its own, not as a derivative of the function g"
        )
        assert wb.expr("E^{(2)} := 3\nE^{(2)} + 1") == 4  # an order on a value is a label

    def test_an_index_on_a_numbered_tensor_is_read(self, wb):
        # a label, then an index, in one subscript: k_{1\mu} is k_1 with its index \mu
        wb.exe("ds^2 = -dt^2 + dx^2 + dy^2 + dz^2\nk_1^\\mu := (E_1, E_1, 0, 0)")
        assert wb.expr(r"k_1^\mu k_{1\mu}") == 0  # a null vector
        assert wb.expr(r"k_1^\mu k_1{}_{\mu}") == 0


class TestOperatorsReadWithAnotherMeaning:
    """Guard 4: an operator the lexer reads as a product, a measure or a derivative that
    would be computed with another meaning is refused until it is built: d\\Omega and a
    Jacobian. The Kronecker product, the exponential of a matrix, the bracket of two
    matrices, their commutator, \\times of vectors written with unit vectors and the
    derivative of a list are built, and read so."""

    SIGMA_X = r"\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}"
    SIGMA_Y = r"\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}"

    def test_a_kronecker_product(self, wb):
        product = wb.expr(rf"{self.SIGMA_X} \otimes {self.SIGMA_Y}")
        i = sympy.I
        assert product.tolist() == [[0, 0, 0, -i], [0, 0, i, 0], [0, -i, 0, 0], [i, 0, 0, 0]]
        wb.exe("Coordinates := [t, x]\nA_{ab} := [[1, 2], [3, 4]]")
        assert wb.expr(r"A_{ab} \otimes A_{cd}").components.shape == (2, 2, 2, 2)  # with indices

    def test_a_cross_product_of_unit_vectors(self, wb):
        wb.exe(r"\mathbf{E} := E_0 \hat{\mathbf{x}}" "\n" r"\mathbf{B} := B_0 \hat{\mathbf{y}}")
        E0, B0 = sympy.symbols("E_0 B_0")
        assert list(wb.expr(r"\mathbf{E} \times \mathbf{B}")) == [0, 0, E0 * B0]
        assert wb.expr(r"2 \times 10^{-5}") == sympy.Rational(1, 50000)  # numbers multiply

    def test_a_cross_product_of_hats_on_no_coordinate(self, wb):
        # the review's program: the note said unit vectors and the cross product were not
        # supported, yet \hat{x} \times \hat{y} is (0, 0, 1)
        found = error(wb, r"\hat{n}\times\hat{m}")
        assert found.notes == [
            r"a hat is a unit vector where it sits on a coordinate, as \hat{x} and \hat{r} do, "
            r"in a statement whose plain hats all do, and \times between vectors is their "
            "cross product"
        ]
        assert found.hint == (
            r"define \hat{m} as a vector, as in \hat{m} := (0, 0, 1), or write each component "
            "of the cross product out"
        )
        assert list(wb.expr(r"\hat{x}\times\hat{y}")) == [0, 0, 1]

    def test_a_tuple_compared_with_a_number(self, wb):
        # the review's program: x < 3 of x := (1, 2) was an internal error, and (1, 2) < 3 True
        wb.exe("x := (1, 2)")
        found = error(wb, "x < 3")
        assert found.message == "x < 3 is an inequality, which orders numbers, and x is a tuple"
        assert found.hint == "compare each of its items, or its length, |x|"
        assert error(wb, "(1, 2) < 3").message.endswith("and (1, 2) is a tuple")
        assert wb.expr("|x| < 3") is sympy.true  # the hint works
        (step,) = list(wb.each("a, b > 0"))  # names, distributed: an assumption, as before
        assert step.error is None

    def test_a_matrix_compared_with_a_number(self, wb):
        # an internal error of SymPy's before, as the tuple's was
        wb.exe(r"A := \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}")
        assert error(wb, r"A \le 3").message == (
            r"A \le 3 is an inequality, which orders numbers, and A is a matrix"
        )
        assert wb.expr("A = A").holds()  # an equation of matrices, as before

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
        r, theta = sympy.symbols("r theta")
        each = [-r * sympy.sin(theta), r * sympy.cos(theta), 0]
        assert list(wb.expr(r"\partial_\theta \mathbf{x}")) == each  # component by component
        space = "\\mathcal{E} := \\mathbb{R}^{3}\ni \\in \\mathcal{E}\n"
        derivative = wb.expr(
            space + r"x^i := [r \cos\theta, r \sin\theta, z]"
            "\n"
            r"\frac{\partial}{\partial \theta} x^i"
        )
        assert derivative.components == sympy.Array(each)  # with an index, as before

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
        t = sympy.Symbol("t")
        rotation = [[sympy.cos(t), sympy.sin(t)], [-sympy.sin(t), sympy.cos(t)]]
        assert wb.expr(r"e^{A t}").tolist() == rotation
        assert wb.expr(r"\exp(2 A)").tolist() == [[c.subs(t, 2) for c in row] for row in rotation]

    def test_the_bracket_of_two_matrices(self, wb):
        wb.exe(f"K_1 := {self.SIGMA_X}\nK_2 := {self.SIGMA_Y}")
        commutator = sympy.Matrix([[2 * sympy.I, 0], [0, -2 * sympy.I]])
        assert sympy.Matrix(wb.expr("[K_1, K_2]")) == commutator  # their commutator
        assert sympy.Matrix(wb.expr("K_1 K_2 - K_2 K_1")) == commutator
        assert sympy.Matrix(wb.expr(r"\{K_1, K_2\}")) == sympy.zeros(2, 2)


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

    def test_the_derivative_written_as_an_operator(self, wb):
        # the review's program: d/dx y, d/dx y^2 and d/dx(x y) gave 0, 0 and y
        wb.exe("x^3 + y^3 = 6 x y")
        for written in (r"\frac{d}{dx} y", r"\frac{d}{dx} y^2", r"\frac{d (y^2)}{dx}"):
            assert error(wb, written).message == (
                f"{written} is not 0: y is tied to x by the equation on line 1, x^3 + y^3 = 6 x y"
            )
        found = error(wb, r"\frac{d}{dx}(x y)")
        assert found.message == (
            r"\frac{d}{dx}(x y) takes y as a constant, but y is tied to x by the equation on "
            "line 1, x^3 + y^3 = 6 x y"
        )
        assert found.notes == [
            "read as written, y would be a constant; differentiating a name an equation "
            "defines implicitly is not supported yet"
        ]
        assert found.hint == (
            r"to hold y constant, write \frac{\partial}{\partial x}; for a function, write "
            r"y(x) for y in the equation, and solve its derivative, as in "
            r"solve(\frac{d}{dx}(...) = 0, y'(x))"
        )
        assert wb.expr(r"\frac{d}{dx} x^2") == 2 * x
        assert wb.expr(r"\frac{d}{dx} z") == 0  # no equation ties z to x

    def test_a_run_of_letters_differentiated_is_its_letters(self, wb):
        # \frac{d(xy)}{dx} was y, read as one name xy that no equation ties to x, with a
        # note that xy does not depend on x; in LaTeX it is x y
        (step,) = list(wb.each(r"\frac{d(GM)}{dM}"))
        assert step.values == [sympy.Symbol("G")]
        assert step.notes == []
        wb.exe("x^3 + y^3 = 6 x y")
        assert error(wb, r"\frac{d(xy)}{dx}").message.startswith(
            r"\frac{d(xy)}{dx} takes y as a constant, but y is tied to x"
        )

    def test_a_partial_derivative_holds_the_other_names_constant(self, wb):
        # as it says: after a constraint, \partial_x (x y) is y, where d/dx (x y) is refused
        wb.exe("x^2 + y^2 = 1")
        y = sympy.Symbol("y")
        assert wb.expr(r"\frac{\partial}{\partial x} (x y)") == y
        assert wb.expr(r"\pdv{x^2 y}{y}") == x**2
        assert error(wb, r"\frac{\partial}{\partial x} y").message.startswith(
            r"\frac{\partial}{\partial x} y is not 0: y is tied to x"
        )  # but what is written with no x at all would be 0, as \frac{\partial y}{\partial x}

    def test_an_operator_written_with_a_derivative(self, wb):
        wb.exe("x^2 + y^2 = 1\n" r"\hat{p} := -i\hbar \frac{d}{dx}")
        found = error(wb, r"\hat{p} (x y)")
        assert found.message == (
            r"\hat{p} (x y) takes y as a constant, but y is tied to x by the equation on "
            "line 1, x^2 + y^2 = 1"
        )
        assert error(wb, r"\left(\frac{d}{dx}\right) (x y)").message.startswith(
            r"\left(\frac{d}{dx}\right) (x y) takes y as a constant"
        )
        hbar = sympy.Symbol("hbar")
        assert wb.expr(r"\hat{p} x^2") == -2 * sympy.I * hbar * x

    def test_an_equation_that_states_a_rate_ties_nothing_to_its_variable(self, wb):
        # Carroll's (8.43) and (8.44): d/dt (\rho a^3) = ... gives no a for each t, so the
        # claim after it is read as it was, and holds as written
        wb.exe(r"\frac{d}{dt}(\rho a^3) = a^3\left(\dot\rho + 3\rho\frac{\dot a}{a}\right)")
        assert wb.expr(r"\frac{d}{dt}(\rho a^3) \leq 0") is sympy.true

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

    def test_a_coefficient_of_a_series_is_read_as_one(self, wb):
        catalan = r"\frac{1 - \sqrt{1 - 4x}}{2x}"
        assert wb.expr(rf"[x^{{10}}] {catalan}") == 16796  # design/notation.md, 37
        x = sympy.Symbol("x")
        assert wb.expr(f"series({catalan}, x, 0, 11)").coeff(x, 10) == 16796
        assert wb.expr("[x + 1] y") == (x + 1) * sympy.Symbol("y")  # brackets that group
