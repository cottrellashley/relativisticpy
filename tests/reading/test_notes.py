r"""What is read silently is said: the notes the reader gives where it takes the less likely
reading."""

import pytest
import sympy

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

    def test_a_superscript_i_on_a_name_that_means_nothing(self, wb):
        # Carroll's E^i, a component of the electric field, was E to the power i, the
        # imaginary unit, and nothing said so
        assert notes(wb, "E^i") == [
            "E is not defined, so E^i is read as E to the power i, the imaginary unit; to use "
            "a vector, define it first: E^{i} := [...]"
        ]
        wb.exe("i := 2")
        assert notes(wb, "E^i") == []  # i has a value

    def test_a_solidus_of_differentials_the_program_defines_is_their_quotient(self, wb):
        # dy/dx is the derivative, unless the program gives dy and dx values of their own
        wb.exe("dx := 1/10\ndy := 1/5")
        assert notes(wb, "dy/dx") == [
            "dy and dx are defined, so dy/dx is a quotient, not a derivative"
        ]
        assert wb.expr("dy/dx") == 2

    def test_a_name_whose_letters_are_written_alone_is_one_name_and_says_so(self, wb):
        # (a+b)^2 - (a^2 + 2ab + b^2) was 2ab - 2 ab, without a word (decision 41)
        assert notes(wb, "(a+b)^2 - (a^2 + 2ab + b^2)") == [
            "ab is one name, as plain text writes names; for the product, write a b or a*b"
        ]
        assert wb.expr("(a+b)^2 - (a^2 + 2a b + b^2)") == 0  # the hint works
        assert notes(wb, "2pq") == []  # p and q are written nowhere alone: one name
        wb.exe("x := 2\ny := 3")
        assert notes(wb, "xy + 1") == [
            "xy is one name, as plain text writes names; for the product, write x y or x*y"
        ]

    def test_a_run_of_a_functions_parameters_in_its_body_is_their_product(self, wb):
        # g(x, y) := x e^(xy) was x times e to a name xy, so that ∂g/∂x was e^(xy) and
        # ∂²g/∂y∂x was 0 (decision 41)
        x, y = sympy.symbols("x y")
        wb.exe("g(x, y) := x e^(xy)\nu(x, y) := sin(xy)")
        assert notes(wb, "∂g/∂x") == [
            "xy is x*y, a product of the parameters of g(x, y), since nothing defines xy"
        ]
        assert wb.expr("∂g/∂x") == sympy.diff(x * sympy.exp(x * y), x)
        assert wb.expr("∂²g/∂y∂x") == sympy.diff(x * sympy.exp(x * y), x, y)
        assert wb.expr("∂u/∂y") == x * sympy.cos(x * y)
        assert wb.expr("g(2, 3)") == 2 * sympy.exp(6)
        wb.exe("h(a, b) := (a + b)^2 - (a^2 + 2ab + b^2)")
        assert wb.expr("h(1, 2)") == 0
        wb.exe("xy := 5\nk(x, y) := xy")  # a name the program defines is that name
        assert notes(wb, "k(1, 2)") == []
        assert wb.expr("k(1, 2)") == 5

    def test_a_name_run_on_after_a_numbered_subscript_is_one_name_and_says_so(self, wb):
        # in plain text x_1x_2 is one name, as a program writes one; where x_1 and x_2 have
        # values it says how to write their product (in LaTeX it is the product)
        assert notes(wb, "x_1x_2") == []
        wb.exe("x_1 := 3\nx_2 := 5")
        assert notes(wb, "x_1x_2") == [
            "x_1x_2 is one name, as plain text writes names; for the product, write x_1 x_2 "
            "or x_1*x_2"
        ]
        assert wb.expr("x_1 x_2") == 15  # the hint works

    def test_a_bar_over_a_quantity_the_program_defines_is_its_conjugate_and_says_so(self, wb):
        # \bar{z} of z := 3 + 4i was a new name, zbar, without a word (decision 8.1)
        wb.exe("z := 3 + 4i")
        assert notes(wb, r"\bar{z}") == [
            r"\bar{z} is read as the complex conjugate of z, since z is defined on line 1; for "
            r"a name of its own, define \bar{z} first"
        ]
        assert notes(wb, r"\overline{1 + i}") == [
            r"\overline{1 + i} is read as the complex conjugate of 1 + i, since a bar over an "
            "expression names nothing"
        ]
        assert notes(wb, r"\bar{q}") == []  # a name of its own, as q means nothing
        wb.exe("x := [1, 2, 3, 6]")
        assert notes(wb, r"\bar{x}") == [
            r"x is a list, so \bar{x} is read as a name of its own, not its mean; for the mean, "
            r"define it, as \bar{x} := \frac{1}{4} \sum_{i=1}^{4} x_i"
        ]
        wb.exe(r"\bar{x} := \frac{1}{4} \sum_{i=1}^{4} x_i")
        assert wb.expr(r"\bar{x}") == 3  # the hint works

    def test_a_numbered_partial_derivative_says_which_variable_it_is_by(self, wb):
        # Stewart's f_1 and D_1 f were a name, and D_1 times f (decisions 5.6 and 18)
        wb.exe("f(x, y) := x^2 y^3")
        assert notes(wb, "f_{12}") == [
            "f_{12} is read as the partial derivative of f(x, y) by x, then y, the variables "
            "its subscript numbers, as Stewart writes it; for a name of its own, define f_{12} "
            "first"
        ]
        assert notes(wb, r"D_1 f + 0 \cdot x") == [
            "D_1 f is read as the partial derivative of f(x, y) by x, the variable the "
            r"subscript of D numbers, as Stewart writes it; for D_1 times f, write D_1 \cdot f"
        ]

    def test_a_t_raised_on_a_matrix_is_its_transpose_and_says_so(self, wb):
        # A^t was the t-th power, silently, where Artin and Lang mean the transpose
        wb.exe(r"A := \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}")
        assert notes(wb, "A^t") == [
            r"A^t is read as the transpose of A, as Artin and Lang write it; for a power of A, "
            r"say what t is first, as t \in \mathbb{N}"
        ]
        assert notes(wb, r"\tilde{A}") == [
            r"\tilde{A} is read as the transpose of A, as Goldstein and Griffiths write it; for "
            r"a name of its own, define \tilde{A} first"
        ]

    def test_a_name_written_as_an_operator_applies_to_what_follows_and_says_so(self, wb):
        # \\operatorname{lb} 8 was 8 lb, a symbol times 8, without a word (decision 26)
        assert notes(wb, r"\operatorname{lb} 8 + 1") == [
            r"\operatorname{lb} is not a function RelativisticPy knows, so \operatorname{lb} 8 "
            "is an unknown function",
            "lb is a function nothing defines, so it is left as written where it is applied "
            "to a number; define it first, as in lb(x) := ...",
        ]

    def test_a_capital_i_beside_a_number_says_it_is_no_imaginary_unit(self, wb):
        # (3 + 4*I)^2 was (4I + 3)^2, silently, where SymPy's users mean I for i
        note = (
            "I is a name of its own, as nothing defines it, and not the imaginary unit, which "
            "is i, as in 3 + 4*i"
        )
        assert notes(wb, "(3 + 4*I)^2") == [note]
        assert notes(wb, "(x + y*I)^2") == [note]
        assert notes(wb, "(sqrt(3)/2)*I") == [note]
        assert sympy.expand(wb.expr("(3 + 4*i)^2")) == -7 + 24 * sympy.I  # the hint works
        assert notes(wb, "I*R") == []  # a current times a resistance, as physics writes it
        wb.exe("I := 2")
        assert notes(wb, "3 + 4*I") == []  # I has a value

    def test_a_defined_letter_run_into_digits_is_one_name_and_says_so(self, wb):
        # (3 + j4)^2 after j := \sqrt{-1} was (j4 + 3)^2, silently, where engineers mean 4j
        wb.exe(r"j := \sqrt{-1}")
        assert notes(wb, r"(3 + j4)^2 + \alpha") == [
            "j4 is one name, as nothing defines it, though j has a value; for j times 4, write 4j"
        ]
        assert sympy.expand(wb.expr(r"(3 + 4j)^2 + 0 \alpha")) == -7 + 24 * sympy.I
        assert notes(wb, "x2 + 1") == []  # x has no value

    def test_a_function_run_into_its_argument_is_one_name_and_says_so(self, wb):
        # sinx and lnx were names, without a word that sin x and ln x were meant
        assert notes(wb, "sinx") == ["sinx is one name; for sin of x, write sin x or sin(x)"]
        assert notes(wb, "1 + lnx") == ["lnx is one name; for ln of x, write ln x or ln(x)"]
        assert notes(wb, "cos2theta") == [
            "cos2theta is one name; for cos of 2theta, write cos 2theta or cos(2theta)"
        ]
        assert notes(wb, "logic + cost2") == []  # words, as names are

    def test_a_function_without_brackets_over_names_says_how_far_it_reaches(self, wb):
        # sin x y was sin(x) y in plain text; it is sin(x y), as \sin x y is (decision 11)
        assert notes(wb, "sin x y") == [
            "sin x y is read as sin(x y), as LaTeX reads it; for sin(x) times y, write sin(x) y"
        ]
        assert notes(wb, "sin 2x + sin(x y) + sin x") == []

    def test_a_letter_followed_by_brackets_is_a_function_and_says_so(self, wb):
        assert notes(wb, r"\gamma(t - v x)") == [
            r"\gamma is not defined, so \gamma(t - v x) is an unknown function applied to "
            r"t - v x; for a product, write \gamma \cdot (t - v x)"
        ]
        assert str(wb.expr(r"\gamma \cdot (t - v x)")) == "gamma*(t - v*x)"  # the hint works
        assert notes(wb, r"\alpha(t, r)") == []  # a function of names, as meant
        wb.exe("f(x) := x^2")
        assert notes(wb, "f(x + 1)") == []  # a function defined

    def test_a_function_nothing_defines_at_a_number_is_said_to_be_left_as_written(self, wb):
        # Black-Scholes, as Hull writes it: N(d_1) at the number d_1 is no value, and a
        # result that holds it said nothing (design/notation.md, 26)
        wb.exe(r"d_1 := \frac{1}{2}")
        assert notes(wb, r"42 \cdot N(d_1)") == [
            "N is a function nothing defines, so it is left as written where it is applied "
            "to a number; define it first, as in N(x) := ..."
        ]
        assert notes(wb, r"42 \cdot N(x)") == []  # a function of x, as meant

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
