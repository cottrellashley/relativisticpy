r"""Quantum notation, read by its operands (design/notation.md, decisions 13 and 20): kets,
bras and their brackets; angle brackets without a bar; commutators of matrices and of
operators; operators as values, applied and composed; and the rules an ``=`` states for
operators and kets nothing defines. The Pauli matrices are written X, Y and Z, as their
labelled names, \sigma_x, are read elsewhere."""

from __future__ import annotations

import pytest
import sympy

from relativisticpy import Workbook, holds
from relativisticpy.errors import CompileError, EvaluationError

IMAGINARY = sympy.I
x, k, a, hbar, m, omega, n = sympy.symbols("x k a hbar m omega n")

X = r"X := \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}"
Y = r"Y := \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}"
Z = r"Z := \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}"
KETS = (
    r"|0\rangle := \begin{pmatrix} 1 \\ 0 \end{pmatrix}" "\n"
    r"|1\rangle := \begin{pmatrix} 0 \\ 1 \end{pmatrix}"
)  # fmt: skip
MOMENTUM = r"\hat{p} := -i\hbar \frac{d}{dx}"
ANGULAR = "\n".join(
    rf"L_{a} := -i\hbar \left( {b} \frac{{\partial}}{{\partial {c}}} - {c} \frac{{\partial}}"
    rf"{{\partial {b}}} \right)"
    for a, b, c in (("x", "y", "z"), ("y", "z", "x"), ("z", "x", "y"))
)


def matrix(value: object) -> sympy.Matrix:
    return sympy.Matrix(value.tolist())  # type: ignore[attr-defined]


def readings(wb: Workbook, source: str) -> list[str]:
    return [
        decision.text()
        for reading in wb.explain(source)
        for decision in reading.decisions
        if decision.kind == "bracket or operator"
    ]


def notes(wb: Workbook, source: str) -> list[str]:
    return [line for step in wb.each(source) for note in step.notes for line in note.split("\n")]


def error(wb: Workbook, source: str) -> CompileError:
    with pytest.raises(CompileError) as raised:
        wb.exe(source)
    return raised.value


def same(found: object, expected: object) -> bool:
    """Whether two values are equal, their symbols compared by name, whatever is assumed of
    them."""
    value = sympy.sympify(found)
    plain = value.xreplace({s: sympy.Symbol(s.name) for s in value.free_symbols})
    return sympy.simplify(plain - sympy.sympify(expected)) == 0


class TestKetsAsColumns:
    def test_a_bracket_of_columns_is_their_inner_product(self, wb):
        wb.exe(KETS)
        assert wb.expr(r"\langle 0|0\rangle") == 1
        assert wb.expr(r"\langle 0|1\rangle") == 0

    def test_a_bra_is_the_conjugate_transpose(self, wb):
        wb.exe(r"|\psi\rangle := \begin{pmatrix} 1 \\ i \end{pmatrix}")
        assert matrix(wb.expr(r"\langle\psi|")) == sympy.Matrix([[1, -IMAGINARY]])
        assert wb.expr(r"\langle\psi|\psi\rangle") == 2

    def test_a_matrix_element_is_a_row_a_matrix_and_a_column(self, wb):
        wb.exe(f"{KETS}\n{Z}\n{X}")
        assert wb.expr(r"\langle 0 | Z | 0 \rangle") == 1
        assert wb.expr(r"\langle 1 | X | 0 \rangle") == 1
        assert readings(wb, r"\langle 0 | Z | 0 \rangle") == [
            r"bracket or operator: \langle 0 | Z | 0 \rangle is a row times a matrix times a "
            "column, since its states are columns"
        ]

    def test_an_outer_product_is_a_matrix(self, wb):
        wb.exe(KETS)
        assert matrix(wb.expr(r"|0\rangle\langle 1|")) == sympy.Matrix([[0, 1], [0, 0]])

    def test_completeness(self, wb):
        wb.exe(
            r"|+\rangle := \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 1 \end{pmatrix}" "\n"
            r"|-\rangle := \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ -1 \end{pmatrix}"
        )  # fmt: skip
        found = wb.expr(r"|+\rangle\langle+| + |-\rangle\langle-|")
        assert matrix(found) == sympy.eye(2)

    def test_a_ket_of_kets_the_program_defines_is_their_tensor_product(self, wb):
        wb.exe(KETS)
        assert matrix(wb.expr(r"|01\rangle")) == sympy.Matrix([0, 1, 0, 0])
        assert notes(wb, r"|00\rangle") == [
            r"|00\rangle is read as |0\rangle \otimes |0\rangle, of the kets the program defines"
        ]

    def test_a_bell_state(self, wb):
        wb.exe(KETS + "\n" r"|\Phi^+\rangle := \frac{|00\rangle + |11\rangle}{\sqrt{2}}")
        rho = matrix(wb.expr(r"|\Phi^+\rangle\langle\Phi^+|"))
        bell = sympy.Matrix([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]])
        assert rho == bell / 2

    def test_a_ket_is_a_column(self, wb):
        found = error(wb, r"|0\rangle := \begin{pmatrix} 1 & 0 \end{pmatrix}")
        assert found.message.startswith(r"|0\rangle is a ket, so its value is a column")


class TestWavefunctions:
    GROUND = r"\psi(x) := \sqrt{\frac{2}{a}} \sin\left(\frac{\pi x}{a}\right)"

    def test_a_bracket_of_wavefunctions_is_an_integral_over_the_stated_domain(self, wb):
        wb.exe(r"0 \leq x \leq a" "\n" + self.GROUND)
        assert same(wb.expr(r"\langle\psi|\psi\rangle"), 1)
        assert notes(wb, r"\langle\psi|\psi\rangle")[0] == (
            r"\langle\psi|\psi\rangle is read as their overlap, an integral over x from (0, a)"
        )

    def test_without_a_domain_the_integral_is_over_the_line(self, wb):
        wb.exe(r"\phi(x) := e^{-x^2/2}")
        assert same(wb.expr(r"\langle\phi|\phi\rangle"), sympy.sqrt(sympy.pi))

    def test_a_bracket_with_a_position_is_the_wavefunction_there(self, wb):
        wb.exe(r"\psi(x) := e^{i k x}")
        assert wb.expr(r"\langle y|\psi\rangle") == sympy.exp(IMAGINARY * k * sympy.Symbol("y"))

    def test_a_matrix_element_of_wavefunctions_applies_the_operator(self, wb):
        wb.exe(r"0 \leq x \leq a" "\n" + self.GROUND + "\n" + MOMENTUM)
        expected = (sympy.pi * hbar / a) ** 2
        assert same(wb.expr(r"\langle\psi|\hat{p}^2|\psi\rangle"), expected)

    def test_a_bracket_the_program_defines_and_its_conjugate(self, wb):
        wb.exe(r"\langle x | p \rangle := \frac{1}{\sqrt{2\pi\hbar}} e^{ipx/\hbar}")
        p = sympy.Symbol("p")
        assert same(
            wb.expr(r"\langle x | p \rangle"),
            sympy.exp(IMAGINARY * p * x / hbar) / sympy.sqrt(2 * sympy.pi * hbar),
        )
        conjugate = wb.expr(r"\langle p | x \rangle")
        assert same(
            conjugate, sympy.exp(-IMAGINARY * p * x / hbar) / sympy.sqrt(2 * sympy.pi * hbar)
        )
        assert notes(wb, r"\langle p | x \rangle")[0] == (
            r"\langle p | x \rangle is read as the conjugate of \langle x|p\rangle, defined on "
            "line 1, with x and p real"
        )

    def test_the_momentum_wavefunction(self, wb):
        wb.exe(
            r"\langle x | p \rangle := \frac{1}{\sqrt{2\pi\hbar}} e^{ipx/\hbar}" "\n"
            r"\langle x | \alpha \rangle := \frac{1}{\pi^{1/4} \sqrt{d}} "
            r"\exp\left(ikx - \frac{x^2}{2d^2}\right)"
        )  # fmt: skip
        found = wb.expr(
            r"\int_{-\infty}^{\infty} \langle p | x \rangle \langle x | \alpha \rangle \, dx"
        )
        p, d = sympy.symbols("p d")
        stated = sympy.sqrt(d / (hbar * sympy.sqrt(sympy.pi))) * sympy.exp(
            -((p - hbar * k) ** 2) * d**2 / (2 * hbar**2)
        )
        values = {p: 0.7, d: 1.3, hbar: 0.9, k: 0.4}
        assert abs(complex(found.subs(values)) - complex(stated.subs(values))) < 1e-12

    def test_a_star_before_the_arguments_conjugates_the_call(self, wb):
        # \psi^*(x) is \psi(x)^*, not \psi^* times x, which made <p^2> of a Gaussian 0
        wb.exe(r"\psi(x) := e^{i k x}")
        conjugate = sympy.conjugate(sympy.exp(IMAGINARY * k * x))
        assert wb.expr(r"\psi^*(x)") == wb.expr(r"\psi(x)^*") == conjugate


class TestAngleBrackets:
    def test_what_a_program_defines_wins(self, wb):
        wb.exe(r"\langle x^2 \rangle := 3" "\n" r"\langle x \rangle := 1")
        assert wb.expr(r"\langle x^2 \rangle - \langle x \rangle^2") == 2

    def test_a_bracket_of_names_is_a_function_of_them(self, wb):
        wb.exe(r"\langle u, v \rangle := u v + 1")
        assert wb.expr(r"\langle 2, 3 \rangle") == 7

    def test_the_inner_product_of_two_vectors(self, wb):
        assert wb.expr(r"\langle (1, 2), (3, 4) \rangle") == 11
        assert wb.expr(r"\langle [1, i], [1, i] \rangle") == 2

    def test_the_expectation_value_in_the_one_state(self, wb):
        wb.exe(r"0 \leq x \leq a" "\n" + TestWavefunctions.GROUND)
        assert same(wb.expr(r"\langle x \rangle"), a / 2)
        assert notes(wb, r"\langle x \rangle")[0].startswith(
            r"\langle x \rangle is read as the expectation value of x in \psi"
        )

    def test_the_expectation_value_of_an_operator(self, wb):
        wb.exe(r"0 \leq x \leq a" "\n" + TestWavefunctions.GROUND + "\n" + MOMENTUM)
        assert same(wb.expr(r"\langle \hat{p}^2 \rangle"), (sympy.pi * hbar / a) ** 2)
        assert same(wb.expr(r"\langle \hat{p} \rangle"), 0)

    def test_the_expectation_value_in_a_column(self, wb):
        wb.exe(Z + "\n" r"|\chi\rangle := \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 1 \end{pmatrix}")
        assert wb.expr(r"\langle Z \rangle") == 0

    def test_the_state_a_subscript_names(self, wb):
        wb.exe(r"\chi(x) := e^{-x^2}" "\n" r"\psi(x) := e^{-2x^2}")
        found = error(wb, r"\langle x^2 \rangle")
        assert (
            found.message
            == r"\langle x^2 \rangle is an expectation value, but in which state: \chi, \psi?"
        )
        assert same(wb.expr(r"\langle x^2 \rangle_\chi"), sympy.Rational(1, 4))

    def test_a_state_need_not_be_normalised(self, wb):
        # the review's programs: <x^2> of e^{-x^2} is 1/4, not its integral, sqrt(2 pi)/8
        wb.exe(r"\psi(x) := e^{-x^2}")
        assert same(wb.expr(r"\langle x^2 \rangle"), sympy.Rational(1, 4))
        assert notes(wb, r"\langle x^2 \rangle") == [
            r"\langle x^2 \rangle is read as the expectation value of x^2 in \psi, an integral "
            r"over x from (-\infty, \infty), divided by \langle\psi|\psi\rangle, as a state need "
            "not be normalised"
        ]

    def test_a_column_need_not_be_normalised(self, wb):
        wb.exe(
            r"|\chi\rangle := \begin{pmatrix} 1 \\ 1 \end{pmatrix}" "\n"
            r"\sigma_x := \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}"
        )  # fmt: skip
        assert wb.expr(r"\langle \sigma_x \rangle_\chi") == 1  # not <chi|sigma_x|chi>, 2

    def test_a_state_that_cannot_be_normalised_has_no_expectation_values(self, wb):
        wb.exe(r"\psi(x) := e^{i k x}")
        with pytest.raises(EvaluationError) as raised:
            wb.expr(r"\langle x \rangle")
        assert raised.value.message.startswith("the state cannot be normalised")

    def test_a_hat_on_the_variable_of_the_state_is_the_position(self, wb):
        wb.exe(r"\psi(x) := \frac{1}{\pi^{1/4}} e^{-(x-2)^2/2}")
        assert wb.expr(r"\langle \hat{x} \rangle") == 2  # not the unit vector [1, 0, 0]
        assert wb.expr(r"\langle \hat{x}^2 \rangle") == sympy.Rational(9, 2)  # not xhat^2
        assert wb.expr(r"\langle\psi|\hat{x}|\psi\rangle") == 2
        assert notes(wb, r"\langle \hat{x} \rangle")[0] == (
            r"\hat{x} is read as the position, x, which acts by multiplying: nothing defines "
            "it, and it is in angle brackets of a wavefunction of x"
        )

    def test_phi_is_a_state_only_where_it_is_written_as_one(self, wb):
        # \Phi and \phi name potentials and phases as often as states
        wb.exe(r"\Phi(x) := \frac{1}{2} k x^2")
        found = error(wb, r"\langle x^2 \rangle")
        assert found.message == (
            r"\langle x^2 \rangle is an expectation value, but the program defines no state"
        )
        assert found.notes[-1] == (
            r"\Phi names a potential or a phase as often as a state, so it is a state only "
            "where a ket or a subscript names it"
        )
        assert found.hint.startswith(r"name the state, as in \langle x^2 \rangle_\Phi, or")
        wb.exe(r"\phi(x) := e^{-x^2}")
        assert same(wb.expr(r"\langle x^2 \rangle_\phi"), sympy.Rational(1, 4))
        assert same(wb.expr(r"\langle\phi|\phi\rangle"), sympy.sqrt(sympy.pi / 2))

    def test_without_a_state_an_expectation_value_is_refused(self, wb):
        found = error(wb, r"\langle x \rangle")
        assert (
            found.message
            == r"\langle x \rangle is an expectation value, but the program defines no state"
        )
        assert found.hint == (
            r"name the state, as in \langle x \rangle_\psi, or define \langle x \rangle itself, "
            r"as in \langle x \rangle := \int x |\psi(x)|^2 \, dx"
        )

    def test_brackets_of_what_is_no_vector_are_refused(self, wb):
        found = error(wb, r"\sqrt{\langle p, p \rangle}")
        assert found.message == (
            r"\langle p, p \rangle is angle brackets around p, p, which are not two vectors, "
            "and no bracket the program defines"
        )

    def test_a_presentation_of_a_group_is_refused(self, wb):
        found = error(wb, r"\langle a, b \mid a b a^{-1} b \rangle")
        assert found.message.endswith(
            "is read as the presentation of a group, by its generators and relations, which is "
            "not supported"
        )

    def test_a_time_ordered_product_is_refused(self, wb):
        found = error(wb, r"\langle 0 | T\{\phi(x_1)\phi(x_2)\} | 0 \rangle = 2")
        assert found.message == (
            r"T\{\phi(x_1)\phi(x_2)\}, between a bra and a ket, is read as a time-ordered "
            "product of fields, which is not supported yet"
        )


class TestCommutators:
    def test_of_matrices(self, wb):
        wb.exe(f"{X}\n{Y}\n{Z}")
        assert matrix(wb.expr("[X, Y]")) == 2 * IMAGINARY * matrix(wb.expr("Z"))
        assert wb.expr("[X, Y] = 2i Z").holds()
        assert notes(wb, "[X, Y]") == [
            "[X, Y] is read as the commutator of the matrices X and Y, X Y - Y X"
        ]

    def test_the_anticommutator_of_matrices(self, wb):
        wb.exe(f"{X}\n{Y}")
        assert matrix(wb.expr(r"\{X, Y\}")) == sympy.zeros(2, 2)
        assert matrix(wb.expr(r"\{X, X\}")) == 2 * sympy.eye(2)

    def test_of_matrices_written_out(self, wb):
        found = wb.expr(
            r"\left[ \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & -1 \\ 0 & 1 & 0 \end{pmatrix}, "
            r"\begin{pmatrix} 0 & 0 & 1 \\ 0 & 0 & 0 \\ -1 & 0 & 0 \end{pmatrix} \right]"
        )
        assert matrix(found) == sympy.Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 0]])

    def test_a_list_and_a_set_stay_as_they_are(self, wb):
        assert wb.expr("[t, r]") == [sympy.Symbol("t"), sympy.Symbol("r")]
        assert notes(wb, "[t, r]") == [
            "[t, r] is read as a list, as t and r are neither operators nor matrices"
        ]
        assert notes(wb, "solve(x + y = 1, [x, y])") == []  # an argument is no commutator
        assert notes(wb, "[0, t]") == []  # nor is a number
        assert readings(wb, "[0, t]") == []
        assert readings(wb, "[t, r]") == [
            "bracket or operator: [t, r] is a list, since t and r are neither operators nor "
            "matrices"
        ]
        assert readings(wb, r"\{t, r\}") == [
            r"bracket or operator: \{t, r\} is a set, since t and r are neither operators nor "
            "matrices"
        ]

    def test_of_numbers_compared_with_a_scalar_is_still_refused(self, wb):
        wb.exe("a := 2\nb := 3")
        assert error(wb, "[a, b] = 0").hint == "for the commutator, write a b - b a"


class TestOperators:
    def test_an_operator_is_applied_by_juxtaposition(self, wb):
        wb.exe(MOMENTUM)
        assert wb.expr(r"\hat{p} e^{ikx}") == hbar * k * sympy.exp(IMAGINARY * k * x)
        assert readings(wb, MOMENTUM) == [
            r"bracket or operator: \hat{p} := ... is a definition of the operator phat, since "
            r"-i\hbar \frac{d}{dx} acts on what follows it, and has nothing to act on"
        ]

    def test_an_operator_is_composed_by_powers_and_products(self, wb):
        wb.exe(MOMENTUM)
        assert same(wb.expr(r"\hat{p}^2 \sin(kx)"), hbar**2 * k**2 * sympy.sin(k * x))
        assert same(wb.expr(r"\hat{p} \hat{p} \sin(kx)"), hbar**2 * k**2 * sympy.sin(k * x))

    def test_an_operator_acts_on_the_rest_of_the_term(self, wb):
        # as d/dx does: \hat{p} x \psi(x) is \hat{p}(x \psi), not (\hat{p} x) \psi
        wb.exe(MOMENTUM + "\n" r"\psi(x) := e^{-x^2}")
        expected = IMAGINARY * hbar * (2 * x**2 - 1) * sympy.exp(-(x**2))
        for written in (r"\hat{p} \hat{x} \psi(x)", r"\hat{p} x \psi(x)", r"\hat{p}\psi(x) x"):
            assert same(wb.expr(written), expected), written
        assert same(wb.expr(r"x \hat{p} \psi(x)"), 2 * IMAGINARY * hbar * x**2 * sympy.exp(-(x**2)))

    def test_only_the_hat_on_the_variable_of_an_operator_is_the_position(self, wb):
        # \hat{y} stays a unit vector beside \hat{p} := -i\hbar d/dx, and \hat{x} is one
        # among unit vectors
        wb.exe(MOMENTUM + "\n" r"\mathbf{F} := 3\hat{x} + 4\hat{y}")
        assert wb.expr(r"|\mathbf{F}|") == 5
        assert wb.expr(r"|3\hat{x}+4\hat{y}|") == 5
        assert list(wb.expr(r"\hat{y}")) == [0, 1, 0]
        assert same(wb.expr(r"\hat{x}"), x)

    def test_a_function_is_read_as_it_was_where_it_is_defined(self, wb):
        wb.exe(r"\mathbf{v}(t) := t\hat{x} + t^2\hat{y}")
        assert list(wb.expr(r"\mathbf{v}(2)")) == [2, 4, 0]
        wb.exe(MOMENTUM)
        assert list(wb.expr(r"\mathbf{v}(2)")) == [2, 4, 0]  # not 2 x + 4 yhat
        assert list(wb.copy().expr(r"\mathbf{v}(2)")) == [2, 4, 0]  # and so in a copy

    def test_an_operator_with_a_hat_and_a_subscript(self, wb):
        # the review's programs, each refused as "cannot define this"
        wb.exe(
            r"\hat{L}_z := -i\hbar \frac{d}{d\phi}" "\n"
            r"\hat{p}_x := -i\hbar \frac{\partial}{\partial x}"
        )  # fmt: skip
        phi, mz = sympy.symbols("phi m")
        assert same(wb.expr(r"\hat{L}_z e^{i m \phi}"), hbar * mz * sympy.exp(IMAGINARY * mz * phi))
        assert same(wb.expr(r"\hat{p}_x e^{i k x}"), hbar * k * sympy.exp(IMAGINARY * k * x))
        assert wb.expr(r"[\hat{x}, \hat{p}_x] = i\hbar") is sympy.true

    def test_what_cannot_be_defined_says_what_it_was_tried_as(self, wb):
        found = error(wb, r"\hat{L}_z := 5")
        assert found.message == (
            r"cannot define \hat{L}_z as a name, a function, a tensor or an operator"
        )
        assert found.notes == [
            r"as a tensor, it would be named with a letter, and \hat{L} is a letter under an "
            r"accent, which takes a number, a sign or a word as a label, as in \hat{x}_0, but "
            "no index",
            r"as an operator, its value would act on what follows it, as -i\hbar \frac{d}{dx} "
            "does, and 5 does not",
        ]
        assert found.hint == "give it a name without the accent, as in L_z := ..."
        wb.exe(found.hint.removeprefix("give it a name without the accent, as in ")[:-3] + "5")
        assert wb.expr("L_z + 1") == 6  # the hint works

    def test_an_operator_in_brackets_acts_on_what_follows(self, wb):
        found = wb.expr(r"\left(-\hbar^2 \frac{d^2}{dx^2} + x\right) \sin(kx)")
        assert same(found, (hbar**2 * k**2 + x) * sympy.sin(k * x))

    def test_an_operator_alone_has_nothing_to_act_on(self, wb):
        wb.exe(MOMENTUM)
        assert (
            error(wb, r"\hat{p}").message == r"\hat{p} needs something to act on, as in \hat{p} f"
        )

    def test_the_canonical_commutator_on_a_function(self, wb):
        wb.exe(r"\hat{p} := \frac{\hbar}{i} \frac{d}{dx}")
        assert wb.expr(r"[x, \hat{p}] f(x) = i\hbar f(x)") is sympy.true

    def test_a_claim_between_operators_is_decided_on_a_test_function(self, wb):
        wb.exe(MOMENTUM)
        assert wb.expr(r"[x, \hat{p}] = i\hbar") is sympy.true
        assert holds(wb.expr(r"[x, \hat{p}] = -i\hbar")) is False
        assert notes(wb, r"[x, \hat{p}] = i\hbar") == [
            r"[x, \hat{p}] = i\hbar is a claim about operators, decided by applying both sides "
            "to a function f(x)"
        ]

    def test_the_position_beside_the_momentum(self, wb):
        wb.exe(MOMENTUM)
        assert wb.expr(r"[\hat{x}, \hat{p}] = i\hbar") is sympy.true
        assert notes(wb, r"[\hat{x}, \hat{p}] = i\hbar")[0] == (
            r"\hat{x} is read as the position, x, which acts by multiplying: nothing defines "
            "it, and an operator differentiates by x"
        )

    def test_the_algebra_of_angular_momentum(self, wb):
        wb.exe(ANGULAR)
        assert wb.expr(r"[L_x, L_y] = i\hbar L_z") is sympy.true
        assert notes(wb, r"[L_x, L_y] = i\hbar L_z")[0].endswith("to a function f(x, y, z)")

    def test_ladder_operators_both_signs_at_once(self, wb):
        wb.exe(
            MOMENTUM + "\n"
            r"a_\pm := \frac{1}{\sqrt{2\hbar m\omega}} \left(\mp i\hat{p} + m\omega x\right)"
        )
        assert wb.expr("[a_-, a_+] = 1") is sympy.true
        assert holds(wb.expr("[a_+, a_-] = 1")) is False

    def test_a_hamiltonian_is_an_operator(self, wb):
        wb.exe(
            MOMENTUM + "\n"
            r"H := \frac{\hat{p}^2}{2m} + \frac{1}{2} m\omega^2 x^2" "\n"
            r"\psi_0(x) := e^{-\frac{m\omega}{2\hbar} x^2}"
        )  # fmt: skip
        assert wb.expr(r"H \psi_0(x) = \frac{1}{2}\hbar\omega\, \psi_0(x)") is sympy.true


class TestRules:
    def test_a_commutator_of_operators_nothing_defines_is_a_rule(self, wb):
        (step,) = list(wb.each(r"[\hat{x}, \hat{p}] = i\hbar"))
        assert step.notes == [
            r"[\hat{x}, \hat{p}] = i\hbar is read as a rule for what follows, not as a claim: "
            r"its left side is the commutator of operators written with \hat{x} and \hat{p}, "
            "which nothing defines"
        ]
        assert wb.expr(r"[\hat{x}^3, \hat{p}] = 3i\hbar\hat{x}^2") is sympy.true
        assert holds(wb.expr(r"[\hat{x}^2, \hat{p}] = i\hbar\hat{x}")) is False

    def test_an_orthonormal_basis(self, wb):
        wb.exe(
            r"\langle m | n \rangle = \delta_{mn}" "\n"
            r"|\alpha\rangle := i|1\rangle - 2|2\rangle - i|3\rangle" "\n"
            r"|\beta\rangle := i|1\rangle + 2|3\rangle"
        )  # fmt: skip
        assert wb.expr(r"\langle \alpha | \beta \rangle") == 1 + 2 * IMAGINARY
        assert wb.expr(r"\langle \beta | \alpha \rangle") == 1 - 2 * IMAGINARY

    def test_ladder_operators_on_number_states(self, wb):
        wb.exe(
            r"\langle n' | n \rangle = \delta_{n'n}" "\n"
            r"a |n\rangle = \sqrt{n}\, |n - 1\rangle" "\n"
            r"a^\dagger |n\rangle = \sqrt{n + 1}\, |n + 1\rangle" "\n"
            r"x := \sqrt{\frac{\hbar}{2m\omega}} \left(a + a^\dagger\right)"
        )  # fmt: skip
        found = wb.expr(r"\langle n | \frac{1}{2} m\omega^2 x^2 | n \rangle")
        assert same(found, hbar * omega * (n + sympy.Rational(1, 2)) / 2)
        assert wb.expr(r"\langle n | x | n \rangle") == 0
        assert wb.expr(r"\langle n | a^\dagger a | n \rangle") == n

    def test_the_modulus_of_a_bracket_is_summed_over_its_label(self, wb):
        # Landau and Lifshitz, section 38: of a cubic anharmonicity, |1> and |3> contribute
        wb.exe(
            r"\langle n' | n \rangle = \delta_{n'n}" "\n"
            r"a |n\rangle = \sqrt{n}\, |n - 1\rangle" "\n"
            r"a^\dagger |n\rangle = \sqrt{n + 1}\, |n + 1\rangle" "\n"
            r"x := \sqrt{\frac{\hbar}{2m\omega}} \left(a + a^\dagger\right)" "\n"
            r"E_n := \hbar\omega\left(n + \frac{1}{2}\right)"
        )  # fmt: skip
        lam = sympy.Symbol("lambda")
        found = wb.expr(r"|\langle 1 | \lambda x^3 | 0 \rangle|^2")
        assert same(found, sympy.Rational(9, 8) * lam**2 * hbar**3 / (m * omega) ** 3)
        summed = r"\frac{|\langle k | \lambda x^3 | 0 \rangle|^2}{E_0 - E_k}"
        found = wb.expr(r"\sum_{k=1}^{\infty} " + summed)
        assert same(found, -sympy.Rational(11, 8) * lam**2 * hbar**2 / (m**3 * omega**4))

    def test_a_clebsch_gordan_coefficient(self, wb):
        written = (
            r"\langle 1, \tfrac{1}{2}; 1, -\tfrac{1}{2} | 1, \tfrac{1}{2}; \tfrac{3}{2}, "
            r"\tfrac{1}{2} \rangle"
        )
        assert wb.expr(written) == 1 / sympy.sqrt(3)
