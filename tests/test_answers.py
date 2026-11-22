r"""What a program answers, as wave 0 of ``curriculum/ROADMAP.md`` asks: a claim is decided,
or refused as undecided, never shown as if it were an answer."""

import pytest
import sympy

from relativisticpy import Workbook, holds
from relativisticpy.checking import Verdict
from relativisticpy.documents import Document
from relativisticpy.errors import EvaluationError
from relativisticpy.tensors.equation import UndecidedError

x, y = sympy.symbols("x y")


@pytest.fixture
def wb():
    with Workbook() as workbook:
        yield workbook


def undecided(wb, source):
    with pytest.raises(UndecidedError) as raised:
        wb.exe(source)
    return raised.value


class TestClaimsAreDecidedOrRefused:
    """Guard 5: a claim whose truth is not decided raises an UndecidedError where it is
    written. ``e^x \\geq 1 + x`` came back as itself, which reads as an answer."""

    def test_an_inequality_sympy_cannot_compare(self, wb):
        found = undecided(wb, r"e^x \geq 1 + x")
        assert found.message == (
            r"whether e^x \geq 1 + x holds is not decided: SymPy decides an inequality only "
            "where it can compare its sides"
        )
        assert found.notes == ["shown as it is, an undecided claim would read as an answer"]
        assert (
            found.hint
            == "to see what is left, write the difference of its sides alone, as an expression"
        )

    def test_an_identity_sympy_cannot_prove(self, wb):
        wb.exe(r"F(K, L) := A K^{\alpha} L^{1 - \alpha}")
        found = undecided(wb, r"F(\lambda K, \lambda L) = \lambda F(K, L)")
        assert found.message.endswith(
            "is not decided: the sides agree numerically, but this was not proven"
        )

    def test_an_equation_of_matrices(self, wb):
        boost = (
            r"\begin{pmatrix} \cosh\zeta & -\sinh\zeta \\ -\sinh\zeta & \cosh\zeta \end{pmatrix}"
        )
        wb.exe(rf"B(\zeta) := {boost}")
        # each component is decided as a scalar equation is, by cosh^2 - sinh^2 = 1 too
        assert wb.expr(r"B(\zeta_1) B(\zeta_2) = B(\zeta_1 + \zeta_2)")
        found = undecided(  # true where x > 0 alone
            wb,
            r"\begin{pmatrix} \arctan(x) + \arctan(1/x) & 0 \end{pmatrix} = "
            r"\begin{pmatrix} \pi/2 & 0 \end{pmatrix}",
        )
        assert "in 1 of 2 components, SymPy could not show the sides equal" in found.message

    def test_an_order_term(self, wb):
        found = undecided(wb, r"e^x \sin x = x + x^2 + \frac{x^3}{3} + O(x^5)")
        assert found.message.endswith(
            "is not decided: it has an order term, O(...), which no value at a point decides"
        )
        assert holds(sympy.Eq(sympy.sin(x), x + sympy.O(x**3))) is None

    def test_what_is_decided_or_no_claim_is_shown(self, wb):
        assert wb.expr(r"\cosh^2 x - \sinh^2 x = 1") is sympy.true
        assert wb.expr("x + 1 = x") is sympy.false
        assert wb.expr("x^2 + y^2 = 1") == sympy.Eq(x**2 + y**2, 1)  # an equation, to solve
        assert wb.expr("solve(x^2 = 4, x)") == [-2, 2]
        assert wb.expr("a > 0") is None  # an assumption, which shows nothing
        assert wb.expr("r' < r") == (sympy.Symbol("r'") < sympy.Symbol("r"))
        assert wb.expr("x >= y + 1") == (x >= y + 1)  # between rational functions: a condition

    def test_a_check_says_it_is_not_decided(self):
        report = Workbook().check(Document(r"\begin{equation} e^x \geq 1 + x \end{equation}"))
        [finding] = report.findings
        assert finding.verdict is Verdict.NOT_COMPUTED
        assert finding.detail.startswith(r"whether e^x \geq 1 + x holds is not decided")


def notes(wb, source):
    """The notes of the last step of a program."""
    *_, step = wb.each(source)
    assert step.error is None, step.error
    return step.notes


class TestUndoneResultsAreNoted:
    """Guard 7: a result that holds a computation left undone, or conditions SymPy puts on
    its parameters, says so, rather than read as the answer."""

    def test_a_sum(self, wb):
        assert notes(wb, r"\sum_{n=1}^{\infty} \frac{1}{n^n}") == [
            "a sum is left as written: SymPy found no closed form for it in the time it was given"
        ]
        assert wb.expr(r"\sum_{n=1}^{\infty} \frac{1}{n^2}") == sympy.pi**2 / 6  # decision 4

    def test_an_integral(self, wb):
        undone = ["an integral is left as written: SymPy could not compute it"]
        assert notes(wb, r"\int_0^\infty \frac{x^3}{e^x - 1} \, dx") == undone
        assert notes(wb, r"[1, \int_0^\infty \frac{x^3}{e^x - 1} \, dx]") == undone  # in a list

    def test_conditions_on_the_parameters(self, wb):
        # that no assumption books make removes, as they do of a Gaussian's width
        found = notes(wb, r"\int_{-\infty}^{\infty} e^{-i \omega t} \, dt")
        assert found[-1] == (
            "this holds under the conditions SymPy gives, as it takes every symbol to be "
            "complex; saying what the parameters are, as in a > 0, may remove them"
        )

    def test_what_is_done_has_no_note(self, wb):
        assert notes(wb, r"\int_0^1 x^2 \, dx") == []
        assert notes(wb, "f'(0)") == []  # the derivative of an unknown function, as it is
        written = r"f(x) := \begin{cases} x & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases}"
        assert notes(wb, written + "\nf(y)") == []  # cases the program wrote


class TestLimitsAreOfTheValuesNear:
    """Guard 7: a limit never takes a case that holds only at the limit point, and one that
    does not exist says so, where SymPy gives a value."""

    def test_a_case_at_the_point_is_not_taken(self, wb):
        cases = (
            r"\begin{cases} 0 & \text{if } k = 0 \\ \frac{\sin k}{k} & \text{otherwise} \end{cases}"
        )
        assert wb.expr(rf"\lim_{{k \to 0}} {cases}") == 1  # SymPy gives 0

    def test_a_limit_from_two_sides_that_differ(self, wb):
        step = r"\begin{cases} 1 & \text{if } x > 0 \\ -1 & \text{otherwise} \end{cases}"
        with pytest.raises(EvaluationError) as raised:  # SymPy gives -1
            wb.exe(rf"\lim_{{x \to 0}} {step}")
        assert (
            raised.value.message == "the limit does not exist: it is -1 from below and 1 from above"
        )
        with pytest.raises(EvaluationError, match="it is -1 from below and 1 from above"):
            wb.exe(r"\lim_{x \to 0} \frac{|x|}{x}")

    def test_values_that_never_settle(self, wb):
        with pytest.raises(EvaluationError) as raised:  # SymPy gives AccumBounds(-1, 1)
            wb.exe(r"\lim_{h \to 0} \sin\frac{1}{h}")
        assert raised.value.message == "the limit does not exist: the values keep between -1 and 1"

    CASES = (
        r"\begin{cases} %s \sin\frac{1}{x} & \text{if } x \neq 0 \\ "
        r"0 & \text{if } x = 0 \end{cases}"
    )

    def test_a_derivative_where_cases_meet(self, wb):
        wb.exe(f"f(x) := {self.CASES % 'x'}")
        with pytest.raises(EvaluationError) as raised:  # case by case, it was 0
            wb.exe("f'(0)")
        assert raised.value.message == (
            "the derivative at 0 does not exist: its difference quotient has no limit, as the "
            "values keep between -1 and 1"
        )
        wb.exe(f"g(x) := {self.CASES % 'x^2'}")
        assert wb.expr("g'(0)") == 0
