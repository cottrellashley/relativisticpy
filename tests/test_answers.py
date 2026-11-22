r"""What a program answers, as wave 0 of ``curriculum/ROADMAP.md`` asks: a claim is decided,
or refused as undecided, never shown as if it were an answer."""

import pytest
import sympy

from relativisticpy import Workbook, holds
from relativisticpy.checking import Verdict
from relativisticpy.documents import Document
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
        found = undecided(wb, r"B(\zeta_1) B(\zeta_2) = B(\zeta_1 + \zeta_2)")
        assert "in 3 of 3 components, SymPy could not show the sides equal" in found.message

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
        assert wb.expr("a > 0") == (sympy.Symbol("a") > 0)  # a condition, as written
        assert wb.expr("r' < r") == (sympy.Symbol("r'") < sympy.Symbol("r"))
        assert wb.expr("x >= y + 1") == (x >= y + 1)  # between rational functions: a condition

    def test_a_check_says_it_is_not_decided(self):
        report = Workbook().check(Document(r"\begin{equation} e^x \geq 1 + x \end{equation}"))
        [finding] = report.findings
        assert finding.verdict is Verdict.NOT_COMPUTED
        assert finding.detail.startswith(r"whether e^x \geq 1 + x holds is not decided")
