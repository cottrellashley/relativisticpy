r"""Solving respects what a program assumes, and does not fail where another path does not.

``solve`` drops the solutions a stated assumption rules out, with a note that says which
and why; it keeps the roots of a fraction that also zero a factor it cancels; it says so
where the equation does not contain the unknown; and a failure inside SymPy that another
path avoids does not end it (design/interface.md, the same answer every time). ``dsolve``
takes conditions written as equations, y(0) = y_0.
"""

import os
import subprocess
import sys

import pytest
import sympy

from relativisticpy import Workbook
from relativisticpy.errors import EvaluationError
from relativisticpy.integrations import equations

x, t, r, g, h = sympy.symbols("x t r g h")


def step(wb, source):
    """The last step of a program, which must not fail."""
    *_, last = wb.each(source)
    assert last.error is None, last.error
    return last


class TestAssumptionsRuleOutSolutions:
    def test_a_sign_drops_roots_and_says_so(self, wb):
        last = step(wb, "v > 0\ng, h > 0\nsolve(v^2 = 2 g h, v)")
        positive = sympy.Symbol("g", positive=True), sympy.Symbol("h", positive=True)
        assert last.values == [[sympy.sqrt(2) * sympy.sqrt(positive[0] * positive[1])]]
        assert last.notes == ["solve dropped v = −√2 √g √h, as v is assumed positive"]

    def test_complex_roots_of_a_positive_unknown(self, wb):
        last = step(wb, "a > 0\nsolve(a^3 = 8, a)")
        assert last.values == [[2]]
        assert last.notes == [
            "solve dropped a = −1 − √3 i and a = √3 i − 1, as a is assumed positive"
        ]

    def test_a_bound_drops_roots(self, wb):
        last = step(wb, "0 < x < 1\nsolve(2x^2 - 3x + 1 = 0, x)")
        assert last.values == [[sympy.Rational(1, 2)]]
        assert last.notes == ["solve dropped x = 1, as x < 1 is assumed"]

    def test_several_unknowns_each_say_why(self, wb):
        last = step(
            wb,
            "K > 0\n\\omega > 0\n"
            "solve((i\\omega)^3 + 3(i\\omega)^2 + 3i\\omega + 1 + K = 0, [K, \\omega])",
        )
        assert [tuple(map(str, s)) for s in last.values[0]] == [("8", "sqrt(3)")]
        assert last.notes == [
            "solve dropped (K, ω) = (−1, 0), as K is assumed positive",
            "solve dropped (K, ω) = (8, −√3), as ω is assumed positive",
        ]

    def test_what_cannot_be_decided_is_kept(self, wb):
        last = step(wb, "v > 0\nsolve(v^2 = 2 g h, v)")  # g h may be negative
        assert len(last.values[0]) == 2
        assert last.notes == []

    def test_without_assumptions_nothing_is_dropped(self, wb):
        assert wb.expr("solve(x^2 = 4, x)") == [-2, 2]


class TestRobustly:
    def test_a_root_that_zeroes_a_cancelled_factor_is_kept(self, wb):
        source = r"""
            f := 1 - \frac{2M}{r} + \frac{Q^2}{r^2}
            solve(\frac{r^2 f^2}{r^2 - 2Mr + Q^2} = 0, r)
        """
        found = wb.expr(source)
        M, Q = sympy.symbols("M Q")
        horizons = [M - sympy.sqrt(M**2 - Q**2), M + sympy.sqrt(M**2 - Q**2)]
        assert len(found) == 2
        assert all(any(sympy.simplify(a - b) == 0 for b in horizons) for a in found)
        assert (
            sympy.solve(r**2 * (1 - 2 * M / r + Q**2 / r**2) ** 2 / (r**2 - 2 * M * r + Q**2), r)
            == []
        )

    def test_a_coefficient_that_vanishes_by_an_identity_is_dropped(self, wb):
        # Compton's equation is linear in E' only through cos^2 + sin^2 = 1: SymPy, keeping
        # the vanishing square, divides by it and finds no solution
        found = wb.expr(r"solve((\cos^2\theta + \sin^2\theta - 1) x^2 + 2 x - 4 = 0, x)")
        assert found == [2]
        theta, x = sympy.symbols("theta x")
        vanishing = (sympy.cos(theta) ** 2 + sympy.sin(theta) ** 2 - 1) * x**2 + 2 * x - 4
        assert sympy.solve(vanishing, x) != [2]

    def test_an_equation_that_holds_for_every_value(self, wb):
        with pytest.raises(EvaluationError) as raised:
            wb.exe("solve(x - x = 0, x)")
        assert raised.value.message == (
            "the equation holds whatever x is, so there is nothing to solve for"
        )
        assert wb.expr("solve(x - x = 1, x)") == []

    def test_an_unknown_the_equation_does_not_contain(self, wb):
        with pytest.raises(EvaluationError) as raised:
            wb.exe(r"solve(\beta = -t, q)")
        assert raised.value.message == "β = −t does not contain q, so it cannot be solved for q"

    def test_a_failure_inside_sympy_takes_another_path(self, monkeypatch):
        solve = sympy.solve
        tried = []

        def failing(*args, **kwargs):
            tried.append(kwargs.get("check", True))
            if kwargs.get("check", True):
                raise TypeError("Invalid NaN comparison")  # as SymPy's own check fails
            return solve(*args, **kwargs)

        monkeypatch.setattr(equations, "_essential", lambda *args: None)
        monkeypatch.setattr(sympy, "solve", failing)
        assert Workbook().expr("solve(x^2 = 4, x)") == [-2, 2]
        assert tried == [True, False]

    def test_a_solution_where_an_equation_is_undefined_is_dropped(self, wb):
        assert wb.expr(r"solve(x e^{-1/x} = 0, x)") == []  # e^{-1/x} is not defined at 0
        assert wb.expr(r"solve(\sqrt{x} = -1, x)") == []

    @pytest.mark.parametrize("seed", ["0", "6"])
    def test_the_dieterici_gas_under_every_seed(self, seed):
        program = r"""
from relativisticpy import Workbook
print(Workbook().expr(r'''
a, b, R > 0
p(V, T) := \frac{R T}{V - b} e^{-\frac{a}{R T V}}
solve([\frac{\partial p}{\partial V} = 0, \frac{\partial^2 p}{\partial V^2} = 0], [V, T])
'''))
"""
        environment = {**os.environ, "PYTHONHASHSEED": seed}
        done = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            env=environment,
            timeout=120,
            check=False,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "[(2*b, a/(4*R*b))]"


class TestDifferentialEquations:
    def test_a_condition_written_as_an_equation(self, wb):
        y = sympy.Function("y")
        found = wb.expr(r"dsolve(y'(t) = r y(t), y(0) = y_0)")
        assert found == sympy.Eq(y(t), sympy.Symbol("y_0") * sympy.exp(r * t))

    def test_a_condition_on_a_derivative(self, wb):
        y = sympy.Function("y")
        assert wb.expr(r"dsolve(y''(t) + y(t) = 0, y(0) = 1, y'(0) = 0)") == sympy.Eq(
            y(t), sympy.cos(t)
        )

    def test_the_condition_chooses_the_branch(self, wb):
        y = sympy.Function("y")
        found = wb.expr(r"dsolve(y'(x) = \frac{3x^2 + 4x + 2}{2(y(x) - 1)}, y(0) = -1)")
        assert found == sympy.Eq(y(x), 1 - sympy.sqrt(x**3 + 2 * x**2 + 2 * x + 4))

    def test_what_is_no_differential_equation_is_refused(self, wb):
        with pytest.raises(EvaluationError) as raised:
            wb.exe("dsolve(x^2 = 1)")
        assert raised.value.message == (
            "x² = 1 is no differential equation: it has no derivative of a function written "
            "with its variable"
        )

    def test_the_unknown_named_without_its_variable(self, wb):
        y = sympy.Function("y")
        y_0 = sympy.Symbol("y_0")
        found = wb.expr(r"dsolve(\frac{dy}{dt} = r y, y(0) = y_0)")
        assert found == sympy.Eq(y(t), y_0 * sympy.exp(r * t))
        (reading,) = wb.explain(r"dsolve(\frac{dy}{dt} = r y, y(0) = y_0)")
        assert [d.text() for d in reading.decisions if d.kind == "function named alone"] == [
            "function named alone: y is the unknown y(t), since dsolve solves for it, and it is "
            "differentiated by t"
        ]
        assert wb.expr("y") == sympy.Symbol("y")  # a symbol again, after the dsolve

    def test_a_primed_unknown_is_a_function_of_the_other_name(self, wb):
        y = sympy.Function("y")
        found = wb.expr("dsolve(t y' + 2y = 4t^2, y(1) = 2)")
        assert sympy.simplify(found.rhs - (t**2 + 1 / t**2)) == 0
        assert found.lhs == y(t)

    def test_a_condition_that_is_no_value_is_refused(self, wb):
        with pytest.raises(EvaluationError, match="and not a condition that SymPy reads as True"):
            wb.exe(r"dsolve(u''(r) = -1, u(1) = 0, |u(0)| < \infty)")


class TestBoundsDecideInequalities:
    def test_an_inequality_is_decided_where_the_bounds_decide_it(self, wb):
        wb.exe("0 < x < 1\na > r > 0")
        assert wb.exe("x^2 < 1\n1 - x > 0\nx - 1 > 0\na - r > 0") == [
            sympy.true,
            sympy.true,
            sympy.false,
            sympy.true,
        ]

    def test_only_what_they_decide_is_given_the_bounds(self, wb):
        wb.exe("0 < x < 1")
        lines = wb.ir("solve(x^2 = 1/4, x)\nx + 1").splitlines()
        given = [line.split("=", 1)[1].split()[:3] for line in lines if "assuming" in line]
        assert given == [["call", "solve", "assuming=(x"]]
        assert "assuming" not in Workbook().ir("solve(x^2 = 1/4, x)")


def test_an_unknown_beside_a_function_nothing_defines_is_not_read(wb):
    """δ(t − 5) that nothing defines would leave the solution unknown too: the equation is
    read as written, and refused."""
    with pytest.raises(EvaluationError, match="is no differential equation"):
        wb.exe(r"dsolve(2y'' + y' + 2y = \delta(t - 5), y(0) = 0, y'(0) = 0)")
