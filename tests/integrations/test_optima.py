r"""Optima: the greatest and the least value over variables, or along an index of data, as
books write them, and where they are; found by the optimizers, SymPy's exactly and SciPy's
numerically, each saying how it found the optimum and whether it is global; and what the
conformance suite says of an optimizer that breaks the pattern."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
import sympy

from relativisticpy import Workbook, integrations
from relativisticpy.errors import CompileError, EvaluationError
from relativisticpy.integrations import Example, Found, Problem, optimizer

x, y = sympy.symbols("x y", real=True)
a, c, m, p = sympy.symbols("a c m p")


def notes(wb: Workbook, source: str) -> list[str]:
    return [note for step in wb.each(source) for note in step.notes]


class TestOneVariable:
    @pytest.mark.parametrize(
        ("source", "value"),
        [
            (r"\min_{x \in \mathbb{R}} \left(x^2 - 4x + 7\right)", 3),
            (r"\min_x \; x^2 - 4x + 7", 3),  # of the whole sum it begins
            (r"\max_{0 \leq x \leq 3} \left(x^3 - 3x\right)", 18),  # at an end
            (r"\arg\max_{x \geq 0} \; x e^{-x}", 1),
            (r"\inf_{x > 0} \left(x + \frac{1}{x}\right)", 2),
            (r"\sup_{x \geq 0} \frac{x}{1 + x}", 1),  # never reached
            (r"\min_x |x - 1| + |x + 2|", 3),  # where it has no derivative
            ("min_{x} (x^2 + 1)", 1),
            (r"\underset{x}{\min} \; (x - 3)^2", 0),
        ],
    )
    def test_from_its_stationary_points_and_the_ends_of_its_domain(self, wb, source, value):
        assert wb.expr(source) == value

    def test_a_note_says_how_it_was_found_and_whether_it_is_global(self, wb):
        assert notes(wb, r"\min_{x} \; (x - 2)^2 + 3") == [
            "the global minimum, found exactly by SymPy, from its stationary points and the "
            "ends of its domain (extrema)"
        ]

    def test_a_maximum_no_point_reaches_is_refused_and_the_supremum_given(self, wb):
        with pytest.raises(EvaluationError, match="has no maximum") as refused:
            wb.expr(r"\max_{x \geq 0} \frac{x}{1 + x}")
        assert "\\sup" in (refused.value.hint or "")
        with pytest.raises(EvaluationError, match="grows without bound"):
            wb.expr(r"\max_x x^2")
        assert wb.expr(r"\sup_x x^2") == sympy.oo

    def test_an_optimum_inside_another_is_of_its_parameter(self, wb):
        payoff = (
            r"\begin{pmatrix} p & 1 - p \end{pmatrix} A \begin{pmatrix} q \\ 1 - q \end{pmatrix}"
        )
        wb.exe(r"A := \begin{pmatrix} 3 & -1 \\ -2 & 1 \end{pmatrix}")
        inner = wb.expr(rf"\min_{{0 \leq q \leq 1}} {payoff}")
        assert inner == sympy.Min(1 - 2 * sympy.Symbol("p"), 5 * sympy.Symbol("p") - 2)
        assert wb.expr(rf"\max_{{0 \leq p \leq 1}} \min_{{0 \leq q \leq 1}} {payoff}") == (
            sympy.Rational(1, 7)
        )

    @pytest.mark.parametrize(
        ("source", "value"),
        [
            (r"\max_{5 \leq x \leq 10} \sin(x)", 1),  # at 5π/2, no principal root of cos x
            (r"\min_{4 \leq x \leq 7} \cos(x)", sympy.cos(4)),  # 2π is a maximum
            (r"\max_x \sin x", 1),
            (r"\max_{0 \leq x \leq 20} \sin(x) e^{-x/10}",
             sympy.sin(sympy.atan(10)) * sympy.exp(-sympy.atan(10) / 10)),
        ],
    )  # fmt: skip
    def test_every_stationary_point_of_a_periodic_function_is_tried(self, wb, source, value):
        assert sympy.simplify(wb.expr(source) - value) == 0

    @pytest.mark.parametrize(
        ("source", "bound"),
        [
            (r"\max_{0 \leq x \leq 3} \tan(x)", "∞"),
            (r"\max_{-1 \leq x \leq 1} \frac{1}{x}", "∞"),
            (r"\min_{-1 \leq x \leq 1} \frac{1}{x}", "−∞"),
            (r"\min_{x} \frac{1}{x^2 - 1}", "−∞"),
            (r"\max_x \tan x", "∞"),  # toward each of its poles
        ],
    )
    def test_toward_a_point_where_it_is_not_defined_it_is_not_reached(self, wb, source, bound):
        with pytest.raises(EvaluationError, match=f"grows without bound, to {bound}"):
            wb.expr(source)

    def test_a_note_says_the_points_where_it_is_not_defined_were_tried(self, wb):
        source = r"\sup_{0 \leq x \leq 3} \tan(x)"
        assert wb.expr(source) == sympy.oo
        assert notes(wb, source) == [
            "the supremum, which no point reaches, found exactly by SymPy, from its stationary "
            "points, the ends of its domain, and the points where it is not defined (extrema)"
        ]
        with pytest.raises(EvaluationError, match="as near as you like to 2, which no point"):
            wb.expr(r"\max_{0 \leq x \leq 1} \frac{x^2 - 1}{x - 1}")  # not defined at 1

    def test_one_whose_poles_sympy_does_not_give_is_left_to_a_numeric_optimizer(self, wb):
        with integrations._optimizers.answering() as answers:
            wb.expr(r"\min_{1 \leq x \leq 3} \Gamma(x)")
        assert [found.by for found in answers] == ["minimize"]

    def test_every_point_of_an_interval_that_reaches_it_is_said(self, wb):
        source = r"\arg\min_x \left(|x-1|+|x+1|\right)"
        assert wb.expr(source) == -1
        assert notes(wb, source)[-1].endswith(
            "(extrema); every point of [−1, 1] reaches it, and −1 is given"
        )
        assert notes(wb, r"\max_{-1 \leq x \leq 1} x^2")[-1].endswith(
            "(extrema); it is reached at −1 and 1, and −1 is given"
        )
        assert notes(wb, r"\max_x \sin x")[-1].endswith(
            "(extrema); it is reached at 2n π + π/2, of every integer n, and π/2 is given"
        )
        assert notes(wb, r"\max_{0 \leq x \leq 40} \sin(x)")[-1].endswith(
            "(extrema); 7 points reach it, and π/2 is given"
        )

    def test_a_periodic_function_has_no_limit_toward_infinity(self, wb):
        # the bounds SymPy gives sin x cos x there, -1 and 1, are no values it comes near
        assert wb.expr(r"\max_x \sin(x)\cos(x)") == sympy.Rational(1, 2)
        assert wb.expr(r"\max_x a \sin(x)") == sympy.Max(a, -a)

    def test_where_a_case_begins_with_a_jump_its_value_is_approached(self, wb):
        wb.exe(r"f(x) := \begin{cases} x & x < 0 \\ x - 5 & x \geq 0 \end{cases}")
        with pytest.raises(EvaluationError, match="as near as you like to 0, which no point"):
            wb.expr(r"\max_{-1 \leq x \leq 1} f(x)")
        assert wb.expr(r"\sup_{-1 \leq x \leq 1} f(x)") == 0

    @pytest.mark.parametrize("seed", ["0", "1", "2", "3"])
    def test_where_it_is_is_the_same_under_every_hash_seed(self, seed):
        program = r"""
from relativisticpy import Workbook
wb = Workbook()
print(wb.expr(r"\arg\min_x \left(|x-1|+|x+1|\right)"))
print(wb.expr(r"\arg\min_x \left(|x-2|+|x+3|\right)"))
print(wb.expr(r"\arg\max_x \left(-|x-1|-|x+4|\right)"))
"""
        done = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
            timeout=120,
            check=False,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.split() == ["-1", "-3", "-4"]


class TestParameters:
    def test_what_the_program_assumes_is_used(self, wb):
        wb.exe("0 < a < 1\np_x, p_y, m > 0")
        found = wb.expr(
            r"\arg\max_{x, y} \; x^{a} y^{1 - a} \quad \text{subject to} \quad p_x x + p_y y = m"
        )
        px, py = sympy.symbols("p_x p_y")
        plain = [f.xreplace({s: sympy.Symbol(s.name) for s in f.free_symbols}) for f in found]
        assert plain == [a * m / px, m * (1 - a) / py]

    def test_where_signs_are_not_said_they_are_taken_as_books_take_them(self, wb):
        wb.exe(r"L(v) = -m c^2 \sqrt{1 - \frac{v^2}{c^2}}" + "\n" + r"H(p) = \max_v (p v - L(v))")
        steps = list(wb.each("H(p)"))
        assert steps[0].values == [c * sympy.sqrt(c**2 * m**2 + p**2)]
        assert steps[0].assuming == ["c > 0", "m > 0"]

    def test_a_function_defined_by_an_optimum_takes_its_parameter(self, wb):
        wb.exe(
            r"L(x_1, x_2, \nu) := x_1^2 + x_2^2 + \nu (x_1 + x_2 - 1)"
            + "\n"
            + r"g(\nu) := \inf_{x_1, x_2} L(x_1, x_2, \nu)"
        )
        nu = sympy.Symbol("nu")
        assert wb.expr(r"g(\nu)") == -(nu**2) / 2 - nu


class TestSeveralVariables:
    @pytest.mark.parametrize(
        ("source", "value", "by"),
        [
            (r"\min_{x_1, x_2} \; x_1 + x_2 \quad \text{subject to} \quad x_1^2 + x_2^2 = 2",
             -2, "kkt"),
            (r"\max_{x_1, x_2} \; 3x_1 + 5x_2 \quad \text{subject to} \quad x_1 \leq 4, \quad "
             r"2x_2 \leq 12, \quad 3x_1 + 2x_2 \leq 18, \quad x_1, x_2 \geq 0", 36, "simplex"),
            (r"\max_{x \in \{0, 1\}^3} \; 60x_1 + 100x_2 + 120x_3 \quad \text{subject to} \quad "
             r"10x_1 + 20x_2 + 30x_3 \leq 50", 220, "enumeration"),
            (r"\max_{x, y} \; 0.5x + 0.3y \quad \text{subject to} \quad x + y \leq 10, \quad "
             r"x \leq 6, \quad x, y \geq 0", 4.2, "linprog"),
        ],
    )  # fmt: skip
    def test_each_by_the_method_that_fits_it(self, wb, source, value, by):
        with integrations._optimizers.answering() as answers:
            assert wb.expr(source) == pytest.approx(value)
        assert [found.by for found in answers] == [by]

    def test_constraints_on_the_lines_after_it(self, wb):
        source = "\\min_{x, y} x + y\n\\text{subject to} \\quad x^2 + y^2 = 2"
        assert wb.exe(source) == [-2]

    def test_a_vector_takes_its_length_from_the_matrix_it_is_multiplied_by(self, wb):
        rows = r"1 & -2 & 4 \\ 1 & -1 & 1 \\ 1 & 0 & 0 \\ 1 & 1 & 1 \\ 1 & 2 & 4"
        wb.exe(
            rf"""
            A := \begin{{pmatrix}} {rows} \end{{pmatrix}}
            \mathbf{{b}} := \begin{{pmatrix}} 4 \\ 1 \\ 1 \\ 2 \\ 7 \end{{pmatrix}}
            """
        )
        found = wb.expr(r"\arg\min_{\mathbf{x}} \|A\mathbf{x} - \mathbf{b}\|_2^2")
        assert found.tolist() == [
            [sympy.Rational(4, 7)],
            [sympy.Rational(7, 10)],
            [sympy.Rational(17, 14)],
        ]

    def test_componentwise_constraints_between_vectors(self, wb):
        wb.exe(
            r"""
            A := \begin{pmatrix} 1 & 0 \\ 0 & 2 \\ 3 & 2 \end{pmatrix}
            b := \begin{pmatrix} 4 \\ 12 \\ 18 \end{pmatrix}
            c := \begin{pmatrix} 3 \\ 5 \end{pmatrix}
            """
        )
        dual = r"\min_{y} \; b^T y \quad \text{subject to} \quad "
        assert wb.expr(dual + r"A^T y \geq c, \quad y \geq 0") == 36
        assert wb.expr(dual + r"A^T y \succeq c, \quad y \succeq 0") == 36  # componentwise

    def test_a_tensor_is_found_over_its_space(self, wb):
        wb.exe(
            r"""
            \mathcal{A} := \mathbb{R}^{2}
            a, b \in \mathcal{A}
            \Sigma_{ab} := [[0.0144, 0.0072], [0.0072, 0.04]]
            """
        )
        found = wb.expr(
            r"\arg\min_{w} \; w^{a} \Sigma_{ab} w^{b} \quad \text{subject to} \quad 1_{a} w^{a} = 1"
        )
        assert np.allclose(found.to_numpy().astype(float), [0.82, 0.18], atol=1e-12)
        assert found.spaces[0].name == "A"

    def test_whole_numbers_with_no_end_are_found_over_the_reals_first(self, wb):
        source = r"\arg\max_{n \in \mathbb{N}} \frac{\pi^{n/2}}{\Gamma\left(\frac{n}{2} + 1\right)}"
        assert wb.expr(source) == 5
        assert "not proven global" in notes(wb, source)[-1]

    @pytest.mark.parametrize(
        ("source", "value"),
        [
            (r"\max_{x, y} \left(x^2 + y^2\right) \quad \text{subject to} \quad x^2 + y^2 \leq 1",
             1),
            (r"\max_{x, y, z} \left(x^2 + y^2 + z^2\right) \quad \text{subject to} \quad "
             r"x^2 + y^2 + z^2 \leq 1", 1),
            (r"\min_{x, y} \left(-(x^2 + y^2)\right) \quad \text{subject to} \quad "
             r"x^2 + y^2 \leq 1", -1),
            (r"\max_{x, y} \left(x^2 + 2 y^2\right) \quad \text{subject to} \quad "
             r"x^2 + y^2 \leq 1", 2),
        ],
    )  # fmt: skip
    def test_an_optimum_reached_along_a_curve_is_found_there(self, wb, source, value):
        # the centre keeps the first-order conditions too, and is the least, not the greatest
        with integrations._optimizers.answering() as answers:
            assert wb.expr(source) == value
        assert [(found.by, found.proven) for found in answers] == [("kkt", True)]

    def test_a_note_says_the_curve_it_is_reached_along(self, wb):
        source = (
            r"\arg\max_{x, y} \left(x^2 + y^2\right) \quad \text{subject to} \quad x^2 + y^2 \leq 1"
        )
        assert wb.expr(source) == [-1, 0]
        assert notes(wb, source)[-1].endswith(
            "(kkt); every point where x = √(1 − y²) or x = −√(1 − y²) reaches it, and (−1, 0) "
            "is given"
        )

    def test_a_point_the_second_order_conditions_rule_out_is_no_optimum(self):
        kkt = integrations.find("kkt").exact
        saddle = Problem("min", x**2 - y**2, (x, y), inequalities=(1 - x**2 - y**2,))
        found = kkt(saddle)
        assert (found.value, found.proven) == (-1, True)  # at (0, ±1), not the saddle at 0
        assert found.tied == sympy.FiniteSet(sympy.Tuple(0, -1), sympy.Tuple(0, 1))

    def test_a_periodic_function_between_constraints_is_left_to_a_numeric_optimizer(self, wb):
        # SymPy solves cos x = 0 for its principal roots alone, neither of them in the box
        source = (
            r"\max_{x, y} \; \sin(x) + \sin(y) \quad \text{subject to} \quad "
            r"5 \leq x \leq 10, \quad 5 \leq y \leq 10"
        )
        with integrations._optimizers.answering() as answers:
            assert wb.expr(source) == pytest.approx(2)
        assert [found.by for found in answers] == ["minimize"]

    def test_one_that_falls_without_bound_along_no_axis(self, wb):
        with pytest.raises(EvaluationError, match="grows without bound, to −∞"):
            wb.expr(r"\min_{x, y} \; (x - y)^2 - x - y")


class TestNumerically:
    """What SciPy finds is an optimum only where its search converged, and nothing lower is
    found far away; and it is the number SciPy found, not a whole number near it."""

    @pytest.mark.parametrize(
        "objective",
        [x**2 - 1.0 * y, x**3 - 3.0 * x, sympy.exp(-x) * 1.0, -1.0 * x * y],
    )
    def test_a_search_that_runs_away_finds_no_minimum(self, objective):
        variables = tuple(sorted(objective.free_symbols, key=str))
        problem = Problem("min", objective, variables)
        assert integrations.find("minimize").exact(problem) is None

    @pytest.mark.parametrize(
        ("source", "said"),
        [
            (r"\min_{x, y} \left(x^2 - 1.0 y\right)", "grows without bound, to −∞"),
            (r"\min_{x} \left(x^3 - 3.0 x\right)", "grows without bound, to −∞"),
            (r"\min_{x} e^{-x} \cdot 1.0", "as near as you like to 0, which no point reaches"),
        ],
    )
    def test_what_has_no_minimum_is_said_to_have_none(self, wb, source, said):
        with pytest.raises(EvaluationError, match=said):
            wb.expr(source)

    def test_one_that_is_bounded_is_found_as_before(self, wb):
        source = r"\min_{x} \; x^4 - 3.0 x^2 + x"
        assert wb.expr(source) == pytest.approx(-3.513905038934789)
        assert "not proven global" in notes(wb, source)[-1]

    def test_a_value_near_a_whole_number_is_not_made_one(self, wb):
        assert wb.expr(r"\min_{x, y} \left(x^2 + y^2 + 0.0000000003\right)") == pytest.approx(
            3e-10, rel=1e-9
        )
        where = wb.expr(r"\arg\min_{x, y} \left((x - 0.0000000004)^2 + y^2\right)")
        assert [float(each) for each in where] == pytest.approx([4e-10, 0.0], abs=1e-18)

    def test_whole_numbers_of_whole_data_stay_whole(self, wb):
        source = (
            r"\max_{x, y \in \mathbb{Z}} \; x + y \quad \text{subject to} \quad "
            r"2x + 2y \leq 7, \quad x, y \geq 0"
        )
        assert isinstance(wb.expr(source), sympy.Integer)
        program = (
            r"\max_{x, y} \; 0.5x + 0.3y \quad \text{subject to} \quad x + y \leq 10, \quad "
            r"x \leq 6, \quad x, y \geq 0"
        )
        assert isinstance(wb.expr(program), sympy.Float)  # of data, a decimal


class TestOverAnIndex:
    def test_the_greatest_value_along_an_index_of_data(self, wb):
        wb["y_t"] = np.array([1.0, 2.0, 0.0, -1.0])
        found = wb.exe(r"\max_t y_t" + "\n" + r"\min_t y_t" + "\n" + r"\arg\min_t y_t")
        assert found == [2.0, -1.0, 3]

    def test_along_one_index_of_several(self, wb):
        wb["R_{ta}"] = np.array([[0.01, 0.02, -0.01], [0.03, -0.02, 0.0]])
        assert wb.expr(r"\max_t R_{ta}").to_numpy().tolist() == [0.03, 0.02, 0.0]

    def test_where_it_is_is_the_label_and_the_first_of_ties_is_given(self, wb):
        wb.exe(r"N := 8" + "\n" + r"x_n := \frac{n}{N}, \quad n = 0, \ldots, N-1")
        assert wb.expr(r"\arg\max_n \sin(\pi x_n)") == 4
        source = r"\arg\max_n \sin(2\pi x_n)^2"
        assert wb.expr(source) == 2
        assert notes(wb, source) == [
            "\\arg\\max_n: the optimum is reached at 2 and 6, and the first is given"
        ]

    def test_complex_data_has_no_order(self, wb):
        wb["z_t"] = np.array([1 + 1j, 2.0])
        with pytest.raises(EvaluationError, match="have no order"):
            wb.expr(r"\max_t z_t")


class TestRefusals:
    def test_an_optimum_over_a_function_is_not_supported_yet(self, wb):
        with pytest.raises(CompileError, match="is over a function, u"):
            wb.expr(r"\min_{u} \int_0^1 u(t)^2 \, dt")

    def test_an_optimum_of_a_function_nothing_defines_is_refused(self, wb):
        with pytest.raises(EvaluationError, match="a function the program does not define") as no:
            wb.expr(r"\max_{0 < c < k} \left[\ln c + \beta V(k - c)\right]")
        assert no.value.hint == "define V before the optimum"

    def test_an_optimum_over_a_set_the_language_does_not_know(self, wb):
        with pytest.raises(CompileError, match="a set the language does not know"):
            wb.expr(r"\inf_{\gamma \in \Pi(\mu, \nu)} \gamma")

    def test_a_variable_before_brackets_that_hold_it_multiplies_them(self, wb):
        assert wb.expr(r"\max_x x(1 - x)") == sympy.Rational(1, 4)  # decision 28

    def test_constraints_nothing_keeps(self, wb):
        source = r"\max_{x, y} \; x + y \quad \text{s.t.} \quad x + y \leq 1, \quad x + y \geq 2"
        with pytest.raises(EvaluationError, match="no point keeps the constraints"):
            wb.expr(source)


class TestGeneralizedInequalities:
    @pytest.mark.parametrize(
        ("source", "holds"),
        [
            (r"\begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix} \succ 0", True),
            (r"\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} \succ 0", False),
            (r"\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} \succeq 0", True),
            (r"\begin{pmatrix} 1 \\ 2 \end{pmatrix} \succeq \begin{pmatrix} 0 \\ 2 \end{pmatrix}",
             True),
        ],
    )  # fmt: skip
    def test_definiteness_and_components(self, wb, source, holds):
        assert bool(wb.expr(source)) is holds


class TestTheKind:
    def test_the_optimizers_are_consulted_exact_first(self):
        found = [(each.name, each.numerical) for each in integrations._optimizers.optimizers()]
        assert found == [
            ("extrema", False), ("kkt", False), ("simplex", False), ("enumeration", False),
            ("linprog", True), ("quadratic", True), ("minimize", True),
        ]  # fmt: skip

    def test_an_optimizer_that_reports_what_breaks_its_constraints_is_told(self):
        @optimizer(
            "careless",
            summary="a careless optimizer",
            problems=[
                Problem("min", (x - 1) ** 2 + (y - 2) ** 2, (x, y), inequalities=(1 - x - y,))
            ],
            examples=[
                Example(r"\min_{x} x^2", 0, "standard"),
                Example(r"\min_{x} x^4", 0, "standard"),
            ],
        )
        class Careless:
            def takes(self, problem):
                return True

            def optimum(self, problem):
                return Found(sympy.Integer(0), (sympy.Integer(1), sympy.Integer(2)), "badly", True)

        failures = integrations.check(Careless, ["feasibility", "agreement"])
        assert failures["feasibility"][0].startswith(
            "careless: at the least (x − 1)² + (y − 2)² over x, y with 1 − x − y ≥ 0, the point "
            "it reports, (1, 2), breaks 1 − x − y ≥ 0"
        )
        assert any("disagree" in failure for failure in failures["agreement"])

    def test_one_that_is_not_stationary_is_told(self):
        @optimizer(
            "hasty",
            summary="a hasty optimizer",
            problems=[Problem("min", (x - 1) ** 2 + y**2, (x, y))],
            examples=[
                Example(r"\min_{x} x^2", 0, "standard"),
                Example(r"\min_{x} x^4", 0, "standard"),
            ],
        )
        class Hasty:
            def takes(self, problem):
                return True

            def optimum(self, problem):
                return Found(sympy.Integer(1), (sympy.Integer(0), sympy.Integer(0)), "badly", False)

        (failure,) = integrations.check(Hasty, ["optimality"])["optimality"]
        assert "is not stationary" in failure

    def test_a_class_without_its_methods_is_told_what_to_declare(self):
        with pytest.raises(TypeError, match="has no takes method"):

            @optimizer("lazy", summary="a lazy optimizer", problems=[], examples=[])
            class Lazy:
                def optimum(self, problem):
                    return None
