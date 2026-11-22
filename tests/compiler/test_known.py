"""What the compiler needs to know is known without computing anything: it comes from a
declaration, or from what is written."""

import numpy as np
import pytest
import sympy

from relativisticpy import Workbook
from relativisticpy.errors import CompileError, EvaluationError


@pytest.fixture
def wb():
    with Workbook() as workbook:
        yield workbook


class TestSpacetimes:
    def test_the_coordinates_are_those_the_spacetime_declares(self, wb):
        wb.exe(r"g_{\mu\nu} := Kerr(M, a)")
        assert [c.name for c in wb.coordinates] == ["t", "r", "theta", "phi"]
        wb.exe(r"g_{\mu\nu} := Minkowski()")
        assert [c.name for c in wb.coordinates] == ["t", "x", "y", "z"]
        wb.exe(r"g_{\mu\nu} := Minkowski(spherical)")
        assert [c.name for c in wb.coordinates] == ["t", "r", "theta", "phi"]

    def test_declared_assumptions_are_kept(self, wb):
        wb.exe("Positive := [r]")
        wb.exe(r"g_{\mu\nu} := Schwarzschild(M)")
        assert wb.coordinates[1].is_positive
        assert wb.expr(r"g_{rr}").has(wb.coordinates[1])

    def test_the_metric_is_an_operation_not_a_constant(self, wb):
        listing = wb.ir(r"g_{\mu\nu} := Schwarzschild(M)" "\n" "g_{tt}")
        assert "call Schwarzschild" in listing
        assert "call core.components coordinates=(t, r, theta, phi)" in listing

    def test_minkowski_takes_its_coordinates_by_name(self, wb):
        wb.exe("k := spherical")
        wb.exe(r"g_{\mu\nu} := Minkowski(k)")
        assert [c.name for c in wb.coordinates] == ["t", "r", "theta", "phi"]
        with pytest.raises(EvaluationError, match="unknown coordinates 3: use 'cartesian'"):
            wb.exe(r"g_{\mu\nu} := Minkowski(3)")
        with pytest.raises(CompileError, match="takes cartesian or spherical coordinates"):
            wb.exe("n := 1 + 1\n" r"g_{\mu\nu} := Minkowski(n)")


class TestSizes:
    def test_a_dimension_is_a_number_or_a_name_given_one(self, wb):
        wb.exe(r"n := 3" "\n" r"\mathcal{A} := \mathbb{R}^{n}")
        assert wb["A"].dim == 3
        with pytest.raises(CompileError, match="whole number, not 2\\+1") as raised:
            wb.exe(r"\mathcal{B} := \mathbb{R}^{2+1}")
        assert "read before anything is computed" in raised.value.hint

    def test_a_count_is_a_number_or_a_name_given_one(self, wb):
        wb.exe(r"\mathcal{A} := \mathbb{R}^{3}" "\n" r"a, b \in \mathcal{A}")
        wb["A^a_b"] = np.diag([1.0, 2.0, 3.0])
        wb["k"] = 2
        assert np.allclose(wb.expr(r"\operatorname{spec}(A^a{}_b, k)").to_numpy(), [2.0, 3.0])
        with pytest.raises(CompileError, match="number of eigenvalues is a whole number"):
            wb.expr(r"\operatorname{spec}(A^a{}_b, 1 + 1)")

    def test_text_in_quotes_is_never_a_size(self, wb):
        with pytest.raises(EvaluationError, match="text in quotes is not a value"):
            wb.exe(r"\mathcal{A} := \mathbb{R}^{'3'}")
        assert sympy.Symbol("A") == wb.expr("A")  # nothing was declared


class TestTheNamesAMetricIsWrittenWith:
    """A definition of a name the metric in force is written with gives it its value there
    too; which names those are is read from what the metric is computed from."""

    LATE = (
        "the metric defined on line {} is written with {}, which has this value there from now on"
    )

    def test_a_parameter_of_a_spacetime(self, wb):
        wb.exe(r"g_{\mu\nu} := Schwarzschild(M)")
        wb.exe("M := 2")
        assert wb._compiler.take_notes() == [self.LATE.format(1, "M")]
        assert wb.expr("Kretschmann") == 192 / sympy.Symbol("r") ** 6

    def test_a_name_given_a_value_before_the_metric(self, wb):
        wb.exe("M := 1\n" r"g_{\mu\nu} := Schwarzschild(M)")
        wb.exe("M := 2")
        assert wb._compiler.take_notes() == []
        assert wb.expr("Kretschmann") == 48 / sympy.Symbol("r") ** 6

    def test_a_parameter_the_spacetime_is_not_written_with(self, wb):
        wb.exe(r"g_{\mu\nu} := FLRW(k, a, G)")
        wb.exe("G := 2")
        assert wb._compiler.take_notes() == []
        wb.exe("a := 5")
        assert wb._compiler.take_notes() == [self.LATE.format(1, "a")]

    def test_what_a_sum_or_an_evaluation_bar_binds(self, wb):
        wb.exe("Coordinates := [t, r]\n" r"g_{ab} := diag(-1 - \sum_{n=0}^{2} n, (1 + x)|_{x=2})")
        wb.exe("n := 2\nx := 3")
        assert wb._compiler.take_notes() == []

    def test_a_name_that_cancels_counts(self, wb):
        # the metric is not computed to find that x has gone
        wb.exe("Coordinates := [t, r]\n" r"g_{ab} := diag(-1, 1 + x - x)")
        wb.exe("x := 2")
        assert wb._compiler.take_notes() == [self.LATE.format(2, "x")]
        assert wb.expr("g_{rr}") == 1


class TestTheScalarsOfNotation:
    """Tensor notation's algebra is done while compiling, with its scalar factors folded
    from the constants they are written with, as the executor would compute them."""

    def test_a_factor_written_with_numbers_symbols_and_functions(self, wb):
        G, t = sympy.symbols("G t")
        equation = wb.expr(r"G_{\mu\nu} = 8\pi G T_{\mu\nu}")
        assert equation.rhs.coeff == 8 * sympy.pi * G
        wb.exe(r"Coordinates := [t, x]")
        equation = wb.expr(r"G_{\mu\nu} = \sqrt{2} \frac{\partial t^2}{\partial t} T_{\mu\nu}")
        assert equation.rhs.coeff == 2 * sympy.sqrt(2) * t

    def test_a_factor_computed_from_tensors_is_refused(self, wb):
        wb.exe(r"g_{\mu\nu} := Schwarzschild(M)")
        with pytest.raises(CompileError, match="computed from tensors") as raised:
            wb.expr(r"G_{\mu\nu} = Ric\, T_{\mu\nu}")
        assert raised.value.span.text == "Ric"


class TestTheChecksOfADefinition:
    """A metric is computed as its definition runs, so that one that is no metric is
    refused where it is defined, and the definition undone; compiling computes nothing."""

    def test_a_definition_makes_a_check_and_computes_nothing(self, wb):
        from relativisticpy.compiler import Compiler, Session
        from relativisticpy.language import Source, parse

        compiler = Compiler(Session())
        program = parse(Source("Coordinates := [x, y]\ng_{ab} := diag(0, 1)"))
        assert compiler.compile(program) == []  # not refused: nothing is computed
        (check,) = compiler.take_checks()
        assert check.node.op.label() == "metric"
        assert check.span.text == "g_{ab} := diag(0, 1)"
        assert compiler.take_checks() == []

    def test_a_degenerate_metric_is_refused_where_it_is_defined(self, wb):
        wb.exe("Coordinates := [x, y]\ng_{ab} := diag(1, x^2)")
        with pytest.raises(EvaluationError, match="degenerate") as raised:
            wb.exe("g_{ab} := diag(0, 1)\nR")
        assert raised.value.span.text == "g_{ab} := diag(0, 1)"
        assert raised.value.notes == ["no term of it has dx"]
        assert wb.expr("g_{yy}") == sympy.Symbol("x") ** 2  # the metric before it

    def test_a_definition_whose_check_fails_is_undone(self, wb):
        wb.exe(r"ds^2 = -\Delta dt^2 + dr^2 + r^2 d\Omega^2")
        steps = list(wb.each(r"\Delta = 0"))
        assert "degenerate" in steps[0].error.message
        assert wb.expr(r"\Delta") == sympy.Symbol("Delta")
        assert wb._compiler.take_notes() == []  # nor said to have been bound in the metric

    def test_the_ir_of_a_metric_that_is_none_is_shown(self, wb):
        listing = wb.ir("Coordinates := [x, y]\ng_{ab} := diag(0, 1)\ng_{xx}")
        assert "metric(" in listing
        assert wb.coordinates is None  # nothing was defined
