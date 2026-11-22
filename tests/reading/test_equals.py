r"""What ``=`` means: the contract of design/interface.md.

Papers use ``=`` both to define and to claim, so RelativisticPy decides by what the left
side already means. ``:=`` always defines; ``=`` defines only what has no meaning yet, and
is otherwise a claim, checked, whose result carries a note saying so.
"""

import pytest
import sympy
from sympy import Eq, Function, Symbol, symbols

from relativisticpy import Workbook
from relativisticpy.abstract import IndexEquation
from relativisticpy.errors import CompileError, EvaluationError

x, y, r, t, M, G, p, q = symbols("x y r t M G p q")
SCHWARZSCHILD = "Coordinates := [t, r, \\theta, \\phi]\ng_{\\mu\\nu} := Schwarzschild(M)"


def step(wb, source):
    """The one step of a one-statement program."""
    (found,) = list(wb.each(source))
    assert found.error is None, found.error
    return found


class TestEqualsDefinesWhatHasNoMeaningYet:
    def test_a_new_name(self, wb):
        assert wb.exe("x = 2") == []
        assert wb.expr("x + 1") == 3

    def test_a_new_function(self, wb):
        assert wb.exe("f(x) = x^2") == []
        assert wb.expr("f(3)") == 9

    def test_a_new_tensor(self, wb):
        wb.exe("Coordinates := [t, x]")
        assert wb.exe("u^\\mu = [1, 0]") == []
        assert wb.expr("u^\\mu").components == sympy.Array([1, 0])

    def test_a_line_element_even_written_as_python_does(self, wb):
        assert wb.exe("ds**2 = -dt**2 + dx**2") == []
        assert wb.expr("g_{tt}") == -1
        assert wb.expr("R") == 0  # the Ricci scalar, not a symbol: a metric is in force

    def test_the_metric_while_there_is_none(self, wb):
        wb.exe("Coordinates := [t, x]")
        assert wb.exe("g_{\\mu\\nu} = diag(-1, 1)") == []
        assert wb.expr("g_{xx}") == 1


class TestEqualsComparesWhatHasAMeaning:
    def test_a_name_with_a_value_is_compared_not_redefined(self, wb):
        wb.exe(SCHWARZSCHILD + "\nK := R_{abcd} R^{abcd}")
        claim = step(wb, "K = \\frac{48 M^2}{r^5}")
        assert claim.values[0] is not sympy.true
        assert claim.notes == [
            "K already has a value, so = compares it; to give K a new value, write K := ..."
        ]
        assert wb.expr("K") == 48 * M**2 / r**6  # unchanged

    def test_colon_equals_redefines(self, wb):
        wb.exe("x := 2\nx := 3")
        assert wb.expr("x") == 3

    def test_a_tensor_with_a_value(self, wb):
        wb.exe("Coordinates := [t, x]\nu^\\mu = [1, 0]")
        claim = step(wb, "u^\\mu = [2, 0]")
        assert not claim.values[0]
        assert claim.notes[0].startswith("u^\\mu already has a value, so = compares it")
        assert wb.expr("u^\\mu").components == sympy.Array([1, 0])

    def test_a_function_with_a_value(self, wb):
        wb.exe("f(x) := x^2")
        claim = step(wb, "f(x) = x^3")
        assert claim.values[0] is not sympy.true
        assert "to give f a new value, write f(x) := ..." in claim.notes[0]
        assert wb.expr("f(2)") == 4

    def test_a_built_in_scalar(self, wb):
        wb.exe(SCHWARZSCHILD)
        claim = step(wb, "R = 0")
        assert claim.values == [sympy.true]
        assert claim.notes == [
            "R is the Ricci scalar, so = compares it; to give R a meaning of your own, "
            "write R := ..."
        ]

    def test_the_ricci_scalar_before_a_metric_is_notation(self, wb):
        # R = 0 neither defines R nor is lost: after a metric, R is the Ricci scalar
        claim = step(wb, "R = 0")
        assert isinstance(claim.values[0], IndexEquation)
        assert "R is the Ricci scalar" in claim.notes[0]
        wb.exe("ds^2 = a^2 (d\\theta^2 + \\sin^2\\theta d\\phi^2)")
        assert wb.expr("R") == 2 / Symbol("a") ** 2

    def test_a_constant(self, wb):
        claim = step(wb, "\\pi = 3")
        assert claim.values == [sympy.false]
        assert claim.notes[0].startswith("\\pi is a constant, so = compares it")

    def test_the_metric_in_force(self, wb):
        wb.exe(SCHWARZSCHILD)
        claim = step(wb, "g_{\\mu\\nu} = diag(-1, 1, r^2, r^2 \\sin^2\\theta)")
        assert not claim.values[0]
        assert "g_{\\mu\\nu} is the metric in force" in claim.notes[0]
        assert wb.expr("Kretschmann") == 48 * M**2 / r**6

    def test_a_component_is_a_claim_without_a_note(self, wb):
        wb.exe(SCHWARZSCHILD)
        claim = step(wb, "g_{tt} = -(1 - 2M/r)")
        assert claim.values == [sympy.true]
        assert claim.notes == []


class TestANameInItsOwnValue:
    def test_is_an_equation(self, wb):
        wb.exe("ds^2 = R^2 (d\\theta^2 + \\sin^2\\theta d\\phi^2)")
        claim = step(wb, "R = \\frac{2}{R^2}")
        assert claim.values == [Eq(Symbol("R"), 2 / Symbol("R") ** 2)]
        assert "R appears on both sides" in claim.notes[0]
        assert wb.expr("R") == Symbol("R")  # still the radius

    def test_through_the_value_of_another_name(self, wb):
        rho, w = symbols("rho w")
        wb.exe("p = w \\rho")
        claim = step(wb, "\\rho = -p")
        assert claim.values == [Eq(rho, -w * rho)]
        assert wb.expr("\\rho") == rho

    def test_an_inverse_transformation_is_checked(self, wb):
        # Carroll's x' = x^{1/3}, then x = (x')^3: a claim that holds
        wb.exe("x' = x^{1/3}")
        assert wb.expr("x = (x')^3") is sympy.true

    def test_a_decorated_name_is_another_name(self, wb):
        assert wb.exe("h = -\\bar h") == []  # \bar h is h̄, neither h nor ℏ
        assert wb.expr("h") == -Symbol("h\N{COMBINING MACRON}")

    def test_a_function_that_calls_itself(self, wb):
        with pytest.raises(CompileError, match="f calls itself"):
            wb.exe("f(x) := f(x - 1) + 1")
        claim = step(wb, "g(x) = x g(x - 1)")
        assert "g appears on both sides" in claim.notes[0]

    def test_a_tensor_written_with_its_own_components_is_defined(self, wb):
        # k^\mu = (\omega, 0, 0, k^3): k^3 is a component or a power, not the tensor
        wb.exe("Coordinates := [t, x, y, z]")
        assert wb.exe("k^\\mu = (\\omega, 0, 0, k^3)") == []
        assert wb.expr("k^{t}") == Symbol("omega")

    def test_a_symmetry_is_a_claim(self, wb):
        assert isinstance(wb.expr("T_{\\mu\\nu} = T_{\\nu\\mu}"), IndexEquation)


class TestFunctions:
    def test_a_value_at_a_point_is_a_claim(self, wb):
        assert wb.expr("f(2) = 4") == Eq(Function("f")(2), 4)

    def test_a_coordinate_along_a_path_is_never_defined(self, wb):
        wb.exe(SCHWARZSCHILD)
        tau = Symbol("tau")
        assert wb.expr("r(\\tau) = 2\\tau") == Eq(Function("r")(tau), 2 * tau)
        assert wb.expr("r") == r

    def test_a_definition_that_means_nothing_as_a_function_is_an_equation(self, wb):
        # f(W) = W_{a} W^{a}: as a function of a symbol W, its value cannot be computed
        wb.exe("Coordinates := [t, x]\nL_\\mu := [1, 0]")
        found = next(iter(wb.each("V(r) = L^2 + r")))
        assert found.error is not None  # L^2, a tensor squared, means nothing either way
        assert "is read as an equation, not as the definition of a function" in str(found.error)
        assert wb.exe("V = 2") == []  # so V was not defined


class TestLateBinding:
    """Where \\Delta = ...: a name the metric is written with, defined after it."""

    def test_a_symbol(self, wb):
        wb.exe("ds^2 = -\\Delta dt^2 + \\Delta^{-1} dr^2 + r^2 d\\Omega^2")
        assert wb.exe("\\Delta = 1 - \\frac{2M}{r}") == []
        notes = wb._compiler.take_notes()
        assert notes == [
            "the metric defined on line 1 is written with \\Delta, which has this value "
            "there from now on"
        ]
        assert wb.expr("R") == 0
        assert wb.expr("R_{abcd} R^{abcd}") == 48 * M**2 / r**6

    def test_a_function(self, wb):
        wb.exe("ds^2 = -f(r) dt^2 + \\frac{dr^2}{f(r)} + r^2 d\\Omega^2")
        wb.exe("f(r) := 1 - \\frac{2M}{r}")
        assert wb.expr("R_{abcd} R^{abcd}") == 48 * M**2 / r**6

    def test_a_function_of_the_coordinates_named_alone(self, wb):
        wb.exe("ds^2 = -e^{2\\alpha(t,r)} dt^2 + dr^2 + r^2 d\\Omega^2")
        wb.exe("\\alpha = \\ln r")
        assert wb.expr("g_{tt}") == -(r**2)

    def test_what_was_computed_before_stays(self, wb):
        wb.exe("ds^2 = -\\Delta dt^2 + \\Delta^{-1} dr^2 + r^2 d\\Omega^2")
        before = wb.expr("R")
        wb.exe("\\Delta = 1 - \\frac{2M}{r}")
        assert before.has(Symbol("Delta"))
        assert wb.expr("R") == 0

    def test_carroll_reissner_nordstrom_as_written(self, wb):
        # (7.110) then (7.111): G is a factor in 2GM, so G(p^2 + q^2) is a product
        wb.exe(r"ds^2 = -\Delta dt^2 + \Delta^{-1} dr^2 + r^2 d\Omega^2")
        wb.exe(r"\Delta = 1-{{2GM}\over r}+{{G(p^2+q^2)}\over{r^2}}")
        assert wb.expr("R") == 0  # an electrovacuum: traceless
        assert wb.expr("g_{tt}") == -(1 - 2 * G * M / r + G * (p**2 + q**2) / r**2)

    def test_a_value_that_makes_the_metric_degenerate_is_refused(self, wb):
        wb.exe("ds^2 = -\\Delta dt^2 + dr^2 + r^2 d\\Omega^2")
        with pytest.raises(EvaluationError, match="the metric is degenerate"):
            wb.exe("\\Delta = 0")
        assert wb.expr("\\Delta") == Symbol("Delta")  # not defined either


class TestFunctionOrProduct:
    def test_a_name_also_written_as_a_factor(self, wb):
        assert wb.expr("\\frac{2GM}{r} + G(p^2 + q^2)") == 2 * G * M / r + G * (p**2 + q**2)

    def test_a_declared_constant(self, wb):
        wb.exe("Constants := [G]")
        constant = Symbol("G", constant=True)
        assert wb.expr("G(p^2 + q^2)") == constant * (p**2 + q**2)

    def test_e_and_i(self, wb):
        assert wb.expr("e(x+1)") == sympy.E * (x + 1)
        assert wb.expr("i(x+1)") == sympy.I * (x + 1)

    def test_a_name_written_as_a_factor_before(self, wb):
        rho, w = symbols("rho w")
        wb.exe("p = w \\rho")
        assert wb.expr("\\rho(1 + 3w)") == rho * (1 + 3 * w)

    def test_a_function_stays_one(self, wb):
        assert wb.expr("f(f(x))") == Function("f")(Function("f")(x))
        wb.exe("ds^2 = -e^{2\\alpha(t,r)} dt^2 + dr^2")
        assert wb.expr("\\alpha + \\alpha(t, r)") == 2 * Function("alpha")(t, r)

    def test_a_value_written_with_what_it_depends_on(self, wb):
        wb.exe("a := t^2")
        assert wb.expr("\\frac{d}{dt} a(t)") == 2 * t


class TestReading:
    def test_in_latex_an_inline_slash_takes_what_follows_it(self, wb):
        alpha = Symbol("alpha")
        assert wb.expr("\\alpha x/4GM") == alpha * x / (4 * G * M)
        assert wb.expr("\\exp(-r/2GM)") == sympy.exp(-r / (2 * G * M))
        assert wb.expr("\\frac{x}{4}GM") == x * G * M / 4
        assert wb.expr("\\alpha x/4 GM") == alpha * x * G * M / 4  # with a space, as written

    def test_a_null_path_is_a_condition(self, wb):
        wb.exe(SCHWARZSCHILD)
        with pytest.raises(CompileError, match="says that a path is null"):
            wb.exe("ds^2 = 0")
        assert wb.expr("Kretschmann") == 48 * M**2 / r**6  # the metric is as it was

    def test_an_equation_in_brackets_is_a_condition(self, wb):
        with pytest.raises(CompileError, match=r"\(k=0\), in brackets, is a condition"):
            wb.exe("(k=0)")
        assert wb.expr("k") == Symbol("k")

    def test_ds_squared_alone_is_the_line_element(self, wb):
        wb.exe("ds^2 = -dt^2 + r^2 d\\phi^2")
        dt, dphi = Symbol("dt"), Symbol("dphi")
        assert wb.expr("ds^2") == -(dt**2) + r**2 * dphi**2


class TestDefinitionsAreWhole:
    def test_a_failed_line_element_keeps_the_coordinates(self, wb):
        wb.exe("ds^2 = -dt^2 + dx^2")
        with pytest.raises(CompileError):
            wb.exe("ds^2 = -dt^2 + a^2 d\\psi^2 + b(c) d")
        assert [c.name for c in wb.coordinates] == ["t", "x"]
        assert wb.expr("g_{xx}") == 1

    def test_a_two_sphere_after_three_dimensions_says_how_to_declare_it(self, wb):
        wb.exe("ds^2 = dr^2 + r^2 d\\theta^2 + r^2 \\sin^2\\theta d\\phi^2")
        with pytest.raises(EvaluationError, match="degenerate") as error:
            wb.exe("ds^2 = d\\theta^2 + \\sin^2\\theta d\\phi^2")
        assert "written in \\theta and \\phi alone" in str(error.value)
        hint = "Coordinates := [\\theta, \\phi]"
        assert hint in str(error.value)
        wb.exe(hint + "\nds^2 = d\\theta^2 + \\sin^2\\theta d\\phi^2")  # the hint works
        assert wb.expr("R") == 2


def test_checking_a_definition_by_equals_leaves_the_session_as_it_was():
    with Workbook() as wb:
        wb.exe("ds^2 = -dt^2 + dx^2")
        wb.exe("V(x) = m(t) x")  # m(t) would make a bare m mean m(t) from then on
        assert "m" not in wb._session.functions
        assert wb.expr("V(2)") == 2 * Function("m")(t)
