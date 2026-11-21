r"""Less common notation and the errors that guard it."""

import pytest
import sympy
from sympy import Array, Function, Symbol, symbols

from relativisticpy import Tensor, TensorEquation
from relativisticpy.errors import CompileError, IndexNotationError

x, y, t, r, theta = symbols("x y t r theta")


@pytest.fixture
def plane(wb):
    wb.exe("Coordinates := [x, y]\ng_{ab} := [[1, 0], [0, x^2]]\nV^{a} := [y, x]")
    return wb


def test_logic_and_chained_comparisons(wb):
    assert wb.expr("x > 0 and x < 1 or y") == sympy.Or(sympy.And(x > 0, x < 1), y)
    assert wb.expr("not x > 0") == sympy.Not(x > 0)
    assert wb.expr("0 < x <= 1") == sympy.And(sympy.Lt(0, x), sympy.Le(x, 1))
    assert wb.expr("+x") == x


def test_coordinates_by_name(plane):
    assert plane.expr("Coordinates") == [x, y]


def test_coordinates_from_a_column_or_a_single_name(wb):
    wb.exe(r"Coordinates := \begin{pmatrix} t \\ r \end{pmatrix}")
    assert wb.coordinates == (t, r)
    wb.exe("Coordinates := t")
    assert wb.coordinates == (t,)
    with pytest.raises(CompileError, match="lists names"):
        wb.expr("Coordinates := 2 + t")


def test_destructuring_needs_names(wb):
    with pytest.raises(CompileError, match="only names can be defined together"):
        wb.expr("a, b^2 := symbols('a b')")


def test_cannot_define_a_power_of_a_call(wb):
    with pytest.raises(CompileError, match="cannot define this"):
        wb.expr("f(x)^{a} := 2")


def test_definition_index_rules(plane):
    with pytest.raises(IndexNotationError, match="must differ"):
        plane.expr("T_{aa} := [[1, 0], [0, 1]]")
    with pytest.raises(IndexNotationError, match="the metric has two indices"):
        plane.expr("g_{abc} := 0")


def test_metric_from_a_tensor(wb):
    wb.exe("Coordinates := [x, y]\neta_{ab} := [[-1, 0], [0, 1]]\ng_{ab} := eta_{ab}")
    assert wb.expr("g^{ab}").components == Array([[-1, 0], [0, 1]])
    with pytest.raises(IndexNotationError, match="expected a tensor with 2 indices, not 1"):
        wb.expr("w_{a} := [1, 2]\ng_{ab} := w_{a}")


def test_a_metric_from_a_tensor_takes_it_as_written(wb):
    wb.exe("Coordinates := [x, y]\nh_{ab} := [[-1, 0], [0, 4]]")
    with pytest.raises(IndexNotationError, match=r"must agree: \^\{a\}\^\{b\} and _\{a\}_\{b\}"):
        wb.exe("g^{ab} := h_{ab}")  # h has its indices down
    with pytest.raises(IndexNotationError, match="must agree"):
        wb.exe(r"g_{\mu\nu} := h_{ab}")
    wb.exe("g_{ab} := h_{ab}")
    assert wb.expr("g^{ab}").components == Array([[-1, 0], [0, sympy.Rational(1, 4)]])


def test_literal_components_are_scalars(plane):
    with pytest.raises(CompileError, match="a component must be a scalar"):
        plane.expr("T_{a} := [V^{b}, 0]")
    with pytest.raises(CompileError, match="expected 2 components"):
        plane.expr("T_{a} := [[1, 2], [3, 4]]")


def test_components_from_a_value(plane):
    plane.exe("m := diag(1, 2)\nT_{ab} := m")
    assert plane.expr("T_{ab}").components == Array([[1, 0], [0, 2]])
    plane.exe("u := [1, 2]\nW_{a} := u")
    assert plane.expr("W_{a}").components == Array([1, 2])
    plane.exe(r"P_{a} := \begin{pmatrix} 1 & 2 \end{pmatrix}")
    assert plane.expr("P_{a}").components == Array([1, 2])


def test_builtins_that_take_tensors(plane):
    assert plane.expr("det(g_{ab})") == x**2
    assert plane.expr("trace(g_{ab})") == 2  # g^{ab} g_{ab}, not the diagonal's sum 1 + x^2
    assert plane.expr("trace([[1, 2], [3, x]])") == 1 + x  # a plain matrix's is its diagonal's
    assert plane.expr("inverse(g_{ab})") == Array([[1, 0], [0, 1 / x**2]])
    assert plane.expr("transpose(V_{a} V_{b})") == plane.expr("V_{b} V_{a}").components
    assert plane.expr("latex(V^{a})") == r"T^{a} = \left[\begin{matrix}y & x\end{matrix}\right]"
    assert plane.expr("det(5)") == 5
    with pytest.raises(CompileError, match="cannot pass a tensor to f"):
        plane.expr("f(V^{a})")


def test_the_trace_of_a_tensor_is_taken_with_the_metric(wb):
    wb.exe(r"g_{\mu\nu} := FLRW(k)")
    assert wb.expr(r"trace(R_{\mu\nu}) = R") is sympy.true
    assert wb.expr(r"trace(R^{\mu}{}_{\nu}) = R") is sympy.true
    assert wb.expr(r"trace(R^{\mu\nu}) = R") is sympy.true


def test_componentwise_builtins(plane):
    result = plane.expr("diff(V^{a} V_{a}, x)")
    assert result == sympy.diff(y**2 + x**4, x)
    assert plane.expr("subs(V^{a}, x, 2)").components == Array([y, 2])
    assert plane.expr("N(V^{a} |_{x=1, y=2})").components == Array([2.0, 1.0])


def test_equation_builtins(wb):
    assert wb.expr("RHS(dsolve(diff(f(x), x) = f(x), f(x)))") == Symbol("C1") * sympy.exp(x)
    assert wb.expr("LHS(x = 2)") == x
    assert wb.expr("RHS([x = 2, y = 3])") == [2, 3]
    assert wb.expr("solve([x + y = 2, x - y = 0])") == {x: 1, y: 1}


def test_solving_a_tensor_equation(wb):
    wb.exe("Coordinates := [x, y]\nV^{a} := [x - 1, y + 2]")
    assert wb.expr("solve(V^{a} = 0, [x, y])") == {x: 1, y: -2}
    assert wb.expr("solve([V^{a} = 0], [x, y])") == {x: 1, y: -2}


def test_doit_and_numerical_values(wb):
    assert wb.expr(r"doit(\sum_{k=1}^{4} k)") == 10
    assert wb.expr("doit(3)") == 3
    assert wb.expr("N(pi, 5)") == sympy.N(sympy.pi, 5)
    assert wb.expr("root(8, 3)") == 2


def test_spacetime_values(wb):
    from relativisticpy import Metric

    assert isinstance(wb.expr("Schwarzschild(M)"), Metric)


@pytest.mark.parametrize(
    ("name", "function"),
    [("Schwarzschild", "schwarzschild"), ("Kerr", "kerr"), ("FLRW", "flrw")],
)
def test_a_spacetime_takes_the_arguments_its_function_does(wb, name, function):
    import inspect

    from relativisticpy import spacetimes

    parameters = inspect.signature(getattr(spacetimes, function)).parameters.values()
    count = sum(1 for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))
    assert wb.expr(f"{name}({', '.join(['1'] * count)})") is not None
    with pytest.raises(CompileError, match=rf"{name}\(\) takes 0 to {count} arguments"):
        wb.expr(f"{name}({', '.join(['1'] * (count + 1))})")


def test_the_double_factorial(wb):
    assert wb.expr("factorial2(5)") == 15
    assert wb.expr("factorial2(6)") == 48


def test_primes(wb):
    with pytest.raises(CompileError, match="takes one argument"):
        wb.expr("f'(x, y)")
    with pytest.raises(CompileError, match="y is not a function"):
        wb.expr("y := 2\ny'(x)")
    with pytest.raises(CompileError, match="needs an argument"):
        wb.expr("f''")


def test_geodesic_equations_arguments(wb):
    wb.exe(r"g_{\mu\nu} := Minkowski()")
    assert len(wb.expr("geodesic_equations()")) == 4
    with pytest.raises(CompileError, match="at most one argument"):
        wb.expr("geodesic_equations(s, u)")


def test_derivatives_with_respect_to_vectors(plane):
    # V^a = (y, x), so the gradient of xy with respect to it is (x, y), a covector
    gradient = plane.expr(r"\frac{\partial}{\partial V^{a}} (x y)")
    assert gradient.components == Array([x, y])
    assert str(gradient.indices) == "_{a}"
    with pytest.raises(IndexNotationError, match="one first derivative"):
        plane.expr(r"\frac{\partial^2 (x y)}{\partial V^{a} \partial x}")
    with pytest.raises(IndexNotationError, match="only a scalar"):
        plane.expr(r"\frac{\partial V^{b}}{\partial V^{a}}")


def test_derivatives_of_tensors(plane):
    assert plane.expr(r"\frac{d}{dx} V^{a}").components == Array([0, 1])
    assert plane.expr("diff(V^{a}, y)").components == Array([1, 0])


def test_raised_derivative_indices(plane):
    assert plane.expr(r"\partial^{a} (x y)").components == Array([y, 1 / x])
    assert plane.expr("V_{a}{}^{,b}").rank == 2


def test_integrals_limits_and_substitutions_of_tensors(plane):
    assert plane.expr(r"\int_0^1 V^{a} \, dx").components == Array([y, sympy.Rational(1, 2)])
    assert plane.expr(r"\lim_{x \to 0} V^{a}").components == Array([y, 0])
    assert plane.expr(r"V^{a} |_{x=3}").components == Array([y, 3])
    with pytest.raises(CompileError, match="give both limits"):
        plane.expr(r"\int_0 x \, dx")


def test_a_symmetrisation_is_over_indices_at_one_level(plane):
    plane.exe("T^{a}{}_{bc} := [[[1, 2], [3, 4]], [[5, 6], [7, x]]]")
    assert plane.expr(r"T_{(a}{}^{b}{}_{c)} = \frac{1}{2}(T_{a}{}^{b}{}_{c} + T_{c}{}^{b}{}_{a})")
    with pytest.raises(IndexNotationError, match="opens on lower indices and closes on upper"):
        plane.expr(r"V_{(a} V^{b)}")


def test_sums_and_products_of_tensors(plane):
    # component by component, as integrals are, and kept as written until doit
    assert isinstance(plane.expr(r"\sum_{n=0}^{3} n V^{a}"), Tensor)
    assert plane.expr(r"doit(\sum_{n=0}^{3} n V^{a})").components == Array([6 * y, 6 * x])
    assert plane.expr(r"doit(\prod_{n=1}^{2} V^{a})").components == Array([y**2, x**2])


def test_integration_variable_must_be_a_name(wb):
    with pytest.raises(CompileError, match="expected a variable"):
        wb.expr(r"\int x \, d{x^2}")


def test_time_derivatives_of_tensors(wb):
    wb.exe(r"Coordinates := [t, x]" "\n" r"q^{a} := [f(t), g(t)]")
    result = wb.expr(r"\dot{q}^{a}")
    assert result.components == Array([Function("f")(t).diff(t), Function("g")(t).diff(t)])
    assert wb.expr(r"\ddot{q^{a}}").rank == 1


def test_accents_need_symbols(wb):
    with pytest.raises(CompileError, match=r"\\hat needs a single symbol"):
        wb.expr(r"\hat{x + y}")


def test_subscripts_need_names(wb):
    with pytest.raises(CompileError, match="only a name can have a subscript"):
        wb.expr("(x + y)_{0}")
    assert wb.expr("x_{i+1}") == Symbol("x_i+1")


def test_symmetrising_needs_free_indices(plane):
    with pytest.raises(IndexNotationError, match="only free indices"):
        plane.expr("V^{(a} V^{b)} V_{b}")
    with pytest.raises(IndexNotationError, match="never closed"):
        plane.expr("V^{(a} V^{b}")


def test_an_operator_rejects_derivative_markers(plane):
    with pytest.raises(IndexNotationError, match="unexpected derivative index"):
        plane.expr(r"\partial_{a,b} x")


def test_bare_tensor_names_and_parameters(plane):
    assert plane.expr("V").components == Array([y, x])
    assert plane.expr("f(W) := W_{a} W^{a}\nf(V_{b})") == plane.expr("V_{a} V^{a}")


def test_index_errors_from_the_parser_surface(plane):
    with pytest.raises(Exception, match="only two or more indices"):
        plane.expr("V^{(a)}")


class TestTensorEquations:
    @pytest.fixture
    def equation(self, plane):
        return plane.expr("V^{a} = 0")

    def test_components_and_equations(self, equation):
        assert isinstance(equation, TensorEquation)
        assert equation.rank == 1
        assert [str(e) for e in equation.equations()] == ["Eq(y, 0)", "Eq(x, 0)"]
        assert not equation

    def test_display(self, equation):
        assert str(equation) == "x: Eq(y, 0)\ny: Eq(x, 0)"
        assert repr(equation) == "TensorEquation(^{a}, 2 equations)"
        assert equation.latex() == r"\begin{aligned}^{x}: &\; y = 0 \\ ^{y}: &\; x = 0\end{aligned}"
        assert equation._repr_latex_().startswith("$")

    def test_comparison(self, plane, equation):
        assert equation == plane.expr("V^{a} = 0")
        assert (equation == True) is False  # noqa: E712
        assert equation.__eq__(1) is NotImplemented

    def test_substitution(self, plane):
        solved = plane.expr("V^{a} = 0 |_{x=0, y=0}")
        assert solved == True  # noqa: E712
        assert str(solved) == "True"
        assert solved.latex() == r"\mathrm{True}"


def test_tensor_values(plane):
    tensor = plane.expr("V^{a}")
    assert list(tensor) == [y, x]
    assert len(Tensor(5, "")) == 1
    assert Tensor(5, "").dim == 0
    assert tensor.latex() == r"V^{a} = \left[\begin{matrix}y & x\end{matrix}\right]"
