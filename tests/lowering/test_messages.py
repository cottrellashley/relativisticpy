r"""What the lowerer says of what it cannot lower as written: the arguments a function
takes, what builtins take, derivatives of functions, and hints that work as written."""

import pytest

from relativisticpy.errors import CompileError, EvaluationError, ParseError

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


class TestArity:
    def test_a_spacetime_names_its_parameters(self, wb):
        assert error(wb, r"g_{\mu\nu} := Schwarzschild(M, G, c, d)").message == (
            "Schwarzschild(M, G, c) takes at most 3 arguments, not 4"
        )

    def test_a_parameter_left_out_is_the_symbol_the_table_names_it_with(self, wb):
        # as docs/tensors.md says: Kerr(M) is Kerr(M, a), not a guess at a value of a
        assert wb.expr("Kerr(M)") == wb.expr("Kerr(M, a)")
        wb.exe(r"g_{\mu\nu} := Kerr(M)")
        assert sorted(map(str, wb.expr(r"g_{t\phi}").free_symbols)) == ["M", "a", "r", "theta"]

    def test_one_argument_is_singular(self, wb):
        assert error(wb, "solve()").message == "solve() takes at least 1 argument, not 0"
        assert error(wb, "log(1, 2, 3)").message == "log() takes 1 or 2 arguments, not 3"

    @pytest.mark.parametrize("source", ["eig()", "eig(1, 2, 3)", "spec(x)"])
    def test_linear_algebra_takes_a_tensor(self, wb, source):
        assert "takes a tensor written with its indices" in error(wb, source).message

    def test_an_argument_by_name(self, wb):
        found = error(wb, "series(sin(x), x, 0, n=5)", ParseError)  # refused as it is read
        assert found.message == "n=5 is a keyword argument, which the language does not have"
        assert found.hint == "give the value in its place, as in series(sin(x), x, 0, 5)"
        assert wb.expr("series(sin(x), x, 0, 5)")  # the hint works


class TestBuiltinsSayWhatTheyTake:
    def test_rhs_of_what_is_no_equation(self, wb):
        assert error(wb, "RHS(x)", EvaluationError).message == (
            "RHS takes an equation, as in RHS(x^2 = 4), but x is not one"
        )

    def test_subs_takes_rules(self, wb):
        x = __import__("sympy").Symbol("x")
        assert wb.expr("subs(x^2, x = 3)") == 9
        assert wb.expr("subs(x^2, x, 3)") == 9
        found = error(wb, "subs(x^2, x)", EvaluationError).message
        assert "subs takes an expression and what to put for what" in found
        assert wb.expr("symbols(x)") == x
        assert "symbols takes names in quotes" in error(wb, "symbols(2)", EvaluationError).message

    def test_a_list_is_not_indexed(self, wb):
        found = error(wb, "[1, 2][0]", ParseError)  # refused as it is read
        assert found.message == "[1, 2][0] indexes as Python does, which the language does not"
        assert found.hint.startswith("a component is written with indices, as in V^{0}")
        wb.exe("a, b := [1, 2]")  # the hint works
        assert wb.expr("a") == 1
        wb.exe(SCHWARZSCHILD)
        assert (
            error(wb, r"geodesic_equations(\lambda)[1]").message == "a list is not indexed with [ ]"
        )

    def test_geodesics_are_integrated_from_python(self, wb):
        wb.exe(SCHWARZSCHILD)
        found = error(wb, "geodesic(g, [0, 10, pi/2, 0])")
        assert found.message == (
            "geodesics are integrated from Python, with relativisticpy.numeric.geodesic"
        )
        assert len(wb.expr(r"geodesic_equations(\tau)")) == 4  # the hint works

    @pytest.mark.parametrize(
        ("source", "message"),
        [
            (r"\det([1, 2])", "a determinant needs a square matrix, but this is a list of 2"),
            (
                r"\det([[1, 2, 3], [4, 5, 6]])",
                "a determinant needs a square matrix, but this is a 2×3 matrix",
            ),
            (
                r"\mathrm{inv}([[1, 2, 3], [4, 5, 6]])",
                "an inverse needs a square matrix, but this is a 2×3 matrix",
            ),
            (
                r"\mathrm{inv}([[1, 2], [2, 4]])",
                "this matrix has no inverse: its determinant is zero",
            ),
        ],
    )
    def test_a_matrix_that_has_no_determinant_or_inverse(self, wb, source, message):
        assert error(wb, source, EvaluationError).message == message
        assert wb.expr(r"\det([[1, 2], [3, 4]])") == -2

    def test_dsolve_solves_for_a_function(self, wb):
        assert error(wb, "dsolve(y'' + y, y)", EvaluationError).message == (
            "dsolve solves for a function written with its variable, "
            "as in dsolve(f''(x) + f(x) = 0, f(x)), not for y"
        )
        found = wb.expr("dsolve(f''(x) + f(x) = 0, f(x))")  # the example works
        assert "sin(x)" in str(found)
        assert "cos(x)" in str(found)


class TestDerivativesOfFunctions:
    def test_a_function_written_alone_is_differentiated_at_its_argument(self, wb):
        x = __import__("sympy").Symbol("x")
        wb.exe("f(x) := x^3")
        assert wb.expr(r"\frac{d^2 f}{dx^2}") == 6 * x
        assert wb.expr(r"\frac{d^2 f(x)}{dx^2}") == 6 * x
        assert wb.expr("diff(f(x), x, 2)") == 6 * x

    def test_a_value_written_with_what_it_depends_on_is_that_value(self, wb):
        x = __import__("sympy").Symbol("x")
        wb.exe("f := x^3")
        assert wb.expr(r"\frac{d^2 f(x)}{dx^2}") == 6 * x  # f(x) is f, not f times x

    def test_a_name_that_means_nothing_is_constant_and_says_so(self, wb):
        assert wb.expr(r"\frac{d^2 f}{dx^2}") == 0
        assert notes(wb, r"\frac{d^2 f}{dx^2}") == [
            "f does not depend on x, so its derivative is 0; for a function, write f(x)"
        ]


class TestHintsThatWork:
    def test_a_sum_over_an_index_without_limits_is_a_contraction(self, wb):
        numpy = pytest.importorskip("numpy")
        wb["w^a"] = numpy.array([1.0, 2.0])
        found = error(wb, r"\sum_a w^a")
        assert found.message == r"\sum needs limits"
        assert found.hint == (
            "to sum the components of w^a, contract it with the covector of ones: w^a 1_a"
        )
        assert wb.expr("w^a 1_a") == 3.0  # the hint works

    def test_solving_a_trace_for_the_ricci_scalar(self, wb):
        wb.exe(r"EFE := G_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G T_{\mu\nu}")
        assert str(wb.expr("solve(trace(EFE), R)")) == r"R = 4 \Lambda - 8 \pi G T"
