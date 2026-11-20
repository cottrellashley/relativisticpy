"""Functions: one-line definitions, block functions and lexical scope (ported)."""

import pytest


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("f() := x**2\nf()", "x**2"),
        ("h(x, y) := x**2 - y*8 + y*x\nh(v**2, 7)", "v**4 + 7*v**2 - 56"),
        ("f(x) := 3*x**3 - 2*x**2 + x - 5\nf(2)", "13"),
        ("g(x) := e**x - ln(x)\ng(1) + exp(ln(x) + 10)", "x*exp(10) + E"),
        ("g(x) := x\ng(1) + exp(ln(x) + 10)", "x*exp(10) + 1"),
        ("t(x) := sin(x)**2 + cos(x)**2\nt(pi/4)", "1"),
        (
            "abs_func(x) := |x - 3|\nabs_func_two(x) := |-x**2|\n"
            "|-x + abs_func(-1) + abs_func_two(10)|",
            "Abs(x - 104)",
        ),
        ("piecewise(x) := { x**2, x > 0; -x, x <= 0 }\npiecewise(2) * piecewise(-2)", "8"),
        (
            "f(x) := x**2 + 2*x - 1\ng(x) := 2*sin(x)\nh(x) := f(g(x))\nh(pi/2)",
            "7",
        ),
        ("r(x) := (x**2 - 4) / (x - 2)\nr(3)", "5"),
        (
            "factorial_func(n) := n!\n"
            "comb(n, k) := n! / (k! * (n - k)!)\n"
            "comb(factorial_func(3), 4)",
            "15",
        ),
        ("y(x) = 10", "Eq(y(x), 10)"),
        ("y(x) = 10 * x |_{x = 10}", "Eq(y(10), 100)"),
    ],
)
def test_one_line_functions(wb, source, expected):
    assert str(wb.expr(source)) == expected


def test_differential_equation_in_a_function(wb):
    source = """
        f : (x) -> {
            diff(y(x), x, 2) + 2*diff(y(x), x) + y(x) = 10
        }
        dsolve(f(x), y(x))
    """
    assert str(wb.expr(source)) == "Eq(y(x), (C1 + C2*x)*exp(-x) + 10)"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("f : (x) -> {\n x^2\n}\nf(3)", "9"),
        ("f : (x, y) -> {\n x*y\n}\nf(3, 4)", "12"),
        ("f : () -> {\n 42\n}\nf()", "42"),
        ("f : (x) -> {\n g : (y) -> {\n y^2\n }\n x + g(x)\n}\nf(3)", "12"),
        (
            "f : (x) -> {\n m := x^2\n s := m^2\n g : (y) -> {\n n := y^2\n n^2\n }\n"
            " s + g(2)\n}\nf(2)",
            "32",
        ),
    ],
)
def test_block_functions(wb, source, expected):
    assert str(wb.expr(source)) == expected


def test_a_variable_defined_in_a_function_stays_inside(wb):
    assert str(wb.expr("f : (x) -> {\n y := x + 1\n y\n}\nf(2) + y")) == "y + 3"


def test_a_variable_of_an_outer_function_stays_inside_it(wb):
    source = "f : (x) -> {\n y := x + 1\n g : (z) -> {\n z + y\n }\n g(2)\n}\nf(2) + y"
    assert str(wb.expr(source)) == "y + 5"


def test_a_function_defined_in_a_function_stays_inside(wb):
    source = "f : (x) -> {\n g : (z) -> {\n z + x\n }\n g(2)\n}\nf(2) + g(2)"
    assert str(wb.expr(source)) == "g(2) + 4"


def test_functions_see_the_scope_they_were_defined_in(wb):
    assert wb.expr("x := 10\nf : (y) -> { x + y }\nf(5)") == 15
    assert wb.expr("x := 10\ng : (x) -> { x^2 }\ng(5)") == 25
