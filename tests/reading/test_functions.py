r"""Functions named as books name them (design/notation.md, decisions 5 to 8 and 11): a name
with scripts, then its parameters in brackets, defines and calls a function of its own;
f^{(4)} of a function of one variable is its derivative; a prime on the left of := is part
of the name; and a function named without its argument is the function at its
parameters. Each was refused, or read as a product, before."""

from __future__ import annotations

import pytest
import sympy

from relativisticpy import Workbook
from relativisticpy.errors import CompileError

r, t, x, theta, phi = sympy.symbols("r t x theta phi")


def readings(wb: Workbook, source: str, kind: str) -> list[str]:
    """How the last statement of ``source`` was read, in decisions of ``kind``."""
    return [d.text() for d in wb.explain(source)[-1].decisions if d.kind == kind]


class TestFunctionsWithScripts:
    @pytest.mark.parametrize(
        ("definition", "call", "expected"),
        [
            (r"R_{nl}(r) := r e^{-r}", r"R_{nl}(1)", sympy.exp(-1)),  # plain text: no command
            (r"\sigma^2(\mu) := \mu^2 + 1", r"\sigma^2(2)", 5),
            (r"C_\mathrm{A}(t) := \frac{1}{1 + t}", r"C_\mathrm{A}(1)", sympy.Rational(1, 2)),
            (r"v_\theta(r) := \frac{1}{r}", r"v_\theta(2)", sympy.Rational(1, 2)),
            (r"\bar{\mathbf{x}}(u) := 2 u", r"\bar{\mathbf{x}}(3)", 6),
        ],
    )
    def test_a_function_with_scripts_is_defined_and_called(self, wb, definition, call, expected):
        wb.exe(definition)
        assert wb.expr(call) == expected

    def test_a_definition_of_ones_own_wins_over_a_book_name(self, wb):
        harmonic = wb.expr(r"Y_2^1(\theta, \phi)")  # the spherical harmonic, by its book name
        assert harmonic.has(sympy.exp(sympy.I * phi))
        wb.exe(r"Y_2^1(\theta, \phi) := \sin\theta")
        assert wb.expr(r"Y_2^1(\theta, \phi)") == sympy.sin(theta)

    def test_the_reading_is_recorded(self, wb):
        assert readings(wb, r"R_{nl}(r) := r", "product or call") == [
            r"product or call: R_{nl}(r) is a function R_{nl}(r), since a name with scripts, "
            "then names in brackets, is a function of them"
        ]


class TestOrders:
    def test_an_order_on_a_function_is_its_derivative(self, wb):
        wb.exe(r"f(x) := \sec x")
        assert wb.expr(r"\frac{f^{(4)}(0)}{4!}") == sympy.Rational(5, 24)
        assert wb.expr(r"f^{(2)}") == sympy.diff(sympy.sec(x), x, 2)  # at its parameter

    def test_a_dotted_function_called(self, wb):
        v_0 = sympy.Symbol("v_0")
        wb.exe(r"x(t) := x_0 \cos \omega t + \frac{v_0}{\omega} \sin \omega t")
        assert wb.expr(r"\dot{x}(0)") == v_0
        assert wb.expr(r"\ddot{x} + \omega^2 x = 0") is sympy.true


class TestPrimes:
    def test_a_primed_function_is_defined_and_called(self, wb):
        wb.exe(r"H'(x) := 2x")
        assert wb.expr(r"H'(3)") == 6
        wb.exe("f'(x) := 1")  # plain text too
        assert wb.expr("f'(5)") == 1

    def test_a_prime_on_a_function_of_one_variable_is_its_derivative(self, wb):
        wb.exe("f(x) := x^3")
        assert wb.expr("f'(2)") == 12
        assert wb.expr("f''") == 6 * x  # at its parameter

    def test_a_primed_function_named_alone_in_a_derivative(self, wb):
        wb.exe(r"\rho'(x, t) := R e^{i(k x - \omega t)}")
        R, k, omega = sympy.symbols("R k omega")
        wave = R * sympy.exp(sympy.I * (k * x - omega * t))
        assert wb.expr(r"\frac{\partial \rho'}{\partial t}") == sympy.diff(wave, t)

    def test_a_primed_tensor(self, wb):
        wb.exe(
            "ds^2 = -dt^2 + dx^2 + dy^2 + dz^2\n"
            r"k^\mu := (E, E, 0, 0)"
            "\n"
            r"k'^\mu := (E', E' \cos\theta, E' \sin\theta, 0)"
            "\n"
            r"p^\mu := (m, 0, 0, 0)"
            "\n"
            r"p'^\mu := p^\mu + k^\mu - k'^\mu"
        )
        E, m = sympy.symbols("E m")
        (found,) = wb.expr(r"solve(p'_\mu p'^\mu = -m^2, E')")  # Compton's law
        assert sympy.simplify(found - E * m / (E * (1 - sympy.cos(theta)) + m)) == 0

    def test_a_primed_parameter(self, wb):
        wb.exe(r"r(t') := 2 t'")
        assert wb.expr("r(3)") == 6


class TestFunctionsNamedAlone:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("y(x) := x^3\ny + 1", x**3 + 1),
            ("u(t) := t^2\n3 u", 3 * t**2),
            ("a(t) := t^{2/3}\n" r"\frac{\dot a}{a}", 2 / (3 * t)),
            ("y_1(t) := e^{-2t}\ny_1'", -2 * sympy.exp(-2 * t)),
        ],
    )
    def test_a_function_named_alone_is_at_its_parameters(self, wb, source, expected):
        assert sympy.simplify(wb.exe(source)[-1] - expected) == 0

    def test_in_a_wronskian(self, wb):
        wb.exe("y_1(t) := e^{-2t}\ny_2(t) := e^{-3t}")
        found = wb.expr(r"\begin{vmatrix} y_1 & y_2 \\ y_1' & y_2' \end{vmatrix}")
        assert sympy.simplify(found + sympy.exp(-5 * t)) == 0

    def test_in_plain_text_a_function_applies_to_what_follows(self, wb):
        assert wb.expr("f(t) := t^2\nf 3") == 9

    def test_the_reading_is_recorded(self, wb):
        wb.exe("y(x) := x^3")
        assert readings(wb, r"y + 1", "function named alone") == [
            "function named alone: y is the function y(x), since y is a function, defined on line 1"
        ]

    def test_in_a_function_that_takes_none_of_its_parameters_it_is_refused(self, wb):
        # a Legendre transform means S eliminated; read as U(S, V), F would depend on S
        wb.exe("U(S, V) := C V^{-2/3} e^{2 S / (3 n R)}\nF(T, V) := U - T S")
        with pytest.raises(CompileError) as raised:
            wb.expr("F(T, V)")
        assert raised.value.message == (
            "U is named without its arguments in the value of F, so it would be U(S, V), and "
            "F takes no S"
        )
        assert (
            wb.expr("G(S, V) := U + 1\nG(1, 1)")
            == sympy.Symbol("C")
            * sympy.exp(sympy.Rational(2, 3) / (sympy.Symbol("n") * sympy.Symbol("R")))
            + 1
        )  # where the function takes them, it is the function at them

    def test_a_function_of_two_variables_needs_its_arguments_with_an_order(self, wb):
        wb.exe("g(x, y) := x^2 y^2")
        with pytest.raises(CompileError, match="read as a name of its own"):
            wb.expr("g^{(2)}(0)")


class TestArgumentsOfFunctions:
    def test_an_unbracketed_argument_stops_before_a_fraction(self, wb):
        # decision 11: \sin\theta \frac{a}{b} is sin(θ) a/b, not sin(θ a/b)
        a, b = sympy.symbols("a b")
        assert wb.expr(r"\sin\theta \frac{a}{b}") == sympy.sin(theta) * a / b
        assert wb.expr(r"\sin \frac{\pi}{2}") == 1
