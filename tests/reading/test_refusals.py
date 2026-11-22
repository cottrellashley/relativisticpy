r"""What the reader refuses rather than misreads: the guards of wave 0 of
``curriculum/ROADMAP.md``. A wrong answer given confidently is the worst thing the language
can do, so where a form would be read with another meaning, it is refused, saying what was
read, why, and what to write instead; and the hint works."""

import pytest
import sympy

from relativisticpy.errors import CompileError

x, theta = sympy.symbols("x theta")


def error(wb, source, kind=CompileError):
    with pytest.raises(kind) as raised:
        wb.exe(source)
    return raised.value


def decisions(wb, source, kind):
    """The decisions of ``kind`` taken in reading the last statement of ``source``."""
    reading = wb.explain(source)[-1]
    return [(d.written, d.reading, d.why) for d in reading.decisions if d.kind == kind]


class TestColonEqualsDefinesOrRefuses:
    """Guard 1: ``:=`` defines, or is refused saying how its left side was read; it never
    makes an equation, as ``\\psi_n(x) := ...`` made Eq(psi_n x, ...) and defined nothing."""

    def test_a_subscripted_function_is_refused_with_its_family(self, wb):
        found = error(wb, r"\psi_n(x) := \sin(n x)")
        assert found.message == r"cannot define \psi_n(x), which is read as \psi_n times x"
        assert found.notes[0] == (
            r"a name with a subscript or a superscript, as \psi_n, is not yet the name of a "
            "function, nor of a family of them"
        )
        assert found.hint == (
            r"define a function of n too, as in \psi(n, x) := ..., and write \psi(1, x) for "
            r"\psi_1(x)"
        )
        assert wb.expr(r"\psi(n, x) := \sin(n x)" "\n" r"\psi(1, x)") == sympy.sin(x)

    def test_a_function_with_a_superscript(self, wb):
        found = error(wb, r"Y_2^1(\theta, \phi) := \sin\theta")
        assert found.message == (
            r"cannot define Y_2^1(\theta, \phi), which is read as Y_2^1 times (\theta, \phi)"
        )
        hint = r"give the function a name without scripts, as in Y(\theta, \phi) := ..."
        assert found.hint == hint
        assert wb.expr(r"Y(\theta, \phi) := \sin\theta" "\n" r"Y(\theta, 0)") == sympy.sin(theta)

    def test_a_product_of_names(self, wb):
        found = error(wb, r"\Delta\phi := 2")
        assert found.message == (
            r"cannot define \Delta\phi, which is read as the product \Delta times \phi"
        )
        assert found.notes == [
            ":= gives a value to a name, a tensor or a function, not to a product"
        ]
        assert found.hint == r"to name one quantity, write one name, as in \Delta_{\phi} := ..."
        assert wb.expr(r"\Delta_{\phi} := 2" "\n" r"3 \Delta_{\phi}") == 6

    def test_a_product_of_subscripted_names(self, wb):
        found = error(wb, r"\omega_e x_e := 52.8")
        assert found.message == (
            r"cannot define \omega_e x_e, which is read as the product \omega_e times x_e"
        )
        assert found.hint == (
            "to name one quantity, give it one name, and write that name where it is used"
        )

    def test_a_time_derivative(self, wb):
        found = error(wb, r"\dot{x} := x(\alpha - \beta y)")
        assert found.message == (
            r"cannot define \dot{x}: it is the time derivative of x, which takes its value "
            "from x"
        )
        assert found.hint.startswith("define x as a function of time, as in x(t) := ...")

    def test_a_function_of_what_is_not_a_name(self, wb):
        found = error(wb, r"H(e^{j\omega}) := 1 + 2e^{-j\omega}")
        assert found.message == (
            r"cannot define H(e^{j\omega}): the parameters of a function are names, and "
            r"e^{j\omega} is not"
        )
        assert found.hint == (
            r"define it with names as its parameters, as in H(z) := ..., and then write "
            r"H(e^{j\omega})"
        )
        wb.exe("H(z) := 1 + 2/z")
        assert wb.expr(r"H(e^{i\omega})") == 1 + 2 * sympy.exp(-sympy.I * sympy.Symbol("omega"))

    def test_nothing_is_defined_and_the_reading_says_how_it_was_read(self, wb):
        source = r"\psi_n(x) := \sin(n x)"
        assert decisions(wb, source, "product or call") == [
            (
                r"\psi_n(x)",
                r"\psi_n times x",
                r"\psi_n is not a function: a name with scripts is a symbol",
            )
        ]
        assert wb.explain(source)[0].error is not None
        assert next(iter(wb.each(source))).error is not None
        assert wb.names() == {}

    def test_equiv_still_states_an_identity(self, wb):
        # Carroll's chart map, \phi_1(x^1, x^2, x^3) \equiv (y^1, y^2): a relation, not refused
        (step,) = list(wb.each(r"\Delta\phi \equiv \phi_2 - \phi_1"))
        assert step.error is None

    def test_a_function_written_as_books_write_one_is_defined(self, wb):
        # Bracewell's sinc x := sin(πx)/(πx): the argument after the name, without brackets
        wb.exe(r"\operatorname{sinc} x := \frac{\sin \pi x}{\pi x}")
        assert wb.expr(r"\operatorname{sinc}\left(\frac{1}{2}\right)") == 2 / sympy.pi
        assert decisions(wb, r"\mathrm{sinc}\, x := 1", "product or call") == [
            (
                r"\mathrm{sinc}\, x",
                "sinc(x)",
                "a name written as an operator, with one name after it, is a function of it",
            )
        ]

    def test_an_operator_name_before_more_than_one_name_is_refused(self, wb):
        found = error(wb, r"\operatorname{sinc} x y := 2")
        assert found.message.startswith(r"cannot define \operatorname{sinc} x y")


class TestEqualsDefinesAFunctionOrRefuses:
    """Guard 1: ``=`` whose left side is a name with arguments defines that function, or is
    refused; as a product of a symbol and its argument, U_{eff}(r) = ... was an equation,
    and the minimum of U_{eff} was [], from dU_{eff}/dr = 0."""

    def test_a_subscripted_name_with_arguments(self, wb):
        found = error(wb, r"U_{\text{eff}}(r) = -\frac{k}{r} + \frac{\ell^2}{2 \mu r^2}")
        assert found.message == (
            r"U_{\text{eff}}(r) is read as U_{\text{eff}} times r, so = neither defines nor "
            "compares it"
        )
        assert found.hint == "give the function a name without scripts, as in U(r) := ..."
        wb.exe(r"U(r) := -\frac{k}{r} + \frac{\ell^2}{2 \mu r^2}")
        k, ell, mu = sympy.symbols("k ell mu")
        assert wb.expr(r"solve\left(\frac{d U(r)}{d r} = 0, r\right)") == [ell**2 / (k * mu)]

    def test_a_function_that_is_defined_is_compared(self, wb):
        wb.exe("f(x) := x^2")
        assert next(iter(wb.each("f(x) = x^3"))).error is None
