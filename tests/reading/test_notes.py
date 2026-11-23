r"""What is read silently is said: the notes the reader gives where it takes the less likely
reading."""

import pytest

from relativisticpy.errors import CompileError, IndexNotationError

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


class TestSilentReadingsAreSaid:
    def test_greek_superscripts_on_a_name_that_means_nothing(self, wb):
        assert notes(wb, r"T^{\mu\nu}") == [
            r"T is not defined, so T^{\mu\nu} is read as T to a power; to use a tensor, define "
            r"it first: T^{\mu\nu} := [...]"
        ]
        assert notes(wb, r"\frac{d x^\mu}{d\tau}")[0].startswith(
            r"x is not defined, so x^\mu is read as x to a power"
        )
        assert notes(wb, "x^n y^n") == []  # a Latin exponent is a power, as written

    def test_a_letter_followed_by_brackets_is_a_function_and_says_so(self, wb):
        assert notes(wb, r"\gamma(t - v x)") == [
            r"\gamma is not defined, so \gamma(t - v x) is an unknown function applied to "
            r"t - v x; for a product, write \gamma \cdot (t - v x)"
        ]
        assert str(wb.expr(r"\gamma \cdot (t - v x)")) == "gamma*(t - v*x)"  # the hint works
        assert notes(wb, r"\alpha(t, r)") == []  # a function of names, as meant
        wb.exe("f(x) := x^2")
        assert notes(wb, "f(x + 1)") == []  # a function defined

    def test_a_function_nothing_defines_at_a_number_is_said_to_be_left_as_written(self, wb):
        # Black-Scholes, as Hull writes it: N(d_1) at the number d_1 is no value, and a
        # result that holds it said nothing (design/notation.md, 26)
        wb.exe(r"d_1 := \frac{1}{2}")
        assert notes(wb, r"42 \cdot N(d_1)") == [
            "N is a function nothing defines, so it is left as written where it is applied "
            "to a number; define it first, as in N(x) := ..."
        ]
        assert notes(wb, r"42 \cdot N(x)") == []  # a function of x, as meant

    def test_a_label_written_up_and_down_on_a_name_that_means_nothing(self, wb):
        assert notes(wb, "u^a u_a = -1") == [
            "u is not defined, so u^a is a power and u_a one symbol; to use a tensor, define "
            "it first: u^{a} := [...]"
        ]
        assert notes(wb, r"u_\alpha u_\alpha = -1")[0].startswith(
            r"u is not defined, so u_\alpha is one symbol, not a component"
        )

    def test_a_derivative_with_an_index_and_no_coordinates(self, wb):
        assert notes(wb, r"\nabla_\mu \nabla^\mu \phi = 0") == [
            r"there are no coordinates, so a derivative with the index \mu is one with respect "
            r"to a variable named \mu; declare coordinates to make \mu an index"
        ]

    def test_an_undefined_name_added_to_a_tensor(self, wb):
        numpy = pytest.importorskip("numpy")
        wb["x_t"] = numpy.ones(3)
        found = error(wb, "x_t + y_t", IndexNotationError)
        assert found.message == "the free indices of a sum must agree: _{t} and none"
        assert found.notes == ["y is not defined, so y_t is read as one symbol, y_t"]
