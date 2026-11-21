"""Simplifying components: exact decimals and deciding when a component vanishes."""

from sympy import Float, Rational, symbols

from relativisticpy.tensors.simplification import is_zero, rationalise

x = symbols("x")


def test_decimals_are_the_fractions_they_denote():
    assert rationalise(Float("1.7") * x) == Rational(17, 10) * x
    assert rationalise(x ** Float("0.5")) == x ** Rational(1, 2)
    assert rationalise(Float(0.1 + 0.2)) == Rational(3, 10)  # to the digits a float carries
    assert rationalise(Float("0.30000000000000004")) == Rational(30000000000000004, 10**17)
    assert rationalise(Float("6.674e-11")) == Rational(6674, 10**14)
    assert rationalise(x + 1) == x + 1


def test_decimals_cannot_hide_a_zero():
    assert is_zero((Float("1.7") * x + 1) ** 2 - (Float("2.89") * x**2 + Float("3.4") * x + 1))
    assert is_zero(Float(0.0))
    assert not is_zero(Float("1.7") * x - Float("1.6") * x)
    assert not is_zero(Float("1e-30") * x)  # small is not zero
