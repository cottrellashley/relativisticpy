"""Simplifying components: exact decimals and deciding when a component vanishes."""

from sympy import Float, Integral, Rational, Symbol, cos, pi, sec, sin, symbols, tan

from relativisticpy.tensors.simplification import is_zero, rationalise

x = symbols("x")


def test_numbers_are_decided():
    assert is_zero(cos(1) ** 2 + sin(1) ** 2 - 1)
    assert is_zero(sec(1) ** 2 - tan(1) ** 2 - 1)
    assert not is_zero(cos(1) ** 2 + sin(1) ** 2)
    assert not is_zero(cos(1) - cos(2))


def test_fractions_of_an_angle_are_related():
    assert is_zero(sin(x / 2) ** 2 - (1 - cos(x)) / 2)
    assert is_zero(tan(x / 2) - sin(x) / (1 + cos(x)))
    assert is_zero(sin(x) - 3 * sin(x / 3) + 4 * sin(x / 3) ** 3)
    assert not is_zero(sin(x / 2) - sin(x) / 2)
    assert not is_zero(sin(x / 2) ** 2 - (1 + cos(x)) / 2)
    k = Symbol("k", integer=True)  # an integer k is not an integer once halved
    assert not is_zero(sin(pi * k / 2))
    assert not is_zero(sin(k / 2) ** 2 - (1 - cos(k)) / 2 + 1)
    # a variable an integral binds is not rescaled
    assert not is_zero(Integral(sin(x / 2), (x, 0, 1)) - Integral(sin(x) / 2, (x, 0, 1)))


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
