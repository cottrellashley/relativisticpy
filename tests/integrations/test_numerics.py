"""Data has the values exact numbers have, on the whole domain of a function: a polynomial
of a whole degree is evaluated with an integer degree, a value is complex where exact's is,
and what exact refuses is refused. Each program is one the review of the engines found
wrong, by the number of its finding."""

import mpmath
import numpy as np
import pytest
import sympy

from relativisticpy import Workbook
from relativisticpy.errors import EvaluationError

PAIR = r"\mathcal{V} := \mathbb{R}^{2}" "\n" r"l \in \mathcal{V}"
COMPLEX_PAIR = r"\mathcal{V} := \mathbb{C}^{2}" "\n" r"l \in \mathcal{V}"


def of_data(program, values, setup=PAIR, name="x^l"):
    """The value of a program of data bound to ``name``, as an array."""
    with Workbook() as wb:
        wb.exe(setup)
        wb[name] = np.asarray(values)
        return np.asarray(wb.expr(program).to_numpy())


def exactly(program):
    """The value of a program, as a complex number."""
    with Workbook() as wb:
        return complex(sympy.N(wb.expr(program), 30))


def refused(program, setup="", data=None, name="x^l"):
    """The message a program is refused with, of data bound to ``name`` where it is given."""
    with Workbook() as wb:
        if setup:
            wb.exe(setup)
        if data is not None:
            wb[name] = np.asarray(data)
        with pytest.raises(EvaluationError) as error:
            wb.expr(program)
    return error.value.message


class TestDegrees:
    """5 and 7: a whole degree is an integer, and other degrees are what exact makes them."""

    def test_a_laguerre_polynomial_of_a_high_degree(self):
        found = of_data(r"L_{40}(x^l)", [25.0, 60.0])
        expected = [float(mpmath.laguerre(40, 0, v)) for v in (25, 60)]
        assert np.allclose(found, expected, rtol=1e-9)

    def test_a_generalised_laguerre_polynomial_of_a_high_degree(self):
        found = of_data(r"assoc_laguerre(100, 3/2, x^l)", [25.0, 60.0])
        expected = [float(mpmath.laguerre(100, 1.5, v)) for v in (25, 60)]
        assert np.allclose(found, expected, rtol=1e-9)

    def test_a_laguerre_polynomial_of_a_negative_degree(self):
        found = of_data(r"laguerre(-2, x^l)", [0.3, 1.5])
        assert np.allclose(found, [np.exp(v) * (1 + v) for v in (0.3, 1.5)], rtol=1e-12)

    def test_a_spherical_bessel_function_of_half_an_order(self):
        found = of_data(r"jn(1/2, x^l)", [1.5, 3.0])
        expected = [
            float(mpmath.sqrt(mpmath.pi / (2 * v)) * mpmath.besselj(1, v)) for v in (1.5, 3)
        ]
        assert np.allclose(found, expected, rtol=1e-12)

    def test_a_spherical_bessel_function_of_a_negative_order(self):
        assert np.allclose(of_data(r"jn(-1, x^l)", [1.5, 3.0]), np.cos([1.5, 3]) / [1.5, 3])
        assert np.allclose(of_data(r"yn(-1, x^l)", [1.5, 3.0]), np.sin([1.5, 3]) / [1.5, 3])

    def test_a_hermite_function_of_half_a_degree(self):
        found = of_data(r"hermite(1/2, x^l)", [1.5, 3.0])
        assert np.allclose(found, [float(mpmath.hermite(0.5, v)) for v in (1.5, 3)], rtol=1e-12)
        assert np.isclose(found[0], exactly("N(hermite(1/2, 3/2))"), rtol=1e-12)

    @pytest.mark.parametrize(
        ("program", "said"),
        [
            ("hermite(-1, x^l)", "hermite takes a degree that is a real number from 0, not -1"),
            ("laguerre(1/2, x^l)", "laguerre takes a degree that is a whole number, not 1/2"),
            ("assoc_laguerre(1/2, 1, x^l)", "assoc_laguerre takes a degree that is a whole"),
            ("Ynm(1/2, 0, x^l, 1)", "Ynm takes a degree that is a whole number, not 1/2"),
        ],
    )
    def test_a_degree_exact_refuses_is_refused_of_data(self, program, said):
        assert said in refused(program, PAIR, [0.3, 0.7])
        assert said in refused(program.replace("x^l", "3/10"))

    def test_an_order_beyond_its_degree_is_refused_by_both(self):
        said = "Ynm is of a degree l from 0 and an order m from -l to l, not l = 2, m = 3"
        assert refused("Ynm(2, 3, x^l, 1)", PAIR, [0.3, 0.7]) == said
        assert refused("Ynm(2, 3, 3/10, 1)") == said

    def test_spherical_harmonics_with_scipy_before_1_15(self, monkeypatch):
        # which names the function sph_harm, with the order first and the azimuth second
        import scipy.special

        newer = scipy.special.sph_harm_y
        older = lambda m, n, azimuth, polar: newer(n, m, polar, azimuth)  # noqa: E731
        monkeypatch.delattr(scipy.special, "sph_harm_y")
        monkeypatch.setattr(scipy.special, "sph_harm", older, raising=False)
        found = of_data("Ynm(2, 1, x^l, 1/2)", [0.3, 1.2])
        assert np.allclose(found, newer(2, 1, np.array([0.3, 1.2]), 0.5))


class TestComplexValues:
    """6: past the real domain, data is complex where exact is, never nan or a real part."""

    @pytest.mark.parametrize(
        ("program", "computed"),
        [
            (r"W_0(x^l)", mpmath.lambertw),
            (r"\operatorname{Ci}(x^l)", mpmath.ci),
            (r"\operatorname{Log}(x^l)", mpmath.log),
            (r"\ln(x^l)", mpmath.log),
            (r"\sqrt{x^l}", mpmath.sqrt),
            (r"besselj(1/2, x^l)", lambda v: mpmath.besselj(0.5, v)),
            (r"Y_0(x^l)", lambda v: mpmath.bessely(0, v)),
            (r"besseli(1/2, x^l)", lambda v: mpmath.besseli(0.5, v)),
            (r"K_0(x^l)", lambda v: mpmath.besselk(0, v)),
        ],
    )
    def test_of_negative_numbers(self, program, computed):
        found = of_data(program, [-1.0, -2.5])
        assert np.allclose(found, [complex(computed(v)) for v in (-1, -2.5)], rtol=1e-9)

    @pytest.mark.parametrize(
        "program",
        [r"\arcsin(x^l)", r"\arccos(x^l)", r"\operatorname{K}(x^l)", r"\operatorname{E}(x^l)"],
    )
    def test_past_one(self, program):
        found = of_data(program, [1.5, 2.5])
        expected = [exactly(program.replace("x^l", value)) for value in ("3/2", "5/2")]
        assert np.allclose(found, expected, rtol=1e-9)
        assert all(value.imag != 0 for value in expected)

    @pytest.mark.parametrize(
        ("program", "exact"),
        [
            (r"\mathrm{B}(z^l, 2)", lambda z: mpmath.beta(z, 2)),
            (r"\operatorname{K}(z^l)", mpmath.ellipk),
            (r"\lfloor z^l \rfloor", lambda z: complex(np.floor(z.real), np.floor(z.imag))),
        ],
    )
    def test_complex_data_a_library_takes_no_complex_numbers_for(self, program, exact):
        # 27: SciPy's beta and elliptic integral, and NumPy's floor, take no complex numbers
        z = [0.5 + 1.0j, 2.0 - 3.0j]
        found = of_data(program, z, COMPLEX_PAIR, "z^l")
        assert np.allclose(found, [complex(exact(v)) for v in z], rtol=1e-12)

    def test_data_in_the_real_domain_stays_real(self):
        found = of_data(r"\sqrt{x^l} + J_0(x^l)", [1.5, 2.5])
        assert found.dtype.kind == "f"
