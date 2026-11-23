"""Data has the values exact numbers have, on the whole domain of a function: a polynomial
of a whole degree is evaluated with an integer degree, a value is complex where exact's is,
and what exact refuses is refused. Each program is one the review of the engines found
wrong, by the number of its finding."""

import mpmath
import numpy as np
import pytest
import sympy

from relativisticpy import Workbook

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
