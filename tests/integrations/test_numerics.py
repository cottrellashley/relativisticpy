"""Data has the values exact numbers have, on the whole domain of a function: a polynomial
of a whole degree is evaluated with an integer degree, a value is complex where exact's is,
and what exact refuses is refused. Each program is one the review of the engines found
wrong, by the number of its finding."""

import mpmath
import numpy as np
import pytest
import sympy

from relativisticpy import Workbook, integrations
from relativisticpy.errors import EvaluationError

PAIR = r"\mathcal{V} := \mathbb{R}^{2}" "\n" r"l \in \mathcal{V}"
COMPLEX_PAIR = r"\mathcal{V} := \mathbb{C}^{2}" "\n" r"l \in \mathcal{V}"
MAP = r"\mathcal{V} := \mathbb{R}^{2}" "\n" r"a, b \in \mathcal{V}"


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


def spherical(cylinder, n, x):
    """A spherical Bessel function through the Bessel function of order n + 1/2."""
    with mpmath.workdps(40):
        x = mpmath.mpf(str(x))
        return complex(mpmath.sqrt(mpmath.pi / (2 * x)) * cylinder(n + mpmath.mpf(1) / 2, x))


class TestHighDegrees:
    """5, beyond the data the review tried: a closed form of decimals was summed in their
    own fifteen digits by SymPy, and a polynomial of a complex number, or of a degree that
    is no whole number, as a series in double precision by SciPy, and each lost its digits
    as the degree grew."""

    @pytest.mark.parametrize(
        ("program", "value"),
        [
            ("laguerre(40, 25.0)", lambda: mpmath.laguerre(40, 0, 25)),  # was 60672
            ("L_{40}(25.0)", lambda: mpmath.laguerre(40, 0, 25)),
            ("assoc_laguerre(100, 1.5, 25.0)", lambda: mpmath.laguerre(100, 1.5, 25)),
            ("chebyshevt(100, 0.9)", lambda: mpmath.chebyt(100, mpmath.mpf("0.9"))),  # 3.5e18
            ("chebyshevu(100, 0.9)", lambda: mpmath.chebyu(100, mpmath.mpf("0.9"))),
            ("legendre(100, 0.9)", lambda: mpmath.legendre(100, mpmath.mpf("0.9"))),
            ("hermite(100, 2.5)", lambda: mpmath.hermite(100, 2.5)),
            ("assoc_legendre(40, 2, 0.3)", lambda: mpmath.legenp(40, 2, mpmath.mpf("0.3"))),
            ("Ynm(40, 1, 0.3, 0.7)",
             lambda: mpmath.spherharm(40, 1, mpmath.mpf("0.3"), mpmath.mpf("0.7"))),
            ("jn(40, 0.9)", lambda: spherical(mpmath.besselj, 40, 0.9)),  # was 7.1e44
            ("yn(40, 0.9)", lambda: spherical(mpmath.bessely, 40, 0.9)),
            ("besselj(81/2, 0.9)", lambda: mpmath.besselj(40.5, mpmath.mpf("0.9"))),
        ],
    )  # fmt: skip
    def test_a_closed_form_of_a_decimal(self, program, value):
        with mpmath.workdps(40):
            expected = complex(value())
        assert exactly(program) == pytest.approx(expected, rel=1e-12)

    def test_a_closed_form_of_symbols_is_as_it_was(self):
        x = sympy.Symbol("x")
        with Workbook() as wb:
            assert wb.expr("chebyshevt(3, x)") == 4 * x**3 - 3 * x
            assert wb.expr("laguerre(2, 1/2)") == sympy.Rational(1, 8)
            assert float(wb.expr("hermite(2.0, 1.5)")) == 7.0

    @pytest.mark.parametrize(
        ("name", "computed"),
        [
            ("legendre", mpmath.legendre),
            ("chebyshevt", mpmath.chebyt),  # T_40(0.3 + 0i) was 356.8
            ("chebyshevu", mpmath.chebyu),
            ("hermite", mpmath.hermite),
            ("laguerre", lambda n, z: mpmath.laguerre(n, 0, z)),
        ],
    )
    def test_a_polynomial_of_complex_data(self, name, computed):
        z = [0.3 + 0.0j, 0.5 + 0.2j, 3.0 - 4.0j]
        found = of_data(
            f"{name}(40, z^l)", z, r"\mathcal{V} := \mathbb{C}^{3}" "\n" r"l \in \mathcal{V}", "z^l"
        )
        with mpmath.workdps(40):
            expected = [complex(computed(40, mpmath.mpc(v))) for v in z]
        assert np.allclose(found, expected, rtol=1e-11, atol=0)

    @pytest.mark.parametrize(
        ("name", "computed"),
        [
            ("legendre", mpmath.legendre),
            ("chebyshevt", mpmath.chebyt),
            ("chebyshevu", mpmath.chebyu),
        ],
    )
    def test_a_function_of_a_degree_that_is_no_whole_number(self, name, computed):
        found = of_data(f"{name}(41/2, x^l)", [0.3, -0.9])  # U_20.5(-0.9) was -979037
        with mpmath.workdps(40):
            expected = [
                complex(computed(mpmath.mpf("20.5"), mpmath.mpf(v))) for v in ("0.3", "-0.9")
            ]
        assert np.allclose(found, expected, rtol=1e-11, atol=0)

    def test_a_polynomial_at_a_root_of_it_is_zero(self):
        # where mpmath finds no digits left of its sum, which cancels altogether
        z = [2.5 + 0.0j, 1.0 + 1.0j]
        found = of_data("assoc_laguerre(1, 3/2, z^l)", z, COMPLEX_PAIR, "z^l")
        assert np.allclose(found, [0, 1.5 - 1j])


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


class TestHypergeometric:
    """21: the hypergeometric function lifts to data, and the lifting check tries it."""

    def test_of_data(self):
        found = of_data(r"{}_2F_1(1, 1; 2; x^l)", [0.3, 0.5])
        assert np.allclose(found, [-np.log(1 - v) / v for v in (0.3, 0.5)], rtol=1e-12)
        assert np.allclose(
            of_data(r"hyper([1], [2], x^l)", [0.3, 0.5]), np.expm1([0.3, 0.5]) / [0.3, 0.5]
        )

    def test_of_complex_numbers(self):
        # 6: SciPy's value of a complex z was cast to a real number
        z = [0.5 + 1.0j, 2.0 - 3.0j]
        found = of_data(r"{}_2F_1(1, 1; 2; z^l)", z, COMPLEX_PAIR, "z^l")
        assert np.allclose(found, [complex(mpmath.hyp2f1(1, 1, 2, v)) for v in z], rtol=1e-12)

    def test_the_lifting_check_tries_it_on_data(self):
        class Wrong:
            exact = integrations.find("hyper").implementation.exact

            def numeric(self, xp, a, b, z):
                return xp.zeros_like(z)

        wrong = integrations.function(
            "wronghyper", summary="the hypergeometric function, wrong of data", examples=[],
            points=integrations.find("hyper").points,
        )(Wrong)  # fmt: skip
        assert integrations.check(integrations.find("hyper"), ["lifting"]) == {"lifting": []}
        (failure,) = integrations.check(wrong, ["lifting"])["lifting"]
        assert failure.startswith("wronghyper applied to data, as its argument 3, gives [0, 0, 0]")


class TestMatrices:
    def test_an_eigenvalue_of_data_is_counted_as_exact_counts_it(self):
        # 10: of data, lambda_0 was the largest, lambda_{3/2} the first, lambda_3 an IndexError
        data = np.array([[2.0, 1.0], [1.0, 3.0]])
        for k in ["0", "-1", "3/2", "3"]:
            of_data = refused(f"eigenvalue({k}, A)", MAP, data, "A^{a}{}_{b}")
            assert of_data == refused(f"eigenvalue({k}, [[2, 1], [1, 3]])")
            assert of_data.endswith(f"and no \\lambda_{{{k}}}")
        with Workbook() as wb:
            wb.exe(MAP)
            wb["A^{a}{}_{b}"] = data
            assert np.isclose(wb.expr("eigenvalue(1, A)"), (5 - np.sqrt(5)) / 2)
            assert np.isclose(wb.expr("eigenvalue(2.0, A)"), (5 + np.sqrt(5)) / 2)

    def test_the_logarithm_of_a_singular_matrix_is_refused(self):
        # 15: of data it was -46.05 times the identity, and of numbers a failure inside (27)
        said = "a matrix with the eigenvalue 0 has no logarithm"
        assert said in refused("logm(A)", MAP, np.zeros((2, 2)), "A^{a}{}_{b}")
        assert said in refused("logm(A)", MAP, np.array([[1.0, 2.0], [2.0, 4.0]]), "A^{a}{}_{b}")
        assert said in refused("logm(A)", MAP + "\nA^{a}{}_{b} := [[0, 0], [0, 0]]")
        assert said in refused("logm([[0, 0], [0, 0]])")
        assert said in refused("logm([[1, 2], [2, 4]])")
        assert said in refused("logm([[1.0, 2.0], [2.0, 4.0]])")
        with Workbook() as wb:  # and a matrix that has one keeps it
            assert wb.expr("logm([[1, 0], [0, e^2]])") == sympy.Array([[0, 0], [0, 2]])


class TestRefusals:
    @pytest.mark.parametrize("digits", ["-5", "0", "1/2", "2.5", "x"])
    def test_numerical_evaluation_to_fewer_than_one_digit(self, digits):
        # 15: N(pi, -5) was 4.0
        said = f"N gives a number to a whole number of significant digits from 1, not {digits}"
        assert refused(f"N(pi, {digits})") == said

    def test_numerical_evaluation_to_digits(self):
        with Workbook() as wb:
            assert str(wb.expr("N(pi, 5)")) == "3.1416"
            assert str(wb.expr("N(pi, 5.0)")) == "3.1416"
