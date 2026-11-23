r"""Transforms and rewrites: the Laplace, Fourier and z-transforms by their book names, with
their variables named by convention and the Fourier transform in the convention a program
chooses; the discrete Fourier transform of data, over a space of frequencies; the transform
written as its sum, computed as an FFT; and what the conformance suite says of a transform
or a rewrite that breaks the pattern."""

from __future__ import annotations

import numpy as np
import pytest
import sympy

from relativisticpy import Workbook, integrations
from relativisticpy.errors import CompileError
from relativisticpy.integrations import Example, _transforms, rewrite, transform
from relativisticpy.integrations.sequences import impulse, step

t, s, x, z, n, a, p = sympy.symbols("t s x z n a p")
omega, k = sympy.symbols("omega k")


def notes(wb: Workbook, source: str) -> list[str]:
    return [note for reading in wb.explain(source) for note in reading.notes]


class TestLaplace:
    @pytest.mark.parametrize(
        ("source", "value"),
        [
            (r"\mathcal{L}\{t^2\}", 2 / s**3),
            (r"\mathcal{L}\{t^2\}(p)", 2 / p**3),
            (r"\mathcal{L}\{\cos at\}", s / (s**2 + a**2)),
            (r"\mathcal{L}\{e^{-2t}\}(1)", sympy.Rational(1, 3)),
            (r"\mathcal{L}^{-1}\left\{\frac{1}{s^2}\right\}", t),
            (r"\mathcal{L}^{-1}\{1/s^2\}(t)", t),
            ("LaplaceTransform(sin(t))", 1 / (s**2 + 1)),
            ("LaplaceTransform(sin(t), t, p)", 1 / (p**2 + 1)),
        ],
    )
    def test_a_transform_written_as_books_write_it(self, wb, source, value):
        assert sympy.simplify(wb.expr(source) - value) == 0

    def test_the_theorems_of_the_transform_are_decided(self, wb):
        wb.exe(r"f(t) := \cos at")
        assert wb.expr(r"\mathcal{L}\{f'(t)\} = s \mathcal{L}\{f(t)\} - f(0)")
        assert wb.expr(r"\mathcal{L}\{f\} = \mathcal{L}\{f(t)\}")  # f, named alone, is f(t)

    def test_a_letter_nothing_defines_is_a_function_of_the_variable(self, wb):
        f = sympy.Function("f")
        assert wb.expr(r"\mathcal{L}\{f\}") == sympy.LaplaceTransform(f(t), t, s)
        assert notes(wb, r"\mathcal{L}\{f\}") == [
            r"f is not defined, so \mathcal{L}\{f\} is read as the transform of a function f(t)"
        ]

    def test_a_function_of_several_variables_none_conventional_is_refused(self, wb):
        with pytest.raises(CompileError, match="none of which is the variable") as refused:
            wb.expr(r"\mathcal{L}\{e^{-a u} \sin b\}")
        assert "LaplaceTransform(f, x, s)" in (refused.value.hint or "")

    def test_braces_make_the_transform_whatever_the_plain_letter_is(self, wb):
        wb.exe(r"\mathcal{L} := \frac{1}{2} m v^2 - V")  # a Lagrangian
        m, v = sympy.symbols("m v")
        assert wb.expr(r"\frac{\partial \mathcal{L}}{\partial v}") == m * v
        assert wb.expr(r"\mathcal{L}\{t^2\}") == 2 / s**3
        assert notes(wb, r"\mathcal{L}\{t^2\}") == [
            r"\mathcal{L}, with braces, is read as the Laplace transform, not as L, which the "
            "program defines: a typeface makes a name of its own"
        ]

    def test_brackets_are_no_braces(self, wb):
        assert wb.expr(r"\mathcal{L}(t)") == sympy.Function("L")(t)

    def test_a_transform_acts_on_a_function_not_a_tensor(self, wb):
        wb["x_t"] = np.array([1.0, 2.0])
        with pytest.raises(CompileError, match="acts on a function of a variable, not on a"):
            wb.expr(r"\mathcal{L}\{x_t\}")

    def test_a_transform_binds_its_variable(self, wb):
        # the review's programs: with t := 2, \mathcal{L}\{t^2\} was 4/s, the transform of 4
        wb.exe("t := 2")
        assert wb.expr(r"\mathcal{L}\{t^2\}") == 2 / s**3
        assert wb.expr("t") == 2  # the program's t is as it was
        fourier = Workbook()
        fourier.exe("x := 5")
        found = fourier.expr(r"\mathcal{F}\{e^{-x^2}\}")
        expected = sympy.sqrt(sympy.pi) * sympy.exp(-(sympy.pi**2) * s**2)
        assert sympy.simplify(found - expected) == 0

    def test_a_function_named_alone_is_at_its_own_variable(self, wb):
        wb.exe(r"f(t) := \cos at" "\n" "t := 3")
        assert sympy.simplify(wb.expr(r"\mathcal{L}\{f\}") - s / (s**2 + a**2)) == 0
        assert sympy.simplify(wb.expr(r"\mathcal{L}\{f(t)\}") - s / (s**2 + a**2)) == 0

    def test_a_transform_of_a_list_is_refused(self, wb):
        # the review's program: \mathcal{Z}\{x\} of a list was an internal error
        wb.exe("x := [1, 2, 3]")
        with pytest.raises(CompileError) as refused:
            wb.expr(r"\mathcal{Z}\{x\}")
        assert refused.value.message == (
            r"\mathcal{Z} acts on a function of a variable, not on a list"
        )
        assert refused.value.hint == (
            "for a sequence, write it with its index, as x[n] := ..., and transform that"
        )
        with pytest.raises(CompileError, match="not on a tuple"):
            wb.expr(r"\mathcal{L}\{(1, 2)\}")


class TestFourier:
    def test_the_letters_choose_the_variables_and_the_convention(self, wb):
        assert wb.expr(r"\mathcal{F}\{e^{-\pi x^2}\}") == sympy.exp(-sympy.pi * s**2)
        assert notes(wb, r"\mathcal{F}\{e^{-\pi x^2}\}") == [
            r"\mathcal{F} is read with e^{-2πi s x}, Bracewell's convention, as it takes x to "
            r"s; to choose another, define \mathcal{F} by its integral"
        ]
        found = wb.expr(r"\mathcal{F}\{e^{-t^2}\}(\omega)")
        assert sympy.simplify(found - sympy.sqrt(sympy.pi) * sympy.exp(-(omega**2) / 4)) == 0
        assert "the convention of angular frequency" in notes(wb, r"\mathcal{F}\{e^{-t^2}\}")[0]

    def test_a_transform_sympy_gives_for_positive_frequencies_alone(self, wb):
        found = wb.expr(r"\mathcal{F}\{\frac{1}{1 + x^2}\}")
        assert sympy.simplify(found - sympy.pi * sympy.exp(-2 * sympy.pi * sympy.Abs(s))) == 0
        assert found.subs(s, -1) == found.subs(s, 1)

    def test_a_program_declares_its_convention_by_the_integral(self, wb):
        wb.exe(
            r"\mathcal{F}\{f\}(\omega) := \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty} f(t)\, "
            r"e^{-i\omega t}\, dt"
        )
        assert wb.expr(r"\mathcal{F}\{e^{-t^2/2}\}(\omega)") == sympy.exp(-(omega**2) / 2)
        assert "as line 1 defines it" in notes(wb, r"\mathcal{F}\{e^{-t^2/2}\}(\omega)")[0]
        unitary = wb.expr(r"\mathcal{F}^{-1}\{e^{-\omega^2/2}\}(t)")
        assert sympy.simplify(unitary - sympy.exp(-(t**2) / 2)) == 0

    def test_an_integral_the_program_writes_shows_its_convention(self, wb):
        wb.exe(r"F(s) := \int_{-\infty}^{\infty} e^{-\pi x^2} e^{-i 2\pi x s} \, dx")
        found = wb.expr(r"\mathcal{F}\{e^{-t^2}\}(\omega)")
        expected = sympy.sqrt(sympy.pi) * sympy.exp(-(sympy.pi**2) * omega**2)
        assert sympy.simplify(found - expected) == 0
        assert (
            "as the integral of F, on line 1, writes it"
            in notes(wb, r"\mathcal{F}\{e^{-t^2}\}(\omega)")[0]
        )

    def test_bracewells_pair_is_a_claim_about_the_transform(self, wb):
        assert wb.expr(r"e^{-\pi x^2} \supset e^{-\pi s^2}")
        assert "Bracewell's convention" in notes(wb, r"e^{-\pi x^2} \supset e^{-\pi s^2}")[0]
        with pytest.raises(CompileError, match="'⊃' is read but not computed"):
            wb.expr(r"A \supset B")  # of no function a Fourier transform takes

    def test_only_a_fourier_transform_is_defined_by_its_integral(self, wb):
        with pytest.raises(CompileError, match="which has one definition"):
            wb.exe(r"\mathcal{L}\{f\}(s) := \int_0^\infty e^{-st} f(t)\, dt")
        with pytest.raises(CompileError, match="is no Fourier integral"):
            wb.exe(r"\mathcal{F}\{f\}(\omega) := \int_0^1 f(t)\, dt")


class TestZTransform:
    @pytest.mark.parametrize(
        ("source", "value"),
        [
            (r"\mathcal{Z}\{(1/2)^n\}(z)", z / (z - sympy.Rational(1, 2))),
            (r"\mathcal{Z}\{n\}", z / (z - 1) ** 2),
            (r"\mathcal{Z}\{\cos an\}", z * (z - sympy.cos(a)) / (z**2 - 2 * z * sympy.cos(a) + 1)),
            (
                r"\mathcal{Z}^{-1}\left\{\frac{z}{z - 1/2}\right\}",
                sympy.Rational(1, 2) ** n * step(n),
            ),
            (
                r"\mathcal{Z}^{-1}\left\{\frac{1}{\left(1 - \frac{1}{4}z^{-1}\right)"
                r"\left(1 - \frac{1}{2}z^{-1}\right)}\right\}",
                (2 * sympy.Rational(1, 2) ** n - sympy.Rational(1, 4) ** n) * step(n),
            ),
            (r"\mathcal{Z}^{-1}\left\{1 + \frac{2}{z}\right\}", impulse(n) + 2 * impulse(n - 1)),
            (r"\mathcal{Z}\{u[n - 2]\}(z)", 1 / (z * (z - 1))),
        ],
    )
    def test_the_transform_by_summation_and_its_inverse_by_partial_fractions(
        self, wb, source, value
    ):
        assert sympy.simplify(wb.expr(source) - value) == 0

    def test_the_inverse_of_what_is_no_rational_function_is_refused(self, wb):
        with pytest.raises(Exception, match="rational function of z"):
            wb.expr(r"\mathcal{Z}^{-1}\{e^{1/z}\}")


SIGNAL = np.array([1.0, 2.0, 0.0, -1.0, 3.0, 0.0, 0.0, 2.0])


class TestDiscreteFourierTransform:
    def test_data_is_transformed_over_the_frequencies_of_its_index(self, wb):
        wb["x_t"] = SIGNAL
        found = wb.expr(r"\operatorname{fft}(x_t)")
        assert np.allclose(found.to_numpy(), np.fft.fft(SIGNAL))
        (space,) = found.spaces
        assert space.name == "the frequencies of the space of t"
        assert space.basis == ("0", "1", "2", "3", "4", "5", "6", "7")  # the bins, as X_k

    def test_the_inverse_gives_data_over_the_samples_again(self, wb):
        wb["x_t"] = SIGNAL
        wb.exe(r"X_k := \operatorname{fft}(x_t)")
        back = wb.expr(r"\operatorname{ifft}(X_k)")
        assert np.allclose(back.to_numpy(), SIGNAL)
        assert back.spaces == wb["x"].spaces
        wb.exe(r"y_t := \operatorname{ifft}(X_k)")
        assert np.isclose(complex(wb.expr("y_t x_t")), SIGNAL @ SIGNAL)  # over one space

    def test_parseval(self, wb):
        wb["x_t"] = SIGNAL
        wb.exe(r"X_k := \operatorname{fft}(x_t)")
        energy = wb.expr(r"\frac{1}{8} \sum_k |X_k|^2")
        assert np.isclose(complex(energy), SIGNAL @ SIGNAL)

    def test_the_frequencies_of_a_grid_are_its_bins_as_x_k_is_written(self, wb):
        wb.exe(r"N := 64" "\n" r"t_n := \frac{n}{N}, \quad n = 0, \ldots, N-1")
        spectrum = wb.expr(r"\operatorname{fft}(\sin(2\pi \cdot 5\, t_n))")
        peak = int(np.argmax(np.abs(spectrum.to_numpy()[:32])))
        assert peak == 5
        assert spectrum.spaces[0].basis[5] == "5"  # bin 5: five cycles over the grid
        wb.exe(r"f_n := \sin(2\pi \cdot 5\, t_n)")
        tone = r"\operatorname*{arg\,max}_k \left| \operatorname{fft}(f)_k \right|"
        assert wb.expr(tone) == 5

    def test_a_list_is_transformed_exactly(self, wb):
        assert list(wb.expr("fft([1, 2, 0, -1])").components) == [
            2, 1 - 3 * sympy.I, 0, 1 + 3 * sympy.I
        ]  # fmt: skip

    def test_data_over_several_indices_is_not_supported_yet(self, wb):
        wb["R_{ta}"] = np.eye(3)
        with pytest.raises(CompileError, match="over 2 indices is not supported yet"):
            wb.expr(r"\operatorname{fft}(R_{ta})")


DFT = r"X_k := \sum_{n=0}^{N-1} x_n e^{-2\pi i k n/N}, \quad k = 0, \ldots, N-1"
GRID = r"x_n := n, \quad n = 0, \ldots, N-1"


class TestTheSumRewritten:
    def test_the_dft_written_as_its_sum_is_computed_as_an_fft(self, wb):
        wb["x_n"] = SIGNAL
        wb.exe("N := 8")
        assert "call fft" in wb.ir(DFT + "\nX_k")
        wb.exe(DFT)
        assert np.allclose(wb["X"].to_numpy(), np.fft.fft(SIGNAL))
        (reading,) = wb.explain(DFT)
        assert any(decision.kind == "rewrite" for decision in reading.decisions)

    def test_it_gives_the_numbers_of_the_sum_as_written(self, wb):
        wb["x_n"] = SIGNAL
        wb.exe("N := 8")
        rewritten = wb.exe(DFT + "\nX_k")[-1].to_numpy()
        with _transforms.written_as_is():
            assert "call fft" not in wb.ir(DFT + "\nX_k")
            written = wb.exe(DFT + "\nX_k")[-1].to_numpy()
        assert np.allclose(rewritten, written)

    def test_a_list_summed_over_whole_is_data(self, wb):
        wb.exe("x := (1, 2, 0, -1, 3, 0, 0, 2)\nN := 8\n" + DFT)
        assert np.allclose(wb["X"].to_numpy().astype(complex), np.fft.fft(SIGNAL))

    @pytest.mark.parametrize(
        "kernel",
        [r"e^{2\pi i k n/N}", r"e^{-2\pi i k n^2/N}", r"e^{-i k n}", r"e^{-2\pi i n/N}"],
    )
    def test_another_kernel_is_computed_as_written(self, wb, kernel):
        wb["x_n"] = SIGNAL
        wb.exe("N := 8")
        program = rf"X_k := \sum_{{n=0}}^{{N-1}} x_n {kernel}, \quad k = 0, \ldots, N-1"
        assert "call fft" not in wb.ir(program + "\nX_k")


class TestTheSuite:
    def test_a_transform_that_is_not_its_integral_is_told(self):
        @transform(
            "brokenlaplace",
            latex=r"\mathcal{B}\{f\}",
            summary="a transform that is not its integral",
            variables=[("t", "s")],
            examples=[Example(r"\mathcal{B}\{t\}", 2 / s**2, "made up")] * 2,
            points=[(t,)],
        )
        class Broken:
            def exact(self, f, t=None, s=None):
                return 2 / s**2

            def integral(self, f, t, s):
                return sympy.Integral(sympy.exp(-s * t) * f, (t, 0, sympy.oo))

        failures = integrations.check(Broken, ["transform"])["transform"]
        assert failures == [
            "brokenlaplace of t at s = 13/10 is 1.183431953, but its integral is 0.5917159763: "
            "fix exact, or the integral"
        ]

    def test_a_rewrite_that_gives_other_numbers_is_told(self):
        @rewrite(
            "brokensum",
            computes="ifft",
            latex=r"\sum_{n=0}^{N-1} x_n e^{-2\pi i k n/N}",
            summary="a sum computed by what it is not",
            examples=[Example("N := 4\n" + GRID + "\n" + DFT, None, "made up")] * 2,
        )
        class Broken:
            def matches(self, kernel, n, k, size):
                return 1

        with integrations.installed(Broken):
            failures = integrations.check(Broken, ["rewrite"])["rewrite"]
        assert any("gives other numbers rewritten than written" in failure for failure in failures)
