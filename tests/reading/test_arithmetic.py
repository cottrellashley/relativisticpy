r"""Number theory, read as books write it (design/notation.md, 19, 30 and 32 to 37): the
letters of the arithmetic functions only of integers, so that φ(x), μ(t) and π(x) of
physics stay what they were; sums and products over the values a condition gives their
index; congruences and remainders; and the coefficients of a generating function."""

from __future__ import annotations

import pytest
import sympy

from relativisticpy import Workbook, display
from relativisticpy.errors import CompileError, EvaluationError
from relativisticpy.language import parse, to_sexpr

n = sympy.Symbol("n", integer=True, nonnegative=True)
x, t = sympy.symbols("x t")


def notes(wb: Workbook, source: str) -> list[str]:
    return [note for step in wb.each(source) for note in step.notes]


def sexpr(source: str) -> str:
    return to_sexpr(parse(source).statements[0].expr)


class TestParsing:
    def test_a_condition_with_a_bar_under_a_sum_is_its_index_set(self):
        assert sexpr(r"\sum_{d \mid 28} d") == "(sum (∣ d 28) d)"
        assert sexpr(r"\prod_{p | n} p") == "(prod (∣ p n) p)"

    def test_the_rows_of_a_substack_are_conditions_that_all_hold(self):
        written = r"\sum_{\substack{1 \le k \le n \\ \gcd(k, n) = 1}} k"
        assert sexpr(written) == "(sum (and (chain 1 <= k <= n) (= (call gcd k n) 1)) k)"

    def test_karamatas_brackets_are_the_stirling_numbers(self):
        assert sexpr(r"{5 \brace k}") == "(call stirling2 5 k)"
        assert sexpr(r"{5 \brack k}") == "(call stirling1 5 k)"

    def test_bmod_binds_as_a_product_and_pmod_takes_all_of_its_side(self):
        assert sexpr(r"2 + 12! \bmod 13") == "(+ 2 (mod (! 12) 13))"
        assert sexpr(r"(3 \equiv 1 + 2 \pmod{5})") == "(≡ 3 (mod (+ 1 2) 5))"

    def test_equiv_with_a_modulus_is_a_congruence_not_a_definition(self):
        statement = parse(r"x \equiv 2 \pmod{3}").statements[0]
        assert to_sexpr(statement.expr) == "(≡ x (mod 2 3))"  # type: ignore[attr-defined]


class TestLetters:
    """Decision 32: φ, μ, Λ, σ_k, ω, Ω, d and p are the arithmetic functions only of
    integers."""

    @pytest.mark.parametrize(
        ("written", "value"),
        [
            (r"\varphi(36)", 12),
            (r"\phi(97)", 96),
            (r"\mu(30)", -1),
            (r"\sigma_1(28)", 56),
            ("d(28)", 6),
            (r"\omega(30)", 3),
            (r"\Omega(1024)", 10),
            (r"\Lambda(8)", sympy.log(2)),
            ("p(5)", 7),
        ],
    )
    def test_of_a_whole_number_a_letter_is_its_arithmetic_function(self, wb, written, value):
        assert wb.expr(written) == value

    def test_a_note_says_how_the_letter_was_read(self, wb):
        assert notes(wb, r"\varphi(36)") == [
            r"\varphi(36) is read as Euler's totient function, totient, as 36 is an integer; "
            r"for a function of your own, define \varphi first"
        ]

    @pytest.mark.parametrize("written", [r"\varphi(x)", r"\mu(t)", "p(t)", r"\phi(0)", r"\mu(1/2)"])
    def test_of_anything_else_it_stays_a_function_of_the_program(self, wb, written):
        value = wb.expr(written)
        assert isinstance(value, sympy.core.function.AppliedUndef)

    def test_of_a_name_assumed_an_integer_it_is_the_function(self, wb):
        wb.exe(r"n \in \mathbb{N}")
        assert wb.expr(r"\varphi(n)") == sympy.totient(n)
        assert wb.expr(r"p(5n + 4)") == sympy.partition(5 * n + 4)

    def test_of_the_index_of_a_sum_it_is_the_function(self, wb):
        assert wb.expr(r"\sum_{k=1}^{10} \varphi(k)") == 32

    def test_a_name_assumed_an_integer_is_the_index_of_a_sum_too(self, wb):
        wb.exe(r"n \in \mathbb{N}")
        assert wb.expr(r"\sum_{n=1}^{\infty} \frac{1}{n^2}") == sympy.pi**2 / 6

    def test_a_definition_of_the_letter_wins(self, wb):
        wb.exe(r"\varphi := \frac{1 + \sqrt{5}}{2}")
        assert sympy.simplify(wb.expr(r"\varphi^2 - \varphi") - 1) == 0

    def test_a_call_of_a_symbol_prints_by_its_name_and_reads_back(self, wb):
        value = wb.expr(r"totient(x)")
        assert display.latex(value) == r"\operatorname{totient}\left(x\right)"
        assert wb.expr(display.latex(value)) == value

    def test_of_an_integer_it_prints_as_books_write_it(self, wb):
        wb.exe(r"n \in \mathbb{N}")
        assert display.latex(wb.expr(r"\varphi(n)")) == r"\varphi\left(n\right)"


class TestPi:
    """Decision 33: π(x) is the prime-counting function only of a name assumed an
    integer."""

    def test_of_a_number_it_is_pi_times_the_number_and_says_how_to_count_primes(self, wb):
        assert wb.expr(r"\pi(10)") == 10 * sympy.pi
        assert notes(wb, r"\pi(10)") == [
            r"\pi(10) is read as \pi times 10; the number of primes up to 10 is primepi(10)"
        ]

    @pytest.mark.parametrize("written", [r"\pi(x)", r"\pi(r^2 + a^2)", r"\sum_{k=1}^{3} \pi(k)"])
    def test_in_physics_it_stays_pi_times_what_follows(self, wb, written):
        assert wb.expr(written).has(sympy.pi)

    def test_of_a_name_assumed_an_integer_it_counts_primes(self, wb):
        wb.exe(r"n \in \mathbb{N}")
        assert wb.expr(r"\left. \pi(n) \right|_{n=100}") == 25


class TestSubscripts:
    """Decisions 34 and 35: H_n and B_n are numbers only of names assumed integers."""

    @pytest.mark.parametrize("written", ["H_0", "B_0", "H_4", "B_{10}"])
    def test_with_a_number_they_stay_names(self, wb, written):
        assert isinstance(wb.expr(written), sympy.Symbol)

    def test_of_a_name_assumed_an_integer_they_are_the_numbers(self, wb):
        wb.exe(r"n \in \mathbb{N}")
        assert wb.expr("H_{n+1} - H_n") == 1 / (n + 1)
        assert wb.expr(r"\left. B_{2n} \right|_{n=3}") == sympy.Rational(1, 42)

    def test_the_coefficients_of_a_series_stay_names(self, wb):
        wb.exe(r"n \in \mathbb{N}")
        assert wb.expr(r"\sum_{n=1}^{2} B_n").free_symbols

    def test_a_family_of_the_program_wins(self, wb):
        wb.exe(r"n \in \mathbb{N}" "\nH_1 := 5")
        assert isinstance(wb.expr("H_n"), sympy.Symbol)

    def test_called_h_n_is_hermites_polynomial(self, wb):
        assert wb.expr("H_2(x)") == 4 * x**2 - 2


class TestIndexSets:
    """Decisions 19 and 36: a condition under a sum or a product gives its index its
    values."""

    @pytest.mark.parametrize(
        ("written", "value"),
        [
            (r"\sum_{d \mid 28} d", 56),
            (r"\sum_{d \mid 36} \mu(d) \frac{36}{d}", 12),
            (r"36 \prod_{p \mid 36} \left(1 - \frac{1}{p}\right)", 12),
            (r"\prod_{p \le 10} p", 210),
            (r"\sum_{k \le 4} k^2", 30),
            (r"\sum_{\substack{1 \le k \le 10 \\ \gcd(k, 10) = 1}} k", 20),
            (r"\sum_{\substack{d \mid 28 \\ d < 28}} d", 28),
            (r"\sum_{\substack{1 \le k \le 9 \\ k \text{ odd}}} k", 25),
            (r"\sum_{k \neq 0} \frac{1}{k^2}", sympy.pi**2 / 3),
        ],
    )
    def test_the_values_are_the_ones_it_gives(self, wb, written, value):
        assert sympy.simplify(wb.expr(written) - value) == 0

    def test_p_ranges_over_the_primes_and_says_so(self, wb):
        assert notes(wb, r"\sum_{p \le 10} 1") == [
            "p ranges over the primes, as number theory writes it"
        ]

    def test_eulers_product_over_every_prime_is_zeta(self, wb):
        assert wb.expr(r"\prod_{p \text{ prime}} \frac{1}{1 - p^{-2}}") == sympy.pi**2 / 6
        assert wb.expr(r"\prod_{p \text{ prime}} \frac{1}{1 - p^{-2}} = \frac{\pi^2}{6}")
        with pytest.raises(EvaluationError, match="diverges where s is 1"):
            wb.expr(r"\prod_{p \text{ prime}} \frac{1}{1 - 1/p}")

    def test_the_divisors_of_a_symbol_are_refused(self, wb):
        wb.exe(r"n \in \mathbb{N}")
        with pytest.raises(EvaluationError, match="where n is a number"):
            wb.expr(r"\sum_{d \mid n} d")

    def test_the_gcd_of_an_index_waits_for_its_values(self, wb):
        assert wb.expr(r"\frac{1}{6} \sum_{k=0}^{5} 2^{\gcd(k, 6)}") == 14
        assert wb.expr("gcd(x^2 - 1, x - 1)") == x - 1

    def test_a_sum_over_every_integer_is_left_to_its_binomial_coefficients(self, wb):
        """Decision 38."""
        wb.exe(r"n \in \mathbb{Z}")
        assert wb.expr(r"\sum_k \binom{r}{k} \binom{s}{n-k} = \binom{r+s}{n}")
        assert wb.expr(r"\sum_k \binom{5}{k}") == 32
        with pytest.raises(CompileError, match="needs limits"):
            wb.expr(r"\sum_k k^2")

    def test_a_sum_of_finitely_many_terms_holds_no_conditions_of_convergence(self, wb):
        wb.exe(r"n \in \mathbb{N}")
        value = wb.expr(r"\sum_{k=0}^{n} \binom{n}{k} x^k y^{n-k}")
        y = sympy.Symbol("y")
        assert sympy.simplify(value - (x + y) ** n) == 0

    def test_the_stirling_numbers_sum_to_the_bell_number_and_to_n_factorial(self, wb):
        assert wb.expr(r"\sum_{k=0}^{5} {5 \brace k}") == 52
        assert wb.expr(r"\sum_{k=0}^{5} {5 \brack k}") == 120


class TestCongruences:
    """Decision 30: congruences and remainders."""

    def test_bmod_is_the_remainder(self, wb):
        assert wb.expr(r"12! \bmod 13") == 12

    def test_a_congruence_is_a_claim_decided_by_the_remainders(self, wb):
        assert wb.expr(r"3^{100} \equiv 1 \pmod{101}") is sympy.true
        assert wb.expr(r"7 \equiv 3 \mod 5") is sympy.false

    def test_a_power_to_minus_one_is_the_inverse(self, wb):
        assert wb.expr(r"17^{-1} \bmod 43") == 38
        with pytest.raises(EvaluationError, match="no inverse modulo 9"):
            wb.expr(r"6^{-1} \bmod 9")

    def test_congruences_and_bounds_assumed_of_a_name_leave_it_one_value(self, wb):
        values = wb.exe(
            r"x \equiv 2 \pmod{3}" "\n" r"x \equiv 3 \pmod{5}" "\n" r"x \equiv 2 \pmod{7}" "\n"
            r"0 \leq x < 105" "\n" "x"
        )  # fmt: skip
        assert values == [23]

    def test_a_congruence_that_contradicts_one_assumed_is_refused(self, wb):
        wb.exe(r"x \equiv 1 \pmod{2}")
        with pytest.raises(CompileError, match="contradicts what was assumed before"):
            wb.exe(r"x \equiv 0 \pmod{4}")

    def test_solve_gives_the_remainders_modulo_the_moduli(self, wb):
        system = r"solve(x \equiv 2 \pmod{3}, x \equiv 3 \pmod{5}, x \equiv 2 \pmod{7}, x)"
        assert wb.expr(system) == [23]
        assert wb.expr(r"solve(x^2 \equiv 1 \pmod{8}, x)") == [1, 3, 5, 7]

    def test_equiv_without_a_modulus_still_defines(self, wb):
        wb.exe(r"y \equiv 2")
        assert wb.expr("y + 1") == 3


class TestCoefficients:
    """Decision 37: [x^n] F(x), the coefficient of x^n in the series of F."""

    def test_it_reads_a_number_from_a_generating_function(self, wb):
        assert wb.expr(r"[x^{10}] \frac{1 - \sqrt{1 - 4x}}{2x}") == 16796
        assert wb.expr(r"[x^3] e^{2x}") == sympy.Rational(4, 3)

    def test_of_a_symbolic_power_it_is_refused(self, wb):
        wb.exe(r"n \in \mathbb{N}")
        with pytest.raises(EvaluationError, match="where the power is a whole number"):
            wb.expr(r"[x^n] \frac{1}{1 - x}")


class TestLettersPhysicsWritesAsFactors:
    """Finding 12 of the review of the reading stack: after ``n \\in \\mathbb{N}``,
    ``E_n := \\hbar\\omega(2n+1)/2`` read ω as the number of distinct primes of 2n + 1, so
    that E_0 was 0 and E_1 ħ/2, and ``\\sum_{n=0}^{3} \\hbar\\omega(2n+1)`` was 3ħ. ω, Ω, μ,
    Λ and φ follow the conservative rule of decision 33: each is its arithmetic function
    only of a whole number or of one name that is an integer, and never where it stands
    right after another symbol, as a factor does."""

    hbar, omega = sympy.symbols("hbar omega")

    def test_a_frequency_beside_hbar_is_a_factor(self, wb):
        wb.exe(r"n \in \mathbb{N}" "\n" r"E_n := \frac{\hbar\omega(2n+1)}{2}")
        assert wb.expr("E_0") == self.hbar * self.omega / 2
        assert wb.expr("E_1") == 3 * self.hbar * self.omega / 2

    def test_it_is_one_whatever_was_written_before_it(self, wb):
        written = r"\sum_{n=0}^{3} \hbar\omega(2n+1)"
        assert wb.expr(written) == 16 * self.hbar * self.omega
        assert notes(wb, written) == [
            r"\omega(2n+1) is read as \omega times 2n+1, as \omega is written right after "
            "another symbol, as a factor is; the number of distinct primes that divide 2n+1 "
            "is primenu(2n+1)"
        ]
        wb.exe(r"T := \frac{2\pi}{\omega}")  # \omega as a factor before: the same reading
        assert wb.expr(written) == 16 * self.hbar * self.omega

    def test_beside_another_symbol_it_is_never_the_arithmetic_function(self, wb):
        wb.exe(r"k \in \mathbb{N}")
        value = wb.expr(r"\hbar\omega(k)")  # a frequency, a function of k
        assert value == self.hbar * sympy.Function("omega")(sympy.Symbol("k", **n.assumptions0))
        assert wb.expr(r"m\mu(2k + 1)") == sympy.Symbol("m") * sympy.Symbol("mu") * (
            2 * sympy.Symbol("k", **n.assumptions0) + 1
        )

    @pytest.mark.parametrize("written", [r"\omega(2n+1)", r"\varphi(n^2)", r"\mu(n - 1)"])
    def test_of_more_than_one_name_it_stays_a_function_of_the_program(self, wb, written):
        wb.exe(r"n \in \mathbb{N}")
        value = wb.expr(written)
        assert isinstance(value, sympy.core.function.AppliedUndef)

    def test_of_a_whole_number_or_one_integer_it_is_the_function_still(self, wb):
        wb.exe(r"n \in \mathbb{N}")
        assert wb.expr(r"\omega(30)") == 3
        assert wb.expr(r"2\varphi(36)") == 24  # after a number, not a symbol
        assert wb.expr(r"\omega(n)") == sympy.primenu(n)
        assert wb.expr(r"\sum_{d \mid 36} \mu(d) \frac{36}{d}") == 12
        assert wb.expr(r"p(5n + 4)") == sympy.partition(5 * n + 4)  # p is no such letter

    def test_the_note_names_the_argument_as_it_is_written(self, wb):
        wb.exe(r"k \in \mathbb{N}")
        assert notes(wb, r"\omega(k)") == [
            r"\omega(k) is read as the number of distinct primes that divide k, primenu, as k "
            r"is an integer; for a function of your own, define \omega first"
        ]
