r"""Random variables, read as books write them (design/notation.md, 14): ``X \sim D``
declares one by its distribution, ``\Pr``, ``P`` and ``\mathbb{P}`` of an event are
probabilities, with ``\mid`` the conditional bar inside them, and ``\mathbb{E}``,
``\operatorname{Var}`` and ``\operatorname{Cov}`` are moments, while ``\mathbb{E}_t`` of data
over an index keeps its meaning."""

from __future__ import annotations

import pytest
import sympy

from relativisticpy import Workbook
from relativisticpy.errors import CompileError, EvaluationError
from relativisticpy.language import parse, to_sexpr

lam, p, t, x, a, b, c, mu, sigma = sympy.symbols("lambda p t x a b c mu sigma")
N = sympy.Symbol("N")


def notes(wb: Workbook, source: str) -> list[str]:
    return [note for step in wb.each(source) for note in step.notes]


def assumed(wb: Workbook, source: str) -> list[str]:
    return [each for step in wb.each(source) for each in step.assuming]


def sexpr(source: str) -> str:
    return to_sexpr(parse(source).statements[0].expr)


def same(value: object, expected: object) -> bool:
    """Whether two values are equal, their symbols compared by name, whatever is assumed of
    them, as the curriculum compares them."""

    def plain(found: object) -> sympy.Expr:
        found = sympy.sympify(found)
        return found.xreplace({s: sympy.Symbol(s.name) for s in found.free_symbols})

    return sympy.simplify(plain(value) - plain(expected)) == 0


class TestParsing:
    def test_mid_inside_a_probability_is_the_conditional_bar(self):
        assert sexpr(r"\Pr(A \mid B_1)") == "(call Pr (∣ A B_1))"
        assert sexpr(r"\mathbb{E}[X \mid N]") == "(* 𝔼 (list (∣ X N)))"
        assert sexpr(r"\Pr(X = 2 \mid N = 3)") == "(call Pr (∣ (= X 2) (= N 3)))"
        assert sexpr(r"\Pr(X > 1 \mid Y = 0)") == "(call Pr (∣ (> X 1) (= Y 0)))"

    def test_a_law_given_another_is_one_statement(self):
        assert sexpr(r"X \mid N \sim \operatorname{Bin}(N, p)") == "(~ (∣ X N) (call Bin N p))"
        assert (
            sexpr(r"X \mid Y = 0 \sim \mathcal{N}(-1, 1)") == "(~ (∣ X (= Y 0)) (call N (- 1) 1))"
        )

    def test_names_listed_before_sim_are_declared_together(self):
        assert sexpr(r"X, Y \sim \mathcal{N}(0, 1)") == "(~ (tuple X Y) (call N 0 1))"
        listed = r"X_1, \ldots, X_n \overset{\text{iid}}{\sim} \mathcal{N}(0, 1)"
        assert sexpr(listed) == "(~ (tuple X_1 ⋯ X_n) (call N 0 1))"

    def test_mid_keeps_its_other_meanings(self):
        assert sexpr(r"\sum_{d \mid 28} d") == "(sum (∣ d 28) d)"
        assert sexpr(r"\langle \phi \mid \psi \rangle").startswith("(braket")
        assert sexpr("|x| + 1") == "(+ (abs x) 1)"

    def test_overset_is_a_label_over_a_relation_only(self):
        assert sexpr(r"X \overset{\text{iid}}{\sim} \operatorname{Exp}(1)") == "(~ X (call Exp 1))"
        with pytest.raises(Exception, match="no relation"):
            parse(r"u \overset{p}{\leadsto} v")


class TestDeclarations:
    def test_sim_with_a_distribution_declares_a_random_variable(self, wb):
        found = notes(wb, r"X \sim \mathcal{N}(0, 4)")
        assert found == [
            r"X \sim \mathcal{N}(0, 4) declares X a random variable, of the normal distribution, "
            "of mean μ and standard deviation σ, with μ = 0 and σ² = 4"
        ]
        assert wb.expr(r"\operatorname{Var}(X)") == 4

    def test_plain_text_calls_a_distribution_by_its_name_and_parameters(self, wb):
        wb.run("X ∼ Normal(0, 2)")
        assert wb.expr("Var(X)") == 4

    def test_a_distribution_is_no_value_of_its_own(self, wb):
        with pytest.raises(CompileError, match=r"declare a random variable with it"):
            wb.run("X Normal(0, 2)")

    @pytest.mark.parametrize(
        ("written", "mean", "variance"),
        [
            (r"\operatorname{Poisson}(3)", 3, 3),
            (r"\operatorname{Bin}(10, 1/2)", 5, sympy.Rational(5, 2)),
            (r"\operatorname{Bernoulli}(1/4)", sympy.Rational(1, 4), sympy.Rational(3, 16)),
            ("U(0, 1)", sympy.Rational(1, 2), sympy.Rational(1, 12)),
            (r"\operatorname{Exp}(2)", sympy.Rational(1, 2), sympy.Rational(1, 4)),
            (r"\Gamma(2, 3)", 6, 18),
            (r"\chi^2_3", 3, 6),
            ("t_5", 0, sympy.Rational(5, 3)),
            (r"\operatorname{Beta}(2, 3)", sympy.Rational(2, 5), sympy.Rational(1, 25)),
            (r"\operatorname{Geom}(1/3)", 3, 6),
        ],
    )
    def test_each_distribution_is_read_as_books_write_it(self, wb, written, mean, variance):
        wb.run(f"X \\sim {written}")
        assert wb.expr(r"\mathbb{E}[X]") == mean
        assert wb.expr(r"\operatorname{Var}(X)") == variance

    def test_sim_between_anything_else_is_read_but_not_computed_as_before(self, wb):
        with pytest.raises(CompileError, match="read but not computed"):
            wb.run(r"\alpha \sim l_P^2")
        with pytest.raises(CompileError, match="read but not computed"):
            wb.run(r"n! \sim \sqrt{2\pi n} \left(\frac{n}{e}\right)^n")

    def test_a_name_with_a_value_is_not_declared_again(self, wb):
        wb.run("X := 3")
        with pytest.raises(CompileError, match="is defined"):
            wb.run(r"X \sim \mathcal{N}(0, 1)")

    def test_a_random_variable_is_not_defined_again(self, wb):
        # P(X < 1) after it would be of the number 5, the probability of nothing
        wb.run(r"X \sim \mathcal{N}(0, 1)")
        with pytest.raises(CompileError) as refused:
            wb.run("X := 5")
        assert refused.value.message == (
            "X is declared a random variable on line 1, so X := 5 cannot define it"
        )
        assert same(wb.expr("P(X < 1)"), (1 + sympy.erf(1 / sympy.sqrt(2))) / 2)

    @pytest.mark.parametrize(
        "program",
        [r"X \sim \mathcal{N}(0,(-2)^2)", "s := -2\n" r"X \sim \mathcal{N}(0, s^2)"],
    )
    def test_a_variance_written_as_the_square_of_a_negative_number(self, wb, program):
        wb.run(program)
        assert wb.expr(r"\operatorname{Var}(X)") == 4

    def test_a_variance_defined_as_a_name_of_its_own_is_its_value(self, wb):
        # \sigma^2 := 4 defines the name σ²: the law's σ is its root, not a symbol σ
        wb.run("\\sigma^2 := 4\nX \\sim \\mathcal{N}(0, \\sigma^2)")
        assert wb.expr(r"\operatorname{Var}(X)") == 4
        assert same(wb.expr("P(X < 2)"), (1 + sympy.erf(1 / sympy.sqrt(2))) / 2)

    def test_a_note_lists_the_variables_as_they_are_written(self, wb):
        source = r"X_1, \ldots, X_4 \overset{\text{iid}}{\sim} \operatorname{Exp}(1)"
        (note,) = notes(wb, source)
        assert note.startswith(source + r" declares X_1, \ldots, X_4 random variables, of ")

    def test_a_random_process_is_refused_saying_what_it_is(self, wb):
        with pytest.raises(CompileError, match="random process"):
            wb.run(r"w[n] \sim \mathcal{N}(0, \sigma^2) \text{ independent}")

    def test_parameters_a_law_does_not_allow_are_refused(self, wb):
        with pytest.raises(EvaluationError, match="no distribution"):
            wb.run("X \\sim \\mathcal{N}(0, -1)\n\\mathbb{E}[X]")

    def test_a_random_variable_shows_as_itself(self, wb):
        wb.run(r"X \sim \mathcal{N}(0, 1)")
        assert str(wb.expr("X + 1")) == "X + 1"

    def test_explain_says_what_a_random_variable_is(self, wb):
        text = str(wb.explain("X \\sim \\operatorname{Exp}(1)\n\\Pr(X > 1)"))
        assert r"a random variable, X \sim \operatorname{Exp}(1)" in text


class TestProbabilities:
    def test_of_a_poisson_variable(self, wb):
        wb.run(r"X \sim \operatorname{Poisson}(\lambda)")
        assert same(wb.expr("P(X = 2)"), lam**2 * sympy.exp(-lam) / 2)
        assert same(wb.expr(r"\Pr(X \le 2)"), (lam**2 / 2 + lam + 1) * sympy.exp(-lam))

    def test_of_a_normal_variable(self, wb):
        wb.run(r"Z \sim \mathcal{N}(0, 1)")
        assert same(wb.expr(r"\Pr(Z > 1)"), (1 - sympy.erf(1 / sympy.sqrt(2))) / 2)
        assert wb.expr(r"\Pr(|Z| > 1.96)") == pytest.approx(0.04999579029644087)

    def test_given_an_event_the_exponential_forgets(self, wb):
        wb.run("s, t > 0\nT \\sim \\operatorname{Exp}(\\lambda)")
        assert same(wb.expr(r"\Pr(T > s + t \mid T > s)"), sympy.exp(-lam * t))

    def test_an_end_nothing_places_is_taken_within_the_values_and_said(self, wb):
        source = "U \\sim U(0, 1)\n\\Pr(U < c)"
        assert wb.run(source)[-1] == c
        assert assumed(Workbook(), source) == ["c ≤ 1", "0 ≤ c"]

    def test_the_probability_of_an_event_the_program_gives(self, wb):
        wb.run(
            "\\Pr(B_1) := 0.0001\n\\Pr(B_2) := 0.9999\n"
            "\\Pr(A \\mid B_1) := 0.9\n\\Pr(A \\mid B_2) := 0.1"
        )
        found = wb.expr(
            r"\frac{\Pr(A \mid B_1) \Pr(B_1)}{\Pr(A \mid B_1) \Pr(B_1) + \Pr(A \mid B_2) \Pr(B_2)}"
        )
        assert found == pytest.approx(9 / 10008)

    def test_an_event_whose_probability_is_not_given_is_refused(self, wb):
        with pytest.raises(CompileError, match="not given"):
            wb.run(r"\Pr(A)")
        with pytest.raises(CompileError, match="outside 0 to 1"):
            wb.run(r"\Pr(A) := 1.5")

    def test_the_probability_of_an_event_of_a_declared_variable_is_its_laws(self, wb):
        # given as written, \Pr(X > 0) would be 0.7 and \Pr(0 < X) the law's 1/2
        wb.run(r"X \sim \mathcal{N}(0, 1)")
        with pytest.raises(CompileError, match="a law, which gives its probability") as refused:
            wb.run(r"\Pr(X>0) := 0.7")
        assert "\\Pr(A) := 0.3" in (refused.value.hint or "")
        half = sympy.Rational(1, 2)
        assert wb.run("\\Pr(X>0)\n\\Pr(0<X)\n\\Pr(X \\le 0)") == [half, half, half]

    def test_p_is_a_probability_only_of_random_variables(self, wb):
        assert isinstance(wb.expr("P(x)"), sympy.core.function.AppliedUndef)
        wb.run("Q(x) := x^2")
        assert wb.expr("Q(3)") == 9
        wb.run("P(v) := 2 v")
        assert wb.expr("P(3)") == 6

    def test_of_a_variable_given_values_of_another(self, wb):
        wb.run(
            "Y \\sim \\operatorname{Bernoulli}(1/2)\n"
            "X \\mid Y = 0 \\sim \\mathcal{N}(-1, 1)\n"
            "X \\mid Y = 1 \\sim \\mathcal{N}(1, 1)"
        )
        assert wb.expr(r"\Pr(X > 0)") == sympy.Rational(1, 2)
        assert same(wb.expr(r"\Pr(Y = 1 \mid X > 0)"), (1 + sympy.erf(1 / sympy.sqrt(2))) / 2)


class TestEventsThatShareAVariable:
    """Events of several random variables that share one are not independent: SymPy
    multiplies the probabilities of events joined by and, and adds those joined by or."""

    POISSON = "X \\sim \\operatorname{Poisson}(2)\nY \\sim \\operatorname{Poisson}(3)"

    @pytest.mark.parametrize(
        ("written", "value"),
        [
            (r"P(X = 1 \mid X + Y = 3)", sympy.Rational(54, 125)),  # of Bin(3, 2/5)
            (r"P(X = 1, X + Y = 3)", 9 * sympy.exp(-5)),  # P(X = 1) P(Y = 2)
            (r"\mathbb{E}[X \mid X + Y = 3]", sympy.Rational(6, 5)),
            (r"P(X + Y = 3)", 125 * sympy.exp(-5) / 6),  # of Poisson(5)
            (r"P(X + Y \le 2)", sympy.Rational(37, 2) * sympy.exp(-5)),
            (r"P(X + Y \ge 2)", 1 - 6 * sympy.exp(-5)),
            (r"P(2X + Y = 4)", sympy.Rational(115, 8) * sympy.exp(-5)),
            (r"\mathbb{E}[X Y \mid X + Y = 3]", sympy.Rational(36, 25)),
            (r"P(X > 0, Y > 0)", (1 - sympy.exp(-2)) * (1 - sympy.exp(-3))),
            (r"P(X > 0 \text{ or } Y > 0)", 1 - sympy.exp(-5)),
            (r"P(X = 1 \mid Y = 1)", 2 * sympy.exp(-2)),  # they are independent
        ],
    )
    def test_of_discrete_variables_they_are_summed_over_the_values_of_one(self, wb, written, value):
        wb.run(self.POISSON)
        assert same(wb.expr(written), value)

    def test_with_rates_that_are_symbols(self, wb):
        wb.run("X \\sim \\operatorname{Poisson}(\\lambda)\nY \\sim \\operatorname{Poisson}(\\mu)")
        assert same(wb.expr(r"P(X = 1 \mid X + Y = 2)"), 2 * lam * mu / (lam + mu) ** 2)

    def test_of_variables_that_take_finitely_many_values(self, wb):
        wb.run("D \\sim \\operatorname{Bin}(2, 1/2)\nE \\sim \\operatorname{Bin}(2, 1/2)")
        assert wb.expr("P(D = E)") == sympy.Rational(3, 8)
        assert wb.expr("P(D E = 1)") == sympy.Rational(1, 4)  # no linear relation
        assert wb.expr(r"\mathbb{E}[D \mid D \ge E]") == sympy.Rational(14, 11)

    def test_of_continuous_variables_apart_and_together(self, wb):
        wb.run("X \\sim \\mathcal{N}(0, 1)\nY \\sim \\mathcal{N}(0, 1)")
        assert wb.expr("P(X > 0, Y > 0)") == sympy.Rational(1, 4)
        assert wb.expr(r"P(X > 0 \text{ or } Y > 0)") == sympy.Rational(3, 4)
        assert wb.expr("P(X = 1, X + Y = 3)") == 0
        assert wb.expr("P(X + Y < 0)") == sympy.Rational(1, 2)  # SymPy's, borne out

    def test_what_sympy_gives_wrongly_is_refused(self, wb):
        # SymPy gives the expectation of X where X + Y > 0, and 0 elsewhere, as 0: it is
        # 1/(2√π), and so E[X | X + Y > 0] is 1/√π
        wb.run("X \\sim \\mathcal{N}(0, 1)\nY \\sim \\mathcal{N}(0, 1)")
        with pytest.raises(EvaluationError, match="SymPy gives 0, and a simulation of it"):
            wb.expr(r"\mathbb{E}[X \mid X + Y > 0]")
        with pytest.raises(EvaluationError, match="still holds X"):
            wb.expr("P(X > 0, X + Y < 1)")


class TestMoments:
    def test_the_variance_of_an_affine_function_holds_whatever_the_law(self, wb):
        assert wb.expr(r"\operatorname{Var}(a X + b) = a^2 \operatorname{Var}(X)") is sympy.true
        assert notes(wb, r"\operatorname{Var}(a X + b)") == [
            "X is a random variable whose law is not given, as probability writes random "
            "variables with capitals: the variance is found whatever the law is"
        ]

    def test_the_law_of_total_expectation(self, wb):
        wb.run(
            "N \\sim \\operatorname{Poisson}(\\lambda)\nX \\mid N \\sim \\operatorname{Bin}(N, p)"
        )
        assert same(wb.expr(r"\mathbb{E}[X]"), lam * p)
        assert same(wb.expr(r"\mathbb{E}[X \mid N]"), p * wb.expr("N"))
        assert same(wb.expr(r"\operatorname{Var}(X)"), lam * p)

    def test_expectations_as_books_write_them(self, wb):
        wb.run(r"X \sim \mathcal{N}(\mu, \sigma^2)")
        assert same(wb.expr(r"\mathbb{E}[X^2]"), mu**2 + sigma**2)
        assert same(wb.expr(r"\mathbb{E}\left[X^2\right] - \mathbb{E}[X]^2"), sigma**2)
        assert same(wb.expr(r"\mathbb{E}X"), mu)

    def test_independent_variables_are_said_so_where_the_program_does_not(self, wb):
        source = "X, Y \\sim \\mathcal{N}(0, 1)\n\\operatorname{Var}(X + Y)"
        assert wb.run(source)[-1] == 2
        said = "X and Y are taken independent, as nothing says how they depend on each other"
        assert said in notes(Workbook(), source)
        iid = "X, Y \\overset{\\text{iid}}{\\sim} \\mathcal{N}(0, 1)\n\\operatorname{Var}(X + Y)"
        assert said not in notes(Workbook(), iid)

    def test_a_sample_written_with_an_ellipsis(self, wb):
        wb.run(r"X_1, \ldots, X_4 \overset{\text{iid}}{\sim} \operatorname{Bernoulli}(p)")
        assert same(wb.expr(r"\mathbb{E}[X_1 + X_2 + X_3 + X_4]"), 4 * p)
        assert same(wb.expr(r"\operatorname{Var}(X_1 + X_4)"), 2 * p * (1 - p))

    def test_a_sample_of_a_size_that_is_a_symbol(self, wb):
        n = sympy.Symbol("n")
        wb.run(
            "X_1, \\ldots, X_n \\overset{\\text{iid}}{\\sim} \\mathcal{N}(\\mu, \\sigma^2)\n"
            "\\bar{X} := \\frac{1}{n} \\sum_{i=1}^{n} X_i"
        )
        assert same(wb.expr(r"\mathbb{E}[\bar{X}]"), mu)
        assert same(wb.expr(r"\operatorname{Var}(\bar{X})"), sigma**2 / n)
        assert same(wb.expr(r"\mathbb{E}\left[\sum_{i=1}^{n} X_i^2\right]"), n * (mu**2 + sigma**2))
        assert same(wb.expr(r"\mathbb{E}[X_1 X_2]"), mu**2)

    def test_a_sample_of_a_symbols_size_says_how_its_members_depend(self, wb):
        with pytest.raises(CompileError, match="without saying how they depend"):
            wb.run(r"X_1, \ldots, X_n \sim \mathcal{N}(0, 1)")

    def test_an_upright_e_with_square_brackets_is_the_expectation(self, wb):
        assert isinstance(wb.expr("E(k)"), sympy.core.function.AppliedUndef)
        assert wb.expr(r"\operatorname{E}(0)") == sympy.pi / 2  # the elliptic integral
        assert wb.expr(r"\operatorname{E}[a X + b] = a \operatorname{E}[X] + b") is sympy.true

    def test_angle_brackets_of_a_random_variable_are_its_mean(self, wb):
        wb.run(r"E \sim \operatorname{Exp}(\lambda)")
        assert same(wb.expr(r"\langle E \rangle"), 1 / lam)
        assert same(wb.expr(r"\langle E^2 \rangle - \langle E \rangle^2"), 1 / lam**2)

    def test_covariance(self, wb):
        wb.run("X \\sim \\mathcal{N}(0, 1)\nY \\sim \\operatorname{Exp}(1)")
        assert wb.expr(r"\operatorname{Cov}(X + Y, X)") == 1

    def test_given_an_event(self, wb):
        wb.run(r"X \sim \operatorname{Poisson}(2)")
        e = sympy.E
        assert same(wb.expr(r"\mathbb{E}[X \mid X > 1]"), (2 - 2 * e**2) / (3 - e**2))

    def test_the_second_moment_of_a_law_without_a_mean_is_infinite(self, wb):
        # the variance and the mean it was built of are both undefined
        wb.run(r"T \sim t_1")
        assert wb.expr(r"\mathbb{E}[T^2]") == sympy.oo
        assert wb.expr(r"\mathbb{E}[T^4]") == sympy.oo
        for written in (r"\mathbb{E}[T]", r"\operatorname{Var}(T)", r"\mathbb{E}[T^3]"):
            with pytest.raises(EvaluationError, match="is not defined") as refused:
                wb.expr(written)
            assert "Cauchy" in refused.value.notes[0]

    def test_a_law_with_a_mean_and_no_variance(self, wb):
        wb.run(r"T \sim t_2")
        assert wb.run("\\mathbb{E}[T]\n\\mathbb{E}[T^2]\n\\operatorname{Var}(T)") == [
            0,
            sympy.oo,
            sympy.oo,
        ]

    def test_of_data_the_statistic_is_written_with_its_index(self, wb):
        import numpy as np

        wb["x_t"] = np.array([1.0, 2.0, 3.0, 6.0])
        assert wb.expr(r"\mathbb{E}_t[x_t]") == pytest.approx(3.0)
        with pytest.raises(CompileError, match="over an index"):
            wb.run(r"\mathbb{E}[x_t]")

    def test_a_number_found_numerically_where_sympy_cannot_says_so(self, wb, monkeypatch):
        from relativisticpy.integrations import _distributions

        def slow(*args, **kwargs):
            raise NotImplementedError

        monkeypatch.setattr(_distributions.stats, "E", slow)
        steps = list(wb.each("X \\sim \\mathcal{N}(0, 1)\n\\mathbb{E}[\\cos(X^3)]"))
        # scipy.stats.norm.expect(lambda x: np.cos(x**3))
        assert float(steps[-1].values[0]) == pytest.approx(0.5965649688262752, rel=1e-8)
        assert any("numerically, by SciPy" in note for note in steps[-1].notes)


class TestFunctionsOfARandomVariable:
    def test_density_distribution_function_and_quantile(self, wb):
        wb.run(r"X \sim \mathcal{N}(0, 1)")
        assert same(wb.expr("f_X(x)"), sympy.exp(-(x**2) / 2) / sympy.sqrt(2 * sympy.pi))
        assert same(wb.expr("F_X(0)"), sympy.Rational(1, 2))
        assert wb.expr("F_X^{-1}(0.975)") == pytest.approx(1.959963984540054)

    def test_the_generating_functions(self, wb):
        wb.run(r"X \sim \operatorname{Poisson}(\lambda)")
        assert same(wb.expr("M_X(t)"), sympy.exp(lam * (sympy.exp(t) - 1)))
        assert same(wb.expr(r"\varphi_X(t)"), sympy.exp(lam * (sympy.exp(sympy.I * t) - 1)))

    def test_a_note_names_the_variable_not_the_function(self, wb):
        wb.run(r"X \sim \mathcal{N}(0, 1)")
        assert notes(wb, "f_X(0)") == ["f_X(0) is the density of X"]
        assert notes(wb, "M_X(t)") == ["M_X(t) is the moment-generating function of X"]

    @pytest.mark.parametrize(
        ("law", "written"),
        [
            (r"T \sim \operatorname{Exp}(1)", r"\mathbb{E}[e^{2T}]"),  # finite where t < λ
            (r"G \sim \Gamma(2, 1)", r"\mathbb{E}[e^{2G}]"),  # where t < 1/θ
            (r"C \sim \chi^2_2", r"\mathbb{E}[e^{C}]"),  # where t < 1/2
            (r"G \sim \operatorname{Geom}(1/2)", r"\mathbb{E}[e^{G}]"),  # where t < -ln(1 - p)
            (r"T \sim \operatorname{Exp}(1)", r"\mathbb{E}[e^{T}]"),  # nor at its end
        ],
    )
    def test_past_where_a_generating_function_is_finite_the_expectation_is_infinite(
        self, wb, law, written
    ):
        # the closed forms give -1, 1, -1 and -3.78 there
        wb.run(law)
        assert wb.expr(written) == sympy.oo

    def test_a_generating_function_is_used_where_it_is_finite(self, wb):
        wb.run(r"T \sim \operatorname{Exp}(1)")
        assert wb.expr(r"\mathbb{E}[e^{T/2}]") == 2
        with pytest.raises(EvaluationError, match=r"the variance of .* is not defined"):
            wb.expr(r"\operatorname{Var}(e^{T})")  # ∞ - ∞

    def test_of_a_symbol_it_says_where_it_is_finite(self, wb):
        wb.run(r"T \sim \operatorname{Exp}(\lambda)")
        assert same(wb.expr("M_T(t)"), lam / (lam - t))
        assert assumed(wb, "M_T(t)") == ["t < λ"]
        assert notes(wb, "M_T(t)")[-1] == (
            "assuming t < λ, where the moment-generating function is finite"
        )
        assert assumed(wb, r"\varphi_T(t)") == []  # the characteristic function always is
        assert assumed(wb, r"\mathbb{E}[e^{-T}]") == []

    def test_of_a_name_that_is_no_random_variable_they_are_as_before(self, wb):
        assert isinstance(wb.expr("f_0(x)"), sympy.Expr)
        assert not str(wb.expr("f_0(x)")).startswith("exp")


class TestMembersOfASampleAtAnIntegerSubscript:
    """Finding 18 of the review of the reading stack: only a subscript that was one name
    or one number made a member of a sample, so that ``\\mathbb{E}[X_{n-1}]`` was the symbol
    ``X_n-1`` and ``X_{k+1}`` in ``\\operatorname{Var}(X_{k+1} - X_k)`` a random variable
    of its own. Any subscript written as an integer is, of whole numbers and names in
    sums, differences and products, is lowered as a member; how members at symbols compare
    is the engine's."""

    SAMPLE = r"X_1, \ldots, X_n \overset{\text{iid}}{\sim} \mathcal{N}(0, 1)"

    def test_a_computed_subscript_is_a_member(self, wb):
        wb.exe(self.SAMPLE)
        assert wb.expr(r"\mathbb{E}[X_{n-1}]") == 0
        assert wb.expr(r"\mathbb{E}[X_{2k} + X_{n-1}]") == 0
        assert notes(wb, r"\mathbb{E}[X_{n-1}]") == []  # no random variable X of its own

    def test_each_member_is_lowered_as_one(self, wb):
        wb.exe(self.SAMPLE)
        lowered = wb.ir(r"\operatorname{Var}(X_{k+1} - X_k)")
        assert lowered.count("call prob.member") == 2
        reading = wb.explain(r"\operatorname{Var}(X_{k+1} - X_k)")[-1]
        meanings = {name.written: name.meaning for name in reading.names}
        assert meanings["X_{k+1}"] == "a member of the sample X, declared on line 1"

    def test_a_subscript_that_is_no_integer_is_refused(self, wb):
        wb.exe(self.SAMPLE)
        with pytest.raises(CompileError) as raised:
            wb.expr(r"\mathbb{E}[X_{k/2}]")
        assert raised.value.message == (
            "X_{k/2} is no member of the sample X: its members are numbered by integers, and "
            "k/2 is not written as one"
        )
