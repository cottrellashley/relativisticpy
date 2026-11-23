"""The curriculum (``curriculum/``): every case is well formed, and every case recorded as
passing still passes. ``benchmarks/curriculum.py`` runs the rest and keeps the record."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest
import sympy

BENCHMARKS = Path(__file__).parent.parent / "benchmarks"


def _curriculum():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("curriculum", BENCHMARKS / "curriculum.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["curriculum"] = module
    spec.loader.exec_module(module)
    return module


curriculum = _curriculum()
CASES = curriculum.load()
PASSING = curriculum.passing()


def test_every_case_is_well_formed():
    assert CASES, "the curriculum has no cases"
    assert curriculum.problems(CASES) == []


def test_every_case_recorded_as_passing_exists():
    assert {case.id for case in CASES} >= PASSING


#: How long a case that passed may take here, in seconds: several times what the benchmark
#: gives it, as a runner of CI is slower than the machine its record was made on, and what
#: passed there must pass here.
PATIENT = 4 * curriculum.TIME_LIMIT


@pytest.mark.parametrize(
    "case", [case for case in CASES if case.id in PASSING], ids=lambda case: case.id
)
def test_a_case_that_passed_still_passes(case):
    result = curriculum.run(case, PATIENT)
    assert result.passed, result.describe()


def test_a_program_in_a_latex_fence_is_latex_in_every_statement(tmp_path, monkeypatch):
    # design/notation.md, 10: S_0 N(d_1), with no command of LaTeX, was plain text, where N
    # is numerical evaluation, so that Hull's call price was 8.42, silently
    monkeypatch.setattr(curriculum, "ROOT", tmp_path / "curriculum")
    topic = tmp_path / "curriculum" / "domain" / "topic.md"
    topic.parent.mkdir(parents=True)
    fence = "```"
    topic.write_text(
        "# Topic\n\n## A product of letters\n\n**Level 1** · `t.letters` · standard\n\n"
        f"{fence}latex\nx := 2\ny := 3\n6xy\n{fence}\n\n"
        f"{fence}python\nexpected = 36\n{fence}\n\n"
        "## The same in plain text\n\n**Level 1** · `t.plain` · standard\n\n"
        f"{fence}relativisticpy\nxy := 5\n6xy\n{fence}\n\n"
        f"{fence}python\nexpected = 30\n{fence}\n",
        encoding="utf-8",
    )
    letters, plain = curriculum.read(topic)
    assert (letters.latex, plain.latex) == (True, False)
    assert curriculum.run(letters).passed  # in LaTeX, 6 x y
    assert curriculum.run(plain).passed  # in plain text, xy is one name


# -- what counts as the answer ---------------------------------------------------------------

s, t, x = sympy.symbols("s t x")
y = sympy.Function("y")
SINC = sympy.sin(sympy.pi * s) / (sympy.pi * s)


def test_a_piecewise_that_fills_a_removable_point_is_the_answer():
    filled = sympy.Piecewise((SINC, sympy.Ne(s, 0)), (1, True))
    line = (s > -sympy.oo) & (s < sympy.oo) & sympy.Ne(s, 0)
    assert curriculum.equal(filled, SINC)
    assert curriculum.equal(sympy.Piecewise((SINC, line), (1, True)), SINC)


def test_a_piecewise_with_another_value_or_condition_is_not():
    assert not curriculum.equal(sympy.Piecewise((SINC, sympy.Ne(s, 0)), (0, True)), SINC)
    assert not curriculum.equal(sympy.Piecewise((SINC, s > 0), (1, True)), SINC)
    assert not curriculum.equal(sympy.Piecewise((SINC, s > -sympy.oo), (0, True)), SINC)


def test_equations_are_compared_by_the_difference_of_their_sides():
    solved = sympy.Eq(y(t), sympy.Symbol("C1") * sympy.exp(t))
    assert curriculum.equal(sympy.Eq(y(t), sympy.exp(t) * sympy.Symbol("C1")), solved)
    assert curriculum.equal(sympy.Eq(sympy.Symbol("C1") * sympy.exp(t), y(t)), solved)
    assert curriculum.equal(sympy.Eq(y(t) - sympy.Symbol("C1") * sympy.exp(t), 0), solved)
    assert not curriculum.equal(sympy.Eq(y(t), sympy.Symbol("C1") * sympy.exp(2 * t)), solved)


def test_derivatives_of_unknown_functions_are_compared_as_values_of_their_own():
    # cm.lagrangian.index-form: comparing failed, "Cannot convert expression to float"
    theta = sympy.Function("theta")(t)
    given = sympy.Derivative(theta, (t, 2))
    expected = sympy.Derivative(theta, (t, 2)) - sympy.sin(theta) * sympy.cos(theta)
    assert not curriculum.equal(given, expected)
    assert curriculum.equal(sympy.sin(theta) ** 2 * given, given - sympy.cos(theta) ** 2 * given)


def test_a_result_left_undone_is_not_its_value():
    assert not curriculum.equal(sympy.Integral(x, (x, 0, 1)), sympy.Rational(1, 2))
    assert curriculum.equal(sympy.Integral(x, (x, 0, 1)), sympy.Integral(x, (x, 0, 1)))


def test_nan_is_no_answer():
    # compared at random points, nan was within every tolerance, as no number is beyond one
    assert not curriculum.equal(sympy.nan, 1 / (4 * sympy.Symbol("M")))
    assert not curriculum.equal(sympy.nan + x, x)
    assert not curriculum.equal(sympy.nan, sympy.Integer(1))


def test_solutions_are_compared_in_any_form_and_order():
    p, K, w = sympy.symbols("p K omega")
    third = sympy.Rational(2, 3)
    assert curriculum.equal({p: third}, [third])
    assert curriculum.equal([{p: third}], [third])
    assert curriculum.equal({K: 8, w: sympy.sqrt(3)}, [(8, sympy.sqrt(3))])
    assert curriculum.equal([{w: sympy.sqrt(3), K: 8}], [{K: 8, w: sympy.sqrt(3)}])
    assert curriculum.equal([(1, 2), (3, 4)], [{K: 3, w: 4}, {K: 1, w: 2}])
    assert curriculum.equal([{K: 1}, {K: 2}], [2, 1])


def test_other_solutions_are_not():
    p, K, w = sympy.symbols("p K omega")
    assert not curriculum.equal({p: sympy.Rational(1, 3)}, [sympy.Rational(2, 3)])
    assert not curriculum.equal([{K: 1, w: 2}], [{K: 2, w: 1}])  # by name, not by order
    assert not curriculum.equal([{K: 1}], [1, 2])  # a solution missing
    assert not curriculum.equal([{K: 1}, {K: 1}], [1, 2])  # one solution twice
    assert not curriculum.equal({p: 1}, [(1, 2)])


LEFT = "an integral is left as written: SymPy could not compute it"
RELU = (
    r"\operatorname{ReLU} is not a function RelativisticPy knows, so "
    r"\operatorname{ReLU}(3) is an unknown function"
)


def test_a_value_kept_as_written_where_a_note_says_so_is_kept():
    unknown = sympy.Function("ReLU")
    assert curriculum.kept(sympy.Integral(x**x, (x, 0, 1)), 0.7834, [LEFT]) == LEFT
    assert curriculum.kept(unknown(-2) + unknown(3), 3, [RELU]) == RELU
    det = sympy.Function("det")(6 * sympy.Symbol("I_5") - sympy.Symbol("J_5"))
    unknown_value = (
        r"\det\left(6 I_5 - J_5\right) is left as an unknown value: nothing defines I_{5} "
        "and J_{5}, which may be matrices of any size, not numbers"
    )
    assert curriculum.kept(sympy.Eq(det, 1296), True, [unknown_value]) == unknown_value


def test_a_function_called_inside_a_sum_kept_as_written_is_kept_with_it():
    # sig.random.autocorrelation: u, the unit step, is unknown, inside the sum left undone
    a, n, lag = sympy.symbols("a n l")
    u = sympy.Function("u")
    terms = a**n * a ** (n - lag) * u(n) * u(n - lag)
    left = "a sum is left as written: SymPy found no closed form for it"
    value = sympy.Sum(terms, (n, -sympy.oo, sympy.oo))
    assert curriculum.kept(value, a ** abs(lag) / (1 - a**2), [left]) == left
    assert curriculum.kept(value + u(lag), a ** abs(lag) / (1 - a**2), [left]) is None


def test_a_value_no_note_says_is_kept_is_given_as_the_answer():
    integral = sympy.Integral(x**x, (x, 0, 1))
    assert curriculum.kept(integral, 0.7834, []) is None
    assert curriculum.kept(integral, 0.7834, ["assuming a > 0; the program does not say"]) is None
    series = sympy.Sum(1 / (x * sympy.log(x)), (x, 2, sympy.oo))
    assert curriculum.kept(series, 1, ["assuming n > 0, as books take a scale to be"]) is None
    assert curriculum.kept(sympy.Function("f")(3), 3, [RELU]) is None  # f in "function"
    assert curriculum.kept(x + 1, 3, [LEFT]) is None  # nothing kept: a value, and wrong
    # an unknown function the answer holds too is none the value keeps
    assert curriculum.kept(y(t) + 1, y(t), [f"y is not defined, so {y(t)} is unknown"]) is None


def _case(program: str, oracle: str, *, latex: bool = True):  # type: ignore[no-untyped-def]
    return curriculum.Case(
        id="test.case",
        title="a case of the tests",
        domain="analysis",
        topic="test",
        level=1,
        source="standard",
        program=program,
        oracle=oracle,
        path=curriculum.ROOT / "analysis" / "test.md",
        line=1,
        latex=latex,
    )


def test_a_program_that_keeps_a_function_nothing_defines_is_kept_not_wrong():
    # num.rounding.cancellation: fl is defined where it is used (design/notation.md, 26)
    program = r"""u := 2^{-53}
x := 10^{-8}
\operatorname{fl}\left(\frac{1 - \cos x}{x^2}\right)"""
    result = curriculum.run(_case(program, "expected = 0.0"))
    assert result.outcome == "kept", result.describe()
    assert r"\operatorname{fl} is not a function RelativisticPy knows" in result.reason


def test_a_program_that_gives_another_value_is_wrong():
    result = curriculum.run(_case("x := 2\nx + 1", "expected = 4"))
    assert result.outcome == "wrong", result.describe()


def test_a_program_written_in_latex_is_given_as_latex_in_every_statement(tmp_path):
    # la.eigen.exponential: e^{At} has no backslash, and given as plain text At was one name
    program = "A := \\begin{pmatrix} 0 & 1 \\\\ -1 & 0 \\end{pmatrix}\ne^{At}"
    oracle = "t = symbols('t')\nexpected = Matrix([[cos(t), sin(t)], [-sin(t), cos(t)]])"
    assert curriculum.run(_case(program, oracle)).outcome == "passed"
    case = "## {0}\n\n**Level 1** · `t.{0}` · standard\n\n```{1}\nx\n```\n\n```python\nx = 1\n```\n"
    topic = tmp_path / "topic.md"
    topic.write_text(
        "# A topic\n\n" + case.format("a", "latex") + case.format("b", "relativisticpy")
    )
    read = [(case.id, case.latex) for case in curriculum.read(topic)]
    assert read == [("t.a", True), ("t.b", False)]


def test_a_column_is_the_vector_a_list_writes():
    column = sympy.Matrix([[sympy.Rational(4, 5)], [0], [sympy.Rational(1, 5)]])
    assert curriculum.equal(column, [sympy.Rational(4, 5), 0, sympy.Rational(1, 5)])
    assert curriculum.equal(column.T, [sympy.Rational(4, 5), 0, sympy.Rational(1, 5)])
    assert not curriculum.equal(sympy.ones(2, 3), sympy.ones(3, 2).tolist())


# -- under several hash seeds ----------------------------------------------------------------


def test_the_cases_whose_outcome_depends_on_the_hash_seed_are_found():
    outcomes = {
        "0": {"a": ("passed", ""), "b": ("passed", ""), "c": ("refused", "why")},
        "1": {"a": ("passed", ""), "b": ("refused", "why"), "c": ("refused", "why")},
        "2": {"a": ("passed", ""), "b": ("unfinished", "stopped")},  # stopped before c
    }
    assert curriculum.differing(outcomes) == {
        "b": {"passed": ["0"], "refused": ["1"], "unfinished": ["2"]}
    }


def test_hash_seeds_are_two_or_more_whole_numbers():
    assert curriculum._seeds("0,1, 2") == ["0", "1", "2"]
    for wrong in ("0", "a,b", "0,-1"):
        with pytest.raises(argparse.ArgumentTypeError, match="two or more whole numbers"):
            curriculum._seeds(wrong)
