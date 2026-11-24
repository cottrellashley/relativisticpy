"""The core of errors: how they read, as text and as data, and the failures of libraries
and of RelativisticPy itself, translated."""

import json
import pickle

import numpy as np
import pytest
import sympy

from relativisticpy import Workbook
from relativisticpy.diffgeom import curvature
from relativisticpy.errors import (
    DEBUG,
    CompileError,
    DataError,
    EvaluationError,
    FileError,
    IndexNotationError,
    InternalError,
    MetricError,
    NotComputedError,
    RelativisticPyError,
    UnknownNameError,
    UsageError,
    internal_error,
    translate,
)
from relativisticpy.language import Source, Span
from relativisticpy.tensors.equation import UndecidedError

DEGENERATE = "Coordinates := [x, y]\ng_{ab} := [[1, 1], [1, 1]]\nR"


def raised(program: str) -> RelativisticPyError:
    with pytest.raises(RelativisticPyError) as error:
        Workbook().exe(program)
    return error.value


class TestReading:
    def test_the_diagnostic_names_its_kind(self):
        error = raised(r"G_{\mu\nu} = 8\pi T_{\mu\nu\rho}")
        assert error.kind == "IndexNotationError"
        text = error.render()
        assert text.startswith("IndexNotationError: the two sides have different free indices")
        assert "\n --> <input>:1:1\n" in text

    def test_ipython_shows_the_diagnostic_and_nothing_else(self, monkeypatch):
        # API-04: no internal frames, and no chained library traceback
        monkeypatch.delenv(DEBUG, raising=False)
        error = raised(DEGENERATE)
        lines = error._render_traceback_()
        assert lines == error.render().split("\n")
        assert lines[0] == ("MetricError: the metric is degenerate: its determinant is zero")
        assert not any("Traceback" in line for line in lines)

    def test_debugging_shows_the_traceback(self, monkeypatch):
        monkeypatch.setenv(DEBUG, "1")
        lines = raised(DEGENERATE)._render_traceback_()
        assert lines[0].startswith("Traceback (most recent call last)")

    def test_an_internal_error_keeps_its_traceback_for_a_report(self):
        try:
            {}["missing"]
        except KeyError as error:
            failure = internal_error(error)
        assert failure.render().startswith("internal error: RelativisticPy failed")
        assert failure.cause.__class__ is KeyError
        assert "please report it" in failure.hint

    def test_a_shown_error_is_shown_in_full(self):
        # ERR-32: an error displayed as a value, as step.error is, keeps its hint
        error = raised("R_{ab}")

        class Printer:
            text = ""

            def text(self, value):  # noqa: F811 - IPython's printer protocol
                Printer.text = value

        error._repr_pretty_(Printer(), False)
        assert "hint: declare them first" in Printer.text

    def test_a_message_is_one_line_without_padding(self):
        # CLI-32: no message starts with a line break
        assert EvaluationError("\nWhen a single argument\nis passed").message == (
            "When a single argument is passed"
        )

    def test_errors_survive_pickling_as_from_a_worker(self):
        error = pickle.loads(pickle.dumps(UnknownNameError("T is not defined", hint="h")))
        assert (str(error), error.hint) == ("T is not defined\nhint: h", "h")


class TestData:
    def test_an_error_is_data(self):
        # ERR-33: what a person reads, a program can read
        data = raised(r"G_{\mu\nu} = 8\pi T_{\mu\nu\rho}").to_dict()
        assert json.loads(json.dumps(data)) == data
        assert data["type"] == "IndexNotationError"
        assert (data["file"], data["line"], data["column"]) == ("<input>", 1, 1)
        assert data["text"] == r"G_{\mu\nu} = 8\pi T_{\mu\nu\rho}"
        assert data["end_column"] == 1 + len(data["text"])
        assert data["hint"] is None
        assert data["notes"] == []

    def test_an_error_nowhere_has_no_place(self):
        data = UsageError("exe takes the program as a str, not int").to_dict()
        assert (data["file"], data["line"], data["text"]) == (None, None, None)

    def test_positions_are_the_files_through_macros(self):
        source = Source("x + y", "paper.tex", origin=(12, 5))
        data = CompileError("no", span=Span(source, 4, 5)).to_dict()
        assert (data["line"], data["column"], data["text"]) == (12, 9, "y")


class TestKinds:
    """API-20: every error from Python is a RelativisticPyError, and also what Python's
    own habits expect."""

    @pytest.mark.parametrize(
        ("kind", "bases"),
        [
            (UsageError, (TypeError, ValueError)),
            (DataError, (CompileError, ValueError)),
            (UnknownNameError, (KeyError,)),
            (FileError, (OSError,)),
            (MetricError, (EvaluationError, ValueError)),
            (NotComputedError, (CompileError,)),
        ],
    )
    def test_each_is_also_the_exception_python_expects(self, kind, bases):
        error = kind("message")
        assert isinstance(error, RelativisticPyError)
        assert all(isinstance(error, base) for base in bases)
        assert str(error) == "message"  # not KeyError's quotes

    def test_internal_errors_are_named_so(self):
        assert InternalError.kind == "internal error"
        assert IndexNotationError.kind == "IndexNotationError"


class TestTranslation:
    """ERR-14, ERR-29: library text is never shown as RelativisticPy's."""

    @pytest.mark.parametrize(
        ("program", "message"),
        [
            (
                r"\begin{pmatrix}1&2\\2&4\end{pmatrix}^{-1}",
                "this matrix has no inverse: its determinant is zero",
            ),
            ("|[[1, 2], [3, 4]]|", "the absolute value of a list is not defined"),
            (r"solve(\sin(x) = x^3 + e^x, x)", "SymPy cannot solve this equation in closed form"),
            ("x := 3\ndiff(sin(x), x)", "cannot differentiate with respect to a number"),
        ],
    )
    def test_known_failures_are_said_in_relativisticpys_words(self, program, message):
        error = raised(program)
        assert type(error) is EvaluationError
        assert error.message == message
        assert error.__cause__ is None  # the library's exception is not chained ...
        assert error.cause is not None  # ... but kept

    @pytest.mark.parametrize(
        ("program", "message"),
        [
            (r"\det([1, 2])", "a determinant needs a square matrix, but this is a list of 2"),
            (
                "subs(x^2, x)",
                "subs takes an expression and what to put for what, as in subs(x^2, x, 3), "
                "or rules, as in subs(x^2, x = 3)",
            ),
            (
                "dsolve(y'' + y, y)",
                "dsolve solves for a function written with its variable, as in "
                "dsolve(f''(x) + f(x) = 0, f(x)), not for y",
            ),
        ],
    )
    def test_builtins_say_what_they_take_before_any_library_runs(self, program, message):
        error = raised(program)
        assert type(error) is EvaluationError
        assert error.message == message

    def test_one_message_for_every_kind_of_singular_matrix(self):
        symbolic = raised(r"\begin{pmatrix}1&2\\2&4\end{pmatrix}^{-1}")
        wb = Workbook()
        wb["S_{ab}"] = np.array([[1.0, 2.0], [2.0, 4.0]])
        with pytest.raises(EvaluationError) as numerical:
            wb.expr(r"(S^{-1})^{ab}")
        assert symbolic.message == numerical.value.message
        wb["S_{ab}"] = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0]])
        with pytest.raises(EvaluationError, match="and this one is 2×3"):
            wb.expr(r"(S^{-1})^{ab}")

    def test_a_failure_of_relativisticpys_own_is_an_internal_error(self, monkeypatch):
        # CLI-32: a Python failure inside RelativisticPy is not the user's mistake
        from dataclasses import replace

        from relativisticpy.runtime import registry

        def broken(*args):
            raise AttributeError("'list' object has no attribute 'is_Float'")

        entry = replace(registry.lookup("factorial2"), func=broken)
        monkeypatch.setitem(registry._ENTRIES, "factorial2", entry)
        error = raised("factorial2(5)")
        assert isinstance(error, InternalError)
        assert error.notes == [
            "Python says: AttributeError: 'list' object has no attribute 'is_Float'"
        ]

    def test_a_library_that_fails_inside_itself_says_so(self):
        # SymPy's solve can fail in its own checks, as when a NaN is compared
        try:
            _ = sympy.nan < 1
        except TypeError as error:
            failure = translate(error)
        assert isinstance(failure, InternalError)
        assert failure.message == "SymPy failed while computing this"
        assert failure.notes == ["SymPy says: TypeError: Invalid NaN comparison"]
        assert failure.hint is not None
        assert "a failure inside SymPy, not a mistake in the input" in failure.hint

    def test_relativisticpys_own_errors_keep_their_words(self):
        # the metric's check raises its own error, a ValueError too, which stays as it is
        with pytest.raises(ValueError, match="symmetric") as error:
            curvature.check_metric(sympy.ImmutableMatrix([[1, 2], [3, 4]]))
        failure = translate(error.value)
        assert type(failure) is MetricError
        assert isinstance(failure, EvaluationError)
        assert failure.message.startswith("a metric must be symmetric")

    def test_a_library_that_declines_is_quoted(self):
        try:
            sympy.Matrix([[1, 2]]).inv()
        except Exception as error:
            failure = translate(error)
        assert type(failure) is EvaluationError

    def test_pythons_own_failures_in_relativisticpy_are_internal(self):
        try:
            int("x")
        except ValueError as error:
            failure = translate(error)
        assert isinstance(failure, InternalError)

    @pytest.mark.parametrize(
        ("error", "message"),
        [
            (ZeroDivisionError("division by zero"), "division by zero"),
            (RecursionError("maximum recursion depth exceeded"), "this is nested too deeply"),
        ],
    )
    def test_arithmetic_failures(self, error, message):
        assert translate(error).message.startswith(message)


class TestRelations:
    """An inequality SymPy cannot compare, which it says by failing, is refused, or not
    decided, in RelativisticPy's words: a side that has no value, as ∞ - ∞ and 1/0 have
    none, is neither greater nor less; and complex numbers have no order."""

    @pytest.mark.parametrize(
        ("program", "message"),
        [
            (r"\infty - \infty < 1", "the left side is undefined here, as 0/0 is"),
            (r"1 \geq 0 \cdot \infty", "the right side is undefined here, as 0/0 is"),
            (r"x := \infty - \infty" "\n" r"x > 2", "the left side is undefined here, as 0/0 is"),
            (r"\frac{1}{0} > 2", "the left side is infinite here: a division by zero"),
            (r"\Gamma(0) \leq 1", "the left side is infinite here: a division by zero"),
        ],
    )
    def test_a_side_with_no_value_is_neither_greater_nor_less(self, program, message):
        # as a check says it of an equation: not the jargon of complex infinity
        error = raised(program)
        assert type(error) is UndecidedError
        assert error.message == message

    @pytest.mark.parametrize(
        ("program", "message"),
        [
            (r"\infty - \infty = 0", "the left side is undefined here, as 0/0 is"),
            (r"\infty - \infty = \infty - \infty", "the left side is undefined here, as 0/0 is"),
            (r"\infty - \infty \neq 1", "the left side is undefined here, as 0/0 is"),
            (r"0 == \frac{0}{0}", "the right side is undefined here, as 0/0 is"),
            (r"\frac{1}{0} = 2", "the left side is infinite here: a division by zero"),
        ],
    )
    def test_a_side_with_no_value_is_neither_equal_nor_unequal(self, program, message):
        # each had a truth value: ∞ - ∞ = 0 was false, ∞ - ∞ = ∞ - ∞ true, ∞ - ∞ ≠ 1 true
        error = raised(program)
        assert type(error) is UndecidedError
        assert error.message == message

    @pytest.mark.parametrize(
        ("program", "message"),
        [
            # a sentence of a test, read as a program, failed inside SymPy
            (
                "cannot add a scalar and an array",
                "a add cannot scalar is neither true nor false, so and cannot join it",
            ),
            ("2 x or y > 0", "2x is neither true nor false, so or cannot join it"),
            ("not (2 x)", "2x is neither true nor false, so not cannot negate it"),
        ],
    )
    def test_a_connective_joins_what_is_true_or_false(self, program, message):
        error = raised(program)
        assert type(error) is EvaluationError
        assert error.message == message
        assert Workbook().expr("x > 0 and y") == sympy.And(sympy.Symbol("x") > 0, sympy.Symbol("y"))

    def test_numbers_that_are_not_real_have_no_order(self):
        error = raised("1 + i > 0")
        assert type(error) is EvaluationError
        assert error.message == (
            "1 + i, on the left side, is not real, and only real numbers are ordered"
        )
        assert Workbook().expr("|1 + i| > 0") is sympy.true  # as its hint says


class TestClaims:
    """An equation, an inequality or a truth value is no number: arithmetic with one, which
    failed inside RelativisticPy, is refused, saying so."""

    @pytest.mark.parametrize(
        ("program", "message"),
        [
            # arXiv 2611.26576, line 2573, for c_u where ε_u² = 1
            (
                r"c_u (\varepsilon_u^2 = 1) = \frac{2}{3} - \frac{2}{15} (2 - q)^2",
                "an equation, ε_u² = 1, is no number, and has no arithmetic",
            ),
            (r"-(x = 1)", "an equation, x = 1, is no number, and has no arithmetic"),
            (r"(x = 1)^2", "an equation, x = 1, is no number, and has no arithmetic"),
            (r"(x > 0) x", "a relation, x > 0, is no number, and has no arithmetic"),
            (r"(1 < 2) \cdot 3", "a truth value, True, is no number, and has no arithmetic"),
            (
                "x := [1, 2]\n(x = [1, 2]) \\cdot 2",
                "an equation is no number, and has no arithmetic",
            ),
        ],
    )
    def test_a_claim_is_no_number(self, program, message):
        error = raised(program)
        assert type(error) is EvaluationError
        assert error.message == message

    def test_a_function_whose_value_is_an_equation_is_no_number(self):
        # arXiv 2611.26585 defines H_1 by a chain, on line 363, so that its value is an
        # equation, which line 404 multiplies
        error = raised(
            r"\mathcal{H}_1(\lambda) := \mathcal{H}_c + \mathcal{N}(\lambda) \mathcal{C}"
            r" = \mathcal{N}(\lambda) \mathcal{C}"
            "\n"
            r"\frac{d\lambda}{d\tilde{\lambda}} \mathcal{H}_1(\lambda)"
        )
        assert type(error) is EvaluationError
        assert error.message.startswith("an equation, C N(λ) + H_c = C N(λ), is no number")

    def test_what_a_relation_in_arithmetic_stands_for_is_written_with_cases(self):
        error = raised(r"(x > 0) x")
        assert error.hint is not None
        written = error.hint.partition(", as ")[2]
        x = sympy.Symbol("x")
        found = sympy.piecewise_fold(Workbook().expr(f"{written} x"))
        assert found == sympy.Piecewise((x, x > 0), (0, True))


class TestWhereANumberIsTaken:
    """A claim, or several values, given where a number is taken, failed inside SymPy or
    RelativisticPy: each is refused, saying what it is, as arithmetic with one is."""

    @pytest.mark.parametrize(
        ("program", "message"),
        [
            ("(x = 1) < 1", "an equation, x = 1, is no number, and only numbers are ordered"),
            (r"\sum_{n=1}^{3} (n > 1)", "a relation, n > 1, is no number, and a sum adds numbers"),
            (
                r"\prod_{n=1}^{(x = 2)} n",
                "an equation, x = 2, is no number, and a product multiplies numbers",
            ),
            (
                r"\lim_{x \to 0} (x > 0)",
                "a relation, x > 0, is no number, and a limit is taken of a number, at a number",
            ),
            (
                r"\left. x^2 \right|_{x = (1 < 2)}",
                "a truth value, True, is no number, and an evaluation bar puts numbers for names",
            ),
            (r"\log(x > 0)", "log takes numbers and expressions, not a relation"),
            (r"\int_0^1 (x > 0) \, dx", "integrate takes numbers and expressions, not a relation"),
            (
                "E := x^2 = 4\n\\min_{x} E",
                "an equation, x² = 4, is no number, and \\min_{x} is of a number",
            ),
            (
                "X \\sim \\mathcal{N}(0, 1)\nf_X((1 < 2))",
                "a truth value, True, is no number, and the density of X is taken at a number",
            ),
        ],
    )
    def test_a_claim(self, program, message):
        error = raised(program)
        assert type(error) is EvaluationError
        assert error.message == message

    @pytest.mark.parametrize(
        ("program", "message"),
        [
            (
                r"\lim_{x \to [1, 2]} x^2",
                "[1, 2] is several values, and a limit is taken at one point",
            ),
            (
                r"\left. x^2 \right|_{x = (1, 2)}",
                "(1, 2) is several values, and an evaluation bar puts one value for each name",
            ),
            (
                r"\int_0^{[1, 2]} x \, dx",
                "the upper limit of the integral is one value, and [1, 2] is several",
            ),
            (r"\log((1, 2))", "log takes numbers and expressions, not a tuple"),
            (r"\binom{(1, 2)}{2}", "binomial takes numbers and expressions, not a tuple"),
            (
                r"\nabla \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}",
                "\\nabla of [[1, 2], [3, 4]] would be a tensor of more indices than a vector "
                "has, which is not taken yet",
            ),
            (
                r"\sum_{n=1}^{\pm 3} n",
                "[3, −3] is several values, and a limit of a sum is one value",
            ),
            (
                r"\prod_{n=1}^{[2, 3]} n",
                "[2, 3] is several values, and a limit of a product is one value",
            ),
        ],
    )
    def test_several_values(self, program, message):
        error = raised(program)
        assert type(error) is EvaluationError
        assert error.message == message


def test_a_metric_or_tensor_built_from_python_raises_relativisticpys_errors():
    # still ValueErrors, as Python code expects, and RelativisticPy's own kinds
    from relativisticpy import Metric, Tensor

    with pytest.raises(MetricError, match="non-degenerate"):
        Metric([[0, 0], [0, 0]], ["t", "x"])
    with pytest.raises(MetricError, match="needs 2 coordinates"):
        Metric([[-1, 0], [0, 1]], ["t", "x", "y"])
    with pytest.raises(DataError, match="must be square"):
        Tensor([[1, 2, 3], [4, 5, 6]], "_{ab}")
    with pytest.raises(ValueError, match="must be square"):
        Tensor([[1, 2, 3], [4, 5, 6]], "_{ab}")
