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
        assert lines[0] == (
            "MetricError: a metric must be non-degenerate, but its determinant is zero"
        )
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
            (r"\det([1, 2])", "only a square matrix has a determinant or an inverse"),
            (
                r"\begin{pmatrix}1&2\\2&4\end{pmatrix}^{-1}",
                "this matrix has no inverse: its determinant is zero",
            ),
            ("|[1, 2]|", "the absolute value of a list is not defined"),
            (
                "Coordinates := [t, r, \\theta, \\phi]\ng_{\\mu\\nu} := Schwarzschild(M)\n"
                "subs(R, M)",
                "subs needs a substitution, as in subs(R, M = 1)",
            ),
            (r"solve(\sin(x) = x^3 + e^x, x)", "SymPy cannot solve this equation in closed form"),
            ("dsolve(y'' + y, y)", "dsolve solves for a function, written with its variable"),
            ("x := 3\ndiff(sin(x), x)", "cannot differentiate with respect to a number"),
        ],
    )
    def test_known_failures_are_said_in_relativisticpys_words(self, program, message):
        error = raised(program)
        assert type(error) is EvaluationError
        assert error.message == message
        assert error.__cause__ is None  # the library's exception is not chained ...
        assert error.cause is not None  # ... but kept

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

    def test_a_failure_of_relativisticpys_own_is_an_internal_error(self):
        # CLI-32: 'list' object has no attribute 'is_Float' is not the user's mistake
        error = raised("Matrix([[1,2],[3,4]])")
        assert isinstance(error, InternalError)
        assert error.notes == [
            "Python says: AttributeError: 'list' object has no attribute 'is_Float'"
        ]

    def test_relativisticpys_own_value_errors_keep_their_words(self):
        # a ValueError RelativisticPy raises says what is wrong with the input
        with pytest.raises(ValueError, match="symmetric") as error:
            curvature.check_metric(sympy.ImmutableMatrix([[1, 2], [3, 4]]))
        failure = translate(error.value)
        assert type(failure) is EvaluationError
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
