"""Exceptions raised by RelativisticPy.

Errors caused by user input carry the :class:`~relativisticpy.language.source.Span`
of source text they refer to, and render it as an underlined excerpt, with notes that
say why and a hint that says how to fix it::

    IndexNotationError: the free indices of a sum must agree: _{a} and _{b}
     --> <input>:3:1
      |
    3 | V_{a} + W_{b}
      | ^^^^^^^^^^^^^

Every error derives from :class:`RelativisticPyError`, and each stage of the
pipeline raises its own kind: the parser raises :class:`ParseError`, the
compiler :class:`CompileError` and the executor :class:`EvaluationError`. A mistake
made from Python raises one too, which is also the exception Python's own habits
expect: a name that is not defined, ``wb["T"]``, raises :class:`UnknownNameError`,
which is a ``KeyError``, and data that cannot be bound :class:`DataError`, which is a
``ValueError``. So ``except RelativisticPyError`` catches every error a user can cause.

A failure of RelativisticPy's own, never the user's, is an :class:`InternalError`,
with an invitation to report it; :func:`internal_error` makes one of any exception.
Failures of SymPy and NumPy on what they were given are translated by
:func:`translate`, so that their text is never shown as RelativisticPy's.

An error can be read as data, for programs and agents, with
:meth:`~RelativisticPyError.to_dict`.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from relativisticpy.language.source import Span

#: Where to report a failure of RelativisticPy's own.
ISSUES = "https://github.com/cottrellashley/relativisticpy/issues"

#: Set this environment variable to 1 to see the Python tracebacks of RelativisticPy's
#: own frames and of the libraries' failures behind an error, which are hidden otherwise.
DEBUG = "RELATIVISTICPY_DEBUG"


def debugging() -> bool:
    """Whether :data:`DEBUG` asks for every traceback in full."""
    return os.environ.get(DEBUG, "").strip() not in ("", "0")


class RelativisticPyError(Exception):
    """Base class for every error raised by RelativisticPy.

    - ``message`` says what is wrong, in one line, in the user's notation;
    - ``span`` says where: the :class:`~relativisticpy.language.source.Span` of source
      text it is about, with its ``line``, ``column`` and ``source.name``, or ``None``;
    - ``notes`` say why, or give context, such as where something was defined;
    - ``hint`` says how to fix it;
    - ``cause`` is the exception of a library behind it, such as SymPy's, if any.

    :meth:`render` gives the whole diagnostic, as the command line prints it, and
    :meth:`to_dict` the same as data.
    """

    #: The name the first line of the diagnostic gives the error: the class's own, or,
    #: for a failure of RelativisticPy's, ``internal error``.
    kind: ClassVar[str] = "RelativisticPyError"

    def __init_subclass__(cls, **options: Any) -> None:
        super().__init_subclass__(**options)
        cls.kind = cls.__dict__.get("kind", cls.__name__)

    def __init__(
        self,
        message: str,
        *,
        span: Span | None = None,
        hint: str | None = None,
        notes: Sequence[str] = (),
    ):
        message = " ".join(message.split("\n")).strip()  # one line, as every headline is
        super().__init__(message)
        self.message = message
        self.span = span
        self.hint = hint
        self.notes = list(notes)
        self.cause: BaseException | None = None

    def __str__(self) -> str:
        parts = [self.message]
        if self.span is not None:
            parts.append(self.span.render())
        parts.extend(f"note: {note}" for note in self.notes)
        if self.hint is not None:
            parts.append(f"hint: {self.hint}")
        return "\n".join(parts)

    def render(self) -> str:
        """The whole diagnostic, as the command line and notebooks show it: its kind and
        message, the excerpt of source it is about, its notes and its hint."""
        return f"{self.kind}: {self}"

    def to_dict(self) -> dict[str, Any]:
        """The error as data, for programs and agents::

            {"type": "IndexNotationError", "message": "...", "file": "paper.tex",
             "line": 12, "column": 5, "end_line": 12, "end_column": 21,
             "text": "R_{\\mu\\nu} = g_{\\mu}", "notes": [...], "hint": "..."}

        Lines and columns are the file's, from 1; the end is just after the text. Without
        a place in the source, they are ``None``. ``type`` is the class, which is stable,
        and ``kind`` the name the diagnostic gives it."""
        return {
            "type": type(self).__name__,
            "kind": self.kind,
            "message": self.message,
            **place(self.span),
            "notes": list(self.notes),
            "hint": self.hint,
        }

    def _render_traceback_(self) -> list[str]:
        """What IPython and Jupyter show for this error: the diagnostic, without
        RelativisticPy's frames or the library exceptions behind it (unless
        :data:`DEBUG` is set). A failure of RelativisticPy's own keeps its traceback."""
        if debugging() or isinstance(self, InternalError):
            import traceback

            frames = traceback.format_exception(type(self), self, self.__traceback__)
            return "".join(frames).rstrip("\n").split("\n")
        return self.render().split("\n")

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        """How IPython shows an error as a value, as ``step.error`` is: in full."""
        printer.text(self.render())


class ParseError(RelativisticPyError):
    """The source text is not valid RelativisticPy (raised by the lexer and parser)."""


class CompileError(RelativisticPyError):
    """A well-formed statement does not make sense: an unknown tensor, a function
    called with the wrong number of arguments, a metric without coordinates."""


class IndexNotationError(CompileError):
    """Tensor indices break the rules of index notation: mismatched free indices in
    a sum, an index repeated three times, two lower indices summed together."""


class NotComputedError(CompileError):
    """A statement RelativisticPy reads but does not compute: a relation such as ``≈``,
    ``∝`` or ``→``, a condition on a coordinate, ``\\theta = \\pi/2``, or a relation
    between differentials. A check calls it *not computed*, whatever its message says."""


class EvaluationError(RelativisticPyError):
    """A computation failed while running: a degenerate metric, an equation SymPy
    cannot solve."""


class MetricError(EvaluationError, ValueError):
    """A metric that cannot be one: not square, not symmetric, or degenerate. The same
    error whether the metric was written in the language or built in Python."""


class UsageError(RelativisticPyError, TypeError, ValueError):
    """A call from Python was given what it cannot take: ``wb.exe(123)``, a backend that
    does not exist, a time limit of no seconds. It is a ``TypeError`` and a
    ``ValueError`` too, as Python's own checks of arguments raise those."""


class DataError(CompileError, ValueError):
    """Data bound from Python, as by ``wb["w^a"] = ...``, cannot be bound as written: it
    is ragged, it does not have an axis for each index, or its labels are not the
    elements of its space."""


class UnknownNameError(RelativisticPyError, KeyError):
    """A name looked up from Python, as in ``wb["T"]``, that nothing defines. It is a
    ``KeyError``, as a missing key of a ``dict`` is."""


class FileError(RelativisticPyError, OSError):
    """A file cannot be read: it does not exist, or it is not UTF-8 text."""


class InternalError(RelativisticPyError):
    """A failure of RelativisticPy's own: never a mistake in the input. Its hint invites
    a report, and it keeps the Python traceback behind it, which the report needs. Every
    entry point reports it as it reports any error, and carries on."""

    kind = "internal error"


def internal_error(
    error: BaseException,
    *,
    span: Span | None = None,
    during: str | None = None,
    inside: str | None = None,
) -> InternalError:
    """An :class:`InternalError` for an exception RelativisticPy did not expect, raised
    while doing what ``during`` says (``"printing this result"``): the exception is
    quoted in a note and kept as the error's ``cause``. One raised ``inside`` a library,
    as ``"SymPy"``, says that it failed there, where the fault may be the library's.

    The command line and the interactive prompt use it so that a failure of their own
    ends neither the statement's neighbours nor the session::

        except RelativisticPyError as error:
            print(error.render(), file=sys.stderr)
        except Exception as error:  # RelativisticPy's own failure: reported, not raised
            print(internal_error(error).render(), file=sys.stderr)
    """
    if isinstance(error, InternalError):
        return error
    what = during or "computing this"
    if inside is not None:
        failure = InternalError(
            f"{inside} failed while {what}",
            span=span,
            notes=[f"{inside} says: {_quoted(error)}"],
            hint=f"this is a failure inside {inside}, not a mistake in the input: a bug in "
            f"{inside}, or in how RelativisticPy asked it; please report it, with the "
            f"input, at {ISSUES}",
        )
    else:
        failure = InternalError(
            f"RelativisticPy failed while {what}",
            span=span,
            notes=[f"Python says: {_quoted(error)}"],
            hint=f"this is a bug in RelativisticPy, not a mistake in the input; please "
            f"report it, with the input, at {ISSUES}",
        )
    failure.cause = error
    failure.__cause__ = error
    return failure


def translate(error: BaseException, *, span: Span | None = None) -> RelativisticPyError:
    """An exception from a computation as a :class:`RelativisticPyError`: itself if it
    is one; the failures of SymPy and NumPy RelativisticPy knows, in its own words; any
    other failure of a library on what it was given, as SymPy's or NumPy's, quoted in a
    note; a ``ValueError`` of RelativisticPy's own, which says what is wrong with the
    input, as an :class:`EvaluationError`; and anything else as an
    :class:`InternalError`, which says where it failed: in a library, or in
    RelativisticPy. The library's exception is kept as the error's ``cause``."""
    if isinstance(error, RelativisticPyError):
        if error.span is None:
            error.span = span
        return error
    known = known_failure(error)
    library = _library(error)
    failure: RelativisticPyError
    if known is not None:  # in RelativisticPy's words, with the library's as its cause
        message, hint = known
        failure = EvaluationError(message, span=span, hint=hint)
    elif library is not None and isinstance(error, _REFUSALS):
        failure = EvaluationError(
            f"{library} could not compute this",
            span=span,
            notes=[f"{library} says: {_quoted(error)}"],
            hint=f"if this should work, please report it at {ISSUES}",
        )
    elif isinstance(error, ValueError) and _raised_by_relativisticpy(error):
        failure = EvaluationError(str(error), span=span)  # its own words, about the input
    else:
        return internal_error(error, span=span, inside=library)
    failure.cause = error
    return failure


#: What a library raises when it declines what it was given, rather than when it fails:
#: a value it cannot take, or a computation it has no method for.
_REFUSALS = (ValueError, ArithmeticError, NotImplementedError)


def place(span: Span | None) -> dict[str, Any]:
    """Where a span is, as data: its file, the line and column it starts at and those
    just after it ends, from 1, and its text, as the file has them (before any macro is
    expanded). Each is ``None`` without a span."""
    if span is None:
        return dict.fromkeys(("file", "line", "column", "end_line", "end_column", "text"))
    source = span.source
    start, end = source.written_range(span.start, span.end)
    written = source.written if source.written is not None else source.text
    line, column = span.line, span.column
    end_line, end_column = source.location(span.end) if span.end > span.start else (line, column)
    return {
        "file": source.name,
        "line": line,
        "column": column,
        "end_line": end_line,
        "end_column": end_column,
        "text": written[start:end],
    }


# -- the failures of libraries ---------------------------------------------------------

#: What a matrix without an inverse is told, whatever kind of matrix it is.
NO_INVERSE = "this matrix has no inverse: its determinant is zero"
#: What a matrix that is not square is told, where only a square one will do.
NOT_SQUARE = "only a square matrix has a determinant or an inverse"

#: The failures of SymPy and NumPy RelativisticPy knows: the kinds of exception, the
#: text of their message, and what is wrong in RelativisticPy's words, with a hint.
_KNOWN: tuple[tuple[frozenset[str], re.Pattern[str], str, str | None], ...] = tuple(
    (frozenset(kinds.split()), re.compile(pattern, re.IGNORECASE | re.DOTALL), message, hint)
    for kinds, pattern, message, hint in (
        (
            "NonInvertibleMatrixError LinAlgError ValueError ZeroDivisionError",
            r"det == 0|not invertible|singular matrix|matrix is singular",
            NO_INVERSE,
            None,
        ),
        (
            "NonSquareMatrixError LinAlgError ValueError",
            r"nonsquarematrix|must be square|not square",
            NOT_SQUARE,
            None,
        ),
        (
            "ValueError",
            r"single argument is passed to subs",
            "subs needs a substitution, as in subs(R, M = 1)",
            "give the symbol and the value it takes, as in subs(R, M = 1)",
        ),
        (
            "NotImplementedError PolynomialError",
            r"multiple generators|no algorithms are implemented to solve",
            "SymPy cannot solve this equation in closed form",
            "simplify it first, or solve it numerically",
        ),
        (
            "ValueError",
            r"only work with functions of one variable",
            "dsolve solves for a function, written with its variable",
            "write the unknown as a function, as in dsolve(y''(x) + y(x) = 0, y(x))",
        ),
        (
            "ValueError",
            r"first variable cannot be a number",
            "cannot differentiate with respect to a number",
            "differentiate with respect to a symbol: a name defined as a number, as by "
            "x := 3, is that number",
        ),
        (
            "TypeError",
            r"bad argument type for abs\(\)",
            "the absolute value of a list is not defined",
            "for the length of a vector, bind it as data, as in w^a, and write |w|",
        ),
        (
            "PolynomialError GeneratorsError",
            r"contains an element of the set of generators",
            "this is not a polynomial in its variables, as it is used here",
            None,
        ),
    )
)


def known_failure(error: BaseException) -> tuple[str, str | None] | None:
    """The message and hint for a failure RelativisticPy knows, or ``None``."""
    name = type(error).__name__
    for kinds, pattern, message, hint in _KNOWN:
        if name in kinds and pattern.search(f"{name}: {error}"):
            return message, hint
    if isinstance(error, ZeroDivisionError):
        return "division by zero", None
    if isinstance(error, RecursionError):
        return (
            "this is nested too deeply to compute",
            "define its parts first, as in x := ..., then combine them",
        )
    if isinstance(error, OverflowError):
        return "a number here is too large to compute with", None
    if isinstance(error, MemoryError):
        return "there is not enough memory to compute this", None
    return None


def _raised_by_relativisticpy(error: BaseException) -> bool:
    """Whether RelativisticPy raised ``error`` itself, with a ``raise`` statement whose
    message says what is wrong with the input, rather than Python failing in its code."""
    import linecache

    traceback = error.__traceback__
    if traceback is None:
        return False
    while traceback.tb_next is not None:
        traceback = traceback.tb_next
    frame = traceback.tb_frame
    if not str(frame.f_globals.get("__name__", "")).startswith("relativisticpy"):
        return False
    filename, line = frame.f_code.co_filename, traceback.tb_lineno
    return any(
        linecache.getline(filename, number).lstrip().startswith("raise ")
        for number in range(max(1, line - 3), line + 1)
    )


def _library(error: BaseException) -> str | None:
    """The library that raised ``error``, by the module of the code that raised it:
    SymPy, NumPy, SciPy or JAX; ``None`` for RelativisticPy itself, or Python."""
    traceback = error.__traceback__
    module = ""
    while traceback is not None:
        module = str(traceback.tb_frame.f_globals.get("__name__", ""))
        traceback = traceback.tb_next
    module = module or type(error).__module__ or ""
    for prefix, name in (
        ("sympy", "SymPy"),
        ("mpmath", "SymPy"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("jax", "JAX"),
        ("jaxlib", "JAX"),
    ):
        if module == prefix or module.startswith(prefix + "."):
            return name
    return None


def _quoted(error: BaseException) -> str:
    """An exception as a note quotes it: its kind and its message, on one line."""
    message = " ".join(str(error).split())
    if len(message) > 300:
        message = message[:299] + "…"
    return f"{type(error).__name__}: {message}" if message else type(error).__name__
