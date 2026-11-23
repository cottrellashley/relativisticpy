r"""Equivalences: two spellings of one thing must agree, and with its value computed
without RelativisticPy, by SymPy or NumPy.

A silent wrong answer is often one spelling read otherwise than another: ``\frac{dy}{dx}``
refused where ``\frac{d}{dx} y`` gives 0, a sum that disagrees with its terms written out.
Each pair here is held to give, both ways, the value SymPy or NumPy gives, or to be refused
both ways. A pair that does not agree yet is marked to fail, strictly, with the finding of
the review of the reading it comes from: its fix removes the mark, and a pair that comes to
agree without one says so, as a mark kept on a passing test fails the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
import scipy.linalg
import scipy.special
import scipy.stats
import sympy

from relativisticpy import Workbook
from relativisticpy.errors import RelativisticPyError

x, y, hbar = sympy.symbols("x y hbar")

#: What a pair refused both ways has in the place of a value.
REFUSED = object()


@dataclass(frozen=True)
class Pair:
    """Two spellings, ``left`` and ``right``, of what a program defines in ``setup``, with
    the Python objects a workbook reads by name; and the value both must have, ``expected``,
    computed apart from them, or :data:`REFUSED`."""

    setup: str
    left: str
    right: str
    expected: Any
    scope: dict[str, Any] = field(default_factory=dict)


def value(pair: Pair, program: str) -> Any:
    """What a program gives after the pair's setup: its value, or the error it is refused
    with."""
    with Workbook(scope=pair.scope or None) as wb:
        wb.exe(pair.setup)
        try:
            found = wb.expr(program)
        except RelativisticPyError as error:
            return error
    return found.to_numpy() if hasattr(found, "to_numpy") else found


def agree(pair: Pair) -> None:
    """Both spellings give the expected value, or are both refused."""
    left, right = value(pair, pair.left), value(pair, pair.right)
    if pair.expected is REFUSED:
        assert isinstance(left, RelativisticPyError), f"{pair.left} gives {left}"
        assert isinstance(right, RelativisticPyError), f"{pair.right} gives {right}"
        return
    for program, found in ((pair.left, left), (pair.right, right)):
        assert not isinstance(found, RelativisticPyError), f"{program}: {found}"
        assert same(found, pair.expected), f"{program} gives {found}, not {pair.expected}"


def same(found: Any, expected: Any) -> bool:
    """Whether a value is the one expected: numbers and arrays of them to rounding error,
    and expressions where their difference simplifies to 0."""
    if isinstance(expected, np.ndarray | float | int) or isinstance(found, np.ndarray):
        try:
            numbers = np.asarray(found, dtype=complex)
        except (TypeError, ValueError):
            return False
        rounding = 1e-12 * float(np.max(np.abs(expected)))
        return numbers.shape == np.shape(expected) and bool(
            np.allclose(numbers, expected, rtol=1e-9, atol=rounding)
        )
    try:
        return bool(sympy.simplify(sympy.sympify(found) - expected) == 0)
    except (sympy.SympifyError, TypeError):
        return False


def case(pair: Pair, name: str, finding: str | None = None) -> Any:
    """A pair as a test case, expected to fail where the review of the reading found the
    spellings disagree, by its finding."""
    marks = []
    if finding is not None:
        reason = f"the review of the reading, {finding}"
        marks.append(pytest.mark.xfail(reason=reason, strict=True))
    return pytest.param(pair, id=name, marks=marks)


# -- a derivative written two ways ------------------------------------------------------


DERIVATIVES = [
    case(
        Pair(r"y(x) := \sin(x^2)", r"\frac{dy}{dx}", r"\frac{d}{dx} y",
             sympy.diff(sympy.sin(x**2), x)),
        "of-a-function-defined",
    ),
    case(
        Pair("y(x) := x^3", r"\frac{dy}{dx}", r"\frac{d}{dx} y(x)", sympy.diff(x**3, x)),
        "of-a-call",
    ),
    case(  # y is tied to x by the curve: neither is 0, and both are refused (guard 6)
        Pair("x^3 + y^3 = 6 x y", r"\frac{dy}{dx}", r"\frac{d}{dx} y", REFUSED),
        "of-a-name-tied-by-an-equation",
        "22: guard 6 catches dy/dx but not d/dx y, which gives 0",
    ),
]  # fmt: skip


@pytest.mark.parametrize("pair", DERIVATIVES)
def test_dy_dx_is_d_dx_of_y(pair):
    agree(pair)


# -- a finite sum, and its terms written out --------------------------------------------


SUMS = [
    case(
        Pair("", r"\sum_{n=1}^{4} \frac{1}{n^2}", r"1 + \frac{1}{4} + \frac{1}{9} + \frac{1}{16}",
             sum(sympy.Rational(1, n**2) for n in range(1, 5))),
        "of-numbers",
    ),
    case(
        Pair("x := [1, 2, 3]", r"\sum_{i=1}^{3} x_i", r"\sum_{n=1}^{3} x_n",
             int(np.sum([1, 2, 3]))),
        "of-a-list",
    ),
    case(
        Pair("a_0 := 1\na_n := \\frac{1}{n}", r"\sum_{n=0}^{3} a_n", "a_0 + a_1 + a_2 + a_3",
             1 + sum(sympy.Rational(1, n) for n in range(1, 4))),
        "of-a-family-with-a-member-given",
        "9: the sum inlines the formula, 1/0 at n = 0, and never consults a_0",
    ),
    case(
        Pair("c_n := \\frac{1}{n^2}\nc_1 := 5", r"\sum_{n=1}^{3} c_n", "c_1 + c_2 + c_3",
             5 + sympy.Rational(1, 4) + sympy.Rational(1, 9)),
        "of-a-family-with-its-first-member-given",
        "9: the sum gives 49/36, its terms 193/36",
    ),
]  # fmt: skip


@pytest.mark.parametrize("pair", SUMS)
def test_a_sum_is_its_terms(pair):
    agree(pair)


# -- the vector of ones, transposed or dotted -------------------------------------------


W = r"w := \begin{pmatrix} 0.2 \\ 0.3 \\ 0.5 \end{pmatrix}"

ONES = [
    case(
        Pair(W, r"\mathbf{1}^T w", r"\mathbf{1} \cdot w", float(np.sum([0.2, 0.3, 0.5]))),
        "of-a-column",
        "6: \\mathbf{1} \\cdot w is w, as \\mathbf{1} is read alone beside \\cdot",
    ),
]


@pytest.mark.parametrize("pair", ONES)
def test_ones_transposed_is_ones_dotted(pair):
    agree(pair)


# -- a probability, its event written either way round ----------------------------------


PROBABILITIES = [
    case(
        Pair(r"X \sim \mathcal{N}(0, 1)", r"\Pr(X > 0)", r"\Pr(0 < X)",
             float(scipy.stats.norm.sf(0))),
        "of-a-law",
    ),
    case(
        Pair(r"X \sim \operatorname{Exp}(2)", r"\Pr(X > 1)", r"\Pr(1 < X)",
             float(scipy.stats.expon(scale=1 / 2).sf(1))),
        "of-another-law",
    ),
    case(
        Pair(r"\Pr(X > 0) := 0.7", r"\Pr(X > 0)", r"\Pr(0 < X)", 0.7),
        "given",
        "19: a probability given is found by the text of its event, and 0 < X is other text",
    ),
]  # fmt: skip


@pytest.mark.parametrize("pair", PROBABILITIES)
def test_an_event_either_way_round(pair):
    agree(pair)


# -- a list from Python, and one the program writes -------------------------------------


SCOPE = {"x": [1.0, 2.0, 3.0], "v": [3.0, 4.0]}

LISTS = [
    case(
        Pair(r"x := \texttt{x}", r"\sum_{i=1}^{3} x_i", r"\sum_{n=1}^{3} x_n",
             float(np.sum(SCOPE["x"])), SCOPE),
        "a-sum-of-its-elements",
        "21: a Python list is a constant array to the reader, and x_i a new symbol",
    ),
    case(
        Pair(r"v := \texttt{v}", r"|v|", r"\sqrt{v \cdot v}", float(np.linalg.norm(SCOPE["v"])),
             SCOPE),
        "its-length",
        "21: the absolute value of a list from Python is not defined",
    ),
]  # fmt: skip

OWN_LISTS = [
    case(
        Pair("x := [1, 2, 3]", r"\sum_{i=1}^{3} x_i", r"\sum_{n=1}^{3} x_n", 6.0),
        "a-sum-of-its-elements",
    ),
    case(Pair("v := [3, 4]", r"|v|", r"\sqrt{v \cdot v}", 5.0), "its-length"),
]


@pytest.mark.parametrize("pair", LISTS)
def test_a_list_from_python_is_a_list(pair):
    agree(pair)


@pytest.mark.parametrize("pair", OWN_LISTS)
def test_the_languages_own_list(pair):
    agree(pair)


# -- d/dx, and an operator defined as it ------------------------------------------------


P = r"\hat{p} := -i \hbar \frac{d}{dx}" "\n" r"\psi(x) := e^{-x^2}"
PSI = sympy.exp(-(x**2))

OPERATORS = [
    case(
        Pair(P, r"\hat{p} \psi(x)", r"-i \hbar \frac{d}{dx} \psi(x)",
             -sympy.I * hbar * sympy.diff(PSI, x)),
        "of-a-function",
    ),
    case(
        Pair(P, r"\hat{p} \hat{p} \psi(x)", r"-\hbar^2 \frac{d^2}{dx^2} \psi(x)",
             -(hbar**2) * sympy.diff(PSI, x, 2)),
        "twice",
    ),
    case(
        Pair(P, r"\hat{p} x \psi(x)", r"-i \hbar \frac{d}{dx} (x \psi(x))",
             -sympy.I * hbar * sympy.diff(x * PSI, x)),
        "of-a-product",
        "1: a defined operator acts on the next factor only",
    ),
]  # fmt: skip


@pytest.mark.parametrize("pair", OPERATORS)
def test_an_operator_is_what_it_is_defined_as(pair):
    agree(pair)


# -- the length of a vector, and the root of its inner product --------------------------


NORMS = [
    case(
        Pair(r"|v\rangle := \begin{pmatrix} 3 \\ 4 \end{pmatrix}", r"\| |v\rangle \|",
             r"\sqrt{\langle v | v \rangle}", float(np.linalg.norm([3, 4]))),
        "of-a-real-ket",
    ),
    case(
        Pair(r"\mathbf{u} := (3, 4)", r"|\mathbf{u}|", r"\sqrt{\mathbf{u} \cdot \mathbf{u}}",
             float(np.linalg.norm([3, 4]))),
        "of-a-real-vector",
    ),
    case(
        Pair(r"|\psi\rangle := \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ i \end{pmatrix}",
             r"\| |\psi\rangle \|", r"\sqrt{\langle \psi | \psi \rangle}",
             float(np.linalg.norm(np.array([1, 1j]) / np.sqrt(2)))),
        "of-a-complex-ket",
        "3: the length of a complex vector is sqrt(sum v^2), without conjugating",
    ),
]  # fmt: skip


@pytest.mark.parametrize("pair", NORMS)
def test_a_length_is_the_root_of_the_inner_product(pair):
    agree(pair)


# -- a function of a matrix, by its notation and by its name ----------------------------


ROWS = np.array([[0.5, 0.2], [0.1, 0.9]])
A = (
    r"\mathcal{V} := \mathbb{R}^{2}" "\n" r"a, b \in \mathcal{V}" "\n"
    r"A^{a}{}_{b} := [[0.5, 0.2], [0.1, 0.9]]"
)  # fmt: skip

MATRIX_FUNCTIONS = [
    case(Pair(A, r"\exp(A)", "expm(A)", scipy.linalg.expm(ROWS)), "the-exponential"),
    case(Pair(A, r"\sqrt{A}", "sqrtm(A)", scipy.linalg.sqrtm(ROWS)), "the-square-root"),
    case(Pair(A, r"\log(A)", "logm(A)", scipy.linalg.logm(ROWS)), "the-logarithm"),
]


@pytest.mark.parametrize("pair", MATRIX_FUNCTIONS)
def test_a_function_of_a_matrix_is_the_function_called_by_name(pair):
    agree(pair)


# -- a decimal, and the fraction it denotes ---------------------------------------------


DECIMALS = [
    case(
        Pair("", "laguerre(40, 25.0)", "N(laguerre(40, 25))",
             float(scipy.special.eval_laguerre(40, 25.0))),
        "in-a-polynomial-of-a-high-degree",
    ),
    case(
        Pair("", "jn(10, 0.9)", "N(jn(10, 9/10))", float(scipy.special.spherical_jn(10, 0.9))),
        "in-a-spherical-bessel-function",
    ),
    case(
        Pair("", "sqrt(0.25) + sin(0.5)", "N(sqrt(1/4) + sin(1/2))",
             float(np.sqrt(0.25) + np.sin(0.5))),
        "in-elementary-functions",
    ),
]  # fmt: skip


@pytest.mark.parametrize("pair", DECIMALS)
def test_a_decimal_is_the_fraction_it_denotes(pair):
    agree(pair)
