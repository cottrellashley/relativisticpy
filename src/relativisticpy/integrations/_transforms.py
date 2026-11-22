r"""Transforms and rewrites: two kinds of capability, each with its own checks.

A *transform* acts on a function of a variable and gives a function of another, as the
Laplace transform takes f(t) to F(s); or on data over one index, and gives data over a new
space, as the discrete Fourier transform takes samples to frequencies::

    @transform(
        "LaplaceTransform",
        latex=[r"\mathcal{L}\{f\}(s)", r"\mathcal{L}\{f\}"],
        summary=r"the Laplace transform, `\int_0^\infty e^{-st} f(t)\, dt`",
        variables=[("t", "s")],
        inverse="InverseLaplaceTransform",
        examples=[...],
        points=[(t**2,), (sympy.exp(-2 * t),)],
    )
    class Laplace:
        def exact(self, f, t=None, s=None):
            return sympy.laplace_transform(f, t, s, noconds=True)

        def integral(self, f, t, s):
            return sympy.Integral(sympy.exp(-s * t) * f, (t, 0, sympy.oo))

The variables of a transform of functions are named by convention, by the letters a program
uses (``design/notation.md``, 21); ``integral`` writes it as its integral or sum, and the
conformance suite checks the transform against it at points, and checks that its
``inverse`` undoes it. A transform of data has instead a ``space`` method, which names the
space its result is over, and a ``numeric`` one, which computes it along the first axis of
an array; the suite checks it against the sum it stands for, its inverse, Parseval's
relation and the size and labels of its space.

A *rewrite* is a faster computation of what a program writes: the discrete Fourier
transform written as its sum, on data, is computed as an FFT. It names the capability that
``computes`` it, and the suite checks that each of its examples gives the same numbers
rewritten as written.
"""

from __future__ import annotations

import contextlib
import itertools
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import sympy

from relativisticpy.errors import RelativisticPyError
from relativisticpy.integrations._declarations import Capability, Example, _declare
from relativisticpy.integrations._discovery import catalogue, on_change
from relativisticpy.language import syntax as ast
from relativisticpy.language.source import Span


@dataclass(frozen=True, eq=False)
class Transform(Capability):
    """What acts on a function of a variable and gives a function of another, as the Laplace
    transform does; or on data over one index, giving data over a new space, as the
    discrete Fourier transform does. A transform of functions names its ``variables`` by
    convention, the function's and its own, a pair for each convention the letters choose,
    as ℱ takes x to s and t to ω; ``inverse`` names the transform that undoes it."""

    kind: ClassVar[str] = "transform"
    takes: ClassVar[str] = "a function of a variable, or data"

    variables: tuple[tuple[str, str], ...] = ()
    inverse: str | None = None

    @property
    def on_data(self) -> bool:
        """Whether it acts on data over an index, giving data over a space of its own."""
        return callable(getattr(self.implementation, "space", None))

    @property
    def fourier(self) -> bool:
        """Whether it is a Fourier transform, whose convention a program may choose."""
        return bool(getattr(self.implementation, "fourier", False))


@dataclass(frozen=True, eq=False)
class Rewrite(Capability):
    """A faster computation of what a program writes, as the discrete Fourier transform
    written as its sum is computed as an FFT: ``computes`` names the capability that
    computes it. Its ``latex`` form writes the computation it stands for, and its
    implementation's ``matches`` says whether what a program writes is it."""

    kind: ClassVar[str] = "rewrite"
    takes: ClassVar[str] = "what a program writes"

    computes: str = ""

    @property
    def exact(self) -> Callable[..., Any]:
        """What says whether a sum is the one the rewrite stands for: its implementation's
        ``matches``, given the sum's kernel in the positions of its two indices, from 0, and
        their size, which gives the factor to scale the computation it names by, or
        ``None``."""
        method: Callable[..., Any] = self.implementation.matches
        return method


def transform(
    name: str,
    *,
    summary: str,
    examples: Sequence[Example],
    aliases: Sequence[str] = (),
    latex: str | Sequence[str] = (),
    unicode: str | Sequence[str] = (),
    requires: str | None = None,
    points: Sequence[Sequence[Any]] = (),
    variables: Sequence[tuple[str, str]] = (),
    inverse: str | None = None,
) -> Callable[[type], Transform]:
    """Declare a transform: of functions, whose ``variables`` are named by convention, a
    pair for each convention the letters choose, and checked at ``points``, each a function
    it transforms, which gives a scalar; or of data, whose implementation has a ``space``
    method, which gives data over it. ``inverse`` names the transform that undoes it."""
    return _declare(
        Transform,
        name,
        summary=summary,
        examples=examples,
        aliases=aliases,
        latex=latex,
        unicode=unicode,
        requires=requires,
        points=points,
        variables=tuple((str(given), str(taken)) for given, taken in variables),
        inverse=inverse,
        scalar=bool(variables),
    )


def rewrite(
    name: str,
    *,
    summary: str,
    examples: Sequence[Example],
    computes: str,
    latex: str | Sequence[str] = (),
    requires: str | None = None,
) -> Callable[[type], Rewrite]:
    """Declare a rewrite: what a program writes in its ``latex`` form, computed by the
    capability it ``computes`` with. Its class has a ``matches`` method, which says whether
    a sum is the one the rewrite stands for, rather than ``exact``: it is never called."""
    written = (latex,) if isinstance(latex, str) else tuple(latex)

    def decorate(cls: type) -> Rewrite:
        if not callable(getattr(cls, "matches", None)):
            raise TypeError(
                f"{name}: {cls.__name__} has no matches method; declare which sums it "
                "rewrites, as def matches(self, kernel, n, k, size): ..."
            )
        return Rewrite(
            cls(),
            name,
            summary,
            tuple(examples),
            latex=written,
            requires=requires,
            computes=computes,
        )

    return decorate


# -- Fourier conventions -----------------------------------------------------------------


@dataclass(frozen=True)
class Convention:
    r"""How a Fourier transform is written: :math:`F(k) = a \int f(x)\, e^{i b x k}\, dx`,
    with the ``factor`` a and the ``exponent`` b, as Bracewell writes it with 1 and -2π,
    and physics with 1 and -1; and what chose it, ``said`` in a note's words."""

    factor: Any
    exponent: Any
    said: str = ""

    def inverse(self) -> Convention:
        r"""The convention of the inverse transform: :math:`f(x) = \frac{|b|}{2\pi a}
        \int F(k)\, e^{-i b x k}\, dk`."""
        factor = sympy.simplify(sympy.Abs(self.exponent) / (2 * sympy.pi * self.factor))
        return Convention(factor, -self.exponent, self.said)

    def kernel(self, given: str, taken: str) -> str:
        """The exponential it is written with, as a note writes it, the transform's variable
        first: e^{-2πi s x}, e^{-i ω t}."""
        exponent = sympy.sympify(self.exponent)
        sign = "-" if exponent.could_extract_minus_sign() else ""
        size = sympy.Abs(exponent)
        scale = "" if size == 1 else "2π" if size == 2 * sympy.pi else f"{size}·"
        factor = sympy.sympify(self.factor)
        before = "" if factor == 1 else f"{_shown(factor)} "
        return f"{before}e^{{{sign}{scale}i {_letter(taken)} {_letter(given)}}}"

    def attributes(self) -> tuple[tuple[str, Any], ...]:
        """The convention as the attributes of the operation that computes the transform."""
        return (("factor", sympy.sympify(self.factor)), ("exponent", sympy.sympify(self.exponent)))


#: Bracewell's convention, of ordinary frequency: F(s) = ∫ f(x) e^{-2πixs} dx.
ORDINARY = Convention(sympy.Integer(1), -2 * sympy.pi, "Bracewell's convention")

#: The convention of angular frequency, physics': F(ω) = ∫ f(t) e^{-iωt} dt.
ANGULAR = Convention(sympy.Integer(1), sympy.Integer(-1), "the convention of angular frequency")


def convention_of(integrand: Any, x: Any, k: Any, factor: Any = 1) -> Convention | None:
    r"""The convention an integral over the whole line writes, of the exponential in
    ``integrand`` that is :math:`e^{i b x k}`: the factor before the integral and b;
    ``None`` where the integrand has no such exponential."""
    for part in sympy.Mul.make_args(sympy.sympify(integrand)):
        if isinstance(part, sympy.exp) or (isinstance(part, sympy.Pow) and part.base == sympy.E):
            power = part.exp if isinstance(part, sympy.Pow) else part.args[0]
            exponent = sympy.simplify(sympy.expand(power).coeff(x * k) / sympy.I)
            if exponent.has(x, k) or exponent == 0:
                continue
            return Convention(sympy.sympify(factor), exponent)
    return None


def _shown(value: Any) -> str:
    from relativisticpy import display

    return display.unicode(value)


def _letter(name: str) -> str:
    from relativisticpy import display

    return display.unicode(sympy.Symbol(name))


# -- written forms ----------------------------------------------------------------------

_BRACES = ("\\{", "\\lbrace", "\\left\\{", "\\left\\lbrace")
_CLOSING = ("\\}", "\\rbrace", "\\right\\}", "\\right\\rbrace")
_ROUND = ("(", "\\left(")


@dataclass(frozen=True)
class Head:
    """How a transform is named where a program writes it: the letter, as the L of
    ``\\mathcal{L}``, how it is set, and whether it is inverted, ``\\mathcal{L}^{-1}``."""

    name: str
    style: str
    inverted: bool = False


@dataclass(frozen=True)
class Written:
    """A transform as a program writes it: the ``capability``, what it acts on, its
    ``operand``, in braces, the variable written after it in brackets, its ``point``, if
    there is one, as the s of ``\\mathcal{L}\\{f\\}(s)``, and the ``head`` it is written
    with."""

    capability: Transform
    operand: ast.Expr
    point: ast.Expr | None
    span: Span
    head: Head


def head_of(node: ast.Node) -> Head | None:
    """The head a transform is written with: a name, as ``\\mathcal{L}``, or a name to the
    power -1, as ``\\mathcal{L}^{-1}``; ``None`` for anything else."""
    from relativisticpy.integrations._forms import _style
    from relativisticpy.typing.syntax import _is_minus_one

    inverted = False
    if isinstance(node, ast.Scripted):
        if len(node.scripts) != 1 or node.scripts[0].kind != "^":
            return None
        if not _is_minus_one(node.scripts[0].body):
            return None
        node, inverted = node.base, True
    if not isinstance(node, ast.Name):
        return None
    return Head(node.id, _style(node), inverted)


def in_braces(node: ast.Node) -> bool:
    """Whether an expression is written in braces, as the f of ``\\mathcal{L}\\{f\\}``."""
    return node.span.text.lstrip().startswith(_BRACES)


def _braced_call(node: ast.Call) -> bool:
    """Whether a call's argument is written in braces, ``\\mathcal{L}\\{f\\}``."""
    after = node.span.source.text[node.func.span.end : node.span.end].lstrip()
    return after.startswith(_BRACES)


def _in_round_brackets(node: ast.Node) -> bool:
    return node.span.text.lstrip().startswith(_ROUND)


def _closed_after(name: ast.Name, call: ast.Call) -> bool:
    """Whether ``\\{F\\}(t)``, which the parser reads as F called with t, is F in braces and
    then t in brackets: the parser gives F the span of its braces."""
    return in_braces(name) and name.span.text.rstrip().endswith(_CLOSING)


def written_at(factors: Sequence[ast.Expr], position: int) -> tuple[Written, int] | None:
    """The transform written at ``position`` among the factors of a product, and how many
    factors it takes: ``\\mathcal{L}\\{f\\}``, ``\\mathcal{L}\\{f\\}(s)`` or
    ``\\mathcal{L}^{-1}\\{F\\}(t)``; ``None`` where none is."""
    first = factors[position]
    point: ast.Expr | None = None
    if isinstance(first, ast.Call) and _braced_call(first):
        head, used = head_of(first.func), 1
        if len(first.args) != 1:
            return None
        operand = first.args[0]
    else:
        head, used = head_of(first), 2
        following = factors[position + 1] if position + 1 < len(factors) else None
        if head is None or following is None or not in_braces(following):
            return None
        operand = following
        if (
            isinstance(following, ast.Call)
            and isinstance(following.func, ast.Name)
            and len(following.args) == 1
            and _closed_after(following.func, following)
        ):  # \{F\}(t), read as F(t) by the parser
            operand, point = following.func, following.args[0]
    if head is None:
        return None
    capability = transforms().get(head)
    if capability is None:
        return None
    last = factors[position + used - 1]
    if point is None and position + used < len(factors):
        after = factors[position + used]
        if _in_round_brackets(after) and after.span.start == last.span.end:
            point, used = after, used + 1
    span = first.span.to(factors[position + used - 1].span)
    return Written(capability, operand, point, span, head), used


def written_call(node: ast.Call) -> Written | None:
    """A transform written alone as a call, ``\\mathcal{L}\\{t^2\\}``; ``None`` for any
    other call."""
    found = written_at([node], 0)
    return found[0] if found is not None else None


def form_head(written: str) -> Head | None:
    """The head of a transform's written form, as ``\\mathcal{L}^{-1}`` of
    ``\\mathcal{L}^{-1}\\{F\\}(t)``; ``None`` where the form is not written so."""
    from relativisticpy.language import parse

    try:
        statements = parse(written).statements
    except RelativisticPyError:
        return None
    if len(statements) != 1 or not isinstance(statements[0], ast.Evaluate):
        return None
    factors = _factors(statements[0].expr)
    first = factors[0]
    if isinstance(first, ast.Call) and _braced_call(first):
        return head_of(first.func)
    if len(factors) > 1 and in_braces(factors[1]):
        return head_of(first)
    return None


def form_failures(capability: Transform, written: str) -> list[str]:
    """A transform's form written with a head, as ``\\mathcal{L}\\{f\\}(s)``, reads as it,
    with a head of its own, and names in braces and brackets only parameters of exact."""
    from relativisticpy.language import parse

    name = capability.name
    head = form_head(written)
    found = transforms().get(head) if head is not None else None
    if found is not capability:
        other = found.name if found is not None else "nothing"
        return [
            f"{name}: the form {written} reads as {other}, not as {name}; write it with a head "
            "of its own, as \\mathcal{Z}"
        ]
    statement = parse(written).statements[0]
    assert isinstance(statement, ast.Evaluate)
    match = written_at(_factors(statement.expr), 0)
    assert match is not None
    slots = (match[0].operand, match[0].point)
    strange = [
        slot.id
        for slot in slots
        if isinstance(slot, ast.Name) and slot.id not in capability.parameters
    ]
    if strange:
        return [
            f"{name}: the form {written} writes {', '.join(strange)}, which is no parameter of "
            "exact; name the variables of a form as exact names its parameters, as "
            f"{capability.text()}"
        ]
    return []


def sum_failures(capability: Rewrite) -> list[str]:
    """Each form of a rewrite is the sum it stands for, which the language reads."""
    from relativisticpy.language import parse

    failures = []
    if not capability.latex:
        failures.append(
            f"{capability.name}: declare the sum it rewrites, as "
            'latex=r"\\sum_{n=0}^{N-1} x_n e^{-2\\pi i k n/N}"'
        )
    for written in capability.latex:
        try:
            statements = parse(written).statements
        except RelativisticPyError as error:
            message = f"its form {written} does not parse: {error.message}"
            failures.append(f"{capability.name}: {message}")
            continue
        expr = statements[0].expr if isinstance(statements[0], ast.Evaluate) else None
        if not isinstance(expr, ast.BigOperator) or expr.op != "sum":
            failures.append(
                f"{capability.name}: its form {written} is no sum; write the sum it computes faster"
            )
    return failures


def _factors(node: ast.Expr) -> list[ast.Expr]:
    found: list[ast.Expr] = []
    while isinstance(node, ast.Binary) and node.op == "*":
        found.insert(0, node.right)
        node = node.left
    found.insert(0, node)
    return found


_found: dict[Head, Transform] | None = None


def transforms() -> dict[Head, Transform]:
    """The transforms written with a head of their own, as ``\\mathcal{L}``, by it."""
    global _found
    if _found is None:
        heads: dict[Head, Transform] = {}
        for capability in catalogue().capabilities:
            if isinstance(capability, Transform):
                for written in capability.latex:
                    head = form_head(written)
                    if head is not None:
                        heads.setdefault(head, capability)
        _found = heads
    return _found


def _forget() -> None:
    global _found
    _found = None


on_change(_forget)


# -- rewrites ---------------------------------------------------------------------------

_rewriting = True


@contextlib.contextmanager
def written_as_is() -> Iterator[None]:
    """The language computing what a program writes as it is written, rewriting nothing,
    while the block runs: to check that a rewrite gives the same numbers."""
    global _rewriting
    before = _rewriting
    _rewriting = False
    try:
        yield
    finally:
        _rewriting = before


def rewriting() -> bool:
    """Whether rewrites are made: everywhere but inside :func:`written_as_is`."""
    return _rewriting


def rewrites() -> list[Rewrite]:
    """Every rewrite the integrations declare."""
    return [c for c in catalogue().capabilities if isinstance(c, Rewrite)]


# -- frequencies --------------------------------------------------------------------------


def frequencies(size: int, spacing: Any = 1) -> list[Any]:
    """The frequencies of the bins of a discrete Fourier transform of ``size`` samples
    ``spacing`` apart, in the order of its values, as ``numpy.fft.fftfreq`` gives them:
    0, 1/(N d), ..., then the negative ones."""
    step = sympy.nsimplify(spacing) if isinstance(spacing, float) else sympy.sympify(spacing)
    half = (size - 1) // 2 + 1
    bins = [*range(half), *range(-(size // 2), 0)]
    return [sympy.Rational(k, size) / step for k in bins]


def label(value: Any) -> str:
    """A number as a label of a space's basis: ``1/8``, ``-0.25``, ``3``."""
    value = sympy.sympify(value)
    if value.is_Rational:
        return str(value)
    return repr(float(value))


def spacing(basis: Sequence[str] | None) -> Any:
    """The step between the points of a space whose basis is evenly spaced numbers, as a
    grid's is; 1, the step between samples, for any other basis."""
    if not basis or len(basis) < 2:
        return sympy.Integer(1)
    try:
        values = [sympy.nsimplify(sympy.sympify(name)) for name in basis]
    except (sympy.SympifyError, TypeError, ValueError):
        return sympy.Integer(1)
    if not all(value.is_number and value.is_real for value in values):
        return sympy.Integer(1)
    step = values[1] - values[0]
    evenly = all(sympy.simplify(b - a - step) == 0 for a, b in itertools.pairwise(values))
    return step if evenly and step != 0 else sympy.Integer(1)


# -- the checks of the conformance suite -------------------------------------------------

#: How far a transform and the integral or sum it stands for may differ, at a point.
AGREEMENT = 1e-7


def transform_checks(capability: Capability) -> list[str]:
    """A transform of functions agrees with its integral or sum at points, and its inverse
    undoes it; a transform of data agrees with the sum it stands for, its inverse undoes
    it, it keeps Parseval's relation, and its space has the size and the labels it
    declares."""
    assert isinstance(capability, Transform)
    if capability.on_data:
        return _data_checks(capability)
    return _function_checks(capability)


def _function_checks(capability: Transform) -> list[str]:
    name, failures = capability.name, []
    implementation = capability.implementation
    integral = getattr(implementation, "integral", None)
    if integral is None and capability.inverse is None:
        return [
            f"{name}: declare how it is written as an integral or a sum, as "
            "def integral(self, f, t, s): return sympy.Integral(sympy.exp(-s*t)*f, "
            "(t, 0, sympy.oo)), which it is checked against, or the inverse that undoes it"
        ]
    if not capability.variables:
        return [f'{name}: declare the variables it maps by convention, as variables=[("t", "s")]']
    if not capability.points:
        return [
            f"{name}: declare points=[(f,), ...], functions it transforms, at which it is "
            "checked against its integral and its inverse"
        ]
    given, taken = (sympy.Symbol(letter) for letter in capability.variables[0])
    inverse = catalogue().get(capability.inverse) if capability.inverse else None
    for point in capability.points:
        function = sympy.sympify(point[0])
        try:
            found = capability.exact(function, given, taken)
        except Exception as error:
            failures.append(f"{name} of {_shown(function)} fails: {type(error).__name__}: {error}")
            continue
        for value in _samples(implementation) if callable(integral) else ():
            assert integral is not None
            written = integral(function, given, taken).subs(taken, value)
            try:
                expected = complex(sympy.N(written, 15))
                computed = complex(sympy.N(found.subs(taken, value), 15))
            except (TypeError, ValueError):
                failures.append(
                    f"{name} of {_shown(function)} is no number at {taken} = {value}: "
                    f"{_shown(found)}; declare points where it is one"
                )
                break
            if not _close(computed, expected):
                failures.append(
                    f"{name} of {_shown(function)} at {taken} = {value} is {_number(computed)}, "
                    f"but its integral is {_number(expected)}: fix exact, or the integral"
                )
                break
        if inverse is None:
            continue
        failures += _undone(capability, inverse, function, found, given, taken)
    if capability.inverse and inverse is None:
        failures.append(f"{name}: its inverse, {capability.inverse}, is declared by no integration")
    return failures


def _undone(
    capability: Transform, inverse: Capability, function: Any, found: Any, given: Any, taken: Any
) -> list[str]:
    """Whether the inverse of a transform gives back the function, at points."""
    try:
        back = inverse.exact(found, taken, given)
    except Exception as error:
        return [
            f"{inverse.name} of {capability.name} of {_shown(function)} fails: "
            f"{type(error).__name__}: {error}"
        ]
    for value in _samples(inverse.implementation):
        try:
            then = complex(sympy.N(back.subs(given, value), 15))
            now = complex(sympy.N(function.subs(given, value), 15))
        except (TypeError, ValueError):
            return [
                f"{inverse.name} of {capability.name} of {_shown(function)} is "
                f"{_shown(back)}, no number at {given} = {value}"
            ]
        if not _close(then, now):
            return [
                f"{inverse.name} does not undo {capability.name}: of {_shown(function)} it "
                f"gives back {_shown(back)}, {_number(then)} at {given} = {value}, not "
                f"{_number(now)}; fix one of them"
            ]
    return []


def _samples(implementation: Any) -> tuple[Any, ...]:
    """The values of a variable a transform is checked at: those its implementation
    declares, or else three."""
    declared = getattr(implementation, "samples", None)
    if declared is not None:
        return tuple(declared)
    return (sympy.Rational(13, 10), sympy.Rational(21, 10), sympy.Rational(37, 10))


def _number(value: complex) -> str:
    """A number as a message writes it: real where it is, to ten digits."""
    if abs(value.imag) <= AGREEMENT * max(1.0, abs(value.real)):
        return f"{value.real:.10g}"
    return f"{value:.10g}"


def _close(computed: complex, expected: complex) -> bool:
    if not (math.isfinite(computed.real) and math.isfinite(computed.imag)):
        return False
    return abs(computed - expected) <= AGREEMENT * max(1.0, abs(expected))


#: The samples a transform of data is checked on: real and complex, of even and odd size.
_DATA = (
    (1.0, 2.0, 0.0, -1.0, 3.0, 0.0, 0.0, 2.0),
    (0.5, -1.5, 2.25, 4.0, -0.75),
    (1 + 2j, -1j, 0.5, 3 - 1j, 2.0, -2.5j),
)


def _data_checks(capability: Transform) -> list[str]:
    import numpy as np

    from relativisticpy.tensors.spaces import Space

    name, failures = capability.name, []
    implementation = capability.implementation
    summed = getattr(implementation, "sum", None)
    if summed is None:
        return [
            f"{name}: declare the sum it stands for, as def sum(self, x, k): return "
            "sum(x[n] * exp(-2j*pi*k*n/len(x)) for n in range(len(x))), which it is checked "
            "against"
        ]
    inverse = catalogue().get(capability.inverse) if capability.inverse else None
    for samples in _DATA:
        x = np.asarray(samples)
        size = len(x)
        found = np.asarray(capability.numeric(np, x))  # type: ignore[misc]
        expected = np.asarray([complex(summed(x, k)) for k in range(size)])
        if found.shape != (size,) or not np.allclose(found, expected, rtol=1e-9, atol=1e-9):
            failures.append(
                f"{name} of {list(samples)} is {np.round(found, 6).tolist()}, but the sum it "
                f"stands for is {np.round(expected, 6).tolist()}: fix numeric, or the sum"
            )
            continue
        exactly = capability.exact([sympy.nsimplify(value) for value in samples])
        if not np.allclose(np.asarray([complex(v) for v in exactly]), expected, atol=1e-9):
            failures.append(f"{name}: exact of {list(samples)} differs from numeric; fix exact")
        energy = getattr(implementation, "energy", None)
        if energy is not None:
            before, after = float(np.sum(np.abs(x) ** 2)), float(np.sum(np.abs(found) ** 2))
            if not math.isclose(after, float(energy(size)) * before, rel_tol=1e-9):
                failures.append(
                    f"{name} of {list(samples)} does not keep Parseval's relation: the "
                    f"energy is {before:.10g} before and {after:.10g} after, not "
                    f"{float(energy(size)):.10g} times it; fix numeric, or energy"
                )
        if inverse is not None and inverse.numeric is not None:
            back = np.asarray(inverse.numeric(np, found))
            if not np.allclose(back, x, atol=1e-9):
                failures.append(
                    f"{inverse.name} does not undo {name}: of {list(samples)} it gives back "
                    f"{np.round(back, 6).tolist()}; fix one of them"
                )
    for size, step in ((8, 1), (5, sympy.Rational(1, 4))):
        basis = tuple(label(step * n) for n in range(size))
        over = Space("samples", size, basis)
        space = implementation.space(over)
        if not isinstance(space, Space) or space.dim != size:
            failures.append(
                f"{name}: of a space of {size} samples, its space is {space!r}, not one of "
                f"{size}: fix space"
            )
            continue
        labels = getattr(implementation, "labels", None)
        wanted = list(labels(size, step)) if labels is not None else None
        if wanted is not None and list(space.basis or ()) != [label(v) for v in wanted]:
            failures.append(
                f"{name}: the labels of its space are {list(space.basis or ())}, not the "
                f"{[label(v) for v in wanted]} it declares; fix space or labels"
            )
        failures += _labelled_in_language(capability, size, step)
    return failures


def _labelled_in_language(capability: Transform, size: int, step: Any) -> list[str]:
    """Applied in the language to data over a grid, whose space is labelled by the values of
    its index, a transform of data gives a tensor over the space its ``space`` names."""
    import numpy as np

    from relativisticpy import Workbook
    from relativisticpy.tensors.spaces import Space

    name = capability.name
    grid = Space(f"n = 0, …, {size - 1}", size, tuple(str(n) for n in range(size)))
    space = capability.implementation.space(grid)
    program = f"N := {size}\nt_n := {_written(step)} n, \\quad n = 0, \\ldots, N-1\n{name}(t_n)"
    try:
        with Workbook() as wb:
            result = wb.exe(program)[-1]
    except RelativisticPyError as error:
        return [f"{name} applied to data over a grid fails: {error.kind}: {error.message}"]
    spaces = getattr(result, "spaces", None)
    if not spaces or spaces[0].dim != size or spaces[0].basis != space.basis:
        return [
            f"{name} applied to data over a grid of {size} points gives {result!r}, not a "
            f"tensor over {space}: lower it over the space its space method names"
        ]
    if np.asarray(result.to_numpy()).shape != (size,):
        return [f"{name} applied to data over a grid gives the wrong number of values"]
    return []


def _written(value: Any) -> str:
    value = sympy.sympify(value)
    if value.is_Rational and not value.is_integer:
        return f"\\frac{{{value.p}}}{{{value.q}}}"
    return str(value)


def rewrite_checks(capability: Capability) -> list[str]:
    """Each example of a rewrite gives the same numbers rewritten as written, and is
    rewritten: computed by the capability the rewrite names."""
    from relativisticpy import Workbook
    from relativisticpy.integrations._conformance import _same
    from relativisticpy.integrations._conformance import _shown as _program

    assert isinstance(capability, Rewrite)
    name, failures = capability.name, []
    if catalogue().get(capability.computes) is None:
        return [f"{name}: it computes with {capability.computes}, which no integration declares"]
    for example in capability.examples:
        shown = _program(example)
        try:
            with Workbook() as wb:
                ir = wb.ir(example.program)
                rewritten = wb.exe(example.program)
            with written_as_is(), Workbook() as wb:
                kept = wb.ir(example.program)
                written = wb.exe(example.program)
        except RelativisticPyError as error:
            failures.append(f"{name}'s example {shown} fails: {error.kind}: {error.message}")
            continue
        if f"call {capability.computes}" not in ir:
            failures.append(
                f"{name}'s example {shown} is not rewritten: it is not computed with "
                f"{capability.computes}; fix matches, or the example"
            )
        if f"call {capability.computes}" in kept:
            failures.append(f"{name}'s example {shown} is rewritten even as written")
        if not rewritten or not written or not _same(rewritten[-1], written[-1], 1e-9):
            failures.append(
                f"{name}'s example {shown} gives other numbers rewritten than written: "
                f"{_value(rewritten)} and {_value(written)}; fix the rewrite"
            )
    return failures


def _value(values: Sequence[Any]) -> str:
    from relativisticpy import display

    return display.unicode(values[-1]) if values else "nothing"
