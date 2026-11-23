r"""Sequences, as signal processing writes them.

The unit impulse ``\delta[n]`` and the unit step ``u[n]`` of Oppenheim and Schafer, of which
discrete-time signals are built: the impulse is 1 at n = 0 and 0 at every other integer,
the step 1 from n = 0 on and 0 before it. And the poles and the zeros of a transfer
function, the roots of its denominator and of its numerator. Exact with SymPy, and numeric
with NumPy. Oppenheim and Schafer (2010) is *Discrete-Time Signal Processing*, 3rd
edition.

A sum over the index of a sequence is taken a term at a time (:func:`pieces`): an impulse
leaves its term at the one point where it is 1, and a step cuts the range of its term to
where it is 1, so that what is left is a sum SymPy can find, over a range of its own.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import sympy

from relativisticpy.errors import EvaluationError
from relativisticpy.integrations import Example, function


class impulse(sympy.Function):  # type: ignore[misc]  # named as its calls print
    """``\\delta[n]``: 1 at n = 0 and 0 at every other n, kept as it is where n is a
    symbol that may be 0."""

    nargs = 1
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, n: Any) -> Any:
        if n.is_zero:
            return sympy.S.One
        if n.is_zero is False:
            return sympy.S.Zero
        return None

    def _eval_Abs(self) -> Any:
        return self

    def _eval_conjugate(self) -> Any:
        return self

    def _eval_power(self, exponent: Any) -> Any:
        return self if exponent.is_positive else None

    def _eval_rewrite_as_KroneckerDelta(self, n: Any, **hints: Any) -> Any:
        return sympy.KroneckerDelta(n, 0)

    def _eval_rewrite_as_Piecewise(self, n: Any, **hints: Any) -> Any:
        return sympy.Piecewise((1, sympy.Eq(n, 0)), (0, True))


class step(sympy.Function):  # type: ignore[misc]  # named as its calls print
    """``u[n]``: 1 from n = 0 on and 0 before it, kept as it is where the sign of n is not
    known."""

    nargs = 1
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, n: Any) -> Any:
        if n.is_extended_nonnegative:
            return sympy.S.One
        if n.is_extended_negative:
            return sympy.S.Zero
        return None

    def _eval_Abs(self) -> Any:
        return self

    def _eval_conjugate(self) -> Any:
        return self

    def _eval_power(self, exponent: Any) -> Any:
        return self if exponent.is_positive else None

    def _eval_rewrite_as_Heaviside(self, n: Any, **hints: Any) -> Any:
        return sympy.Heaviside(n, 1)

    def _eval_rewrite_as_Piecewise(self, n: Any, **hints: Any) -> Any:
        return sympy.Piecewise((1, n >= 0), (0, True))


#: The functions sequences are built of, which a sum over their index takes apart.
MEMBERS = (impulse, step)


@function(
    "impulse",
    latex=r"\delta[n]",
    unicode="δ[n]",
    summary="the unit impulse; 1 at n = 0 and 0 at every other integer",
    examples=[
        Example(r"\delta[0] + \delta[3]", 1, "Oppenheim and Schafer (2010), eq. (2.3)"),
        Example(r"\sum_{k=-\infty}^{\infty} 2^k \delta[k - 3]", 8,
                "Oppenheim and Schafer (2010), eq. (2.6): an impulse picks one term"),
    ],
    points=[(0,), (1,), (-2,)],
)  # fmt: skip
class UnitImpulseSequence:
    def exact(self, n: Any) -> Any:
        return impulse(n)

    def numeric(self, xp: Any, n: Any) -> Any:
        return xp.where(n == 0, 1.0, 0.0)


@function(
    "step",
    latex="u[n]",
    unicode="u[n]",
    summary="the unit step; 1 from n = 0 on and 0 before it",
    examples=[
        Example("u[0] - u[-1]", 1, "Oppenheim and Schafer (2010), eq. (2.7)"),
        Example(r"\sum_{n=-\infty}^{\infty} \left(\frac{1}{2}\right)^n u[n]", 2,
                "Oppenheim and Schafer (2010), eq. (2.7): the step cuts the sum to n ≥ 0"),
    ],
    points=[(0,), (2,), (-1,)],
)  # fmt: skip
class UnitStepSequence:
    def exact(self, n: Any) -> Any:
        if sympy.sympify(n).is_extended_real is False:
            raise _not_real()
        return step(n)

    def numeric(self, xp: Any, n: Any) -> Any:
        if xp.iscomplexobj(n) and xp.any(xp.imag(n) != 0):
            raise _not_real()
        return xp.where((xp.real(n) if xp.iscomplexobj(n) else n) >= 0, 1.0, 0.0)


def _not_real() -> EvaluationError:
    return EvaluationError(
        "the unit step u[n] is a function of a real number, and this is not one",
        hint="u[n] is 1 where n ≥ 0 and 0 where n < 0, which compares n with 0, as only a "
        "real n can be",
    )


def _roots(H: Any, z: Any, part: Callable[[Any], Any], what: str) -> list[Any]:
    """The roots of the numerator or the denominator of a rational function, ``part`` of
    it, each as often as its order, ordered by their real parts and then their imaginary
    parts, where they are numbers."""
    H = sympy.sympify(H)
    if not isinstance(H, sympy.Expr):
        raise EvaluationError(
            f"{what} takes a rational function, not {H}",
            hint=f"for example {what}(1/(1 - z^{{-1}}/2))",
        )
    variable = _variable(H, z, what)
    rational = sympy.together(H)
    if not rational.is_rational_function(variable):
        raise EvaluationError(
            f"{what} takes a rational function of {variable}, and {H} is not one",
            hint=f"for example {what}(1/(1 - z^{{-1}}/2))",
        )
    polynomial = sympy.Poly(part(sympy.fraction(sympy.cancel(rational))), variable)
    if polynomial.is_zero:  # 0 is 0 at every point
        raise EvaluationError(
            f"{H} is 0 for every {variable}, so every {variable} is one of its {what}, "
            "which no list can give",
            hint=f"{what} takes a rational function that is not 0",
        )
    if polynomial.degree() <= 0:
        return []
    found = sympy.roots(polynomial, multiple=True)
    if len(found) != polynomial.degree():
        found = polynomial.all_roots()
    numbers = all(root.is_number for root in found)
    if numbers:  # by their values: the real part of CRootOf(z^5 - z - 1, 2) is no number
        values = {root: complex(sympy.N(root, 30)) for root in found}
        return sorted(found, key=lambda root: (values[root].real, values[root].imag))
    return list(found)


def _variable(H: Any, z: Any, what: str) -> Any:
    """The variable of a rational function: the one given, its one symbol, or else z or s,
    as transfer functions are written in them."""
    if z is not None:
        return z
    symbols = sorted(H.free_symbols, key=str)
    if len(symbols) == 1:
        return symbols[0]
    named = [s for s in symbols if str(s) in ("z", "s")]
    if len(named) == 1:
        return named[0]
    if not symbols:
        return sympy.Symbol("z")
    raise EvaluationError(
        f"{what} of a function of {', '.join(map(str, symbols))} needs its variable",
        hint=f"name it, as in {what}(H, z)",
    )


@function(
    "poles",
    summary="the poles of a rational function, the roots of its denominator, each as often "
    "as its order",
    examples=[
        Example(r"poles(1/(1 - 3/4 z^(-1) + 1/8 z^(-2)))",
                [sympy.Rational(1, 4), sympy.Rational(1, 2)],
                "Oppenheim and Schafer (2010), example 5.1"),
        Example("poles(1/(s^2 + 3 s + 2), s)", [-2, -1], "standard: s^2 + 3s + 2 = (s + 1)(s + 2)"),
    ],
    lifts=False,
    scalar=False,
)  # fmt: skip
class Poles:
    def exact(self, H: Any, z: Any = None) -> Any:
        return _roots(H, z, lambda parts: parts[1], "poles")


@function(
    "zeros",
    summary="the zeros of a rational function, the roots of its numerator, each as often as "
    "its order",
    examples=[
        Example("zeros((1 + z^(-1))/2)", [-1], "Oppenheim and Schafer (2010), section 7.1"),
        Example("zeros((s - 1)(s + 2)/(s + 3), s)", [-2, 1], "standard"),
    ],
    lifts=False,
    scalar=False,
)  # fmt: skip
class Zeros:
    def exact(self, H: Any, z: Any = None) -> Any:
        return _roots(H, z, lambda parts: parts[0], "zeros")


# -- sums over the index of a sequence ---------------------------------------------------


def of_sequences(expr: Any, index: Any) -> bool:
    """Whether an expression holds an impulse or a step of ``index``, as a sum over a
    sequence's index does."""
    expr = sympy.sympify(expr)
    return isinstance(expr, sympy.Basic) and any(
        member.has(index) for member in expr.atoms(*MEMBERS)
    )


def pieces(body: Any, index: Any, lower: Any, upper: Any) -> list[tuple[Any, Any, Any]] | None:
    """The sum of ``body`` over the integers ``index`` from ``lower`` to ``upper``, as sums
    each over a range of its own, ``(term, first, last)``: a term at a time, each impulse of
    the index taken where it is 1, its term then a value at that one point, and each step of
    the index cutting the range of its term to where it is 1. ``None`` where a term holds an
    impulse or a step of the index that this does not take apart, as one of n^2 or of 2n - 1
    at a symbolic point, which SymPy sums as it is."""
    found: list[tuple[Any, Any, Any]] = []
    for term in sympy.Add.make_args(sympy.expand_mul(sympy.sympify(body))):
        piece = _piece(term, index, lower, upper)
        if piece is None:
            return None
        if isinstance(piece, tuple):
            found.append(piece)
    return found


def _piece(term: Any, index: Any, lower: Any, upper: Any) -> tuple[Any, Any, Any] | None:
    """One term of :func:`pieces`; of an impulse, its value at its point, where the range
    holds the point, as steps of the point's distance from the ends say."""
    factors = sympy.Mul.make_args(term)
    impulses = [f for f in factors if isinstance(f, impulse) and f.has(index)]
    if impulses:
        point = _point(impulses[0].args[0], index)
        if point is None:
            return None
        if point is False:
            return sympy.S.Zero, sympy.S.Zero, sympy.S.Zero
        value = term.xreplace({index: point})
        for end, inside in ((lower, point - lower), (upper, upper - point)):
            if not sympy.sympify(end).is_infinite:
                value *= step(inside)
        return value, point, point
    first, last, kept = lower, upper, []
    for factor in factors:
        if not (isinstance(factor, step) and factor.has(index)):
            kept.append(factor)
            continue
        cut = _cut(factor.args[0], index)
        if cut is None:
            return None
        side, end = cut
        if side == "from":
            first = sympy.Max(first, end)
        else:
            last = sympy.Min(last, end)
    rest = sympy.Mul(*kept)
    if any(member.has(index) for member in rest.atoms(*MEMBERS)):
        return None
    return rest, first, last


def linear(argument: Any, index: Any) -> tuple[Any, Any] | None:
    """The coefficient of the index and the rest of an argument linear in it, as 1 and -3
    of n - 3, the argument of an impulse or a step; ``None`` for any other argument."""
    expanded = sympy.expand(argument)
    coefficient = expanded.coeff(index)
    rest = expanded - coefficient * index
    if coefficient == 0 or coefficient.has(index) or rest.has(index):
        return None
    if not (coefficient.is_Integer or coefficient.is_Rational):
        return None
    return coefficient, rest


def _point(argument: Any, index: Any) -> Any:
    """Where an impulse of the index is 1: the integer that makes its argument 0; ``False``
    where no integer does, as for 2n - 1, and ``None`` where that is not known, as for
    2n - m."""
    found = linear(argument, index)
    if found is None:
        return None
    coefficient, rest = found
    point = -rest / coefficient
    if point.is_integer:
        return point
    if point.is_number:
        return False
    return point if abs(coefficient) == 1 else None


def _cut(argument: Any, index: Any) -> tuple[str, Any] | None:
    """Where a step of the index is 1: ``("from", m)`` of u[n - m], every n from m, and
    ``("to", m)`` of u[m - n], every n up to m; ``None`` for an argument not linear in the
    index."""
    found = linear(argument, index)
    if found is None:
        return None
    coefficient, rest = found
    end = -rest / coefficient
    if coefficient > 0:
        return "from", end if abs(coefficient) == 1 else sympy.ceiling(end)
    return "to", end if abs(coefficient) == 1 else sympy.floor(end)


def emptied(value: Any, first: Any, last: Any) -> Any:
    """A sum over a range from ``first`` to ``last``, which has no terms where the last is
    before the first: 0 there, by a step of their difference, where that may be."""
    width = sympy.sympify(last - first)
    if width.is_infinite or width.is_extended_nonnegative:
        return value
    if width.is_extended_negative:
        return sympy.S.Zero
    return value * step(width)
