"""Simplifying the components of tensors.

Components are kept in a normal form that suits relativity: trigonometric
functions are written with ``sin`` and ``cos`` only, and multiple angles are
expanded (``sin 2θ`` becomes ``2 sin θ cos θ``). Besides matching how papers
write Christoffel symbols, the normal form makes identities checkable by
algebra alone: ``cot θ`` and ``cos θ / sin θ`` become the same expression.

Deciding whether a component vanishes is the most frequent question, so it
has a fast, exact answer (:func:`is_zero`): combine the expression into one
fraction, expand its numerator, and use :math:`\\cos^2 = 1 - \\sin^2` until no
cosine has a power above one. The numerator is then zero exactly when the
component vanishes. This never computes a polynomial GCD, the step that makes
SymPy's ``cancel`` and ``simplify`` crawl on the curvature of the Kerr metric.

A decimal is the exact fraction it denotes (:func:`rationalise`): roundoff
never cancels exactly, so a metric given with ``1.7`` would otherwise have
curvature that is only nearly zero.
"""

from __future__ import annotations

import sympy
from sympy import Expr, cos, cot, csc, sec, sin, tan

#: Above this many operations, components are tidied rather than fully simplified.
LARGE = 400


def rationalise(expr: Expr) -> Expr:
    """Write every decimal as the exact fraction it denotes: ``1.7`` becomes ``17/10``.

    A decimal is read to the digits its precision carries, as SymPy prints it,
    so a typed decimal keeps every digit and the sum ``0.1 + 0.2`` is ``3/10``.
    Works on expressions and matrices alike.
    """
    decimals = expr.atoms(sympy.Float)
    if not decimals:
        return expr
    return expr.xreplace({decimal: sympy.Rational(str(decimal)) for decimal in decimals})


def normal_trig(expr: Expr) -> Expr:
    """Write every trigonometric function with ``sin`` and ``cos``, multiple angles expanded."""
    if not expr.has(sin, cos, tan, cot, sec, csc):
        return expr
    expr = expr.replace(tan, lambda a: sin(a) / cos(a)).replace(cot, lambda a: cos(a) / sin(a))
    expr = expr.replace(sec, lambda a: 1 / cos(a)).replace(csc, lambda a: 1 / sin(a))
    return sympy.expand_trig(expr)


def _reduce_pythagorean(polynomial: Expr) -> Expr:
    """Use ``cos²x = 1 - sin²x`` until no cosine has a power above one.

    A polynomial in ``sin x`` and ``cos x`` reduced this way is zero exactly when
    it vanishes by the Pythagorean identity.
    """

    def is_cos_power(e: Expr) -> bool:
        return bool(e.is_Pow and isinstance(e.base, cos) and e.exp.is_Integer and e.exp >= 2)

    def reduce(e: Expr) -> Expr:
        angle, n = e.base.args[0], int(e.exp)
        return (1 - sin(angle) ** 2) ** (n // 2) * cos(angle) ** (n % 2)

    return sympy.expand(polynomial.replace(is_cos_power, reduce))


def is_zero(expr: object) -> bool:
    """Whether an expression is identically zero, decided exactly for rational
    functions of the variables, ``sin`` and ``cos``. Decimals count as the exact
    fractions they denote."""
    expr = sympy.sympify(expr)
    if not isinstance(expr, sympy.Expr):
        return False
    exact = rationalise(expr)  # roundoff in decimals would never cancel
    if exact == 0:
        return True
    if exact.is_number and exact != 0:
        return False
    normal = normal_trig(exact)
    if _has_denominator(normal):  # combine into one fraction; its numerator decides
        numerator = sympy.expand(sympy.fraction(sympy.together(normal))[0])
    else:  # a polynomial in its atoms: expanding it decides, with no common denominator
        numerator = sympy.expand(normal)
    if numerator == 0:
        return True
    return bool(numerator.has(cos) and _reduce_pythagorean(numerator) == 0)


def _has_denominator(expr: Expr) -> bool:
    """Whether anything but a number divides part of ``expr``, as in ``x/r``."""
    return any(
        power.exp.is_negative and not power.base.is_number for power in expr.atoms(sympy.Pow)
    )


def simplify_component(expr: object) -> Expr:
    """Simplify one component.

    Vanishing components are recognised exactly and cheaply. The rest go to
    :func:`sympy.simplify`, except very large ones, which are combined into a
    single fraction with common factors pulled out, since a full simplification
    could take minutes.
    """
    expr = sympy.sympify(expr)
    if expr == 0 or not isinstance(expr, sympy.Expr) or expr.is_Atom:
        return expr
    if is_zero(expr):
        return sympy.S.Zero
    normal = normal_trig(expr)
    if sympy.count_ops(normal) > LARGE:
        return sympy.factor_terms(sympy.together(normal))
    return normal_trig(sympy.simplify(sympy.cancel(normal)))


def tidy(expr: object) -> Expr:
    """Cheaply show exact cancellations; otherwise keep the expression as written.

    Used on results the user did not ask to simplify: a component that
    vanishes becomes ``0``, one that is constant becomes the constant
    (``g_{ab} g^{ab}`` is ``4``), and a small expression is cancelled when that
    makes it shorter.
    """
    expr = sympy.sympify(expr)
    if not isinstance(expr, sympy.Expr) or expr.is_Atom or expr.is_number:
        return expr
    if is_zero(expr):
        return sympy.S.Zero
    if sympy.count_ops(expr) > LARGE:
        return expr
    cancelled = sympy.cancel(normal_trig(expr))
    if cancelled.is_number or sympy.count_ops(cancelled) < sympy.count_ops(expr):
        return cancelled
    return expr
