r"""The operations of abstract index notation, as the engines compute them.

An equation with a tensor that has no components, as
:math:`G_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G T_{\mu\nu}` before T has any, is kept as
notation: a SymPy tensor expression, from :mod:`relativisticpy.abstract`. The compiler
emits the operations below for it, ``notation.tensor``, ``notation.add``,
``notation.solve`` and the rest, whose types it knows from their rules
(:mod:`relativisticpy.typing.notation`); the engines compute them here, as they compute
any other operation, while the statement that writes them runs.

Each is a function of the values of its arguments, then of its attributes, by name. The
terms of notation, before they are made an expression or an equation, are SymPy's tensor
expressions and scalars; the values a program shows are
:class:`~relativisticpy.abstract.IndexExpression` and
:class:`~relativisticpy.abstract.IndexEquation`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import sympy
from sympy import Symbol
from sympy.tensor import tensor as symbolic

from relativisticpy import abstract
from relativisticpy.abstract import Algebra, IndexEquation, IndexExpression
from relativisticpy.errors import CompileError, IndexNotationError
from relativisticpy.tensors import Indices

if TYPE_CHECKING:
    from relativisticpy.ir import Node

Kept = IndexEquation | IndexExpression


# -- terms ------------------------------------------------------------------------------


def algebra(*, dim: int, metric: str, number: int) -> Algebra:
    """The tensors notation is written with: their indices, the metric and a head for
    every tensor, in ``dim`` dimensions (see :class:`~relativisticpy.ir.NotationAlgebra`)."""
    return Algebra(dim, metric=metric)


def tensor(space: Algebra, *, name: str, indices: Indices, quantity: str | None = None) -> Any:
    """A tensor with its indices, as :math:`T^{\\mu}{}_{\\nu}`; ``quantity`` names a
    built-in one, whose symmetries it has."""
    head = space.head(name, len(indices), quantity)
    try:
        return head(*(space.index(index.name, index.upper) for index in indices))
    except ValueError as error:
        raise IndexNotationError(str(error)) from None


def ricci_scalar(space: Algebra) -> Any:
    """``R``, written without a metric: the Ricci scalar of notation."""
    return space.ricci_scalar


def scalar(value: Any) -> Any:
    """A scalar factor of notation, as :math:`8\\pi G`."""
    found = sympy.sympify(value)
    if not isinstance(found, sympy.Expr):
        raise IndexNotationError(
            "in tensor notation, each factor is a scalar or a tensor with indices"
        )
    return found


def add(*terms: Any, signs: tuple[int, ...]) -> Any:
    """A sum of terms of notation, each with its sign. Their free indices must agree, as
    SymPy checks."""
    total: Any = sympy.S.Zero
    for sign, term in zip(signs, terms, strict=True):
        value = term if sign > 0 else -term
        try:
            total = value if total == 0 else total + value
        except ValueError as error:  # SymPy's own check of free indices
            raise IndexNotationError(f"the free indices of a sum must agree: {error}") from None
    return total


def mul(*factors: Any) -> Any:
    """A product of terms of notation, summing each index written once up and once down."""
    product: Any = sympy.S.One
    for factor in factors:
        try:
            product = product * factor
        except ValueError as error:
            raise IndexNotationError(str(error)) from None
    return product


def denominator(value: Any) -> Any:
    """What notation is divided by: a scalar, never a tensor."""
    if isinstance(value, symbolic.TensExpr):
        raise IndexNotationError("cannot divide by a tensor")
    return value


def divide(numerator: Any, by: Any) -> Any:
    return numerator / by


def negative(value: Any) -> Any:
    return -value


def value(kept: IndexExpression) -> Any:
    """What an expression kept as notation is, as a term of another."""
    return kept.value


# -- expressions and equations ----------------------------------------------------------------


def expression(space: Algebra, term: Any) -> IndexExpression:
    return IndexExpression(term, space)


def equation(space: Algebra, lhs: Any, rhs: Any) -> IndexEquation:
    return IndexEquation(lhs, rhs, space)


def unknown(subject: IndexEquation, target: Any) -> Any:
    """What ``solve`` is asked to solve for: a tensor with its indices, or a scalar that
    an equation of scalars is of."""
    if _of_scalars(subject, target):
        return target
    if not isinstance(target, symbolic.Tensor):
        raise CompileError(
            "give the tensor to solve for with its indices, as in T_{\\mu\\nu}, or a scalar "
            "the equation is of"
        )
    return target


def solve(subject: IndexEquation, target: Any, *, unknown: str) -> IndexEquation:
    """``solve(E, T_{\\mu\\nu})``: the equation solved for the tensor, in which it is
    linear; or, of an equation of scalars, for a scalar, written ``unknown``."""
    space = subject.space
    if _of_scalars(subject, target):  # solve(trace(EFE), R): an equation of scalars
        found = sympy.solve(subject.lhs - subject.rhs, target)
        if len(found) != 1:
            raise CompileError(f"the equation does not fix {unknown}")
        return IndexEquation(target, found[0], space)
    return IndexEquation(target, abstract.solve(subject.lhs, subject.rhs, target, space), space)


def _of_scalars(subject: IndexEquation, target: Any) -> bool:
    """Whether ``target`` is a scalar that ``subject``, an equation of scalars, is of."""
    sides = (subject.lhs, subject.rhs)
    return isinstance(target, Symbol) and not any(
        isinstance(side, symbolic.TensExpr) for side in sides
    )


def lhs(subject: IndexEquation) -> IndexExpression:
    return IndexExpression(subject.lhs, subject.space)


def rhs(subject: IndexEquation) -> IndexExpression:
    return IndexExpression(subject.rhs, subject.space)


def substitute(subject: Kept, rule: IndexEquation) -> Kept:
    """``subs(E, G_{ab} = R_{ab} - \\frac{1}{2} R g_{ab})``: the rule put into each side."""
    space = subject.space
    return _each_side(subject, lambda side: abstract.substitute(side, rule.lhs, rule.rhs, space))


def trace(subject: Kept) -> Kept:
    """The trace of each side, contracted with the metric."""
    return _each_side(subject, lambda side: abstract.trace(side, subject.space))


def simplify(subject: Kept) -> Kept:
    """Each side in its canonical form, with the identities of General Relativity."""
    return _each_side(subject, lambda side: abstract.canonical(side, subject.space))


def _each_side(subject: Kept, transform: Callable[[Any], Any]) -> Kept:
    if isinstance(subject, IndexEquation):
        return IndexEquation(transform(subject.lhs), transform(subject.rhs), subject.space)
    return IndexExpression(transform(subject.value), subject.space)


def rebase(space: Algebra, kept: Kept) -> Kept:
    """Notation kept with an earlier algebra, read with ``space``: in the dimensions and
    with the metric's name in force."""
    return abstract.rebased(kept, space)


# -- components ----------------------------------------------------------------------------


def lower(kept: IndexExpression, *references: Node, context: Any) -> Node:
    """The IR that computes the components of an expression of notation, made by the
    lowerer once the notation is known, from the ``references``, the nodes of the metric
    and the tensors the notation may be written with (see
    :func:`relativisticpy.compiler.compiler.lower_notation`)."""
    from relativisticpy.compiler.compiler import lower_notation

    return lower_notation(kept, references, context)
