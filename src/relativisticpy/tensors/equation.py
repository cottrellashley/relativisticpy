r"""Equations between tensors, and whether a claim holds.

A :class:`TensorEquation`, such as the vacuum field equations :math:`R_{\mu\nu} = 0`
of a metric, stands for one equation per component. It says whether it holds and, if
not, where::

    >>> import sympy
    >>> r, M = sympy.symbols("r M")
    >>> zero = sympy.ImmutableDenseNDimArray([[0, 0], [0, 0]])
    >>> g = sympy.ImmutableDenseNDimArray([[2*M/r - 1, 0], [0, r/(r - 2*M)]])
    >>> equation = TensorEquation(zero, g, coordinates=sympy.symbols("t r"),
    ...                           source=r"R_{\mu\nu} = g_{\mu\nu}")
    >>> equation.holds()
    False
    >>> print(equation)
    R_{μν} = g_{μν}: differs in 2 of 3 components
      R_{tt} = 0 but g_{tt} = 2M/r − 1
      R_{rr} = 0 but g_{rr} = r/(r − 2M)

:func:`holds` answers the same question of any result: an equation of tensors, of
index notation, or of scalars.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import sympy
from sympy import Eq, Symbol
from sympy.core.function import AppliedUndef

from relativisticpy.errors import RelativisticPyError
from relativisticpy.tensors import arrays, numerical
from relativisticpy.tensors.indices import Indices
from relativisticpy.tensors.simplification import is_zero
from relativisticpy.tensors.spaces import Space


class UndecidedError(RelativisticPyError, TypeError):
    """An equation was asked for a truth value it does not have, or a program claims one
    that is not decided: SymPy could not show its sides equal, nor numbers show that they
    differ. ``holds()`` says ``None``."""


@dataclass
class Failure:
    """Components where the sides of an equation differ, or are not shown to agree: one
    component, or several that are equal, or opposite, by an exchange of indices.

    ``members`` are the sign and the position of each, the first first; ``lhs`` and
    ``rhs`` the sides' values at the first; ``differs`` whether numbers show the sides to
    differ (``None`` if they could not be computed, ``False`` if they agree where tried).
    """

    members: list[tuple[int, tuple[int, ...]]]
    lhs: Any
    rhs: Any
    differs: bool | None = True

    @property
    def index(self) -> tuple[int, ...]:
        return self.members[0][1]


@dataclass
class Analysis:
    """What became of each component of an equation: how many there are, counting those
    equal by a symmetry of both sides once, and where the sides differ."""

    holds: bool | None
    total: int
    failures: list[Failure] = field(default_factory=list)
    failing: int = 0  # independent components that differ, or are not shown to agree
    differing: int = 0  # of those, the components that numbers show to differ
    largest: tuple[tuple[int, ...], float] | None = None  # numbers: the worst, and by how much


class TensorEquation:
    r"""An equation between two tensors with the same free indices, such as the
    vacuum field equations :math:`R_{\mu\nu} = 0`.

    It stands for one equation per component. :meth:`holds` says whether every
    component holds: ``True``, ``False``, or ``None`` when that could not be decided;
    ``bool()`` agrees with it, and is an :class:`UndecidedError` where it is ``None``.
    :meth:`failures` gives the components where the sides differ, keyed by coordinate
    names, and :meth:`equations` their equations, to solve::

        R_{tt} = 0, R_{rr} = 0, ...

    Numerical components hold when they agree to rounding error, relative to each other
    or to ``scale``, the magnitude of the numbers the sides were computed from, which is
    that of the sides unless given; while an equation is decided, it is the magnitude of
    the terms of a side that is a sum (see :func:`~relativisticpy.tensors.numerical.close`).

    ``source`` is the equation as written, ``R_{\mu\nu} = g_{\mu\nu}``, by which its
    sides and their components are named when it is shown.
    """

    def __init__(
        self,
        lhs: Any,
        rhs: Any,
        indices: Indices | None = None,
        coordinates: Sequence[Symbol] = (),
        spaces: Sequence[Space] | None = None,
        *,
        scale: float | None = None,
        source: str | None = None,
    ) -> None:
        if numerical.involves([lhs, rhs]):  # one side may be the number 0
            np = numerical.numpy()
            lhs, rhs = (numerical.dense(numerical.as_array(side)) for side in (lhs, rhs))
            lhs, rhs = np.broadcast_arrays(lhs, rhs)
        else:  # a side that is zero has the shape of the other
            lhs, rhs = _zero_shaped(lhs, rhs), _zero_shaped(rhs, lhs)
        self.lhs = lhs
        self.rhs = rhs
        self.indices = indices
        self.coordinates = tuple(coordinates)
        self.spaces = tuple(spaces) if spaces is not None else None
        self.scale = numerical.operand_scale() if scale is None else scale
        self.source = source
        self._analysis: Analysis | None = None

    @property
    def rank(self) -> int:
        """The number of free indices: 2 for :math:`R_{\\mu\\nu} = 0`."""
        return arrays.rank(self.lhs)

    @property
    def shape(self) -> tuple[int, ...]:
        """The number of components along each axis."""
        return tuple(int(n) for n in getattr(self.lhs, "shape", ()))

    @property
    def numerical(self) -> bool:
        """Whether the sides are numbers in arrays, rather than SymPy's."""
        return numerical.is_numerical(self.lhs)

    # -- does it hold? -----------------------------------------------------------

    def holds(self) -> bool | None:
        """Whether every component holds: ``True`` when that is shown, ``False`` when the
        sides differ in some component, and ``None`` when SymPy could not show them equal
        nor numbers that they differ."""
        return self.analysis().holds

    def __bool__(self) -> bool:
        verdict = self.holds()
        if verdict is None:
            from relativisticpy import display

            raise UndecidedError(
                f"whether {display.unicode(self).splitlines()[0]} is not decided",
                hint="ask equation.holds(), which is None for an equation not decided, or "
                "simplify its components first",
            )
        return verdict

    def analysis(self) -> Analysis:
        """What became of each component, worked out once: see :class:`Analysis`."""
        if self._analysis is None:
            self._analysis = _numerical(self) if self.numerical else _symbolic(self)
        return self._analysis

    # -- where it does not hold --------------------------------------------------

    def components(self) -> dict[tuple[int, ...], Eq]:
        """Every component that does not hold identically, keyed by position."""
        if self.numerical:
            np = numerical.numpy()
            return {
                tuple(int(i) for i in index): Eq(
                    sympy.sympify(self.lhs[tuple(index)]),
                    sympy.sympify(self.rhs[tuple(index)]),
                    evaluate=False,
                )
                for index in np.argwhere(self._wrong())
            }
        shape = self.shape
        result: dict[tuple[int, ...], Eq] = {}
        for index in product(*(range(n) for n in shape)):
            lhs, rhs = self.lhs[index], self.rhs[index]
            if not _shown_equal(lhs, rhs):
                result[index] = Eq(lhs, rhs, evaluate=False)
        return result

    def failures(self) -> dict[tuple[Any, ...], tuple[Any, Any]]:
        """The components where the sides are not shown equal, keyed by the names of
        their coordinates (or of the elements of their spaces' bases, or else their
        positions), as :meth:`Tensor.nonzero <relativisticpy.tensors.Tensor.nonzero>`
        keys components: ``{('t', 't'): (0, 2*M/r - 1), ...}``."""
        return {
            tuple(self.labels(axis)[i] for axis, i in enumerate(index)): (eq.lhs, eq.rhs)
            for index, eq in self.components().items()
        }

    def labels(self, axis: int) -> list[Any]:
        """The names of the components along an axis: the basis of its space, the
        coordinates, or else positions."""
        space = self.spaces[axis] if self.spaces is not None else None
        if self.coordinates and (space is None or space.spacetime):
            return [str(c) for c in self.coordinates]
        if space is not None and space.basis is not None:
            return list(space.basis)
        return list(range(self.shape[axis]))

    def subs(self, *args: Any) -> TensorEquation:
        """Substitute into both sides, as SymPy's ``subs`` does."""
        lhs, rhs = self.lhs.subs(*args), self.rhs.subs(*args)
        return TensorEquation(
            lhs, rhs, self.indices, self.coordinates, self.spaces, source=self.source
        )

    def equations(self) -> list[Eq]:
        """The component equations that do not hold identically, in order."""
        return list(self.components().values())

    def _wrong(self) -> Any:
        """Numerical sides: where they differ beyond rounding error."""
        np = numerical.numpy()
        magnitudes = (numerical.magnitude(side) for side in (self.lhs, self.rhs))
        atol = numerical.ROUNDING * max(self.scale, *magnitudes)
        return ~np.isclose(self.lhs, self.rhs, rtol=numerical.RELATIVE, atol=atol)

    # -- comparison and display ----------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if isinstance(other, bool):
            return self.holds() is other
        if not isinstance(other, TensorEquation):
            return NotImplemented
        if numerical.involves([self.lhs, other.lhs]):
            return numerical.close(self.lhs, other.lhs) and numerical.close(self.rhs, other.rhs)
        return bool(self.lhs == other.lhs and self.rhs == other.rhs)

    __hash__ = None  # type: ignore[assignment]

    # The forms of relativisticpy.display, by which every surface shows an equation.

    def _display_unicode(self) -> str:
        from relativisticpy import display

        return display.equation_form(self, "unicode")

    def _display_text(self) -> str:
        from relativisticpy import display

        return display.equation_form(self, "text")

    def _display_latex(self) -> str:
        from relativisticpy import display

        return display.equation_form(self, "latex")

    def _display_data(self) -> dict[str, Any]:
        from relativisticpy import display

        return display.equation_data(self)

    def __str__(self) -> str:
        from relativisticpy import display

        return display.unicode(self)

    __repr__ = __str__

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        printer.text(str(self))

    def latex(self) -> str:
        """Typeset the equation, and whether it holds: where not, each component that
        differs, with ``\\neq``."""
        from relativisticpy import display

        return display.latex(self)

    def _latex(self, printer: Any = None) -> str:
        return self.latex()

    def _repr_latex_(self) -> str:
        from relativisticpy import display

        return display.jupyter_latex(self)

    def _display_(self) -> Any:
        """What marimo shows: the equation typeset, by :mod:`~relativisticpy.display`."""
        from relativisticpy import display

        return display.marimo(self)


def _zero_shaped(side: Any, other: Any) -> Any:
    """``side``, or zeros of the shape of ``other`` where ``side`` is zero and ``other`` an
    array of another shape: the zero of T_{ta} = 0 has T's shape, square or not."""
    if not isinstance(other, sympy.NDimArray) or getattr(side, "shape", None) == other.shape:
        return side
    if all(value == 0 for value in arrays.flatten(side)):
        return sympy.ImmutableDenseNDimArray.zeros(*other.shape)
    return side


def _shown_equal(lhs: Any, rhs: Any) -> bool:
    """Whether two components are shown equal, as the sides of a scalar equation are:
    exactly, or where that fails, with hyperbolic functions written as exponentials and by
    a bounded simplification, as :math:`\\cosh^2 v - \\sinh^2 v = 1` is."""
    if not (isinstance(lhs, sympy.Expr) and isinstance(rhs, sympy.Expr)):
        return False
    if is_zero(lhs - rhs):
        return True
    from relativisticpy.runtime.kernels import simplifies_to_zero

    return simplifies_to_zero(lhs, rhs)


def _differs(lhs: Any, rhs: Any) -> bool | None:
    """Whether numbers show two components not shown equal to differ."""
    if not (isinstance(lhs, sympy.Expr) and isinstance(rhs, sympy.Expr)):
        return True
    from relativisticpy.runtime.kernels import differs

    try:
        return differs(lhs, rhs)
    except Exception:  # SymPy fails on some expressions: then it is not decided
        return None


def _classes(equation: TensorEquation) -> list[Any]:
    """For each axis, what it can be exchanged with: axes in the same position, of the
    same size, over the same space."""
    positions: Sequence[bool] = (
        [index.upper for index in equation.indices]
        if equation.indices is not None
        else [False] * equation.rank
    )
    spaces = equation.spaces or [None] * equation.rank
    return [(positions[axis], spaces[axis], equation.shape[axis]) for axis in range(equation.rank)]


def _symmetries(sides: Sequence[Any], classes: Sequence[Any]) -> list[tuple[int, int, int]]:
    """The exchanges of two axes that leave every side alike, ``(i, j, 1)``, or that negate
    every side, ``(i, j, -1)``."""
    found = []
    rank = len(classes)
    for i in range(rank):
        for j in range(i + 1, rank):
            if classes[i] != classes[j]:
                continue
            order = list(range(rank))
            order[i], order[j] = j, i
            swapped = [sympy.permutedims(side, order) for side in sides]
            if all(s == t for s, t in zip(sides, swapped, strict=True)):
                found.append((i, j, 1))
            elif all(s == -t for s, t in zip(sides, swapped, strict=True)):
                found.append((i, j, -1))
    return found


def _orbit(
    index: tuple[int, ...], symmetries: Sequence[tuple[int, int, int]]
) -> list[tuple[int, tuple[int, ...]]]:
    """The positions an index reaches by the symmetries, each with its sign."""
    reached = {index: 1}
    frontier = [index]
    while frontier:
        current = frontier.pop()
        for i, j, sign in symmetries:
            other = list(current)
            other[i], other[j] = other[j], other[i]
            candidate = tuple(other)
            if candidate not in reached:
                reached[candidate] = reached[current] * sign
                frontier.append(candidate)
    return [(sign, position) for position, sign in reached.items()]


def _symbolic(equation: TensorEquation) -> Analysis:
    shape = equation.shape
    sides = [equation.lhs, equation.rhs]
    symmetries = _symmetries(sides, _classes(equation)) if equation.rank > 1 else []
    seen: set[tuple[int, ...]] = set()
    total = 0
    failures = []
    for index in product(*(range(n) for n in shape)):
        if index in seen:
            continue
        orbit = _orbit(index, symmetries)
        seen.update(position for _, position in orbit)
        total += 1
        lhs, rhs = equation.lhs[index], equation.rhs[index]
        if _shown_equal(lhs, rhs):
            continue
        members = sorted(orbit, key=lambda member: member[1])  # index itself first
        failures.append(Failure(members, lhs, rhs, _differs(lhs, rhs)))
    differing = sum(1 for failure in failures if failure.differs)
    if not failures:
        holds: bool | None = True
    elif differing:
        holds = False
    else:
        holds = None
    return Analysis(holds, total, failures, len(failures), differing)


#: The most failing components an analysis of numbers keeps, to show.
SHOWN = 10


def _numerical(equation: TensorEquation) -> Analysis:
    np = numerical.numpy()
    with numerical.operands(equation.scale):
        holds = numerical.close(equation.lhs, equation.rhs)
    total = int(np.size(equation.lhs)) or 1
    if holds:
        return Analysis(True, total)
    wrong = equation._wrong()
    count = int(np.count_nonzero(wrong))
    difference = np.abs(np.asarray(equation.lhs) - np.asarray(equation.rhs))
    difference = np.where(wrong, difference, 0)
    worst = np.unravel_index(int(np.argmax(difference)), np.shape(difference))
    largest = (tuple(int(i) for i in worst), float(difference[worst]))
    failures = []
    for flat in np.flatnonzero(wrong)[:SHOWN]:
        index = tuple(int(i) for i in np.unravel_index(int(flat), np.shape(wrong)))
        failures.append(
            Failure([(1, index)], equation.lhs[index].item(), equation.rhs[index].item())
        )
    return Analysis(False, total, failures, count, count, largest)


def holds(value: Any) -> bool | None:
    """Whether a result that claims something holds: ``True``, ``False``, or ``None``
    where that is not decided. It answers for an equation of tensors, of index notation
    or of scalars, for a comparison such as ``x < 1``, and for a truth value::

        >>> import sympy
        >>> x = sympy.Symbol("x")
        >>> holds(sympy.Eq(x**2 - 1, (x - 1) * (x + 1))), holds(sympy.Eq(25.0, 24))
        (True, False)
        >>> holds(sympy.true), holds(sympy.Eq(sympy.Function("f")(x), 0))
        (True, False)

    A value with a method ``holds()``, such as an equation of tensors or of index
    notation, answers for itself; an expression in index notation, such as a simplified
    identity, is read as the claim that it vanishes. Anything else is no claim, and a
    ``TypeError``.
    """
    if isinstance(value, bool):
        return value
    method = getattr(value, "holds", None)
    if callable(method):
        verdict = method()
        return None if verdict is None else bool(verdict)
    if value is sympy.true or value is sympy.false:
        return value is sympy.true
    numpy_scalar = type(value).__module__.split(".")[0] == "numpy"
    if numpy_scalar and getattr(value, "shape", None) == () and value.dtype.kind == "b":
        return bool(value)
    if isinstance(value, sympy.Basic) and value.is_Relational:
        return _relation(value)
    from relativisticpy import display

    raise TypeError(
        f"holds() decides claims, such as equations, and {display.unicode(value)} is a "
        f"{type(value).__name__}, not a claim"
    )


def _relation(relation: Any) -> bool | None:
    """Whether a relation of scalars holds: an equation by its sides' difference, an
    inequality by SymPy, where it decides it."""
    if isinstance(relation, sympy.Equality | sympy.Unequality):
        lhs, rhs = relation.lhs, relation.rhs
        if not (isinstance(lhs, sympy.Expr) and isinstance(rhs, sympy.Expr)):
            equal: bool | None = bool(lhs == rhs)
        elif lhs.is_number and rhs.is_number:
            equal = _numbers_equal(lhs, rhs)
        elif _shown_equal(lhs, rhs):
            equal = True
        elif _asymptotic(relation):  # e^x \sin x = x + ... + O(x^5): no number decides it
            equal = None
        else:
            shown = _differs(lhs, rhs)
            equal = False if shown else None
        if equal is None:
            return None
        return equal if isinstance(relation, sympy.Equality) else not equal
    decided = relation.func(relation.lhs, relation.rhs)  # evaluated, where SymPy can
    if decided is sympy.true or decided is sympy.false:
        return decided is sympy.true
    return None


def _asymptotic(relation: Any) -> bool:
    """Whether a relation has an order term, SymPy's or Landau's O written as a function,
    as ``O(x^5)``: it says how a remainder grows, which no value at a point decides."""
    if relation.has(sympy.Order):
        return True
    return any(f.func.__name__ == "O" for f in relation.atoms(AppliedUndef))


def _numbers_equal(lhs: Any, rhs: Any) -> bool | None:
    """Whether two numbers are equal: exactly, or to rounding error if either is a float."""
    if lhs.has(sympy.Float) or rhs.has(sympy.Float):
        try:
            left, right = complex(lhs), complex(rhs)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(abs(value)) for value in (left, right)):
            return None
        return abs(left - right) <= numerical.RELATIVE * max(abs(left), abs(right))
    return bool(is_zero(lhs - rhs))
