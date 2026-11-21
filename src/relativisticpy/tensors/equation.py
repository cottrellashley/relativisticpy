"""Equations between tensors."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from typing import Any

import sympy
from sympy import Eq, Symbol

from relativisticpy.tensors import arrays, numerical
from relativisticpy.tensors.indices import Indices, latex_label
from relativisticpy.tensors.simplification import is_zero
from relativisticpy.tensors.spaces import Space


class TensorEquation:
    r"""An equation between two tensors with the same free indices, such as the
    vacuum field equations :math:`R_{\mu\nu} = 0`.

    It stands for one equation per component; :meth:`equations` lists those that
    do not hold identically, keyed by coordinate names, so they can be solved::

        R_{tt} = 0, R_{rr} = 0, ...

    An equation whose every component holds is ``True`` in a boolean context. Numerical
    components hold when they agree to rounding error, relative to each other or to
    ``scale``, the magnitude of the numbers the sides were computed from, which is that
    of the sides unless given; while an equation is decided, it is the magnitude of the
    terms of a side that is a sum (see :func:`~relativisticpy.tensors.numerical.close`).
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

    @property
    def rank(self) -> int:
        return arrays.rank(self.lhs)

    def components(self) -> dict[tuple[int, ...], Eq]:
        """Every component that does not hold identically, keyed by position."""
        if numerical.is_numerical(self.lhs):
            np = numerical.numpy()
            magnitudes = (numerical.magnitude(side) for side in (self.lhs, self.rhs))
            atol = numerical.ROUNDING * max(self.scale, *magnitudes)
            wrong = ~np.isclose(self.lhs, self.rhs, rtol=numerical.RELATIVE, atol=atol)
            return {
                tuple(int(i) for i in index): Eq(
                    sympy.sympify(self.lhs[tuple(index)]),
                    sympy.sympify(self.rhs[tuple(index)]),
                    evaluate=False,
                )
                for index in np.argwhere(wrong)
            }
        shape = tuple(self.lhs.shape) if self.rank else ()
        result: dict[tuple[int, ...], Eq] = {}
        for index in product(*(range(n) for n in shape)):
            lhs, rhs = self.lhs[index], self.rhs[index]
            comparable = isinstance(lhs, sympy.Expr) and isinstance(rhs, sympy.Expr)
            if not (comparable and is_zero(lhs - rhs)):
                result[index] = Eq(lhs, rhs, evaluate=False)
        return result

    def subs(self, *args: Any) -> TensorEquation:
        """Substitute into both sides, as SymPy's ``subs`` does."""
        lhs, rhs = self.lhs.subs(*args), self.rhs.subs(*args)
        return TensorEquation(lhs, rhs, self.indices, self.coordinates, self.spaces)

    def equations(self) -> list[Eq]:
        """The component equations that do not hold identically, in order."""
        return list(self.components().values())

    def __bool__(self) -> bool:
        if numerical.is_numerical(self.lhs):
            with numerical.operands(self.scale):
                return numerical.close(self.lhs, self.rhs)
        return not self.components()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, bool):
            return bool(self) is other
        if not isinstance(other, TensorEquation):
            return NotImplemented
        if numerical.involves([self.lhs, other.lhs]):
            return numerical.close(self.lhs, other.lhs) and numerical.close(self.rhs, other.rhs)
        return bool(self.lhs == other.lhs and self.rhs == other.rhs)

    __hash__ = None  # type: ignore[assignment]

    def __str__(self) -> str:
        components = self.components()
        if not components:
            return "True"
        return "\n".join(f"{self._label(index)}: {eq}" for index, eq in components.items())

    def __repr__(self) -> str:
        return f"TensorEquation({self.indices or ''}, {len(self.components())} equations)"

    def _label(self, index: tuple[int, ...]) -> str:
        return ",".join(self._names(index))

    def _names(self, index: tuple[int, ...]) -> list[str]:
        """The names of the component at ``index``: from the bases of the spaces of its
        axes, or the coordinates, or else its position along each axis."""
        if self.spaces is not None:
            return [
                space.basis[i] if space.basis else str(i)
                for i, space in zip(index, self.spaces, strict=True)
            ]
        coordinates = [str(c) for c in self.coordinates]
        return [coordinates[i] if coordinates else str(i) for i in index]

    def latex(self) -> str:
        components = self.components()
        if not components:
            return r"\mathrm{True}"
        lines = []
        for index, eq in components.items():
            labels = [latex_label(name) for name in self._names(index)]
            head = self.indices.latex(labels) if self.indices else ",".join(labels)
            lines.append(f"{head}: &\\; {sympy.latex(eq)}")
        return "\\begin{aligned}" + " \\\\ ".join(lines) + "\\end{aligned}"

    def _repr_latex_(self) -> str:
        return f"$\\displaystyle {self.latex()}$"


def _zero_shaped(side: Any, other: Any) -> Any:
    """``side``, or zeros of the shape of ``other`` where ``side`` is zero and ``other`` an
    array of another shape: the zero of T_{ta} = 0 has T's shape, square or not."""
    if not isinstance(other, sympy.NDimArray) or getattr(side, "shape", None) == other.shape:
        return side
    if all(value == 0 for value in arrays.flatten(side)):
        return sympy.ImmutableDenseNDimArray.zeros(*other.shape)
    return side
