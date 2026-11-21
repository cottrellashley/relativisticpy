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
    components hold when they agree to rounding error.
    """

    def __init__(
        self,
        lhs: Any,
        rhs: Any,
        indices: Indices | None = None,
        coordinates: Sequence[Symbol] = (),
        spaces: Sequence[Space] | None = None,
    ) -> None:
        if numerical.involves([lhs, rhs]):  # one side may be the number 0
            np = numerical.numpy()
            lhs, rhs = (numerical.dense(numerical.as_array(side)) for side in (lhs, rhs))
            lhs, rhs = np.broadcast_arrays(lhs, rhs)
        self.lhs = lhs
        self.rhs = rhs
        self.indices = indices
        self.coordinates = tuple(coordinates)
        self.spaces = tuple(spaces) if spaces is not None else None

    @property
    def rank(self) -> int:
        return arrays.rank(self.lhs)

    def components(self) -> dict[tuple[int, ...], Eq]:
        """Every component that does not hold identically, keyed by position."""
        if numerical.is_numerical(self.lhs):
            np = numerical.numpy()
            scale = max(1.0, float(np.max(np.abs(self.lhs), initial=0.0)))
            wrong = ~np.isclose(self.lhs, self.rhs, rtol=1e-9, atol=1e-12 * scale)
            return {
                tuple(int(i) for i in index): Eq(
                    sympy.sympify(self.lhs[tuple(index)]),
                    sympy.sympify(self.rhs[tuple(index)]),
                    evaluate=False,
                )
                for index in np.argwhere(wrong)
            }
        dim = self.lhs.shape[0] if self.rank else 0
        result: dict[tuple[int, ...], Eq] = {}
        for index in product(range(dim), repeat=self.rank):
            lhs, rhs = self.lhs[index], self.rhs[index]
            comparable = isinstance(lhs, sympy.Expr) and isinstance(rhs, sympy.Expr)
            if not (comparable and is_zero(lhs - rhs)):
                result[index] = Eq(lhs, rhs, evaluate=False)
        return result

    def subs(self, *args: Any) -> TensorEquation:
        """Substitute into both sides, as SymPy's ``subs`` does."""
        lhs, rhs = self.lhs.subs(*args), self.rhs.subs(*args)
        return TensorEquation(lhs, rhs, self.indices, self.coordinates)

    def equations(self) -> list[Eq]:
        """The component equations that do not hold identically, in order."""
        return list(self.components().values())

    def __bool__(self) -> bool:
        if numerical.is_numerical(self.lhs):
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
        if self.spaces is not None:
            return ",".join(
                space.basis[i] if space.basis else str(i)
                for i, space in zip(index, self.spaces, strict=True)
            )
        names = [str(c) for c in self.coordinates] or [str(i) for i in range(self.lhs.shape[0])]
        return ",".join(names[i] for i in index)

    def latex(self) -> str:
        components = self.components()
        if not components:
            return r"\mathrm{True}"
        names = [latex_label(str(c)) for c in self.coordinates] or [str(i) for i in range(99)]
        lines = []
        for index, eq in components.items():
            labels = [names[i] for i in index]
            head = self.indices.latex(labels) if self.indices else ",".join(labels)
            lines.append(f"{head}: &\\; {sympy.latex(eq)}")
        return "\\begin{aligned}" + " \\\\ ".join(lines) + "\\end{aligned}"

    def _repr_latex_(self) -> str:
        return f"$\\displaystyle {self.latex()}$"
