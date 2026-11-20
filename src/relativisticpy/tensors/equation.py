"""Equations between tensors."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from typing import Any

import sympy
from sympy import Eq, Symbol

from relativisticpy.tensors.indices import Indices, latex_label
from relativisticpy.tensors.simplification import is_zero


class TensorEquation:
    r"""An equation between two tensors with the same free indices, such as the
    vacuum field equations :math:`R_{\mu\nu} = 0`.

    It stands for one equation per component; :meth:`equations` lists those that
    do not hold identically, keyed by coordinate names, so they can be solved::

        R_{tt} = 0, R_{rr} = 0, ...

    An equation whose every component holds is ``True`` in a boolean context.
    """

    def __init__(
        self,
        lhs: Any,
        rhs: Any,
        indices: Indices | None = None,
        coordinates: Sequence[Symbol] = (),
    ) -> None:
        self.lhs = lhs
        self.rhs = rhs
        self.indices = indices
        self.coordinates = tuple(coordinates)

    @property
    def rank(self) -> int:
        return int(self.lhs.rank())

    def components(self) -> dict[tuple[int, ...], Eq]:
        """Every component that does not hold identically, keyed by position."""
        dim = self.lhs.shape[0] if self.rank else 0
        result: dict[tuple[int, ...], Eq] = {}
        for index in product(range(dim), repeat=self.rank):
            lhs, rhs = self.lhs[index], self.rhs[index]
            if not is_zero(lhs - rhs):
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
        return not self.components()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, bool):
            return bool(self) is other
        if not isinstance(other, TensorEquation):
            return NotImplemented
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
