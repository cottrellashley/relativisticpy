r"""Spaces: what an index ranges over.

In General Relativity every index ranges over spacetime, whose basis is the
coordinates. Index notation never needed spacetime, though: an index may range
over 500 assets, 2,520 trading days or the two states of a qubit. A tensor's type
records the space of each of its axes, so that an index is only ever summed over one
space, and components can be named by the basis of theirs, as ``g_{tt}`` names a
component by the coordinate ``t``::

    >>> assets = Space("assets", 3, basis=("AAPL", "MSFT", "NVDA"))
    >>> assets.dim, assets.position("MSFT")
    (3, 1)

A space's metric, which raises and lowers its indices, and its measure, which
expectations average with, belong to the session the space is used in: a covariance
can be made the metric of the assets after the fact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from sympy import Symbol


@dataclass(frozen=True)
class Space:
    """What an index ranges over: its ``name``, its dimension ``dim``, optionally the
    names of its ``basis`` and whether its components are ``real`` or ``complex``.
    Spacetime, the space of the coordinates, is marked as such: its metric is the
    metric in force."""

    name: str
    dim: int
    basis: tuple[str, ...] | None = None
    field: Literal["real", "complex"] = "real"
    spacetime: bool = False

    def __post_init__(self) -> None:
        if self.dim < 0:
            raise ValueError(f"a space cannot have dimension {self.dim}")
        if self.basis is not None and len(self.basis) != self.dim:
            raise ValueError(
                f"the space {self.name} has dimension {self.dim} but {len(self.basis)} names"
            )

    def __str__(self) -> str:
        return self.name

    def position(self, name: str) -> int | None:
        """The number of the basis element called ``name``, if there is one."""
        if self.basis is not None and name in self.basis:
            return self.basis.index(name)
        return None


def spacetime(coordinates: Sequence[Symbol]) -> Space:
    """The space of the coordinates: its basis is the coordinates' names."""
    return Space("spacetime", len(coordinates), tuple(str(c) for c in coordinates), spacetime=True)
