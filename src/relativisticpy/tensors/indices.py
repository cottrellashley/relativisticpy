"""Tensor indices: a label and a position, upper (contravariant) or lower (covariant)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from relativisticpy.language.vocabulary import GREEK


@dataclass(frozen=True)
class Index:
    """One index of a tensor, such as the ``\\mu`` in ``g_{\\mu\\nu}``."""

    name: str
    upper: bool = False

    def __str__(self) -> str:
        return f"{'^' if self.upper else '_'}{{{self.name}}}"

    def flip(self) -> Index:
        """The same label in the other position."""
        return Index(self.name, not self.upper)

    def latex(self) -> str:
        """The label as LaTeX: ``mu`` → ``\\mu``."""
        return latex_label(self.name)


class Indices(tuple[Index, ...]):
    """An ordered tuple of indices.

    ``str`` gives the plain form RelativisticPy has always printed,
    ``^{a}_{b}_{c}``, and :meth:`latex` the typeset form, ``^{a}{}_{bc}``.
    """

    def __new__(cls, indices: Iterable[Index] = ()) -> Indices:
        return super().__new__(cls, indices)

    def __str__(self) -> str:
        return "".join(str(index) for index in self)

    def __repr__(self) -> str:
        return f"Indices({str(self)!r})"

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(index.name for index in self)

    def latex(self, labels: Iterable[str] | None = None) -> str:
        """Typeset the indices, grouping neighbours in the same position.

        ``labels`` replaces the index names, e.g. with coordinate names to
        typeset one component: ``^{t}{}_{t r}``.
        """
        names = list(labels) if labels is not None else [i.latex() for i in self]
        groups: list[tuple[bool, list[str]]] = []
        for index, name in zip(self, names, strict=True):
            if groups and groups[-1][0] == index.upper:
                groups[-1][1].append(name)
            else:
                groups.append((index.upper, [name]))
        rendered = [
            ("^" if upper else "_") + "{" + " ".join(group) + "}" for upper, group in groups
        ]
        return "{}".join(rendered)


def latex_label(name: str) -> str:
    """A symbol name as LaTeX: Greek names gain a backslash, ``x_0`` becomes ``x_{0}``."""
    if name in GREEK:
        return "\\" + name
    head, _, tail = name.partition("_")
    if tail:
        return f"{latex_label(head)}_{{{tail}}}"
    return name
