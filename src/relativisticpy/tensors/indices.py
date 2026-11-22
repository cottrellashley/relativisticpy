r"""Tensor indices: a label and a position, upper (contravariant) or lower (covariant).

Indices are shown as they are written, in each of the forms of
:mod:`relativisticpy.display`: ``_{\mu\nu}`` as text, ``_{μν}`` in Unicode and
``_{\mu\nu}`` in LaTeX, never ``_{mu}_{nu}``::

    >>> indices = Indices("^{a}_{bc}")
    >>> str(indices), indices.unicode(), indices.latex()
    ('^{a}_{bc}', '^a_{bc}', '^{a}{}_{bc}')
    >>> print(Indices(r"_{\mu\nu}"))
    _{\mu\nu}
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from relativisticpy.language.vocabulary import GREEK

#: The forms indices are written in.
IndexForm = Literal["unicode", "text", "latex"]

_GREEK_LETTERS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "zeta": "ζ",
    "eta": "η", "theta": "θ", "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ",
    "nu": "ν", "xi": "ξ", "omicron": "ο", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ",
    "upsilon": "υ", "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π",
    "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}  # fmt: skip
_SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


@dataclass(frozen=True)
class Index:
    r"""One index of a tensor, such as the ``\mu`` in ``g_{\mu\nu}``: its label ``name``
    and whether it is ``upper``."""

    name: str
    upper: bool = False

    def __str__(self) -> str:
        return Indices([self]).text()

    def flip(self) -> Index:
        """The same label in the other position."""
        return Index(self.name, not self.upper)

    def latex(self) -> str:
        """The label as LaTeX: ``mu`` → ``\\mu``."""
        return latex_label(self.name)

    def unicode(self) -> str:
        """The label in Unicode: ``mu`` → ``μ``, ``mu'`` → ``μ′``."""
        return unicode_label(self.name)


class Indices(tuple[Index, ...]):
    r"""An ordered tuple of indices, made from :class:`Index` values or read from text
    as a tensor's indices are written: ``Indices("^{a}_{bc}")``.

    Neighbours in the same position are written together: ``str`` gives the text form,
    ``^{a}_{bc}``, :meth:`unicode` the form for a terminal, ``^a_{bc}``, and
    :meth:`latex` the typeset form, ``^{a}{}_{bc}``.
    """

    def __new__(cls, indices: Iterable[Index] | str = ()) -> Indices:
        if isinstance(indices, str):
            indices = parse(indices)
        items = tuple(indices)
        for item in items:
            if not isinstance(item, Index):
                raise TypeError(
                    f"Indices are made of Index values or read from text, as in "
                    f"Indices('^{{a}}_{{bc}}'), not of {type(item).__name__}"
                )
        return super().__new__(cls, items)

    def __str__(self) -> str:
        return self.text()

    def __repr__(self) -> str:
        return f"Indices({self.text()!r})"

    @property
    def names(self) -> tuple[str, ...]:
        """The labels, in order: ``('a', 'b', 'c')``."""
        return tuple(index.name for index in self)

    @property
    def positions(self) -> tuple[bool, ...]:
        """Whether each index is upper, in order."""
        return tuple(index.upper for index in self)

    def text(self, labels: Sequence[str] | None = None) -> str:
        r"""The indices as text the language reads: ``^{a}_{bc}``, ``_{\mu\nu}``.
        ``labels`` replaces the labels, already written as text."""
        return scripts(self.positions, self._labels("text", labels), "text")

    def unicode(self, labels: Sequence[str] | None = None) -> str:
        """The indices in Unicode: ``^a_{bc}``, ``_{μν}``."""
        return scripts(self.positions, self._labels("unicode", labels), "unicode")

    def latex(self, labels: Iterable[str] | None = None) -> str:
        """Typeset the indices, grouping neighbours in the same position.

        ``labels`` replaces the index names, already typeset, e.g. with coordinate
        names to typeset one component: ``^{t}{}_{tr}``.
        """
        return scripts(self.positions, self._labels("latex", labels), "latex")

    def _labels(self, form: IndexForm, labels: Iterable[str] | None) -> list[str]:
        if labels is not None:
            given = list(labels)
            if len(given) != len(self):
                raise ValueError(f"{len(self)} indices but {len(given)} labels")
            return given
        render = {"text": text_label, "unicode": unicode_label, "latex": latex_label}[form]
        return [render(index.name) for index in self]


def scripts(positions: Sequence[bool], labels: Sequence[str], form: IndexForm) -> str:
    r"""Labels written as scripts, neighbours in the same position together: ``^{a}_{bc}``
    as text, ``^a_{bc}`` in Unicode and ``^{a}{}_{bc}`` in LaTeX. The labels are already
    in the form, as ``\mu``, ``μ`` or ``\theta``."""
    groups: list[tuple[bool, list[str]]] = []
    for upper, label in zip(positions, labels, strict=True):
        if groups and groups[-1][0] == upper:
            groups[-1][1].append(label)
        else:
            groups.append((upper, [label]))
    rendered = []
    bare_before = False  # whether the last group was an unbraced ASCII letter or digit
    for upper, group in groups:
        mark = "^" if upper else "_"
        joined = join_labels(group, form)
        plain = joined.isascii() and joined.isalnum()
        # x^a_b would read as x^{a_b}: a letter after an unbraced one is braced
        if form == "unicode" and len(joined) == 1 and not (bare_before and plain):
            rendered.append(mark + joined)
            bare_before = plain
            continue
        rendered.append(f"{mark}{{{joined}}}")
        bare_before = False
    return ("{}" if form == "latex" else "").join(rendered)


def join_labels(labels: Sequence[str], form: IndexForm) -> str:
    r"""Labels side by side, as one group of scripts: ``\mu\nu`` and ``ab``, but ``\mu a``,
    since ``\mua`` would be a command of its own, and ``x1 x2`` or ``AAPL MSFT``, which
    would run together. LaTeX spaces labels of several letters with ``\,``."""
    text = ""
    positions = len(labels) > 1 and all(label.isdigit() for label in labels)  # N_{0 1}
    several = positions or any(_several(label, form) for label in labels)
    for label in labels:
        if text:
            if several:
                text += "\\," if form == "latex" else " "
            elif label[:1].isalpha() and re.search(r"\\[A-Za-z]+$", text):
                text += " "
        text += label
    return text


def _several(label: str, form: IndexForm) -> bool:
    """Whether a label is more than one letter, as written in a form: ``AAPL``, ``x1``
    or ``10``, but not ``\\mu``, ``μ′`` or ``α₁``."""
    core = label.rstrip("'′")
    if re.fullmatch(r"\\[A-Za-z]+", core) or len(core) <= 1:
        return False
    return not (form == "unicode" and all(c in "₀₁₂₃₄₅₆₇₈₉" for c in core[1:]))


def latex_label(name: str) -> str:
    """A symbol name as LaTeX: Greek names gain a backslash, ``x_0`` becomes ``x_{0}``
    and ``mu'`` becomes ``\\mu'``; a name of two, as the change ``Delta x``, ``\\Delta x``."""
    if name in GREEK:
        return "\\" + name
    head, space, rest = name.partition(" ")
    if space:
        return f"{latex_label(head)} {latex_label(rest)}"
    unprimed = name.rstrip("'")
    if unprimed and unprimed != name:
        return latex_label(unprimed) + name[len(unprimed) :]
    head, _, tail = name.partition("_")
    if tail:
        return f"{latex_label(head)}_{{{tail}}}"
    return name


def text_label(name: str) -> str:
    r"""An index label as text the language reads: ``mu`` → ``\mu``, ``mu'`` → ``\mu'``."""
    unprimed = name.rstrip("'")
    primes = name[len(unprimed) :]
    if unprimed in GREEK:
        return "\\" + unprimed + primes
    return name


def unicode_label(name: str) -> str:
    """An index label in Unicode: ``mu`` → ``μ``, ``mu'`` → ``μ′``, ``alpha_1`` → ``α₁``."""
    unprimed = name.rstrip("'")
    primes = "′" * (len(name) - len(unprimed))
    head, _, tail = unprimed.partition("_")
    shown = _GREEK_LETTERS.get(head, head)
    if tail:
        shown += tail.translate(_SUBSCRIPT_DIGITS) if tail.isdigit() else "_" + tail
    return shown + primes


def parse(text: str) -> Indices:
    r"""Read indices as a tensor's are written: ``"^{a}_{bc}"``, ``"_{\mu\nu}"``,
    ``"_{μν}"``, ``"_{mu nu}"`` or ``"^a{}_b"``. Anything else is a ``ValueError``."""
    from relativisticpy.errors import ParseError
    from relativisticpy.language.parser import parse_expression
    from relativisticpy.language.syntax import Scripted

    try:
        node = parse_expression("T " + text)  # the space keeps "_a" from joining the name
    except ParseError as error:
        raise ValueError(f"cannot read indices from {text!r}") from error
    if text and not isinstance(node, Scripted):
        raise ValueError(f"cannot read indices from {text!r}")
    result = []
    for script in node.scripts if isinstance(node, Scripted) else ():
        if script.indices is None:
            raise ValueError(f"cannot read indices from {text!r}")
        for item in script.indices.items:
            if item.label is None or item.value is not None:
                raise ValueError(f"index labels only, please: {text!r}")
            result.append(Index(str(item.label), script.kind == "^"))
    return Indices(result)
