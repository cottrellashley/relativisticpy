r"""The types of abstract index notation: what each operation on notation gives.

Notation, as :math:`G_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G T_{\mu\nu}` before T has
components, is computed by the engines (:mod:`relativisticpy.runtime.notation`). What the
compiler needs to know of it comes from these rules, from the types of an operation's
arguments: whether it is an equation, and the free indices of each side, in the order
they are written. A rule never computes.

The free indices a rule gives are those of the notation as written. Computing can lose
them only where the value vanishes, as ``simplify(R_{ab} - R_{ba})`` does: zero has
whatever indices it is given.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

from relativisticpy.ir import (
    SCALAR,
    VALUE,
    Node,
    NotationAlgebra,
    NotationType,
    ScalarType,
    TensorType,
    Type,
    dependencies,
    ops,
)
from relativisticpy.tensors import Index, Indices
from relativisticpy.tensors.spaces import Space


def contracted(indices: Iterable[Index]) -> Indices:
    """The free indices of what is written with ``indices``, in order: each label
    written once. A label written twice is summed, as in :math:`R^a{}_{bac}`."""
    written = list(indices)
    counts = Counter(index.name for index in written)
    return Indices(index for index in written if counts[index.name] == 1)


#: No indices: those of a scalar.
NONE = Indices()


def term(algebra: NotationAlgebra, free: Indices = NONE) -> NotationType:
    """A term of notation: a scalar, with no free indices, or what has ``free``."""
    return NotationType("term", (free,), algebra)


def tensor(algebra: NotationAlgebra, indices: Indices) -> NotationType:
    """``notation.tensor``: a tensor with its indices, as :math:`T^{\\mu}{}_{\\nu}`."""
    return term(algebra, contracted(indices))


def add(terms: Sequence[NotationType]) -> NotationType:
    """``notation.add``: a sum has the free indices of its terms, which agree; a term
    without any, as 0, has whatever the others have."""
    free = next((each.free for each in terms if each.free), NONE)
    return term(terms[0].algebra, free)


def mul(factors: Sequence[NotationType]) -> NotationType:
    """``notation.mul``: a product's free indices are those of its factors, less each
    written once up and once down, which is summed."""
    return term(factors[0].algebra, contracted(i for each in factors for i in each.free))


def quotient(numerator: NotationType) -> NotationType:
    """``notation.div``: what is divided by a scalar keeps its free indices."""
    return numerator


def value(kept: NotationType) -> NotationType:
    """``notation.value``: what an expression kept as notation is, as a term of another."""
    return term(kept.algebra, kept.free)


def expression(raw: NotationType) -> NotationType:
    """``notation.expression``: an expression of notation, as a value."""
    return NotationType("expression", (raw.free,), raw.algebra)


def equation(lhs: NotationType, rhs: NotationType) -> NotationType:
    """``notation.equation``: an equation of notation; each side has its own free
    indices, as a side written 0 has none."""
    return NotationType("equation", (lhs.free, rhs.free), lhs.algebra)


def solve(subject: NotationType, unknown: NotationType) -> NotationType:
    """``notation.solve``: the unknown, equal to what solves for it, which has its free
    indices, in its positions."""
    return NotationType("equation", (unknown.free, unknown.free), subject.algebra)


def side(subject: NotationType, which: int) -> NotationType:
    """``notation.lhs`` and ``notation.rhs``: a side of an equation, as an expression."""
    return NotationType("expression", (subject.sides[which],), subject.algebra)


def kept(subject: NotationType) -> NotationType:
    """``notation.subs`` and ``notation.simplify``: what is substituted in or simplified
    keeps its free indices."""
    return subject


def trace(subject: NotationType) -> NotationType:
    """``notation.trace``: a trace contracts the two free indices of each side."""
    return NotationType(subject.role, tuple(NONE for _ in subject.sides), subject.algebra)


def rebased(subject: NotationType, algebra: NotationAlgebra) -> NotationType:
    """``notation.rebase``: notation kept in one algebra, read in another."""
    return NotationType(subject.role, subject.sides, algebra)


def components(free: Indices, spaces: Sequence[Space]) -> TensorType | ScalarType:
    """``notation.components``: the components of an expression of notation with the
    ``free`` indices, in the order they are written, over ``spaces``: a tensor, but for
    each index named after an element of its space, as a coordinate's name is in
    spacetime, which is that component; a scalar, without free indices."""
    kept = [
        (index, space)
        for index, space in zip(free, spaces, strict=True)
        if index.name not in (space.basis or ())
    ]
    if not kept:
        return SCALAR
    return TensorType(Indices(index for index, _ in kept), tuple(space for _, space in kept))


def notation_of(node: Node) -> NotationType | None:
    """The type of a value kept as notation, an expression or an equation; ``None`` for
    anything else, a term of notation included."""
    kind = node.type
    if isinstance(kind, NotationType) and kind.role != "term":
        return kind
    return None


#: What a scalar factor of notation is made with: constants, arithmetic, arrays of
#: scalars and the registry's functions, as :math:`8\pi G` or :math:`\sqrt{-g}` are.
_FACTORS = (ops.Const, ops.Add, ops.Mul, ops.Pow, ops.MakeArray, ops.Call)


def a_factor(node: Node) -> bool:
    """Whether ``node`` can be a scalar factor of notation: made of constants by
    arithmetic and functions, not computed from tensors, which notation does not read."""
    return all(isinstance(each.op, _FACTORS) for each in dependencies([node]))


def _passed(kind: Type) -> Type:
    """The type of what an operation gives of a value of this type, as a derivative of
    it: the same, but of notation, which only notation's own operations give."""
    return VALUE if isinstance(kind, NotationType) else kind
