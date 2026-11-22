r"""Abstract index notation: tensor equations kept, and solved, as tensor equations.

The component engine turns index notation into arrays of components at once.
This module keeps it as notation. :math:`G_{\mu\nu} + \Lambda g_{\mu\nu} =
8\pi G T_{\mu\nu}` stays an equation between tensors, which can be simplified,
traced and solved for :math:`T_{\mu\nu}` without a metric, and turned into
components later, when there is one.

Canonical forms under the tensors' symmetries and the renaming of dummy
indices come from :mod:`sympy.tensor.tensor`, which implements the
Butler–Portugal algorithm. The identities of General Relativity are added here:
:math:`R^a{}_a = R`, :math:`G^a{}_a = (1 - n/2) R`, a contraction of the
Riemann tensor is the Ricci tensor, the Weyl tensor is traceless, and the trace
of a tensor :math:`T_{ab}` is written :math:`T`.
"""

from __future__ import annotations

import itertools
import operator
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import reduce
from typing import Any

import sympy
from sympy import Expr, Symbol
from sympy.printing.latex import LatexPrinter
from sympy.tensor.tensor import (
    TensAdd,
    TensExpr,
    TensMul,
    Tensor,
    TensorHead,
    TensorIndex,
    TensorIndexType,
    TensorSymmetry,
)

from relativisticpy.display import _LatexPrinter, _PlainPrinter, text_name, unicode_name
from relativisticpy.errors import CompileError
from relativisticpy.tensors.indices import latex_label, scripts, text_label, unicode_label

#: A value in abstract index notation: a SymPy tensor expression, or a scalar.
Value = Any

_SYMMETRIES = {
    "riemann": lambda: TensorSymmetry.riemann(),
    "weyl": lambda: TensorSymmetry.riemann(),
    "ricci": lambda: TensorSymmetry.fully_symmetric(2),
    "einstein": lambda: TensorSymmetry.fully_symmetric(2),
    "christoffel": lambda: TensorSymmetry.direct_product(1, 2),
}
_RANKS = {"riemann": 4, "weyl": 4, "ricci": 2, "einstein": 2, "christoffel": 3}
# The labels dummy indices are given when shown, in order of preference.
_DUMMY_LABELS = (
    "alpha", "beta", "gamma", "lambda", "kappa", "rho", "sigma", "tau", "epsilon", "zeta",
    "eta", "iota", "xi", "omega", "chi", "psi",
)  # fmt: skip


class Algebra:
    """The tensors of a computation in ``dim`` dimensions: their indices, the metric
    and Kronecker delta, and a head for every tensor by name and rank.

    >>> Algebra(dim=3)
    Algebra(dim=3, metric g, Ricci scalar R, no tensors)
    """

    def __init__(self, dim: int = 4, *, metric: str = "g", ricci_scalar: str = "R") -> None:
        self.dim = dim
        self.type = TensorIndexType("L", dummy_name="L", dim=dim, metric_name=metric)
        self.metric = self.type.metric
        self.delta = self.type.delta
        self.ricci_scalar = Symbol(ricci_scalar)
        self._heads: dict[tuple[str, int], TensorHead] = {}
        self._quantities: dict[TensorHead, str] = {}
        self._indices: dict[str, TensorIndex] = {}

    def __repr__(self) -> str:
        names = sorted({name for name, _ in self._heads})
        tensors = (
            f"{len(names)} tensor{'s' if len(names) != 1 else ''}: {', '.join(names)}"
            if names
            else "no tensors"
        )
        return (
            f"Algebra(dim={self.dim}, metric {self.metric.name}, "
            f"Ricci scalar {self.ricci_scalar}, {tensors})"
        )

    def index(self, label: str, upper: bool = False) -> TensorIndex:
        """The index ``label``, upper or lower: ``algebra.index("a", upper=True)``."""
        index = self._indices.get(label)
        if index is None:
            index = self._indices[label] = TensorIndex(label, self.type)
        return index if upper else -index

    def head(self, name: str, rank: int, quantity: str | None = None) -> TensorHead:
        """The tensor called ``name`` with ``rank`` indices; ``quantity`` names a
        built-in one (``"riemann"``, ``"ricci"``, ...), whose symmetries it has."""
        if quantity == "metric":
            return self.metric
        if quantity == "kronecker":
            return self.delta
        key = (name, rank)
        head = self._heads.get(key)
        if head is None:
            make = _SYMMETRIES.get(quantity or "")
            symmetry = make() if make and _RANKS[quantity or ""] == rank else None
            head = TensorHead(name, [self.type] * rank, symmetry)
            self._heads[key] = head
            if quantity is not None:
                self._quantities[head] = quantity
        return head

    def quantity(self, head: TensorHead) -> str | None:
        return self._quantities.get(head)

    def traces(self) -> dict[Symbol, TensorHead | None]:
        """The scalars that stand for traces: the Ricci scalar R, and T for each tensor
        T_{ab} of one's own (with its head; R has none of its own to write)."""
        found: dict[Symbol, TensorHead | None] = {self.ricci_scalar: None}
        for head in self._heads.values():
            symbol = self.trace(head)
            if symbol is not None and symbol != self.ricci_scalar:
                found[symbol] = head
        return found

    def trace(self, head: TensorHead) -> Symbol | None:
        """The scalar standing for the trace of a tensor with two indices: ``R`` for the
        Ricci tensor, ``T`` for a tensor ``T_{ab}`` of one's own."""
        if len(head.index_types) != 2 or head in (self.metric, self.delta):
            return None
        quantity = self.quantity(head)
        if quantity == "ricci":
            return self.ricci_scalar
        if quantity is not None:
            return None
        return Symbol(head.name)


# -- simplification ------------------------------------------------------------------


def terms(value: Value) -> Iterator[tuple[Expr, list[Tensor]]]:
    """Each term of a value: its scalar coefficient and its tensor factors."""
    if isinstance(value, TensExpr):
        value = value.expand()  # a scalar times a sum, as written, is a sum of terms
    parts = value.args if isinstance(value, TensAdd) else (value,)
    for part in parts:
        if isinstance(part, Tensor):
            yield sympy.S.One, [part]
        elif isinstance(part, TensMul):
            yield part.coeff, [arg for arg in part.args if isinstance(arg, Tensor)]
        elif part != 0:
            yield sympy.sympify(part), []


def build(pieces: Sequence[tuple[Expr, Sequence[Value]]]) -> Value:
    """The sum of ``coefficient * factor * factor ...`` over ``pieces``."""
    total: Value = sympy.S.Zero
    for coefficient, factors in pieces:
        term = reduce(operator.mul, factors, coefficient)
        total = term if total == 0 else total + term
    return total


def canonical(value: Value, space: Algebra) -> Value:
    """A canonical form: expanded, with metrics and deltas contracted, the tensors'
    symmetries and dummy indices in canonical order, and the identities of General
    Relativity applied. Two values are equal exactly when their difference is 0."""
    return _without_cyclic_sums(_canonical(value, space), space)


def _canonical(value: Value, space: Algebra) -> Value:
    """The canonical form, but for sums the first Bianchi identity makes vanish where
    contracted indices keep :func:`_cyclic` from applying it."""
    for _ in range(16):  # each round can reveal another contraction
        for step in ("expand", "contract_metric", "contract_delta", "canon_bp"):
            if not isinstance(value, TensExpr):  # it has cancelled to a scalar
                return sympy.expand(value)
            if step == "contract_metric":
                value = value.contract_metric(space.metric)
            elif step == "contract_delta":
                value = value.contract_delta(space.delta)
            else:
                value = getattr(value, step)()
        if not isinstance(value, TensExpr):
            return sympy.expand(value)
        changed, value = _identities(value, space)
        if not changed:
            return value
    return value


def _identities(value: Value, space: Algebra) -> tuple[bool, Value]:
    """Replace contractions within a single tensor by what they are."""
    pieces = []
    changed = False
    for coefficient, tensors in terms(value):
        factors: list[Value] = []
        for tensor in tensors:
            replacement = _self_contraction(tensor, space)
            if replacement is None:
                replacement = _cyclic(tensor, space)
            if replacement is None:
                replacement = _mixed_metric(tensor, space)
            if replacement is None:
                factors.append(tensor)
            else:
                factors.append(replacement)
                changed = True
        pieces.append((coefficient, factors))
    return changed, build(pieces) if changed else value


def _self_contraction(tensor: Tensor, space: Algebra) -> Value | None:
    """What a tensor contracted with itself is, if an identity says: ``R^a_a = R``."""
    indices = list(tensor.indices)
    pair = next(
        (
            (i, j)
            for i, j in itertools.combinations(range(len(indices)), 2)
            if indices[i] == -indices[j]
        ),
        None,
    )
    if pair is None:
        return None
    quantity = space.quantity(tensor.head)
    if quantity == "weyl":
        return sympy.S.Zero
    if quantity == "einstein":
        return (1 - sympy.Rational(space.dim, 2)) * space.ricci_scalar
    if quantity == "riemann":
        # Ric_{bd} = R^a_{bad}: contracting slots (0, 2); the others follow by symmetry.
        signs = {(0, 2): 1, (1, 3): 1, (0, 3): -1, (1, 2): -1, (0, 1): 0, (2, 3): 0}
        sign = signs[pair]
        if sign == 0:
            return sympy.S.Zero
        rest = [indices[k] for k in range(4) if k not in pair]
        return sign * space.head("R", 2, "ricci")(*rest)
    trace = space.trace(tensor.head)
    return trace if trace is not None else None


def _mixed_metric(tensor: Tensor, space: Algebra) -> Value | None:
    """The metric with one index up and one down is the Kronecker delta."""
    if tensor.head != space.metric:
        return None
    first, second = tensor.indices
    if first.is_up == second.is_up:
        return None
    return space.delta(first, second) if first.is_up else space.delta(second, first)


def _cyclic(tensor: Tensor, space: Algebra) -> Value | None:
    """The first Bianchi identity, :math:`R_{a[bcd]} = 0`, as a choice of basis.

    Four distinct indices can be paired in three ways, and the identity relates
    the three. Writing them in an order, w1 < w2 < w3 < w4, the pairing of w1
    with w4 is replaced: :math:`R_{w_1 w_4 w_2 w_3} = R_{w_1 w_3 w_2 w_4} - R_{w_1 w_2 w_3 w_4}`.
    Pairings survive every one-term symmetry, so this terminates. It is applied
    where all four indices are free, so that renaming dummies cannot undo it."""
    if space.quantity(tensor.head) not in ("riemann", "weyl"):
        return None
    indices = list(tensor.indices)
    names = [index.name for index in indices]
    if len(set(names)) != 4 or any(name.startswith("L_") for name in names):
        return None
    w1, w2, w3, w4 = sorted(names)
    first, second = {names[0], names[1]}, {names[2], names[3]}
    if {w1, w4} not in (first, second):
        return None
    # The sign taking the tensor, by one-term symmetries, to R(w1, w4, w2, w3).
    pairs = (names[0:2], names[2:4]) if w1 in first else (names[2:4], names[0:2])
    sign = (1 if pairs[0][0] == w1 else -1) * (1 if pairs[1][0] == w2 else -1)
    by_name = {index.name: index for index in indices}
    head = tensor.head
    kept = head(*(by_name[n] for n in (w1, w3, w2, w4))) - head(
        *(by_name[n] for n in (w1, w2, w3, w4))
    )
    return sign * kept


def _without_cyclic_sums(value: Value, space: Algebra) -> Value:
    """``value`` without the sums of terms that vanish by the first Bianchi identity where
    some indices of a Riemann or Weyl tensor are contracted, as in
    :math:`R_{\\rho\\sigma\\mu\\nu} u^\\nu + R_{\\rho\\mu\\nu\\sigma} u^\\nu
    + R_{\\rho\\nu\\sigma\\mu} u^\\nu`, where :func:`_cyclic` does not apply.

    Only terms with the same tensors can cancel one another, so the terms are taken in
    groups of those. A group is dropped where it vanishes once each Riemann or Weyl
    tensor is written as its projection onto the solutions of the identity,
    :math:`R_{abcd} = \\tfrac23 R_{abcd} - \\tfrac13 R_{adbc} + \\tfrac13 R_{acbd}`: an
    identity itself, which canonical forms see through. Nothing that does not vanish is
    dropped."""
    if not isinstance(value, TensExpr):
        return value
    groups: dict[tuple[tuple[str, int], ...], list[tuple[Expr, list[Tensor]]]] = {}
    for coefficient, tensors in terms(value):
        key = tuple(sorted((t.head.name, len(t.head.index_types)) for t in tensors))
        groups.setdefault(key, []).append((coefficient, tensors))
    kept: list[tuple[Expr, list[Tensor]]] = []
    for group in groups.values():
        contracted = any(
            space.quantity(t.head) in ("riemann", "weyl")
            and any(i.name.startswith("L_") for i in t.indices)
            for _, tensors in group
            for t in tensors
        )
        if len(group) > 1 and contracted and _canonical(_projected(group, space), space) == 0:
            continue
        kept.extend(group)
    return value if len(kept) == sum(map(len, groups.values())) else build(kept)


def _projected(pieces: list[tuple[Expr, list[Tensor]]], space: Algebra) -> Value:
    """Terms with each Riemann or Weyl tensor replaced by its projection onto the
    solutions of the first Bianchi identity, which it equals."""
    third = sympy.Rational(1, 3)
    projected = []
    for coefficient, tensors in pieces:
        factors: list[Value] = []
        for tensor in tensors:
            if space.quantity(tensor.head) not in ("riemann", "weyl"):
                factors.append(tensor)
                continue
            m, n, p, q = tensor.indices
            head = tensor.head
            factors.append(
                2 * third * head(m, n, p, q) - third * head(m, q, n, p) + third * head(m, p, n, q)
            )
        projected.append((coefficient, factors))
    return build(projected)


# -- operations --------------------------------------------------------------------------


def rebased(
    kept: IndexEquation | IndexExpression, space: Algebra
) -> IndexEquation | IndexExpression:
    """Notation written with the tensors of one algebra, with those of ``space``: the
    same tensors and indices, by name. Notation kept before the coordinates change, or
    the metric is renamed, is read in the dimensions and names in force."""
    if kept.space is space:
        return kept
    if isinstance(kept, IndexEquation):
        lhs, rhs = (_rebased(side, kept.space, space) for side in (kept.lhs, kept.rhs))
        return IndexEquation(lhs, rhs, space)
    return IndexExpression(_rebased(kept.value, kept.space, space), space)


def _rebased(value: Value, old: Algebra, new: Algebra) -> Value:
    pieces = []
    for coefficient, tensors in terms(value):
        factors = []
        for tensor in tensors:
            if tensor.head == old.metric:
                head = new.metric
            elif tensor.head == old.delta:
                head = new.delta
            else:
                rank = len(tensor.head.index_types)
                head = new.head(tensor.head.name, rank, old.quantity(tensor.head))
            factors.append(head(*(new.index(i.name, i.is_up) for i in tensor.indices)))
        pieces.append((coefficient, factors))
    return build(pieces)


def free(value: Value) -> list[TensorIndex]:
    return list(value.get_free_indices()) if isinstance(value, TensExpr) else []


def positioned(value: Value, wanted: Sequence[TensorIndex], space: Algebra) -> Value:
    """``value`` with its free indices raised or lowered to the positions in ``wanted``."""
    for index in free(value):
        target = next((w for w in wanted if w.name == index.name), None)
        if target is None or target.is_up == index.is_up:
            continue
        fresh = TensorIndex(f"{index.name}_moved", space.type)
        base = TensorIndex(index.name, space.type)
        # Contract with the metric, which leaves the index in the other position, renamed.
        value = value * (
            space.metric(fresh, -index) if target.is_up else space.metric(-fresh, -index)
        )
        value = value.substitute_indices((fresh, base))
    return canonical(value, space)


def trace(value: Value, space: Algebra) -> Value:
    """Contract a value's two free indices with the metric. Zero, which has whatever
    indices the other side of its equation has, traces to zero."""
    if not isinstance(value, TensExpr) and value == 0:
        return sympy.S.Zero
    indices = free(value)
    if len(indices) != 2:
        raise CompileError(
            f"a trace contracts two free indices, and this has {len(indices)}",
        )
    first, second = indices
    return canonical(value * space.metric(-first, -second), space)


def solve(lhs: Value, rhs: Value, target: Tensor, space: Algebra) -> Value:
    """Solve ``lhs = rhs`` for the tensor ``target``, in which it must be linear.

    When the trace of the target appears too, as the Ricci scalar does beside
    the Ricci tensor, the equation is traced first and solved for the trace."""
    wanted = list(target.indices)
    expression = canonical(lhs - rhs, space)
    if not _mentions(expression, target.head) and space.quantity(target.head) == "ricci":
        expression = expand_einstein(expression, space)
    if free(expression) and {i.name for i in free(expression)} == {i.name for i in wanted}:
        expression = positioned(expression, wanted, space)
    trace_symbol = space.trace(target.head)
    coefficient = sympy.S.Zero
    others: list[tuple[Expr, list[Tensor]]] = []
    for term_coefficient, tensors in terms(expression):
        mine = [t for t in tensors if t.head == target.head]
        if not mine:
            others.append((term_coefficient, tensors))
            continue
        if len(mine) > 1 or len(tensors) > 1:
            raise CompileError(
                f"{target.head.name} appears multiplied by other tensors; "
                "solve handles equations linear in it",
            )
        if list(mine[0].indices) != wanted:
            raise CompileError(
                f"{target.head.name} appears with its indices in another order or position: "
                f"solve for it as it is written, or give it a symmetry",
            )
        coefficient += term_coefficient
    if coefficient == 0:
        raise CompileError(f"the equation does not contain {target.head.name} to solve for")
    if trace_symbol is not None and coefficient.has(trace_symbol):
        # T T_{ab} = S_{ab}: the trace of T times T, so not linear in T
        raise CompileError(
            f"{target.head.name} appears multiplied by its own trace, {trace_symbol}; "
            "solve handles equations linear in it"
        )
    rest = build(others)
    if trace_symbol is not None and _has(rest, trace_symbol):
        # a T_{ab} + rest(T) = 0; its trace is a T + trace(rest)(T) = 0, linear in T.
        traced = canonical(coefficient * trace_symbol + trace(rest, space), space)
        if isinstance(traced, TensExpr):
            raise CompileError("the trace of the equation is not a scalar equation")
        values = sympy.solve(traced, trace_symbol)
        if len(values) != 1:
            raise CompileError(f"the trace of the equation does not fix {trace_symbol}")
        rest = _substitute_scalar(rest, trace_symbol, values[0])
    return canonical(-rest / coefficient, space)


def expand_einstein(value: Value, space: Algebra) -> Value:
    """Write the Einstein tensor out: :math:`G_{ab} = R_{ab} - \\tfrac12 R g_{ab}`."""
    einstein = space.head("G", 2, "einstein")
    if not _mentions(value, einstein):
        return value
    a, b = (TensorIndex(name, space.type) for name in ("einstein_a", "einstein_b"))
    ricci = space.head("R", 2, "ricci")(-a, -b)
    definition = ricci - space.ricci_scalar / 2 * space.metric(-a, -b)
    return substitute(value, einstein(-a, -b), definition, space)


def _mentions(value: Value, head: TensorHead) -> bool:
    return any(tensor.head == head for _, tensors in terms(value) for tensor in tensors)


def substitute(value: Value, rule_lhs: Value, rule_rhs: Value, space: Algebra) -> Value:
    """Replace what ``rule_lhs`` matches in ``value`` by ``rule_rhs``.

    The rule is a definition of one tensor, ``G_{ab} = R_{ab} - \\tfrac12 R g_{ab}``,
    a contraction with no free indices, ``u^a u_a = -1``, or a value for a scalar,
    ``\\Lambda = 0``, which is put into the coefficients."""
    rule_lhs = canonical(rule_lhs, space)
    if not isinstance(rule_lhs, TensExpr):
        return _substitute_scalars(value, rule_lhs, canonical(rule_rhs, space), space)
    rule_terms = list(terms(rule_lhs))
    if len(rule_terms) != 1:
        raise CompileError("the left side of a rule must be one term")
    rule_coefficient, rule_tensors = rule_terms[0]
    replacement = canonical(rule_rhs / rule_coefficient, space)
    if len(rule_tensors) == 1 and free(rule_lhs):
        result = _replace_tensor(value, rule_tensors[0], replacement, space)
    elif not free(rule_lhs):
        result = _replace_contraction(value, rule_lhs, rule_tensors, replacement, space)
    else:
        raise CompileError(
            "a rule defines one tensor, as in G_{ab} = ..., or a contraction without "
            "free indices, as in u^a u_a = -1"
        )
    return canonical(result, space)


def _substitute_scalars(value: Value, old: Expr, new: Value, space: Algebra) -> Value:
    """``value`` with the scalar ``old``, such as \\Lambda or 8\\pi G, given the scalar value
    ``new`` in every coefficient."""
    if old.is_number:
        raise CompileError("a rule gives a value to a symbol or an expression, not to a number")
    if isinstance(new, TensExpr):
        raise CompileError("a rule for a scalar gives it a scalar value, without tensors")
    pieces = [(coefficient.subs(old, new), tensors) for coefficient, tensors in terms(value)]
    return canonical(build(pieces), space)


def _own_dummies(tensors: Sequence[Tensor], taken: Iterable[str], space: Algebra) -> list[Tensor]:
    """The factors of a term, with each dummy index renamed to a name used nowhere else:
    neither in the term nor among ``taken``, the names of the rule put into it. SymPy
    names the dummies of every product L_0, L_1, ..., so a rule's would meet the term's
    where a rule is put in; the factors are multiplied only once it has been."""
    names = [index.name for tensor in tensors for index in tensor.indices]
    used = {*names, *taken}
    renamed: dict[str, str] = {}
    for name in dict.fromkeys(name for name in names if names.count(name) == 2):
        number = next(n for n in itertools.count() if f"{name}_{n}" not in used)
        renamed[name] = f"{name}_{number}"
        used.add(renamed[name])
    return [
        tensor.head(
            *(
                TensorIndex(renamed.get(index.name, index.name), space.type, index.is_up)
                for index in tensor.indices
            )
        )
        for tensor in tensors
    ]


def _replace_tensor(value: Value, pattern: Tensor, replacement: Value, space: Algebra) -> Value:
    indices = replacement.get_indices() if isinstance(replacement, TensExpr) else []
    taken = {index.name for index in (*indices, *pattern.indices)}
    pieces = []
    for coefficient, tensors in terms(canonical(value, space)):
        factors: list[Value] = []
        for tensor in _own_dummies(tensors, taken, space):
            if tensor.head != pattern.head:
                factors.append(tensor)
                continue
            # Rename the rule's free indices to this occurrence's, positions included,
            # through temporary names, so that G_{ba} can use a rule written for G_{ab}.
            # The occurrence's dummies have names of their own, which the rule's cannot.
            slots = list(zip(pattern.indices, tensor.indices, strict=True))
            rule = positioned(
                replacement, [TensorIndex(p.name, space.type, t.is_up) for p, t in slots], space
            )
            if isinstance(rule, TensExpr):
                temporary = [TensorIndex(f"{p.name}_rule", space.type) for p, _ in slots]
                rule = rule.substitute_indices(
                    *(
                        (TensorIndex(p.name, space.type), tmp)
                        for (p, _), tmp in zip(slots, temporary, strict=True)
                    )
                )
                rule = rule.substitute_indices(
                    *(
                        (tmp, TensorIndex(t.name, space.type))
                        for (_, t), tmp in zip(slots, temporary, strict=True)
                    )
                )
            factors.append(rule)
        pieces.append((coefficient, factors))
    return build(pieces)


def _replace_contraction(
    value: Value, pattern: Value, pattern_tensors: list[Tensor], replacement: Value, space: Algebra
) -> Value:
    heads = sorted(t.head.name for t in pattern_tensors)
    pieces = []
    for coefficient, tensors in terms(canonical(value, space)):
        remaining = list(tensors)
        replaced = sympy.S.One
        while True:
            match = None
            for chosen in itertools.combinations(range(len(remaining)), len(pattern_tensors)):
                group = [remaining[i] for i in chosen]
                if sorted(t.head.name for t in group) != heads:
                    continue
                product = canonical(reduce(operator.mul, group), space)
                if not free(product) and canonical(product - pattern, space) == 0:
                    match = chosen
                    break
            if match is None:
                break
            remaining = [t for i, t in enumerate(remaining) if i not in match]
            replaced = replaced * replacement
        pieces.append((coefficient * replaced, remaining))
    return build(pieces)


def _vanishes(value: Value, space: Algebra) -> bool:
    """Whether a value is zero, with the Einstein tensor written out, so that
    :math:`G_{ab}` and :math:`R_{ab} - \\tfrac12 R g_{ab}` count as equal, and each
    coefficient with its common factors cancelled, which SymPy leaves as they are."""
    reduced = canonical(expand_einstein(canonical(value, space), space), space)
    if reduced == 0:
        return True
    totals: dict[str, Expr] = {}  # the coefficient of each product of tensors
    for coefficient, tensors in terms(reduced):
        product = _product_latex(tensors)
        totals[product] = totals.get(product, sympy.S.Zero) + coefficient
    return all(_cancelled(total) == 0 for total in totals.values())


def _same_free(one: Value, other: Value) -> bool:
    """Whether two values have the same free indices, in the same positions."""
    return {(i.name, i.is_up) for i in free(one)} == {(i.name, i.is_up) for i in free(other)}


def _cancelled(coefficient: Expr) -> Expr:
    """A coefficient with the factors its numerator and denominator share cancelled, as
    :math:`(a^2 - b^2)/(a - b)` is :math:`a + b`; any other as it is, so that
    :math:`\\Lambda - R/2` is not put over a common denominator."""
    together = sympy.together(coefficient)
    numerator, denominator = sympy.fraction(together)
    if denominator.is_number:
        return coefficient
    try:
        common = sympy.gcd(numerator, denominator)
    except sympy.PolynomialError:
        return coefficient
    return sympy.cancel(together) if common.free_symbols else coefficient


def _has(value: Value, symbol: Symbol) -> bool:
    return any(coefficient.has(symbol) for coefficient, _ in terms(value))


def _substitute_scalar(value: Value, symbol: Symbol, by: Expr) -> Value:
    return build([(coefficient.subs(symbol, by), tensors) for coefficient, tensors in terms(value)])


# -- display -------------------------------------------------------------------------------


def latex(value: Value) -> str:
    """Typeset a value in index notation, as in
    ``\\frac{G_{\\mu\\nu} + \\Lambda g_{\\mu\\nu}}{8 \\pi G}``."""
    return notation(value, "latex")


def notation(value: Value, form: str = "unicode") -> str:
    """A value in index notation in one of the forms of :mod:`relativisticpy.display`:
    ``G_{μν} + Λ g_{μν}`` in Unicode, ``G_{\\mu\\nu} + \\Lambda*g_{\\mu\\nu}`` as text and
    ``G_{\\mu\\nu} + \\Lambda g_{\\mu\\nu}`` in LaTeX."""
    if form not in ("unicode", "text", "latex"):
        raise ValueError(f"the forms are unicode, text and latex, not {form!r}")
    if not isinstance(value, TensExpr):
        if form == "latex":
            return str(_Printer({}).doprint(value))
        return str(_PlainNotation({}, {}, unicode=form == "unicode").doprint(value))
    placeholders: dict[str, Symbol] = {}
    names: dict[Symbol, str] = {}
    ranks: dict[Symbol, tuple[bool, int]] = {}
    coefficients: dict[Symbol, Expr] = {}
    for coefficient, tensors in terms(value):
        text = _product(tensors, form)
        placeholder = placeholders.get(text)
        if placeholder is None:
            placeholder = placeholders[text] = Symbol(f"zz{len(placeholders):04d}")
            names[placeholder] = text
            # Products of more tensors first, and terms of the metric alone last.
            ranks[placeholder] = (all(map(_is_metric, tensors)), -len(tensors))
        coefficients[placeholder] = coefficients.get(placeholder, sympy.S.Zero) + coefficient
    coefficients = {tensor: _cancelled(c) for tensor, c in coefficients.items()}
    printer: _Printer | _PlainNotation
    if form == "latex":
        printer = _Printer(names, ranks)
    else:
        printer = _PlainNotation(names, ranks, unicode=form == "unicode")
    return str(printer.doprint(_factored(coefficients, printer._order)))


def _is_metric(tensor: Tensor) -> bool:
    index_type = tensor.head.index_types[0]
    return tensor.head in (index_type.metric, index_type.delta)


def _factored(coefficients: dict[Symbol, Expr], order: Any) -> Expr:
    """The sum of each tensor times its coefficient, with a factor common to every
    coefficient taken out so that the first term's is 1, as in
    :math:`\\frac{1}{8\\pi G}(G_{\\mu\\nu} + \\Lambda g_{\\mu\\nu})`."""
    coefficients = {tensor: c for tensor, c in coefficients.items() if c != 0}
    total = sympy.Add(*(c * tensor for tensor, c in coefficients.items()))
    if len(coefficients) < 2:
        return total
    shared = sympy.factor_terms(
        sympy.Add(*(c.as_coeff_Mul()[1] * tensor for tensor, c in coefficients.items()))
    )
    symbolic = shared.as_independent(*coefficients, as_Add=False)[0]
    if symbolic == 1:  # only a number in common: leave the coefficients as they are
        return total
    first = min(coefficients, key=lambda tensor: order(coefficients[tensor] * tensor))
    common = coefficients[first].as_coeff_Mul()[0] * symbolic
    return sympy.Mul(common, sympy.Add(*(c / common * t for t, c in coefficients.items())))


def _term_order(
    ranks: dict[Symbol, tuple[bool, int]], term: Expr
) -> tuple[tuple[bool, int], int, bool]:
    tensors = term.free_symbols & ranks.keys()
    rank = min((ranks[tensor] for tensor in tensors), default=(False, 0))
    scalars = len(term.free_symbols - ranks.keys())
    return rank, scalars, term.could_extract_minus_sign()


class _Printer(_LatexPrinter):
    """SymPy's LaTeX as a physicist writes it, as in ``(\\rho + p) u_{\\mu} u_{\\nu} + p
    g_{\\mu\\nu}`` and ``R_{\\mu\\nu} - \\frac{1}{2} R g_{\\mu\\nu}``: coefficients before
    tensors, terms in the order of their tensors' ``ranks`` and then the simplest first, and
    a fraction times a product as a coefficient. The tensors stand in as the symbols that
    ``names`` maps to their LaTeX."""

    def __init__(
        self, names: dict[Symbol, str], ranks: dict[Symbol, tuple[bool, int]] | None = None
    ) -> None:
        super().__init__({"symbol_names": names})
        self._ranks = ranks or {}

    def factored(self, coefficients: dict[Symbol, Expr]) -> Expr:
        """See :func:`_factored`."""
        return _factored(coefficients, self._order)

    def _as_ordered_terms(self, expr: Expr, order: Any = None) -> list[Expr]:
        return sorted(LatexPrinter._as_ordered_terms(self, expr, order=order), key=self._order)

    def _order(self, term: Expr) -> tuple[tuple[bool, int], int, bool]:
        return _term_order(self._ranks, term)

    def _print_Mul(self, expr: Expr) -> str:
        tensors = [arg for arg in expr.args if arg in self._ranks]
        if tensors:
            scalar = sympy.Mul(*(arg for arg in expr.args if arg not in self._ranks))
            text = " ".join(self._print(tensor) for tensor in tensors)
            if scalar in (1, -1):
                return text if scalar == 1 else f"- {text}"
            written = self._print(scalar)
            return (rf"\left({written}\right)" if scalar.is_Add else written) + f" {text}"
        coefficient, rest = expr.as_coeff_Mul()
        if coefficient.is_Rational and coefficient.q != 1 and sympy.denom(rest) == 1:
            sign = "- " if coefficient < 0 else ""
            fraction = rf"\frac{{{abs(coefficient.p)}}}{{{coefficient.q}}}"
            text = self._print(rest)
            return f"{sign}{fraction} " + (rf"\left({text}\right)" if rest.is_Add else text)
        return str(super()._print_Mul(expr))


class _PlainNotation(_PlainPrinter):
    """Index notation on one line, in Unicode or as text: ``R_{μν} − ½ R g_{μν}`` is
    ``R_{μν} − 1/2 R g_{μν}``, with the order of terms of :class:`_Printer`."""

    def __init__(
        self,
        names: dict[Symbol, str],
        ranks: dict[Symbol, tuple[bool, int]],
        *,
        unicode: bool,
    ) -> None:
        super().__init__(unicode=unicode, latex_mode=not unicode, names=names)
        self._ranks = ranks

    def _as_ordered_terms(self, expr: Expr, order: Any = None) -> list[Expr]:
        return sorted(expr.as_ordered_terms(), key=self._order)

    def _order(self, term: Expr) -> tuple[tuple[bool, int], int, bool]:
        return _term_order(self._ranks, term)

    def _print_Mul(self, expr: Expr) -> str:
        tensors = [arg for arg in expr.args if arg in self._ranks]
        if not tensors:
            return str(super()._print_Mul(expr))
        scalar = sympy.Mul(*(arg for arg in expr.args if arg not in self._ranks))
        text = " ".join(self._print(tensor) for tensor in tensors)
        minus = "−" if self.unicode else "-"
        if scalar in (1, -1):
            return text if scalar == 1 else f"{minus}{text}"
        if scalar.could_extract_minus_sign() and not scalar.is_Add:
            return minus + self._print_Mul(-expr)
        written = self._print(scalar)
        if scalar.is_Add:
            written = f"({written})"
        return f"{written} {text}" if self.unicode else f"{written}*{text}"


def latex_labels(labels: Sequence[str]) -> str:
    """Index labels side by side as LaTeX: ``\\mu\\nu`` and ``ab``, but ``\\mu a``, since
    ``\\mua`` would be a command of its own."""
    text = ""
    for label in map(latex_label, labels):
        if label[:1].isalpha() and re.search(r"\\[A-Za-z]+$", text):
            text += " "
        text += label
    return text


def _product_latex(tensors: Sequence[Tensor]) -> str:
    """Tensors multiplied, in LaTeX; see :func:`_product`."""
    return _product(tensors, "latex")


def _product(tensors: Sequence[Tensor], form: str) -> str:
    """Tensors multiplied, with dummy indices named as a physicist would: Greek letters,
    then, for products with more dummies than those, the letters numbered."""
    used = {index.name for tensor in tensors for index in tensor.indices}
    numbered = (f"{label}_{n}" for n in itertools.count(1) for label in _DUMMY_LABELS)
    candidates = itertools.chain(_DUMMY_LABELS, numbered)
    available = (label for label in candidates if label not in used)
    labels: dict[str, str] = {}
    pieces = []
    label_in = {"unicode": unicode_label, "text": text_label, "latex": latex_label}[form]
    for tensor in tensors:
        positions, written = [], []
        for index in tensor.indices:
            if index.name.startswith("L_") and index.name not in labels:
                labels[index.name] = next(available)
            positions.append(bool(index.is_up))
            written.append(label_in(labels.get(index.name, index.name)))
        if tensor.head.name == "KD":
            name = "δ" if form == "unicode" else r"\delta"
        elif form == "unicode":
            name = unicode_name(tensor.head.name)
        elif form == "latex":
            name = latex_label(tensor.head.name)
        else:
            name = text_name(tensor.head.name, latex_mode=True)
        pieces.append(name + scripts(positions, written, form))  # type: ignore[arg-type]
    return " ".join(pieces)


@dataclass(frozen=True, eq=False)
class IndexExpression:
    """An expression in abstract index notation, such as :math:`8\\pi G\\,T_{\\mu\\nu}`.

    It compares with others and with scalars by canonical forms, so that a simplified
    identity is zero: ``simplify(R_{a b c d} + R_{a c d b} + R_{a d b c}) == 0``. As for
    any number, it is true in a boolean context when it is not zero; :attr:`is_zero`
    says so plainly, and :meth:`holds` reads it as the claim that it vanishes."""

    value: Value
    space: Algebra

    @property
    def algebra(self) -> Algebra:
        """The tensors of the computation it belongs to (also ``space``)."""
        return self.space

    def latex(self) -> str:
        return latex(self.value)

    def display(self, form: str = "unicode") -> str:
        """The expression in a form of :mod:`relativisticpy.display`."""
        return notation(self.value, form)

    def _display_unicode(self) -> str:
        return self.display("unicode")

    def _display_text(self) -> str:
        return self.display("text")

    def _display_latex(self) -> str:
        return self.display("latex")

    def _display_data(self) -> dict[str, Any]:
        return {"kind": "index expression", "is_zero": self.is_zero}

    def __str__(self) -> str:
        return self.latex()  # LaTeX, as it always has been; repr is Unicode

    def __repr__(self) -> str:
        return f"IndexExpression({self.display('unicode')})"

    def _repr_latex_(self) -> str:
        from relativisticpy import display

        return display.jupyter_latex(self)

    def _display_(self) -> Any:
        """What marimo shows: the expression typeset, by :mod:`~relativisticpy.display`."""
        from relativisticpy import display

        return display.marimo(self)

    def _latex(self, printer: Any = None) -> str:
        return self.latex()

    @property
    def is_zero(self) -> bool:
        """Whether the expression vanishes identically, by canonical forms."""
        return _vanishes(self.value, self.space)

    def holds(self) -> bool:
        """Whether the expression vanishes identically: an expression read as the claim
        that it is zero, as a simplified identity is written."""
        return self.is_zero

    def __bool__(self) -> bool:
        """Whether the expression is not zero, as for any number."""
        return not self.is_zero

    def __eq__(self, other: object) -> bool:
        if isinstance(other, IndexExpression):
            if not _same_free(self.value, other.value):
                return False
            return _vanishes(self.value - other.value, self.space)
        if isinstance(other, int | float | sympy.Expr) and not isinstance(other, bool):
            scalar = sympy.sympify(other)
            if scalar == 0:  # zero has whatever indices the other side has
                return self.is_zero
            if free(self.value):
                return False
            return _vanishes(self.value - scalar, self.space)
        return NotImplemented

    def __hash__(self) -> int:
        reduced = canonical(self.value, self.space)
        if not isinstance(reduced, TensExpr):
            return hash(reduced)  # as the scalar it equals
        return hash(("IndexExpression", _free_key(reduced)))


@dataclass(frozen=True, eq=False)
class IndexEquation:
    """An equation in abstract index notation, such as Einstein's.

    :meth:`holds` says whether it is an identity, and ``bool()`` agrees with it."""

    lhs: Value
    rhs: Value
    space: Algebra

    @property
    def algebra(self) -> Algebra:
        """The tensors of the computation it belongs to (also ``space``)."""
        return self.space

    def latex(self) -> str:
        return f"{latex(self.lhs)} = {latex(self.rhs)}"

    def display(self, form: str = "unicode") -> str:
        """The equation in a form of :mod:`relativisticpy.display`."""
        return f"{notation(self.lhs, form)} = {notation(self.rhs, form)}"

    def _display_unicode(self) -> str:
        return self.display("unicode")

    def _display_text(self) -> str:
        return self.display("text")

    def _display_latex(self) -> str:
        return self.display("latex")

    def _display_data(self) -> dict[str, Any]:
        return {"kind": "index equation", "holds": self.holds()}

    def __str__(self) -> str:
        return self.latex()  # LaTeX, as it always has been; repr is Unicode

    def __repr__(self) -> str:
        return f"IndexEquation({self.display('unicode')})"

    def _repr_latex_(self) -> str:
        from relativisticpy import display

        return display.jupyter_latex(self)

    def _display_(self) -> Any:
        """What marimo shows: the equation typeset, by :mod:`~relativisticpy.display`."""
        from relativisticpy import display

        return display.marimo(self)

    def _latex(self, printer: Any = None) -> str:
        return self.latex()

    def holds(self) -> bool:
        """Whether the equation is an identity, by canonical forms."""
        return _vanishes(self.lhs - self.rhs, self.space)

    def __bool__(self) -> bool:
        """Whether the equation is an identity: see :meth:`holds`."""
        return self.holds()

    def __eq__(self, other: object) -> bool:
        """Equations are equal when they say the same: their sides differ alike."""
        if not isinstance(other, IndexEquation):
            return NotImplemented
        mine, theirs = self.lhs - self.rhs, other.lhs - other.rhs
        if not _same_free(mine, theirs):
            return False
        return _vanishes(mine - theirs, self.space) or _vanishes(mine + theirs, self.space)

    def __hash__(self) -> int:
        # Equal equations have sides that differ by the same free indices.
        return hash(("IndexEquation", _free_key(self.lhs - self.rhs)))


def _free_key(value: Value) -> frozenset[tuple[str, bool]]:
    return frozenset((index.name, bool(index.is_up)) for index in free(value))
