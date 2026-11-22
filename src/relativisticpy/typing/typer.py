"""The typer: the rules of index notation and spaces.

What the indices of a value are, free and summed, up and down; the space each axis ranges
over; and each rule's one message, in the user's notation. The lowerer asks the typer
whether what it is about to emit keeps the rules, and of what type it is.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from relativisticpy.compiler.passes import Pass
from relativisticpy.compiler.scope import Context, Scope, TensorBinding, VariableBinding
from relativisticpy.errors import CompileError, IndexNotationError
from relativisticpy.ir import SCALAR, Node, TensorType, Type, ops
from relativisticpy.language import syntax as ast
from relativisticpy.language.source import Span
from relativisticpy.language.vocabulary import GREEK, UNICODE_LETTERS
from relativisticpy.runtime.kernels import refuse_quoted
from relativisticpy.tensors import Index, Indices
from relativisticpy.tensors.spaces import Space, spacetime
from relativisticpy.typing import notation as rules
from relativisticpy.typing.syntax import (
    _NOTATION_OPERATIONS,
    _indexed_name,
    _is_minus_one,
    _operand,
    _plain_label,
    _product_factors,
    _signed_terms,
    _tensor_pattern,
    _walk,
    _zero_literal,
)


class Typer(Pass):
    """The rules of index notation and spaces, one message each.

    Each rule has one phrasing, on every path that checks it: as components and as
    notation. The phrasings of README.md, docs/spaces.md and docs/getting-started.md are
    kept as their examples show them. An index label ranges over a space: one it was
    declared in (a, b \\in \\mathcal{A}), or, within a statement, the space of the first
    axis it is written on. Labels that neither declare nor infer range over spacetime."""

    def _written_label(self, label: str) -> str:
        """An index label as the statement writes it: ``\\mu`` in LaTeX, ``μ`` in
        Unicode, ``mu`` in plain text, and a Latin letter as it is."""
        name = _display(label)
        if name not in GREEK:
            return name
        text = self._text
        if "\\" + name in text:
            return "\\" + name
        character = _GREEK_CHARACTERS.get(name)
        if character is not None and character in text:
            return character
        return name if re.search(rf"(?<![\\A-Za-z]){name}(?![A-Za-z])", text) else "\\" + name

    def _written_indices(self, indices: Sequence[Index]) -> str:
        """Indices as the statement writes them, grouped by position: ``_{\\mu\\nu}``,
        ``^{a}_{b}``, or ``none``."""
        groups: list[tuple[bool, list[str]]] = []
        for index in indices:
            label = self._written_label(index.name)
            if groups and groups[-1][0] == index.upper:
                groups[-1][1].append(label)
            else:
                groups.append((index.upper, [label]))
        rendered = []
        for upper, labels in groups:
            plain = any(len(label) > 1 and label.isalpha() for label in labels)
            joined = " ".join(labels) if plain else "".join(labels)
            rendered.append(("^" if upper else "_") + "{" + joined + "}")
        return "".join(rendered) or "none"

    def _written_free(self, free: Iterable[Index]) -> str:
        """A set of free indices, upper first and in order of their labels."""
        return self._written_indices(sorted(free, key=lambda index: (not index.upper, index.name)))

    def _twice(self, name: str, upper: bool, span: Span) -> IndexNotationError:
        """An index written twice in one term in the same position."""
        label = self._written_label(name)
        notes = []
        coordinates = [c.name for c in self.session.coordinates or ()]
        if (
            coordinates
            and _display(name) in _COORDINATE_NAMES
            and _display(name) not in coordinates
        ):  # R_{zz} in (t, r, \theta, \phi): a component that is none
            shown = _listed([self._written_label(c) for c in coordinates])
            notes.append(
                f"{label} is not a coordinate, so it is an index; the coordinates are {shown}"
            )
        position = "an upper" if upper else "a lower"
        return IndexNotationError(
            f"index {label} appears twice as {position} index; a summed index must appear "
            "once up and once down",
            span=span,
            notes=notes,
        )

    def _thrice(self, name: str, count: int, span: Span) -> IndexNotationError:
        """An index written three times or more in one term."""
        return IndexNotationError(
            f"index {self._written_label(name)} appears {count} times in one term; an index "
            "appears once, or twice to be summed",
            span=span,
        )

    def _two_spaces(self, name: str, one: Space, other: Space, span: Span) -> IndexNotationError:
        """An index written on axes over two spaces."""
        ranges, written = _apart(one, other)
        return IndexNotationError(
            f"the index {self._written_label(name)} ranges over {ranges}, but is written here "
            f"on an axis over {written}",
            span=span,
        )

    def _are(self, labels: Sequence[str]) -> str:
        """``\\rho is`` or ``\\rho and \\sigma are``, as the statement writes the labels."""
        shown = _listed([self._written_label(label) for label in labels])
        return f"{shown} {'is' if len(labels) == 1 else 'are'}"

    def _odd_term(
        self,
        term: ast.Expr,
        indices: frozenset[Index],
        common: frozenset[Index],
        scope: Scope,
    ) -> IndexNotationError:
        """The error for the one term of a sum whose free indices are not the others':
        ``\\rho is free in this term but in no other``, and what the others have."""
        mine = {index.name: index.upper for index in indices}
        theirs = {index.name: index.upper for index in common}
        extra = sorted(name for name in mine if name not in theirs)
        missing = sorted(name for name in theirs if name not in mine)
        moved = sorted(name for name in mine if name in theirs and mine[name] != theirs[name])
        others = self._written_free(common)
        notes = [f"the other terms have {others} free"]
        hint = None
        if extra:
            message = f"{self._are(extra)} free in this term but in no other"
            if missing:
                notes.append(f"{self._are(missing)} missing here")
            if len(extra) == 1 and len(missing) == 1:
                text = term.span.text
                wrong, right = self._written_label(extra[0]), self._written_label(missing[0])
                if text.count(wrong) == 1:
                    hint = f"did you mean {text.replace(wrong, right)}?"
        elif moved:
            name = moved[0]
            here, there = ("up" if mine[name] else "down"), ("up" if theirs[name] else "down")
            message = (
                f"{self._written_label(name)} is {here} in this term, but {there} in the others"
            )
        elif not indices:
            message = "this term has no free indices, but the others do"
            notes.extend(self._undefined_notes([term], scope))
        else:
            message = f"{self._are(missing)} free in the other terms, but not in this one"
        return IndexNotationError(message, span=term.span, notes=notes, hint=hint)

    def _undefined_notes(self, terms: Sequence[ast.Expr], scope: Scope) -> list[str]:
        """Notes for terms that are scalars only because a name is not defined, as y_t is
        one symbol where y is no tensor."""
        notes = []
        for term in terms:
            if isinstance(term, ast.Name) and "_" in term.id:
                base = term.id.partition("_")[0]
                if scope.lookup(base) is None:
                    notes.append(
                        f"{base} is not defined, so {term.span.text} is read as one symbol, "
                        f"{term.span.text}"
                    )
            elif (
                isinstance(term, ast.Scripted)
                and isinstance(term.base, ast.Name)
                and scope.lookup(term.base.id) is None
                and _tensor_pattern(term)
            ):
                written = term.base.span.text
                notes.append(f"{written} is not defined, so {term.span.text} is not a tensor")
        return notes

    def _tensor_power(
        self, metric: str | None, indices: Sequence[Index], exponent: str | None, span: Span
    ) -> IndexNotationError:
        """A tensor raised to a power: to -1, as for an inverse, or to any other."""
        flipped = self._written_indices([index.flip() for index in indices])
        if exponent == "-1":
            if metric is not None:
                hint = f"the inverse metric is {metric}{flipped}"
            elif len(indices) == 2:
                tensor = f"S{self._written_indices(indices)}"
                hint = f"the inverse of a tensor {tensor} is (S^{{-1}}){flipped}"
            else:
                hint = "only a tensor with two indices has an inverse"
            return IndexNotationError("cannot raise a tensor to the power -1", span=span, hint=hint)
        return IndexNotationError(
            "cannot raise a tensor to a power",
            span=span,
            hint="contract it with itself instead, as in V_{a} V^{a}",
        )

    def _check_terms(
        self, terms: Sequence[tuple[Node, ast.Expr]], span: Span, scope: Scope
    ) -> None:
        """That the terms of a sum have the same free indices, or say which term does
        not, and which index is wrong in it, where the others agree."""
        free = [
            frozenset(_indices(value)) if isinstance(value.type, TensorType) else frozenset()
            for value, _ in terms
        ]
        written = [
            (value, term, indices)
            for (value, term), indices in zip(terms, free, strict=True)
            if indices or not _is_zero(value)
        ]
        kinds = Counter(indices for _, _, indices in written)
        if len(kinds) <= 1:
            return
        common, count = kinds.most_common(1)[0]
        odd = [(value, term, indices) for value, term, indices in written if indices != common]
        if count > 1 and len(odd) == 1:  # the others agree: say what is wrong in this one
            _, term, indices = odd[0]
            raise self._odd_term(term, indices, common, scope)
        first, second = list(kinds)[:2]
        shown = []
        for kind in (first, second):
            value, term, _ = next(entry for entry in written if entry[2] == kind)
            shown.append(str(_indices(value)) if kind else "none")
            where = term.span
        raise IndexNotationError(
            f"the free indices of a sum must agree: {shown[0]} and {shown[1]}",
            span=where,
            notes=self._undefined_notes(
                [term for _, term, indices in written if not indices], scope
            ),
        )

    def _index_structure(
        self, node: ast.Expr, scope: Scope
    ) -> tuple[frozenset[tuple[str, bool]] | None, bool]:
        """The free indices of notation, with their positions, read from its syntax, and
        whether any index is summed. The free indices are ``None`` where the syntax cannot
        tell them, as for the result of ``solve``.

        Raises :class:`IndexNotationError` where the indices break the rules: an index
        three times in one term, a summed index that is not once up and once down, a
        division by a tensor, or terms or sides whose free indices differ."""
        summed = False

        def contract(entries: list[tuple[str, bool]], span: Span) -> frozenset[tuple[str, bool]]:
            nonlocal summed
            free: set[tuple[str, bool]] = set()
            for label in dict.fromkeys(label for label, _ in entries):
                positions = [up for other, up in entries if other == label]
                if len(positions) > 2:
                    raise self._thrice(label, len(positions), span)
                if len(positions) == 2 and positions[0] == positions[1]:
                    raise self._twice(label, positions[0], span)
                if len(positions) == 2:
                    summed = True
                else:
                    free.add((label, positions[0]))
            return frozenset(free)

        def side(node: ast.Expr) -> frozenset[tuple[str, bool]] | None:
            if _zero_literal(node):  # zero has whatever indices the other side has
                return None
            found = uses(node)
            return None if found is None else contract(found, node.span)

        def uses(node: ast.Expr) -> list[tuple[str, bool]] | None:
            """Every index written in a term, summed or free; ``None`` if unknown."""
            node = _operand(node)
            if isinstance(node, ast.Equation):
                lhs, rhs = side(node.lhs), side(node.rhs)
                if lhs is not None and rhs is not None and lhs != rhs:
                    raise IndexNotationError(
                        "the two sides have different free indices: "
                        f"{self._written_free(_as_indices(lhs))} on the left, "
                        f"{self._written_free(_as_indices(rhs))} on the right",
                        span=node.span,
                    )
                either = lhs if lhs is not None else rhs
                return None if either is None else sorted(either)
            if isinstance(node, ast.Binary) and node.op in ("+", "-"):
                common: frozenset[tuple[str, bool]] | None = None
                kinds: list[tuple[ast.Expr, frozenset[tuple[str, bool]]]] = []
                for _, term in _signed_terms(node):
                    free = side(term)
                    if free is None:
                        continue
                    kinds.append((term, free))
                    common = free if common is None else common
                if len({free for _, free in kinds}) > 1:
                    counted = Counter(free for _, free in kinds)
                    most, count = counted.most_common(1)[0]
                    odd = [(term, free) for term, free in kinds if free != most]
                    if count > 1 and len(odd) == 1:
                        raise self._odd_term(
                            odd[0][0], _as_indices(odd[0][1]), _as_indices(most), scope
                        )
                    first, second = list(counted)[:2]
                    raise IndexNotationError(
                        "the free indices of a sum must agree: "
                        f"{self._written_free(_as_indices(first))} and "
                        f"{self._written_free(_as_indices(second))}",
                        span=node.span,
                    )
                return None if common is None else sorted(common)
            if isinstance(node, ast.Binary) and node.op == "*":
                entries: list[tuple[str, bool]] = []
                known = True
                for factor in _product_factors(node):
                    written = uses(factor)
                    if written is None:
                        known = False
                    else:
                        entries.extend(written)
                if not known:
                    contract(entries, node.span)  # what the known factors break, still
                    return None
                return entries
            if isinstance(node, ast.Binary) and node.op == "/":
                if side(node.right):
                    raise IndexNotationError("cannot divide by a tensor", span=node.right.span)
                return uses(node.left)
            if isinstance(node, ast.Unary) and node.op in ("-", "+", "±", "∓"):
                return uses(node.operand)
            if isinstance(node, ast.Scripted):
                if pattern := _tensor_pattern(node):
                    return pattern[1]
                items = [
                    (item, script.kind == "^")
                    for script in node.scripts
                    if script.indices is not None
                    for item in script.indices.items
                ]
                if any(item.derivative or item.opens or item.closes for item, _ in items):
                    return None  # derivatives and symmetrisations
                plain = [(str(item.label), up) for item, up in items if _plain_label(item)]
                if isinstance(node.base, ast.Operator):  # \partial_\mu, a factor of a product
                    return plain
                if isinstance(node.base, ast.Name) or not plain:  # a component, or a power
                    return None if side(node.base) else []
                return None
            if isinstance(node, ast.Name):
                binding = scope.lookup(node.id)
                if (
                    isinstance(binding, VariableBinding)
                    and (kept := rules.notation_of(binding.node)) is not None
                ):
                    if kept.equation:
                        return None
                    return [(index.name, index.upper) for index in kept.free]
                return []
            if isinstance(node, ast.Call):
                # An operator applied, as in \nabla_\mu(...), is a derivative.
                if not isinstance(node.func, ast.Name):
                    return None
                arguments = [side(argument) for argument in node.args]
                if node.func.id in _NOTATION_OPERATIONS or any(a is None or a for a in arguments):
                    return None  # a function of tensors, or of what cannot be told
                return []
            if isinstance(node, ast.Number | ast.Accent | ast.Root | ast.Abs | ast.Factorial):
                return []
            return None

        return side(node), summed

    def _euclidean(self, space: Space) -> bool:
        """Whether the space's metric is the identity, so that an index may be summed in
        any position: neither spacetime nor a space given a metric."""
        return not space.spacetime and space not in self.session.space_metrics

    def _entries(self, node: ast.Scripted, purpose: str) -> list[tuple[ast.IndexItem, bool]]:
        """The index items of every script, each with whether it is upper."""
        entries: list[tuple[ast.IndexItem, bool]] = []
        for script in node.scripts:
            if script.indices is None:
                if script.index_error is not None:
                    raise script.index_error
                raise IndexNotationError(f"expected indices in {purpose}", span=script.span)
            for item in script.indices.items:
                if item.label == "⋯":
                    raise CompileError(
                        f"{item.span.text} stands for indices that are not written, so this "
                        "cannot be computed",
                        span=item.span,
                    )
                entries.append((item, script.kind == "^"))
        return entries

    def _index_value(self, item: ast.IndexItem, space: Space) -> int | None:
        """The component an index item fixes: ``a=0``, ``a:t``, or the name of an element
        of the space's basis, a coordinate's name in spacetime."""
        names = list(space.basis or ())
        what = "coordinates" if space.spacetime else f"elements of {space.name}"
        value = item.value
        if value is None:
            return names.index(item.label) if item.label in names else None
        if isinstance(value, ast.Number) and value.text in map(str, names):
            return list(map(str, names)).index(value.text)  # a label such as 2020, as .loc
        if isinstance(value, ast.Number):
            number = int(value.text)
            if number >= space.dim:
                raise IndexNotationError(
                    f"index value {number} is out of range for {space.dim} {what}",
                    span=value.span,
                )
            return number
        assert isinstance(value, ast.Name)
        if value.id not in names:
            noun = "a coordinate" if space.spacetime else f"an element of {space.name}"
            raise IndexNotationError(
                f"{value.id} is not {noun}",
                span=value.span,
                hint=f"the {what} are {', '.join(names)}" if names else None,
            )
        return names.index(value.id)

    def _constant(self, node: Node) -> Any:
        """The value of a node that is known when compiling: a constant, as a number written
        or a name given one, or data bound from Python; ``None`` for a value computed, which
        is not known until the program runs. Text in quotes is never a value."""
        if not isinstance(node.op, ops.Const):
            return None
        refuse_quoted(node, node.op.value)
        return node.op.value

    def _space_of(self, name: str, span: Span, why: str = "index notation") -> Space:
        space = self._labels.get(name) or self.session.labels.get(name)
        if space is not None:
            return space
        if self.session.coordinates is None:
            label = self._written_label(name)
            data = bool(self.session.spaces) or any(
                not space.spacetime
                for _, binding in self.session.globals.bindings()
                if isinstance(binding, TensorBinding)
                for space in _spaces(binding.node) or ()
            )
            hint = (
                "name a space for it, as in \\mathcal{A} := \\mathbb{R}^{3}, and write "
                f"{label} \\in \\mathcal{{A}}"
                if data
                else "declare coordinates first, as in Coordinates := [t, r, \\theta, \\phi]"
            )
            raise CompileError(
                f"{why} needs coordinates, or a space for the index {label}",
                span=span,
                notes=[
                    "an index ranges over spacetime, over a space declared for it, or over the "
                    "axis of data it is written on in the same statement"
                ],
                hint=hint,
            )
        return self.session.spacetime(span, why)

    def _spaces_of(self, names: Sequence[str], span: Span, why: str) -> tuple[Space, ...]:
        return tuple(self._space_of(name, span, why) for name in names)

    def _check_label(self, name: str, space: Space, span: Span) -> None:
        """That an index written on an axis over ``space`` may range over it; the first
        time in a statement, it then does, for the rest of the statement. On an axis of
        spacetime, a coordinate's name, as t in \\Gamma^{t}_{tr}, names a component and is
        no index, whatever t is elsewhere."""
        if space.spacetime and name in (space.basis or ()):
            return
        expected = self._labels.get(name) or self.session.labels.get(name)
        if expected is None:
            self._labels[name] = space
        elif expected != space:
            raise self._two_spaces(name, expected, space, span)

    def _infer_labels(self, statement: ast.Statement, scope: Scope) -> dict[str, Space]:
        """The spaces of the labels written on tensors with spaces of their own in a
        statement, found before it is compiled, in the order they are written, so that
        𝟙_c in 𝟙_c (\\Sigma^{-1})^{cd}, and 1^k in 1^k \\operatorname{spec}(S)_k, know their
        spaces from what comes after them."""
        labels: dict[str, Space] = {}
        candidates = (
            node for node in _walk(statement) if isinstance(node, ast.Scripted | ast.Name)
        )
        for current in sorted(candidates, key=lambda node: node.span.start):
            found = self._written_on(current, scope, statement.latex, labels)
            if found is None:
                continue
            entries, spaces = found
            for (label, _), space in zip(entries, spaces, strict=True):
                if space.spacetime:
                    continue
                expected = labels.get(label) or self.session.labels.get(label)
                if expected is not None and expected != space:
                    raise self._two_spaces(label, expected, space, current.span)
                labels[label] = space
        return labels

    def _names_tensor(self, node: ast.Expr, scope: Scope) -> bool:
        """Whether an expression mentions a tensor: written with indices, or by a name."""
        for current in _walk(node):
            if isinstance(current, ast.Scripted) and _tensor_pattern(current):
                return True
            if isinstance(current, ast.Name):
                binding = scope.lookup(current.id)
                if isinstance(binding, TensorBinding) or (
                    isinstance(binding, VariableBinding)
                    and isinstance(binding.node.type, TensorType)
                ):
                    return True
        return False

    def _written_on(
        self, node: ast.Scripted | ast.Name, scope: Scope, latex: bool, labels: dict[str, Space]
    ) -> tuple[list[tuple[str, bool]], tuple[Space, ...]] | None:
        """The index labels written on a tensor with spaces of its own, and those spaces:
        a tensor defined or bound, ``y_t`` for ``y_{t}``, an inverse, whose spaces go the
        other way, or the result of a call such as ``\\operatorname{diag}(k)_{ij}``, which
        is compiled to find them; ``None`` for anything else."""
        if isinstance(node, ast.Name):
            indexed = _indexed_name(node, scope)
            if indexed is None:
                return None
            node = indexed
        base, inverse = node.base, False
        if isinstance(base, ast.Scripted) and len(base.scripts) == 1:  # (\Sigma^{-1})^{cd}
            if not _is_minus_one(base.scripts[0].body):
                return None
            base, inverse = base.base, True
        pattern = _tensor_pattern(replace(node, base=ast.Name("#", span=node.span)))
        spaces: tuple[Space, ...] | None = None
        if isinstance(base, ast.Name):
            binding = scope.lookup(base.id)
            if isinstance(binding, VariableBinding | TensorBinding):
                spaces = _spaces(binding.node)
        elif isinstance(base, ast.Call) and pattern is not None and self._names_tensor(base, scope):
            from relativisticpy.errors import RelativisticPyError

            saved, self._labels = self._labels, labels  # its labels are this statement's
            checks = len(self._checks)
            try:
                spaces = _spaces(self.value(base, scope, Context(latex=latex)))
            except RelativisticPyError:
                return None  # said where it is compiled
            finally:
                self._labels = saved
                del self._checks[checks:]  # compiled to see its spaces, not to run
        if inverse:  # of two indices, and going the other way
            spaces = spaces[::-1] if spaces is not None and len(spaces) == 2 else None
        if not spaces or pattern is None or len(pattern[1]) != len(spaces):
            return None
        return pattern[1], spaces

    def _notation_space(self, label: str, span: Span) -> Space:
        """The space a free index of notation's components ranges over: one declared for it
        or written in the statement, or spacetime. Without coordinates, the space of none:
        the lowerer, as the statement runs, says what the components need."""
        space = self._labels.get(label) or self.session.labels.get(label)
        if space is not None:
            return space
        if self.session.coordinates is None:
            return spacetime(())
        return self.session.spacetime(span)

    # -- the rules of what the lowerer emits ------------------------------------------

    def _contraction(
        self,
        factors: Sequence[tuple[Node, Indices]],
        span: Span,
        output: Indices | None = None,
    ) -> tuple[ops.Contract, Type]:
        """The contraction of tensors, and its type: each index that appears once up and
        once down is summed, or, in a space whose metric is the identity, twice in any
        position. An index ranges over one space. A label an expectation binds is shared by
        every factor and kept."""
        seen: dict[str, list[tuple[Index, Space]]] = {}
        for node, indices in factors:
            spaces = _spaces(node)
            assert spaces is not None
            assert len(spaces) == len(indices)
            for index, space in zip(indices, spaces, strict=True):
                seen.setdefault(index.name, []).append((index, space))
        for name, found in seen.items():
            over = list(dict.fromkeys(space for _, space in found))
            if len(over) > 1:
                raise self._two_spaces(name, over[0], over[1], span)
            if name in self._bound:
                continue
            if len(found) > 2:
                raise self._thrice(name, len(found), span)
            positions = {index.upper for index, _ in found}
            if len(found) == 2 and len(positions) == 1 and not self._euclidean(over[0]):
                raise self._twice(name, found[0][0].upper, span)
        if output is None:
            kept = {
                name: found[0][0]
                for name, found in seen.items()
                if len(found) == 1 or name in self._bound
            }
            output = Indices(kept.values())
        numbers = {name: number for number, name in enumerate(seen)}
        labels = tuple(tuple(numbers[i.name] for i in indices) for _, indices in factors)
        sizes = tuple(found[0][1].dim for found in seen.values())
        spaces = tuple(seen[index.name][0][1] for index in output)
        kind: Type = TensorType(output, spaces) if output else SCALAR
        return ops.Contract(labels, tuple(numbers[i.name] for i in output), sizes), kind

    def _alignment(
        self, have: Indices, indices: Indices, span: Span, what: str = "a sum"
    ) -> list[int]:
        """Where each of ``indices`` is among ``have``, the free indices of a value that is
        to have them: those of the other terms of ``what``, a sum, the other side of an
        equation or a definition, or the other factor of an elementwise product."""
        if sorted(have, key=str) != sorted(indices, key=str):
            wanted, found = self._written_indices(indices), self._written_indices(have)
            if what in ("an equation", "a definition"):
                message = (
                    f"the two sides have different free indices: {wanted} on the left, "
                    f"{found} on the right"
                )
            elif what == "a sum":
                message = f"the free indices of a sum must agree: {indices} and {have}"
            else:
                message = f"the free indices of {what} must agree: {wanted} and {found}"
            raise IndexNotationError(message, span=span)
        return [have.index(i) for i in indices]

    def _check_sides(self, value: Node, tensor: TensorType, lhs: Node, equation: Span) -> None:
        """That one side of a tensor equation has the free indices of the first side's."""
        assert isinstance(value.type, TensorType)
        if sorted(value.type.indices, key=str) == sorted(tensor.indices, key=str):
            return
        left = _indices(lhs) if isinstance(lhs.type, TensorType) else value.type.indices
        right = value.type.indices if value is not lhs else tensor.indices
        raise IndexNotationError(
            f"the two sides have different free indices: {self._written_indices(left)} on the "
            f"left, {self._written_indices(right)} on the right",
            span=equation,
        )

    def _check_zero_side(
        self, value: Node, tensor: TensorType, side: ast.Expr, left: bool, scope: Scope
    ) -> None:
        """That the side of a tensor equation that is a scalar is zero, which has whatever
        indices the other side has."""
        if not _is_zero(value):
            here = self._written_indices(tensor.indices)
            left_text, right_text = ("none", here) if left else (here, "none")
            raise IndexNotationError(
                f"the two sides have different free indices: {left_text} on the left, "
                f"{right_text} on the right",
                span=side.span,
                notes=[
                    *self._undefined_notes([side], scope),
                    "a scalar side can only be 0, which has whatever indices the other has",
                ],
            )

    def _symmetrized(self, kind: Type, labels: Sequence[str], span: Span) -> tuple[int, ...]:
        """The axes an (anti)symmetrisation over ``labels`` takes of a value of ``kind``:
        only free indices can be (anti)symmetrised."""
        if not isinstance(kind, TensorType):
            raise IndexNotationError("only free indices can be (anti)symmetrised", span=span)
        names = list(kind.indices.names)
        missing = [label for label in labels if label not in names]
        if missing:
            raise IndexNotationError(
                f"only free indices can be (anti)symmetrised, not {', '.join(missing)}", span=span
            )
        return tuple(names.index(label) for label in labels)


@dataclass(frozen=True)
class TensorSource:
    """Something indices can be put on: a defined tensor or a built-in one."""

    node: Node
    indices: Indices
    name: str


def _indices(node: Node) -> Indices:
    assert isinstance(node.type, TensorType)
    return node.type.indices


def _spaces(node: Node) -> tuple[Space, ...] | None:
    """The spaces of a tensor's axes; ``None`` for anything else."""
    return node.type.spaces if isinstance(node.type, TensorType) else None


def _shares(one: Node, other: Node) -> bool:
    """Whether two tensors have an index label in common."""
    return bool(set(_indices(one).names) & set(_indices(other).names))


def _complex(kind: TensorType) -> bool:
    """Whether a tensor has an axis over a complex space."""
    return any(space.field == "complex" for space in kind.spaces)


def _public(indices: Indices) -> Indices:
    return Indices(Index(_display(i.name), i.upper) for i in indices)


def _display(name: str) -> str:
    return name.lstrip("#")


def _is_zero(node: Node) -> bool:
    return isinstance(node.op, ops.Const) and node.op.value == 0


def _as_indices(free: Iterable[tuple[str, bool]]) -> frozenset[Index]:
    """Free indices read from syntax, ``(label, upper)``, as indices."""
    return frozenset(Index(label, upper) for label, upper in free)


def _apart(one: Space, other: Space) -> tuple[str, str]:
    """Two different spaces in words that tell them apart: their names, or, where they
    share a name, as the spaces of one label in two sets of data do, their sizes,
    elements or fields."""
    if str(one) != str(other):
        return str(one), str(other)
    if one.dim != other.dim:
        return f"{one}, of dimension {one.dim}", f"{other}, of dimension {other.dim}"
    if one.basis != other.basis:
        return f"{one}, {_elements(one)}", f"{other}, {_elements(other)}"
    return f"{one}, {one.field}", f"{other}, {other.field}"


def _elements(space: Space) -> str:
    if space.basis is None:
        return "its elements unnamed"
    shown = ", ".join(space.basis[:3])
    return f"of {shown}{', ...' if len(space.basis) > 3 else ''}"


def _listed(items: Sequence[str], conjunction: str = "and") -> str:
    """``t, r, \\theta and \\phi``."""
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} {conjunction} {items[-1]}"


# Index labels for examples in messages, where the labels written name components.
_EXAMPLE_LABELS = ("\\mu", "\\nu", "\\rho", "\\sigma", "\\lambda", "\\kappa")


#: Greek letters as Unicode writes them, to write a label as the user did.
_GREEK_CHARACTERS: dict[str, str] = {}


for _character, _name in UNICODE_LETTERS.items():
    if _name in GREEK:
        _GREEK_CHARACTERS.setdefault(_name, _character)


#: Names usual for coordinates: written twice down on a tensor, as z in R_{zz}, and
#: not a coordinate in force, such a label is more likely a component than a sum.
_COORDINATE_NAMES = frozenset({"t", "x", "y", "z", "r", "theta", "phi", "rho", "chi", "psi"})
