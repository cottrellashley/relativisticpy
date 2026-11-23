r"""The lowering of transforms, of the ranges of index labels, and of sums over an index
of data.

A transform written as books write it, ``\mathcal{L}\{f(t)\}(s)``, or called by its name,
``LaplaceTransform(f, t, s)``, is a call of its capability, with the variables its
convention names (``design/notation.md``, 21): ℒ takes t to s, and ℱ x to s or t to ω, by
the letters used. A transform of data, ``\operatorname{fft}(x_t)``, gives data over a space
of its own, the frequencies of the index it transforms.

A range written beside a definition, ``x_n := n h, \quad n = 0, \ldots, N-1``, declares the
space its label ranges over, of N elements labelled by their values, and the definition is
made for each value: n, used as a number, is the value itself. A sum over an index of data,
``\sum_{n=0}^{N-1} x_n e^{-2\pi i k n/N}``, binds the index as a sum does, and is a
contraction over its space, unless a rewrite computes it faster, as an FFT.
"""

from __future__ import annotations

import contextlib
import itertools
from collections.abc import Iterator
from typing import Any

import sympy

from relativisticpy import integrations
from relativisticpy.compiler.builtins import BUILTINS, Builtin
from relativisticpy.compiler.passes import Term
from relativisticpy.compiler.scope import (
    Context,
    FunctionBinding,
    Scope,
    TensorBinding,
    VariableBinding,
)
from relativisticpy.compiler.session import Session
from relativisticpy.errors import CompileError, IndexNotationError, RelativisticPyError
from relativisticpy.integrations import _transforms as transforms
from relativisticpy.integrations._transforms import Convention, Transform, Written
from relativisticpy.ir import SCALAR, VALUE, EquationType, Node, TensorType, dependencies, ops
from relativisticpy.language import syntax as ast
from relativisticpy.language.parser import split_index_labels
from relativisticpy.language.source import Span
from relativisticpy.language.vocabulary import GREEK
from relativisticpy.lowering.definitions import Definitions
from relativisticpy.reading.names import _elements
from relativisticpy.reading.subscripts import Kind
from relativisticpy.tensors import Index, Indices
from relativisticpy.tensors.spaces import Space
from relativisticpy.typing.syntax import _tensor_pattern, _walk
from relativisticpy.typing.typer import _KNOWN_FIRST, _listed

#: The decision of what a transform's variables, and its convention, are.
TRANSFORM = "transform"
#: The decision of what a label used as a number is, where it ranges over values.
RANGE = "range of an index"


class Transforms(Definitions):
    """The lowering of transforms, of the ranges of index labels, and of sums over an
    index of data."""

    def __init__(self, session: Session) -> None:
        super().__init__(session)
        # The labels of the definition being made over their ranges, each with whether it
        # is written up: the n of x_n := n h, \quad n = 0, \ldots, N-1.
        self._ranged: dict[str, bool] = {}
        # The values of the labels a sum over an index of data binds, while its body is
        # compiled: n, as a number, in \sum_{n=0}^{N-1} x_n e^{-2\pi i k n/N}.
        self._summed: dict[str, Node] = {}

    # -- transforms written as books write them -----------------------------------------

    def _Call(self, node: ast.Call, scope: Scope, ctx: Context) -> Node:
        written = transforms.written_call(node)
        if written is not None:  # \mathcal{L}\{t^2\}
            self._note_letter(written, scope)
            return self._written_transform(written, scope, ctx)
        return super()._Call(node, scope, ctx)

    def _statistic_factors(
        self, factors: list[ast.Expr], scope: Scope, ctx: Context
    ) -> list[ast.Expr | Node]:
        """The factors of a product, each transform written among them, as
        ``\\mathcal{L}\\{f\\}(s)`` or ``\\mathcal{L}^{-1}\\{F\\}(t)``, compiled to one node."""
        grouped = super()._statistic_factors(factors, scope, ctx)
        found: list[ast.Expr | Node] = []
        position = 0
        while position < len(grouped):
            run: list[ast.Expr] = []
            for factor in grouped[position:]:
                if not isinstance(factor, ast.Expr):
                    break
                run.append(factor)
            match = transforms.written_at(run, 0) if run else None
            if match is not None:
                written, used = match
                self._note_letter(written, scope)
                found.append(self._written_transform(written, scope, ctx))
                position += used
                continue
            found.append(grouped[position])
            position += 1
        return found

    def _note_letter(self, written: Written, scope: Scope) -> None:
        """A calligraphic letter with braces after it is the transform, whatever the plain
        letter is (``design/notation.md``, 9): a note says so where the program gives the
        letter a meaning, as F(s) := ... or a Lagrangian \\mathcal{L} := ... does."""
        name = written.head.name
        if scope.lookup(name) is not None or name in self.session.functions:
            head = _head_written(written.span.text)
            self._note(
                f"{head}, with braces, is read as {written.capability.brief}, not as {name}, "
                "which the program defines: a typeface makes a name of its own"
            )

    def _written_transform(self, written: Written, scope: Scope, ctx: Context) -> Node:
        """A transform as books write it: of what is in braces, at the variable written
        after it, if any."""
        capability = written.capability
        point = self.value(written.point, scope, ctx) if written.point is not None else None
        operand = self._transform_operand(written, point, scope, ctx)
        head = _head_written(written.span.text)
        return self._transform(capability, operand, None, point, written.span, head)

    def _transform_operand(
        self, written: Written, point: Node | None, scope: Scope, ctx: Context
    ) -> Node:
        """What a transform acts on: a function the program defines, named alone, applied to
        its parameter, as f of ``f(t) := \\cos at`` in ``\\mathcal{L}\\{f\\}``; a letter
        nothing defines, a function of the transform's variable, as f in
        ``\\mathcal{L}\\{f\\}``, said in a note; anything else, as it is."""
        operand = written.operand
        capability = written.capability
        if isinstance(operand, ast.Name) and not capability.on_data:
            binding = scope.lookup(operand.id)
            if isinstance(binding, FunctionBinding) and len(binding.definition.params) == 1:
                (param,) = binding.definition.params
                call = ast.Call(operand, (ast.Name(param, span=operand.span),), span=operand.span)
                return self.value(call, scope, ctx)
            if binding is None and self._a_function_letter(operand.id, capability):
                given, _ = self._pair(capability, None, point)
                shown = _unbraced(operand.span.text)
                self._note(
                    f"{shown} is not defined, so {written.span.text} is read as the transform "
                    f"of a function {shown}({_shown(given)})"
                )
                symbol = self.graph.const(self.session.symbol(given), SCALAR, operand.span)
                attributes = (("name", operand.id),)
                return self._call("core.function", [symbol], SCALAR, operand.span, attributes)
        return self.value(operand, scope, ctx)

    def _a_function_letter(self, name: str, capability: Transform) -> bool:
        """Whether a name a transform acts on stands for a function: one letter, Latin or
        Greek, that is none of the transform's variables and no constant."""
        from relativisticpy.compiler.builtins import CONSTANTS

        letters = {letter for pair in capability.variables for letter in pair}
        single = (len(name) == 1 and name.isalpha()) or name in GREEK
        return single and name not in letters and name not in CONSTANTS

    def _builtin(
        self, builtin: Builtin, args: list[Node], span: Span, attributes: ops.Attributes = ()
    ) -> Node:
        capability = integrations._discovery.catalogue().get(builtin.name)
        if not isinstance(capability, Transform):
            return super()._builtin(builtin, args, span, attributes)
        if message := builtin.check_arity(len(args)):  # LaplaceTransform(f, t, s), fft(x_t)
            raise CompileError(message, span=span)
        variable = args[1] if len(args) > 1 else None
        point = args[2] if len(args) > 2 else None
        return self._transform(capability, args[0], variable, point, span, builtin.name)

    def _transform(
        self,
        capability: Transform,
        operand: Node,
        variable: Node | None,
        point: Node | None,
        span: Span,
        head: str,
    ) -> Node:
        """The call of a transform: of data, over a space of its own; of a function of a
        variable, at the variable its convention names, or the point written after it."""
        if capability.on_data:
            return self._transform_data(capability, operand, span)
        if isinstance(operand.type, TensorType):
            raise CompileError(
                f"{head} acts on a function of a variable, not on a tensor",
                span=span,
                hint="for data, the discrete Fourier transform is \\operatorname{fft}(x_t)",
            )
        given, taken = self._transform_variables(capability, operand, variable, point, span, head)
        attributes: ops.Attributes = ()
        if capability.fourier:
            letters = (given.name, _name_of(point) or taken.name)
            convention = self._fourier(capability, letters, head, span)
            attributes = convention.attributes()
        at = point if point is not None else self.graph.const(taken, SCALAR, span)
        args = [operand, self.graph.const(given, SCALAR, span), at]
        kind = SCALAR if operand.type == SCALAR else VALUE
        return self._call(capability.name, args, kind, span, attributes)

    def _transform_variables(
        self,
        capability: Transform,
        operand: Node,
        variable: Node | None,
        point: Node | None,
        span: Span,
        head: str,
    ) -> tuple[sympy.Symbol, sympy.Symbol]:
        """The variable of the function a transform acts on, and its own: the ones written,
        or else those its convention names, by the letters used. The function's variable
        is the conventional one where the function is written with it, as t is in ``\\cos
        at``, or the one variable it is written with; a function of several, none
        conventional, is refused, saying how to name it."""
        if variable is not None:
            given = self._constant(variable)
            if not isinstance(given, sympy.Symbol):
                raise CompileError(
                    f"the variable {capability.name} transforms is a name, not "
                    f"{_shown_value(given)}",
                    span=span,
                    hint=f"for example {capability.text()}",
                )
            pair = self._pair(capability, given.name, point)
            return given, self.session.symbol(pair[1])
        symbols = _symbols(operand)
        names = {symbol.name: symbol for symbol in symbols}
        pair = self._pair(capability, None, point)
        letters = [pair[0], *(g for g, _ in capability.variables if g != pair[0])]
        conventional = [letter for letter in letters if letter in names]
        if conventional:
            pair = self._pair(capability, conventional[0], point)
            return names[conventional[0]], self.session.symbol(pair[1])
        if len(symbols) == 1:
            (only,) = symbols
            return only, self.session.symbol(pair[1])
        if not symbols:
            return self.session.symbol(pair[0]), self.session.symbol(pair[1])
        shown = _listed(sorted(names))
        raise CompileError(
            f"{head} acts here on a function of {shown}, none of which is the variable it "
            f"takes by convention, {pair[0]}",
            span=span,
            hint=f"name the variable, as in {capability.name}(f, x, {pair[1]})",
        )

    def _pair(
        self, capability: Transform, given: str | None, point: Node | None
    ) -> tuple[str, str]:
        """The variables of a transform's convention: the pair whose variable is the point
        written after it, as ω in ``\\mathcal{F}\\{f\\}(\\omega)``; or else whose function's
        variable is ``given``; or else its first."""
        pairs = capability.variables or (("x", "k"),)
        written = _name_of(point)
        for pair in pairs:
            if written is not None and pair[1] == written:
                return pair
        for pair in pairs:
            if given is not None and pair[0] == given:
                return pair
        return pairs[0]

    def _fourier(
        self, capability: Transform, letters: tuple[str, str], head: str, span: Span
    ) -> Convention:
        """The convention of a Fourier transform (``design/notation.md``, 21): the one a
        program declares, defining ℱ by its integral; or else the one an integral it writes
        shows, by its 2π; or else the one its letters choose: x to s, Bracewell's, with
        e^{-2πixs}, and t to ω, of angular frequency, with e^{-iωt}. Said in a note."""
        given, taken = letters
        forward = capability.implementation.fourier != "inverse"
        declared = self.session.conventions.get("fourier")
        if declared is None:
            declared = self._fourier_written()
        if declared is not None:
            convention = declared
        else:
            pair = self._pair(capability, given if forward else taken, None)
            angular = pair[1 if forward else 0] == "omega" or "omega" in letters
            convention = transforms.ANGULAR if angular else transforms.ORDINARY
        shown = convention if forward else convention.inverse()
        variables = (given, taken)
        kernel = shown.kernel(*variables)
        if declared is not None:
            said = f"{head} is read with {kernel}, {convention.said}"
        else:
            said = (
                f"{head} is read with {kernel}, {convention.said}, as it takes "
                f"{_shown(given)} to {_shown(taken)}; to choose another, define "
                "\\mathcal{F} by its integral"
            )
        why = "a Fourier transform is computed in the convention a program chooses"
        self._decide(TRANSFORM, span, kernel, why, said)
        self._note(said)
        return shown

    def _fourier_written(self) -> Convention | None:
        """The convention an integral the program writes shows: the last function it defines
        by an integral over the whole line with e^{ibxk} in it, as Bracewell's
        ``F(s) := \\int_{-\\infty}^{\\infty} f(x) e^{-i 2\\pi x s}\\, dx``."""
        found: Convention | None = None
        for name, binding in self.session.globals.bindings():
            if not isinstance(binding, FunctionBinding) or len(binding.definition.params) != 1:
                continue
            body = binding.definition.body
            if not isinstance(body, ast.Expr):
                continue
            (param,) = binding.definition.params
            read = self._integral_convention(body, param, self.session.globals)
            if read is not None:
                line = binding.definition.span.line
                said = f"as the integral of {name}, on line {line}, writes it"
                found = Convention(read[0].factor, read[0].exponent, said)
        return found

    def _defines_transform(self, statement: ast.Define, scope: Scope, ctx: Context) -> bool:
        """``\\mathcal{F}\\{f\\}(\\omega) := \\int_{-\\infty}^{\\infty} f(t)\\, e^{-i \\omega t}\\,
        dt``: a Fourier transform defined by its integral, which declares the convention
        every later one is computed in, its factor, the sign and the 2π of its exponent, and
        its variables; ``False`` for any other definition."""
        if len(statement.targets) != 1:
            return False
        factors = _factors(statement.targets[0])
        match = transforms.written_at(factors, 0)
        if match is None or match[1] != len(factors):
            return False
        written = match[0]
        capability = written.capability
        head = _head_written(statement.targets[0].span.text)
        example = (
            "\\mathcal{F}\\{f\\}(\\omega) := \\int_{-\\infty}^{\\infty} f(t)\\, "
            "e^{-i\\omega t}\\, dt"
        )
        if not capability.fourier:
            raise CompileError(
                f"{head} is {capability.brief}, which has one definition; defined by its "
                "integral, a Fourier transform chooses its convention",
                span=statement.span,
                hint=f"for example {example}",
            )
        point, function = written.point, written.operand
        if not (isinstance(point, ast.Name) and isinstance(function, ast.Name)):
            raise CompileError(
                f"{head} is defined by its integral as {example} is: of a function named "
                "in braces, at a variable named in brackets",
                span=statement.span,
            )
        read = self._integral_convention(statement.value, point.id, scope, function.id)
        if read is None:
            raise CompileError(
                f"{statement.value.span.text} is no Fourier integral: an integral over the "
                f"whole line of {function.id} times e^{{i b x {point.id}}}",
                span=statement.value.span,
                hint=f"for example {example}",
            )
        convention, variable = read
        said = f"as line {statement.span.line} defines it"
        if capability.implementation.fourier == "inverse":
            convention = convention.inverse()
        self.session.conventions["fourier"] = Convention(
            convention.factor, convention.exponent, said
        )
        given, taken = (variable, point.id)
        if capability.implementation.fourier == "inverse":
            given, taken = taken, given
        kernel = convention.kernel(given, taken)
        why = "it defines the transform by its integral"
        self._decide(TRANSFORM, statement.span, f"the convention {kernel}", why)
        return True

    def _integral_convention(
        self, value: ast.Expr, point: str, scope: Scope, function: str | None = None
    ) -> tuple[Convention, str] | None:
        """The convention a Fourier integral is written in, and its variable: of
        ``a \\int_{-\\infty}^{\\infty} f(x) e^{i b x k}\\, dx``, the factor a and b, with k
        the ``point`` and f the ``function``, if one is named; ``None`` for any other
        value."""
        factors = _factors(value)
        integrals = [factor for factor in factors if isinstance(factor, ast.Integral)]
        if len(integrals) != 1 or len(integrals[0].bounds) != 1:
            return None
        integral = integrals[0]
        bound = integral.bounds[0]
        if not isinstance(bound.variable, ast.Name) or bound.lower is None or bound.upper is None:
            return None
        ctx = Context(latex=True)
        try:
            ends = (bound.lower, bound.upper)
            limits = [self._folded(self.value(end, scope, ctx)) for end in ends]
            constants = [
                self._folded(self.value(factor, scope, ctx))
                for factor in factors
                if factor is not integral
            ]
        except RelativisticPyError:
            return None
        if limits != [-sympy.oo, sympy.oo] or any(c is None or c.free_symbols for c in constants):
            return None
        variable = bound.variable.id
        x, k = sympy.Dummy(variable), sympy.Dummy(point)
        inner = scope.child()
        inner.bind(variable, VariableBinding(self.graph.const(x, SCALAR)))
        inner.bind(point, VariableBinding(self.graph.const(k, SCALAR)))
        kernel: Any = None
        for part in _factors(integral.integrand):
            if function is not None and _calls(part, function, variable):
                continue
            try:
                folded = self._folded(self.value(part, inner, ctx))
            except RelativisticPyError:
                return None
            if folded is None:
                if function is None:
                    continue  # the function transformed, as e^{-\pi x^2}
                return None
            kernel = folded if kernel is None else kernel * folded
        if kernel is None:
            return None
        convention = transforms.convention_of(kernel, x, k, sympy.Mul(*constants))
        return (convention, variable) if convention is not None else None

    def _Relation(self, node: ast.Relation, scope: Scope, ctx: Context) -> Node:
        if node.op == "⊃" and (pair := self._fourier_pair(node, scope, ctx)) is not None:
            return pair
        return super()._Relation(node, scope, ctx)

    def _fourier_pair(self, node: ast.Relation, scope: Scope, ctx: Context) -> Node | None:
        """``e^{-\\pi x^2} \\supset e^{-\\pi s^2}``, Bracewell's notation for a Fourier pair
        between functions (``design/notation.md``, 21): the claim that the right side is the
        transform of the left, at the variable the right side is written in. ``None`` where
        the left side is no function of a variable a Fourier transform takes, as of a set."""
        capability = integrations._discovery.catalogue().get("FourierTransform")
        if not isinstance(capability, Transform):
            return None
        function = self.value(node.lhs, scope, ctx)
        transformed = self.value(node.rhs, scope, ctx)
        if function.type != SCALAR or transformed.type != SCALAR:
            return None
        given = {symbol.name for symbol in _symbols(function)}
        if not given & {letter for letter, _ in capability.variables}:
            return None
        written = [symbol for symbol in _symbols(transformed) if symbol.name not in given]
        point = self.graph.const(written[0], SCALAR) if len(written) == 1 else None
        why = "between functions, it is Bracewell's notation for a Fourier pair"
        self._decide(TRANSFORM, node.span, "a Fourier pair", why)
        transform = self._transform(capability, function, None, point, node.span, "\\supset")
        return self.graph.add(
            ops.Equation(), (transform, transformed), EquationType(SCALAR), node.span
        )

    # -- transforms of data ---------------------------------------------------------------

    def _transform_data(self, capability: Transform, operand: Node, span: Span) -> Node:
        """A transform of data over one index, or of a list, data over the space its
        capability names, as the frequencies of the samples."""
        if not isinstance(operand.type, TensorType):
            size = _elements(operand)
            if size is None:
                raise CompileError(
                    f"{capability.name} transforms data over an index, or a list written in "
                    "the program, not this",
                    span=span,
                    hint=f"for example {capability.name}(x_t), or {capability.name}([1, 2, 0, -1])",
                )
            written = operand.span.text if operand.span is not None else "the list"
            elements = Space.from_data(f"the elements of {written}", size)
            kind = TensorType(Indices([Index("#n")]), (elements,))
            operand = self.graph.add(ops.ToArray((size,)), (operand,), kind, span)
        assert isinstance(operand.type, TensorType)
        if operand.type.rank != 1:
            raise CompileError(
                f"{capability.name} of data over {operand.type.rank} indices is not supported "
                "yet: it transforms data over one index",
                span=span,
                hint=f"transform one column at a time, as {capability.name}(x_t)",
            )
        (index,), (over,) = operand.type.indices, operand.type.spaces
        space = capability.implementation.space(over)
        kind = TensorType(Indices([Index("#k", index.upper)]), (space,))
        return self._call(capability.name, [operand], kind, span)

    # -- ranges of index labels ------------------------------------------------------------

    def _declares(self, relation: ast.Relation) -> bool:
        return self._declares_range(relation, local=False) or super()._declares(relation)

    def _declares_range(self, relation: ast.Relation, *, local: bool) -> bool:
        """``n = 0, \\ldots, N-1``: the labels range over a space of the values, each its
        own element, labelled by it, for the definition it is written beside, where
        ``local``, and for the rest of the program otherwise, where it is written with an
        ellipsis, so that ``\\mu = 0, 1, 2, 3`` still says what a spacetime index is.
        ``False`` for any other relation."""
        if relation.op != "∈" or relation.written or not isinstance(relation.rhs, ast.Tuple):
            return False
        members = relation.lhs.items if isinstance(relation.lhs, ast.Tuple) else (relation.lhs,)
        if not all(isinstance(member, ast.Name) for member in members):
            return False
        labels = [member.id for member in members if isinstance(member, ast.Name)]
        items = relation.rhs.items
        dots = [k for k, item in enumerate(items) if isinstance(item, ast.Dots)]
        if not items or (dots and dots[-1] == len(items) - 1) or (not dots and not local):
            return False  # n = 0, 1, 2, \ldots: no end
        ctx = Context(latex=True)
        values: list[Any] = []
        for item in items:
            if isinstance(item, ast.Dots):
                values.append(None)
                continue
            values.append(self._folded(self.value(item, self.session.globals, ctx)))
        known = [v for v in values if v is not None]
        if not all(getattr(v, "is_integer", False) for v in known):
            if not local:
                return False
            raise CompileError(
                f"{relation.span.text} gives the values of {_listed(labels)} by what is not "
                "known before anything is computed",
                span=relation.span,
                hint=_KNOWN_FIRST,
            )
        first, last = int(known[0]), int(known[-1])
        listed = [int(v) for v in known[:-1]] if dots else [int(v) for v in known]
        if {b - a for a, b in itertools.pairwise(listed)} not in (set(), {1}):
            raise CompileError(
                f"{relation.span.text}: a range of values one apart is supported, not yet another",
                span=relation.span,
            )
        if last < first:
            raise CompileError(f"{relation.span.text} gives no values", span=relation.span)
        named = f"{', '.join(labels)} = {first}, …, {last}" if dots else relation.span.text
        space = Space(named, last - first + 1, tuple(str(v) for v in range(first, last + 1)))
        self.session.ranges.add(space)
        where = self._labels if local else self.session.labels
        for label in labels:
            where[label] = space
        shown = _listed([self._written_label(label) for label in labels])
        why = "a range of values written for an index declares the space it ranges over"
        self._decide(RANGE, relation.span, f"{shown} range over {first}, …, {last}", why)
        return True

    def _range_of(self, label: str) -> Space | None:
        """The space a label ranges over where a range declared it, its elements its values,
        consecutive integers; ``None`` for any other, as the months of data numbered 1 to 6,
        over which r_f, after r_t, is not r for each month but a name of its own."""
        space = self._labels.get(label) or self.session.labels.get(label)
        if space is None or space not in self.session.ranges:
            return None
        return space if _first(space) is not None else None

    @contextlib.contextmanager
    def _over_ranges(self, statement: ast.Define) -> Iterator[None]:
        """``x_n := n h, \\quad n = 0, \\ldots, N-1``: the ranges written beside a definition
        declare the spaces of its labels, and the definition, of a target written with a
        label that ranges over values, is made for each of them."""
        for relation in statement.ranges:
            if not self._declares_range(relation, local=True):
                raise CompileError(
                    f"{relation.span.text} gives no range of values for "
                    f"{statement.targets[0].span.text}",
                    span=relation.span,
                    hint="write the range as n = 0, \\ldots, N-1",
                )
        target = statement.targets[0] if len(statement.targets) == 1 else None
        labels = {
            label: upper
            for label, upper in (_target_labels(target) if target is not None else ())
            if self._range_of(label) is not None
        }
        for label in labels:
            space = self._range_of(label)
            assert space is not None
            self._labels.setdefault(label, space)
        saved = self._ranged, self._bound
        self._ranged = labels
        self._bound = self._bound | frozenset(labels)
        try:
            yield
        finally:
            self._ranged, self._bound = saved

    def _define_now(self, target: ast.Expr, value: ast.Expr, scope: Scope, ctx: Context) -> None:
        if not self._ranged:
            super()._define_now(target, value, scope, ctx)
            return
        named = _target(target)
        if named is None:
            raise CompileError("cannot define this over a range", span=target.span)
        name, entries = named
        indices = Indices(Index(label, upper) for label, upper in entries)
        node = self.value(value, scope, ctx)
        have = node.type.indices.names if isinstance(node.type, TensorType) else ()
        missing = [index for index in indices if index.name not in have]
        if missing:  # x_n := 1: the same for each n
            factors = [(node, node.type.indices)] if isinstance(node.type, TensorType) else []
            for index in missing:
                space = self._range_of(index.name) or self._space_of(index.name, target.span)
                ones = self._ones((index.name, index.upper), target.span, space)
                factors.append((ones, Indices([index])))
            spread = self._contract(factors, value.span) if len(factors) > 1 else factors[0][0]
            node = (
                spread
                if isinstance(node.type, TensorType)
                else self._multiply([node, spread], value.span)
            )
        node = self._align(node, indices, value.span, what="a definition")
        scope.bind(name, TensorBinding(node, indices))

    def _Name(self, node: ast.Name, scope: Scope, ctx: Context) -> Term:
        name = node.id
        if name in self._summed and scope.lookup(name) is None:
            return self._summed[name]
        if name in self._bound and scope.lookup(name) is None:
            space = self._range_of(name)
            if space is not None:  # n of x_n := n h, \quad n = 0, \ldots, N-1
                upper = self._ranged.get(name, False)
                first = _first(space)
                assert first is not None
                why = f"{self._written_label(name)} ranges over its values here"
                reading = f"the values of {name}, {first} to {first + space.dim - 1}"
                self._decide(RANGE, node.span, reading, why)
                return self._positions(name, upper, space, first, node.span)
        return super()._Name(node, scope, ctx)

    def _positions(self, label: str, upper: bool, space: Space, first: int, span: Span) -> Node:
        """The values a label takes over its space, from ``first``: a vector over it."""
        kind = TensorType(Indices([Index(label, upper)]), (space,))
        args = (self.graph.const(first, VALUE), self.graph.const(space.dim, VALUE))
        return self.graph.add(ops.Call("core.positions"), args, kind, span)

    def _shared(self, node: Node) -> bool:
        """Whether a tensor's indices are all labels shared by every factor, as a sum over
        them or a definition over their ranges makes them: its powers and quotients are
        taken for each component."""
        return isinstance(node.type, TensorType) and set(node.type.indices.names) <= self._bound

    def _power(self, base: Node, exponent: Node, span: Span) -> Node:
        if self._shared(base) and exponent.type == SCALAR:  # n^2, for each n
            return self.graph.add(ops.Map("core.power"), (base, exponent), base.type, span)
        if self._shared(exponent) and base.type == SCALAR:  # e^{-2\pi i k n/N}
            op = ops.Map("core.power", (), 1)
            return self.graph.add(op, (exponent, base), exponent.type, span)
        return super()._power(base, exponent, span)

    def _absolute(self, node: ast.Abs, operand: Node) -> Node:
        if self._shared(operand):  # |X_k|, for each k of a sum over k
            return self._builtin(BUILTINS["Abs"], [operand], node.span)
        return super()._absolute(node, operand)

    def _divide(self, numerator: Node, denominator: Node, span: Span) -> Node:
        if self._shared(denominator):  # 1/(n + 1), for each n
            minus_one = self.graph.const(sympy.S.NegativeOne, SCALAR)
            reciprocal = self._power(denominator, minus_one, span)
            return self._multiply([numerator, reciprocal], span)
        return super()._divide(numerator, denominator, span)

    # -- sums over an index of data --------------------------------------------------------

    def _BigOperator(self, node: ast.BigOperator, scope: Scope, ctx: Context) -> Node:
        if node.op == "sum" and (summed := self._index_sum(node, scope, ctx)) is not None:
            return summed
        return super()._BigOperator(node, scope, ctx)

    def _index_sum(self, node: ast.BigOperator, scope: Scope, ctx: Context) -> Node | None:
        """``\\sum_{n=0}^{N-1} x_n e^{-2\\pi i k n/N}``: a sum whose body writes its variable
        as an index of data, or of a list, which it ranges over from its first element to
        its last, from the lower limit (``design/notation.md``, 17). It binds the index as a
        sum does, every factor sharing it, and is a contraction over its space; n, as a
        number, is its value. A rewrite may compute it faster. ``None`` for any other sum,
        computed as before."""
        if not isinstance(node.variable, ast.Name):
            return None
        label = node.variable.id
        carriers = self._carriers(label, node.body, scope)
        if not carriers:
            return None
        runs = [
            run
            for run in self._runs_through(node.body, label, scope, ctx)
            if run.kind is not Kind.ELEMENT
        ]
        if runs and all(space is None for _, space, _, _ in carriers):
            return None  # x_i y_i, of a list x and a family y: taken term by term
        for run in runs:
            if run.kind is Kind.LABEL and run.written is not None:
                text = run.written.node.span.text
                raise CompileError(
                    f"{text} names nothing, and the sum runs over its subscript: read as "
                    "written, it would be one symbol in every term",
                    span=node.span,
                    hint=f"define it over the index, as in {text} := [...]",
                )
        written = f"\\{node.op}"
        limits = self._sum_limits(node, scope, ctx)
        tensors = [(name, space, upper) for name, space, upper, _ in carriers if space is not None]
        lists = [(name, size) for name, space, _, size in carriers if space is None]
        if tensors:
            space = tensors[0][1]
            upper = tensors[0][2]
        else:
            if limits is None:
                raise CompileError(
                    f"{written} over the elements of a list needs its limits, from which "
                    "they are counted",
                    span=node.span,
                    hint=f"for example \\sum_{{{label}=0}}^{{N-1}}",
                )
            first, last = limits
            space = Space(
                f"{label} = {first}, …, {last}",
                last - first + 1,
                tuple(str(v) for v in range(first, last + 1)),
            )
            upper = False
        given = _first(space)
        first = limits[0] if limits is not None else given if given is not None else 0
        if limits is not None and limits[1] - limits[0] + 1 != space.dim:
            raise CompileError(
                f"{written} runs over {limits[1] - limits[0] + 1} values of "
                f"{self._written_label(label)}, which ranges over {space.dim}: a sum over a "
                "part of its space is not supported yet",
                span=node.span,
                hint=f"sum over all of it, from {first} to {first + space.dim - 1}",
            )
        if given is not None and first != given:
            raise CompileError(
                f"{self._written_label(label)} ranges over {given}, …, "
                f"{given + space.dim - 1}, but {written} counts it from {first}",
                span=node.span,
                hint=f"sum from {given} to {given + space.dim - 1}",
            )
        for name, size in lists:
            if size != space.dim:
                raise CompileError(
                    f"{name} has {size} elements, but {written} runs over {space.dim}: a sum "
                    "over a part of a list is not supported yet",
                    span=node.span,
                )
        inner = scope.child()
        positions = self._positions(label, upper, space, first, node.variable.span)
        for name, _ in lists:  # x_n of x := (1, 2, 0, -1): the list over the sum's space
            binding = scope.lookup(name)
            assert isinstance(binding, VariableBinding)
            index = Indices([Index(label, upper)])
            kind = TensorType(index, (space,))
            array = self.graph.add(ops.ToArray((space.dim,)), (binding.node,), kind, node.span)
            inner.bind(name, TensorBinding(array, index))
            why = f"the sum over {label} runs over all of {name}, from {first}"
            self._decide("index or power", node.span, f"the elements of {name}", why)
        with self._binding(label), self._summing(label, positions):
            body = self.value(node.body, inner, ctx)
        if not (isinstance(body.type, TensorType) and label in body.type.indices.names):
            raise IndexNotationError(
                f"{written} over {self._written_label(label)} sums what has no index "
                f"{self._written_label(label)}",
                span=node.span,
            )
        summed_index = body.type.indices[body.type.indices.names.index(label)]
        ones = self._ones((label, not summed_index.upper), node.span, space)
        assert isinstance(ones.type, TensorType)
        with self._unbinding(label):
            summed = self._contract(
                [(body, body.type.indices), (ones, ones.type.indices)], node.span
            )
        if transforms.rewriting():
            with self._summing(label, positions):
                rewritten = self._rewritten(node, label, space, first, inner, scope, ctx)
            if rewritten is not None:
                if isinstance(summed.type, TensorType):
                    return self._align(rewritten, summed.type.indices, node.span)
                return rewritten
        return summed

    @contextlib.contextmanager
    def _summing(self, label: str, positions: Node) -> Iterator[None]:
        """The values of a label a sum binds, while what it sums is compiled."""
        saved = self._summed
        self._summed = {**saved, label: positions}
        try:
            yield
        finally:
            self._summed = saved

    def _carriers(
        self, label: str, body: ast.Expr, scope: Scope
    ) -> list[tuple[str, Space | None, bool, int]]:
        """What a sum's body writes its variable on as an index: each tensor, with the space
        of that axis and whether the index is up, and each list, with its length, as x is
        in x_n of x := (1, 2, 0, -1), where the resolver reads the name as its element."""
        found: list[tuple[str, Space | None, bool, int]] = []
        for current in _walk(body):
            entries: list[tuple[str, bool]] | None = None
            name: str | None = None
            if isinstance(current, ast.Scripted) and (pattern := _tensor_pattern(current)):
                name, entries = pattern[0], list(pattern[1])
            elif isinstance(current, ast.Name) and "_" in current.id:
                base, _, tail = current.id.partition("_")
                if base and tail.isalpha() and scope.lookup(current.id) is None:
                    name = base
                    entries = [(each, False) for each in split_index_labels(tail)]
            if name is None or entries is None:
                continue
            labels = [each for each, _ in entries]
            if label not in labels:
                continue
            binding = scope.lookup(name)
            node = binding.node if isinstance(binding, VariableBinding | TensorBinding) else None
            if node is not None and isinstance(node.type, TensorType):
                axis = labels.index(label)
                if axis < len(node.type.spaces):
                    found.append((name, node.type.spaces[axis], entries[axis][1], 0))
            elif len(labels) == 1 and (size := self._list_summed(current, label, scope)):
                found.append((name, None, False, size))
        return found

    def _sum_limits(
        self, node: ast.BigOperator, scope: Scope, ctx: Context
    ) -> tuple[int, int] | None:
        """The first and the last value of a sum over an index of data, which are known
        before anything is computed; ``None`` where none are written, as in ``\\sum_n``."""
        if node.lower is None and node.upper is None:
            return None
        if node.lower is None or node.upper is None:
            raise CompileError("give both limits of the sum, or neither", span=node.span)
        ends = [self._folded(self.value(end, scope, ctx)) for end in (node.lower, node.upper)]
        if not all(end is not None and getattr(end, "is_integer", False) for end in ends):
            raise CompileError(
                f"the limits of \\{node.op} over an index of data are whole numbers known "
                "before anything is computed",
                span=node.span,
                hint=_KNOWN_FIRST,
            )
        return int(ends[0]), int(ends[1])

    def _rewritten(
        self,
        node: ast.BigOperator,
        label: str,
        space: Space,
        first: int,
        inner: Scope,
        scope: Scope,
        ctx: Context,
    ) -> Node | None:
        """A sum over an index of data computed as a rewrite says, faster, as the discrete
        Fourier transform written as its sum is computed as an FFT: its body is the data,
        times a kernel in the index and one other, which ranges over as many values, from
        0; ``None`` where no rewrite is the sum."""
        factors = _factors(node.body)
        data = [f for f in factors if self._carriers(label, f, scope)]
        kernel = [f for f in factors if f not in data]
        if len(data) != 1 or not kernel or any(self._names_tensor(f, scope) for f in kernel):
            return None
        mentioned = {n.id for f in kernel for n in _walk(f) if isinstance(n, ast.Name)}
        free = [
            other
            for other in mentioned
            if other != label and other in self._bound and self._range_of(other) is not None
        ]
        if len(free) != 1:
            return None
        (other,) = free
        over = self._range_of(other)
        assert over is not None
        start = _first(over)
        if over.dim != space.dim or start is None:
            return None
        j, m = sympy.Dummy("j", integer=True), sympy.Dummy("m", integer=True)
        symbolic = scope.child()
        symbolic.bind(label, VariableBinding(self.graph.const(first + j, SCALAR)))
        symbolic.bind(other, VariableBinding(self.graph.const(start + m, SCALAR)))
        read = dict(self._decided), dict(self._resolved), list(self._notes)
        try:  # compiled to see what the kernel is, not to be read so
            values = [self._folded(self.value(f, symbolic, ctx)) for f in kernel]
        except RelativisticPyError:
            return None
        finally:
            self._decided, self._resolved, self._notes = read
        if any(value is None for value in values):
            return None
        written = sympy.Mul(*values)
        for rewrite in transforms.rewrites():
            scale = rewrite.exact(written, j, m, space.dim)
            if scale is not None:
                break
        else:
            return None
        with self._binding(label):
            values_node = self.value(data[0], inner, ctx)
        if not isinstance(values_node.type, TensorType):
            return None
        names = values_node.type.indices.names
        axis = names.index(label)
        order = [axis, *(k for k in range(len(names)) if k != axis)]
        moved = self._permute(
            values_node, order, Indices(values_node.type.indices[k] for k in order)
        )
        assert isinstance(moved.type, TensorType)
        upper = self._ranged.get(other, False)
        indices = Indices([Index(other, upper), *moved.type.indices[1:]])
        kind = TensorType(indices, (over, *moved.type.spaces[1:]))
        result = self._call(rewrite.computes, [moved], kind, node.span)
        if sympy.sympify(scale) != 1:
            factor = self.graph.const(sympy.sympify(scale), SCALAR)
            result = self.graph.add(ops.Scale(), (factor, result), kind, node.span)
        computes = rewrite.computes
        why = f"it is {rewrite.brief}, which {computes} computes to the same numbers, faster"
        self._decide("rewrite", node.span, f"computed by {rewrite.computes}", why)
        return result

    # -- what is known before anything is computed ----------------------------------------

    def _folded(self, node: Node) -> Any:
        """The value of a node made only of numbers, symbols and arithmetic, and the
        elementary functions of them, as -2πi·k·n/N or N - 1 is, which is known when
        compiling; ``None`` for a value computed, which is not."""
        op = node.op
        if isinstance(op, ops.Const):
            value = op.value
            if isinstance(value, bool) or not isinstance(value, int | float | complex | sympy.Expr):
                return None
            return sympy.sympify(value)
        args = [self._folded(arg) for arg in node.args]
        if any(arg is None for arg in args):
            return None
        if isinstance(op, ops.Add):
            return sympy.Add(*(sign * arg for sign, arg in zip(op.signs, args, strict=True)))
        if isinstance(op, ops.Mul | ops.Scale):
            return sympy.Mul(*args)
        if isinstance(op, ops.Pow):
            return sympy.Pow(*args)
        if isinstance(op, ops.Call) and not op.attributes and op.name in _elementary():
            return integrations.find(op.name).exact(*args)
        return None


def _elementary() -> frozenset[str]:
    """The names of the elementary functions, which fold with their arguments."""
    return frozenset(
        name
        for capability in integrations.capabilities()
        if capability.module.endswith(".elementary")
        for name in capability.names
    )


def _symbols(node: Node) -> list[sympy.Symbol]:
    """The symbols a value is written with, as t and a are in ``\\cos at``, in the order
    they are first met; not the dummies of a derivative's body."""
    found: dict[sympy.Symbol, None] = {}
    for current in dependencies([node]):
        value = current.op.value if isinstance(current.op, ops.Const) else None
        if isinstance(value, sympy.Basic):
            for symbol in sorted(value.free_symbols, key=str):
                if isinstance(symbol, sympy.Symbol) and not isinstance(symbol, sympy.Dummy):
                    found.setdefault(symbol, None)
    return list(found)


def _name_of(node: Node | None) -> str | None:
    """The name of a symbol a node is, as the ω of ``\\mathcal{F}\\{f\\}(\\omega)``."""
    if node is not None and isinstance(node.op, ops.Const):
        value = node.op.value
        if isinstance(value, sympy.Symbol):
            return str(value.name)
    return None


def _first(space: Space) -> int | None:
    """The first value of a space whose elements are consecutive integers, as a range's
    are; ``None`` for any other."""
    basis = space.basis
    if not basis:
        return None
    try:
        values = [int(name) for name in basis]
    except ValueError:
        return None
    if values != list(range(values[0], values[0] + len(values))):
        return None
    return values[0]


def _factors(node: ast.Expr) -> list[ast.Expr]:
    """The factors of a product, in order."""
    found: list[ast.Expr] = []
    while isinstance(node, ast.Binary) and node.op == "*":
        found.insert(0, node.right)
        node = node.left
    found.insert(0, node)
    return found


def _calls(node: ast.Expr, function: str, variable: str) -> bool:
    """Whether an expression is ``f(x)``, the function called with the variable."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == variable
    )


def _target_labels(target: ast.Expr) -> list[tuple[str, bool]]:
    """The index labels a definition's target is written with, each with whether it is
    up: n of ``x_n``, and a, up, and b of ``T^{a}{}_{b}``."""
    named = _target(target)
    return named[1] if named is not None else []


def _target(target: ast.Expr) -> tuple[str, list[tuple[str, bool]]] | None:
    """The name a definition over ranges defines, and its labels: x and n of ``x_n``."""
    if isinstance(target, ast.Name):
        base, underscore, tail = target.id.partition("_")
        if base and underscore and tail.isalpha():
            return base, [(label, False) for label in split_index_labels(tail)]
        return None
    if isinstance(target, ast.Scripted) and isinstance(target.base, ast.Name):
        pattern = _tensor_pattern(target)
        if pattern is not None:
            return pattern[0], list(pattern[1])
    return None


def _shown(name: str) -> str:
    """A variable's name as a note writes it: ω, t."""
    from relativisticpy import display

    return display.unicode(sympy.Symbol(name))


def _head_written(written: str) -> str:
    """The head of a transform as written, ``\\mathcal{L}^{-1}`` of
    ``\\mathcal{L}^{-1}\\{F\\}(t)``."""
    return written.split("\\{")[0].split("\\left")[0].split("\\lbrace")[0].strip()


def _unbraced(written: str) -> str:
    """A name as written, without the braces a transform puts it in: F of ``\\{F\\}``."""
    for opening, closing in (("\\left\\{", "\\right\\}"), ("\\{", "\\}"), ("\\lbrace", "\\rbrace")):
        if written.startswith(opening) and written.endswith(closing):
            return written[len(opening) : -len(closing)].strip()
    return written


def _shown_value(value: Any) -> str:
    from relativisticpy import display

    return display.text(value) if value is not None else "what is computed"
