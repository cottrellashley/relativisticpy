r"""The lowering of convolutions, as :mod:`relativisticpy.reading.convolutions` reads them
(design/notation.md, decision 27):

- ``*`` or ``\ast`` between two functions applied at the same variable, ``f(t) * g(t)``, is
  their convolution, the integral over every τ of f(τ) g(t − τ), and between two sequences
  at points of the same index, ``x[n] * h[n]``, the sum over every integer k of
  x[k] h[n − k], each with a note. In their operands, u and δ applied at the variable,
  where nothing defines them, are the unit step and the Dirac delta, as signal processing
  writes them. Between anything else, ``*`` multiplies;
- ``x \circledast h`` is the circular convolution, the sum over m from 0 to N − 1 of
  x[m] h[(n − m) mod N], with N its subscript, as in ``\circledast_N``, or else the length
  of the longer sequence: at a point, of sequences the program defines,
  ``(x \circledast h)[0]``, and of sequences at points of the same index,
  ``x[n] \circledast h[n]``.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import sympy

from relativisticpy import integrations
from relativisticpy.compiler.builtins import BUILTINS, CONSTANTS
from relativisticpy.compiler.passes import Term
from relativisticpy.compiler.scope import Context, FunctionBinding, Scope, VariableBinding
from relativisticpy.compiler.session import Session
from relativisticpy.errors import CompileError
from relativisticpy.ir import SCALAR, VALUE, Node, ops
from relativisticpy.language import syntax as ast
from relativisticpy.lowering.sequences import Sequences
from relativisticpy.reading import convolutions, sequences
from relativisticpy.reading.convolutions import CONVOLUTION
from relativisticpy.reading.families import Family
from relativisticpy.reading.sequences import Response, Sample
from relativisticpy.reading.sequences import Sequence as Defined
from relativisticpy.typing.notation import _passed

#: The names signal processing writes the unit step and the Dirac delta with, applied at a
#: time, and the capabilities they are.
_SIGNALS = {"u": "Heaviside", "delta": "DiracDelta"}


class Convolutions(Sequences):
    """The lowering of convolutions."""

    def __init__(self, session: Session) -> None:
        super().__init__(session)
        # The calls of u and δ written in the operands of the convolutions being lowered,
        # which are the unit step and the Dirac delta, each with the capability it is.
        self._signals: dict[int, str] = {}

    # -- an asterisk between two signals ---------------------------------------------------

    def _product(self, node: ast.Binary, scope: Scope, ctx: Context) -> Term:
        if (circular := self._circular_at(node, scope, ctx)) is not None:
            return circular  # (x \circledast h)[0]
        written = convolutions.factors(node)
        if not any(convolutions.starred(*pair) for pair in itertools.pairwise(written)):
            return super()._product(node, scope, ctx)
        factors = self._samples_in(written, scope)
        for position in reversed(range(len(factors) - 1)):  # the last asterisk first
            right, rest = factors[position + 1], factors[position + 2 :]
            star = convolutions.starred(factors[position], right)
            if star is None or self._signal_names(right, scope) is None:
                continue
            left = convolutions.product_of(factors[: position + 1])
            found = self._convolved(left, right, scope)
            if found is None:
                continue
            variable, discrete = found
            value = self._convolution(left, right, variable, discrete, star, scope, ctx)
            if not rest:
                return value
            after = self.value(convolutions.product_of(rest), scope, ctx)
            return self._multiply([value, after], node.span)
        return super()._product(node, scope, ctx)

    def _samples_in(self, factors: Sequence[ast.Expr], scope: Scope) -> list[ast.Expr]:
        """Factors with each name and the point in square brackets right after it made one
        sample, where it is a sequence there, as x and [n] of ``x[n] * h[n]`` are."""
        found: list[ast.Expr] = []
        position = 0
        while position < len(factors):
            factor = factors[position]
            following = factors[position + 1] if position + 1 < len(factors) else None
            written = sequences.written_after(factor, following) if following else None
            if following and written and not written[1]:
                assert isinstance(factor, ast.Name)
                if self._a_sequence(factor, written[0], scope):
                    span = factor.span.to(following.span)
                    found.append(Sample(factor, (written[0],), span=span))
                    position += 2
                    continue
            found.append(factor)
            position += 1
        return found

    def _convolved(
        self, left: ast.Expr, right: ast.Expr, scope: Scope
    ) -> tuple[ast.Name, bool] | None:
        """The variable two signals are convolved in, as it is written on the right, and
        whether they are sequences: the one name that every function applied in both is
        applied at, or that every sequence in both is at. ``None`` where they are not both
        signals, both functions or both sequences, of one variable."""
        first, second = self._signal_names(left, scope), self._signal_names(right, scope)
        if first is None or second is None or first[1] != second[1]:
            return None
        shared = set(first[0]) & set(second[0])
        if len(shared) != 1:
            return None
        (name,) = shared
        return second[0][name], second[1]

    def _signal_names(
        self, side: ast.Expr, scope: Scope
    ) -> tuple[dict[str, ast.Name], bool] | None:
        """The names a signal is written at, each with a place it is written, and whether it
        is a sequence: those that every function applied in it is applied at, or that every
        sequence in it is at. ``None`` for what is no signal."""
        calls = convolutions.applied(side)
        written = [(call.func, call.args[0]) for call in calls if isinstance(call, Sample)]
        written += sequences.samples(side)
        points = [
            point
            for name, point in written
            if isinstance(name, ast.Name) and self._a_sequence(name, point, scope)
        ]
        arguments = [
            call.args[0]
            for call in calls
            if not isinstance(call, Sample) and self._a_signal(call, scope)
        ]
        if bool(points) == bool(arguments):
            return None  # no signal, or sequences beside functions
        names = [convolutions.names_in(each) for each in points or arguments]
        shared = set(names[0]).intersection(*names[1:]) - set(CONSTANTS)
        return {name: names[0][name] for name in shared}, bool(points)

    def _a_signal(self, call: ast.Call, scope: Scope) -> bool:
        """Whether a call is a signal at a time: of a function of one variable the program
        defines, or of a name nothing defines, as f(t), u(t) and \\delta(t) are; not of a
        builtin, as \\sin t is, nor of a function of the coordinates."""
        func = call.func
        assert isinstance(func, ast.Name)
        binding = scope.lookup(func.id)
        if isinstance(binding, FunctionBinding):
            definition = binding.definition
            return len(definition.params) == 1 and not isinstance(
                definition, Defined | Family | Response
            )
        if binding is not None or func.id in BUILTINS or func.id in CONSTANTS:
            return False
        return not self._has_a_meaning(func.id)

    def _convolution(
        self,
        left: ast.Expr,
        right: ast.Expr,
        variable: ast.Name,
        discrete: bool,
        star: str,
        scope: Scope,
        ctx: Context,
    ) -> Node:
        """The convolution of two signals in their variable: the integral over every τ of the
        left at τ times the right at the variable less τ; of sequences, the sum over every
        integer k of the left at k times the right at the index less k."""
        span = left.span.to(right.span)
        shown = variable.span.text
        if discrete:
            what = f"the sum over every integer k of the first at k times the second at {shown} - k"
            why = f"{star} stands between two sequences at points of {shown}"
        else:
            what = f"the integral over every τ of the first at τ times the second at {shown} - τ"
            why = f"{star} stands between two functions applied at {shown}"
        note = (
            f"{span.text} is read as the convolution of {left.span.text} and "
            f"{right.span.text}, {what}, as signal processing writes it; for their product, "
            f"write them side by side, without {star}"
        )
        self._note(note)
        self._decide(CONVOLUTION, span, f"their convolution, in {shown}", why, note)
        at = self.value(variable, scope, ctx)
        dummy = sympy.Dummy("k", integer=True) if discrete else sympy.Dummy("tau", real=True)
        moved = self.graph.const(dummy, SCALAR, span)
        before = self.graph.add(ops.Add((1, -1)), [at, moved], SCALAR, span)
        first, second = scope.child(), scope.child()
        first.bind(variable.id, VariableBinding(moved))
        second.bind(variable.id, VariableBinding(before))
        with self._signals_of(left, right, variable, scope):
            values = [self.value(left, first, ctx), self.value(right, second, ctx)]
        body = self._multiply(values, span)
        ends = [self.graph.const(end, SCALAR) for end in (-sympy.oo, sympy.oo)]
        if discrete:
            return self._call("seq.sum", [body, moved, *ends], _passed(body.type), span)
        limits = self._call("core.limits", [moved, *ends], VALUE, span)
        return self._call("integrate", [body, limits], _passed(body.type), span)

    @contextmanager
    def _signals_of(
        self, left: ast.Expr, right: ast.Expr, variable: ast.Name, scope: Scope
    ) -> Iterator[None]:
        """While the operands of a convolution are lowered: u and δ applied at its variable
        in them, where nothing defines them, are the unit step and the Dirac delta."""
        saved = dict(self._signals)
        for call in [*convolutions.applied(left), *convolutions.applied(right)]:
            func = call.func
            assert isinstance(func, ast.Name)
            if func.id not in _SIGNALS or isinstance(call, Sample):
                continue
            if scope.lookup(func.id) is not None or self._has_a_meaning(func.id):
                continue
            if variable.id in convolutions.names_in(call.args[0]):
                self._signals[id(call)] = _SIGNALS[func.id]
        try:
            yield
        finally:
            self._signals = saved

    def _Call(self, node: ast.Call, scope: Scope, ctx: Context) -> Node:
        if (name := self._signals.get(id(node))) is not None:
            return self._signal(node, name, scope, ctx)
        return super()._Call(node, scope, ctx)

    def _signal(self, node: ast.Call, name: str, scope: Scope, ctx: Context) -> Node:
        """u(t) or \\delta(t) in a convolution, where nothing defines u or δ: the unit step or
        the Dirac delta, with a note."""
        what = integrations.find(name).brief.split(",")[0]
        written = node.func.span.text
        note = (
            f"{node.span.text} is read as {what}, {name}, as signal processing writes it; for "
            f"a function of your own, define {written}(t) := ... first"
        )
        self._note(note)
        why = f"nothing defines {written}, and it is applied in a convolution"
        self._decide(CONVOLUTION, node.span, f"{what}, {name}", why, note)
        return self._builtin(BUILTINS[name], [self.value(node.args[0], scope, ctx)], node.span)

    # -- circular convolution ---------------------------------------------------------------

    def _CircularConvolution(
        self, node: ast.CircularConvolution, scope: Scope, ctx: Context
    ) -> Node:
        """``x[n] \\circledast h[n]``: the circular convolution of two sequences at points of
        one index, as a sequence of it."""
        found = self._convolved(node.left, node.right, scope)
        if found is None or not found[1]:
            if all(self._named_sequence(side, scope) for side in (node.left, node.right)):
                raise CompileError(
                    f"{node.span.text} is the sequence of their circular convolution, which is "
                    "computed at a point",
                    span=node.span,
                    hint=f"write it at a point, as ({node.span.text})[n]",
                )
            raise CompileError(
                f"{node.span.text} is a circular convolution, which is of two sequences",
                span=node.span,
                hint="write the sequences at their points, as in x[n] \\circledast h[n]",
            )
        variable, _ = found
        label = self.graph.const(sympy.Dummy("m", integer=True), SCALAR, node.span)
        values = []
        for side in (node.left, node.right):
            inner = scope.child()
            inner.bind(variable.id, VariableBinding(label))
            values.append(self.value(side, inner, ctx))
        point = self.value(variable, scope, ctx)
        return self._circular(node, values, label, point, variable.span.text, scope, ctx)

    def _circular_at(self, node: ast.Binary, scope: Scope, ctx: Context) -> Node | None:
        """``(x \\circledast h)[0]``: the circular convolution of sequences the program
        defines, at the point in the square brackets right after it, times the rest of the
        product it is in. ``None`` for any other product."""
        factors = convolutions.factors(node)
        for position, (factor, following) in enumerate(itertools.pairwise(factors)):
            found = sequences.bracketed(following)
            if not isinstance(factor, ast.CircularConvolution) or found is None or found[1]:
                continue
            bindings = [self._named_sequence(side, scope) for side in (factor.left, factor.right)]
            if following.span.start != factor.span.end or None in bindings:
                continue
            label = self.graph.const(sympy.Dummy("m", integer=True), SCALAR, factor.span)
            values = [
                self._inline(binding, [label], factor.span, ctx)
                for binding in bindings
                if binding is not None
            ]
            point = self.value(found[0], scope, ctx)
            value = self._circular(factor, values, label, point, found[0].span.text, scope, ctx)
            rest = [*factors[:position], *factors[position + 2 :]]
            if not rest:
                return value
            return self._multiply(
                [value, self.value(convolutions.product_of(rest), scope, ctx)], node.span
            )
        return None

    def _named_sequence(self, side: ast.Expr, scope: Scope) -> FunctionBinding | None:
        """The binding of a sequence a program defines, named alone, as x is in
        ``x \\circledast h``; ``None`` for anything else."""
        if not isinstance(side, ast.Name):
            return None
        binding = scope.lookup(side.id)
        if not isinstance(binding, FunctionBinding):
            return None
        definition = binding.definition
        if isinstance(definition, Defined) or len(definition.params) == 1:
            return binding
        return None

    def _circular(
        self,
        node: ast.CircularConvolution,
        values: Sequence[Node],
        label: Node,
        point: Node,
        at: str,
        scope: Scope,
        ctx: Context,
    ) -> Node:
        """The circular convolution of two sequences, given as their values at ``label``, at
        ``point``, which is written ``at``: of the length the program writes, or else of that
        of the longer sequence, said in a note."""
        args = [*values, label, point]
        note = None
        if node.length is not None:
            args.append(self.value(node.length, scope, ctx))
            length = f"of length {node.length.span.text}"
        else:
            length = "of the length of the longer sequence"
            note = (
                f"{node.span.text} is the circular convolution of the length of the longer "
                "sequence, from 0 to its last member that is not 0; for another, write it as "
                f"the subscript, as in {node.left.span.text} \\circledast_N "
                f"{node.right.span.text}"
            )
            self._note(note)
        reading = f"their circular convolution {length}, at {at}"
        self._decide(CONVOLUTION, node.span, reading, "\\circledast is circular convolution", note)
        return self._call("seq.circular", args, SCALAR, node.span)
