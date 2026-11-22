r"""Functions named as books name them (``design/notation.md``, decisions 5 to 8), each
reading recorded with why, and a note where another reading was possible:

- a name with scripts, then its parameters in brackets, defines a function under that name,
  and a call of it calls the function: ``R_{nl}(r)``, ``\sigma^2(\mu)``,
  ``Y_2^1(\theta, \phi)``, ``C_\mathrm{A}(t)``, ``\bar{\mathbf{x}}(u, v)``. A definition
  of the program's own wins over a name books give a function, as Y_2^1 does over the
  spherical harmonic;
- ``f^{(n)}`` of a function of one variable the program defines is its nth derivative, as
  ``f^{(4)}(0)`` is at 0, and ``\dot{x}(0)``, of a function of time, its derivative at 0;
- a prime on the left of ``:=`` is part of the name: ``H'(x) := ...`` defines a function H',
  ``k'^\mu := ...`` a tensor k'; a primed name the program defines is that, and a prime on
  any other function of one variable is its derivative;
- a function named without its argument is the function at its parameters, as y is y(x)
  after ``y(x) := ...``, in ``\frac{dy}{dx}``, ``y''``, ``\dot a`` and products.
"""

from __future__ import annotations

import sympy

from relativisticpy.compiler.passes import Term
from relativisticpy.compiler.scope import (
    Context,
    FunctionBinding,
    FunctionReference,
    Scope,
    TensorBinding,
)
from relativisticpy.errors import CompileError
from relativisticpy.ir import SCALAR, VALUE, Node
from relativisticpy.language import syntax as ast
from relativisticpy.language.source import Span
from relativisticpy.reading.families import Families, Family
from relativisticpy.reading.labels import _base_name, label_name
from relativisticpy.reading.names import _bracketed_order, _shown, _with_arguments

#: The decision of whether a name before brackets is called or multiplies.
CALL = "product or call"

#: The decision of what a function named without its argument is.
ALONE = "function named alone"


def _function_name(func: ast.Expr) -> str | None:
    """The name a function written with scripts or an accent is defined under: ``R_nl`` of
    ``R_{nl}``, ``sigma^2`` of ``\\sigma^2``, ``Y^1_2`` of ``Y_2^1``, ``xbar`` of
    ``\\bar{\\mathbf{x}}``; ``None`` for what names no function so."""
    if isinstance(func, ast.Scripted):
        return label_name(func)
    if isinstance(func, ast.Accent) and func.accent not in ("dot", "ddot"):
        return _base_name(func)
    return None


def _an_operator_name(node: ast.Name) -> bool:
    """Whether a name is written as an operator's, as ``\\operatorname{sinc}`` is, which
    applies to what follows it as books write one."""
    return node.verbatim and node.span.text.lstrip().startswith(("\\operatorname", "\\mathrm"))


class Functions(Families):
    """Functions named with scripts, primes and accents, and named without arguments."""

    def _define_function(
        self, definition: ast.FunctionDefinition, scope: Scope, ctx: Context
    ) -> None:
        """Define a function, which the lowerer does (see
        :class:`~relativisticpy.lowering.definitions.Definitions`)."""
        raise NotImplementedError

    # -- definitions ------------------------------------------------------------------------

    def _define_scripted_function(
        self, target: ast.Expr, value: ast.Expr, scope: Scope, ctx: Context
    ) -> bool:
        """``R_{nl}(r) := ...``, ``\\sigma^2(\\mu) := ...`` or ``Y_2^1(\\theta, \\phi) :=
        ...``: a name with scripts, then its parameters in brackets, which the parser reads
        as a product, defines a function under that name (decisions 5 and 6); ``False`` for
        any other definition."""
        applied = _with_arguments(target)
        if applied is None:
            return False
        func, group = applied
        name = _function_name(func)
        if name is None:
            return False
        items = group.items if isinstance(group, ast.Tuple) else (group,)
        params = tuple(item.id for item in items if isinstance(item, ast.Name))
        shown = ", ".join(_shown(p, ctx.latex) for p in params)
        why = "a name with scripts, then names in brackets, is a function of them"
        reading = f"a function {func.span.text}({shown})"
        self._decide(CALL, target.span, reading, why, written=target.span.text)
        span = target.span.to(value.span)
        definition = ast.FunctionDefinition(name, params, value, span=span, latex=ctx.latex)
        self._define_function(definition, scope, ctx)
        return True

    # -- calls ------------------------------------------------------------------------------

    def _call_of_a_name(self, node: ast.Call, scope: Scope, ctx: Context) -> Node | None:
        """A call of what a name with scripts or an accent names: a member of a family of
        functions, as ``\\psi_3(x)``, a function the program defines under that name, as
        ``R_{nl}(r)``, a derivative, as ``f^{(4)}(0)`` and ``\\dot{x}(0)``; ``None`` for any
        other call, which is read as before."""
        if (member := self._member_call(node, scope, ctx)) is not None:
            return member
        func = node.func
        if isinstance(func, ast.Accent) and func.accent in ("dot", "ddot"):
            return self._dotted_call(node, func, scope, ctx)
        if isinstance(func, ast.Scripted) and (order := self._order_of(func, scope)) is not None:
            differentiated, number = order
            (point,) = self._arguments(node, differentiated, scope, ctx)
            why = f"{func.base.span.text} is a function of one variable, defined on line "
            why += str(differentiated.definition.span.line)
            self._decide(CALL, node.span, f"its derivative of order {number}", why)
            return self._derivative_at(differentiated, number, point, node.span, ctx)
        name = _function_name(func) if not isinstance(func, ast.Name) else None
        binding = scope.lookup(name) if name is not None else None
        if not isinstance(binding, FunctionBinding):
            return None
        why = f"{func.span.text} is a function, defined on line {binding.definition.span.line}"
        self._decide(CALL, node.span, f"a call of {func.span.text}", why)
        return self._inline(binding, self._arguments(node, binding, scope, ctx), node.span, ctx)

    def _arguments(
        self, node: ast.Call, binding: FunctionBinding, scope: Scope, ctx: Context
    ) -> list[Node]:
        """The values a call gives a function, as many as it has parameters."""
        args = [self.value(arg, scope, ctx) for arg in node.args]
        count = len(binding.definition.params)
        if len(args) != count and not isinstance(binding.definition, Family):
            written = node.func.span.text
            raise CompileError(
                f"{written} takes {count} argument{'s' * (count != 1)}, not {len(args)}",
                span=node.span,
            )
        return args

    def _dotted_call(
        self, node: ast.Call, func: ast.Accent, scope: Scope, ctx: Context
    ) -> Node | None:
        """``\\dot{x}(0)``: the derivative of a function of time the program defines, at 0,
        as books write the initial velocity; ``None`` where the dotted name is no such
        function."""
        base = func.base
        binding = scope.lookup(base.id) if isinstance(base, ast.Name) else None
        if not isinstance(binding, FunctionBinding) or len(binding.definition.params) != 1:
            return None
        if isinstance(binding.definition, Family) or len(node.args) != 1:
            return None
        order = 1 if func.accent == "dot" else 2
        why = f"{base.span.text} is a function of one variable, defined on line "
        why += str(binding.definition.span.line)
        self._decide(CALL, node.span, f"the derivative of {base.span.text} at its argument", why)
        point = self.value(node.args[0], scope, ctx)
        return self._derivative_at(binding, order, point, node.span, ctx)

    def _order_of(self, node: ast.Scripted, scope: Scope) -> tuple[FunctionBinding, int] | None:
        """The function and the order of ``f^{(4)}``, where f is a function of one variable
        the program defines (decision 6.2 of design/notation.md); ``None`` otherwise."""
        base = node.base
        if not isinstance(base, ast.Name) or len(node.scripts) != 1:
            return None
        script = node.scripts[0]
        body = script.body
        if script.kind != "^" or not _bracketed_order(body) or not isinstance(body, ast.Number):
            return None
        binding = scope.lookup(base.id)
        if not isinstance(binding, FunctionBinding) or isinstance(binding.definition, Family):
            return None
        if len(binding.definition.params) != 1:
            return None
        return binding, int(body.text)

    def _derivative_at(
        self, binding: FunctionBinding, order: int, point: Node, span: Span, ctx: Context
    ) -> Node:
        """A function's derivative of ``order`` at ``point``, as ``f'(x)`` is taken."""
        dummy = self.graph.const(sympy.Dummy(binding.definition.name), SCALAR)
        body = self._inline(binding, [dummy], span, ctx)
        number = self.graph.const(order, VALUE)
        return self._call("core.derivative_at", [body, dummy, number, point], SCALAR, span)

    def _names_a_function(self, node: ast.Expr, scope: Scope) -> bool:
        """Whether what a bracket follows is called: a name with scripts the program defines
        as a function, or a family of them (see :meth:`Families._names_a_function`), an
        order raised on a function of one variable, as ``f^{(4)}``, or a dot on one, as
        ``\\dot{x}``."""
        if super()._names_a_function(node, scope):
            return True
        if isinstance(node, ast.Scripted):
            return self._order_of(node, scope) is not None
        if not isinstance(node, ast.Accent):
            return False
        if node.accent in ("dot", "ddot"):
            base = node.base
            binding = scope.lookup(base.id) if isinstance(base, ast.Name) else None
            return isinstance(binding, FunctionBinding) and len(binding.definition.params) == 1
        name = _function_name(node)
        return name is not None and isinstance(scope.lookup(name), FunctionBinding)

    # -- names written without their arguments ----------------------------------------------

    def _Name(self, node: ast.Name, scope: Scope, ctx: Context) -> Term:
        """In LaTeX, a function the program defines named without its argument is its value
        at its parameters, even before a factor, as y in ``x y``; in plain text it applies
        to what follows it, as ``f 3`` is f(3), and is its value where nothing follows (see
        :meth:`_at_parameters`)."""
        binding = scope.lookup(node.id)
        function = isinstance(binding, FunctionBinding) and not isinstance(
            binding.definition, Family
        )
        if function and ctx.latex and not _an_operator_name(node):
            assert isinstance(binding, FunctionBinding)
            return self._function_alone(node.span.text, node, binding, scope, ctx)
        return super()._Name(node, scope, ctx)

    def _at_parameters(
        self, reference: FunctionReference, scope: Scope, ctx: Context
    ) -> Node | None:
        """A function the program defines, named with nothing to act on, as f in ``f + 1``:
        its value at its parameters; ``None`` for a builtin's name, said as before."""
        binding = scope.lookup(reference.name)
        if not isinstance(binding, FunctionBinding) or isinstance(binding.definition, Family):
            return None
        name = ast.Name(reference.name, span=reference.span)
        return self._function_alone(reference.span.text, name, binding, scope, ctx)

    def _function_alone(
        self,
        written: str,
        node: ast.Expr,
        binding: FunctionBinding,
        scope: Scope,
        ctx: Context,
    ) -> Node:
        """A function the program defines, named without its arguments, as y after
        ``y(x) := ...``: the function at its parameters, y(x) (``fn-bare`` of
        curriculum/ROADMAP.md). Before, it was refused, needing an argument."""
        definition = binding.definition
        params = definition.params
        shown = ", ".join(_shown(p, ctx.latex) for p in params)
        self._check_parameters_taken(written, node, params, scope, ctx)
        why = f"{written} is a function, defined on line {definition.span.line}"
        self._decide(ALONE, node.span, f"the function {written}({shown})", why)
        arguments = [self._parameter_here(param, node, scope, ctx) for param in params]
        return self._inline(binding, arguments, node.span, ctx)

    def _parameter_here(self, param: str, node: ast.Expr, scope: Scope, ctx: Context) -> Node:
        """What a parameter of a function named alone is where it is named: the value its
        name has there, as in the value of a function that takes it, or the symbol."""
        if scope.lookup(param) is not None:
            return self.value(ast.Name(param, span=node.span), scope, ctx)
        return self._symbol_of(param)

    def _check_parameters_taken(
        self,
        written: str,
        node: ast.Expr,
        params: tuple[str, ...],
        scope: Scope,
        ctx: Context,
    ) -> None:
        """Refuse a function named without its arguments in the value of another function
        that takes none of its parameters, as U is in ``F(T, V) := U - T S`` after U(S, V):
        read as U(S, V), F would depend on S, which it does not take, where the book means
        S eliminated, as a Legendre transform does."""
        if not ctx.calls:
            return
        free = [p for p in params if scope.lookup(p) is None]
        if not free:
            return
        within = _shown(ctx.calls[-1], ctx.latex)
        shown = ", ".join(_shown(p, ctx.latex) for p in params)
        missing = ", ".join(_shown(p, ctx.latex) for p in free)
        raise CompileError(
            f"{written} is named without its arguments in the value of {within}, so it would "
            f"be {written}({shown}), and {within} takes no {missing}",
            span=node.span,
            hint=f"write {written} with the arguments meant, as in {written}({shown})",
        )

    def _derivative_alone(
        self, node: ast.Prime, binding: FunctionBinding, scope: Scope, ctx: Context
    ) -> Node:
        """``y''`` of a function of one variable the program defines, written without its
        argument: the derivative at its parameter, y''(x)."""
        (param,) = binding.definition.params
        written = node.span.text
        self._check_parameters_taken(node.func.span.text, node, (param,), scope, ctx)
        why = f"{node.func.span.text} is a function of {param}, defined on line "
        why += str(binding.definition.span.line)
        self._decide(ALONE, node.span, f"{written}({_shown(param, ctx.latex)})", why)
        point = self._parameter_here(param, node, scope, ctx)
        return self._derivative_at(binding, node.order, point, node.span, ctx)

    def _labelled(self, node: ast.Scripted, scope: Scope, ctx: Context) -> Term | None:
        """``f^{(4)}`` of a function of one variable, named without its argument: its
        derivative at its parameter; otherwise as :meth:`Families._labelled` reads it."""
        if (order := self._order_of(node, scope)) is not None:
            binding, number = order
            (param,) = binding.definition.params
            why = f"{node.base.span.text} is a function of {param}, defined on line "
            why += str(binding.definition.span.line)
            self._decide(ALONE, node.span, f"its derivative of order {number} at {param}", why)
            return self._derivative_at(binding, number, self._symbol_of(param), node.span, ctx)
        return super()._labelled(node, scope, ctx)

    def _a_primed_tensor(self, prime: ast.Prime, scope: Scope) -> bool:
        """Whether a primed name is a tensor the program defines, as k' after ``k'^\\mu :=
        ...``: its scripts are then its indices, raised or lowered."""
        return isinstance(scope.lookup(prime.func.id + "'" * prime.order), TensorBinding)
