r"""The lowering of random variables, as :mod:`relativisticpy.reading.probability` reads
them (design/notation.md, 14): ``X \sim \mathcal{N}(\mu, \sigma^2)`` declares one, for the
rest of the program, as an assumption does; ``\Pr(A) := 0.3`` gives the probability of an
event; ``\Pr(X > 1)``, ``P(X = 2)``, ``\mathbb{E}[X \mid N]``, ``\operatorname{Var}(aX + b)``
and ``\operatorname{Cov}(X, Y)`` are computed of them, and so are ``f_X(x)``, ``F_X(x)``,
``F_X^{-1}(q)``, ``M_X(t)`` and ``\varphi_X(t)``. Inside a probability, an expectation, a
variance or a covariance, a capital letter nothing defines is a random variable whose law
is not given, as probability writes one, so that ``\operatorname{Var}(aX + b) = a^2
\operatorname{Var}(X)`` holds whatever X is.
"""

from __future__ import annotations

from collections.abc import Sequence

import sympy
from sympy import Symbol

from relativisticpy.compiler.builtins import BUILTINS, CONSTANTS
from relativisticpy.compiler.scope import (
    Binding,
    Context,
    Scope,
    TensorBinding,
    VariableBinding,
)
from relativisticpy.errors import CompileError
from relativisticpy.integrations import _distributions as distributions
from relativisticpy.integrations import _forms as forms
from relativisticpy.ir import SCALAR, VALUE, Node, TensorType, ops
from relativisticpy.language import syntax as ast
from relativisticpy.language.source import Span
from relativisticpy.lowering.convolutions import Convolutions
from relativisticpy.reading import probability
from relativisticpy.reading.labels import label_name
from relativisticpy.reading.probability import DISTRIBUTED, RANDOM, Declaration, Law
from relativisticpy.reading.quantum import expectation_name
from relativisticpy.reading.subscripts import SAMPLE as _SAMPLE
from relativisticpy.reading.subscripts import Reading
from relativisticpy.typing.syntax import _walk

#: The operation of the IR that computes each operator of random variables.
_COMPUTED = {
    "probability": "prob.probability",
    "expectation": "prob.expectation",
    "variance": "prob.variance",
    "covariance": "prob.covariance",
    "density": "prob.density",
    "distribution function": "prob.cdf",
    "quantile": "prob.quantile",
    "moment-generating function": "prob.mgf",
    "characteristic function": "prob.cf",
}
#: How each operator is written, as a message says it.
_WRITTEN = {
    "probability": r"\Pr(X > 1)",
    "expectation": r"\mathbb{E}[X]",
    "variance": r"\operatorname{Var}(X)",
    "covariance": r"\operatorname{Cov}(X, Y)",
}
#: Where a program says which random variables are independent, in its session's
#: conventions: the names declared together as such.
_INDEPENDENT = "independent random variables"


class Probability(Convolutions):
    """The lowering of random variables."""

    # -- declarations -------------------------------------------------------------------

    def _declares_random(self, statement: ast.Statement, scope: Scope, ctx: Context) -> bool:
        """``X \\sim D`` declares random variables, and ``\\Pr(A) := 0.3`` gives the
        probability of an event: each shows nothing, and a note says what it declares.
        ``False`` for any other statement, which is read as before."""
        if isinstance(statement, ast.Evaluate):
            declared = probability.declaration(statement.expr)
            return declared is not None and self._declare_variables(declared, scope, ctx)
        if isinstance(statement, ast.FunctionDefinition | ast.Define):
            self._refuse_defining_a_random_variable(statement, scope)
            return self._gives_probability(statement, scope, ctx)
        return False

    def _refuse_defining_a_random_variable(self, statement: ast.Statement, scope: Scope) -> None:
        """Refuse a definition of a name the program declares a random variable, as
        ``X := 5`` after ``X \\sim \\mathcal{N}(0, 1)``: what follows would be of the number,
        ``P(X < 1)`` of no random variable, as a declaration of a name defined is refused."""
        if isinstance(statement, ast.FunctionDefinition):
            names = [statement.name]
        else:
            assert isinstance(statement, ast.Define)
            names = [name for name in map(_name, statement.targets) if name is not None]
        for name in names:
            binding = scope.lookup(name)
            if not isinstance(binding, VariableBinding) or not _random(binding):
                continue
            declared = binding.node.span
            where = f" on line {declared.line}" if declared is not None else ""
            raise CompileError(
                f"{name} is declared a random variable{where}, so "
                f"{' '.join(statement.span.text.split())} cannot define it",
                span=statement.span,
                hint="define a name of its own, or declare the variable again with another law",
            )

    def _declare_variables(self, declared: Declaration, scope: Scope, ctx: Context) -> bool:
        """Bind each variable ``declared`` to a random variable of its law; ``False`` where
        its left side is no list of names, as ``\\Lambda \\sim m_P^4`` is not."""
        if (size := _sized(declared)) is not None:  # X_1, \ldots, X_n iid ~ D
            return self._declare_sample(declared, size, scope, ctx)
        names = self._declared_names(declared)
        if names is None:
            return False
        for name, node in names:
            binding = scope.lookup(name)
            if binding is not None and not _random(binding):
                raise CompileError(
                    f"{node.span.text} is defined, so {declared.span.text} cannot declare it a "
                    "random variable",
                    span=declared.span,
                    hint="declare a random variable with a name of its own",
                )
        law = self._law(declared.law, scope, ctx)
        pieces = self._pieces(declared, law, scope, ctx)
        for name, node in names:
            symbol = self.graph.const(Symbol(name), VALUE, node.span)
            previous = self._previous_pieces(name, declared, scope)
            variable = self._call(
                "prob.variable", [symbol, *previous, *pieces], SCALAR, declared.span
            )
            scope.bind(name, VariableBinding(variable))
            self._resolve(node.span, f"a random variable, {_shown(declared)}")
        if declared.independent and len(names) > 1:
            together = frozenset(name for name, _ in names)
            known = self.session.conventions.get(_INDEPENDENT, frozenset())
            self.session.conventions[_INDEPENDENT] = known | {together}
        listed = ", ".join(_unbraced(item) for item in declared.variables)  # X_1, \ldots, X_4
        what = "a random variable" if len(names) == 1 else "random variables"
        note = f"{_shown(declared)} declares {listed} {what}, of {_described(declared.law)}"
        if declared.given is not None:
            note += f", given {declared.given.span.text}"
        if declared.independent and len(names) > 1:
            note += ", independent"
        why = "\\sim between names nothing defines and a distribution declares them, as books do"
        self._decide(DISTRIBUTED, declared.span, what, why, note, written=_shown(declared))
        self._note(note)
        return True

    def _declare_sample(
        self,
        declared: Declaration,
        size: tuple[str, ast.Expr, ast.Expr],
        scope: Scope,
        ctx: Context,
    ) -> bool:
        """``X_1, \\ldots, X_n \\overset{\\text{iid}}{\\sim} D``, of a number n of them that
        is a symbol: a sample, whose members ``X_i``, ``X_{i}`` and ``X_1`` are, each of its own
        subscript, a random variable of the law, independent of the others."""
        base, first, last = size
        if not declared.independent:
            raise CompileError(
                f"{declared.span.text} declares {_shown_subscript(last)} random variables, a "
                "number of them that is a symbol, without saying how they depend on each other",
                span=declared.span,
                hint="say they are independent, with \\overset{\\text{iid}}{\\sim}",
            )
        if declared.given is not None:
            raise CompileError(
                f"{declared.span.text}: a sample of laws given another is not supported",
                span=declared.span,
            )
        law = self._law(declared.law, scope, ctx)
        ends = [self.value(_subscript(end), scope, ctx) for end in (first, last)]
        one = self.graph.const(sympy.S.One, SCALAR)
        count = self.graph.add(ops.Add((1, -1, 1)), (ends[1], ends[0], one), SCALAR, last.span)
        symbol = self.graph.const(Symbol(base), VALUE, first.span)
        found = self._call("prob.sample", [symbol, law, count, ends[0]], VALUE, declared.span)
        scope.bind(_SAMPLE.format(base), VariableBinding(found))
        many = _shown_subscript(last)
        note = (
            f"{_shown(declared)} declares a sample of {many} independent random variables, of "
            f"{_described(declared.law)}: {base}_i is the member at i"
        )
        why = "an ellipsis between two members of a family, iid, declares a sample"
        self._decide(DISTRIBUTED, declared.span, "a sample", why, note, written=_shown(declared))
        self._note(note)
        return True

    def _sample_member(self, reading: Reading, scope: Scope, ctx: Context) -> Node:
        """``X_i``, ``X_{i}`` or ``X_1``, read by the resolver as a member of a sample the
        program declares: its member at the subscript."""
        written, binding = reading.written, reading.binding
        assert written is not None
        assert isinstance(binding, VariableBinding)
        assert written.subscript is not None
        node = written.node
        index = self.value(written.subscript, scope, ctx)
        declared = binding.node.span
        where = f", declared on line {declared.line}" if declared is not None else ""
        self._resolve(node.span, f"a member of the sample {written.base}{where}")
        return self._call("prob.member", [binding.node, index], SCALAR, node.span)

    def _declared_names(self, declared: Declaration) -> list[tuple[str, ast.Expr]] | None:
        """The names a declaration declares, with how each is written: those listed, and those
        an ellipsis stands for between two of a family, as ``X_1, \\ldots, X_{10}``."""
        found: list[tuple[str, ast.Expr]] = []
        items = declared.variables
        for position, item in enumerate(items):
            if isinstance(item, ast.Dots):
                if not 0 < position < len(items) - 1:
                    return None
                between = self._between(items[position - 1], items[position + 1], declared)
                found.extend(between)
                continue
            name = _name(item)
            if name is None:
                if isinstance(item, ast.Binary) and isinstance(item.right, ast.List):
                    raise CompileError(
                        f"{declared.span.text} declares a random process, a random variable "
                        f"for each {item.right.span.text.strip('[]')}, which is not supported",
                        span=declared.span,
                        hint="declare each random variable by name, as X_1, X_2 \\sim ...",
                    )
                return None
            found.append((name, item))
        return found

    def _between(
        self, first: ast.Expr, last: ast.Expr, declared: Declaration
    ) -> list[tuple[str, ast.Expr]]:
        """The names an ellipsis stands for, from the one after ``first`` to the one before
        ``last``, of one family: ``X_2``, ..., ``X_9`` of ``X_1, \\ldots, X_{10}``."""
        written = [probability.capital(each) or _name(each) for each in (first, last)]
        parts = [name.partition("_") if name else ("", "", "") for name in written]
        (base, _, low), (other, _, high) = parts
        if base and base == other and low.isdigit() and high.isdigit() and int(low) < int(high):
            return [(f"{base}_{index}", first) for index in range(int(low) + 1, int(high))]
        raise CompileError(
            f"{declared.span.text} declares random variables up to {last.span.text}, a number "
            "of them that is not given, which is not supported",
            span=declared.span,
            hint=f"give the last, as {first.span.text}, \\ldots, {base or 'X'}_{{10}}",
        )

    def _law(self, written: Law, scope: Scope, ctx: Context) -> Node:
        """The law a declaration writes: its distribution of the parameters as written, each
        the root of what is written where the form writes a power of it, as the standard
        deviation is of the variance."""
        parameters: list[Node] = []
        for argument, power in written.arguments:
            if power == 1:
                parameters.append(self.value(argument, scope, ctx))
                continue
            base = _power_of(argument, power)
            if base is not None and not self._a_name_of_its_own(argument, scope):
                parameters.append(self._root_written(base, power, scope, ctx))
                continue  # \sigma^2 of \mathcal{N}(\mu, \sigma^2)
            value = self.value(argument, scope, ctx)
            root = self.graph.const(sympy.Rational(1, power), SCALAR)
            parameters.append(self.graph.add(ops.Pow(), (value, root), SCALAR, argument.span))
        return self._call(written.capability.name, parameters, VALUE, written.span)

    def _a_name_of_its_own(self, argument: ast.Expr, scope: Scope) -> bool:
        """Whether an argument written as a power is a name the program defines as it is
        written, as ``\\sigma^2 := 4`` defines σ²: its value is then the variance, and σ
        its root, not a symbol σ."""
        if not isinstance(argument, ast.Scripted):
            return False
        name = label_name(argument)
        return name is not None and scope.lookup(name) is not None

    def _root_written(self, base: ast.Expr, power: int, scope: Scope, ctx: Context) -> Node:
        """The parameter an argument written as its power gives, the base written: σ of
        σ², and of an even power of what is known to be negative, as (-2)², what it is the
        power of that is positive, 2, as a standard deviation is."""
        node = self.value(base, scope, ctx)
        known = self._folded(node)
        if power % 2 == 0 and known is not None and known.is_negative:
            return self.graph.add(ops.Add((-1,)), (node,), SCALAR, base.span)
        return node

    def _pieces(self, declared: Declaration, law: Node, scope: Scope, ctx: Context) -> list[Node]:
        """The law of a declaration, under the condition it is given, if any: of ``X \\mid N``,
        a law whose parameters are random; of ``X \\mid Y = 0``, the condition Y = 0."""
        given = declared.given
        if given is None:
            return [law]
        if isinstance(given, ast.Equation):
            parent = scope.lookup(_name(given.lhs) or "")
            if parent is None or not _random(parent):
                raise self._not_random(given.lhs, declared)
            sides = [self.value(side, scope, ctx) for side in (given.lhs, given.rhs)]
            condition = self._call("core.eq", sides, SCALAR, given.span)
            return [condition, law]
        for item in given.items if isinstance(given, ast.Tuple) else (given,):
            parent = scope.lookup(_name(item) or "")
            if parent is None or not _random(parent):
                raise self._not_random(item, declared)
        return [law]

    def _previous_pieces(self, name: str, declared: Declaration, scope: Scope) -> list[Node]:
        """The laws a variable is given under other conditions already, as X's under Y = 0
        where ``X \\mid Y = 1 \\sim ...`` follows ``X \\mid Y = 0 \\sim ...``."""
        if not isinstance(declared.given, ast.Equation):
            return []
        binding = scope.lookup(name)
        if not isinstance(binding, VariableBinding) or not _random(binding):
            return []
        earlier = list(binding.node.args[1:])
        return earlier if len(earlier) % 2 == 0 else []

    def _not_random(self, given: ast.Expr, declared: Declaration) -> CompileError:
        return CompileError(
            f"{declared.span.text} gives a law given {given.span.text}, which is no random "
            "variable",
            span=declared.span,
            hint=f"declare it first, as {given.span.text} "
            "\\sim \\operatorname{Poisson}(\\lambda)",
        )

    def _gives_probability(self, statement: ast.Statement, scope: Scope, ctx: Context) -> bool:
        """``\\Pr(B_1) := 0.0001`` or ``\\Pr(A \\mid B_1) := 0.9``: the probability of an
        event, given another, for the rest of the program. ``P(A) := 0.3`` is so too where A
        is written as an event is, with a capital, and the value does not depend on it."""
        if isinstance(statement, ast.FunctionDefinition):
            if len(statement.params) != 1 or isinstance(statement.body, ast.Block):
                return False
            head, value = statement.name, statement.body
            event: ast.Expr = ast.Name(statement.params[0], span=statement.span)
            given: ast.Expr | None = None
            shown = statement.params[0]
            if head == "P" and (
                probability.capital(event) is None or statement.params[0] in _names_in(value)
            ):
                return False
        else:
            assert isinstance(statement, ast.Define)
            target = statement.targets[0]
            if len(statement.targets) != 1 or not isinstance(target, ast.Call):
                return False
            if not isinstance(target.func, ast.Name) or len(target.args) != 1:
                return False
            head, value = target.func.id, statement.value
            event, given = probability.conditioned(target.args[0])
            if head == "P" and given is None:
                return False
            shown = _unbraced(event)
        if probability.OPERATORS.get(head, probability.LETTERS.get(head)) != "probability":
            return False
        if scope.lookup(head) is not None:
            return False
        declared = self._random_names([event, given], scope)
        if declared:  # its law gives the probability: Pr(0 < X) would not be the one given
            names = ", ".join(name.replace("_…", "") for name in declared)
            raise CompileError(
                f"{statement.span.text} gives the probability of an event of {names}, "
                "declared with a law, which gives its probability",
                span=statement.span,
                hint="give the probability of an event with a name of its own, as "
                "\\Pr(A) := 0.3, or declare the variable with the law that gives it",
            )
        node = self.value(value, scope, ctx)
        if isinstance(node.op, ops.Const) and _outside(node.op.value):
            raise CompileError(
                f"{statement.span.text} gives a probability outside 0 to 1",
                span=statement.span,
            )
        key = probability.event_key(event, given)
        scope.bind(key, VariableBinding(node))
        about = f"the event {shown}"
        if given is not None:
            about += f", given {_unbraced(given)}"
        note = f"{statement.span.text} gives the probability of {about}"
        why = "it gives the value of a probability of events, as a book does"
        self._decide(RANDOM, statement.span, "the probability of an event", why, note)
        self._note(note)
        return True

    # -- probabilities, expectations and the rest --------------------------------------------

    def _Call(self, node: ast.Call, scope: Scope, ctx: Context) -> Node:
        func = node.func
        if isinstance(func, ast.Name) and scope.lookup(func.id) is None:
            kind = probability.operator(func)
            if kind is not None:
                square = probability.squared(node)
                found = self._random_operator(
                    kind, func, node.args, node.span, scope, ctx, square=square
                )
                if found is not None:
                    return found
            function = probability.function_of(func)
            if function is not None and len(node.args) == 1:
                found = self._function_of(function, func, node.args[0], node.span, scope, ctx)
                if found is not None:
                    return found
            law = probability.law(node)
            if law is not None and func.id in law.capability.names:
                raise CompileError(
                    f"{node.span.text} is a distribution: declare a random variable with it, as "
                    f"X \\sim {node.span.text}, or X ∼ {node.span.text} in plain text, where ~ is "
                    "a space, as in LaTeX",
                    span=node.span,
                )
        return super()._Call(node, scope, ctx)

    def _statistic_factors(
        self, factors: list[ast.Expr], scope: Scope, ctx: Context
    ) -> list[ast.Expr | Node]:
        """The factors of a product, each operator of random variables written among them
        with what it acts on, as ``\\mathbb{E}[X]``, ``\\operatorname{Var}[X]`` or
        ``\\varphi_X(t)``, compiled to one node."""
        grouped = super()._statistic_factors(factors, scope, ctx)
        found: list[ast.Expr | Node] = []
        position = 0
        while position < len(grouped):
            factor = grouped[position]
            after = grouped[position + 1] if position + 1 < len(grouped) else None
            if isinstance(factor, ast.Expr) and isinstance(after, ast.Expr):
                node = self._random_factor(factor, after, scope, ctx)
                if node is not None:
                    found.append(node)
                    position += 2
                    continue
            found.append(factor)
            position += 1
        return found

    def _random_factor(
        self, head: ast.Expr, operand: ast.Expr, scope: Scope, ctx: Context
    ) -> Node | None:
        """An operator of random variables and the factor after it, as ``\\mathbb{E}[X]``,
        ``\\mathbb{E}[X]^2`` or ``\\mathbb{E}X``; a function of a random variable at the
        bracket after it, as ``F_X^{-1}(0.975)``; ``None`` for anything else."""
        span = head.span.to(operand.span)
        if isinstance(head, ast.Name) and scope.lookup(head.id) is None:
            kind = probability.operator(head)
            if kind is None:
                return None
            power: ast.Expr | None = None
            inner = operand
            if (
                isinstance(operand, ast.Scripted)
                and len(operand.scripts) == 1
                and operand.scripts[0].kind == "^"
                and _bracket(operand.base) is not None
            ):
                inner, power = operand.base, operand.scripts[0].body  # \mathbb{E}[X]^2
            arguments = _bracket(inner)
            if arguments is None:
                if kind != "expectation" or head.id == "E" or not _a_name(inner):
                    return None
                arguments = (inner,)  # \mathbb{E} X
            square = isinstance(inner, ast.List)
            found = self._random_operator(kind, head, arguments, span, scope, ctx, square=square)
            if found is None or power is None:
                return found
            return self._power(found, self.value(power, scope, ctx), span)
        function = probability.function_of(head)
        arguments = _bracket(operand)
        if function is None or arguments is None or len(arguments) != 1:
            return None
        return self._function_of(function, head, arguments[0], span, scope, ctx)

    def _random_operator(
        self,
        kind: str,
        head: ast.Name,
        arguments: Sequence[ast.Expr],
        span: Span,
        scope: Scope,
        ctx: Context,
        *,
        square: bool = False,
    ) -> Node | None:
        """A probability, an expectation, a variance or a covariance of what ``arguments``
        write, given what follows the conditional bar; ``None`` where a letter, P or E, acts
        on no random variable, and is read as before. ``square`` says it is written with
        square brackets."""
        letter = probability.lettered(head, square=square)
        written = head.span.text
        if not arguments:
            return None
        *rest, last = arguments
        operand, given = probability.conditioned(last)
        operands = [*rest, operand]
        if kind == "probability" and (named := self._named_event(operands, given, scope)):
            return named
        random = self._random_names([*operands, *((given,) if given is not None else ())], scope)
        if kind == "probability":
            for event in (*operands, given):
                if event is None or not _a_name(event) or _name(event) in random:
                    continue
                if letter:
                    return None
                raise CompileError(
                    f"{span.text} is not given: {event.span.text} is an event whose probability "
                    "the program does not give",
                    span=span,
                    hint=f"give it, as {written}({event.span.text}) := 0.3",
                )
        if letter and not random and given is None:
            return None
        expected = 2 if kind == "covariance" else 1
        if kind == "probability" and len(operands) > 1:  # \Pr(X > 1, Y < 2): both
            operands = [_together(operands)]
        if len(operands) != expected:
            if letter:
                return None
            raise CompileError(
                f"{written} takes {'two expressions' if expected == 2 else 'one expression'}, "
                f"but {len(operands)} {'is' if len(operands) == 1 else 'are'} given",
                span=span,
                hint=f"for example {_WRITTEN[kind]}",
            )
        inner = scope if letter else self._with_unknowns([*operands, given], scope, kind)
        nodes = [self._operand(kind, each, inner, ctx, written) for each in operands]
        givens = self._given(given, inner, ctx) if given is not None else []
        self._note_independence(random, scope)
        if letter:
            note = f"{span.text} is read as {_article(kind)}, as {written} acts on random variables"
            why = f"{written} acts on random variables"
            self._decide(RANDOM, span, f"{_article(kind)}", why, note)
            self._note(note)
        else:
            why = f"{written} is {_article(kind)}"
            self._decide(RANDOM, span, _article(kind), why)
        return self._call(_COMPUTED[kind], [*nodes, *givens], SCALAR, span)

    def _named_event(
        self, operands: Sequence[ast.Expr], given: ast.Expr | None, scope: Scope
    ) -> Node | None:
        """The probability a program gives an event, as ``\\Pr(A \\mid B_1)``'s by
        ``\\Pr(A \\mid B_1) := 0.9``; ``None`` where it gives none."""
        if len(operands) != 1:
            return None
        key = probability.event_key(operands[0], given)
        binding = scope.lookup(key)
        if isinstance(binding, VariableBinding):
            self._resolve(operands[0].span, "the probability the program gives the event")
            return binding.node
        return None

    def _operand(self, kind: str, expr: ast.Expr, scope: Scope, ctx: Context, written: str) -> Node:
        """What an operator acts on: an event, of a probability; a value of random variables,
        of the rest. Data over an index is refused, as its statistics are written with it."""
        if kind == "probability":
            return self._event(expr, scope, ctx)
        node = self.value(expr, scope, ctx)
        if isinstance(node.type, TensorType):
            label = node.type.indices.names[0] if node.type.indices else "t"
            raise CompileError(
                f"{written} of data is over an index, written with it",
                span=expr.span,
                hint=f"write the statistic over its index, as {written}_{label}[...]",
            )
        return node

    def _event(self, expr: ast.Expr, scope: Scope, ctx: Context) -> Node:
        """An event: a relation of random variables, ``X = 2``, ``X > 1`` or
        ``|X - \\mu| \\ge 2``, or several, joined by and and or."""
        if isinstance(expr, ast.Equation):
            sides = [self.value(side, scope, ctx) for side in (expr.lhs, expr.rhs)]
            return self._call("core.eq", sides, SCALAR, expr.span)
        if isinstance(expr, ast.Binary) and expr.op in ("and", "or"):
            parts = [self._event(side, scope, ctx) for side in (expr.left, expr.right)]
            return self._call(f"core.{expr.op}", parts, SCALAR, expr.span)
        return self.value(expr, scope, ctx)

    def _given(self, given: ast.Expr, scope: Scope, ctx: Context) -> list[Node]:
        """What follows the conditional bar: random variables, of which what is computed is a
        function; one taking a value, ``Y = 0``; or an event."""
        items = given.items if isinstance(given, ast.Tuple) else (given,)
        return [
            self._event(item, scope, ctx) if _an_event(item) else self.value(item, scope, ctx)
            for item in items
        ]

    def _Expectation(self, node: ast.Expectation, scope: Scope, ctx: Context) -> Node:
        """``\\langle X \\rangle`` of a random variable the program declares, where it defines
        no bracket of its own: X's expectation, as physics writes a mean; any other angle
        brackets are Dirac's or the program's own, as before."""
        operand = node.operand
        if (
            not isinstance(operand, ast.Tuple)
            and scope.lookup(expectation_name(operand)) is None
            and self._random_names([operand], scope)
        ):
            value = self._operand("expectation", operand, scope, ctx, node.span.text)
            note = f"{node.span.text} is read as the expectation of {operand.span.text}, a mean"
            why = f"{operand.span.text} is of random variables the program declares"
            self._decide(RANDOM, node.span, "an expectation", why, note)
            self._note(note)
            return self._call("prob.expectation", [value], SCALAR, node.span)
        return super()._Expectation(node, scope, ctx)

    def _function_of(
        self,
        function: tuple[str, ast.Expr],
        head: ast.Expr,
        argument: ast.Expr,
        span: Span,
        scope: Scope,
        ctx: Context,
    ) -> Node | None:
        """``f_X(x)``, ``F_X(x)``, ``F_X^{-1}(q)``, ``M_X(t)`` or ``\\varphi_X(t)``, of a random
        variable X the program declares; ``None`` where X is none, as of ``f_0(x)``."""
        kind, written = function
        name = _name(written)
        binding = scope.lookup(name) if name is not None else None
        if binding is None or not _random(binding):
            return None
        variable = self.value(written, scope, ctx)
        at = self.value(argument, scope, ctx)
        why = f"{written.span.text} is a random variable the program declares"
        reading = f"the {kind} of {written.span.text}"
        self._decide(RANDOM, span, reading, why, f"{span.text} is {reading}")
        self._note(f"{span.text} is {reading}")
        return self._call(_COMPUTED[kind], [variable, at], SCALAR, span)

    # -- what names mean inside an operator ---------------------------------------------------

    def _random_names(self, exprs: Sequence[ast.Expr | None], scope: Scope) -> list[str]:
        """The random variables the program declares that ``exprs`` name, in order."""
        found: list[str] = []
        for expr in exprs:
            if expr is None:
                continue
            for name in _names_in(expr):
                binding = scope.lookup(name)
                if binding is not None and _random(binding) and name not in found:
                    found.append(name)
            for node in _capitals(expr):  # X_i, of a sample
                written = _member_of(node)
                if written is not None and scope.lookup(_SAMPLE.format(written[0])):
                    found.append(_SAMPLE.format(written[0]))
        return list(dict.fromkeys(found))

    def _with_unknowns(self, exprs: Sequence[ast.Expr | None], scope: Scope, kind: str) -> Scope:
        """A scope in which each capital letter nothing defines, as X is in
        ``\\operatorname{Var}(aX + b)``, is a random variable whose law is not given, as
        probability writes one; said in a note."""
        unknown: list[tuple[str, ast.Expr]] = []
        for expr in exprs:
            if expr is None:
                continue
            for node in _capitals(expr):
                name = probability.capital(node)
                written = _member_of(node)
                if written is not None and scope.lookup(_SAMPLE.format(written[0])):
                    continue  # X_i, a member of a sample
                if (
                    name is not None
                    and scope.lookup(name) is None
                    and name not in BUILTINS
                    and name not in CONSTANTS
                    and name not in (each for each, _ in unknown)
                ):
                    unknown.append((name, node))
        if not unknown:
            return scope
        inner = scope.child()
        for name, node in unknown:
            value = self.graph.const(distributions.unknown(name), SCALAR, node.span)
            inner.bind(name, VariableBinding(value))
        listed = " and ".join(node.span.text for _, node in unknown)
        verb = "is a random variable" if len(unknown) == 1 else "are random variables"
        self._note(
            f"{listed} {verb} whose law is not given, as probability writes random variables "
            f"with capitals: the {kind} is found whatever the law is"
        )
        return inner

    def _note_independence(self, names: Sequence[str], scope: Scope) -> None:
        """Say so where random variables a program declares meet without its saying how they
        depend on each other: they are taken independent."""
        groups = self.session.conventions.get(_INDEPENDENT, frozenset())
        names = [name for name in names if not name.endswith("_…")]  # a sample's are
        nodes = {}
        for name in names:
            binding = scope.lookup(name)
            assert isinstance(binding, VariableBinding)
            nodes[name] = binding.node
        alone = []
        for name in names:
            others = [other for other in names if other != name]
            related = any(
                _depends(nodes[name], nodes[other]) or _depends(nodes[other], nodes[name])
                for other in others
            )
            together = all(any({name, other} <= group for group in groups) for other in others)
            if not related and not together:
                alone.append(name)
        if len(alone) > 1:
            alone.sort()
            listed = ", ".join(alone[:-1]) + f" and {alone[-1]}"
            self._note(
                f"{listed} are taken independent, as nothing says how they depend on each other"
            )

    def _resolve_binding(self, span: Span, binding: VariableBinding | TensorBinding) -> None:
        if isinstance(binding, VariableBinding) and _random(binding):
            declared = binding.node.span
            where = f", declared on line {declared.line}" if declared is not None else ""
            shown = " ".join(declared.text.split()) if declared is not None else ""
            self._resolve(span, f"a random variable, {shown}{where}")
            return
        super()._resolve_binding(span, binding)


# -- the syntax ---------------------------------------------------------------------------------


def _sized(declared: Declaration) -> tuple[str, ast.Expr, ast.Expr] | None:
    """Of ``X_1, \\ldots, X_n \\sim D``, where the last subscript is no number: the name of
    the members, X, and the first and the last; ``None`` for any other declaration."""
    items = declared.variables
    if len(items) != 3 or not isinstance(items[1], ast.Dots):
        return None
    first, last = (_member_of(each) for each in (items[0], items[2]))
    if first is None or last is None or first[0] != last[0]:
        return None
    if isinstance(last[1], ast.Number):
        return None
    return first[0], items[0], items[2]


def _member_of(node: ast.Expr) -> tuple[str, ast.Expr] | None:
    """A capital letter with a subscript, as ``X_i``, ``X_{i}`` or ``X_1``: the letter and
    the subscript; ``None`` for anything else."""
    named = probability.capital(node)
    if named is None or "_" not in named:
        return None
    base, _, label = named.partition("_")
    if isinstance(node, ast.Scripted):
        body = node.scripts[0].body
        assert body is not None
        return base, body
    subscript: ast.Expr = (
        ast.Number(label, span=node.span) if label.isdigit() else ast.Name(label, span=node.span)
    )
    return base, subscript


def _shown_subscript(node: ast.Expr) -> str:
    """The subscript of a member as written, n of ``X_n`` and ``X_{n}``."""
    subscript = _subscript(node)
    if isinstance(subscript, ast.Name):
        return subscript.id
    if isinstance(subscript, ast.Number):
        return subscript.text
    return _unbraced(subscript)


def _subscript(node: ast.Expr) -> ast.Expr:
    written = _member_of(node)
    assert written is not None
    return written[1]


def _random(binding: Binding | None) -> bool:
    """Whether a name is bound to a random variable: one the program declares, or one
    whose law is not given."""
    if not isinstance(binding, VariableBinding):
        return False
    op = binding.node.op
    if isinstance(op, ops.Call):
        return op.name == "prob.variable"
    return isinstance(op, ops.Const) and isinstance(op.value, sympy.stats.rv.RandomSymbol)


def _depends(node: Node, other: Node) -> bool:
    """Whether the random variable ``node`` is given ``other``: its law, or a condition it
    is under, is written with it."""
    pending = list(node.args[1:])
    while pending:
        current = pending.pop()
        if current is other:
            return True
        pending.extend(current.args)
    return False


def _name(node: ast.Expr) -> str | None:
    """The name a variable is written with: ``X``, ``X_1`` or ``X_{1}``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Scripted) and isinstance(node.base, ast.Name):
        return probability.capital(node)
    return None


def _a_name(node: ast.Expr) -> bool:
    return _name(node) is not None


def _names_in(expr: ast.Expr) -> list[str]:
    """The names an expression is written with, a subscripted one as one name, and the
    base of a power as itself."""
    bases = _named_bases(expr)
    found: list[str] = []
    for node in _walk(expr):
        name = None
        if isinstance(node, ast.Scripted) and id(node.base) in bases:
            name = _name(node)
        elif isinstance(node, ast.Name) and id(node) not in bases:
            name = node.id
        if name is not None and name not in found:
            found.append(name)
    return found


def _capitals(expr: ast.Expr) -> list[ast.Expr]:
    """The names of ``expr`` written as random variables are, with capitals, as X of
    ``X^2`` and ``X_1``, but for those an accent makes a name of their own, as
    ``\\bar{X}``."""
    bases = _named_bases(expr)
    found: list[ast.Expr] = []
    for node in _walk(expr):
        if isinstance(node, ast.Scripted) and id(node.base) not in bases:
            continue  # X^2: its base is the name
        if isinstance(node, ast.Name) and id(node) in bases:
            continue
        if isinstance(node, ast.Scripted | ast.Name) and probability.capital(node) is not None:
            found.append(node)
    return found


def _named_bases(expr: ast.Expr) -> set[int]:
    """The names in ``expr`` that are part of a name of their own: the base of a subscripted
    name, as X of ``X_1``, and of an accented one, as X of ``\\bar{X}``."""
    found: set[int] = set()
    for node in _walk(expr):
        if isinstance(node, ast.Accent) or (
            isinstance(node, ast.Scripted) and _name(node) is not None
        ):
            found.add(id(node.base))
    return found


def _bracket(node: ast.Expr) -> tuple[ast.Expr, ...] | None:
    """What a bracket after an operator holds: ``[X]``, ``(X)`` or ``[X, Y]``."""
    if isinstance(node, ast.List):
        return tuple(node.items)
    return forms.bracketed(node)


def _an_event(node: ast.Expr) -> bool:
    return isinstance(node, ast.Equation | ast.Comparison) or (
        isinstance(node, ast.Binary) and node.op in ("and", "or")
    )


def _together(events: Sequence[ast.Expr]) -> ast.Expr:
    """Several events, written with commas, as one: that each holds."""
    joined = events[0]
    for event in events[1:]:
        joined = ast.Binary("and", joined, event, span=joined.span.to(event.span))
    return joined


def _power_of(argument: ast.Expr, power: int) -> ast.Expr | None:
    """What ``argument`` is the ``power`` of, where it is written so, as σ of σ²."""
    if (
        isinstance(argument, ast.Scripted)
        and len(argument.scripts) == 1
        and argument.scripts[0].kind == "^"
        and isinstance(argument.scripts[0].body, ast.Number)
        and argument.scripts[0].body.text == str(power)
    ):
        return argument.base
    return None


def _outside(value: object) -> bool:
    """Whether a number is no probability, outside 0 to 1."""
    number = sympy.sympify(value)
    return bool(number.is_number and number.is_real and (number < 0 or number > 1))


def _article(kind: str) -> str:
    return f"an {kind}" if kind[0] in "aeiou" else f"a {kind}"


def _unbraced(node: ast.Expr) -> str:
    return " ".join(node.span.text.split())


def _shown(declared: Declaration) -> str:
    return " ".join(declared.span.text.split())


def _described(law: Law) -> str:
    """A law, as a note says it: the distribution, and the value of each parameter written
    otherwise than its form writes it, as σ² = 4 of \\mathcal{N}(0, 4)."""
    from relativisticpy import display

    said = law.capability.brief
    values = []
    for slot, (argument, _) in zip(law.slots, law.arguments, strict=False):
        name, _, power = slot.partition("^")
        letter = Symbol(name) ** int(power) if power else Symbol(name)
        written = argument.text if isinstance(argument, ast.Number) else _unbraced(argument)
        if written.lstrip("\\").replace("{", "").replace("}", "") not in (slot, name):
            values.append(f"{display.unicode(letter)} = {written}")
    if values:
        said += ", with " + (" and ".join(values) if len(values) < 3 else ", ".join(values))
    return said
