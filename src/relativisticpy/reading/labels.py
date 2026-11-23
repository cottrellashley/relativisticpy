r"""The names books write with scripts, read by what each script labels
(``design/notation.md``, 5 and 6), each reading recorded with why, and a note where
another reading was possible:

- a subscript that labels is part of the name, whatever the base: ``\lambda_1``,
  ``\delta_0``, ``V_{10}``, ``M_\odot``, ``R_\oplus``, ``T_{\rm eff}``, ``r_+``,
  ``h_\times``, and ``\sigma_x := ...`` where no space is declared for x. An index of a
  tensor in force, or a label declared or ranging over values, stays an index, as in
  ``T_{\mu\nu}``, ``g_{tt}`` and ``x_t`` over data; and a built-in tensor claims a
  subscripted name only with as many indices as it has, or while a metric is in force;
- on the left of ``:=`` a number or an order raised on a name is part of the name, as in
  ``\sigma^2 := ...`` and ``\gamma^0 := ...``, and so is an accent, as in
  ``\hat\beta_1 := ...``;
- a name a program defines wins over any other reading of it, as ``R_\odot`` does over the
  Ricci tensor; and ``^*`` on it, or on a call, is its complex conjugate.
"""

from __future__ import annotations

from collections.abc import Sequence

from relativisticpy.compiler.builtins import BUILTINS, CONSTANTS
from relativisticpy.compiler.passes import Term
from relativisticpy.compiler.scope import (
    Context,
    FunctionBinding,
    Scope,
    TensorBinding,
    VariableBinding,
)
from relativisticpy.errors import CompileError, IndexNotationError
from relativisticpy.ir import SCALAR, Node, TensorType
from relativisticpy.language import syntax as ast
from relativisticpy.language.parser import split_index_labels
from relativisticpy.language.vocabulary import GREEK
from relativisticpy.reading.names import (
    _letter,
    _number_then_letters,
    _shown,
    _written_as_letter,
)
from relativisticpy.reading.subscripts import (
    Subscripts,
    _base_name,
    _index_like,
    _labelled_by_index,
    _raised,
)
from relativisticpy.reading.subscripts import label_name as label_name
from relativisticpy.typing.syntax import _plain_label, _walk

#: The decision of whether a script on a name labels it or indexes it.
LABEL = "label or index"

#: The names written before another to name a change in it, as \Delta x is the change in x.
_INCREMENTS = frozenset({"Delta"})


def compound_name(increment: ast.Expr, operand: ast.Expr) -> str | None:
    """The name ``\\Delta x`` makes, written as one quantity, the change in x: ``Delta x``,
    with a space, which no name written can have; ``Delta tau_1`` of ``\\Delta\\tau_1`` and
    ``Delta x'`` of ``\\Delta x'``. ``None`` for anything else, as ``\\Delta \\cdot x``, or
    ``\\Delta x^\\mu``, a vector of changes."""
    if not (isinstance(increment, ast.Name) and increment.id in _INCREMENTS):
        return None
    if not _written_as_letter(increment):
        return None
    if increment.span.source.text[increment.span.end : operand.span.start].strip():
        return None  # \Delta \cdot x, \Delta\, x
    if isinstance(operand, ast.Name) and operand.id == "d":
        return None  # \Delta {\rm d} t^2: \Delta times a differential
    if isinstance(operand, ast.Name):
        changed: str | None = operand.id
    elif isinstance(operand, ast.Prime):
        changed = operand.func.id + "'" * operand.order
    elif isinstance(operand, ast.Scripted) and all(s.kind == "_" for s in operand.scripts):
        changed = label_name(operand)
    else:
        changed = None
    return f"{increment.id} {changed}" if changed is not None else None


class Labels(Subscripts):
    """How the scripts on a name are read: as part of the name, or as its indices."""

    def _value_or_notation(
        self, node: ast.Expr, scope: Scope, ctx: Context, *, definition: bool = False
    ) -> Node:
        """A value, or notation kept as it is written, which the lowerer gives (see
        :class:`~relativisticpy.lowering.notation.Notation`)."""
        raise NotImplementedError

    # -- names in expressions -------------------------------------------------------------

    def _Name(self, node: ast.Name, scope: Scope, ctx: Context) -> Term:
        if ctx.latex and (run := self._run_with_a_label(node, scope, ctx)) is not None:
            return run
        if ctx.latex and (factors := self._after_a_number(node, scope, ctx)) is not None:
            return factors
        return super()._Name(node, scope, ctx)

    def _after_a_number(self, node: ast.Name, scope: Scope, ctx: Context) -> Node | None:
        """In LaTeX, ``x_1x_2``, a name with a number as its subscript and a letter right
        after it, which the lexer joins into one name: as TeX sets it, the letter starts a
        factor, x_1 times x_2 (design/notation.md, decision 5.9). ``None`` for any other
        name."""
        split = _number_then_letters(node.id)
        if split is None or node.verbatim or scope.lookup(node.id) is not None:
            return None
        first, rest = split
        why = "in LaTeX a subscript without braces ends where a letter follows its number"
        self._decide("letters", node.span, f"{first} times {rest}", why)
        factors = [ast.Name(first, span=node.span), ast.Name(rest, span=node.span)]
        return self._multiply([self.value(factor, scope, ctx) for factor in factors], node.span)

    def _run_with_a_label(self, node: ast.Name, scope: Scope, ctx: Context) -> Node | None:
        """In LaTeX, ``iE_1``, a run of letters that names nothing, with a subscript: as TeX
        sets it, the subscript binds to the last letter, i times E_1 (decision 5.7 of
        design/notation.md). ``None`` for any other name."""
        letters = self._labelled_run(node, scope)
        if letters is None:
            return None
        why = "in LaTeX a subscript binds to the last letter of a run that names nothing"
        reading = f"{' '.join(letters[:-1])} times {letters[-1]}"
        self._decide("letters", node.span, reading, why)
        factors = [
            _letter(letter, node.span, first=position == 0)
            for position, letter in enumerate(letters[:-1])
        ]
        factors.append(ast.Name(letters[-1], span=node.span))
        return self._multiply([self.value(factor, scope, ctx) for factor in factors], node.span)

    def _labelled_run(self, node: ast.Name, scope: Scope) -> list[str] | None:
        """The letters of ``iE_1``, a run of letters that names nothing, with a subscript,
        which binds to the last: i and E_1. ``None`` for any other name."""
        if node.verbatim or scope.lookup(node.id) is not None:
            return None  # \mathrm{iE_1}, or a name read as one, as xbar_00 of \bar x_{00}
        run, underscore, subscript = node.id.partition("_")
        if not (underscore and subscript and len(run) > 1 and run.isalpha() and run.isascii()):
            return None
        known = run in BUILTINS or run in CONSTANTS or scope.lookup(run) is not None
        if known or run in self.session.names.values():
            return None
        if any(self._builtin_tensor(run, rank) for rank in range(5)):
            return None
        letters = split_index_labels(run)
        if len(letters) < 2:
            return None
        return [*letters[:-1], f"{letters[-1]}_{subscript}"]

    def _labelled(self, node: ast.Scripted, scope: Scope, ctx: Context) -> Term | None:
        """A name with scripts read as a name of its own (decision 5): one the program
        defines, as ``\\lambda_1`` after ``\\lambda_1 := ...``, with what is raised on it
        after, as the power of ``\\lambda_1^2``; or one nothing defines whose base claims
        its subscripts for nothing else, as ``R_\\odot`` without a metric, or whose base
        is an accent, as ``\\hat\\beta_1``. ``None`` where the scripts are read otherwise:
        as indices, an element or a power, or, of a name nothing defines, as before."""
        if (numbered := self._numbered_tensor(node, scope, ctx)) is not None:  # k_{1\mu}
            return numbered
        lowered = any(script.kind == "_" for script in node.scripts)
        readings = [raised for raised in (True, False) if raised or lowered]
        for raised in readings:  # \sigma^2 or \lambda_1^2 as the program names them
            name = label_name(node, raised=raised)
            if name is None or scope.lookup(name) is None:
                continue
            rest = () if raised else tuple(s for s in node.scripts if s.kind == "^")
            return self._named_label(node, name, rest, scope, ctx)
        if not lowered:
            return None
        base = _base_name(node.base)
        if base is None or scope.lookup(base) is not None:
            return None  # the scripts of what is defined: an index, an element, a power
        if isinstance(node.base, ast.Name):
            built_in = any(self._builtin_tensor(base, rank) for rank in range(1, 5))
            if not built_in or self._indexes(node, base):
                return None  # \lambda_1 is a symbol of its own, as before
        elif not isinstance(node.base, ast.Accent) or _labelled_by_index(node):
            return None  # \bar h_{\mu\nu} is written with indices, which no name has
        name = label_name(node, raised=False)
        if name is None:
            return None
        rest = tuple(s for s in node.scripts if s.kind == "^")
        return self._named_label(node, name, rest, scope, ctx)

    def _numbered_tensor(self, node: ast.Scripted, scope: Scope, ctx: Context) -> Term | None:
        """``k_{1\\mu}`` beside a tensor k_1: a label, then an index, in one subscript, as
        books write the momentum k_1 with its index \\mu (label-index-subscript of
        curriculum/ROADMAP.md); ``None`` for any other name."""
        base = node.base
        if not isinstance(base, ast.Name) or len(node.scripts) != 1:
            return None
        script = node.scripts[0]
        items = script.indices.items if script.indices is not None else ()
        if script.kind != "_" or len(items) < 2 or not isinstance(items[0].value, ast.Number):
            return None
        rest = items[1:]
        if items[0].label is not None or not all(map(_plain_label, rest)):
            return None
        tensor = f"{base.id}_{items[0].value.text}"
        binding = scope.lookup(tensor)
        defined = isinstance(binding, TensorBinding) or (
            isinstance(binding, VariableBinding) and isinstance(binding.node.type, TensorType)
        )
        if not defined:
            return None
        labels = "".join(self._written_label(str(item.label)) for item in rest)
        written = f"{base.span.text}_{items[0].value.text}"
        why = f"{written} is a tensor{self._line_of(binding)}, and {labels} no number"
        self._decide(LABEL, node.span, f"the tensor {written} with its index {labels}", why)
        indices = ast.IndexList(tuple(rest), span=script.span)
        lowered = ast.Script("_", None, indices, span=script.span)
        named = ast.Name(tensor, verbatim=True, span=base.span)
        return self.term(ast.Scripted(named, (lowered,), span=node.span), scope, ctx)

    def _named_label(
        self,
        node: ast.Scripted,
        name: str,
        rest: Sequence[ast.Script],
        scope: Scope,
        ctx: Context,
    ) -> Term:
        """The name a name with scripts makes, with what is raised on it after: a power, a
        mark, as ``T`` and ``\\dagger`` are on a matrix, or ``*``, its conjugate."""
        written = node.base.span.text + "".join(
            script.span.text for script in node.scripts if script not in rest
        )
        binding = scope.lookup(name)
        if binding is not None:
            why = f"{written} is defined{self._line_of(binding)}"
        else:
            base = _shown(_base_name(node.base) or "", ctx.latex)
            why = f"its subscript labels {base}, which is no tensor here"
        self._decide(LABEL, node.span, f"the name {written}", why, written=written)
        named = ast.Name(name, verbatim=True, span=node.span)
        if not rest:
            return self.term(named, scope, ctx)
        if (conjugate := self._starred(node, named, rest, scope, ctx)) is not None:
            return conjugate
        return self.term(ast.Scripted(named, tuple(rest), span=node.span), scope, ctx)

    def _starred(
        self,
        node: ast.Scripted,
        operand: ast.Expr,
        rest: Sequence[ast.Script],
        scope: Scope,
        ctx: Context,
    ) -> Node | None:
        """``\\psi_1(x)^*`` or ``\\psi_{200}^*``: the complex conjugate of a call or of a
        number the program defines (decision 6.4 of design/notation.md), where SymPy would
        raise it to the power of a symbol named *; ``None`` for any other script."""
        if len(rest) != 1 or not _a_star(rest[0]):
            return None
        value = self.value(operand, scope, ctx)
        if value.type != SCALAR:
            return None
        why = f"{operand.span.text} has a value, and * on it is its complex conjugate"
        self._decide(LABEL, node.span, f"the complex conjugate of {operand.span.text}", why)
        return self._call("conjugate", [value], SCALAR, node.span)

    def _starred_conjugate(self, node: ast.Scripted, scope: Scope, ctx: Context) -> Node | None:
        """``\\psi_1(x)^*`` or ``z^*`` after ``z := 1 + i``: the complex conjugate of what a
        call gives, or of a number the program defines; ``None`` for anything else, as
        ``r^*``, a name of its own where r means nothing."""
        base = node.base
        named = isinstance(base, ast.Name) and isinstance(scope.lookup(base.id), VariableBinding)
        if not (isinstance(base, ast.Call) or named):
            return None
        return self._starred(node, base, node.scripts, scope, ctx)

    def _check_statistic_name(self, factor: ast.Expr, scope: Scope) -> None:
        """A Greek letter with a subscript, as ``\\sigma_1``, or a name with a subscript the
        program defines, is no statistic misspelt, nor an operator RelativisticPy lacks."""
        if isinstance(factor, ast.Scripted) and isinstance(factor.base, ast.Name):
            name = label_name(factor)
            defined = name is not None and scope.lookup(name) is not None
            if defined or _written_as_letter(factor.base):
                return
        super()._check_statistic_name(factor, scope)

    @staticmethod
    def _line_of(binding: object) -> str:
        """`` on line 3``, where a binding was defined, if that is known."""
        if isinstance(binding, FunctionBinding):
            return f" on line {binding.definition.span.line}"
        if isinstance(binding, VariableBinding | TensorBinding) and binding.node.span is not None:
            return f" on line {binding.node.span.line}"
        return ""

    # -- compound names -----------------------------------------------------------------

    def _compound_names(
        self, factors: list[ast.Expr], scope: Scope, ctx: Context
    ) -> list[ast.Expr]:
        """Factors with \\Delta and a name written right after it made one name, the change
        in it, as ``\\Delta x`` and ``\\Delta\\phi`` are (decision 4 of
        design/notation.md's roadmap, compound names), where it is so defined or \\Delta
        means nothing; \\Delta with a value multiplies, as the Δ of Kerr's metric does."""
        if ctx.line_element:  # -\Delta dt^2: \Delta times a differential
            return factors
        merged: list[ast.Expr] = []
        position = 0
        while position < len(factors):
            factor = factors[position]
            following = factors[position + 1] if position + 1 < len(factors) else None
            name = compound_name(factor, following) if following is not None else None
            if (
                following is not None
                and name is not None
                and self._reads_compound(factor, following, name, scope)
            ):
                merged.append(ast.Name(name, verbatim=True, span=factor.span.to(following.span)))
                position += 2
                continue
            merged.append(factor)
            position += 1
        return merged

    def _reads_compound(
        self, increment: ast.Expr, operand: ast.Expr, name: str, scope: Scope
    ) -> bool:
        """Whether ``\\Delta x`` is one name, the change in x: defined so, or where \\Delta
        means nothing. Before a function, \\Delta is its Laplacian, which is refused."""
        span = increment.span.to(operand.span)
        written = span.text
        binding = scope.lookup(name)
        if binding is not None:
            why = f"{written} is defined{self._line_of(binding)}"
            self._decide(LABEL, span, f"the name {written}", why)
            return True
        delta = increment.id if isinstance(increment, ast.Name) else ""
        session = self.session
        if scope.lookup(delta) is not None or delta in session.assumptions:
            return False  # \Delta has a value: \Delta times x
        if delta in session.functions or (session.metric and delta in session.metric.names):
            return False
        changed = operand.span.text
        if isinstance(operand, ast.Name) and (
            isinstance(scope.lookup(operand.id), FunctionBinding) or operand.id in session.functions
        ):
            raise CompileError(
                f"{written}, with {changed} a function, is its Laplacian, which is not "
                "supported yet",
                span=span,
                hint=f"write its second derivatives out, as in \\frac{{\\partial^2 {changed}}}"
                f"{{\\partial x^2}} + \\frac{{\\partial^2 {changed}}}{{\\partial y^2}}",
            )
        note = f"{written} is read as one name, the change in {changed}"
        why = f"{increment.span.text} means nothing here, and before a name writes a change in it"
        self._decide(LABEL, span, f"the name {written}", why, note)
        self._note(note)
        return True

    def _define_compound(
        self, target: ast.Binary, value: ast.Expr, scope: Scope, ctx: Context
    ) -> bool:
        """``\\Delta x := 0.1``: the change in x, one name, defined as a name is."""
        name = compound_name(target.left, target.right) if target.op == "*" else None
        if name is None:
            return False
        operand = target.right
        if isinstance(operand, ast.Name) and isinstance(scope.lookup(operand.id), FunctionBinding):
            return False  # \Delta f, of a function: its Laplacian, which := cannot define
        written = target.span.text
        note = f"{written} is defined as one name, the change in {operand.span.text}"
        why = "\\Delta before a name writes a change in it"
        self._decide(LABEL, target.span, f"the name {written}", why, note, written=written)
        self._note(note)
        scope.bind(
            name, VariableBinding(self._value_or_notation(value, scope, ctx, definition=True))
        )
        return True

    # -- definitions --------------------------------------------------------------------

    def _define_labelled(
        self, target: ast.Expr, value: ast.Expr, scope: Scope, ctx: Context
    ) -> bool:
        """Define what a name with scripts labels, where they are no indices (decisions 5
        and 6): ``\\lambda_1 := ...``, ``r_+ := ...``, ``\\sigma^2 := ...``,
        ``\\hat\\beta_1 := ...``, and ``\\sigma_x := ...`` with no space declared for x. A
        definition whose scripts are indices, as of ``T^{\\mu\\nu}`` or of ``v_a`` with a
        declared over a space, is a tensor's, and left to it: ``False`` then."""
        if isinstance(target, ast.Binary):  # \Delta x := ...
            return self._define_compound(target, value, scope, ctx)
        if not isinstance(target, ast.Scripted):
            return False
        base = _base_name(target.base)
        if base is None or base == self.session.names["metric"]:
            return False
        name = label_name(target)
        if name is None or not self._labels_a_definition(target, value):
            return False
        written = target.span.text
        raised = any(script.kind == "^" for script in target.scripts)
        why = "its scripts label it, as no index does"
        note = None
        if raised:
            shown = _shown(base.partition("_")[0], ctx.latex)
            why = "what is raised on a name being defined is part of it"
            note = f"{written} is defined as a name of its own, not as a power of {shown}"
            self._note(note)
        self._decide(LABEL, target.span, f"the name {written}", why, note, written=written)
        node = self._value_or_notation(value, scope, ctx, definition=True)
        if isinstance(node.type, TensorType):  # \Psi_2 := C_{abcd} l^a m^b \bar m^c n^d
            free = self._written_indices(node.type.indices)
            raise IndexNotationError(
                f"{written} is a name of its own, with no indices, but its value has the free "
                f"indices {free}",
                span=value.span,
                notes=["a name with a label, as a number below it, is no tensor"],
                hint="contract every index of the value, or define a tensor with its indices",
            )
        scope.bind(name, VariableBinding(node))
        return True

    def _labels_a_definition(self, target: ast.Scripted, value: ast.Expr) -> bool:
        """Whether the scripts of a name being defined label it rather than index a tensor:
        a number, a mark, a sign or a word below, a number or an order above, or one Latin
        letter below that can be no index, with no coordinates
        and no space declared for it, and a matrix as the value, as x of
        ``\\sigma_x := \\begin{pmatrix} ... \\end{pmatrix}``: a vector of one index has no
        matrix as its components."""
        items = [
            item
            for script in target.scripts
            if script.indices is not None
            for item in script.indices.items
        ]
        if any(item.label is not None and self._an_index_here(item.label) for item in items):
            return False
        if any(s.kind == "^" and _raised(s) is None for s in target.scripts):
            return False  # T^{\mu\nu}, k'^\mu: indices
        if any(s.kind == "_" and not _index_like(s) for s in target.scripts):
            return True  # r_+, h_\times, M_\odot, T_{\rm eff}
        if any(item.label is None and isinstance(item.value, ast.Number) for item in items):
            return True  # \lambda_1, V_{10}, \sigma^2; but T_{a=0} is a component
        if not all(_plain_label(item) for item in items):
            return False
        labels = [str(item.label) for item in items]
        if len(labels) != 1 or not isinstance(target.base, ast.Name):
            return False  # T_{ab}, \bar h_{\mu\nu}: indices
        if labels[0] in GREEK or not labels[0].isascii():
            return False
        return _writes_a_matrix(value) and self.session.coordinates is None


def _writes_a_matrix(value: ast.Expr) -> bool:
    """Whether a value is written with a matrix of more than one row and column, which no
    tensor with one index has as its components: ``\\begin{pmatrix} 0 & 1 \\\\ 1 & 0
    \\end{pmatrix}`` or ``[[0, 1], [1, 0]]``, maybe times a number."""
    for node in _walk(value):
        if isinstance(node, ast.Matrix) and len(node.rows) > 1 and len(node.rows[0]) > 1:
            return True
        rows = node.items if isinstance(node, ast.List) and len(node.items) > 1 else ()
        if rows and all(isinstance(row, ast.List) and len(row.items) > 1 for row in rows):
            return True
    return False


def _a_star(script: ast.Script) -> bool:
    """Whether a superscript is a star, ``^*`` or ``^{*}``."""
    body = script.body
    return script.kind == "^" and isinstance(body, ast.Name) and body.verbatim and body.id == "*"
