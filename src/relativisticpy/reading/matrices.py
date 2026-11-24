r"""The notation of matrices that is no call, read by its operands (``design/notation.md``,
6, 12, 13 and 23), each reading recorded with why, and a note where another was possible:

- a superscript T, ``\top`` or ``\intercal`` on a matrix is its transpose where T has no
  value; ``\dagger`` and * its conjugate transpose; + its pseudoinverse; and, with a note,
  a superscript t, after the matrix or before it, as Artin and Lang write ``A^t`` and
  ``{}^tA``, where t means nothing, and a tilde over it, as Goldstein and Griffiths write
  ``\tilde{A}``;
- ``e^{A t}`` and ``\exp`` of a matrix are its exponential, and ``\log`` and ``\sqrt`` of
  a matrix its logarithm and square root; a matrix to a power that is no whole number, as
  ``\Lambda^k`` or ``V^{1/2}``, is its power, or its square root;
- I, ``\mathbb{1}`` and 𝟙, which nothing defines, added to a matrix are the identity of
  its size, and ``\mathbf{1}`` too, but beside a vector the vector of ones; a number added
  to a square matrix is that multiple of the identity; I, ``\mathbb{1}`` and 𝟙 multiplying
  a matrix, or beside one in ``\otimes``, are the identity too; a definition of any of them
  wins;
- ``\mathbf{1}`` in a product with a matrix is the vector of ones it meets, and
  ``\mathbf{1}^T`` the row;
- a product that is a 1 × 1 matrix, as a row times a matrix times a column is, is the
  number it holds;
- ``\otimes`` between matrices is their Kronecker product, and of a matrix and what is no
  matrix refused; ``\oplus`` between matrices is their direct sum;
- ``\|A\|_p`` of a matrix or a vector is its norm, and ``(A^3)_{12}`` or ``A_{12}`` of a
  matrix its entry, counted from 1.

The functions of a matrix these call are the capabilities of
:mod:`relativisticpy.integrations.matrices`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import sympy

from relativisticpy import integrations
from relativisticpy.compiler.builtins import BUILTINS
from relativisticpy.compiler.passes import Term
from relativisticpy.compiler.scope import Context, Scope, VariableBinding
from relativisticpy.errors import CompileError, RelativisticPyError
from relativisticpy.ir import SCALAR, VALUE, Node, TensorType, dependencies, ops
from relativisticpy.language import syntax as ast
from relativisticpy.language.source import Span
from relativisticpy.reading.names import (
    _BOLD,
    BOLD_ONE,
    _a_matrix,
    _bold_one,
    _joined_by,
    _matrix_shape,
    _named,
)
from relativisticpy.reading.reader import Reader
from relativisticpy.typing.syntax import _operand, _product_factors

#: The decision of how the notation of a matrix is read.
MATRIX = "matrix or number"

#: The superscripts that mark a matrix, each with the function of a matrix it calls and
#: what that is.
_MARKS = {
    "T": ("transpose", "the transpose"),
    "top": ("transpose", "the transpose"),
    "intercal": ("transpose", "the transpose"),
    "dagger": ("ctranspose", "the conjugate transpose"),
    "*": ("ctranspose", "the conjugate transpose"),
    "+": ("pinv", "the pseudoinverse"),
}

#: A t raised before a name, as the parser names Lang's {}^tA.
_LEFT_T = re.compile(r"\^\{t\}(?P<name>[A-Za-z]\w*)")

#: The names of the identity, where nothing defines them.
_IDENTITIES = ("I", "𝟙")

#: The written orders of a norm, as the subscripts of \|A\|_p write them.
_ORDERS = {"F": "F", "infty": sympy.oo, "1": 1, "2": 2}

#: The functions of numbers that are, of a matrix, the functions of a matrix.
_OF_A_MATRIX = {
    "exp": ("expm", "the exponential"),
    "log": ("logm", "the logarithm"),
    "ln": ("logm", "the logarithm"),
    "sqrt": ("sqrtm", "the square root"),
}


class Matrices(Reader):
    """How the notation of matrices that is no call is read."""

    # -- what a capability is called with ----------------------------------------------

    def _of_a_matrix(self, name: str, args: Sequence[Node], span: Span) -> Node:
        """The function of a matrix called ``name``, applied to ``args``."""
        capability = integrations.find(name)
        kind = SCALAR if capability.scalar else VALUE
        message = BUILTINS[name].check_arity(len(args))
        if message is not None:
            raise CompileError(message, span=span)
        return self._call(name, list(args), kind, span)

    def _matrix_operand(self, node: ast.Expr, scope: Scope, ctx: Context) -> Node | None:
        """What a matrix's mark or script is written on, if it is a matrix: a name the
        program gave one as its value, decorated or not, as ``\\bar{\\mathbf{e}}``, or what
        is written as one, as a call or a bracket whose value is one; ``None`` for anything
        else, as a symbol."""
        node = _named(node)
        if isinstance(node, ast.Name):
            binding = scope.lookup(node.id)
            if not (isinstance(binding, VariableBinding) and _a_matrix(binding.node)):
                return None
        elif isinstance(node, ast.Number):  # but a program's own \mathbf{1}
            if not (_bold_one(node) and scope.lookup(BOLD_ONE) is not None):
                return None
        elif isinstance(node, ast.Operator | ast.Accent | ast.Prime):
            return None
        value = self.value(node, scope, ctx)
        return value if _a_matrix(value) else None

    # -- scripts -----------------------------------------------------------------------

    def _matrix_scripted(self, node: ast.Scripted, scope: Scope, ctx: Context) -> Node | None:
        """A matrix with a mark, as ``A^T`` and ``A^+``; a norm, ``\\|A\\|_2``; or an entry,
        ``(A^3)_{12}``. ``None`` for any other script."""
        if isinstance(node.base, ast.Abs) and node.scripts[0].kind == "_":
            return self._norm_scripted(node, scope, ctx)
        if len(node.scripts) != 1:
            return None
        script = node.scripts[0]
        if script.kind == "_":
            return self._entry(node, scope, ctx)
        mark = script.body.id if isinstance(script.body, ast.Name) else None
        if mark == "t" and self._means_nothing("t", scope):  # A^t, as Artin writes it
            return self._book_transpose(node.base, node.base.span.text, node.span, scope, ctx)
        if mark not in _MARKS or (mark == "T" and scope.lookup("T") is not None):
            return None
        name, what = _MARKS[mark]
        written = node.base.span.text
        matrix = self._matrix_operand(node.base, scope, ctx)
        bold = any(command in written for command in _BOLD)  # \mathbf{x}^T: a vector meant
        if matrix is None and (mark in ("top", "intercal", "dagger") or (mark == "T" and bold)):
            notes = (
                ["the adjoint of an operator is not supported yet"]
                if mark == "dagger"
                else ["with indices, the transpose of A_{ab} is A_{ba}"]
            )
            if bold:
                notes.append(
                    "a bold name is a vector or a matrix in books, but nothing defines "
                    "this one as one"
                )
            ones = _bold_one(node.base)
            named = isinstance(node.base, ast.Name | ast.Scripted | ast.Accent) or ones
            example = written if named else "\\mathbf{e}_1"
            column = "1 \\\\ 1" if ones else "1 \\\\ 0"
            raise CompileError(
                f"{node.span.text} is {what} of {written}, which is not a matrix",
                span=node.span,
                notes=notes,
                hint=f"define {written if named else 'what it is written with'} as a matrix "
                f"first, as in {example} := \\begin{{pmatrix}} {column} \\end{{pmatrix}}",
            )
        if matrix is None:
            return None
        why = f"{written} is a matrix" + (", and T has no value" if mark == "T" else "")
        self._decide(MATRIX, node.span, f"{what} of {written}", why)
        return self._of_a_matrix(name, [matrix], node.span)

    def _Name(self, node: ast.Name, scope: Scope, ctx: Context) -> Term:
        if (transposed := self._left_transpose(node, scope, ctx)) is not None:
            return transposed  # {}^tA, as Lang writes it
        return super()._Name(node, scope, ctx)

    def _left_transpose(self, node: ast.Name, scope: Scope, ctx: Context) -> Node | None:
        """``{}^tA`` or ``{}^{t}A``, a t raised before a matrix, which the parser makes a
        name: its transpose, as Lang writes it, where t means nothing; ``None`` otherwise."""
        match = _LEFT_T.fullmatch(node.id) if node.verbatim else None
        if match is None or not self._means_nothing("t", scope):
            return None
        base = ast.Name(match["name"], span=node.span)
        return self._book_transpose(base, match["name"], node.span, scope, ctx)

    def _tilde_transpose(self, node: ast.Accent, scope: Scope, ctx: Context) -> Node | None:
        """``\\tilde{A}`` of a matrix the program defines, where nothing defines the name
        with its tilde: its transpose, as Goldstein and Griffiths write it, with a note;
        ``None`` otherwise."""
        base = node.base
        if not isinstance(base, ast.Name) or scope.lookup(f"{base.id}tilde") is not None:
            return None
        return self._book_transpose(base, base.span.text, node.span, scope, ctx, tilde=True)

    def _means_nothing(self, name: str, scope: Scope) -> bool:
        """Whether nothing gives a letter a meaning: no value, no assumption, and no
        coordinate, as t of a Markov chain's P^t has, once ``t \\in \\mathbb{N}``."""
        coordinates = {c.name for c in self.session.coordinates or ()}
        known = name in self.session.assumptions or name in coordinates
        return scope.lookup(name) is None and not known

    def _book_transpose(
        self,
        base: ast.Expr,
        written: str,
        span: Span,
        scope: Scope,
        ctx: Context,
        *,
        tilde: bool = False,
    ) -> Node | None:
        """``A^t``, ``{}^tA`` or ``\\tilde{A}`` of a matrix, ``written`` so: its transpose,
        as Artin and Lang, and Goldstein and Griffiths, write it, with a note that says how
        to write the other reading. ``None`` where what the mark is on is no matrix."""
        matrix = self._matrix_operand(base, scope, ctx)
        if matrix is None:
            return None
        whole = span.text if not span.text.startswith("^") else "{}" + span.text  # {}^tA
        if tilde:
            books = "Goldstein and Griffiths"
            other = f"for a name of its own, define {whole} first"
        else:
            books = "Artin and Lang"
            other = f"for a power of {written}, say what t is first, as t \\in \\mathbb{{N}}"
        why = f"{written} is a matrix" + ("" if tilde else ", and t means nothing")
        note = f"{whole} is read as the transpose of {written}, as {books} write it; {other}"
        self._decide(MATRIX, span, f"the transpose of {written}", why, note)
        self._note(note)
        return self._of_a_matrix("transpose", [matrix], span)

    def _entry(self, node: ast.Scripted, scope: Scope, ctx: Context) -> Node | None:
        """``(A^3)_{12}`` or ``A_{12}`` of a matrix: its entry in row 1 and column 2, counted
        from 1 as books count them."""
        indices = node.scripts[0].indices
        items = indices.items if indices is not None else ()
        numbers = [item.value for item in items if isinstance(item.value, ast.Number)]
        if len(items) != 2 or len(numbers) != 2 or not all(n.text.isdigit() for n in numbers):
            return None
        row, column = (int(number.text) for number in numbers)
        if row < 1 or column < 1:
            return None
        matrix = self._matrix_operand(node.base, scope, ctx)
        if matrix is None:
            return None
        written = node.base.span.text
        why = f"{written} is a matrix, whose entries are counted from 1"
        self._decide(
            MATRIX, node.span, f"the entry of {written} in row {row}, column {column}", why
        )
        place = [self.graph.const(row - 1, VALUE), self.graph.const(column - 1, VALUE)]
        return self._call("linalg.entry", [matrix, *place], SCALAR, node.span)

    def _norm_scripted(self, node: ast.Scripted, scope: Scope, ctx: Context) -> Node | None:
        """``\\|A\\|_2``, ``\\|A\\|_F``, ``\\|A\\|_\\infty`` or ``\\|x\\|_1``, maybe squared,
        as ``\\|A x - b\\|_2^2``: the norm of a matrix or a vector, of that order."""
        base = node.base
        assert isinstance(base, ast.Abs)
        subscript, *powers = node.scripts
        order = subscript.span.text.lstrip("_").strip("{} ").lstrip("\\")
        if not _double_bars(base) or order not in _ORDERS or any(s.kind == "_" for s in powers):
            return None
        matrix = self._matrix_operand(base.operand, scope, ctx)
        if matrix is None:
            matrix = self._vector_operand(base.operand, scope)  # \|x\|_2 of x := [3, 4]
        if matrix is None:
            return None
        written = base.operand.span.text
        why = f"{written} is a matrix or a vector, and double bars with a subscript write a norm"
        self._decide(MATRIX, node.span, f"the norm of {written} of order {order}", why)
        value = self._of_a_matrix(
            "norm", [matrix, self.graph.const(_ORDERS[order], VALUE)], node.span
        )
        for power in powers:
            if power.body is None:
                return None
            value = self._power(value, self.value(power.body, scope, ctx), node.span)
        return value

    def _vector_operand(self, node: ast.Expr, scope: Scope) -> Node | None:
        """What a norm's bars hold, where it is a name the program gives a list of numbers,
        a vector, as x of ``x := [3, 4]``; ``None`` for anything else."""
        if not isinstance(node, ast.Name):
            return None
        binding = scope.lookup(node.id)
        if not isinstance(binding, VariableBinding):
            return None
        value = binding.node
        listed = isinstance(value.op, ops.Call) and value.op.name == "core.list"
        if not listed or not value.args or any(arg.type != SCALAR for arg in value.args):
            return None
        return value

    def _norm_of(self, node: ast.Abs, operand: Node) -> Node | None:
        """``\\|x\\|`` of a matrix: its norm of order 2, the length of a vector, and the
        largest singular value of a matrix, said in a note."""
        if not (_double_bars(node) and _a_matrix(operand)):
            return None
        written = node.operand.span.text
        why = f"{written} is a matrix, and double bars write a norm"
        shape = _matrix_shape(operand)
        vector = shape is not None and 1 in shape
        note = None
        if not vector:
            note = (
                f"{node.span.text} is read as the 2-norm of {written}, its largest singular "
                f"value; for another, write \\|{written}\\|_F, \\|{written}\\|_1 or "
                f"\\|{written}\\|_\\infty"
            )
            self._note(note)
        self._decide(MATRIX, node.span, f"the norm of {written} of order 2", why, note)
        return self._of_a_matrix("norm", [operand, self.graph.const(2, VALUE)], node.span)

    # -- functions and powers ------------------------------------------------------------

    def _function_of_a_matrix(self, name: str, args: Sequence[Node], span: Span) -> Node | None:
        """``\\exp``, ``\\log`` or ``\\sqrt`` of a matrix: the function of the matrix, not of
        each of its entries, as ``\\exp(-i\\theta\\sigma_y/2)``; with a base, ``\\log_2 \\rho``
        is the logarithm divided by that of the base. ``None`` for anything else."""
        if name not in _OF_A_MATRIX or not args or not _a_matrix(args[0]):
            return None
        capability, what = _OF_A_MATRIX[name]
        why = "its argument is a matrix"
        self._decide(MATRIX, span, f"{what} of a matrix, {capability}", why)
        value = self._of_a_matrix(capability, args[:1], span)
        if len(args) == 2:  # \log_2 \rho
            base = self._call("log", [args[1]], SCALAR, span)
            minus_one = self.graph.const(sympy.S.NegativeOne, SCALAR)
            reciprocal = self.graph.add(ops.Pow(), (base, minus_one), SCALAR, span)
            value = self._multiply([reciprocal, value], span)
        return value

    def _matrix_power(self, base: Node, exponent: Node, span: Span) -> Node | None:
        """``e^{A t}`` of a matrix: its exponential. A matrix to a power that is no whole
        number: its square root, as ``V^{1/2}``, or its power, as ``\\Lambda^k`` and
        ``D^{-1/2}``. ``None`` for anything else, as ``A^2``, a product of matrices."""
        euler = isinstance(base.op, ops.Const) and base.op.value is sympy.E
        if euler and _a_matrix(exponent):
            why = "its exponent is a matrix"
            self._decide(MATRIX, span, "the exponential of a matrix, expm", why)
            return self._of_a_matrix("expm", [exponent], span)
        if not _a_matrix(base) or _a_matrix(exponent) or exponent.type != SCALAR:
            return None
        value = _known(exponent)
        if value is not None and sympy.sympify(value).is_integer:
            return None
        if value == sympy.Rational(1, 2):
            why = "it is a matrix to the power 1/2"
            self._decide(MATRIX, span, "the square root of a matrix, sqrtm", why)
            return self._of_a_matrix("sqrtm", [base], span)
        why = "it is a matrix to a power that is no whole number"
        self._decide(MATRIX, span, "the power of a matrix, matrix_power", why)
        return self._of_a_matrix("matrix_power", [base, exponent], span)

    # -- sums ----------------------------------------------------------------------------

    def _terms_of_a_sum(
        self, terms: Sequence[tuple[int, ast.Expr]], scope: Scope, ctx: Context
    ) -> list[tuple[int, Node, ast.Expr]]:
        """The terms of a sum, each with its sign. A term that writes the identity, I,
        𝟙, ``\\mathbb{1}`` or ``\\mathbf{1}`` where nothing defines them, is read after the
        others, as the identity of the size of the matrix among them, if there is one; and
        a number added to a square matrix is that multiple of the identity (decision 12)."""
        values = self._beside_a_matrix([term for _, term in terms], scope, ctx, "added to")
        return [(sign, value, term) for (sign, term), value in zip(terms, values, strict=True)]

    def _sides_of_an_equation(
        self, node: ast.Equation, scope: Scope, ctx: Context
    ) -> tuple[Node, Node]:
        """The sides of an equation, read as the terms of a sum are: in ``U^\\dagger U = I``,
        I is the identity, and in ``A = 2``, 2 times it; but ``A = 0`` is the matrix of
        zeros, as before."""
        lhs, rhs = self._beside_a_matrix([node.lhs, node.rhs], scope, ctx, "equated to")
        return lhs, rhs

    def _beside_a_matrix(
        self, parts: Sequence[ast.Expr], scope: Scope, ctx: Context, relation: str
    ) -> list[Node]:
        """The terms of a sum or the sides of an equation, the ``parts`` ``relation`` each
        other, each with the identity read beside a matrix."""
        sum_ = relation == "added to"
        identities = [self._identity_written(part, scope) for part in parts]
        values: dict[int, Node] = {}
        for position, part in enumerate(parts):
            if identities[position] is None:
                values[position] = self._term(part, scope, ctx, grouped=sum_)
        beside = next(((value, parts[k]) for k, value in values.items() if _a_matrix(value)), None)
        if any(isinstance(value.type, TensorType) for value in values.values()):
            beside = None
        shape = _matrix_shape(beside[0]) if beside is not None else None
        square = shape is None or None in shape or shape[0] == shape[1]
        for position, part in enumerate(parts):
            identity = identities[position]
            if identity is not None:
                values[position] = self._identity_term(
                    part, identity, beside, scope, ctx, relation, grouped=sum_
                )
                continue
            value = values[position]
            zero = not sum_ and _known(value) == 0  # A = 0: the matrix of zeros, as before
            if beside is not None and square and value.type == SCALAR and not zero:
                values[position] = self._times_identity(value, beside, part, relation)
        return [values[position] for position in range(len(parts))]

    def _term(self, term: ast.Expr, scope: Scope, ctx: Context, *, grouped: bool = True) -> Node:
        written = _operand(term) if grouped else term  # [x + 1] in a sum is a bracket
        return self._components_of(self.value(written, scope, ctx), scope, ctx, term.span)

    def _identity_written(self, term: ast.Expr, scope: Scope) -> ast.Expr | None:
        """The identity a term of a sum is, maybe times a number: I, 𝟙 or ``\\mathbf{1}``
        where nothing defines them, as a factor of it; ``None`` where it is none."""
        while isinstance(term, ast.Unary) and term.op in ("-", "+"):
            term = term.operand
        while isinstance(term, ast.Binary) and term.op == "/":  # I/2
            term = term.left
        for factor in _product_factors(_operand(term)):
            if isinstance(factor, ast.Name) and factor.id in _IDENTITIES:
                if scope.lookup(factor.id) is None:
                    return factor
            elif _bold_one(factor) and scope.lookup(BOLD_ONE) is None:
                return factor
        return None

    def _identity_term(
        self,
        term: ast.Expr,
        identity: ast.Expr,
        beside: tuple[Node, ast.Expr] | None,
        scope: Scope,
        ctx: Context,
        relation: str,
        *,
        grouped: bool,
    ) -> Node:
        """A term that writes the ``identity``, read with the identity of the size of the
        matrix ``beside`` it, as written; beside no matrix, I is a symbol and
        ``\\mathbf{1}`` the number 1, as before."""
        if beside is None:
            return self._term(term, scope, ctx, grouped=grouped)
        matrix, written = beside[0], beside[1].span.text
        bold = _bold_one(identity)
        one = identity.span.text
        reading = f"the identity of the size of {written}"
        if bold:
            reading += ", or the vector of ones beside a vector"
        note = f"{one} is read as {reading}, as it is {relation} it; to name it, define {one}"
        self._note(note)
        self._decide(
            MATRIX,
            identity.span,
            reading,
            f"nothing defines {one}, and it is {relation} the matrix {written}",
            note,
        )
        attributes = (("ones", bold),)
        saved = self._identity
        self._identity = self._call("linalg.identity", [matrix], VALUE, term.span, attributes)
        try:
            return self._term(term, scope, ctx, grouped=grouped)
        finally:
            self._identity = saved

    def _times_identity(
        self, value: Node, beside: tuple[Node, ast.Expr], term: ast.Expr, relation: str
    ) -> Node:
        """A number added to a square matrix, or equated to one: that multiple of the identity
        (decision 12)."""
        matrix, written = beside[0], beside[1].span.text
        note = (
            f"{term.span.text} is {relation} the matrix {written}, so it is read as "
            f"{term.span.text} times the identity"
        )
        self._note(note)
        self._decide(
            MATRIX,
            term.span,
            f"{term.span.text} times the identity",
            f"it is a number {relation} the matrix {written}",
            note,
        )
        identity = self._call("linalg.identity", [matrix], VALUE, term.span, (("ones", False),))
        return self.graph.add(ops.Mul(), (value, identity), VALUE, term.span)

    def _check_unknown_matrix(self, name: str, args: Sequence[Node], span: Span) -> None:
        """Refuse ``solve`` of an equation of matrices for an unknown that multiplies a
        matrix, as P in ``A^T P + P A = -I``: nothing defines it, so it is a number, where the
        book means a matrix, which is not supported yet."""
        if name != "solve" or len(args) < 2 or not isinstance(args[0].op, ops.Equation):
            return
        if not any(_a_matrix(side) for side in args[0].args):
            return
        unknowns = {symbol for arg in args[1:] for symbol in _symbols_of(arg)}
        for node in dependencies(list(args[0].args)):
            if not (isinstance(node.op, ops.Mul) and any(_a_matrix(arg) for arg in node.args)):
                continue
            for arg in node.args:
                symbol = arg.op.value if isinstance(arg.op, ops.Const) else None
                if isinstance(symbol, sympy.Symbol) and symbol in unknowns:
                    raise CompileError(
                        f"{symbol} multiplies a matrix in the equation solved for it, and has "
                        "no value, so it is read as a number, not as the matrix it is meant to be",
                        span=span,
                        notes=["an unknown matrix is not supported yet"],
                        hint=f"write {symbol} with its entries, as {symbol} := "
                        "\\begin{pmatrix} p & q \\\\ q & r \\end{pmatrix}, and solve for "
                        "them: solve(..., [p, q, r])",
                    )

    # -- products ------------------------------------------------------------------------

    def _kronecker(self, node: ast.Binary, scope: Scope, ctx: Context) -> Node | None:
        """``A \\otimes B`` of two matrices: their Kronecker product; beside a matrix, I or
        𝟙, which nothing defines, is the identity of its size, as in ``\\sigma_x \\otimes I``
        (decision 12). A matrix and what is no matrix are refused; between anything else,
        as tensors written with their indices, ``\\otimes`` is a product, as before."""
        if _joined_by(node) not in ("\\otimes", "⊗"):
            return None
        sides = (node.left, node.right)
        identities = [self._identity_alone(side, scope) for side in sides]
        try:
            values = [
                None if identity else self.value(side, scope, ctx)
                for side, identity in zip(sides, identities, strict=True)
            ]
        except RelativisticPyError:  # an operator, as \partial_\mu \otimes dx^\nu
            return None
        matrices = [
            (value, side)
            for value, side in zip(values, sides, strict=True)
            if value is not None and _a_matrix(value)
        ]
        if not matrices:
            return None
        met = matrices[0]
        factors: list[Node] = []
        for value, side in zip(values, sides, strict=True):
            if value is None:
                factors.append(self._identity_of(side, met, "it meets it in a Kronecker product"))
            elif _a_matrix(value):
                factors.append(value)
            else:
                raise CompileError(
                    f"{node.span.text} is read as a Kronecker product, which is of two "
                    f"matrices, and {side.span.text} is no matrix",
                    span=node.span,
                    hint="for the identity beside a matrix, write I; for a product, write "
                    "the factors side by side",
                )
        why = f"{node.left.span.text} and {node.right.span.text} are matrices"
        self._decide(MATRIX, node.span, "their Kronecker product, kron", why)
        return self._of_a_matrix("kron", factors, node.span)

    def _identity_alone(self, written: ast.Expr, scope: Scope) -> bool:
        """Whether what is written is I, 𝟙 or ``\\mathbb{1}`` alone, which nothing defines."""
        return (
            isinstance(written, ast.Name)
            and written.id in _IDENTITIES
            and scope.lookup(written.id) is None
        )

    def _identity_of(self, identity: ast.Expr, beside: tuple[Node, ast.Expr], how: str) -> Node:
        """The identity of the size of the matrix ``beside`` it, which I or 𝟙 is, as ``how``
        says it meets the matrix, said in a note (decision 12)."""
        matrix, written = beside[0], beside[1].span.text
        one = identity.span.text
        reading = f"the identity of the size of {written}"
        note = f"{one} is read as {reading}, as {how}; to name it, define {one}"
        self._note(note)
        why = f"nothing defines {one}, and {how}"
        self._decide(MATRIX, identity.span, reading, why, note)
        attributes = (("ones", False),)
        return self._call("linalg.identity", [matrix], VALUE, identity.span, attributes)

    def _identities_in_a_product(
        self, factors: Sequence[ast.Expr | Node], scope: Scope, ctx: Context
    ) -> list[ast.Expr | Node]:
        """The factors of a product with I or 𝟙, which nothing defines, beside a matrix:
        the identity that meets the matrix, which multiplying leaves as it is, so that it is
        left out, said in a note (decision 12), as of ``I \\sigma_x`` and ``I|0\\rangle``.
        Beside no matrix, I is a symbol, as before; in a term of a sum that writes it, it is
        read there."""
        read = list(factors)
        if self._identity is not None:
            return read
        identities = [
            position
            for position, factor in enumerate(factors)
            if isinstance(factor, ast.Expr) and self._identity_alone(factor, scope)
        ]
        if not identities or len(identities) == len(factors):
            return read
        met = next(
            (
                other
                for position, other in enumerate(factors)
                if position not in identities
                and (value := self._factor_value(other, scope, ctx)) is not None
                and _a_matrix(value)
            ),
            None,
        )
        if met is None:
            return read
        written = met.span.text if isinstance(met, ast.Expr) else "the matrix"
        for position in identities:
            identity = factors[position]
            assert isinstance(identity, ast.Expr)
            one = identity.span.text
            note = (
                f"{one} is read as the identity, as it multiplies the matrix {written}; to "
                f"name it, define {one}"
            )
            self._note(note)
            why = f"nothing defines {one}, and it multiplies the matrix {written}"
            self._decide(MATRIX, identity.span, "the identity", why, note)
        return [factor for position, factor in enumerate(read) if position not in identities]

    def _kronecker_written(self, node: ast.Binary, scope: Scope, ctx: Context) -> bool:
        """Whether a product written with ``\\otimes`` is of two matrices, and so a factor
        of its own in a longer product, as ``A \\otimes B`` is in ``(A \\otimes B) C``."""
        return self._kronecker(node, scope, ctx) is not None

    def _direct_sum(self, node: ast.Binary, scope: Scope, ctx: Context) -> Node:
        """``A \\oplus B`` of matrices: their direct sum, the block diagonal matrix of them;
        of anything else, refused."""
        left, right = (self.value(side, scope, ctx) for side in (node.left, node.right))
        if not (_a_matrix(left) and _a_matrix(right)):
            raise CompileError(
                f"{node.span.text} is a direct sum, which is of matrices",
                span=node.span,
                notes=["the direct sum of representations or spaces is not supported yet"],
                hint="write the matrices out, as \\begin{pmatrix} 1 & 0 \\\\ 0 & 1 \\end{pmatrix} "
                "\\oplus \\begin{pmatrix} 1 \\end{pmatrix}",
            )
        why = f"{node.left.span.text} and {node.right.span.text} are matrices"
        self._decide(MATRIX, node.span, "their direct sum, direct_sum", why)
        return self._of_a_matrix("direct_sum", [left, right], node.span)

    def _ones_vectors(
        self, factors: Sequence[ast.Expr | Node], scope: Scope, ctx: Context
    ) -> list[ast.Expr | Node]:
        """The factors of a product with ``\\mathbf{1}``, which nothing defines, read as the
        vector of ones of the size of the matrix it meets (decision 23): in ``A \\mathbf{1}``
        the column of as many ones as A has columns, and in ``\\mathbf{1}^T A`` the row of as
        many as A has rows. Beside no matrix, it is the number 1, as before, and where the
        program defines it, its own."""
        read: list[ast.Expr | Node] = list(factors)
        if scope.lookup(BOLD_ONE) is not None:
            return read
        for position, factor in enumerate(factors):
            if not isinstance(factor, ast.Expr):
                continue
            row = isinstance(factor, ast.Scripted) and _bold_one(factor.base)
            if row:
                assert isinstance(factor, ast.Scripted)
                script = factor.scripts[0] if len(factor.scripts) == 1 else None
                mark = script.body.id if script and isinstance(script.body, ast.Name) else None
                if mark not in ("T", "top", "intercal") or scope.lookup("T") is not None:
                    continue
            elif not _bold_one(factor):
                continue
            beside = factors[position + 1 :] if row else factors[:position][::-1]
            met = next(
                (
                    (value, other)
                    for other in beside
                    if (value := self._factor_value(other, scope, ctx)) is not None
                    and _a_matrix(value)
                ),
                None,
            )
            if met is None:
                continue
            matrix, other = met
            side = "rows" if row else "columns"
            written = other.span.text if isinstance(other, ast.Expr) else "the matrix"
            shown = factor.span.text
            what = "the row of ones" if row else "the vector of ones"
            reading = f"{what} as long as {written} has {side}"
            note = f"{shown} is read as {reading}; to name it, define it"
            self._note(note)
            self._decide(
                MATRIX,
                factor.span,
                reading,
                f"nothing defines it, and it meets the matrix {written}",
                note,
            )
            attributes = (("side", side), ("row", row))
            read[position] = self._call("linalg.ones", [matrix], VALUE, factor.span, attributes)
        return read

    def _factor_value(self, factor: ast.Expr | Node, scope: Scope, ctx: Context) -> Node | None:
        """The value of a factor, if it is one, and nothing that acts on what follows it."""
        if isinstance(factor, Node):
            return factor
        if _bold_one(factor) or (isinstance(factor, ast.Scripted) and _bold_one(factor.base)):
            return None
        try:
            term = self.term(_operand(factor), scope, ctx)
        except RelativisticPyError:
            return None
        return term if isinstance(term, Node) else None

    def _one_by_one(self, product: Node, span: Span) -> Node:
        """A product of matrices that is a 1 × 1 matrix, as a row times a matrix times a
        column, ``\\mathbf{x}^T L \\mathbf{x}``, is: the number it holds, as books read it,
        said in a note. Any other value is as it is."""
        if product.type != VALUE or not isinstance(product.op, ops.Mul):
            return product
        if _matrix_shape(product) != (1, 1):
            return product
        matrices = [arg for arg in product.args if _a_matrix(arg)]
        if len(matrices) < 2:
            return product
        note = f"{span.text} is a 1 × 1 matrix, so it is read as the number it holds"
        self._note(note)
        why = "it is a product of matrices whose value has one row and one column"
        self._decide(MATRIX, span, "the number in the 1 × 1 matrix", why, note)
        zero = self.graph.const(0, VALUE)
        return self._call("linalg.entry", [product, zero, zero], SCALAR, span)


def _symbols_of(node: Node) -> set[Any]:
    """The symbols of ``solve``'s unknowns, a name or a list of them."""
    if isinstance(node.op, ops.Const):
        return set(getattr(node.op.value, "free_symbols", ()))
    return {symbol for arg in node.args for symbol in _symbols_of(arg)}


def _known(node: Node) -> Any:
    """The value of a scalar known when compiling, as 1/2 is, made of what is written by
    arithmetic; ``None`` for a value computed."""
    if isinstance(node.op, ops.Const):
        return node.op.value
    if node.type != SCALAR or not isinstance(node.op, ops.Mul | ops.Pow | ops.Add):
        return None
    values = [_known(arg) for arg in node.args]
    if any(value is None for value in values):
        return None
    if isinstance(node.op, ops.Mul):
        return sympy.Mul(*values)
    if isinstance(node.op, ops.Pow):
        return sympy.Pow(*values)
    return sympy.Add(*(sign * value for sign, value in zip(node.op.signs, values, strict=True)))


def _double_bars(node: ast.Abs) -> bool:
    """Whether bars are double, as a norm is written: ``\\|x\\|``, ``\\lVert x \\rVert``,
    ``‖x‖``, and ``||x||`` in plain text, two bars side by side around what they hold, not
    the absolute value of one in another, as ``||x| - |y||`` is."""
    if node.span.text.startswith("||"):
        return node.operand.span.start == node.span.start + 2
    return node.span.text.lstrip().startswith(
        ("\\|", "\\left\\|", "‖", "\\lVert", "\\Vert", "\\left\\lVert", "\\left\\Vert")
    )
