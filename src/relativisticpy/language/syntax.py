"""The syntax tree produced by the parser.

The tree records what was *written*, not what it *means*. Where notation is
ambiguous the tree keeps every reading and leaves the choice to the compiler,
which knows what each name refers to. The main example is a script: ``x^{a}``
is a :class:`Script` holding both an expression (``x`` to the power ``a``) and
an :class:`IndexList` (the contravariant component ``a`` of a vector ``x``).

Nodes are immutable. Each carries the :class:`~relativisticpy.language.source.Span`
it was parsed from, and spans are ignored when nodes are compared, so two trees
are equal when they have the same structure.

:func:`to_sexpr` renders any node as a compact S-expression, which is how the
parser's tests state what a piece of notation means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from relativisticpy.errors import ParseError
from relativisticpy.language.source import Span

# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Node:
    """Base class of every syntax-tree node."""

    span: Span = field(kw_only=True, compare=False, repr=False)


class Expr(Node):
    """A node that denotes a value."""


# ---------------------------------------------------------------------------
# Atoms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Number(Expr):
    """A numeric literal, kept as written: ``2``, ``2.5``, ``6.674e-11``."""

    text: str


@dataclass(frozen=True)
class String(Expr):
    value: str


@dataclass(frozen=True)
class Name(Expr):
    """A name: ``x``, ``theta`` (also written ``\\theta`` or ``θ``), ``Ric``, ``eq0``.

    ``verbatim`` is set for names written inside ``\\mathrm{...}`` or ``\\text{...}``.
    """

    id: str
    verbatim: bool = False


@dataclass(frozen=True)
class Operator(Expr):
    """A differential operator symbol: ``\\partial`` (``∂``), ``\\nabla`` (``∇``), the
    d'Alembertian ``\\Box`` (``□``) or the Lie derivative ``\\pounds`` (``£``)."""

    kind: Literal["partial", "nabla", "box", "lie"]


@dataclass(frozen=True)
class Dots(Expr):
    """An ellipsis, ``\\cdots``: terms that are written only as a pattern."""


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Unary(Expr):
    """A prefix operation. ``*`` is the Hodge dual, as in ``*F``."""

    op: Literal["-", "+", "not", "*", "±", "∓"]
    operand: Expr


@dataclass(frozen=True)
class Binary(Expr):
    """A binary operation. ``implicit`` marks multiplication by juxtaposition (``2x``), and
    ``mod`` a remainder or the modulus of a congruence (``a \\bmod m``, ``b \\pmod{m}``)."""

    op: Literal["+", "-", "*", "/", "**", "and", "or", "∧", "±", "∓", "∘", "⊙", "⊕", "mod"]
    left: Expr
    right: Expr
    implicit: bool = False


@dataclass(frozen=True)
class CircularConvolution(Expr):
    """``x \\circledast h``, the circular convolution of two sequences, of the ``length`` its
    subscript writes, as in ``x \\circledast_N h``, or else, where none is written, of theirs."""

    left: Expr
    right: Expr
    length: Expr | None = None


@dataclass(frozen=True)
class Comparison(Expr):
    """One or more chained comparisons: ``x > 0`` or ``0 < x <= 1``."""

    operands: tuple[Expr, ...]
    ops: tuple[Literal["<", ">", "<=", ">=", "==", "!="], ...]


@dataclass(frozen=True)
class Equation(Expr):
    """``lhs = rhs``. Whether this defines ``lhs`` or states an equation is the compiler's call."""

    lhs: Expr
    rhs: Expr


@dataclass(frozen=True)
class Relation(Expr):
    """A relation other than equality or order, such as ``a \\sim b``, ``a \\propto b``,
    ``x \\in X`` or ``x \\to x'``. RelativisticPy reads relations but does not compute them.

    ``op`` is the relation's symbol, one for each relation however it is written, and
    ``written`` the relation as it was written, ``\\propto`` or ``∝``, for messages. It is
    empty for a relation the parser makes, as it makes ``∈`` of ``\\mu, \\nu = 0, 1``."""

    lhs: Expr
    op: str
    rhs: Expr
    written: str = field(default="", compare=False)


@dataclass(frozen=True)
class Chain(Expr):
    """Relations in a row, as in a derivation: ``a = b = c``, ``a \\approx b = c`` or
    ``a = b \\geq c``. Each link is a statement of its own. ``written`` is each relation as
    it was written."""

    operands: tuple[Expr, ...]
    relations: tuple[str, ...]
    written: tuple[str, ...] = field(default=(), compare=False)

    def links(self) -> list[Equation | Relation | Comparison]:
        """Each relation with its two operands: ``a = b`` and ``b = c``; ``b \\geq c`` a
        comparison."""
        links: list[Equation | Relation | Comparison] = []
        written = self.written or ("",) * len(self.relations)
        for left, op, right, spelling in zip(
            self.operands, self.relations, self.operands[1:], written, strict=False
        ):
            span = left.span.to(right.span)
            if op == "=":
                links.append(Equation(left, right, span=span))
            elif op in COMPARISONS:
                links.append(Comparison((left, right), (COMPARISONS[op],), span=span))
            else:
                links.append(Relation(left, op, right, spelling, span=span))
        return links


#: The comparisons, by their symbol, as a chain of relations holds them.
COMPARISONS: dict[str, Literal["<", ">", "<=", ">=", "==", "!="]] = {
    "<": "<",
    ">": ">",
    "<=": "<=",
    ">=": ">=",
    "==": "==",
    "!=": "!=",
}


@dataclass(frozen=True)
class Factorial(Expr):
    operand: Expr


@dataclass(frozen=True)
class Abs(Expr):
    operand: Expr


@dataclass(frozen=True)
class Call(Expr):
    """A function applied to arguments: ``f(x)``, ``f'(x)``, ``\\sin x``."""

    func: Expr
    args: tuple[Expr, ...]


@dataclass(frozen=True)
class Prime(Expr):
    """Lagrange's derivative notation, ``f'`` or ``f''``; normally the target of a call."""

    func: Name
    order: int


@dataclass(frozen=True)
class Primed(Expr):
    """Primes after brackets, ``(x^3)'``: the derivative of what the brackets hold, by the
    one variable it is written with, as ``f'`` is of a function of one variable."""

    operand: Expr
    order: int


@dataclass(frozen=True)
class Accent(Expr):
    """An accented symbol: ``\\dot{a}``, ``\\bar{x}``, ``\\hat{n}``.

    ``accent`` names the decoration: ``dot``, ``bar``, ``hat`` and so on.
    """

    accent: str
    base: Expr


@dataclass(frozen=True)
class Substitution(Expr):
    """An evaluation bar that substitutes: ``expr |_{x = a, y = b}``, or
    ``\\left. expr \\right|_{x = a, y = b}``, every substitution at once."""

    expr: Expr
    bindings: tuple[tuple[Expr, Expr], ...]


@dataclass(frozen=True)
class Evaluation(Expr):
    """An evaluation bar or bracket, as books write a function's value at a point or the
    difference of its values: ``\\left. F \\right|_{x=a}`` is F at a, with no ``upper``;
    ``\\left. F \\right|_a^b``, ``\\left[F\\right]_a^b`` and ``[F]_{x=a}^{x=b}`` are
    F(b) − F(a). ``variable`` is what is written with the limits, as x of ``_{x=a}``, or
    ``None`` where nothing is, for the reader to find."""

    expr: Expr
    lower: Expr
    upper: Expr | None
    variable: Expr | None


# ---------------------------------------------------------------------------
# Dirac's notation, and angle brackets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ket(Expr):
    """A ket, ``|\\psi\\rangle``. Its ``label`` is what is written between the bar and the
    angle bracket: a name, a number, a sign alone as in ``|+\\rangle``, an expression as in
    ``|n - 1\\rangle``, or several, ``|j, m\\rangle``, a tuple, whose groups a semicolon
    separates, as Clebsch–Gordan's ``|j_1, j_2; j, m\\rangle`` is a tuple of tuples."""

    label: Expr


@dataclass(frozen=True)
class Bra(Expr):
    """A bra, ``\\langle\\phi|``, with a label as a ket's."""

    label: Expr


@dataclass(frozen=True)
class BraKet(Expr):
    """A bracket of a bra and a ket, ``\\langle\\phi|\\psi\\rangle``, or a matrix element,
    ``\\langle\\phi|A|\\psi\\rangle``, with the ``operator`` between their bars. ``bra`` and
    ``ket`` are their labels."""

    bra: Expr
    operator: Expr | None
    ket: Expr


@dataclass(frozen=True)
class Expectation(Expr):
    """Angle brackets around what holds no bar: ``\\langle x \\rangle``, an expectation value,
    or ``\\langle u, v \\rangle``, an inner product, or a bracket a program defines. Whatever
    it is, the reader decides."""

    operand: Expr


# ---------------------------------------------------------------------------
# Scripts and indices
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexItem(Node):
    """One slot of an index list: a label (``a``), a label with a value (``a=0``,
    ``mu:t``) or a bare value (``0``).

    ``derivative`` marks slots written after a comma (partial derivative, as in
    ``T_{\\mu\\nu,\\rho}``) or a semicolon (covariant derivative, ``V^{a}{}_{;b}``).
    """

    label: str | None
    value: Expr | None = None
    derivative: Literal[",", ";"] | None = None
    opens: Literal["(", "["] | None = None
    closes: Literal[")", "]"] | None = None


@dataclass(frozen=True)
class IndexList(Node):
    """The contents of a script read as tensor indices: ``{\\mu\\nu}``, ``{a=0 b}``.

    Index labels follow LaTeX: every letter is its own index, so ``{ab}`` is
    ``a`` and ``b``, while Greek names stay whole (``{munu}`` is ``mu`` and ``nu``).

    Round brackets symmetrise and square brackets antisymmetrise the enclosed
    indices: ``T_{(ab)}``, ``F_{[\\mu\\nu]}``. Each item records whether it opens
    or closes a bracket, because a bracket may span several tensors, as in the
    Bianchi identity ``\\nabla_{[\\lambda} R_{\\rho\\sigma]\\mu\\nu}``.
    """

    items: tuple[IndexItem, ...]


@dataclass(frozen=True)
class Script(Node):
    """A superscript (``^``) or subscript (``_``).

    ``body`` is the script read as an expression and ``indices`` the script read
    as an index list; whichever reading is impossible is ``None``. ``^{2}`` has
    both, ``^{-1}`` only a body and ``_{a=0}`` only indices. When only the body
    exists, ``index_error`` explains what stopped the index reading.
    """

    kind: Literal["^", "_"]
    body: Expr | None
    indices: IndexList | None
    #: Why the script cannot be read as indices, when it can only be read as an expression.
    index_error: ParseError | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class Scripted(Expr):
    """A base with scripts, in the order written: ``x^2``, ``R^{a}{}_{bcd}``."""

    base: Expr
    scripts: tuple[Script, ...]


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class List(Expr):
    """``[a, b, c]``."""

    items: tuple[Expr, ...]


@dataclass(frozen=True)
class Tuple(Expr):
    """``(a, b, c)``, as in ``integrate(f, (x, 0, 1))``."""

    items: tuple[Expr, ...]


@dataclass(frozen=True)
class Matrix(Expr):
    """A LaTeX matrix environment; ``environment`` is its name (``pmatrix``, ``vmatrix``, ...)."""

    environment: str
    rows: tuple[tuple[Expr, ...], ...]


@dataclass(frozen=True)
class Piecewise(Expr):
    """``\\begin{cases}`` or ``{ x**2, x > 0; -x, x <= 0 }``. A ``None`` condition means
    *otherwise*."""

    pieces: tuple[tuple[Expr, Expr | None], ...]


# ---------------------------------------------------------------------------
# Calculus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Root(Expr):
    """``\\sqrt{x}`` or ``\\sqrt[n]{x}``."""

    radicand: Expr
    degree: Expr | None = None


@dataclass(frozen=True)
class Written:
    """What a node was written as, where that is a reading of its own: the quotient
    ``dy/dx`` of a derivative written with a solidus, which is what it means where the
    program defines one of ``names``, as dx. It is no node of the tree, so that no walk of
    the tree meets the names written in it twice, and trees compare without it."""

    expr: Expr
    names: tuple[str, ...]


@dataclass(frozen=True)
class Derivative(Expr):
    """A derivative in Leibniz notation: ``\\frac{d^2 f}{dx^2}``, ``\\pdv{f}{x}{y}``.

    ``variables`` pairs each variable with its order. When ``expr`` is ``None``
    the derivative is an operator, as in ``\\frac{d}{dx}``, which acts on the
    factor that follows it. ``written`` is, for one written with a solidus, as ``dy/dx``
    is in a line of text, the quotient as written.
    """

    expr: Expr | None
    variables: tuple[tuple[Expr, int], ...]
    partial: bool
    written: Written | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class Differential(Expr):
    """``d{f}`` or ``d^n{f}``: the plain-text differential, as in ``d{f(x)}/d{x}``."""

    expr: Expr
    order: int = 1


@dataclass(frozen=True)
class Bound(Node):
    """An integration variable and its optional limits."""

    variable: Expr
    lower: Expr | None = None
    upper: Expr | None = None
    dimension: Expr | None = None  # d^4x integrates over four dimensions


@dataclass(frozen=True)
class Integral(Expr):
    """``\\int_a^b f\\,dx``. Multiple integrals have one bound per differential."""

    integrand: Expr
    bounds: tuple[Bound, ...]
    domain: Expr | None = None  # \\int_M \\omega: a form integrated over a region


@dataclass(frozen=True)
class BigOperator(Expr):
    """``\\sum_{n=0}^{N} body`` or ``\\prod_{n=0}^{N} body``."""

    op: Literal["sum", "prod"]
    variable: Expr
    lower: Expr | None  # None for a sum over an index, as in \\sum_b
    upper: Expr | None
    body: Expr


@dataclass(frozen=True)
class Limit(Expr):
    """``\\lim_{x \\to a} body``; ``direction`` is ``+`` or ``-`` for one-sided limits."""

    variable: Expr
    point: Expr
    body: Expr
    direction: Literal["+", "-"] | None = None


@dataclass(frozen=True)
class Optimum(Expr):
    """The greatest or the least value of ``body`` over its ``variables``, as
    ``\\max_{x} f(x)`` and ``\\inf_{x \\in S} f(x)`` write it, or where it is, as
    ``\\arg\\min_{w} f(w)`` does. ``constraints`` are the conditions written beneath the
    operator, as the ``0 \\le x \\le 3`` of ``\\max_{0 \\le x \\le 3}``, and after it, after
    "subject to": comparisons, equations and memberships, ``x \\in S``. A condition on
    several names at once, as ``x_1, x_2 \\geq 0``, has them as a tuple."""

    op: Literal["max", "min", "argmax", "argmin", "sup", "inf"]
    variables: tuple[Expr, ...]
    body: Expr
    constraints: tuple[Expr, ...] = ()


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Statement(Node):
    """Base class of statements.

    ``latex`` records whether the statement was written in LaTeX (it uses a
    LaTeX command or Unicode mathematics). In LaTeX, as in typeset maths, a run
    of letters such as ``GM`` is a product of symbols; in plain text it is one
    name.
    """

    latex: bool = field(default=False, kw_only=True)


@dataclass(frozen=True)
class Evaluate(Statement):
    """An expression to evaluate. Equations are evaluated too: ``ds^2 = ...``."""

    expr: Expr


@dataclass(frozen=True)
class Define(Statement):
    """``target := value``; several targets destructure: ``x, y := symbols('x y')``. The
    ``ranges`` written after it on the line give the values of its index labels:
    ``x_n := n h, \\quad n = 0, \\ldots, N-1``."""

    targets: tuple[Expr, ...]
    value: Expr
    ranges: tuple[Relation, ...] = field(default=(), kw_only=True)


@dataclass(frozen=True)
class Block(Node):
    """The body of a function written with ``->``: statements, the last one's value returned."""

    statements: tuple[Statement, ...]


@dataclass(frozen=True)
class FunctionDefinition(Statement):
    """``f(x, y) := body`` or ``f : (x, y) -> { statements }``."""

    name: str
    params: tuple[str, ...]
    body: Expr | Block


@dataclass(frozen=True)
class Program(Node):
    """A program's statements. Parsed with ``recover``, it also holds ``errors``: the
    error of each statement that could not be read, in the order of the source, while
    ``statements`` holds those that could. :meth:`in_order` gives both, as written."""

    statements: tuple[Statement, ...]
    errors: tuple[ParseError, ...] = field(default=(), compare=False)

    def in_order(self) -> list[Statement | ParseError]:
        """The statements and the errors of those that could not be read, in the order
        they are written, so that each can be run or reported in turn."""

        def start(item: Statement | ParseError) -> int:
            span = item.span
            return span.start if span is not None else 0

        return sorted([*self.statements, *self.errors], key=start)


# ---------------------------------------------------------------------------
# S-expressions
# ---------------------------------------------------------------------------


def to_sexpr(node: Node) -> str:
    """Render a node as a compact S-expression, e.g. ``2x^2 + 1`` → ``(+ (* 2 (^ x 2)) 1)``.

    Scripts print as ``^2`` or ``^(+ n 1)`` when read as an expression and as
    ``_[mu nu]`` when they can be read as indices. Operator-form derivatives
    print with ``_`` for the missing operand: ``(d/dx _)``.
    """
    return _Printer().render(node)


class _Printer:
    def render(self, node: Node | None) -> str:
        if node is None:
            return "_"
        method = getattr(self, f"_{type(node).__name__}")
        result: str = method(node)
        return result

    def _many(self, nodes: tuple[Node, ...] | list[Node]) -> str:
        return " ".join(self.render(n) for n in nodes)

    def _Number(self, node: Number) -> str:
        return node.text

    def _String(self, node: String) -> str:
        return repr(node.value)

    def _Name(self, node: Name) -> str:
        return node.id

    def _Operator(self, node: Operator) -> str:
        return {"partial": "∂", "nabla": "∇", "box": "□", "lie": "£"}[node.kind]

    def _Dots(self, node: Dots) -> str:
        return "⋯"

    def _Unary(self, node: Unary) -> str:
        return f"({node.op} {self.render(node.operand)})"

    def _Binary(self, node: Binary) -> str:
        return f"({node.op} {self.render(node.left)} {self.render(node.right)})"

    def _Comparison(self, node: Comparison) -> str:
        if len(node.ops) == 1:
            return f"({node.ops[0]} {self._many(node.operands)})"
        parts = [self.render(node.operands[0])]
        for op, operand in zip(node.ops, node.operands[1:], strict=True):
            parts += [op, self.render(operand)]
        return f"(chain {' '.join(parts)})"

    def _Equation(self, node: Equation) -> str:
        return f"(= {self.render(node.lhs)} {self.render(node.rhs)})"

    def _Relation(self, node: Relation) -> str:
        return f"({node.op} {self.render(node.lhs)} {self.render(node.rhs)})"

    def _Chain(self, node: Chain) -> str:
        parts = [self.render(node.operands[0])]
        for op, operand in zip(node.relations, node.operands[1:], strict=True):
            parts += [op, self.render(operand)]
        return f"(chain {' '.join(parts)})"

    def _Factorial(self, node: Factorial) -> str:
        return f"(! {self.render(node.operand)})"

    def _Abs(self, node: Abs) -> str:
        return f"(abs {self.render(node.operand)})"

    def _Call(self, node: Call) -> str:
        args = self._many(node.args)
        return f"(call {self.render(node.func)}{' ' + args if args else ''})"

    def _Prime(self, node: Prime) -> str:
        return node.func.id + "'" * node.order

    def _Primed(self, node: Primed) -> str:
        return f"({chr(39) * node.order} {self.render(node.operand)})"

    def _Accent(self, node: Accent) -> str:
        return f"({node.accent} {self.render(node.base)})"

    def _Substitution(self, node: Substitution) -> str:
        bindings = " ".join(f"({self.render(t)} {self.render(v)})" for t, v in node.bindings)
        return f"(at {self.render(node.expr)} {bindings})"

    def _CircularConvolution(self, node: CircularConvolution) -> str:
        length = f"_{self.render(node.length)}" if node.length is not None else ""
        return f"(⊛{length} {self.render(node.left)} {self.render(node.right)})"

    def _Evaluation(self, node: Evaluation) -> str:
        variable = self.render(node.variable) if node.variable is not None else "?"
        limits: list[Node] = [node.lower] if node.upper is None else [node.lower, node.upper]
        return f"(eval {self.render(node.expr)} ({variable} {self._many(limits)}))"

    def _Ket(self, node: Ket) -> str:
        return f"(ket {self.render(node.label)})"

    def _Bra(self, node: Bra) -> str:
        return f"(bra {self.render(node.label)})"

    def _BraKet(self, node: BraKet) -> str:
        operator = f" {self.render(node.operator)}" if node.operator is not None else ""
        return f"(braket {self.render(node.bra)}{operator} {self.render(node.ket)})"

    def _Expectation(self, node: Expectation) -> str:
        return f"(expectation {self.render(node.operand)})"

    def _IndexItem(self, node: IndexItem) -> str:
        if node.value is None:
            text = node.label or ""
        else:
            value = self.render(node.value)
            text = value if node.label is None else f"{node.label}={value}"
        # Square brackets would read as a list here, so antisymmetrisation prints as <...>.
        opening = {"(": "(", "[": "<", None: ""}[node.opens]
        closing = {")": ")", "]": ">", None: ""}[node.closes]
        return opening + text + closing

    def _IndexList(self, node: IndexList) -> str:
        rendered = [self.render(item) for item in node.items]
        marker = None
        for position, item in enumerate(node.items):
            if item.derivative != marker:
                marker = item.derivative
                rendered[position] = f"{marker} {rendered[position]}"
        return "[" + " ".join(rendered) + "]"

    def _Script(self, node: Script) -> str:
        prefer_indices = node.kind == "_" or node.body is None or not _is_simple(node.body)
        if node.indices is not None and prefer_indices:
            return node.kind + self.render(node.indices)
        body = node.body
        assert body is not None
        return node.kind + self.render(body)

    def _Scripted(self, node: Scripted) -> str:
        if len(node.scripts) == 1 and node.scripts[0].kind == "^":
            script = node.scripts[0]
            if script.body is not None and (script.indices is None or _is_simple(script.body)):
                return f"(^ {self.render(node.base)} {self.render(script.body)})"
        scripts = " ".join(self.render(s) for s in node.scripts)
        return f"({self.render(node.base)} {scripts})"

    def _List(self, node: List) -> str:
        return f"(list {self._many(node.items)})".replace("(list )", "(list)")

    def _Tuple(self, node: Tuple) -> str:
        return f"(tuple {self._many(node.items)})".replace("(tuple )", "(tuple)")

    def _Matrix(self, node: Matrix) -> str:
        rows = " ".join(f"({self._many(row)})" for row in node.rows)
        return f"({node.environment} {rows})"

    def _Piecewise(self, node: Piecewise) -> str:
        pieces = " ".join(
            f"({self.render(value)} if {self.render(condition) if condition else 'otherwise'})"
            for value, condition in node.pieces
        )
        return f"(piecewise {pieces})"

    def _Root(self, node: Root) -> str:
        if node.degree is None:
            return f"(sqrt {self.render(node.radicand)})"
        return f"(root {self.render(node.degree)} {self.render(node.radicand)})"

    def _Derivative(self, node: Derivative) -> str:
        d = "∂" if node.partial else "d"
        total = sum(order for _, order in node.variables)
        top = d if total == 1 else f"{d}^{total}"
        bottom = "".join(
            f"{d}{self.render(var)}" + (f"^{order}" if order > 1 else "")
            for var, order in node.variables
        )
        return f"({top}/{bottom} {self.render(node.expr)})"

    def _Differential(self, node: Differential) -> str:
        d = "d" if node.order == 1 else f"d^{node.order}"
        return f"({d} {self.render(node.expr)})"

    def _Bound(self, node: Bound) -> str:
        if node.dimension is not None:
            return f"(d^{self.render(node.dimension)} {self.render(node.variable)})"
        if node.lower is None and node.upper is None:
            return self.render(node.variable)
        return f"({self.render(node.variable)} {self.render(node.lower)} {self.render(node.upper)})"

    def _Integral(self, node: Integral) -> str:
        if node.domain is not None:
            measure = f" {self._many(node.bounds)}" if node.bounds else ""
            return f"(int {self.render(node.integrand)}{measure} over {self.render(node.domain)})"
        return f"(int {self.render(node.integrand)} {self._many(node.bounds)})"

    def _BigOperator(self, node: BigOperator) -> str:
        bounds = (
            f"({self.render(node.variable)} {self.render(node.lower)} {self.render(node.upper)})"
            if node.lower is not None
            else self.render(node.variable)
        )
        return f"({node.op} {bounds} {self.render(node.body)})"

    def _Limit(self, node: Limit) -> str:
        point = self.render(node.point) + (node.direction or "")
        return f"(lim {self.render(node.variable)} {point} {self.render(node.body)})"

    def _Optimum(self, node: Optimum) -> str:
        constraints = "".join(f" {self.render(constraint)}" for constraint in node.constraints)
        return f"({node.op} ({self._many(node.variables)}) {self.render(node.body)}{constraints})"

    def _Evaluate(self, node: Evaluate) -> str:
        return self.render(node.expr)

    def _Define(self, node: Define) -> str:
        ranges = "".join(f" {self.render(relation)}" for relation in node.ranges)
        return f"(:= {self._many(node.targets)} {self.render(node.value)}{ranges})"

    def _Block(self, node: Block) -> str:
        return f"(block {self._many(node.statements)})"

    def _FunctionDefinition(self, node: FunctionDefinition) -> str:
        return f"(def {node.name} ({' '.join(node.params)}) {self.render(node.body)})"

    def _Program(self, node: Program) -> str:
        return "\n".join(self.render(statement) for statement in node.statements)


def _is_simple(node: Expr) -> bool:
    """A number: the script that is most naturally an exponent."""
    return isinstance(node, Number)
