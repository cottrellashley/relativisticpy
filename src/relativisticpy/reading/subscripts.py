r"""What a name with a subscript means, and a name with brackets right after it: the one
resolver, which every layer asks (``design/notation.md``, decisions 5, 17, 31, 34 and 35).

``c_1``, ``x_i``, ``f_x``, ``H_n``, ``X_k``, ``T_{\mu\nu}`` and ``x[n]`` are written alike,
and are read by what their base is. The readings of ``base_sub`` are tried in this order,
and the first that applies is taken. A user's definition always wins, so the readings of
what the program defines come before those of what is built in:

1. a name the program defines under the name as written, as ``c_1`` after ``c_1 := 5``;
2. an index of a tensor the program defines, braced or not, as t of ``y_t`` over data;
   within it, a subscript that names a coordinate is the component by that coordinate, as
   ``g_{tt}``, which the typer decides index by index;
3. an entry of a matrix, by its row and its column, counted from 1, as ``A_{12}``;
4. a member of a family, as ``c_1``, ``c_{2k}`` and ``a_{n+1}`` after ``c_n := \ldots``,
   and of a recurrence, computed from the members given;
5. an element of a list, counted from the lower limit of the sum that ranges over the
   whole list (decision 17), and refused anywhere else, where its positions have no start;
6. a member of a sample, as ``X_i`` after ``X_1, \ldots, X_n \sim \ldots``;
7. a sequence at a point, as ``x_n`` after ``x[n] := \ldots``;
8. a partial derivative, as ``f_x`` of a function f of x the program defines;
9. an index of a built-in tensor in force, as those of ``R_{01}`` with a metric;
10. a book name of an integer, as the harmonic number ``H_n`` of an n assumed an integer;
11. a label, part of the name, whatever the base: ``\lambda_1``, ``r_s``, ``M_\odot``.

Whether a subscript picks a member is decided on the subscript as the parser read it, a
number, a name or an expression, never on its text. A name whose subscript is a label that
has a value where it is read, as the label of a family whose member is being made has, is
the name at that value, read again: so ``\sigma_a`` is ``\sigma_1`` in ``T_1`` of
``T_a := \sigma_a/2``.

``base[point]`` is read in this order: a sequence the program defines, or a function of
one variable, at the point; an element of a list, as its subscript is, in a sum over all
of the list, and refused anywhere else; an unknown sequence of the equation being read;
a member of a family of the letter, as ``x[n]`` after ``x_n := \ldots``; an operator of
random variables, as the expectation ``E[X]``; the unit impulse ``\delta[n]`` or the unit
step ``u[n]``, at an integer; and otherwise as it was read before, a product in LaTeX and
Python's indexing, which is refused, in plain text. A sequence written with its point as
a subscript, ``x_n`` after ``x[n] := \ldots``, is the sequence at the point, so that a
sequence and a family of one letter are read alike however they are written.

Each reading is found by a test here, in :class:`Subscripts`, and lowered by the layer it
belongs to: the layers hold no test of their own.
"""

from __future__ import annotations

import contextlib
import enum
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import sympy

from relativisticpy.compiler.builtins import CONSTANTS
from relativisticpy.compiler.passes import Term
from relativisticpy.compiler.scope import (
    Context,
    FunctionBinding,
    Scope,
    TensorBinding,
    VariableBinding,
)
from relativisticpy.compiler.session import Session
from relativisticpy.errors import CompileError, RelativisticPyError
from relativisticpy.integrations import _forms as forms
from relativisticpy.ir import SCALAR, Node, TensorType, ops
from relativisticpy.language import syntax as ast
from relativisticpy.language.parser import python_indexing, split_index_labels
from relativisticpy.language.source import Span
from relativisticpy.language.vocabulary import GREEK
from relativisticpy.reading import probability, sequences
from relativisticpy.reading.matrices import Matrices
from relativisticpy.reading.names import (
    _a_matrix,
    _bracketed_order,
    _elements,
    _mentions_symbol,
    _shown,
    accented,
)
from relativisticpy.reading.reader import BookCall, BookReading
from relativisticpy.reading.sequences import Response
from relativisticpy.reading.sequences import Sequence as SequenceDefinition
from relativisticpy.typing.syntax import _indexed_name, _plain_label


class Kind(enum.Enum):
    """A reading of a name with a subscript, or of a name with brackets after it; each
    says what the form is read as."""

    DEFINED = "a name the program defines"
    INDEX = "a tensor the program defines, with its index"
    ENTRY = "an entry of a matrix"
    MEMBER = "a member of a family"
    RECURRENCE = "a member of a recurrence"
    ELEMENT = "an element of a list"
    SAMPLE = "a member of a sample"
    SEQUENCE = "a sequence at a point"
    PARTIAL = "a partial derivative"
    BUILTIN = "a built-in tensor, with its index"
    BOOK = "a book name of an integer"
    LABEL = "a name of its own"
    OPERATOR = "an operator of random variables"
    PRODUCT = "a product"


#: The readings of a name with a subscript, in the order they are tried. A member of a
#: recurrence is a member of a family, found by the same test.
PRECEDENCE = (
    Kind.DEFINED,
    Kind.INDEX,
    Kind.ENTRY,
    Kind.MEMBER,
    Kind.ELEMENT,
    Kind.SAMPLE,
    Kind.SEQUENCE,
    Kind.PARTIAL,
    Kind.BUILTIN,
    Kind.BOOK,
    Kind.LABEL,
)

#: The readings of a name with brackets right after it, in the order they are tried.
BRACKETS = (
    Kind.SEQUENCE,
    Kind.ELEMENT,
    Kind.MEMBER,
    Kind.OPERATOR,
    Kind.BOOK,
    Kind.PRODUCT,
)

#: The capabilities signal processing writes with a letter and a point in brackets.
BRACKET_BOOKS = {"delta": "impulse", "u": "step"}

#: The name a sample is bound under, by the letter its members are written with, which no
#: name written can be.
SAMPLE = "{}_…"

#: The commands written as subscripts that label a name and never index it, as the Sun's
#: \odot does in M_\odot.
_MARKS = frozenset(
    {
        "odot", "oplus", "earth", "sun", "star", "ast", "bullet", "circ", "times", "pm", "mp",
        "dagger", "ddagger", "infty", "parallel", "perp", "prime", "max", "min",
    }
)  # fmt: skip

#: How a script writes a word, as \rm does in T_{\rm eff}.
_WORD = re.compile(r"\\(?:rm|mathrm|text[a-z]*|mbox|operatorname)(?![A-Za-z])")


# -- what is written ------------------------------------------------------------------------


@dataclass(frozen=True)
class Family(ast.FunctionDefinition):
    """A family defined with a free label, as ``c_n := 1/n^2`` or ``\\psi_n(x) := ...``:
    the function of its label, then of its parameters, that each of its members is, bound
    under the family's name as written, ``c_n``. ``base`` is what its members are written
    with, c; ``fixed`` the names its value writes with the base at a subscript free of the
    label, as x_0 of ``x_n := x_0 + n d``, which are names of their own; ``shift`` how far
    the member it defines is from its label, as 2 of ``a_{n+2} := ...``; and ``recurring``
    whether its value is written with its own members."""

    base: str = field(default="", kw_only=True)
    fixed: frozenset[str] = field(default=frozenset(), kw_only=True)
    shift: int = field(default=0, kw_only=True)
    recurring: bool = field(default=False, kw_only=True)

    @property
    def label(self) -> str:
        """The label a member's subscript gives a value, n of ``c_n``."""
        return self.params[0]

    @property
    def parameters(self) -> tuple[str, ...]:
        """The parameters of each member, x of ``\\psi_n(x)``; none of ``c_n``."""
        return self.params[1:]


@dataclass(frozen=True)
class Subscripted:
    """A name with one subscript, as it is written: the ``node``; what the subscript is
    written on, its ``head``, and the name that is, its ``base``, as ``lambda`` of
    ``\\lambda_1`` and ``xbar`` of ``\\bar{x}_u``; the ``subscript`` as the parser read it,
    a number, a name or an expression, ``None`` where it is no expression, as ``a=0``; its
    index ``labels``, where it is written with labels alone, as x and y of ``f_{xy}``; the
    subscript as ``shown``; what is ``raised`` on the name, as the 2 of ``T_1^2``; and
    whether the lexer ``joined`` it into one name, as it does ``c_1``."""

    node: ast.Expr
    head: ast.Expr
    base: str
    subscript: ast.Expr | None
    labels: tuple[str, ...] | None
    shown: str
    raised: tuple[ast.Script, ...] = ()
    joined: bool = False

    @property
    def name(self) -> str:
        """The name it makes where its subscript labels it: ``c_1``, ``a_n+1``."""
        return f"{self.base}_{self.shown}"

    @property
    def plain(self) -> bool:
        """Whether its base is a name written plainly, with no accent or prime."""
        return isinstance(self.head, ast.Name)

    @property
    def bracketed(self) -> bool:
        """Whether it is written with its subscript in square brackets after the name, as
        ``x[n]`` is (see :func:`~relativisticpy.reading.sequences.subscript_form`)."""
        end = self.head.span.end
        return not self.joined and self.head.span.source.text[end : end + 1] == "["


def subscripted(node: ast.Expr) -> Subscripted | None:
    """A name with one subscript, taken apart: ``c_1`` and ``f_xy``, which the lexer joins
    into one name, and ``c_{2k}``, ``a_{n+1}`` and ``T_1^2``, a name with one subscript
    and, maybe, scripts raised on it; ``None`` for anything else."""
    if isinstance(node, ast.Name):
        head, underscore, tail = node.id.partition("_")
        if not (underscore and head and tail):
            return None
        subscript, labels = _joined_subscript(tail, node.span)
        return Subscripted(node, node, head, subscript, labels, tail, joined=True)
    if not isinstance(node, ast.Scripted):
        return None
    base = _base_name(node.base)
    lowered = [script for script in node.scripts if script.kind == "_"]
    if base is None or "_" in base or len(lowered) != 1:
        return None
    script = lowered[0]
    labels = None
    if script.indices is not None and all(_plain_label(item) for item in script.indices.items):
        labels = tuple(str(item.label) for item in script.indices.items)
    body = script.body
    shown = body.span.text if body is not None else script.span.text.lstrip("_").strip("{}")
    raised = tuple(script for script in node.scripts if script.kind == "^")
    return Subscripted(node, node.base, base, body, labels, shown.replace(" ", ""), raised)


def _numbered_variables(subscript: ast.Expr | None, params: Sequence[str]) -> list[str] | None:
    """The variables of a function of several variables that a subscript numbers, as
    Stewart writes f_1 for the derivative by the first and f_{12} for that by the first and
    then the second: x of ``1`` and x and y of ``12``, of f(x, y). ``None`` for a subscript
    that is no whole number, or names a variable the function does not have, as 3 of f(x,
    y), and for a function of one variable, whose f_1 is a name of its own."""
    if len(params) < 2 or not isinstance(subscript, ast.Number):
        return None
    digits = subscript.text
    if not digits.isdigit() or not all(1 <= int(digit) <= len(params) for digit in digits):
        return None
    return [params[int(digit) - 1] for digit in digits]


def member_written(node: ast.Expr, base: str, subscript: ast.Expr, shown: str) -> Subscripted:
    """A name of ``base`` with a ``subscript`` that a layer read apart from the name it is
    written in, as the 1 of ``E^{(0)}_1``, whose base is ``E^(0)``."""
    return Subscripted(node, ast.Name(base, span=node.span), base, subscript, None, shown)


def _joined_subscript(tail: str, span: Span) -> tuple[ast.Expr | None, tuple[str, ...] | None]:
    """The subscript of a name the lexer joined, as the parser would have read it in
    braces, and its index labels: 1 of ``c_1``, k of ``c_k``, and x and y of ``f_xy``."""
    if tail.isdigit():
        return ast.Number(tail, span=span), None
    if not tail.isalpha():
        return None, None
    labels = tuple(split_index_labels(tail))
    return (ast.Name(tail, span=span) if len(labels) == 1 else None), labels


def _base_name(base: ast.Expr) -> str | None:
    """The name a base with scripts is written as: ``lambda`` of ``\\lambda_1``,
    ``betahat`` of ``\\hat\\beta_1``, ``x'`` of ``x'_1``; ``None`` for what is no name."""
    if isinstance(base, ast.Name):
        return base.id
    if (
        isinstance(base, ast.Accent)
        and base.accent not in ("dot", "ddot")
        and isinstance(base.base, ast.Name)
    ):
        return accented(base.base.id, base.accent)
    if isinstance(base, ast.Prime):
        return base.func.id + "'" * base.order
    return None


def _lowered(script: ast.Script) -> str:
    """A subscript as the name it labels writes it: ``1`` of ``\\lambda_1``, ``10`` of
    ``V_{10}``, ``odot`` of ``M_\\odot``, ``eff`` of ``T_{\\rm eff}``, ``+`` of ``r_+``,
    ``times`` of ``h_\\times``."""
    if script.indices is not None:
        return "".join(
            str(item.label)
            if item.label is not None
            else (item.value.text if isinstance(item.value, ast.Number) else "")
            for item in script.indices.items
        )
    written = script.span.text.lstrip("_").strip()
    if written.startswith("{") and written.endswith("}"):
        written = written[1:-1]
    return written.replace(" ", "").replace("\\", "")


def _raised(script: ast.Script) -> str | None:
    """A superscript that labels a name, as it writes the name: ``2`` of ``\\sigma^2``,
    ``0`` of ``\\gamma^0``, ``(0)`` of ``E^{(0)}``; ``None`` for one that is no number or
    order, as ``\\mu`` or ``-1``."""
    body = script.body
    if isinstance(body, ast.Number) and body.text.isdigit():
        return body.text
    if _bracketed_order(body):
        assert body is not None
        return body.span.text.replace(" ", "")
    return None


def label_name(node: ast.Scripted, *, raised: bool = True) -> str | None:
    """The name a name with scripts makes where its scripts label it, those raised first,
    as TeX sets ``E^{(0)}_n`` and ``E_n^{(0)}`` alike: ``lambda_1`` of ``\\lambda_1``,
    ``sigma^2`` of ``\\sigma^2``, ``E^(0)_n`` of either, ``Y^1_2`` of ``Y_2^1``, whose
    ``Y_2`` the lexer joins. With ``raised`` false, of its base and subscripts alone, as
    ``lambda_1`` of ``\\lambda_1^2``. ``None`` where the base is no name, or a superscript
    is no label."""
    base = _base_name(node.base)
    if base is None:
        return None
    head, underscore, joined = base.partition("_")
    if not head:
        return None
    below = [joined] if underscore and joined else []
    marks = ""
    for script in node.scripts:
        if script.kind == "_":
            below.append(_lowered(script))
        elif raised:
            mark = _raised(script)
            if mark is None:
                return None
            marks += "^" + mark
    if not below and not marks:
        return None
    return head + marks + ("_" + "".join(below) if below else "")


def _an_index_label(label: str) -> bool:
    """Whether a label is written as an index is: a letter or a Greek letter, maybe primed
    or numbered, as ``\\mu'``, ``\\mu_n`` and ``i0`` are, or ``\\cdots``; not a mark, as
    ``odot``, nor a word, as ``eff``."""
    if label == "⋯":
        return True
    if label in _MARKS:
        return False
    letters = re.sub(r"[0-9']", "", label.partition("_")[0])
    return (len(letters) == 1 and letters.isascii() and letters.isalpha()) or letters in GREEK


def _index_like(script: ast.Script) -> bool:
    """Whether a subscript could be index labels or numbers, as those of ``R_{01}``,
    ``g_{tt}`` and ``S_{\\mu'\\nu'}`` are; not a mark, as ``\\odot``, a word, as ``eff``
    or ``{\\rm eff}``, nor what is no index list, as ``+`` or ``n+1``."""
    if script.indices is None or _WORD.search(script.span.text):
        return False
    return all(
        item.label is None or (_an_index_label(item.label) and item.label not in _MARKS)
        for item in script.indices.items
    )


def _labelled_by_index(node: ast.Scripted) -> bool:
    """Whether a subscript of a name is written with an index label, as ``\\mu`` and ``i``
    are in ``\\bar h_{\\mu\\nu}`` and ``\\bar h_{i0}``, and the frame's index ``(\\mu)``
    is in ``\\hat e_{(\\mu)}``, rather than with numbers, marks, signs and words alone."""
    for script in node.scripts:
        if script.kind != "_":
            continue
        if script.indices is None:
            body = script.body
            name = _base_name(body) if body is not None else None
            if name is not None and _an_index_label(name):
                return True
        elif any(i.label is not None and _an_index_label(i.label) for i in script.indices.items):
            return True
    return False


def picks_a_member(subscript: ast.Expr | None, bound: Callable[[str], bool]) -> bool:
    """Whether a subscript picks a member of a family or an element of a list: a whole
    number, as 1 in ``c_1`` and -1 in ``E_{-1}``; an expression computed from numbers and
    names, as ``n+1``, ``n-1`` and ``2k``; or a name that ``bound`` says has a value where
    it is written, as the index of a sum has. A letter that nothing binds, as x in
    ``v_x``, may name a component of its own; and what is written side by side with no
    number first, as ``\\mu\\nu`` and ``i0``, are labels, not a product."""
    if isinstance(subscript, ast.Number):
        return subscript.text.isdigit()
    if isinstance(subscript, ast.Name):
        return bound(subscript.id)
    if isinstance(subscript, ast.Unary):
        return subscript.op in ("-", "+")
    if not isinstance(subscript, ast.Binary):
        return False
    if subscript.op == "*" and subscript.implicit:  # 2k, but not \mu\nu or \mathrm{A}0
        first: ast.Expr = subscript
        while isinstance(first, ast.Binary) and first.op == "*" and first.implicit:
            first = first.left
        return isinstance(first, ast.Number)
    return subscript.op in ("+", "-", "*", "/", "**")


def family_of(base: str, scope: Scope) -> FunctionBinding | None:
    """The family the program defines whose members are written with ``base``, as c_n is
    for c."""
    current: Scope | None = scope
    while current is not None:
        for _, binding in reversed(current.bindings()):
            definition = binding.definition if isinstance(binding, FunctionBinding) else None
            if isinstance(definition, Family) and definition.base == base:
                return binding  # type: ignore[return-value]
        current = current.parent
    return None


def a_tensor(binding: object) -> bool:
    """Whether a name is bound to a tensor: one defined with its indices, or a value with
    free indices."""
    if isinstance(binding, TensorBinding):
        return True
    return isinstance(binding, VariableBinding) and isinstance(binding.node.type, TensorType)


# -- the readings ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reading:
    """How a name with a subscript, or a name with brackets after it, is read: the
    ``kind`` of the reading; what was ``written``; and what the layer that lowers it needs
    of it: the ``binding`` it is read by, as the family, the sample or the function; the
    ``name`` the program defines, or the capability a bracket writes; the tensor with its
    indices that a joined name writes, ``indexed``; the parameters a partial derivative is
    by, its ``labels``; and the ``book`` name. ``refusal`` is the error it is refused
    with, where it cannot be read, raised where the reading is taken, and ``why`` the
    reason a decision records for it. ``note`` says which reading was taken where the
    program defines the base as two things, as a family and a list. A label is
    ``at_a_value`` where its subscript is a label that has a value where it is read: it is
    then the name at that value. ``position`` is where an element is in its list, counted
    from 0."""

    kind: Kind
    written: Subscripted | None = None
    binding: FunctionBinding | VariableBinding | None = None
    name: str | None = None
    indexed: ast.Scripted | None = None
    labels: tuple[str, ...] = ()
    book: BookReading | None = None
    refusal: RelativisticPyError | None = None
    why: str = ""
    note: str | None = None
    at_a_value: bool = False
    position: int | None = None


#: The readings by what the program itself defines: where two of them apply to one name,
#: the first is taken, and a note says which the other was.
_OF_THE_PROGRAM = frozenset(
    {Kind.INDEX, Kind.ENTRY, Kind.MEMBER, Kind.RECURRENCE, Kind.ELEMENT, Kind.SAMPLE, Kind.PARTIAL}
)


def _described(reading: Reading, latex: bool) -> str:
    """What a name is read as, as a note says it: ``a member of the family c_n``."""
    written = reading.written
    assert written is not None
    base = _shown(written.base, latex)
    binding = reading.binding
    if reading.kind in (Kind.MEMBER, Kind.RECURRENCE) and isinstance(binding, FunctionBinding):
        return f"a member of the family {base}_{binding.definition.params[0]}"
    described = {
        Kind.INDEX: f"the tensor {base} with its index",
        Kind.ENTRY: f"an entry of the matrix {base}",
        Kind.ELEMENT: f"an element of the list {base}",
        Kind.SAMPLE: f"a member of the sample {base}",
        Kind.PARTIAL: f"a partial derivative of the function {base}",
    }
    return described.get(reading.kind, reading.kind.value)


class Subscripts(Matrices):
    """The resolver of names with a subscript, and of names with brackets after them: the
    test of each reading, in order, the state the tests read, and the one place a reading
    is handed to the layer that lowers it."""

    def __init__(self, session: Session) -> None:
        super().__init__(session)
        # The labels that have a value where a name is being read, each with it: the
        # label of a family whose member is being made, as \sigma_a of T_a := \sigma_a/2
        # is \sigma_1 in T_1, and the index of a sum taken term by term.
        self._at_members: dict[str, Node] = {}
        # The names the equation being read writes at points that nothing defines: its
        # unknown sequences, as y and x of y[n] = x[n] + \frac{1}{2} y[n-1].
        self._unknown_sequences: frozenset[str] = frozenset()
        # The indices of the sums and products being read, innermost last: each an integer,
        # with a value in each term.
        self._sum_indices: list[sympy.Symbol] = []
        # The sums and products being taken term by term: each index with its first and
        # its last value, from which the elements of a list are counted (decision 17).
        self._term_ranges: dict[str, tuple[int, int]] = {}

    # -- what the layers above give ---------------------------------------------------------

    def _folded(self, node: Node) -> Any:
        """The value of a node that is known before anything is computed, which the
        lowerer gives (see :class:`~relativisticpy.lowering.transforms.Transforms`)."""
        raise NotImplementedError

    def _member_read(
        self, reading: Reading, arguments: Sequence[ast.Expr] | None, scope: Scope, ctx: Context
    ) -> Node | None:
        """A member of a family, lowered (see
        :class:`~relativisticpy.reading.families.Families`)."""
        raise NotImplementedError

    def _sample_member(self, reading: Reading, scope: Scope, ctx: Context) -> Node:
        """A member of a sample, lowered (see
        :class:`~relativisticpy.lowering.probability.Probability`)."""
        raise NotImplementedError

    def _partial_derivative(self, reading: Reading, scope: Scope, ctx: Context) -> Node:
        """A partial derivative written with a subscript, lowered (see
        :class:`~relativisticpy.lowering.vectors.VectorCalculus`)."""
        raise NotImplementedError

    def _sequence_value(self, reading: Reading, scope: Scope, ctx: Context) -> Node:
        """A sequence at the point its subscript writes, lowered (see
        :class:`~relativisticpy.lowering.sequences.Sequences`)."""
        raise NotImplementedError

    def _random_names(self, exprs: Sequence[ast.Expr | None], scope: Scope) -> list[str]:
        """The random variables the program declares that expressions name, which the
        lowerer says (see :class:`~relativisticpy.lowering.probability.Probability`)."""
        raise NotImplementedError

    def _capability(self, book: BookCall, span: Span, scope: Scope, ctx: Context) -> Node:
        """A call of a capability, lowered (see
        :class:`~relativisticpy.lowering.expressions.Expressions`)."""
        raise NotImplementedError

    def _a_sum_index(self, name: str, scope: Scope) -> bool:
        """Whether a name is bound as the index of a sum or a product being read."""
        binding = scope.lookup(name)
        if not (isinstance(binding, VariableBinding) and isinstance(binding.node.op, ops.Const)):
            return False
        return binding.node.op.value in self._sum_indices

    @contextlib.contextmanager
    def _members_at(self, label: str, point: Node) -> Iterator[None]:
        """Read names where a label has a value: its subscripts are the value's."""
        saved = dict(self._at_members)
        self._at_members[label] = point
        try:
            yield
        finally:
            self._at_members = saved

    # -- a name with a subscript ------------------------------------------------------------

    def _subscript_reading(
        self,
        written: Subscripted,
        scope: Scope,
        *,
        latex: bool = False,
        called: bool = False,
        summed: frozenset[str] = frozenset(),
    ) -> Reading:
        """How a name with a subscript is read: the first reading, in the order of
        :data:`PRECEDENCE`, whose test finds it. ``latex`` says how its messages write
        names; ``called`` that it is called with arguments, as ``\\psi_3(x)`` is, so that
        any subscript picks a member; and ``summed`` names the indices of the sums it is
        read in, which have a value in each term."""
        tests: dict[Kind, Callable[[], Reading | None]] = {
            Kind.DEFINED: lambda: self._reads_as_defined(written, scope),
            Kind.INDEX: lambda: self._reads_as_index(written, scope),
            Kind.ENTRY: lambda: self._reads_as_entry(written, scope),
            Kind.MEMBER: lambda: self._reads_as_member(written, scope, latex, called, summed),
            Kind.ELEMENT: lambda: self._reads_as_element(written, scope, latex, summed),
            Kind.SAMPLE: lambda: self._reads_as_sample(written, scope),
            Kind.SEQUENCE: lambda: self._reads_as_sequence(written, scope),
            Kind.PARTIAL: lambda: self._reads_as_partial(written, scope),
            Kind.BUILTIN: lambda: self._reads_as_builtin(written, scope),
            Kind.BOOK: lambda: self._reads_as_book(written, scope),
        }
        found: Reading | None = None
        others: list[Reading] = []
        for kind in PRECEDENCE:
            if found is not None and not {found.kind, kind} <= _OF_THE_PROGRAM:
                continue  # only two things the program defines are worth a note
            reading = tests[kind]() if kind in tests else None
            if reading is None:
                continue
            if found is None:
                found = reading
            else:
                others.append(reading)
        if found is None:
            subscript = written.subscript
            valued = isinstance(subscript, ast.Name) and subscript.id in self._at_members
            return Reading(Kind.LABEL, written, at_a_value=valued and written.plain)
        if not others:
            return found
        shown = written.node.span.text
        rivals = " or ".join(_described(other, latex) for other in others)
        note = f"{shown} is read as {_described(found, latex)}, not as {rivals}"
        return replace(found, note=note)

    def _subscripted_term(self, written: Subscripted, scope: Scope, ctx: Context) -> Term | None:
        """What a name with a subscript is where it is read as more than a name: the
        reading the resolver finds, lowered by its layer, with what is raised on a member
        its power. ``None`` for a name the program defines, a label, and a tensor written
        with braces, which are read as names and tensors are."""
        reading = self._subscript_reading(written, scope, latex=ctx.latex)
        if reading.refusal is not None:
            symbol = f"the symbol {written.name}"
            self._decide("index or power", written.node.span, symbol, reading.why)
            raise reading.refusal
        node, kind = written.node, reading.kind
        if reading.note is not None:
            self._note(reading.note)
        if kind in (Kind.MEMBER, Kind.RECURRENCE) or reading.at_a_value:
            if any(script.body is None for script in written.raised):
                return None
            member = self._member_read(reading, None, scope, ctx)
            if member is None:
                return None
            for script in written.raised:
                assert script.body is not None
                member = self._power(member, self.value(script.body, scope, ctx), node.span)
            return member
        if kind is Kind.ELEMENT and reading.position is not None and not written.raised:
            return self._element(reading)
        if written.raised:
            return None  # what is raised on it is read with the name it makes
        if kind is Kind.INDEX and reading.indexed is not None:  # y_t, where y is a tensor
            why = f"{written.base} is a tensor, and {node.span.text} names nothing"
            self._decide("index or power", node.span, f"{written.base} with its indices", why)
            return self.term(reading.indexed, scope, ctx)
        if kind is Kind.SAMPLE:
            return self._sample_member(reading, scope, ctx)
        if kind is Kind.SEQUENCE:
            return self._sequence_value(reading, scope, ctx)
        if kind is Kind.PARTIAL:
            return self._partial_derivative(reading, scope, ctx)
        if kind is Kind.BOOK and reading.book is not None:
            call = self._book_said(reading.book, node)
            return self._capability(call, node.span, scope, ctx) if call is not None else None
        return None

    def _reads_as_defined(self, written: Subscripted, scope: Scope) -> Reading | None:
        """A name the program defines under the name as written: with what is raised on
        it, as ``\\sigma^2``, or without, as ``\\lambda_1`` of ``\\lambda_1^2``."""
        node = written.node
        names: list[str | None] = [written.name]
        if isinstance(node, ast.Scripted):
            names = [label_name(node), label_name(node, raised=False)]
        elif isinstance(node, ast.Name):
            names = [node.id]
        for name in names:
            if name is not None and scope.lookup(name) is not None:
                return Reading(Kind.DEFINED, written, name=name)
        return None

    def _reads_as_index(self, written: Subscripted, scope: Scope) -> Reading | None:
        """An index of a tensor the program defines: on a name the lexer joined, as
        ``y_t``, the tensor with its indices; in braces, it is read with the tensor."""
        if isinstance(written.node, ast.Name):
            indexed = _indexed_name(written.node, scope) if written.joined else None
            return Reading(Kind.INDEX, written, indexed=indexed) if indexed is not None else None
        if written.plain and a_tensor(scope.lookup(written.base)):
            return Reading(Kind.INDEX, written)
        return None

    def _reads_as_entry(self, written: Subscripted, scope: Scope) -> Reading | None:
        """An entry of a matrix the program gives a name, by its row and its column in
        braces, counted from 1, as ``A_{12}``."""
        node = written.node
        if not (isinstance(node, ast.Scripted) and written.plain and not written.raised):
            return None
        indices = node.scripts[0].indices
        items = indices.items if indices is not None else ()
        numbers = [item.value for item in items if isinstance(item.value, ast.Number)]
        if len(items) != 2 or len(numbers) != 2:
            return None
        if not all(number.text.isdigit() and int(number.text) >= 1 for number in numbers):
            return None
        binding = scope.lookup(written.base)
        if not (isinstance(binding, VariableBinding) and _a_matrix(binding.node)):
            return None
        return Reading(Kind.ENTRY, written, binding=binding)

    def _reads_as_member(
        self,
        written: Subscripted,
        scope: Scope,
        latex: bool,
        called: bool,
        summed: frozenset[str] = frozenset(),
    ) -> Reading | None:
        """A member of a family of the base, as ``c_1``, ``c_{2k}`` and ``a_{n+1}`` are.
        Not one of the names a family's value writes, as x_0; nor, unless it is called, a
        subscript that is a letter nothing binds, as k of c_k, a symbol of its own. A name
        defined with a letter that could be no label is no family, and what would be its
        members are refused."""
        subscript = written.subscript
        if not written.plain or subscript is None:
            return None
        binding = family_of(written.base, scope)
        if binding is None:
            return self._reads_as_no_family(written, scope, latex)
        family = binding.definition
        assert isinstance(family, Family)
        if written.name in family.fixed:
            return None
        own = isinstance(subscript, ast.Name) and subscript.id == family.label
        if not (called or own or picks_a_member(subscript, _has_a_binding(scope, summed))):
            return None
        kind = Kind.RECURRENCE if family.recurring else Kind.MEMBER
        return Reading(kind, written, binding=binding)

    def _reads_as_no_family(
        self, written: Subscripted, scope: Scope, latex: bool
    ) -> Reading | None:
        """The refusal of a name whose base the program defines with a letter, as ``c_1``
        after ``c_n := n^2`` where n could be no label: read as written, it would be a
        symbol of its own, silently (guard 3 of wave 0, curriculum/ROADMAP.md)."""
        if not picks_a_member(written.subscript, _has_a_value(scope, frozenset())):
            return None
        letter = self._defined_with_a_letter(written.base, written.name, scope)
        if letter is None:
            return None
        shown, text = _shown(written.base, latex), written.node.span.text
        family = f"{shown}_{_shown(letter, latex)}"
        why = f"{family} is a family in {letter}, which is not supported yet"
        refusal = CompileError(
            f"{text} names nothing, and {family} is a family in {letter}, which is not "
            "supported yet",
            span=written.node.span,
            notes=[
                f"read as written, {text} would be a symbol of its own, not {family} at "
                f"{letter} = {written.shown}"
            ],
            hint=f"define {shown} as a function of {letter}, as in {shown}({letter}) := ..., "
            f"and write {shown}({written.shown}) for {text}",
        )
        return Reading(Kind.MEMBER, written, refusal=refusal, why=why)

    def _defined_with_a_letter(self, base: str, used: str, scope: Scope) -> str | None:
        """The letter a name of ``base`` is defined with, as n of ``c_n := n^2``: a name of
        base and one letter, bound to a value written with that letter, and not with
        ``used``, which is then a symbol it is written with, as x_0 is of
        ``x_n := x_0 + n d``."""
        current: Scope | None = scope
        while current is not None:
            for name, binding in current.bindings():
                prefix, _, label = name.partition("_")
                single = (len(label) == 1 and label.isalpha()) or label in GREEK
                if prefix != base or not single or not isinstance(binding, VariableBinding):
                    continue
                if _mentions_symbol(binding.node, label) and not _mentions_symbol(
                    binding.node, used
                ):
                    return label
            current = current.parent
        return None

    def _reads_as_element(
        self, written: Subscripted, scope: Scope, latex: bool, summed: frozenset[str]
    ) -> Reading | None:
        """An element of a list the program defines, where the subscript picks one,
        counted from the lower limit of the sum that ranges over the whole list (decision
        17): a sum over an index of a list alone binds the list as a tensor over the sum's
        space, whose index the subscript then is, and one taken term by term reads each
        element where the index is. Anywhere else its positions have no start, and it is
        refused. A list written with the name, as p := (p_x, p_y), has it as a name of its
        own."""
        within = scope.lookup(written.base)
        if not (written.plain and isinstance(within, VariableBinding)):
            return None
        size = _elements(within.node)
        if size is None:
            return None
        picked = picks_a_member(written.subscript, _has_a_value(scope, summed))
        if not (picked or written.bracketed):  # x[n] is of the list, whatever n is
            return None
        if _mentions_symbol(within.node, written.name):
            return None
        shown, text = _shown(written.base, latex), written.node.span.text
        counted = self._counted(written.subscript)
        if counted is not None:
            first, last, at = counted
            why = f"the sum runs over all of {shown}, from {first}"
            if last - first + 1 == size:
                return Reading(Kind.ELEMENT, written, binding=within, position=at - first, why=why)
            refusal = CompileError(
                f"{shown} has {size} elements, but the sum runs over {last - first + 1}: a sum "
                "over a part of a list is not supported yet",
                span=written.node.span,
            )
            return Reading(Kind.ELEMENT, written, binding=within, refusal=refusal, why=why)
        if written.bracketed:  # x[2], outside a sum over all of x
            why = f"{shown} is a list, and {text} is in no sum over all of it"
            if not latex:
                return Reading(
                    Kind.ELEMENT,
                    written,
                    binding=within,
                    refusal=python_indexing(text, written.node.span),
                    why=why,
                )
            refusal = CompileError(
                "a list is not indexed with [ ]",
                span=written.node.span,
                notes=[
                    "the elements of a list are read in a sum over all of it, counted from "
                    f"its lower limit, as in \\sum_{{n=0}}^{{{size - 1}}} {shown}[n]"
                ],
                hint="to name the elements of a list, write them on the left of :=, as in "
                "a, b := [1, 2]; for a product, put a space before the bracket",
            )
            return Reading(Kind.ELEMENT, written, binding=within, refusal=refusal, why=why)
        label = written.shown if len(written.shown) == 1 and written.shown.isalpha() else "i"
        refusal = CompileError(
            f"{text} names nothing, and {shown} is a list, whose elements are not "
            "written with a subscript yet",
            span=written.node.span,
            notes=[f"read as written, {text} would be a symbol of its own"],
            hint="give the list an index over a space: "
            f"\\mathcal{{N}} := \\mathbb{{R}}^{{{size}}}, {label} \\in \\mathcal{{N}} and "
            f"{shown}^{label} := [...]; then {shown}^{label} 1_{label} is the sum of its "
            "elements",
        )
        why = f"{shown} is a list"
        return Reading(Kind.ELEMENT, written, binding=within, refusal=refusal, why=why)

    def _counted(self, subscript: ast.Expr | None) -> tuple[int, int, int] | None:
        """The first and the last value of the index of a sum taken term by term, and the
        value it has in the term being read, where ``subscript`` is that index."""
        if not isinstance(subscript, ast.Name) or subscript.id not in self._term_ranges:
            return None
        point = self._at_members.get(subscript.id)
        value = point.op.value if point is not None and isinstance(point.op, ops.Const) else None
        if not isinstance(value, sympy.Integer):
            return None
        first, last = self._term_ranges[subscript.id]
        return first, last, int(value)

    def _element(self, reading: Reading) -> Node:
        """An element of a list, at its position: the item written there, or, of a list a
        value holds, the item it has."""
        written, binding, position = reading.written, reading.binding, reading.position
        assert written is not None
        assert isinstance(binding, VariableBinding)
        assert position is not None
        base, span = written.base, written.node.span
        self._decide("index or power", span, f"element {position + 1} of {base}", reading.why)
        node = binding.node
        if isinstance(node.op, ops.Call) and node.op.name in ("core.list", "core.tuple"):
            return node.args[position]
        assert isinstance(node.op, ops.Const)
        return self.graph.const(sympy.sympify(node.op.value[position]), SCALAR, span)

    def _list_summed(self, node: ast.Node, index: str, scope: Scope) -> int | None:
        """How many elements a list has, where ``node`` is a name with the index of a sum
        as its subscript that is read as an element of the list, as ``x_n`` is under
        ``\\sum_n`` after x := (1, 2, 0, -1); ``None`` for any other name, and for one read
        otherwise, as a member of a family of the same letter is."""
        written = subscripted(node) if isinstance(node, ast.Expr) else None
        if written is None:
            return None
        reading = self._subscript_reading(written, scope, summed=frozenset({index}))
        if reading.kind is not Kind.ELEMENT or not isinstance(reading.binding, VariableBinding):
            return None
        return _elements(reading.binding.node)

    def _reads_as_sample(self, written: Subscripted, scope: Scope) -> Reading | None:
        """A member of a sample the program declares, written with its capital letter and
        a subscript that is written as an integer is, as ``X_i``, ``X_1`` and ``X_{k+1}``
        are: whole numbers and names, in sums, differences and products. Any other
        subscript is refused: read as written, it would be a symbol of its own."""
        base, subscript = written.base, written.subscript
        if not (written.plain and len(base) == 1 and base.isascii() and base.isupper()):
            return None
        if written.raised or subscript is None:
            return None
        binding = scope.lookup(SAMPLE.format(base))
        if not isinstance(binding, VariableBinding):
            return None
        if sequences.integral(subscript, lambda label: self._integral_name(label, scope)):
            return Reading(Kind.SAMPLE, written, binding=binding)
        text = written.node.span.text
        refusal = CompileError(
            f"{text} is no member of the sample {base}: its members are numbered by integers, "
            f"and {written.shown} is not written as one",
            span=written.node.span,
            hint=f"write a member with a whole number or a name, as {base}_1 or {base}_{{k+1}}",
        )
        why = f"{base} is a sample, declared on line {_line_of(binding)}"
        return Reading(Kind.SAMPLE, written, binding=binding, refusal=refusal, why=why)

    def _reads_as_sequence(self, written: Subscripted, scope: Scope) -> Reading | None:
        """A sequence the program defines with brackets, ``x[n] := ...``, written with its
        point as a subscript, as ``x_n`` and ``x_{n-1}`` are: the sequence at the point."""
        binding = scope.lookup(written.base)
        if not (written.plain and written.subscript is not None):
            return None
        if not isinstance(binding, FunctionBinding):
            return None
        if not isinstance(binding.definition, SequenceDefinition):
            return None
        return Reading(Kind.SEQUENCE, written, binding=binding)

    def _reads_as_partial(self, written: Subscripted, scope: Scope) -> Reading | None:
        """A partial derivative of a function the program defines, by the parameters the
        subscript names in turn, a run of letters each naming one, as ``f_x`` and
        ``f_{xy}`` of f(x, y) (decision 5.6), or numbers them, as Stewart's ``f_1`` and
        ``f_{12}`` of a function of several variables; not of a family, whose subscripts
        pick its members, nor of a sequence or a frequency response, which are written at
        points."""
        if isinstance(written.head, ast.Prime):
            return None
        binding = scope.lookup(written.base)
        if not isinstance(binding, FunctionBinding):
            return None
        if isinstance(binding.definition, Family | SequenceDefinition | Response):
            return None
        params = binding.definition.params
        if written.labels is None:  # f_1 and f_{12}: by the first, then the second
            numbered = _numbered_variables(written.subscript, params)
            if numbered is None:
                return None
            return Reading(Kind.PARTIAL, written, binding=binding, labels=tuple(numbered))
        labels = [written.shown] if written.joined else list(written.labels)
        if not all(label in params for label in labels):
            labels = [letter for label in labels for letter in label]
        if not labels or not set(labels) <= set(params):
            return None
        if scope.lookup(f"{written.base}_{''.join(labels)}") is not None:
            return None
        return Reading(Kind.PARTIAL, written, binding=binding, labels=tuple(labels))

    def _reads_as_builtin(self, written: Subscripted, scope: Scope) -> Reading | None:
        """An index of a built-in tensor, where nothing defines the name and its subscript
        is an index (see :meth:`_indexes`), as that of ``R_{01}`` with a metric in force."""
        node = written.node
        if not (isinstance(node, ast.Scripted) and written.plain):
            return None
        if scope.lookup(written.base) is not None:
            return None
        if not any(self._builtin_tensor(written.base, rank) for rank in range(1, 5)):
            return None
        return Reading(Kind.BUILTIN, written) if self._indexes(node, written.base) else None

    def _reads_as_book(self, written: Subscripted, scope: Scope) -> Reading | None:
        """A name books write for a function of an integer, with the integer as its
        subscript, as the Bernoulli number ``B_n`` (decisions 34 and 35)."""
        node = written.node
        if isinstance(node, ast.Name):
            shape = forms.subscripted_shape(node)
        elif isinstance(node, ast.Scripted):
            shape = forms.scripted_shape(node)
        else:
            return None
        if shape is None or shape.kind != "subscripted":
            return None
        book = self._book_reading(shape, node, node.span.text, scope)
        return Reading(Kind.BOOK, written, book=book) if book is not None else None

    # -- what a script on a name is -----------------------------------------------------

    def _an_index_here(self, label: str) -> bool:
        """Whether a label is an index wherever it is written: declared to range over a
        space, ranging over one in this statement, or bound as an index is by a sum."""
        return label in self._labels or label in self.session.labels or label in self._bound

    def _indexes(self, node: ast.Scripted, base: str) -> bool:
        """Whether the subscripts of a name nothing defines are indices (decision 5 of
        design/notation.md): a label declared or ranging over a space; of a built-in
        tensor, as many as it has, any index label, as \\nu' of ``W_{\\nu'}``, and numbers
        while a metric is in force, as in ``R_{01}``. Numbers without a metric, marks and
        words label it, as in ``\\delta_0`` and ``R_\\odot``."""
        items = [
            item
            for script in node.scripts
            if script.indices is not None
            for item in script.indices.items
        ]
        if any(item.label is not None and self._an_index_here(item.label) for item in items):
            return True
        if not any(self._builtin_tensor(base, rank) for rank in range(1, 5)):
            return False
        indexed = all(s.indices is not None for s in node.scripts)
        if indexed and self._builtin_tensor(base, len(items)) is not None:
            return True
        if _labelled_by_index(node):
            return True
        lowered = [s for s in node.scripts if s.kind == "_"]
        return self.session.metric is not None and all(map(_index_like, lowered))

    # -- a name with brackets after it --------------------------------------------------

    def _bracket_reading(self, name: ast.Name, point: ast.Expr, scope: Scope) -> Reading:
        """How ``name[point]`` is read: the first reading, in the order of
        :data:`BRACKETS`, whose test finds it."""
        binding = scope.lookup(name.id)
        if isinstance(binding, FunctionBinding):
            definition = binding.definition
            sequence = isinstance(definition, SequenceDefinition) or (
                len(definition.params) == 1 and not isinstance(definition, Family | Response)
            )
            return Reading(Kind.SEQUENCE if sequence else Kind.PRODUCT, binding=binding)
        integral = sequences.integral(point, lambda label: self._integral_name(label, scope))
        if isinstance(binding, VariableBinding) and integral and self._a_plain_list(binding.node):
            return Reading(Kind.ELEMENT, binding=binding)
        if binding is not None:
            return Reading(Kind.PRODUCT)
        if name.id in self._unknown_sequences:
            return Reading(Kind.SEQUENCE)
        family = family_of(name.id, scope)
        if family is not None and len(family.definition.params) == 1:  # x[n] of x_n := n^2
            return Reading(Kind.MEMBER, binding=family)
        operator = probability.operator(name)
        if operator is not None and (
            not probability.lettered(name, square=True) or self._random_names([point], scope)
        ):
            return Reading(Kind.OPERATOR, name=operator)  # E[X], of a random variable X
        if self._has_a_meaning(name.id) or name.id not in BRACKET_BOOKS:
            return Reading(Kind.PRODUCT)
        if integral:
            return Reading(Kind.BOOK, name=BRACKET_BOOKS[name.id])
        return Reading(Kind.PRODUCT)

    @staticmethod
    def _a_plain_list(node: Node) -> bool:
        """Whether a value is a list of items that are no lists, whose elements a subscript
        or brackets pick: not a matrix, which brackets after it multiply."""
        if _elements(node) is None:
            return False
        if isinstance(node.op, ops.Call):
            return all(_elements(item) is None for item in node.args)
        return all(not isinstance(item, list | tuple) for item in node.op.value)  # type: ignore[attr-defined]

    def _integral_name(self, label: str, scope: Scope) -> bool | None:
        """Whether a name in a point is an integer: one bound to a whole number, or assumed
        an integer; ``None`` where it is free, or bound to an expression of symbols, as the
        index of a sum and the label of a sequence are, which a letter may be; and ``False``
        for one bound to any other number, as j := \\sqrt{-1}, or to a function."""
        binding = scope.lookup(label)
        if isinstance(binding, VariableBinding):
            value = self._folded(binding.node)
            if not isinstance(value, sympy.Basic):
                return None  # computed, as the point a sequence is taken at may be
            if value.is_integer:
                return True
            return False if value.is_number else None
        if binding is not None:
            return False
        if label in CONSTANTS:
            return False
        symbol = self.session.assumptions.get(label)
        return True if symbol is not None and symbol.is_integer else None

    def _has_a_meaning(self, name: str) -> bool:
        """Whether the program gives a name a meaning beyond its scope: a coordinate, a
        function of the coordinates, or a symbol it says something of."""
        coordinates = {c.name for c in self.session.coordinates or ()}
        session = self.session
        return name in coordinates or name in session.functions or name in session.assumptions


def _line_of(binding: VariableBinding) -> int | str:
    """The line a value was defined on, where it is known."""
    return binding.node.span.line if binding.node.span is not None else "?"


def _has_a_binding(scope: Scope, summed: frozenset[str]) -> Callable[[str], bool]:
    """Whether a name is bound where a subscript writes it, to anything, or is the index of
    a sum being read."""
    return lambda name: name in summed or scope.lookup(name) is not None


def _has_a_value(scope: Scope, summed: frozenset[str]) -> Callable[[str], bool]:
    """Whether a name has a value known where a subscript writes it: the index of a sum
    being read, or a name bound to a constant."""

    def valued(name: str) -> bool:
        if name in summed:
            return True
        binding = scope.lookup(name)
        return isinstance(binding, VariableBinding) and isinstance(binding.node.op, ops.Const)

    return valued
