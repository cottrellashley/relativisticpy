r"""Written forms: how a capability is written in books, read and printed.

A capability declares the forms it is written in, in the language itself: in LaTeX, as
``J_{\nu}(x)``, and in Unicode, as ``J_ν(x)``; in plain text it is its name, called, as
``besselj(nu, x)``. Each form is parsed once, by the language's own parser, into a
:class:`Pattern`: the shape of a call, the name it is written with and how that name is
set, its scripts and its arguments, each a variable named after a parameter of ``exact``.

The reader takes what a program writes apart into the same :class:`Shape`, and finds the
capability whose pattern it has (:func:`matching`); a reservation is matched so too, so
that a book name no integration provides is refused, saying what it is. The display
prints a call of a capability in its first form (:func:`printed`), so that it reads back.

    >>> from relativisticpy.language import parse
    >>> shape = form_shape(parse(r"J_{1/2}(\pi)").statements[0].expr)
    >>> shape.head, shape.style, len(shape.below), len(shape.arguments)
    ('J', 'plain', 1, 1)
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import sympy
from sympy.core.function import AppliedUndef

from relativisticpy.errors import RelativisticPyError
from relativisticpy.integrations._declarations import Capability, Function, Reservation
from relativisticpy.integrations._discovery import catalogue, on_change
from relativisticpy.language import syntax as ast
from relativisticpy.language.parser import split_index_labels
from relativisticpy.language.vocabulary import GREEK

#: How the name a function is written with is set: upright, as ``\operatorname{erfc}``
#: and ``\mathrm{B}``; calligraphic, as ``\mathcal{L}``; a Greek letter written as one, as
#: ``\Gamma`` and ``Γ``; or plainly, as ``J`` and ``besselj``.
Style = Literal["upright", "calligraphic", "letter", "plain"]

#: What a shape is: a call, in brackets; an operator, applied to what follows it, as
#: ``\operatorname{tr} A``; or a name with indices, as ``\epsilon_{ijk}``.
Kind = Literal["call", "operator", "indexed"]

_UPRIGHT = ("\\operatorname", "\\mathrm", "\\text", "\\rm", "{\\rm", "\\mathup")
_CALLIGRAPHIC = ("\\mathcal", "\\mathscr")
_OPENERS = ("(", "\\left(")
_CLOSERS = (")", "\\right)")


@dataclass(frozen=True)
class Shape:
    """What a program writes, taken apart as a call: its ``kind``; the ``head``, the name
    it is written with, as ``written``, and how that is set; its scripts, those ``before``
    it, as the p of ``{}_pF_q``, ``below`` and ``above`` it, each one item; and its
    ``arguments``, in round brackets, or in braces, as ``\\mathcal{L}\\{f\\}``."""

    kind: Kind
    head: str
    style: Style
    written: str = ""
    before: tuple[ast.Node, ...] = ()
    below: tuple[ast.Node, ...] = ()
    above: tuple[ast.Node, ...] = ()
    arguments: tuple[ast.Expr, ...] = ()
    bracket: str = "("

    @property
    def scripts(self) -> tuple[ast.Node, ...]:
        return (*self.before, *self.below, *self.above)


@dataclass(frozen=True)
class Pattern:
    """A written form of a capability or a reservation, parsed: the ``owner``, the form
    as it was ``written``, in LaTeX or Unicode (its ``form``), and its :class:`Shape`, whose
    scripts and arguments are variables named after the owner's parameters."""

    owner: Capability | Reservation
    written: str
    form: Literal["latex", "unicode"]
    shape: Shape

    @property
    def separator(self) -> str:
        """What separates the arguments: a comma, or a semicolon, as in ``{}_pF_q``."""
        return ";" if ";" in self.written else ","

    def variables(self) -> list[str]:
        """The names of the variables the form is written with, in order."""
        found: list[str] = []
        for slot in (*self.shape.scripts, *self.shape.arguments):
            if isinstance(slot, ast.Name) and slot.id not in found:
                found.append(slot.id)
        return found


@dataclass(frozen=True)
class Match:
    """What a call a program writes was found to be: the ``pattern`` it has, and what each
    variable of the pattern is in the call, its ``bindings``."""

    pattern: Pattern
    bindings: Mapping[str, ast.Node] = field(default_factory=dict)


# -- taking what is written apart ------------------------------------------------------


#: The commands that set a word as a label: ``\\text{eff}``, ``\\mathrm{in}``, ``{\\rm max}``.
_LABEL = re.compile(r"\\(?:text[a-z]*|mathrm|rm|mbox|operatorname)(?![A-Za-z])")


def call_shape(node: ast.Call) -> Shape | None:
    """``\\Gamma(x)``, ``J_0(1)`` or ``\\operatorname{erfc}(x)``: a call of a name, which
    may carry scripts, as the lexer joins ``J_0`` into one name."""
    head = _head(node.func)
    if head is None:
        return None
    name, style, written, before, below, above = head
    after = node.span.source.text[node.func.span.end : node.span.end].lstrip()
    bracket = "{" if after.startswith(("\\{", "\\lbrace", "\\left\\{")) else "("
    return Shape("call", name, style, written, before, below, above, tuple(node.args), bracket)


def pair_shape(head: ast.Expr, bracket: ast.Expr) -> Shape | None:
    """``J_{\\nu}(x)`` or ``Y_l^m(\\theta, \\phi)``: a name with scripts and, right after
    it, its arguments in brackets, which the parser reads as a product."""
    arguments = bracketed(bracket)
    if arguments is None or head.span.end != bracket.span.start:
        return None
    parts = _head(head)
    if parts is None:
        return None
    name, style, written, before, below, above = parts
    return Shape("call", name, style, written, before, below, above, arguments)


def scripted_shape(node: ast.Scripted) -> Shape | None:
    """``(a)_n``, a bracket with one subscript, the Pochhammer symbol; or a name with
    indices, as ``\\epsilon_{ijk}``."""
    arguments = bracketed(node.base)
    if arguments is not None:
        if len(node.scripts) != 1 or node.scripts[0].kind != "_":
            return None
        slot = _slot(node.scripts[0])
        return Shape("call", "", "plain", "", (), (slot,), (), arguments) if slot else None
    if not isinstance(node.base, ast.Name):
        return None
    labels = [
        item
        for script in node.scripts
        for item in (script.indices.items if script.indices is not None else ())
    ]
    if len(labels) < 2 or any(item.label is None for item in labels):
        return None
    name = node.base.id
    return Shape("indexed", name, _style(node.base), node.base.span.text)


def operator_shape(head: ast.Expr, operand: ast.Expr) -> Shape | None:
    """``\\operatorname{tr} A``: a name, which may carry scripts, applied to what follows
    it without brackets."""
    parts = _head(head, loose=True)
    if parts is None:
        return None
    name, style, written, before, below, above = parts
    return Shape("operator", name, style, written, before, below, above, (operand,))


def form_shape(expr: ast.Expr) -> Shape | None:
    """The shape of a written form, whichever it is."""
    if isinstance(expr, ast.Call):
        return call_shape(expr)
    if isinstance(expr, ast.Binary) and expr.op == "*" and expr.implicit:
        return pair_shape(expr.left, expr.right) or operator_shape(expr.left, expr.right)
    if isinstance(expr, ast.Scripted):
        return scripted_shape(expr)
    return None


def bracketed(node: ast.Expr) -> tuple[ast.Expr, ...] | None:
    """The arguments a bracket holds, ``(x)`` or ``(\\theta, \\phi)``; ``None`` for
    anything else, as ``(x)^2``."""
    text = node.span.text.strip()
    if not (text.startswith(_OPENERS) and text.endswith(_CLOSERS)):
        return None
    if isinstance(node, ast.Scripted | ast.Factorial):
        return None
    if isinstance(node, ast.Tuple):
        return tuple(node.items)
    return (node,)


_Head = tuple[str, Style, str, tuple[ast.Node, ...], tuple[ast.Node, ...], tuple[ast.Node, ...]]


def _head(node: ast.Expr, *, loose: bool = False) -> _Head | None:
    """The name a call is written with, how it is set, as it is written, and its scripts,
    before, below and above it, each one item. A name the lexer joined, as ``J_0`` or the
    ``_{2}F_1`` of ``{}_2F_1``, is taken apart. ``loose`` takes a script of any kind."""
    scripts: tuple[ast.Script, ...] = ()
    if isinstance(node, ast.Scripted):
        scripts = node.scripts
        node = node.base
    if not isinstance(node, ast.Name):
        return None
    parts = _parts(node)
    if parts is None:
        return None
    name, before, below = parts
    written = name if before or below else node.span.text  # J of J_0, as it was joined
    above: list[ast.Node] = []
    for script in scripts:
        slot: ast.Node | None = script if loose else _slot(script)
        if slot is None:
            return None
        (below if script.kind == "_" else above).append(slot)
    return name, _style(node), written, tuple(before), tuple(below), tuple(above)


_PRESCRIPT = re.compile(r"_\{(?P<label>[^{}]*)\}(?P<rest>.+)")


def _parts(node: ast.Name) -> tuple[str, list[ast.Node], list[ast.Node]] | None:
    """A name, its prescript and its subscript, where the lexer joined them into one:
    ``J_0`` is J below 0, and ``_{2}F_1`` F with 2 before it and 1 below."""
    text, before = node.id, list[ast.Node]()
    prescript = _PRESCRIPT.fullmatch(text)
    if prescript is not None:
        label = _part(prescript["label"], node)
        if label is None:
            return None
        before.append(label)
        text = prescript["rest"]
    name, underscore, tail = text.partition("_")
    if not underscore:
        return name, before, []
    below: ast.Node | None = _part(tail, node)
    return (name, before, [below]) if below is not None and name else None


def _part(text: str, node: ast.Name) -> ast.Expr | None:
    """One item of a script written in a name, as the 0 of ``J_0``: a number, or a name;
    ``None`` for several, as the ab of ``T_ab``."""
    if re.fullmatch(r"-?\d+", text):
        return ast.Number(text, span=node.span)
    labels = split_index_labels(text)
    return ast.Name(labels[0], span=node.span) if len(labels) == 1 else None


def _slot(script: ast.Script) -> ast.Expr | None:
    """What a script holds, if it is one item: ``\\nu``, ``1/2`` or ``(\\alpha)``; not
    an index list of several, as ``_{ab}``."""
    if script.body is None:
        return None
    if script.indices is not None and len(script.indices.items) != 1:
        return None
    return script.body


def _style(node: ast.Name) -> Style:
    written = node.span.text.strip()
    if written.startswith(_UPRIGHT):
        return "upright"
    if written.startswith(_CALLIGRAPHIC):
        return "calligraphic"
    if node.id in GREEK and (written.startswith("\\") or len(written) == 1):
        return "letter"
    return "plain"


def _in_brackets(slot: ast.Node) -> bool:
    """Whether a script is in brackets of its own, as the order of ``L_n^{(\\alpha)}``."""
    return slot.span.text.strip().lstrip("^_").strip("{}").startswith("(")


# -- the patterns ---------------------------------------------------------------------


@dataclass
class _Patterns:
    """The patterns of every written form, by the name they are written with and by their
    owner, and the capabilities whose calls are printed in their forms, by the SymPy
    function that makes the calls, with which parameter each argument of a call is."""

    by_head: dict[str, list[Pattern]]
    by_owner: dict[int, list[Pattern]]
    printing: dict[type, tuple[Capability, _Places]] | None = None


_found: _Patterns | None = None


def _patterns() -> _Patterns:
    global _found
    if _found is None:
        by_head: dict[str, list[Pattern]] = {}
        by_owner: dict[int, list[Pattern]] = {}
        for pattern in _parsed():
            by_head.setdefault(pattern.shape.head, []).append(pattern)
            by_owner.setdefault(id(pattern.owner), []).append(pattern)
        _found = _Patterns(by_head, by_owner)
    return _found


def patterns() -> Mapping[str, list[Pattern]]:
    """The patterns of every written form, by the name they are written with: those of
    capabilities first, then those of the book names reserved."""
    return _patterns().by_head


def forms_of(owner: Capability | Reservation) -> Sequence[Pattern]:
    """The patterns of the written forms of a capability or a reservation, in the order
    it declares them; a form that does not parse into a call has none."""
    return _patterns().by_owner.get(id(owner), [])


def _forget() -> None:
    global _found
    _found = None


on_change(_forget)


def _parsed() -> Iterator[Pattern]:
    found = catalogue()
    for capability in found.capabilities:
        for form, written in (("latex", capability.latex), ("unicode", capability.unicode)):
            for each in written:
                pattern = parse_form(capability, each, form)  # type: ignore[arg-type]
                if pattern is not None:
                    yield pattern
    for reservation in found.reservations:
        for each in reservation.written:
            pattern = parse_form(reservation, each, "latex")
            if pattern is not None:
                yield pattern


def parse_form(
    owner: Capability | Reservation, written: str, form: Literal["latex", "unicode"]
) -> Pattern | None:
    """A written form, parsed by the language into a pattern; ``None`` where it does not
    parse, or is no call, which the conformance suite says."""
    from relativisticpy.language import parse

    try:
        statements = parse(written).statements
    except RelativisticPyError:
        return None
    if len(statements) != 1 or not isinstance(statements[0], ast.Evaluate):
        return None
    shape = form_shape(statements[0].expr)
    return Pattern(owner, written, form, shape) if shape is not None else None


def matching(shape: Shape) -> Match | None:
    """The pattern a shape has, with what each of its variables is; ``None`` where none
    has it. A capability's pattern comes before a reservation's."""
    for pattern in patterns().get(shape.head, ()):
        bindings = _binds(pattern, shape)
        if bindings is not None:
            return Match(pattern, bindings)
    return None


def _binds(pattern: Pattern, shape: Shape) -> dict[str, ast.Node] | None:
    form = pattern.shape
    if form.kind != shape.kind or form.bracket != shape.bracket or not _set_alike(form, shape):
        return None
    counts = (len(form.before), len(form.below), len(form.above), len(form.arguments))
    if shape.kind == "indexed":
        return {}
    if counts != (len(shape.before), len(shape.below), len(shape.above), len(shape.arguments)):
        return None
    if [_in_brackets(s) for s in form.above] != [_in_brackets(s) for s in shape.above]:
        return None
    if isinstance(pattern.owner, Reservation):
        return {}
    bindings: dict[str, ast.Node] = {}
    scripts = len(form.scripts)
    for place, (slot, written) in enumerate(
        zip((*form.scripts, *form.arguments), (*shape.scripts, *shape.arguments), strict=True)
    ):
        if isinstance(slot, ast.Name):
            if place < scripts and _a_label(written):
                return None
            bound = bindings.setdefault(slot.id, written)
            if bound is not written and bound != written:
                return None
        elif slot != written:
            return None
    return bindings


def _a_label(script: ast.Node) -> bool:
    """Whether a script names what a quantity is, as eff does in ``U_{\\text{eff}}``, the
    effective potential, rather than holding a parameter, as the n of ``U_n`` does: text, a
    word in roman type, or a word of letters written out."""
    written = script.span.text.strip().lstrip("^_").strip("{} ")
    if _LABEL.match(written):
        return True
    return re.fullmatch(r"[A-Za-z]{2,}", written) is not None


def _set_alike(form: Shape, shape: Shape) -> bool:
    """Whether a name is set as the form sets it: a single letter, Latin, in the same type
    (J is not \\mathrm{J}); a Greek letter as a letter, or spelled out, or upright; and a
    word in any type but a letter's, so that the form zeta(s) is never \\zeta(s)."""
    if len(form.head) == 1 and form.head.isascii():
        return form.style == shape.style
    if form.style == "letter":
        return shape.style != "calligraphic"
    return shape.style in ("plain", "upright")


# -- printing --------------------------------------------------------------------------


def _printers() -> dict[type, tuple[Capability, _Places]]:
    """The capabilities with written forms whose calls are printed in them, by the SymPy
    function that makes the calls; not one whose function another capability makes too,
    as the principal logarithm makes log, which ln and log make."""
    found = _patterns()
    if found.printing is None:
        makers: dict[type, list[tuple[Capability, _Places]]] = {}
        for capability in catalogue().capabilities:
            if isinstance(capability, Function):
                made = _made_by(capability)
                if made is not None:
                    makers.setdefault(made[0], []).append((capability, made[1]))
        found.printing = {
            kind: owners[0]
            for kind, owners in makers.items()
            if len(owners) == 1 and (owners[0][0].latex or owners[0][0].unicode)
        }
    return found.printing


#: Which parameter each argument of a call of a capability is, by its name, or the value
#: the argument always has, as the 1/2 of Heaviside(t, 1/2), Heaviside(t)'s value at 0;
#: ``None`` where the capability says how a call prints (an ``arguments`` method).
_Places = tuple[str | sympy.Basic, ...] | None


def _made_by(capability: Capability) -> tuple[type, _Places] | None:
    """The SymPy function a call of a capability makes, and what each of its arguments
    is, found by calling ``exact`` with a symbol for each parameter it needs; ``None``
    where it gives no call, as a closed form."""
    low, _ = capability.arity
    names = capability.parameters[:low]
    symbols = [sympy.Dummy(name) for name in names]
    try:
        made = capability.exact(*symbols)
    except Exception:
        return None
    if not isinstance(made, sympy.Function) or isinstance(made, AppliedUndef):
        return None
    if hasattr(capability.implementation, "arguments"):
        return type(made), None
    if sorted(map(str, symbols)) != sorted(str(a) for a in made.args if a in symbols):
        return None  # each parameter an argument, once
    places = tuple(names[symbols.index(arg)] if arg in symbols else arg for arg in made.args)
    return type(made), places


def printed(
    expr: Any, form: Literal["latex", "unicode"], show: Callable[[Any], str]
) -> tuple[str, bool] | None:
    """A call of a capability, as the first of its forms in ``form`` that its arguments
    fill writes it, each shown by ``show``; with whether it is written with scripts, so
    that a power puts it in brackets. ``None`` for anything else, or a capability with no
    form there."""
    if not isinstance(expr, sympy.Function):
        return None
    found = _printers().get(type(expr))
    if found is None:
        return None
    capability, parameters = found
    values = _values(capability, parameters, expr)
    for pattern in forms_of(capability):
        if pattern.form != form:
            continue
        shown = _render(pattern, values, show, latex=form == "latex")
        if shown is not None:
            return shown, bool(pattern.shape.scripts) or not pattern.shape.head
    return None


def _values(capability: Capability, places: _Places, expr: Any) -> dict[str, Any]:
    """The value of each parameter of a call, with the default of each the call does not
    give; none where the call is not one ``exact`` makes, as W_{-1}(x) with its branch."""
    if places is None:
        found: dict[str, Any] = capability.implementation.arguments(expr)
        return found
    if len(places) != len(expr.args):
        return {}
    values: dict[str, Any] = {}
    for place, value in zip(places, expr.args, strict=True):
        if isinstance(place, str):
            values[place] = value
        elif place != value:
            return {}
    for parameter in capability.signature.parameters.values():
        if parameter.name not in values and parameter.default is not parameter.empty:
            values[parameter.name] = parameter.default
    return values


def _render(
    pattern: Pattern, values: Mapping[str, Any], show: Callable[[Any], str], *, latex: bool
) -> str | None:
    shape = pattern.shape
    parts: dict[int, str] = {}
    for slot in (*shape.scripts, *shape.arguments):
        if isinstance(slot, ast.Name):
            if slot.id not in values:
                return None
            value = values[slot.id]
            listed = pattern.separator == ";" and isinstance(value, list | tuple | sympy.Tuple)
            parts[id(slot)] = ", ".join(map(show, value)) if listed else show(value)
        else:
            parts[id(slot)] = slot.text if isinstance(slot, ast.Number) else slot.span.text

    def script(mark: str, slot: ast.Node) -> str:
        text = parts[id(slot)]
        if _in_brackets(slot):
            text = f"({text})"
        if not latex and (len(text) == 1 or text.isdigit()):
            return f"{mark}{text}"
        return f"{mark}{{{text}}}"

    before = "".join("{}" + script("_", slot) for slot in shape.before)
    scripts = "".join(script("_", slot) for slot in shape.below)
    scripts += "".join(script("^", slot) for slot in shape.above)
    separator = f"{pattern.separator} "
    arguments = separator.join(parts[id(slot)] for slot in shape.arguments)
    opened, closed = ("\\left(", "\\right)") if latex else ("(", ")")
    if not shape.head:  # (a)_n
        return f"{opened}{arguments}{closed}{scripts}"
    return f"{before}{shape.written}{scripts}{opened}{arguments}{closed}"
