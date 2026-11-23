"""What names and forms are written: runs of letters, decorated and primed names,
subscripted symbols, line elements, the names a statement writes as factors and those a
value is written with, and the tables of the built-in names they are read against."""

from __future__ import annotations

import difflib
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import replace
from functools import reduce
from typing import Any

import sympy
from sympy import Symbol
from sympy.core.function import AppliedUndef

from relativisticpy import display, spacetimes
from relativisticpy.compiler.builtins import _SPACETIMES, BUILTINS, CONSTANTS, SPACETIMES
from relativisticpy.compiler.scope import Scope, VariableBinding
from relativisticpy.compiler.session import ASSUMPTION_STATEMENTS, RENAMING_STATEMENTS, MetricState
from relativisticpy.errors import CompileError
from relativisticpy.integrations import MapFunction
from relativisticpy.integrations._discovery import catalogue
from relativisticpy.ir import SCALAR, Node, NotationType, TensorType, dependencies, ops
from relativisticpy.language import syntax as ast
from relativisticpy.language.parser import split_index_labels
from relativisticpy.language.source import Span
from relativisticpy.language.vocabulary import FUNCTION_WORDS, GREEK, SYMBOL_COMMANDS
from relativisticpy.tensors import Indices
from relativisticpy.tensors.indices import latex_label
from relativisticpy.typing.syntax import _plain_label, _tensor_pattern, _used, _walk

#: Built-in tensors by name and number of indices.
_BUILTIN_TENSORS: dict[str, dict[int, str]] = {
    "Gamma": {3: "christoffel"},
    "R": {4: "riemann", 2: "ricci"},
    "delta": {2: "kronecker"},
}


_ROLE_TENSORS = {
    "connection": {3: "christoffel"},
    "riemann": {4: "riemann", 2: "ricci"},
    "ricci": {2: "ricci"},
    "einstein": {2: "einstein"},
    "weyl": {4: "weyl"},
}


#: Functions that are linear algebra when given a tensor.
_LINEAR_ALGEBRA = frozenset(
    {"eig", "spec", "svd", "det", "tr", "trace", "inv", "diag", "exp", "sqrt", "log"}
)


#: Names of statistics written as operators, for the statistic they mean.
_STATISTIC_ALIASES = {
    "E": "\\mathbb{E}",
    "Mean": "\\mathbb{E}",
    "mean": "\\mathbb{E}",
    "Expectation": "\\mathbb{E}",
    "Variance": "\\operatorname{Var}",
    "Covariance": "\\operatorname{Cov}",
}


#: The statistics written with the index they are over, as \mathbb{E}_t.
_STATISTICS = {
    "𝔼": "expectation",
    "Var": "variance",
    "var": "variance",
    "Cov": "covariance",
    "cov": "covariance",
    "LedoitWolf": "ledoit-wolf",
}


def _statistic(node: ast.Expr) -> tuple[str, str] | None:
    """``\\mathbb{E}_t``, ``\\operatorname{Var}_t``, ...: the statistic and its index."""
    if not (
        isinstance(node, ast.Scripted)
        and isinstance(node.base, ast.Name)
        and node.base.id in _STATISTICS
        and len(node.scripts) == 1
        and node.scripts[0].kind == "_"
        and node.scripts[0].indices is not None
        and len(node.scripts[0].indices.items) == 1
        and _plain_label(node.scripts[0].indices.items[0])
    ):
        return None
    return _STATISTICS[node.base.id], str(node.scripts[0].indices.items[0].label)


# Constants that an equation can only make a claim about: \pi = 3.14159 is not a new \pi.
# (e and i are often named afresh, as a charge or an index.)
_CLAIMED = frozenset({"pi", "infty", "oo"})


# Statements that declare, as := does, whether written with := or =.
_DECLARATIONS = frozenset({"Coordinates", *ASSUMPTION_STATEMENTS, *RENAMING_STATEMENTS})


# The built-in scalars, by the geometric quantity a name denotes.
_BUILT_IN_SCALARS = {
    "determinant": "the determinant of the metric",
    "ricci_scalar": "the Ricci scalar",
    "kretschmann": "the Kretschmann scalar",
}


# The spacetimes, by the names the language calls them.
_SPACETIME_NAMES = tuple(name for name in BUILTINS if name in SPACETIMES)


#: The built-in tensors and scalars by the names they are spelled out with, lower case,
#: as an agent may guess them: what to write instead.
_CONCEPTS = {
    "ricci": "Ric, the Ricci scalar, or R_{\\mu\\nu}, the Ricci tensor",
    "ricciscalar": "R or Ric, the Ricci scalar",
    "riccitensor": "R_{\\mu\\nu}, the Ricci tensor",
    "riemann": "R^{\\rho}{}_{\\sigma\\mu\\nu}, the Riemann tensor",
    "riemanntensor": "R^{\\rho}{}_{\\sigma\\mu\\nu}, the Riemann tensor",
    "christoffel": "\\Gamma^{\\lambda}_{\\mu\\nu}, the Christoffel symbols",
    "christoffelsymbols": "\\Gamma^{\\lambda}_{\\mu\\nu}, the Christoffel symbols",
    "einstein": "G_{\\mu\\nu}, the Einstein tensor",
    "einsteintensor": "G_{\\mu\\nu}, the Einstein tensor",
    "weyl": "C_{\\mu\\nu\\rho\\sigma}, the Weyl tensor",
    "weyltensor": "C_{\\mu\\nu\\rho\\sigma}, the Weyl tensor",
    "metric": "g_{\\mu\\nu}, the metric",
    "metrictensor": "g_{\\mu\\nu}, the metric",
    "curvaturescalar": "R or Ric, the Ricci scalar",
    "kretschmannscalar": "Kretschmann, or R_{abcd} R^{abcd}",
}


#: Names of built-in tensors that are as often names of one's own: a covariance C,
#: weights W, a matrix G.
_SHARED_NAMES = frozenset({"C", "G", "W"})


#: Words written as subscripts, as in T_{eff}: names, not one index per letter.
_SUBSCRIPT_WORDS = frozenset(
    {
        "eff", "vac", "max", "min", "tot", "obs", "crit", "irr", "ext", "int", "rad",
        "kin", "pot", "rel", "ret", "adv", "out", "in", "cm", "lab", "rest", "sun", "orb",
        "com", "phys", "BH", "ADM", "Pl", "em", "dm", "de",
    }
)  # fmt: skip


#: Superscripts that write a transpose, as in A^T.
_TRANSPOSES = frozenset({"T", "top", "intercal"})


def _a_word(run: str) -> bool:
    """Whether a subscript written as a run of letters is a word that names something,
    as eff in T_{eff}, rather than one index per letter, as in T_{abc}: a word often
    written as a subscript, or three letters or more of which some repeat."""
    if not (run.isalpha() and run.isascii()):
        return False
    return run in _SUBSCRIPT_WORDS or (len(run) >= 3 and len(set(run)) < len(run))


def _has_a_value(written: str) -> str:
    """The note on ``K = ...`` once K has a value: a claim, not a new value."""
    return (
        f"{written} already has a value, so = compares it; to give {written} a new value, "
        f"write {written} := ..."
    )


def _built_in(written: str, what: str, again: str) -> str:
    """The note on ``R = 0``: a claim about what a built-in name already means."""
    return (
        f"{written} is {what}, so = compares it; to give {written} a meaning of your own, "
        f"write {again}"
    )


def _late_note(written: str, metric: MetricState | None) -> str:
    """The note on ``\\Delta = ...`` after a line element written with \\Delta."""
    span = metric.node.span if metric is not None else None
    where = f"on line {span.line}" if span is not None else "before"
    return (
        f"the metric defined {where} is written with {written}, which has this value there "
        "from now on"
    )


def _shown(name: str, latex: bool) -> str:
    """A name as the user writes it: ``\\Delta`` in LaTeX, ``Delta`` in plain text."""
    return latex_label(name) if latex else name


def _close(name: str, candidates: Iterable[str], shortest: int = 4) -> str | None:
    """The candidate a name is likely a misspelling of, as Schwarzchild is of
    Schwarzschild or simplfy of simplify; ``None`` where none is close. Names of fewer
    than four letters are never matched, and candidates shorter than ``shortest`` never
    offered, so that ln is no misspelt N, nor erfc a misspelt erf."""
    if len(name) < 4:
        return None
    known: dict[str, str] = {}
    for candidate in candidates:  # the first spelling of each, as Cov before cov
        if len(candidate) >= shortest:
            known.setdefault(candidate.lower(), candidate)
    found = difflib.get_close_matches(name.lower(), list(known), n=1, cutoff=0.8)
    return known[found[0]] if found else None


def _splittable(node: ast.Name) -> bool:
    """Whether a name, in LaTeX, may be a run of letters, each a factor, as GM is: not a
    name set as one, a Greek letter's name, or the name of one of LaTeX's functions written
    in letters, as sin is, whose i is never the imaginary unit."""
    name = node.id
    return (
        not node.verbatim
        and len(name) > 1
        and name.isalpha()
        and name not in GREEK
        and name not in FUNCTION_WORDS
    )


def _side_by_side(left: ast.Expr, right: ast.Expr) -> bool:
    """Whether two factors are written side by side, with nothing but space between them,
    as D and f(x) are in ``D f(x)``, and not in ``D \\cdot f(x)``."""
    if left.span.source is not right.span.source or left.span.end > right.span.start:
        return False
    between = Span(left.span.source, left.span.end, right.span.start).text
    for space in ("\\,", "\\;", "\\:", "\\ ", "~"):
        between = between.replace(space, "")
    return not between.strip()


#: How a bracket that groups begins: (, [, \{, \left( or \left[, and \big( or \Bigl(.
_GROUP_OPENERS = ("(", "[", "\\{", "\\lbrace", "\\lbrack", "\\left", "\\big", "\\Big")


def _in_brackets(node: ast.Name) -> bool:
    """Whether a name is written in brackets of its own, as xy is in ``(xy)^2``: the
    parser gives it the span of its brackets, which then begins with one."""
    return node.span.text != node.id and node.span.text.startswith(_GROUP_OPENERS)


#: LaTeX's functions whose argument books write right after the name, as in sin x, ln x and
#: e^x: a name that begins with one and goes on with an argument, as sinx, runs them
#: together. Not det, dim, arg, min and the like, which begin names of their own, as args.
_ARGUMENT_AFTER = frozenset(
    {
        "sin", "cos", "tan", "cot", "sec", "csc", "sinh", "cosh", "tanh", "coth", "sech",
        "csch", "arcsin", "arccos", "arctan", "arccot", "exp", "ln", "log",
    }
)  # fmt: skip
_ARGUMENT = re.compile(
    r"[0-9]*(?:" + "|".join(sorted(GREEK, key=len, reverse=True)) + r"|[A-Za-z])|[0-9]+"
)


def _a_differential(name: str) -> bool:
    """Whether a name reads as a differential, d and a letter or a Greek letter's name, as
    dx and dtheta do."""
    return name.startswith("d") and (len(name) == 2 or name[1:] in GREEK)


def glued_function(name: str) -> tuple[str, str] | None:
    """The function and the argument that a name runs together, as ``sinx`` runs sin and
    x, ``cos2theta`` cos and 2theta, and ``ln2`` ln and 2: one of LaTeX's functions whose
    argument is written after it, then a letter or a Greek letter's name, maybe after a
    number, or a number alone. ``None`` for any other name, as sinh, cost2 and logic."""
    if name in FUNCTION_WORDS:
        return None
    for word in sorted(_ARGUMENT_AFTER, key=len, reverse=True):
        if name.startswith(word) and _ARGUMENT.fullmatch(name, len(word)):
            return word, name[len(word) :]
    return None


def _letter(label: str, span: Span, *, first: bool) -> ast.Name:
    """One letter, or Greek name, of a run of letters read as a product, as G is of GM.
    No letter is split again. A d is the letter d, as in ad - bc, except first in its run,
    where it is the d of a differential, as in ds or dx, as {\\rm d} is."""
    return ast.Name(label, verbatim=first and label == "d", span=span)


def _written_as_letter(node: ast.Name) -> bool:
    """Whether a name is a Greek letter written as one, ``\\gamma`` or ``γ``, rather than
    spelled out as a function's name is, as in ``gamma(x)``."""
    if node.id not in GREEK:
        return False
    written = node.span.text
    return written == "\\" + node.id or len(written) == 1  # the command, or the character


def accented(name: str, accent: str) -> str:
    """The name of a symbol under an accent: ``xbar`` of ``\\bar{x}``, which SymPy prints as
    \\bar{x} too; but where that would spell another symbol's name, as hbar of ``\\bar{h}``
    spells ℏ's, the letter with the accent Unicode sets over it, ``h̄``, shown as \\bar{h}."""
    joined = name + accent
    if joined in SYMBOL_COMMANDS.values():
        return name + display.ACCENTS[accent]
    return joined


def _named(target: ast.Expr) -> ast.Expr:
    """A decorated name, as ``\\tilde u`` or ``E^{(2)}``, as the name it is read as,
    ``utilde`` or ``E^(2)``; a dot is a time derivative, and anything else is left as it
    is."""
    if (
        isinstance(target, ast.Accent)
        and target.accent not in ("dot", "ddot")
        and isinstance(target.base, ast.Name)
    ):
        name = accented(target.base.id, target.accent)
        return ast.Name(name, verbatim=True, span=target.span)
    if (
        isinstance(target, ast.Scripted)
        and isinstance(target.base, ast.Name)
        and all(script.kind == "^" and _decoration(script) for script in target.scripts)
    ):
        marks = "".join(_mark(script) for script in target.scripts)
        return ast.Name(f"{target.base.id}^{marks}", verbatim=True, span=target.span)
    return target


def _mark(script: ast.Script) -> str:
    """How a decoration marks a name: ``*`` in ``r^*``, ``(2)`` in ``E^{(2)}``."""
    body = script.body
    if isinstance(body, ast.Name) and not _bracketed_order(body):
        return body.id
    return body.span.text if body is not None else ""


def _a_component(script: ast.Script) -> bool:
    """Whether a superscript can only be an index, never a power: a lone 0, as in J^0, or
    digits that are no number as numbers are written, as 00 and 01 in \\eta^{00} and
    F^{01}; or index letters beside a 0, as in F^{0i} and F^{\\mu 0}, or before a digit,
    as in F^{\\mu 1}. A number written as numbers are, as 2 and 10, is an exponent, and so
    is a number before letters, as 2n is in x^{2n}."""
    if script.kind != "^" or script.indices is None or not script.indices.items:
        return False
    items = script.indices.items
    kinds = []  # each item a digit or a letter, in order
    for item in items:
        if isinstance(item.value, ast.Number) and item.label is None:
            if not item.value.text.isdigit():
                return False
            kinds.append(item.value.text)
        elif _plain_label(item):
            kinds.append("")
        else:
            return False
    digits = [kind for kind in kinds if kind]
    if not digits:
        return False
    if len(digits) == len(kinds):  # 0, 00 and 01, but not 2 or 10
        return digits[0] == "0"
    first_letter = kinds.index("")
    return "0" in digits or any(kinds[first_letter:])  # 0i and \mu 0, \mu 1; not 2n


def _decoration(script: ast.Script) -> bool:
    """A script that marks a name rather than raising it to a power: ``r^*``, ``i^+``, or
    an order in brackets, as of a perturbation, ``E^{(2)}``."""
    body = script.body
    if isinstance(body, ast.Name) and body.verbatim and body.id in ("*", "+", "-", "±", "∓"):
        return True
    return _bracketed_order(body)


def _bracketed_order(body: ast.Expr | None) -> bool:
    """``(2)`` or ``(n)``: a whole number or one letter in brackets."""
    if isinstance(body, ast.Number):
        whole = body.text.isdigit()
    else:
        whole = isinstance(body, ast.Name) and (len(body.id) == 1 or body.id in GREEK)
    text = body.span.text if body is not None else ""
    return whole and text.startswith("(") and text.endswith(")")


def _script_labels(statement: ast.Node) -> dict[str, Counter[str]]:
    """How often a statement writes each index label as a subscript (``"_"``), as
    \\mu in x_\\mu, and as a superscript (``"^"``)."""
    counts: dict[str, Counter[str]] = {"_": Counter(), "^": Counter()}
    for node in _walk(statement):
        if isinstance(node, ast.Script) and node.indices is not None:
            counts[node.kind].update(
                str(item.label) for item in node.indices.items if _plain_label(item)
            )
    return counts


def _names_inside(name: str, node: ast.Expr) -> bool:
    """Whether ``name`` is written in ``node`` other than as a function called there: M
    is in r - 2M, and f is not in f(x)."""
    called = {id(n.func) for n in _walk(node) if isinstance(n, ast.Call)}
    return any(
        isinstance(n, ast.Name) and n.id == name and id(n) not in called for n in _walk(node)
    )


def _unprimed(node: ast.Expr, *, every: bool = False) -> ast.Expr:
    """A primed name with subscripts, as ``R'_{\\mu\\nu}``, read as the name ``R'`` with
    them: a tensor or symbol of its own. A power of a primed name, ``A'^2``, is left as
    it is, to be the power of what ``A'`` means; with ``every``, as on the left of := or of
    a tensor k', a superscript too is the primed name's, as in ``k'^\\mu``."""
    if (
        isinstance(node, ast.Scripted)
        and isinstance(node.base, ast.Prime)
        and (every or any(script.kind == "_" for script in node.scripts))
    ):
        prime = node.base
        name = ast.Name(prime.func.id + "'" * prime.order, verbatim=True, span=prime.span)
        return replace(node, base=name)
    return node


def _names_builtin(node: ast.Name, *, latex: bool = False) -> bool:
    """Whether a name, bound to nothing in the program, is a builtin function: never a
    Greek letter written as one, so that \\gamma(r) in a metric is an unknown function,
    and in LaTeX never a single letter, so that N(t) there is the lapse, not ``N``."""
    if latex and len(node.id) == 1:
        return False
    return node.id in BUILTINS and not _written_as_letter(node)


def _names(value: ast.Expr, what: str) -> list[str]:
    """The names listed in ``[t, r, \\theta]``, a one-row matrix, or a single name."""
    if isinstance(value, ast.Name):
        return [value.id]
    if isinstance(value, ast.List | ast.Tuple):
        items: Sequence[ast.Expr] = value.items
    elif isinstance(value, ast.Matrix) and (len(value.rows) == 1 or len(value.rows[0]) == 1):
        items = [cell for row in value.rows for cell in row]
    else:
        raise CompileError(f"{what} lists names, as in [t, r, \\theta, \\phi]", span=value.span)
    names = []
    for item in items:
        if not isinstance(item, ast.Name):
            raise CompileError("expected a name", span=item.span)
        names.append(item.id)
    return names


def _indexed_target(name: str, value: Node) -> tuple[str, Indices] | None:
    """``v_a := ...`` with a value whose free indices are exactly ``a``: the tensor
    ``v_{a}``, with its indices where the value has them. A value whose indices no one
    named, as the eigenvalues of s_k := \\operatorname{spec}(S), takes them in order."""
    base, underscore, tail = name.partition("_")
    if not (underscore and tail.isalpha() and isinstance(value.type, TensorType)):
        return None
    labels = split_index_labels(tail)
    indices = value.type.indices
    if len(labels) == len(indices) and all(index.name.startswith("#") for index in indices):
        return base, indices  # named by the labels where they are written, s_k or s_{k}
    have = {index.name: index for index in indices}
    if sorted(labels) != sorted(have):
        return None
    return base, Indices(have[label] for label in labels)


def _is_line_element(target: ast.Expr) -> bool:
    """``ds^2`` (also ``ds^{2}``, ``(ds)^2``, ``d s^2`` and, in Python's way, ``ds**2``)."""

    def squared(node: ast.Expr, name: str) -> bool:
        if (
            isinstance(node, ast.Binary)
            and node.op == "**"
            and isinstance(node.left, ast.Name)
            and node.left.id == name
        ):
            return isinstance(node.right, ast.Number) and node.right.text == "2"
        return (
            isinstance(node, ast.Scripted)
            and isinstance(node.base, ast.Name)
            and node.base.id == name
            and len(node.scripts) == 1
            and node.scripts[0].kind == "^"
            and isinstance(node.scripts[0].body, ast.Number)
            and node.scripts[0].body.text == "2"
        )

    if squared(target, "ds"):
        return True
    return (
        isinstance(target, ast.Binary)
        and target.op == "*"
        and isinstance(target.left, ast.Name)
        and target.left.id == "d"
        and squared(target.right, "s")
    )


_DIFFERENTIAL_NAME = re.compile(
    r"d(" + "|".join(sorted(GREEK, key=len, reverse=True)) + r"|[A-Za-z][0-9]*)"
)


def _line_element_coordinates(value: ast.Expr, *, solid_angle: bool = True) -> list[str]:
    r"""The coordinates a line element's differentials name, in the order they appear:
    ``-dt^2 + dr^2 + r^2 d\Omega^2`` names t, r, θ and φ, or, without ``solid_angle``, t, r
    and Ω. Empty if a differential is abstract, as in ``g_{\mu\nu} dx^\mu dx^\nu``, or if
    there are none."""
    atoms: list[ast.Expr] = []
    pending: list[ast.Expr] = [value]
    while pending:  # left to right, without recursion
        node = pending.pop()
        if isinstance(node, ast.Binary):
            pending.append(node.right)
            pending.append(node.left)
        elif isinstance(node, ast.Unary):
            pending.append(node.operand)
        else:
            atoms.append(node)
    names: list[str] = []
    for atom, following in zip(atoms, [*atoms[1:], None], strict=True):
        base = atom.base if isinstance(atom, ast.Scripted) else atom
        if isinstance(base, ast.Name) and base.id == "d" and following is not None:
            target = following
        elif isinstance(base, ast.Name) and (match := _DIFFERENTIAL_NAME.fullmatch(base.id)):
            target = ast.Name(match.group(1), span=base.span)
        else:
            continue
        if isinstance(target, ast.Scripted):
            if not all(isinstance(s.body, ast.Number) for s in target.scripts):
                return []  # dx^\mu: an abstract differential
            target = target.base
        if not isinstance(target, ast.Name):
            continue
        found = ["theta", "phi"] if target.id == "Omega" and solid_angle else [target.id]
        names.extend(name for name in found if name not in names)
    return names


def _names_omega(node: ast.Expr) -> bool:
    """``\\Omega`` or ``\\Omega^2``, as after the ``d`` of a solid angle."""
    base = node.base if isinstance(node, ast.Scripted) else node
    return isinstance(base, ast.Name) and base.id == "Omega"


def _factor_names(statement: ast.Node) -> frozenset[str]:
    """The names a statement writes as factors rather than as functions it calls: G and
    M in 2GM, where in LaTeX a run of letters is a product of them, but neither f in
    f(x) or f'(x), nor what a substitution sets, as r in |_{r=r(\\tau)}."""
    other = _other_names(statement)
    for node in _walk(statement):
        if isinstance(node, ast.Substitution):
            other.update(id(target) for target, _ in node.bindings)
        elif isinstance(node, ast.Evaluation) and node.variable is not None:
            other.add(id(node.variable))
    latex = bool(getattr(statement, "latex", False))
    found: set[str] = set()
    for node in _walk(statement):
        if isinstance(node, ast.Name) and id(node) not in other:
            found.add(node.id)
            known = node.id in BUILTINS or node.id in CONSTANTS
            if latex and _splittable(node) and not known:
                found.update(split_index_labels(node.id))
    return frozenset(found)


def _other_names(node: ast.Node) -> set[int]:
    """The ids of the names in an expression that are part of another name, or of a
    function's: f in f(x), x in x', h in \\bar h, E in E^{(2)}."""
    found: set[int] = set()
    for current in _walk(node):
        if isinstance(current, ast.Call | ast.Prime):
            found.add(id(current.func))
        elif isinstance(current, ast.Accent | ast.Scripted) and isinstance(
            _named(current), ast.Name
        ):  # \bar x is xbar, and E^{(2)} is E^(2); \dot h is h, differentiated
            found.add(id(current.base))
    return found


def _writes_tensor(value: ast.Expr, name: str) -> bool:
    """Whether ``value`` writes the tensor ``name`` with index labels, or ``name`` alone,
    other than as the tensor solve is asked to solve for: T_{\\nu\\mu} in
    T_{\\mu\\nu} = T_{\\nu\\mu}, but not k in k^3, a power or a component."""
    bases = {id(node.base) for node in _used(value) if isinstance(node, ast.Scripted)}
    for node in _used(value):
        if isinstance(node, ast.Scripted) and (pattern := _tensor_pattern(node)):
            if pattern[0] == name:
                return True
        elif isinstance(node, ast.Name) and node.id == name and id(node) not in bases:
            return True
    return False


def _slashed(factors: list[ast.Expr]) -> list[ast.Expr]:
    """Factors with an inline slash read as in physics: ``x/4GM`` is x/(4GM), and
    ``e^{-r/2GM}`` is e^{-r/(2GM)}. The letters and numbers written right after the
    denominator, with no space, belong to it; ``x/4 GM``, with a space, and
    ``\\frac{x}{4}GM`` are (x/4) GM, and a tensor never joins a denominator."""

    def simple(node: ast.Expr) -> bool:
        if isinstance(node, ast.Number | ast.Name):
            return True
        return (
            isinstance(node, ast.Scripted)
            and isinstance(node.base, ast.Name)
            and _tensor_pattern(node) is None
            and all(script.kind == "^" for script in node.scripts)
        )

    result: list[ast.Expr] = []
    position = 0
    while position < len(factors):
        factor = factors[position]
        slash = (
            isinstance(factor, ast.Binary)
            and factor.op == "/"
            and Span(factor.span.source, factor.left.span.end, factor.right.span.start).text.strip()
            == "/"
            and simple(factor.right)
            and factor.span.start == factor.left.span.start  # not \{1/s^2\}(t), in brackets
        )
        run: list[ast.Expr] = []
        end = factor.span.end
        while slash and position + 1 + len(run) < len(factors):
            following = factors[position + 1 + len(run)]
            if following.span.start != end or not simple(following):
                break
            run.append(following)
            end = following.span.end
        if not run:
            result.append(factor)
            position += 1
            continue
        assert isinstance(factor, ast.Binary)
        denominator = reduce(
            lambda a, b: ast.Binary("*", a, b, implicit=True, span=a.span.to(b.span)),
            [factor.right, *run],
        )
        span = factor.span.to(run[-1].span)
        result.append(ast.Binary("/", factor.left, denominator, span=span))
        position += 1 + len(run)
    return result


def _greek_labels(scripts: Sequence[ast.Script]) -> bool:
    """Whether scripts are index labels, all Greek, as \\alpha in u_\\alpha."""
    items = [item for s in scripts if s.indices is not None for item in s.indices.items]
    return bool(items) and all(_plain_label(item) and item.label in GREEK for item in items)


def _styled(span: Span) -> Span:
    """A span grown over the style command around it, as B to \\mathcal{B}."""
    text = span.source.text
    for command in ("\\mathcal{", "\\mathbb{", "\\mathscr{", "\\mathfrak{", "\\mathrm{"):
        start = span.start - len(command)
        if text[max(start, 0) : span.start] == command and text[span.end : span.end + 1] == "}":
            return Span(span.source, start, span.end + 1)
    return span


def _an_index(before: ast.Expr, factor: ast.Expr) -> bool:
    """Whether ``factor`` is written as Python indexes a list: a whole number in square
    brackets, right after what it would index, as [0] in [1, 2][0]."""
    return (
        isinstance(factor, ast.List)
        and len(factor.items) == 1
        and isinstance(factor.items[0], ast.Number)
        and factor.items[0].text.isdigit()
        and factor.span.text.startswith("[")
        and before.span.end == factor.span.start
    )


def _bracketed(equation: ast.Equation) -> bool:
    """Whether an equation is written in brackets of its own, as the case ``(k = 0)``
    beside a formula is."""
    text = equation.span.text
    for opener, closer in (("(", ")"), ("\\left(", "\\right)")):
        if (
            text.startswith(opener)
            and text.endswith(closer)
            and equation.lhs.span.start > equation.span.start
            and equation.rhs.span.end < equation.span.end
        ):
            return True
    return False


def _unbracketed(equation: ast.Equation) -> str:
    return Span(equation.span.source, equation.lhs.span.start, equation.rhs.span.end).text


def _mentioned(value: ast.Expr, *, latex: bool) -> frozenset[str]:
    """Every name written in an expression and, in LaTeX, every letter of a run of
    letters read as a product, as GM is in \\frac{2GM}{r}. The name of a builtin, such as
    FLRW in FLRW(k) or diag, is one name and never its letters."""
    found: set[str] = set()
    for node in _walk(value):
        if isinstance(node, ast.Name):
            found.add(node.id)
            builtin = node.id in BUILTINS or node.id in CONSTANTS
            if latex and _splittable(node) and not builtin:
                found.update(split_index_labels(node.id))
    return frozenset(found)


def _mentions_symbol(node: Node, name: str) -> bool:
    """Whether a value is written with the symbol ``name``: a constant it is computed
    from has it among its free symbols."""
    for value in _constants(node):
        written = isinstance(value, sympy.Basic | sympy.NDimArray)
        if written and any(symbol.name == name for symbol in value.free_symbols):
            return True
    return False


def _constants(root: Node) -> Iterator[Any]:
    """The values of the constants a value is computed from, as it is written: not those
    of notation, which is no expression, but those of the scalars of notation that its
    components are computed with."""
    pending, seen = [root], set()
    while pending:
        node = pending.pop()
        if node.id in seen:
            continue
        seen.add(node.id)
        if isinstance(node.op, ops.Const):
            yield node.op.value
        elif _calls(node, "notation.lower"):
            written = dependencies([node.args[0]])
            yield from (each.op.value for each in written if isinstance(each.op, ops.Const))
            pending.extend(node.args[1:])
        elif not isinstance(node.type, NotationType):
            pending.extend(node.args)


def _written_with(root: Node) -> frozenset[str]:
    """The names of the symbols and of the functions that have no value that ``root``'s
    value is written with, read from the IR without computing it: those of its constants
    and of the functions it applies, but not the differentials a line element takes away,
    the variables a sum, a product, an integral, a limit or an evaluation bar binds, a name
    a definition made after the metric gives its value to, or a parameter a spacetime is
    not written with, as G of flat space."""
    found: dict[int, frozenset[str]] = {}
    for node in dependencies([root]):
        found[node.id] = _names_written(node, found)
    return found[root.id]


def _symbols_written(value: Node) -> list[str]:
    """The names of the symbols a value is written with, read from its IR as
    :func:`_written_with` reads them, without computing it, but not the functions it applies
    that nothing defines, as f of f(x)."""
    functions: set[str] = set()
    for node in dependencies([value]):
        op = node.op
        if isinstance(op, ops.Call) and op.name == "core.function":
            functions.add(str(dict(op.attributes)["name"]))
        elif isinstance(op, ops.Const) and isinstance(op.value, sympy.Basic):
            functions.update(each.func.__name__ for each in op.value.atoms(AppliedUndef))
    return sorted(_written_with(value) - functions)


def _names_written(node: Node, found: Mapping[int, frozenset[str]]) -> frozenset[str]:
    """The names ``node`` is written with (see :func:`_written_with`), from those of its
    arguments, in ``found``."""
    op = node.op
    if isinstance(op, ops.Const):
        return _symbol_names(op.value)
    if isinstance(node.type, NotationType):  # notation, which is no expression
        return frozenset()
    args = [found[arg.id] for arg in node.args]
    every = frozenset().union(*args)
    if isinstance(op, ops.LineElement):
        return every - {differential.name for differential in op.differentials}
    if not isinstance(op, ops.Call | ops.Map):
        return every
    name, rest = op.name, frozenset().union(*args[2:])
    if name == "notation.lower":  # components: their notation's scalars, and its tensors
        return every | frozenset().union(*map(_symbol_names, _constants(node.args[0])))
    if name == "core.function":
        return every | {str(dict(op.attributes)["name"])}
    if name in ("core.sum", "core.prod", "core.limit"):  # body, variable, then the rest
        return (args[0] - args[1]) | rest
    if name == "core.derivative_at":  # body, the variable, the order and the point
        return (args[0] - args[1]) | args[3]
    if name == "core.subs":  # expr, then pairs of what and what for
        return (args[0] - frozenset().union(*args[1::2])) | frozenset().union(*args[2::2])
    if name == "integrate":  # what a definite integral's limits bind, as core.limits gives
        bound = frozenset().union(
            *(found[arg.args[0].id] for arg in node.args[1:] if _calls(arg, "core.limits"))
        )
        return every - bound
    if name in ("core.where", "core.where_function"):  # the metric, the name, the value
        defined = node.args[1].op
        assert isinstance(defined, ops.Const)
        parameters: frozenset[str] = frozenset()
        if name == "core.where_function":
            defined_name, symbols = defined.value
            parameters = frozenset(symbol.name for symbol in symbols)
        else:
            defined_name = defined.value
        return (args[0] - {defined_name}) | (args[2] - parameters)
    if name in _SPACETIMES:
        absent = spacetimes.absent_from(_SPACETIMES[name])
        written = BUILTINS[name].parameters
        return frozenset().union(
            *(names for names, p in zip(args, written, strict=False) if p not in absent)
        )
    return every


def _calls(node: Node, name: str) -> bool:
    """Whether ``node`` applies the operation registered as ``name``."""
    return isinstance(node.op, ops.Call) and node.op.name == name


def _symbol_names(value: Any) -> frozenset[str]:
    """The names of the symbols and of the functions that have no value in a constant."""
    if isinstance(value, list | tuple):
        return frozenset().union(*(_symbol_names(item) for item in value))
    if not isinstance(value, sympy.Basic):
        return frozenset()
    names = {symbol.name for symbol in value.free_symbols if isinstance(symbol, Symbol)}
    names.update(function.func.__name__ for function in value.atoms(AppliedUndef))
    return frozenset(names)


def _with_arguments(target: ast.Expr) -> tuple[ast.Expr, ast.Expr] | None:
    """``\\psi_n(x)`` read as a product, \\psi_n times x: a name, maybe with scripts, and
    names in brackets written right after it, as the arguments of a function are; the name
    and the bracketed names. ``None`` for anything else."""
    if not (isinstance(target, ast.Binary) and target.op == "*"):
        return None
    func, group = target.left, target.right
    named = isinstance(func, ast.Name) or (
        isinstance(func, ast.Scripted | ast.Accent) and isinstance(func.base, ast.Name)
    )
    items = group.items if isinstance(group, ast.Tuple) else (group,)
    if not named or not all(isinstance(item, ast.Name) for item in items):
        return None
    if not group.span.text.startswith("(") or group.span.start != func.span.end:
        return None
    return func, group


def _family_label(func: ast.Expr, value: ast.Expr) -> str | None:
    """The label of a family, as n is of ``\\psi_n(x) := \\sin(n x)``: the one letter
    written as the subscript of the name being defined that its value is written with too,
    as it is written; ``None`` where there is none."""
    if not isinstance(func, ast.Scripted):
        return None
    subscripts = [script for script in func.scripts if script.kind == "_"]
    if len(subscripts) != 1 or subscripts[0].indices is None:
        return None
    items = subscripts[0].indices.items
    if len(items) != 1 or not _plain_label(items[0]):
        return None
    if str(items[0].label) not in _mentioned(value, latex=True):
        return None
    return subscripts[0].span.text.lstrip("_").strip("{}")


_OPERATOR_FORM = re.compile(
    r"\\(?:operatorname\*?|mathrm)\{[^{}]+\}(?:\s|\\[,;:!])*(?!\\left\b)(?:\\[A-Za-z]+|[A-Za-z])"
)


def _operator_form(written: str) -> str | None:
    """The left side of a function defined as books write one, its argument after its name
    without brackets, as ``\\operatorname{sinc} x`` of ``\\operatorname{sinc} x := ...``;
    ``None`` for one written with brackets, as ``sinc(x) := ...``."""
    match = _OPERATOR_FORM.match(written)
    return match.group(0) if match is not None else None


def _a_name(node: ast.Expr) -> bool:
    """Whether ``node`` is a name, maybe with subscripts, as ``\\Delta`` and ``x_e`` are."""
    if isinstance(node, ast.Scripted):
        return isinstance(node.base, ast.Name) and all(s.kind == "_" for s in node.scripts)
    return isinstance(node, ast.Name)


def _a_matrix(node: Node) -> bool:
    """Whether a value is a matrix, written as one, as ``\\begin{pmatrix} ...`` or a list of
    rows, or made of one by arithmetic or a function of a matrix, as ``-S_3``,
    ``\\frac{2\\pi}{3} M`` and ``A^T`` are; known without computing."""
    pending, seen = [node], set()
    while pending:
        current = pending.pop()
        if current.id in seen or isinstance(current.type, TensorType):  # A_{ab}: indices
            continue
        if current.type == SCALAR:  # a number, and what it is made of
            continue
        seen.add(current.id)
        if isinstance(current.op, ops.MakeArray):
            return len(current.op.shape) == 2
        if _rows(current) is not None or _makes_a_matrix(current):
            return True
        if isinstance(current.op, ops.Mul | ops.Scale | ops.Add | ops.Pow):
            pending.extend(current.args)
    return False


#: The operations whose value is a matrix, of the core.
_MATRICES = frozenset({"linalg.identity", "linalg.ones"})


def _makes_a_matrix(node: Node) -> bool:
    """Whether a value is made by a function whose value is a matrix: one of a matrix that
    is no number, as the inverse is and the spectrum, a list, is not, or the identity."""
    if not isinstance(node.op, ops.Call):
        return False
    name = node.op.name
    if name in _MATRICES:
        return True
    capability = catalogue().get(name)
    return isinstance(capability, MapFunction) and not capability.scalar and name != "spec"


def _rows(node: Node) -> list[Node] | None:
    """The rows of a matrix written as a list of them, ``[[1, 2], [3, 4]]``: lists of as
    many items each; ``None`` for anything else."""
    if not (isinstance(node.op, ops.Call) and node.op.name == "core.list" and node.args):
        return None
    rows = list(node.args)
    if not all(isinstance(row.op, ops.Call) and row.op.name == "core.list" for row in rows):
        return None
    return rows if len({len(row.args) for row in rows}) == 1 and rows[0].args else None


#: The functions of a matrix whose value has the shape of their first argument, and of its
#: transpose.
_SAME_SHAPE = frozenset({"inverse", "inv", "adjugate", "adj", "expm", "logm", "sqrtm",
                         "matrix_power", "linalg.identity"})  # fmt: skip
_TRANSPOSED = frozenset({"transpose", "ctranspose", "pinv"})


def _matrix_shape(node: Node) -> tuple[int | None, int | None] | None:
    """The rows and the columns of a matrix, each where it is known without computing it,
    as of a matrix written out, and of products and functions of one; ``None`` for a value
    that is no matrix."""
    if not _a_matrix(node):
        return None
    op, args = node.op, node.args
    if isinstance(op, ops.MakeArray):
        rows, columns = op.shape
        return rows, columns
    if (rows_written := _rows(node)) is not None:
        return len(rows_written), len(rows_written[0].args)
    if isinstance(op, ops.Mul):
        shapes = [shape for arg in args if (shape := _matrix_shape(arg)) is not None]
        return (shapes[0][0], shapes[-1][1]) if shapes else None
    if isinstance(op, ops.Add | ops.Pow | ops.Scale):
        shapes = [shape for arg in args if (shape := _matrix_shape(arg)) is not None]
        return shapes[0] if shapes else None
    if not isinstance(op, ops.Call) or not args:
        return None, None
    first = _matrix_shape(args[0])
    if op.name in _SAME_SHAPE:
        return first
    if op.name in _TRANSPOSED and first is not None:
        return first[1], first[0]
    if op.name == "linalg.ones" and first is not None:
        attributes = dict(op.attributes)
        length = first[0] if attributes["side"] == "rows" else first[1]
        return (1, length) if attributes["row"] else (length, 1)
    return None, None


def _writes_a_matrix(nodes: Iterable[ast.Expr], scope: Scope) -> bool:
    """Whether what is written is a matrix, known from what is written and the names the
    program defined: a matrix written out, a name whose value is one, as T_j is after
    T_j := D^{-1}(D - A), or a map written with its indices, as A^a{}_b; but no symbol, as
    the density in \\rho(1 + 3w), nor a form, as T_{\\mu\\nu} in \\kappa(T_{\\mu\\nu})."""
    for node in nodes:
        for current in _walk(node):
            if isinstance(current, ast.Matrix) and current.environment != "vmatrix":
                return True
            pattern = _tensor_pattern(current) if isinstance(current, ast.Scripted) else None
            if (
                pattern is not None
                and [upper for _, upper in pattern[1]] == [True, False]
                and scope.lookup(pattern[0]) is not None
            ):  # A^a{}_b, a map
                return True
            if isinstance(current, ast.List) and all(
                isinstance(row, ast.List) for row in current.items
            ):
                return bool(current.items)
            if isinstance(current, ast.Name):
                binding = scope.lookup(current.id)
                if isinstance(binding, VariableBinding) and _a_matrix(binding.node):
                    return True
    return False


#: The commands that set a number in bold, as books write the vector of ones.
_BOLD = ("\\mathbf", "\\boldsymbol", "\\bm", "\\pmb")

#: The name a program's own ``\mathbf{1}`` is bound to, as ``\mathbf{1} := ...`` defines it,
#: which no name written can be.
BOLD_ONE = "\\mathbf{1}"


def _bold_one(node: ast.Node) -> bool:
    """Whether a number is a bold one, ``\\mathbf{1}`` or ``{\\bf 1}``, which books write for
    the vector of ones, or the identity (design/notation.md, 12 and 23)."""
    if not (isinstance(node, ast.Number) and node.text == "1"):
        return False
    if node.span.text.lstrip().startswith(_BOLD):
        return True
    return node.span.source.text[: node.span.start].rstrip().endswith("{\\bf")


def _joined_by(node: ast.Binary) -> str:
    """What is written between the two sides of a product, as ``\\otimes``, which the lexer
    reads as a product's ``*``."""
    return Span(node.span.source, node.left.span.end, node.right.span.start).text.strip()


def _operators_written(node: ast.Binary) -> list[tuple[ast.Expr, ast.Expr, str]]:
    """The products of a product as written, each with what is written between its two
    sides, as ``\\otimes`` or ``\\times``, which the lexer reads as a product's ``*``."""
    found = []
    pending: list[ast.Expr] = [node]
    while pending:
        current = pending.pop()
        if isinstance(current, ast.Binary) and current.op == "*":
            left, right = current.left, current.right
            between = Span(current.span.source, left.span.end, right.span.start).text.strip()
            found.append((left, right, between))
            pending.extend((left, right))
    return found


def _elements(node: Node) -> int | None:
    """How many elements a list or a tuple has, as ``x := [2, 4, 4]`` makes one; ``None``
    for any other value."""
    if isinstance(node.op, ops.Call) and node.op.name in ("core.tuple", "core.list"):
        return len(node.args)
    if isinstance(node.op, ops.Const) and isinstance(node.op.value, list | tuple):
        return len(node.op.value)
    return None


def _a_member(subscript: str, bound: bool) -> bool:
    """Whether a subscript picks a member of a family or an element of a list, as 1 in c_1,
    -1 in E_{-1}, n+1 in a_{n+1}, or a variable a sum binds, as i in \\sum_i x_i, does;
    a letter that nothing binds, as x in v_x, may name a component of its own."""
    return bound or bool(re.fullmatch(r"-?\d+|.*[-+*/].*", subscript))


_RECURRENCE_LABEL = re.compile(r"(?P<label>[A-Za-z])(?:[-+]\d+)?")


def _subscripted(node: ast.Expr) -> tuple[str, str] | None:
    """A name with a subscript and the subscript as written: ``("T", "n-1")`` of
    ``T_{n-1}`` and of the plain name ``T_n-1``; ``None`` for anything else."""
    if isinstance(node, ast.Name):
        base, underscore, subscript = node.id.partition("_")
        return (base, subscript) if underscore and base and subscript else None
    if (
        isinstance(node, ast.Scripted)
        and isinstance(node.base, ast.Name)
        and len(node.scripts) == 1
        and node.scripts[0].kind == "_"
    ):
        return node.base.id, node.scripts[0].span.text.lstrip("_").strip("{}").replace(" ", "")
    return None


def _recurrence(target: ast.Expr, value: ast.Expr) -> tuple[str, ast.Expr] | None:
    """The label and the term that make ``target := value`` a recurrence, as n and
    T_{n-1} do of ``T_n = 2 T_{n-1} + 1``: the value writes the name being defined at
    another subscript in its label. ``None`` where it does not, as x_0 in
    ``x_n := x_0 + n d`` does not."""
    named = _subscripted(target)
    if named is None or (label := _RECURRENCE_LABEL.fullmatch(named[1])) is None:
        return None
    letter = label["label"]
    for node in _walk(value):
        term = _subscripted(node) if isinstance(node, ast.Name | ast.Scripted) else None
        if (
            term is not None
            and term[0] == named[0]
            and term[1] != named[1]
            and re.search(rf"(?<![A-Za-z]){letter}(?![A-Za-z])", term[1])
        ):
            assert isinstance(node, ast.Expr)
            return letter, node
    return None


def _a_pair(node: Node) -> bool:
    """Whether a value is a pair or a list written in brackets, of two items or more, as
    ``(a, b)``, ``[a, b]``, ``\\{a, b\\}`` and ``\\langle a, b \\rangle`` are read."""
    names = ("core.tuple", "core.list")
    return isinstance(node.op, ops.Call) and node.op.name in names and len(node.args) >= 2


def _may_be_a_pair(node: Node) -> bool:
    """Whether a scalar may stand for a pair: an unknown function's value, as the chart map
    ``\\phi_1(x^1, x^2, x^3)`` is, or a symbol nothing defines."""
    if isinstance(node.op, ops.Call) and node.op.name == "core.function":
        return True
    return isinstance(node.op, ops.Const) and isinstance(node.op.value, Symbol)


#: What two items in brackets often are in books, by the brackets they are written in, and
#: how to write it today, with {0} and {1} for the two items. Numbers in brackets are
#: none of them: [4, 9] is a list.
_BRACKETS = (
    (
        ("\\{", "\\left\\{", "\\lbrace", "\\left\\lbrace"),
        "a Poisson bracket",
        "for the Poisson bracket, write it out, as in \\frac{{\\partial {0}}}{{\\partial q}} "
        "\\frac{{\\partial {1}}}{{\\partial p}} - \\frac{{\\partial {0}}}{{\\partial p}} "
        "\\frac{{\\partial {1}}}{{\\partial q}}, summed over the canonical pairs (q, p)",
    ),
    (
        ("[", "\\left[", "\\big[", "\\Big[", "\\lbrack"),
        "a commutator",
        "for the commutator, write {0} {1} - {1} {0}",
    ),
)


def _bracket_meaning(written: str) -> tuple[str, str] | None:
    """What two items in the brackets ``written`` starts with are often in books, as a
    Poisson bracket is in ``\\{L_x, L_y\\}``, and the hint that says how to write it;
    ``None`` for other brackets, and for numbers in brackets."""
    items = _items_written(written)
    if items is None or all(_NUMBER.fullmatch(item) for item in items):
        return None
    for openers, meaning, hint in _BRACKETS:
        if written.startswith(openers):
            return meaning, hint
    return None


def _often(meant: tuple[str, str] | None) -> list[str]:
    """The note on what brackets often are in books, as a Poisson bracket, where they are."""
    if meant is None:
        return []
    if meant[0] == "a commutator":  # read so of operators and matrices
        return ["in these brackets, two items are often a commutator, which is read so of "
                "operators and matrices"]  # fmt: skip
    return [f"in these brackets, two items are often {meant[0]}, which is not supported yet"]


_OPENED = re.compile(
    r"^(?:\\left|\\big|\\Big)?(?:\\langle|⟨|\\\{|\\lbrace|\[|\\lbrack|\()(?P<inside>.*?)"
    r"(?:\\right|\\big|\\Big)?(?:\\rangle|⟩|\\\}|\\rbrace|\]|\\rbrack|\))$",
    re.DOTALL,
)
_NUMBER = re.compile(r"-?\d+(?:\.\d*)?")


def _items_written(written: str) -> list[str] | None:
    """The items of brackets as they are written: ``p`` and ``q`` of
    ``\\langle p, q \\rangle``; ``None`` where ``written`` is no bracketed list."""
    match = _OPENED.match(written.strip())
    if match is None:
        return None
    items, depth, start = [], 0, 0
    inside = match["inside"]
    for position, character in enumerate(inside):
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        elif character == "," and depth == 0 and not inside[:position].endswith("\\"):
            items.append(inside[start:position].strip())
            start = position + 1
    items.append(inside[start:].strip())
    return items


def _subscripted_symbol(target: ast.Scripted) -> bool:
    """``r_{s}``: a single subscript, which may name a symbol."""
    return (
        len(target.scripts) == 1
        and target.scripts[0].kind == "_"
        and target.scripts[0].indices is not None
        and len(target.scripts[0].indices.items) == 1
    )


def _subscripted_name(node: ast.Scripted) -> str:
    """The symbol a subscript names: ``r_{s}`` is ``r_s`` and ``x_{0}`` is ``x_0``."""
    assert isinstance(node.base, ast.Name)
    parts = []
    for script in node.scripts:
        if script.indices is not None:
            parts.append(
                "".join(
                    item.label if item.label is not None else to_text(item.value)
                    for item in script.indices.items
                )
            )
        else:
            parts.append(script.span.text.lstrip("_").strip("{}"))
    return f"{node.base.id}_{''.join(parts)}"


def to_text(node: ast.Expr | None) -> str:
    return (
        node.text
        if isinstance(node, ast.Number)
        else (node.id if isinstance(node, ast.Name) else "")
    )
