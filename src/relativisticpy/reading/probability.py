r"""How random variables are written (design/notation.md, 14).

- ``X \sim \mathcal{N}(\mu, \sigma^2)`` declares a random variable by its distribution, whose
  forms the integrations declare (:mod:`relativisticpy.integrations._distributions`): the form
  says what each argument is, as ``\sigma^2`` says that the second is the square of the
  standard deviation. ``X, Y \sim ...`` declares several, ``X_1, \ldots, X_n
  \overset{\text{iid}}{\sim} ...`` a sample, and ``\text{independent}`` after the
  distribution says so; ``X \mid N \sim \operatorname{Bin}(N, p)`` and ``X \mid Y = 0 \sim
  ...`` give a law given another variable.
- ``\Pr(A)``, ``\mathbb{P}(A)`` and ``P(X = 2)`` are probabilities, ``\mathbb{E}[X]`` an
  expectation, ``\operatorname{Var}(X)`` and ``\operatorname{Cov}(X, Y)`` the variance and the
  covariance; inside them ``\mid`` is the conditional bar.
- ``f_X(x)``, ``F_X(x)``, ``F_X^{-1}(q)``, ``M_X(t)`` and ``\varphi_X(t)`` are the density,
  the distribution function, the quantile, and the moment-generating and characteristic
  functions of a random variable X.

Each is taken apart here, from the syntax; :mod:`relativisticpy.lowering.probability` reads
it in the program.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from relativisticpy.errors import RelativisticPyError
from relativisticpy.integrations import _forms as forms
from relativisticpy.integrations._discovery import on_change
from relativisticpy.integrations._distributions import Distribution, distributions
from relativisticpy.language import syntax as ast
from relativisticpy.language.source import Span
from relativisticpy.reading import sequences

#: The decision of what ``\sim`` declares.
DISTRIBUTED = "distributed as"
#: The decision of whether ``P(...)``, ``E[...]`` or ``f_X(x)`` is of random variables.
RANDOM = "of random variables"

#: The operators of random variables, by the names they are written with: those always
#: theirs, whatever they act on.
OPERATORS = {
    "Pr": "probability",
    "ℙ": "probability",
    "𝔼": "expectation",
    "Var": "variance",
    "var": "variance",
    "Cov": "covariance",
    "cov": "covariance",
}
#: Letters that are a probability and an expectation only where they act on random
#: variables, as P(X = 2) and E[X] do: P is a pressure, and E an energy, elsewhere.
LETTERS = {"P": "probability", "E": "expectation"}


#: The functions of a random variable, by the letter written with it as a subscript.
FUNCTIONS = {
    "f": "density",
    "p": "density",
    "F": "distribution function",
    "M": "moment-generating function",
    "psi": "moment-generating function",
    "phi": "characteristic function",
}

#: What says, after a distribution, that the variables declared are independent.
_INDEPENDENT = re.compile(r"i\.?\s*i\.?\s*d\.?|independent(?:ly)?", re.IGNORECASE)


@dataclass(frozen=True)
class Law:
    """A distribution as a program writes it: the ``capability`` whose form it has, and the
    value written for each parameter, in order, with the power of the parameter it is: 2 for
    the variance a normal distribution is written with, of which the standard deviation is
    the root; and how its ``form`` writes each, as σ² of ``\\mathcal{N}(\\mu, \\sigma^2)``."""

    capability: Distribution
    arguments: tuple[tuple[ast.Expr, int], ...]
    form: str
    span: Span
    slots: tuple[str, ...] = ()


@dataclass(frozen=True)
class Declaration:
    """``X \\sim D``: the ``variables`` declared, the ``law`` they have, what they are
    ``given``, if anything, as N of ``X \\mid N``, and whether they are said to be
    ``independent``, by iid or a word."""

    variables: tuple[ast.Expr, ...]
    law: Law
    given: ast.Expr | None
    independent: bool
    span: Span


# -- distributions ------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Pattern:
    owner: Distribution
    written: str
    shape: forms.Shape


_PATTERNS: list[_Pattern] | None = None


def _forget() -> None:
    global _PATTERNS
    _PATTERNS = None


on_change(_forget)


def _patterns() -> list[_Pattern]:
    global _PATTERNS
    if _PATTERNS is None:
        found = []
        for capability in distributions():
            for written in (*capability.latex, *capability.unicode):
                expr = parsed(written)
                shape = _shape(expr) if expr is not None else None
                if shape is not None:
                    found.append(_Pattern(capability, written, shape))
        _PATTERNS = found
    return _PATTERNS


def parsed(written: str) -> ast.Expr | None:
    """A form, parsed by the language; ``None`` where it is no expression."""
    from relativisticpy.language import parse

    try:
        statements = parse(written).statements
    except RelativisticPyError:
        return None
    if len(statements) != 1 or not isinstance(statements[0], ast.Evaluate):
        return None
    return statements[0].expr


def form(written: str) -> list[str] | None:
    """The parameters a form of a distribution is written with, in order; ``None`` where it
    does not parse as one, a call, as ``\\mathcal{N}(\\mu, \\sigma^2)``, or a name with
    scripts, as ``\\chi^2_k``."""
    expr = parsed(written)
    shape = _shape(expr) if expr is not None else None
    if shape is None:
        return None
    found = []
    for slot in (*shape.scripts, *shape.arguments):
        variable = _variable(slot)
        if variable is not None:
            found.append(variable[0])
        elif not isinstance(slot, ast.Number):
            return None
    return found


def law(expr: ast.Expr) -> Law | None:
    """The distribution ``expr`` writes, with its parameters, as a form declares it, or the
    plain text that calls it by name; ``None`` where it writes none."""
    shape = _shape(expr)
    if shape is None:
        return None
    for pattern in _patterns():
        arguments = _bound(pattern.shape, shape)
        if arguments is not None and len(arguments) == len(pattern.owner.parameters):
            slots = tuple(_written(slot) for slot in _slots(pattern.shape))
            return Law(pattern.owner, arguments, pattern.written, expr.span, slots)
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
        for capability in distributions():
            if expr.func.id in capability.names and len(expr.args) == len(capability.parameters):
                arguments = tuple((argument, 1) for argument in expr.args)
                slots = capability.parameters
                return Law(capability, arguments, capability.text(), expr.span, slots)
    return None


def _slots(form: forms.Shape) -> list[ast.Node]:
    """The slots of a form that are its parameters, in order."""
    return [slot for slot in (*form.scripts, *form.arguments) if _variable(slot) is not None]


def _written(slot: ast.Node) -> str:
    """A parameter as a form writes it, as a note says it: σ²."""
    found = _variable(slot)
    assert found is not None
    name, power = found
    return name if power == 1 else f"{name}^{power}"


def _shape(expr: ast.Expr) -> forms.Shape | None:
    """A distribution as written, taken apart: its name, how it is set, its scripts and its
    arguments, as a call's are."""
    found = forms.form_shape(expr)
    if found is not None and found.kind in ("call", "subscripted"):
        return found
    if isinstance(expr, ast.Scripted) and isinstance(expr.base, ast.Name):  # \chi^2_k
        below: list[ast.Node] = []
        above: list[ast.Node] = []
        for script in expr.scripts:
            if script.body is None:
                return None
            (below if script.kind == "_" else above).append(script.body)
        base = expr.base
        style = forms._style(base)
        return forms.Shape(
            "subscripted", base.id, style, base.span.text, below=tuple(below), above=tuple(above)
        )
    return None


def _bound(form: forms.Shape, shape: forms.Shape) -> tuple[tuple[ast.Expr, int], ...] | None:
    """What each parameter of a form is in a distribution as written, with the power it is
    written to; ``None`` where it does not have the form's shape."""
    if form.kind != shape.kind or form.head != shape.head:
        return None
    if not forms._set_alike(form, shape):
        return None
    counts = (len(form.before), len(form.below), len(form.above), len(form.arguments))
    if counts != (len(shape.before), len(shape.below), len(shape.above), len(shape.arguments)):
        return None
    bound: list[tuple[ast.Expr, int]] = []
    for slot, written in zip(
        (*form.scripts, *form.arguments), (*shape.scripts, *shape.arguments), strict=True
    ):
        variable = _variable(slot)
        if variable is not None:
            if not isinstance(written, ast.Expr):
                return None
            bound.append((written, variable[1]))
        elif isinstance(slot, ast.Number):
            if not (isinstance(written, ast.Number) and written.text == slot.text):
                return None
        else:
            return None
    return tuple(bound)


def _variable(slot: ast.Node) -> tuple[str, int] | None:
    """A slot of a form that is a parameter, ``\\mu``, or a power of one, ``\\sigma^2``: its
    name and the power; ``None`` for anything else, as the 2 of ``\\chi^2_k``."""
    if isinstance(slot, ast.Name):
        return slot.id, 1
    if (
        isinstance(slot, ast.Scripted)
        and isinstance(slot.base, ast.Name)
        and len(slot.scripts) == 1
        and slot.scripts[0].kind == "^"
        and isinstance(slot.scripts[0].body, ast.Number)
        and slot.scripts[0].body.text.isdigit()
    ):
        return slot.base.id, int(slot.scripts[0].body.text)
    return None


# -- declarations -------------------------------------------------------------------------------


def declaration(expr: ast.Expr) -> Declaration | None:
    """``X \\sim D``, ``X, Y \\sim D``, ``X_1, \\ldots, X_n \\overset{\\text{iid}}{\\sim} D``,
    ``X \\mid N \\sim D``: what it declares; ``None`` for a relation whose right side is no
    distribution, as ``n! \\sim \\sqrt{2\\pi n}(n/e)^n``."""
    if not (isinstance(expr, ast.Relation) and expr.op == "~"):
        return None
    left, right, written = expr.lhs, expr.rhs, expr.written
    given: ast.Expr | None = None
    if isinstance(left, ast.Relation) and left.op == "∣":  # X \mid N \sim D
        given = left.rhs
        left = left.lhs
    right, said = _independence(right)
    found = law(right)
    if found is None:
        return None
    variables = left.items if isinstance(left, ast.Tuple) else (left,)
    independent = said or _INDEPENDENT.search(written) is not None
    return Declaration(tuple(variables), found, given, independent, expr.span)


def _independence(right: ast.Expr) -> tuple[ast.Expr, bool]:
    """A distribution followed by a word that says the variables are independent, as
    ``\\mathcal{N}(0, 1) \\text{ independent}``: the distribution, and whether it says so."""
    if (
        isinstance(right, ast.Binary)
        and right.op == "*"
        and isinstance(right.right, ast.Name)
        and right.right.verbatim
        and _INDEPENDENT.fullmatch(right.right.id) is not None
    ):
        return right.left, True
    return right, False


def conditioned(expr: ast.Expr) -> tuple[ast.Expr, ast.Expr | None]:
    """What an operator acts on, and what it is conditioned on, after the conditional bar:
    ``X > 1 \\mid Y = 0`` is ``X > 1`` given ``Y = 0``; ``None`` where there is no bar."""
    if isinstance(expr, ast.Relation) and expr.op == "∣":
        return expr.lhs, expr.rhs
    return expr, None


# -- the names random variables are written with --------------------------------------------------


def operator(name: ast.Name) -> str | None:
    """The operator of random variables a name writes: one always theirs, as ``\\Pr``, or a
    letter that is one only of random variables, P or E; ``None`` for any other name."""
    return OPERATORS.get(name.id) or LETTERS.get(name.id)


def lettered(name: ast.Name, *, square: bool = False) -> bool:
    """Whether a name is a letter that is an operator only where it acts on random
    variables, as P and E are; an upright E before square brackets,
    ``\\operatorname{E}[X]``, is always the expectation, as books set it, while
    ``\\operatorname{E}(m)`` is an elliptic integral's."""
    upright = name.id == "E" and name.verbatim and square
    return name.id in LETTERS and not upright


def squared(call: ast.Call) -> bool:
    """Whether a call is written with square brackets, as ``\\operatorname{E}[X]`` is."""
    after = call.span.source.text[call.func.span.end : call.span.end].lstrip()
    return after.startswith(("[", "\\left["))


def capital(node: ast.Expr) -> str | None:
    """The name of a random variable written as probability writes one, a capital Latin
    letter, maybe with a subscript that is written as an integer is, as X, Y, X_1 and
    X_{k+1} are; ``None`` for any other name."""
    if isinstance(node, ast.Name):
        base, _, label = node.id.partition("_")
        if len(base) == 1 and base.isascii() and base.isupper() and _label(label):
            return node.id
        return None
    if not (
        isinstance(node, ast.Scripted)
        and isinstance(node.base, ast.Name)
        and len(node.scripts) == 1
        and node.scripts[0].kind == "_"
    ):
        return None
    base, body = node.base.id, node.scripts[0].body
    if not (len(base) == 1 and base.isascii() and base.isupper()) or body is None:
        return None
    if isinstance(body, ast.Name | ast.Number):
        label = body.id if isinstance(body, ast.Name) else body.text
        return f"{base}_{label}" if _label(label) else None
    if not sequences.integral(body, lambda label: None):
        return None
    return f"{base}_{body.span.text.replace(' ', '')}"  # X_{k+1}, X_{n-1}


def _label(label: str) -> bool:
    return not label or label.isdigit() or (len(label) == 1 and label.isalpha())


def function_of(head: ast.Expr) -> tuple[str, ast.Expr] | None:
    """``f_X``, ``F_X``, ``F_X^{-1}``, ``M_X`` or ``\\varphi_X``: which function of a random
    variable it is, and the variable's name as written; ``None`` for anything else."""
    inverse = False
    if (
        isinstance(head, ast.Scripted)
        and len(head.scripts) >= 1
        and head.scripts[-1].kind == "^"
        and _minus_one(head.scripts[-1].body)
    ):
        inverse = True
        head = (
            ast.Scripted(head.base, head.scripts[:-1], span=head.span)
            if len(head.scripts) > 1
            else head.base
        )
    letter: str | None = None
    variable: ast.Expr | None = None
    if isinstance(head, ast.Name) and "_" in head.id:
        letter, _, rest = head.id.partition("_")
        variable = ast.Name(rest, span=head.span) if rest else None
    elif (
        isinstance(head, ast.Scripted)
        and isinstance(head.base, ast.Name)
        and len(head.scripts) == 1
        and head.scripts[0].kind == "_"
        and head.scripts[0].body is not None
    ):
        letter, variable = head.base.id, head.scripts[0].body
    if letter is None or variable is None or letter not in FUNCTIONS:
        return None
    kind = FUNCTIONS[letter]
    if inverse:
        if kind != "distribution function":
            return None
        kind = "quantile"
    return kind, variable


def _minus_one(node: ast.Expr | None) -> bool:
    return (
        isinstance(node, ast.Unary)
        and node.op == "-"
        and isinstance(node.operand, ast.Number)
        and node.operand.text == "1"
    )


def event_key(event: ast.Expr, given: ast.Expr | None) -> str:
    """The name the probability of an event the program gives is bound to: ``P(A|B_1)`` of
    ``\\Pr(A \\mid B_1)``, however it is written."""
    shown = _text(event)
    if given is not None:
        shown += f"|{_text(given)}"
    return f"P({shown})"


def _text(node: ast.Expr) -> str:
    named = capital(node)
    if named is not None:
        return named
    if isinstance(node, ast.Name):
        return node.id
    return re.sub(r"\s+|\\[,;!:]|[{}]", "", node.span.text)
