r"""The language reference, generated from the language's own tables.

:func:`markdown` gives, on one page, every statement, LaTeX command, environment,
function, constant, spacetime, statistic, verdict and kind of error RelativisticPy
knows, with what each means::

    >>> from relativisticpy.reference import markdown
    >>> page = markdown()
    >>> print(page.splitlines()[0])
    # The language reference
    >>> "Schwarzschild(M, G, c)" in page, r"\frac" in page
    (True, True)

``relativisticpy help reference`` prints it. The names come from the tables the
lexer, parser and compiler read; where a table says what a name is but not what it
means, a short table here says it. ``docs/reference.md`` is the same page, written by
``python -m relativisticpy.reference --write docs/reference.md``, and the tests keep it
current and complete.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from relativisticpy import display, integrations
from relativisticpy.checking import EMPTY, LISTED, STRICT, TIME_LIMIT, Verdict
from relativisticpy.compiler import builtins
from relativisticpy.compiler.compiler import (
    _BUILTIN_TENSORS,
    _LINEAR_ALGEBRA,
    _NOTATION_OPERATIONS,
    _STATISTICS,
)
from relativisticpy.compiler.session import (
    ASSUMPTION_STATEMENTS,
    DEFAULT_NAMES,
    RENAMING_STATEMENTS,
)
from relativisticpy.documents import DISPLAY_ENVIRONMENTS
from relativisticpy.errors import (
    CompileError,
    DataError,
    EvaluationError,
    FileError,
    IndexNotationError,
    InternalError,
    MetricError,
    NotComputedError,
    ParseError,
    UnknownNameError,
    UsageError,
)
from relativisticpy.integrations._objects import shown_type
from relativisticpy.language import vocabulary as vocab
from relativisticpy.tensors.equation import UndecidedError

__all__ = ["TITLE", "main", "markdown"]

#: The title of the page, by which ``relativisticpy help`` lists it.
TITLE = "The language reference"

# -- what the tables do not say -----------------------------------------------------------
#
# The functions and the spacetimes are described by their declarations, in the
# integrations (relativisticpy.integrations); what the tables of the parser and the
# compiler name but do not describe, a short table here says.

#: What the commands the parser and the lexer read themselves mean, a family at a time.
#: Every one of them is in exactly one family.
_STRUCTURE: tuple[tuple[tuple[str, ...], str], ...] = (
    (("frac", "dfrac", "tfrac", "cfrac", "over"),
     r"fractions, `\frac{a}{b}` and plain TeX's `{a \over b}`; `\frac{d}{dx}` and "
     r"`\frac{\partial f}{\partial x}` are derivatives"),
    (("sqrt",), r"roots, `\sqrt{x}` and `\sqrt[n]{x}`"),
    (("binom", "dbinom", "tbinom", "choose"),
     r"binomial coefficients, `\binom{n}{k}` and `{n \choose k}`"),
    (("sum", "prod"),
     r"a sum, `\sum_{n=1}^{N}`, and a product, in closed form where SymPy finds one"),
    (("int", "oint", "iint", "iiint"),
     r"integrals, single, double and triple, which end at their differentials: `\int_0^1 f \, dx`"),
    (("lim",), r"a limit, `\lim_{x \to 0}`, or from one side, `\lim_{x \to 0^+}`"),
    (("sup", "inf", "underset"),
     r"the supremum and the infimum over a variable, `\sup_{0 \le x < 1} x^n`, as "
     r"`\max_x` and `\arg\min_w` are optima; `\underset{x}{\min}` sets the variable beneath"),
    (("partial", "nabla"),
     r"the partial and the covariant derivative, `\partial_\mu` and `\nabla_\mu`; "
     r"`\partial_t^2` is the second derivative"),
    (("Box", "square"), r"the d'Alembertian, `\nabla^\mu \nabla_\mu`, as is `\nabla^2`"),
    (("pounds",), r"the Lie derivative, `\pounds_V T_{ab}`"),
    (("pdv", "dv"),
     r"the physics package's derivatives: `\pdv{f}{x}`, `\pdv[2]{f}{x}`, `\dv{f}{x}`"),
    (("lfloor", "rfloor", "lceil", "rceil"), r"the floor `\lfloor x \rfloor` and the ceiling"),
    (("left", "right"),
     r"`\left( ... \right)` groups, with any delimiters; between bars it is an absolute "
     r"value, and from `\left.` to a bar with `_{x=a}` an evaluation bar"),
    (("begin", "end"), r"an environment, `\begin{pmatrix} ... \end{pmatrix}`"),
    (("bgroup", "egroup"), "plain TeX's braces"),
    (("cr",), r"ends a row of plain TeX's `\matrix` and `\cases`"),
    (("mathbb",),
     r"blackboard bold: `\mathbb{R}^{n}` and `\mathbb{C}^{n}` are spaces, `\mathbb{E}_t` "
     r"an expectation and `\mathbb{1}_a` the covector of ones"),
)  # fmt: skip

#: What the operator commands mean, by the kinds of token they are read as, in order.
_OPERATORS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("STAR",), "multiplication, `*`; between tensors, their product, and `\\otimes` between "
     "matrices their Kronecker product"),
    (("SLASH",), "division, `/`"),
    (("LESS", "GREATER", "LESS_EQ", "GREATER_EQ", "NOT_EQ"),
     "comparisons, in this order: `<`, `>`, `<=` (three spellings), `>=` (three), `!=` (two)"),
    (("ARROW",), r"an arrow, as in `\lim_{x \to 0}`"),
    (("DEFINE",), "a definition, `:=`"),
    (("BAR",), "a bar: absolute values, norms and evaluation bars"),
    (("LPAREN", "RPAREN"), r"brackets: `\langle x \rangle` groups"),
    (("PLUS_MINUS", "MINUS_PLUS"), "both signs, so two values"),
    (("WEDGE",), "the wedge product"),
    (("COMPOSE",), "composition: read, not computed"),
)  # fmt: skip

#: The built-in tensors: the role whose name each takes, if any, the quantity it is, its
#: number of indices and what it means.
_TENSORS: tuple[tuple[str | None, str, int, str], ...] = (
    ("metric", "metric", 2, r"the metric, `g_{\mu\nu}`, and its inverse, `g^{\mu\nu}`"),
    ("connection", "christoffel", 3, r"the Christoffel symbols, `\Gamma^a{}_{bc}`"),
    ("riemann", "riemann", 4, r"the Riemann tensor, `R^a{}_{bcd}`"),
    ("ricci", "ricci", 2, r"the Ricci tensor, `R_{ab}`"),
    ("einstein", "einstein", 2, r"the Einstein tensor, `G_{ab}`"),
    ("weyl", "weyl", 4, r"the Weyl tensor, `C_{abcd}`"),
    (None, "kronecker", 2, r"the Kronecker delta, `\delta^a{}_b`"),
)

#: Names the compiler gives a built-in tensor besides those of its tables: most texts
#: write the Weyl tensor C_{abcd}.
_ALSO: Mapping[str, str] = {"weyl": "C"}

#: What each statistic means, by its kind in the compiler's table.
_STATISTIC_MEANINGS: Mapping[str, tuple[str, str]] = {
    "expectation": (r"\mathbb{E}_t[X]", "the expectation over t, a contraction with its measure"),
    "variance": (r"\operatorname{Var}_t(X)", "the variance over t of each component"),
    "covariance": (r"\operatorname{Cov}_t(X, Y)",
                   "the covariance over t, with the free indices of both"),
    "ledoit-wolf": (r"\operatorname{LedoitWolf}_t(R_{ta})",
                    "the Ledoit–Wolf estimate of a covariance, with scikit-learn"),
}  # fmt: skip

#: What each function of linear algebra does, given a tensor.
_LINEAR_ALGEBRA_MEANINGS: Mapping[str, str] = {
    "eig": r"`\lambda_k, v^a{}_k := \operatorname{eig}(A^a{}_b)`: eigenvalues and eigenvectors",
    "spec": r"`\operatorname{spec}(A^a{}_b)`, the eigenvalues; with a second argument n, the "
    "n largest, or smallest if n is negative",
    "svd": r"`U_{tk}, s_k, V_{ak} := \operatorname{svd}(X_{ta})`: the singular values and vectors",
    "det": "the determinant of a map",
    "tr": "the trace of a map",
    "trace": "the trace of a map",
    "inv": r"the inverse; `(\Sigma^{-1})^{ab}` is written as on paper, and solved for, not formed",
    "diag": r"`\operatorname{diag}(d)_{ab}`, the diagonal matrix of a vector",
    "exp": r"the exponential of a tensor named without indices: `\exp(-i H t)`",
    "sqrt": r"the square root of a matrix, `\sqrt{\Sigma}`",
    "log": "the logarithm of a matrix",
}

#: What each operation does to an equation between tensors without components.
_NOTATION_MEANINGS: Mapping[str, str] = {
    "solve": r"`solve(E, T_{\mu\nu})`, for a tensor, with its indices up or down",
    "trace": "contracts its two free indices with the metric",
    "simplify": "its canonical form, with the identities of General Relativity",
    "subs": r"`subs(E, rule)`: a definition, a contraction, or a value such as `\Lambda = 0`",
    "LHS": "its left side",
    "RHS": "its right side",
}

#: How the declarations of symbols are worded, by the assumption each makes.
_ASSUMPTIONS: Mapping[str, str] = {
    "constant": "constants",
    "real": "real",
    "integer": "integers",
    "complex": "complex",
    "positive": "positive",
}

# -- writing Markdown ---------------------------------------------------------------------


def _code(text: str) -> str:
    """``text`` as inline code."""
    return f"`{text}`"


def _command(name: str) -> str:
    return _code(f"\\{name}")


def _order(name: str) -> tuple[str, str]:
    return (name.lower(), name)


def _commands(names: Iterable[str]) -> str:
    """Commands, as one piece of code: `\\alpha \\beta`."""
    return _code(" ".join(f"\\{name}" for name in sorted(names, key=_order)))


def _names(names: Iterable[str]) -> str:
    return ", ".join(_code(name) for name in sorted(names, key=_order))


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    """A Markdown table. A bar in a cell would be read differently by GitHub and by the
    docs site, so the cells hold none."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        if any("|" in cell for cell in row):
            raise ValueError(f"a cell of the reference holds a bar: {row}")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _arity(capability: integrations.Capability) -> str:
    low, high = capability.arity
    if high is None:
        return f"{low} or more"
    if low == high:
        return str(low)
    return f"{low} to {high}"


def _first_sentence(text: str | None) -> str:
    """The first sentence of a docstring, on one line, without its full stop."""
    words = " ".join((text or "").split())
    sentence = words.split(". ")[0].rstrip(".").replace("``", "`")
    return sentence[:1].lower() + sentence[1:]


_MATHEMATICS = re.compile(r"[\\_^{}()|/=]|^[b-zB-Z]$|^[-+]$|^\.\.\.$")


def _prose(text: str) -> str:
    """A message of the language's, with its mathematics as code: ``write d x, or
    \\mathrm{d} x`` is ``write `d x`, or `\\mathrm{d} x```."""
    words, run = [], []
    for word in text.split(" "):
        bare = word.rstrip(",;:")
        if _MATHEMATICS.search(bare):
            run.append(bare)
            if bare == word:
                continue
            words.append(_code(" ".join(run)) + word[len(bare) :])
        else:
            if run:
                words.append(_code(" ".join(run)))
            words.append(word)
        run = []
    if run:
        words.append(_code(" ".join(run)))
    return " ".join(words)


# -- the sections -------------------------------------------------------------------------


def _header() -> list[str]:
    return [
        f"# {TITLE}",
        "",
        "> **For:** anyone, person or agent, who needs to know what the language accepts. "
        "**You will learn:** every statement, command, environment, function, constant, "
        "spacetime, statistic, verdict and error, on one page.",
        "",
        "This page is generated from the tables the language is read with. "
        "`relativisticpy help reference` prints it offline, and "
        "`relativisticpy.reference.markdown()` gives it in Python. The guides explain each "
        "part, with examples: [The language](language.md), [LaTeX](latex.md), "
        "[Tensors and geometry](tensors.md) and [Beyond spacetime](spaces.md).",
    ]


def _statements() -> list[str]:
    defines = [name for name, kind in vocab.OPERATOR_COMMANDS.items() if kind.name == "DEFINE"]
    unicode = [c for c, kind in vocab.UNICODE_OPERATORS.items() if kind.name == "DEFINE"]
    lines = [
        "## Statements",
        "",
        "A program is a sequence of statements, one to a line; a semicolon, `\\\\` and "
        f"{_commands(vocab.LINE_BREAK_COMMANDS)} also end one. `#` and `%` start a comment, "
        "so `%` is not a remainder. A statement goes on to the next line while a bracket "
        "is open, after a line that ends with an operator, and before a line that starts "
        "with one such as `+` or `=`; a line that starts with `-` goes on only if it is "
        "indented further.",
        "",
        "`:=` defines, as do "
        + " ".join(_command(name) for name in sorted(defines, key=_order))
        + " and "
        + " ".join(_code(c) for c in unicode)
        + "; between two sides neither of which can be defined, `\\equiv` states that they "
        "are equal. `=` defines only what has no meaning yet; otherwise it states an "
        "equation, or a claim, which is checked.",
        "",
    ]
    declarations = " ".join(_code(name) for name in ASSUMPTION_STATEMENTS)
    words = [
        " and ".join(_ASSUMPTIONS[key] for key in assumption)
        for assumption in ASSUMPTION_STATEMENTS.values()
    ]
    assumptions = ", ".join(words[:-1]) + " or " + words[-1]
    rows = [
        ("`x := 2`", "defines x, or gives it a new value"),
        ("`x = 2`, where x is new",
         "defines x; so do `f(x) = x^2` and a new tensor `T_{ab} = ...`"),
        ("`x = 3`, where x has a value",
         "a claim, checked: does x equal 3? `x := 3` gives x a new value"),
        ("`ds^2 = ...`", "a line element: the metric, and the coordinates if none are declared"),
        ("`R = 0`, `g_{tt} = -1`",
         "with a metric in force, a claim about a built-in or a component"),
        ("`t = 0`, where t is a coordinate", "a condition: read, not computed"),
        ("`x^2 + y^2 = 1`", "an equation, to `solve`; `a = b = c` is a claim for each link"),
        ("`f(x) := x^2`", "a function; `f : (x) -> { ... }` is one with a block of statements"),
        (r"`\mathcal{H} := \mathbb{C}^{2}`",
         r"a space, complex, of 2 dimensions; `\mathbb{R}^{n}` is real"),
        (r"`i, j \in \mathcal{H}`", "index labels that range over a space"),
        (r"`x_n := n h, \quad n = 0, \ldots, N-1`",
         "a grid: n ranges over its N values, and x is made for each"),
        (r"`a > 0`, `0 < x < 1`, `a, b > 0`, `x \in \mathbb{R}`, `n \in \mathbb{N}`",
         "an assumption about names nothing defines, from then on"),
        (r"`Coordinates := [t, r, \theta, \phi]`", "the coordinates, in order"),
        (f"{declarations} `:= [a, b]`", f"declare a and b {assumptions}, before they are used"),
    ]  # fmt: skip
    lines += _table(("Statement", "Meaning"), rows)
    renamings = ", ".join(
        f"{_code(statement)} ({_code(DEFAULT_NAMES[role])})"
        for statement, role in RENAMING_STATEMENTS.items()
    )
    lines += [
        "",
        "A renaming statement gives a built-in another name: `MetricSymbol := G` makes the "
        f"metric `G`. They are, with the names they change: {renamings}.",
    ]
    return lines


def _names_and_numbers() -> list[str]:
    lines = [
        "## Names and numbers",
        "",
        "- A name that nothing defines is a symbol. `theta`, `\\theta` and `θ` are one "
        "symbol, as is every spelling of a Greek letter.",
        "- In LaTeX, a run of letters is a product of one symbol for each letter: "
        "`\\frac{2GM}{r}` is 2·G·M/r. Greek names, defined names and builtins stay whole, "
        "and `\\mathrm{GM}` or `\\text{GM}` is one name. In plain text, `GM` is one name.",
        "- A subscript on a symbol makes another symbol: `r_s` and `r_{s}` are both `r_s`. "
        "On a tensor, it is an index.",
        "- A name followed by a bracket, with no space, is a call: `f(x)`. A function nothing "
        "defines is unknown, as `A(r)` in a metric. A function written as a command takes "
        "what follows it: `\\sin x`, `\\sin^2\\theta`.",
        "- Arithmetic is exact: `1/2` is a half, and a decimal stays a decimal. `^` and `**` "
        "are powers, a space between factors multiplies, and `299\\,792\\,458` is one number.",
        "- `[1, 2]` is a list, which is a vector in arithmetic; a list of rows, or a matrix "
        "environment, is a matrix.",
        "- A bar with values, as in `x^2 |_{x=2}`, puts the values in.",
        "",
    ]
    values = {
        "pi": "π",
        "e": "Euler's number",
        "i": "the imaginary unit",
        "oo": "infinity",
        "infty": "infinity, as `\\infty`",
        "True": "true",
        "False": "false",
    }
    rows = [
        (_code(name), values.get(name, str(value))) for name, value in builtins.CONSTANTS.items()
    ]
    lines += _table(("Constant", "Value"), rows)
    return lines


def _index_notation() -> list[str]:
    lines = [
        "## Index notation",
        "",
        "- `^{...}` holds upper indices and `_{...}` lower ones. In braces each letter is an "
        "index and a Greek name is one: `_{ab}` is a, b and `_{\\mu\\nu}` (or `_{mu nu}`) is "
        "μ, ν. A script without braces is one letter, command or number, as in TeX: `R^a_b` "
        "is `R^{a}_{b}`. `{}` staggers indices: `R^{a}{}_{bcd}`.",
        "- An index once up and once down in a term is summed; the others are free, and are "
        "the same in every term and on both sides. These rules are checked before anything "
        "is computed.",
        "- With a metric in force, it raises and lowers indices: write a tensor with its "
        "indices where you want them.",
        "- A component is chosen by number, `g_{01}`, by a coordinate's name, "
        "`\\Gamma^{r}_{\\theta\\theta}`, or with `=` or `:`, as in `A^{\\mu=0}` and "
        "`w^{a:MSFT}`.",
        "- `\\partial_\\mu`, or `d_{a}`, and `\\nabla_\\mu`, or `D_{a}`, are the partial and "
        "the covariant derivative of the factor after them; `T^{\\mu\\nu}{}_{,\\nu}` and "
        "`T^{\\mu\\nu}{}_{;\\nu}` mean the same.",
        "- `T_{(ab)}` symmetrises and `F_{[ab]}` antisymmetrises, also across tensors.",
        "- An equation between tensors with components stands for one equation for each "
        "component. One between tensors without components, as `T_{\\mu\\nu}` before it is "
        "defined, is kept as an equation of notation.",
        "",
        "### Built-in tensors",
        "",
        "With a metric in force, these names mean the tensors of its geometry, by name and "
        "number of indices. The conventions are those of Misner, Thorne and Wheeler, and of "
        "Carroll.",
        "",
    ]
    rows = []
    for role, quantity, count, meaning in _TENSORS:
        names = [DEFAULT_NAMES[role]] if role is not None else []
        names += [name for name, kinds in _BUILTIN_TENSORS.items() if kinds.get(count) == quantity]
        names += [_ALSO[quantity]] if quantity in _ALSO else []
        written = (f"\\{name}" if name in vocab.GREEK else name for name in dict.fromkeys(names))
        rows.append((" ".join(_code(name) for name in written), str(count), meaning))
    rows += [
        (_code(DEFAULT_NAMES["metric"]), "0", "the determinant of the metric"),
        (f"{_code(DEFAULT_NAMES['ricci'])} `R`", "0", "the Ricci scalar"),
        ("`Kretschmann`", "0", "the Kretschmann scalar, `R_{abcd} R^{abcd}`"),
        (r"`geodesic_equations(\tau)`", "", r"the geodesic equations, in the parameter `\tau`"),
    ]
    lines += _table(("Name", "Indices", "Meaning"), rows)
    lines += [
        "",
        "`R` and `g` stay symbols where the metric is written with them, as the radius R of "
        "a sphere is; `Ric` is always the Ricci scalar. In a line element, `d\\Omega^2` is "
        "`d\\theta^2 + \\sin^2\\theta\\, d\\phi^2`.",
        "",
        "### Equations between tensors without components",
        "",
    ]
    lines += _table(
        ("Operation", "Meaning"),
        ((_code(name), _NOTATION_MEANINGS[name]) for name in sorted(_NOTATION_OPERATIONS)),
    )
    return lines


def _spacetimes() -> list[str]:
    lines = [
        "## Spacetimes",
        "",
        "Each is a function that gives a metric, as in `g_{\\mu\\nu} := Schwarzschild(M)`. "
        "Parameters are names or numbers, and units are geometric, G = c = 1, unless `G` "
        "and `c` follow the spacetime's own parameters.",
        "",
    ]
    rows = []
    for capability in _declared(spacetime=True):
        factory = builtins._SPACETIMES[capability.name]
        coordinates = ", ".join(display.unicode(c) for c in factory().coordinates)
        rows.append((_code(capability.text()), coordinates, capability.summary))
    lines += _table(("Spacetime", "Coordinates", "Meaning"), rows)
    return lines


def _declared(*, spacetime: bool) -> list[integrations.Capability]:
    """The capabilities the integrations declare: the spacetimes, or the rest."""
    return [
        capability
        for capability in integrations.capabilities()
        if (capability.name in builtins._SPACETIMES) == spacetime
    ]


def _title(module: str) -> str:
    """The title of an integration: the first line of its module's docstring, or its
    name."""
    documented = getattr(sys.modules.get(module), "__doc__", None)
    first = (documented or module.rsplit(".", 1)[-1]).strip().splitlines()[0]
    return first.rstrip(".")


def _structure() -> list[tuple[tuple[str, ...], str]]:
    """The families of the commands the parser and the lexer read themselves, checked
    against the vocabulary: every one of them in one family, and nothing else."""
    named = [name for names, _ in _STRUCTURE for name in names]
    expected = vocab.PARSER_COMMANDS | vocab.STRUCTURE_COMMANDS
    if sorted(named) != sorted(expected):
        raise AssertionError(
            "the commands the parser reads are out of date: not described "
            f"{sorted(expected - set(named))}, not read {sorted(set(named) - expected)}"
        )
    return list(_STRUCTURE)


def _functions() -> list[str]:
    lines = [
        "## Functions",
        "",
        "A function is called by its name, `simplify(x)`, or, if LaTeX has it, as a command, "
        "`\\sin x`, or as books write it, `\\Gamma(x)`: a definition of the name in the "
        "program comes first. Most apply to each component of a tensor: `simplify(R_{abcd})` "
        "simplifies every component. Each is a capability of an integration, a module of "
        "`relativisticpy.integrations` ([Integrations](integrations.md)), from whose "
        "declaration this list is written.",
    ]
    modules: dict[str, list[integrations.Capability]] = {}
    rewrites: list[integrations.Rewrite] = []
    for capability in _declared(spacetime=False):
        if isinstance(capability, integrations.Rewrite):
            rewrites.append(capability)
            continue
        if isinstance(capability, integrations.Optimizer):
            continue  # consulted in turn, never called: see _optimizers
        modules.setdefault(capability.module, []).append(capability)
    for module, declared in modules.items():
        rows = [
            (_called(capability), _arity(capability), capability.summary) for capability in declared
        ]
        lines += ["", f"### {_title(module)}", ""]
        lines += _table(("Function", "Arguments", "Meaning"), rows)
    lines += [
        "",
        "### Rewrites",
        "",
        "What a program writes, computed faster by another capability, to the same numbers:",
        "",
    ]
    rows = [
        (" ".join(_code(form) for form in rewrite.latex), _code(rewrite.computes), rewrite.summary)
        for rewrite in rewrites
    ]
    lines += _table(("Written", "Computed by", "Meaning"), rows)
    lines += _optimizers()
    reserved = [
        f"{_code(reservation.written[0])}, {reservation.what}"
        for reservation in integrations.reservations()
    ]
    lines += [
        "",
        "### Book names not supported yet",
        "",
        "A name books write for a function no integration provides is refused, saying what it "
        "is, never read as an unknown function or a symbol: " + "; ".join(reserved) + ".",
    ]
    return lines


def _optimizers() -> list[str]:
    """The optimizers, in the order an optimum consults them."""
    from relativisticpy.integrations._optimizers import optimizers

    lines = [
        "",
        "### Optimizers",
        "",
        "An optimum, `\\min_{x} f(x)`, `\\arg\\max_{x, y} u \\quad \\text{subject to} \\quad "
        "g = 0` or `\\sup_{0 \\le x < 1} f(x)`, is found by the optimizers, in this order: "
        "those that compute exactly, then, where it has no parameters, those that compute "
        "numerically, which take first one written with decimals, as data is. A note says "
        "which found it, how, and whether it is global. Over an index of data, `\\max_t x_t` "
        "is the greatest value along it, and `\\arg\\max_t x_t` its label.",
        "",
    ]
    rows = [
        (_code(each.name), "numerically" if each.numerical else "exactly", each.summary)
        for each in optimizers()
    ]
    return lines + _table(("Optimizer", "Computes", "Meaning"), rows)


def _called(capability: integrations.Capability) -> str:
    """How a capability is written: called by its name, then its other names, then the
    forms books write it in."""
    written = (capability.text(), *capability.aliases, *capability.latex)
    return " ".join(_code(name) for name in written)


def _data() -> list[str]:
    lines = [
        "## Data, statistics and linear algebra",
        "",
        'Data is bound from Python, `wb["R_{ta}"] = array`, and computed with NumPy: '
        '`pip install "relativisticpy[numeric]"`, or `[data]` for pandas and scikit-learn. '
        "An index ranges over a space: of coordinates, assets, days or states. `1_a` is the "
        "covector of ones, `A_{ta} \\odot B_{ta}` the elementwise product and `\\|v\\|` the "
        "length of a vector.",
        "",
        "### Statistics",
        "",
        "A statistic binds its index, as a sum does: inside it, every factor shares t.",
        "",
    ]
    spellings: dict[str, list[str]] = {}
    for written, kind in _STATISTICS.items():
        spellings.setdefault(kind, []).append(written)
    rows = [
        (_code(example), f"{meaning}; named {_names(spellings[kind])}")
        for kind, (example, meaning) in _STATISTIC_MEANINGS.items()
    ]
    lines += _table(("Statistic", "Meaning"), rows)
    lines += [
        "",
        "### Linear algebra",
        "",
        "Given a tensor, these functions are linear algebra. They are written "
        "`\\operatorname{eig}`, or as a command where LaTeX has one, `\\det`.",
        "",
    ]
    names: dict[str, list[str]] = {}
    for name in sorted(_LINEAR_ALGEBRA):
        names.setdefault(_LINEAR_ALGEBRA_MEANINGS[name], []).append(name)
    lines += _table(
        ("Function", "Meaning"),
        ((" ".join(_code(name) for name in same), meaning) for meaning, same in names.items()),
    )
    lines += [
        "",
        "### Python objects",
        "",
        "`\\texttt{prices}` is the object called prices in the Python scope a workbook is "
        "given, `Workbook(scope=globals())`, or of the objects passed to `show`. Attributes, "
        'keys and positions follow, `\\texttt{prices["NVDA"]}`, and are read, never run as '
        "Python. A document checked, and a program sent to the MCP server, have no scope. "
        "Defined with indices, `P_{ta} := \\texttt{prices}`, data is bound as "
        '`wb["P_{ta}"] = prices` binds it, and meets other data by label; written with them, '
        "`\\texttt{signal}_t`, it has them. What an object becomes, by the object integrations:",
        "",
    ]
    lines += _table(
        ("Object", "Becomes"),
        (
            (" ".join(_code(shown_type(taken)) for taken in conversion.takes), conversion.summary)
            for conversion in integrations.conversions()
        ),
    )
    return lines


def _latex() -> list[str]:
    operators: dict[str, list[str]] = {}
    for name, kind in vocab.OPERATOR_COMMANDS.items():
        operators.setdefault(kind.name, []).append(name)
    renamed = ", ".join(
        f"{_command(name)} calls {_code(builtin)}"
        for name, builtin in sorted(
            vocab.FUNCTION_COMMANDS.items(), key=lambda item: _order(item[0])
        )
        if name != builtin
    )
    variants = vocab.GREEK_VARIANTS
    if any(letter != name.removeprefix("var") for name, letter in variants.items()):
        raise AssertionError("a variant letterform is not the letter after var")
    symbols = set(vocab.SYMBOL_COMMANDS) - vocab.GREEK - set(variants)
    sizes = _code(" ".join(f"\\{name}" for name in sorted(vocab.SIZE_COMMANDS)))
    rows: list[tuple[str, str]] = [
        (_commands(vocab.GREEK), "Greek letters, each one symbol"),
        (_commands(variants), "variant letterforms: each the letter after `var`"),
        (_commands(symbols), r"symbols; `\odot` is the Sun, as in `M_\odot`, and between "
         r"tensors the elementwise product; `\oplus` between matrices is their direct sum, and "
         r"`A^\top`, `A^\intercal` and `A^\dagger` of a matrix its transpose and its conjugate "
         "transpose"),
        (_commands(vocab.FUNCTION_COMMANDS), "functions, each calling the builtin of its name, "
         f"except that {renamed}; `\\sin^{{-1}}` is the inverse, and only `\\log` takes a "
         "subscript, its base"),
        (_commands(vocab.ACCENT_COMMANDS), r"accents: `\dot{a}` and `\ddot{a}` are time "
         "derivatives of a function of one variable; the others name a new symbol"),
    ]  # fmt: skip
    if {kind for kinds, _ in _OPERATORS for kind in kinds} != set(operators):
        raise AssertionError("the operator commands are out of date: say what each kind means")
    rows += [
        (
            _code(" ".join(f"\\{name}" for kind in kinds for name in sorted(operators[kind]))),
            meaning,
        )
        for kinds, meaning in _OPERATORS
    ]
    rows += [
        (_commands(vocab.RELATION_COMMANDS), r"relations, read so that the equations around "
         r"them run, but not computed; `\in` also declares index labels"),
        (_commands(vocab.CONNECTIVE_COMMANDS), "between equations, each side a statement"),
        (_commands(vocab.ELLIPSIS_COMMANDS), "terms not written: read, not computed"),
        (_commands(vocab.TEXT_COMMANDS - {"texttt"}), r"text: one name, as `\text{otherwise}`; "
         r"`\operatorname{name}` names a function"),
        (_command("texttt"), r"a Python object, read by name from the workbook's scope: "
         r"`\texttt{prices}` (see Python objects)"),
        (_commands(vocab.STYLE_COMMANDS), r"typefaces: what they hold is read as it is, so "
         r"`\mathrm{d}x` is `d x`"),
        (_commands(vocab.FONT_DECLARATIONS), r"plain TeX's typefaces, for the rest of the "
         r"group: `{\rm d}x`"),
        (_commands(vocab.LAYOUT_COMMANDS | vocab.LAYOUT_COMMANDS_WITH_ARGUMENT),
         r"layout, ignored; `\quad` and `\qquad` also separate equations"),
        (sizes, "delimiter sizes, ignored"),
        (_commands(vocab.LINE_BREAK_COMMANDS), r"end a statement, as `\\` does"),
        (_commands(vocab.PLAIN_ENVIRONMENTS), r"plain TeX's matrices and cases: "
         r"`\matrix{a & b \cr c & d}`"),
    ]  # fmt: skip
    rows += [(_commands(names), meaning) for names, meaning in _structure()]
    lines = [
        "## LaTeX",
        "",
        "Typesetting is ignored: spacing, sizes, `\\label`, `\\tag`, `$`, `$$`, `\\[ \\]` "
        "and the `&` of alignments. Every statement inside a display is LaTeX, however it "
        "is written. An unknown command is an error that suggests the closest one.",
        "",
        "### Commands",
        "",
    ]
    lines += _table(("Command", "Meaning"), rows)
    return lines


def _environments() -> list[str]:
    kinds = {
        "matrices": "a matrix; `vmatrix` is its determinant",
        "cases": r"a piecewise function, with `\text{if}` and `\text{otherwise}`",
        "equations": r"statements, separated by `\\`",
    }
    lines = ["### Environments", ""]
    lines += _table(
        ("Environment", "Meaning"),
        ((_names(names), kinds[kind]) for kind, names in vocab.ENVIRONMENTS.items()),
    )
    equations = vocab.ENVIRONMENTS["equations"]
    lines += [
        "",
        f"- In {_names(vocab.COLUMN_PAIR_ENVIRONMENTS)}, each pair of columns is an equation; "
        f"{_names(vocab.SINGLE_EQUATION_ENVIRONMENTS)} holds one equation, broken over rows.",
        f"- {_names(vocab.ENVIRONMENTS_WITH_ARGUMENT)} take an argument first, such as the "
        "columns of `array`.",
        "- `relativisticpy check` reads the equation environments of a document as displays, "
        f"with {_names(DISPLAY_ENVIRONMENTS - equations)}, `$$ ... $$` and `\\[ ... \\]`; "
        f"it reads {_names(equations - DISPLAY_ENVIRONMENTS)} inside them.",
    ]
    return lines


def _unsupported() -> list[str]:
    lines = [
        "### Not supported",
        "",
        "These commands are LaTeX's, or a package's, but not the language's. Each is an "
        "error that says what to write instead:",
        "",
    ]
    groups: dict[tuple[str, str], list[str]] = {}
    for name, (what, instead) in sorted(
        vocab.UNSUPPORTED_COMMANDS.items(), key=lambda item: _order(item[0])
    ):
        groups.setdefault((_without(what, name), instead), []).append(name)
    for (what, instead), names in groups.items():
        lines.append(f"- {_commands(names)}: {_prose(what)}; {_prose(instead)}.")
    return lines


def _without(message: str, name: str) -> str:
    """A message about one command, said of it without its name: ``the remainder, \\mod,
    is not supported yet`` is ``the remainder is not supported yet``."""
    command = re.escape(f"\\{name}")
    message = re.sub(rf"^{command},? ", "", message)
    message = re.sub(rf", {command},", "", message)
    message = re.sub(rf" {command}(?= )", "", message)
    return message[:1].lower() + message[1:]


def _unicode() -> list[str]:
    commands = ", ".join(f"{c} {_command(name)}" for c, name in vocab.UNICODE_COMMANDS.items())
    operators = " ".join(vocab.UNICODE_OPERATORS)
    relations = " ".join(vocab.UNICODE_RELATIONS)
    return [
        "### Unicode",
        "",
        "Unicode reads as the LaTeX it stands for. A Greek letter, `θ`, is its command, and so "
        f"are {commands}. The operators {operators} and the relations {relations} are "
        "LaTeX's. Superscripts and subscripts are scripts, so `r²` is `r^{2}` and `Aᵀ` is "
        "`A^{T}`, † after a matrix is `^\\dagger`, ⊕ is `\\oplus`, and the double-struck "
        "letters, ℝ ℂ 𝔼 𝟙, are `\\mathbb`'s. A combining accent, as in ȧ, is an error that "
        "names it: write the command, `\\dot{a}`.",
    ]


def _verdicts() -> list[str]:
    lines = [
        "## Verdicts of a check",
        "",
        "`relativisticpy check paper.tex`, or `wb.check(document)` in Python, runs the "
        "display equations of a document in order and gives each statement a verdict.",
        "",
    ]

    def fails(verdict: Verdict) -> str:
        if verdict in STRICT:
            return "yes"
        return "over a time limit" if verdict is Verdict.NOT_COMPUTED else "no"

    rows = [(verdict.mark, verdict.value, verdict.meaning, fails(verdict)) for verdict in Verdict]
    lines += _table(("Mark", "Verdict", "Meaning", "Fails `--strict`"), rows)
    listed = " ".join(_code(verdict) for verdict in Verdict if verdict in LISTED)
    lines += [
        "",
        "`--strict` also fails on a document in which no equation was found. A report lists "
        f"the statements that are {listed}, and counts the rest; `--all` lists every one. "
        "In Python, `report.failed(...)` takes the verdicts to fail on, and "
        f"{_code(TIME_LIMIT)} and {_code(EMPTY)} besides.",
    ]
    return lines


def _errors() -> list[str]:
    kinds: tuple[type[Exception], ...] = (
        ParseError,
        CompileError,
        IndexNotationError,
        NotComputedError,
        EvaluationError,
        MetricError,
        DataError,
        UnknownNameError,
        UsageError,
        FileError,
        UndecidedError,
        InternalError,
    )
    lines = [
        "## Errors",
        "",
        "Every error a user can cause is a `RelativisticPyError`. It says what is wrong, "
        "where, as `--> file:line:column` with the text underlined, why, in a `note:`, and "
        "how to fix it, in a `hint:`. `error.to_dict()` gives the same as data.",
        "",
    ]
    lines += _table(
        ("Error", "When"), ((_code(kind.__name__), _first_sentence(kind.__doc__)) for kind in kinds)
    )
    return lines


_SECTIONS: tuple[Callable[[], list[str]], ...] = (
    _header,
    _statements,
    _names_and_numbers,
    _index_notation,
    _spacetimes,
    _functions,
    _data,
    _latex,
    _environments,
    _unsupported,
    _unicode,
    _verdicts,
    _errors,
)


def markdown() -> str:
    """The language reference, as Markdown."""
    return "\n\n".join("\n".join(section()) for section in _SECTIONS) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m relativisticpy.reference``: print the reference; with ``--write PATH``,
    write it to a file, as UTF-8."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) == 2 and arguments[0] == "--write":
        Path(arguments[1]).write_text(markdown(), encoding="utf-8")
        return 0
    if arguments:
        print("usage: python -m relativisticpy.reference [--write PATH]", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(markdown().encode("utf-8"))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
