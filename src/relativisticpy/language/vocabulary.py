"""The LaTeX and Unicode vocabulary RelativisticPy understands.

This module is pure data. The lexer and parser consult it, so it is the one
place to teach the language a new command or symbol.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from relativisticpy.language.tokens import TokenKind

# --------------------------------------------------------------------------
# Letters and symbols
# --------------------------------------------------------------------------

#: Greek letter names, as LaTeX spells them. Each is a single symbol, whether
#: typed as a command (``\theta``), as plain text (``theta``) or in Unicode (``θ``).
GREEK: frozenset[str] = frozenset(
    {
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
        "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho",
        "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
        "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon",
        "Phi", "Psi", "Omega",
    }
)  # fmt: skip

#: Variant letterforms that denote the same symbol as the plain letter, including
#: amsmath's slanted capitals such as ``\varOmega``.
GREEK_VARIANTS: Mapping[str, str] = MappingProxyType(
    {
        "varepsilon": "epsilon",
        "vartheta": "theta",
        "varphi": "phi",
        "varrho": "rho",
        "varsigma": "sigma",
        "varkappa": "kappa",
        **{f"var{capital}": capital for capital in GREEK if capital[0].isupper()},
    }
)

#: Infinity as plain text writes it, as SymPy does: one symbol, as ``\infty`` and ∞ are.
INFINITY = "oo"

#: Commands that stand for a symbol, mapped to the name of that symbol.
SYMBOL_COMMANDS: Mapping[str, str] = MappingProxyType(
    {
        **{letter: letter for letter in GREEK},
        **GREEK_VARIANTS,
        "varpi": "varpi",
        "ell": "ell",
        "hbar": "hbar",
        "infty": "infty",
        "odot": "odot",  # the Sun, as in M_\odot
        "star": "star",  # as in N_\star
        "oplus": "oplus",  # the direct sum, between two matrices
        "top": "top",  # the transpose, as in A^\top
        "intercal": "intercal",
        "dagger": "dagger",  # the conjugate transpose, as in A^\dagger
        "imath": "i",  # the dotless i, set under a hat: \hat{\imath}, a unit vector
        "jmath": "j",
    }
)

#: Commands that apply an elementary function, mapped to the builtin they call.
FUNCTION_COMMANDS: Mapping[str, str] = MappingProxyType(
    {
        "sin": "sin", "cos": "cos", "tan": "tan", "cot": "cot", "sec": "sec", "csc": "csc",
        "arcsin": "asin", "arccos": "acos", "arctan": "atan", "arccot": "acot",
        "sinh": "sinh", "cosh": "cosh", "tanh": "tanh", "coth": "coth",
        "sech": "sech", "csch": "csch",
        "exp": "exp", "ln": "ln", "log": "log",
        "det": "det", "min": "min", "max": "max", "gcd": "gcd", "arg": "arg",
        "Re": "re", "Im": "im", "sgn": "sign", "ker": "null", "dim": "dim", "Pr": "Pr",
    }
)  # fmt: skip

#: The names of LaTeX's functions, as plain text and Unicode write them, in letters: where
#: one applies to what follows it, it is the command, so that ``sin²x`` is ``\sin^2 x`` and
#: ``log_2 8`` is ``\log_2 8``, and it is never a product of letters. Not ``Re``, ``Im`` and
#: ``Pr``, which LaTeX sets as symbols, and which are also names, as the Reynolds and
#: Prandtl numbers are in ``Re^{1/2} Pr^{1/3}``, or letters, as in ``Re^{i\theta}``.
FUNCTION_WORDS: frozenset[str] = frozenset(FUNCTION_COMMANDS) - {"Re", "Im", "Pr"}

#: The names of LaTeX's operators with limits, as plain text writes them: with their
#: scripts, ``int_0^1 x dx``, ``sum_{n=1}^oo`` and ``lim_{x->0}`` are the commands.
LIMITS_WORDS: frozenset[str] = frozenset({"int", "iint", "iiint", "oint", "sum", "prod", "lim"})

#: Of those, the ones that apply to what follows without scripts too, as ``int x dx`` does:
#: ``sum``, ``prod`` and ``lim`` without them are names, and ``lim(f, x, 0)`` a call.
UNSCRIPTED_WORDS: frozenset[str] = frozenset({"int", "iint", "iiint", "oint"})

#: Words that write a sign of LaTeX's, mapped to its command: AsciiMath's ``del``, which is
#: ∂, so that ``(del f)/(del x)`` is ``\frac{\partial f}{\partial x}``.
SIGN_WORDS: Mapping[str, str] = MappingProxyType({"del": "partial"})

#: The words a subscript belongs to, as its base, its variable or its limits: ``log_2``,
#: ``min_x``, ``argmax_x``, ``sup_x``, ``int_0``, ``del_x``. The bounds and the places of
#: optima are words only with a script.
SCRIPTED_WORDS: frozenset[str] = LIMITS_WORDS | {
    "log", "min", "max", "sup", "inf", "argmin", "argmax", "del",
}  # fmt: skip

#: The inverse of each function, used for the ``\sin^{-1} x`` notation.
INVERSE_FUNCTIONS: Mapping[str, str] = MappingProxyType(
    {
        "sin": "asin", "cos": "acos", "tan": "atan", "cot": "acot",
        "sec": "asec", "csc": "acsc", "sinh": "asinh", "cosh": "acosh",
        "tanh": "atanh", "coth": "acoth", "sech": "asech", "csch": "acsch",
    }
)  # fmt: skip

#: Accents, mapped to the decoration they add. ``\\bar{x}`` is the symbol ``xbar``,
#: which SymPy also prints as ``\\bar{x}``; ``\\dot{a}`` can denote a time derivative.
ACCENT_COMMANDS: Mapping[str, str] = MappingProxyType(
    {
        "dot": "dot", "ddot": "ddot", "hat": "hat", "widehat": "hat", "bar": "bar",
        "overline": "bar", "tilde": "tilde", "widetilde": "tilde", "vec": "vec",
        "check": "check", "breve": "breve",
    }
)  # fmt: skip

# --------------------------------------------------------------------------
# Commands the lexer resolves on its own
# --------------------------------------------------------------------------

#: Commands that are operators, mapped to the token they lex as.
OPERATOR_COMMANDS: Mapping[str, TokenKind] = MappingProxyType(
    {
        "cdot": TokenKind.STAR,
        "times": TokenKind.STAR,
        "ast": TokenKind.STAR,
        "div": TokenKind.SLASH,
        "lt": TokenKind.LESS,
        "gt": TokenKind.GREATER,
        "le": TokenKind.LESS_EQ,
        "leq": TokenKind.LESS_EQ,
        "leqslant": TokenKind.LESS_EQ,
        "ge": TokenKind.GREATER_EQ,
        "geq": TokenKind.GREATER_EQ,
        "geqslant": TokenKind.GREATER_EQ,
        "ne": TokenKind.NOT_EQ,
        "neq": TokenKind.NOT_EQ,
        "to": TokenKind.ARROW,
        "rightarrow": TokenKind.ARROW,
        "longrightarrow": TokenKind.ARROW,
        "mapsto": TokenKind.ARROW,
        "equiv": TokenKind.DEFINE,
        "coloneqq": TokenKind.DEFINE,
        "coloneq": TokenKind.DEFINE,
        "triangleq": TokenKind.DEFINE,
        "defeq": TokenKind.DEFINE,
        "vert": TokenKind.BAR,
        "lvert": TokenKind.BAR,
        "rvert": TokenKind.BAR,
        "Vert": TokenKind.BAR,
        "lVert": TokenKind.BAR,
        "rVert": TokenKind.BAR,
        "mid": TokenKind.BAR,
        "langle": TokenKind.LANGLE,  # Dirac's notation, or an expectation value
        "rangle": TokenKind.RANGLE,
        "otimes": TokenKind.STAR,  # in index notation a tensor product is a product
        "pm": TokenKind.PLUS_MINUS,
        "mp": TokenKind.MINUS_PLUS,
        "wedge": TokenKind.WEDGE,
        "circ": TokenKind.COMPOSE,
    }
)

#: Relations other than equality and order, mapped to their symbol. RelativisticPy
#: reads them, so that the equations around them run, but does not compute them.
RELATION_COMMANDS: Mapping[str, str] = MappingProxyType(
    {
        "sim": "~", "simeq": "≃", "approx": "≈", "cong": "≅", "doteq": "≐",
        "asymp": "≍", "propto": "∝", "ll": "≪", "gg": "≫",
        "in": "∈", "notin": "∉", "subset": "⊂", "subseteq": "⊆", "supset": "⊃",
        "supseteq": "⊇", "leftrightarrow": "↔", "longleftrightarrow": "↔",
        "succeq": "⪰", "preceq": "⪯", "succ": "≻", "prec": "≺",
    }
)  # fmt: skip

#: Connectives between whole equations, such as ``a = b \Rightarrow c = d``. Each
#: side is a statement of its own.
CONNECTIVE_COMMANDS: frozenset[str] = frozenset(
    {"Rightarrow", "Longrightarrow", "implies", "Leftrightarrow", "Longleftrightarrow", "iff"}
)

#: Ellipses, which stand for terms that are not written.
ELLIPSIS_COMMANDS: frozenset[str] = frozenset(
    {"cdots", "ldots", "dots", "dotsc", "dotsb", "dotsm", "dotsi", "dotso", "vdots", "ddots"}
)

#: Commands whose argument is prose, not mathematics. The argument becomes a
#: single name (``\text{otherwise}``) or, if it contains spaces, a string.
TEXT_COMMANDS: frozenset[str] = frozenset(
    {
        "text", "textrm", "textit", "textbf", "textsf", "texttt", "textup", "textnormal",
        "mbox", "hbox", "operatorname",
    }
)  # fmt: skip

#: Commands that only change the typeface of their argument. The argument is
#: lexed as ordinary mathematics: ``\mathrm{d}x`` reads exactly like ``d x``.
STYLE_COMMANDS: frozenset[str] = frozenset(
    {
        "mathrm", "mathit", "mathbf", "mathsf", "mathtt", "mathcal", "mathbb",
        "mathfrak", "mathscr", "mathnormal", "boldsymbol", "bm", "pmb",
        "mathop", "mathord", "mathrel", "mathbin", "mathpunct", "mathinner", "lefteqn",
        "mathopen", "mathclose", "substack",
    }
)  # fmt: skip

#: Plain TeX's font switches, which change the typeface of the rest of the group
#: they are in: ``{\rm d}x`` reads like ``\mathrm{d}x``.
FONT_DECLARATIONS: frozenset[str] = frozenset(
    {
        "rm", "bf", "it", "sl", "sf", "tt", "sc", "em", "cal", "mit", "boldmath",
        "rmfamily", "bfseries", "itshape", "upshape", "mdseries", "normalfont",
    }
)  # fmt: skip

#: Typesetting commands without arguments; they are ignored.
LAYOUT_COMMANDS: frozenset[str] = frozenset(
    {
        "displaystyle", "textstyle", "scriptstyle", "scriptscriptstyle",
        "limits", "nolimits", "nonumber", "notag", "allowbreak", "displaybreak",
        "quad", "qquad", "enspace", "thinspace", "medspace", "thickspace",
        "negthinspace", "negmedspace", "negthickspace", "hfill", "hline",
        "noindent", "centering", "aftergroup",
    }
)  # fmt: skip

#: The layout commands that insert horizontal space.
SPACING_COMMANDS: frozenset[str] = frozenset(
    {
        "quad", "qquad", "enspace", "thinspace", "medspace", "thickspace",
        "negthinspace", "negmedspace", "negthickspace",
    }
)  # fmt: skip

#: The spacing commands wide enough to separate equations: ``a = b \\qquad c = d``.
WIDE_SPACING_COMMANDS: frozenset[str] = frozenset({"quad", "qquad", "hfill", "hfil"})

#: Typesetting commands with one braced argument (``\label{eq:1}``); ignored.
LAYOUT_COMMANDS_WITH_ARGUMENT: frozenset[str] = frozenset(
    {"label", "tag", "vspace", "hspace", "phantom", "hphantom", "vphantom"}
)

#: Commands that end a line, and therefore a statement.
LINE_BREAK_COMMANDS: frozenset[str] = frozenset({"newline", "linebreak"})

#: Delimiter sizing commands (``\big(``, ``\Bigl[`` ...).
SIZE_COMMANDS: frozenset[str] = frozenset(
    size + suffix for size in ("big", "Big", "bigg", "Bigg") for suffix in ("", "l", "r", "m")
)

# --------------------------------------------------------------------------
# Environments
# --------------------------------------------------------------------------

#: Environments that only lay out equations. Their contents are statements,
#: ``\\`` separates statements and ``&`` alignment markers are ignored. Among them
#: are breqn's ``dmath`` and ``dgroup`` and IEEEtrantools' ``IEEEeqnarray``.
ALIGNMENT_ENVIRONMENTS: frozenset[str] = frozenset(
    {
        "align", "align*", "aligned", "alignat", "alignat*", "alignedat", "flalign",
        "flalign*", "equation", "equation*", "gather", "gather*", "gathered", "split",
        "multline", "multline*", "eqnarray", "eqnarray*", "displaymath", "math",
        "dmath", "dmath*", "dgroup", "dgroup*", "IEEEeqnarray", "IEEEeqnarray*",
    }
)  # fmt: skip

#: Alignments whose columns come in pairs, each pair an equation, so that the second
#: ``&`` of a pair starts the next equation: ``a &= 1 & b &= 2``.
COLUMN_PAIR_ENVIRONMENTS: frozenset[str] = frozenset(
    {"align", "align*", "aligned", "alignat", "alignat*", "alignedat", "flalign", "flalign*"}
)

#: Environments that hold one equation, however many rows it is broken over.
SINGLE_EQUATION_ENVIRONMENTS: frozenset[str] = frozenset({"multline", "multline*"})

#: Matrix environments. ``vmatrix`` denotes the determinant of its entries.
MATRIX_ENVIRONMENTS: frozenset[str] = frozenset(
    {"matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix", "smallmatrix", "array"}
)

#: Piecewise definitions.
CASES_ENVIRONMENTS: frozenset[str] = frozenset({"cases", "dcases"})

#: Plain TeX's matrices and cases, written ``\matrix{a & b \cr c & d}``. They
#: read as the environment of the same name, with ``\cr`` ending each row.
PLAIN_ENVIRONMENTS: frozenset[str] = frozenset({"matrix", "pmatrix", "cases"})

#: Environments followed by a mandatory argument that is not mathematics,
#: such as the column specification of ``\begin{array}{cc}``.
ENVIRONMENTS_WITH_ARGUMENT: frozenset[str] = frozenset(
    {"array", "alignat", "alignat*", "alignedat", "IEEEeqnarray", "IEEEeqnarray*"}
)

#: Every environment a program may use, by kind, as an error lists them.
ENVIRONMENTS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "matrices": MATRIX_ENVIRONMENTS,
        "cases": CASES_ENVIRONMENTS,
        "equations": ALIGNMENT_ENVIRONMENTS - {"displaymath"},
    }
)

# --------------------------------------------------------------------------
# Unicode
# --------------------------------------------------------------------------

#: Unicode letters, mapped to the name of the symbol they denote. Each is one symbol,
#: as the command it stands for is: ``ℏ`` is ``\hbar``, not h b a r, and ``⊙`` is
#: ``\odot``, the Sun in ``M_⊙`` or the elementwise product in ``A ⊙ B``.
UNICODE_LETTERS: Mapping[str, str] = MappingProxyType(
    {
        "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
        "ϵ": "epsilon", "ζ": "zeta", "η": "eta", "θ": "theta", "ϑ": "theta",
        "ι": "iota", "κ": "kappa", "ϰ": "kappa", "λ": "lambda", "μ": "mu",
        "µ": "mu", "ν": "nu", "ξ": "xi", "ο": "omicron", "π": "pi", "ϖ": "varpi",
        "ρ": "rho", "ϱ": "rho", "σ": "sigma", "ς": "sigma", "τ": "tau",
        "υ": "upsilon", "φ": "phi", "ϕ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
        "Γ": "Gamma", "Δ": "Delta", "∆": "Delta", "Θ": "Theta", "Λ": "Lambda",
        "Ξ": "Xi", "Π": "Pi", "Σ": "Sigma", "Υ": "Upsilon", "Φ": "Phi", "Ψ": "Psi",
        "Ω": "Omega",
        "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
        "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Χ": "X",
        "∞": "infty", "ℏ": "hbar", "ℓ": "ell", "⊙": "odot", "⊕": "oplus",
    }
)  # fmt: skip

#: Unicode mathematical operators that have a LaTeX command equivalent.
UNICODE_COMMANDS: Mapping[str, str] = MappingProxyType(
    {
        "∂": "partial", "∇": "nabla", "√": "sqrt", "∑": "sum", "∏": "prod",
        "∫": "int", "∬": "iint", "∭": "iiint", "∮": "oint", "⊛": "circledast",
    }
)  # fmt: skip


#: Unicode operators that lex directly as an operator token.
def double_struck(letter: str) -> str:
    """The double-struck form of a letter or digit, as ``\\mathbb`` sets it: ℝ for R."""
    special = {"C": "ℂ", "H": "ℍ", "N": "ℕ", "P": "ℙ", "Q": "ℚ", "R": "ℝ", "Z": "ℤ"}
    if letter in special:
        return special[letter]
    if letter.isdigit():
        return chr(0x1D7D8 + int(letter))
    if letter.isupper():
        return chr(0x1D538 + ord(letter) - ord("A"))
    return chr(0x1D552 + ord(letter) - ord("a"))


#: Blackboard bold, ``\mathbb{R}`` or ℝ: a name of its own, so that ℝ is not R, the
#: Ricci scalar, and 𝔼, an expectation, is not E. Each maps to itself.
DOUBLE_STRUCK: Mapping[str, str] = MappingProxyType(
    {
        double_struck(c): double_struck(c)
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    }
)

UNICODE_OPERATORS: Mapping[str, TokenKind] = MappingProxyType(
    {
        "·": TokenKind.STAR,
        "⋅": TokenKind.STAR,
        "∗": TokenKind.STAR,
        "×": TokenKind.STAR,
        "÷": TokenKind.SLASH,
        "⁄": TokenKind.SLASH,  # the fraction slash, as in 3⁄4
        "−": TokenKind.MINUS,
        "≤": TokenKind.LESS_EQ,
        "≥": TokenKind.GREATER_EQ,
        "≠": TokenKind.NOT_EQ,
        "→": TokenKind.ARROW,
        "↦": TokenKind.ARROW,
        "≡": TokenKind.DEFINE,
        "≔": TokenKind.DEFINE,
        "′": TokenKind.PRIME,
        "±": TokenKind.PLUS_MINUS,
        "∓": TokenKind.MINUS_PLUS,
        "∧": TokenKind.WEDGE,
        "∘": TokenKind.COMPOSE,
        "⊗": TokenKind.STAR,
        "‖": TokenKind.BAR,
        "⟨": TokenKind.LANGLE,
        "⟩": TokenKind.RANGLE,
        "〈": TokenKind.LANGLE,
        "〉": TokenKind.RANGLE,
        "…": TokenKind.ELLIPSIS,
        "⋯": TokenKind.ELLIPSIS,
    }
)

#: Unicode relations, mapped to the symbol :data:`RELATION_COMMANDS` uses.
UNICODE_RELATIONS: Mapping[str, str] = MappingProxyType(
    {
        "∼": "~", "≃": "≃", "≈": "≈", "≅": "≅", "∝": "∝", "≪": "≪", "≫": "≫",
        "∈": "∈", "∉": "∉", "⊂": "⊂", "⊆": "⊆", "↔": "↔",
    }
)  # fmt: skip

#: Unicode superscript characters (``r²``, ``eˣ``) and their plain equivalents.
UNICODE_SUPERSCRIPTS: Mapping[str, str] = MappingProxyType(
    dict(
        zip(
            "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿᵀᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐᵒᵖʳˢᵗᵘᵛʷˣʸᶻ",
            "0123456789+-nTabcdefghijklmoprstuvwxyz",
            strict=True,
        )
    )
)

#: Unicode marks set as a superscript after what they mark, as ``A†`` is ``A^\dagger``.
UNICODE_MARKS: Mapping[str, str] = MappingProxyType({"†": "dagger"})

#: Unicode subscript characters (``x₀``, ``rₛ``) and their plain equivalents.
UNICODE_SUBSCRIPTS: Mapping[str, str] = MappingProxyType(
    dict(zip("₀₁₂₃₄₅₆₇₈₉₊₋ₐₑₒₓₕₖₗₘₙₚₛₜᵢⱼᵣᵤᵥ", "0123456789+-aeoxhklmnpstijruv", strict=True))
)

#: Combining accents, which Unicode puts after a letter, mapped to the accent command
#: that writes them: ``ψ`` and a combining macron is ``\bar{\psi}``.
COMBINING_ACCENTS: Mapping[str, str] = MappingProxyType(
    {
        "\N{COMBINING CIRCUMFLEX ACCENT}": "hat",
        "\N{COMBINING TILDE}": "tilde",
        "\N{COMBINING MACRON}": "bar",
        "\N{COMBINING OVERLINE}": "bar",
        "\N{COMBINING BREVE}": "breve",
        "\N{COMBINING DOT ABOVE}": "dot",
        "\N{COMBINING DIAERESIS}": "ddot",
        "\N{COMBINING CARON}": "check",
        "\N{COMBINING RIGHT ARROW ABOVE}": "vec",
    }
)

# --------------------------------------------------------------------------
# Every command
# --------------------------------------------------------------------------

#: The physics and braket packages' commands for Dirac's notation, which the parser reads as
#: what they print (:mod:`relativisticpy.language.dirac`).
DIRAC_COMMANDS: frozenset[str] = frozenset({"ket", "bra", "braket", "ketbra", "expval", "mel"})

#: Commands the parser reads itself: fractions, roots, big operators, derivatives,
#: binomials, floors and ceilings, TeX's generalised fractions, the brackets of the
#: Stirling numbers among them, circular convolution, and Dirac's notation.
PARSER_COMMANDS: frozenset[str] = frozenset(
    {
        "frac", "dfrac", "tfrac", "cfrac", "sqrt", "sum", "prod", "int", "oint", "iint",
        "iiint", "lim", "partial", "nabla", "Box", "square", "pounds", "pdv", "dv",
        "binom", "dbinom", "tbinom", "lfloor", "rfloor", "lceil", "rceil", "over", "choose",
        "sup", "inf", "underset",
        "brace", "brack", "bmod", "mod", "pmod", "circledast",
        *DIRAC_COMMANDS,
    }
)  # fmt: skip

#: Commands the lexer reads itself: brackets, environments, plain TeX's groups and rows, and
#: a label set over a relation, as iid is in ``\overset{\text{iid}}{\sim}``.
STRUCTURE_COMMANDS: frozenset[str] = frozenset(
    {"left", "middle", "right", "begin", "end", "bgroup", "egroup", "cr", "mathbb", "overset",
     "stackrel"}
)  # fmt: skip

#: Every command RelativisticPy reads, which is where "did you mean" looks.
COMMANDS: frozenset[str] = frozenset().union(
    SYMBOL_COMMANDS,
    FUNCTION_COMMANDS,
    ACCENT_COMMANDS,
    OPERATOR_COMMANDS,
    RELATION_COMMANDS,
    CONNECTIVE_COMMANDS,
    ELLIPSIS_COMMANDS,
    TEXT_COMMANDS,
    STYLE_COMMANDS,
    FONT_DECLARATIONS,
    LAYOUT_COMMANDS,
    LAYOUT_COMMANDS_WITH_ARGUMENT,
    LINE_BREAK_COMMANDS,
    SIZE_COMMANDS,
    PLAIN_ENVIRONMENTS,
    PARSER_COMMANDS,
    STRUCTURE_COMMANDS,
)

_MACRO = "write what it stands for; relativisticpy check reads the macros of a document"

#: Commands of LaTeX and its packages, such as physics, that RelativisticPy does not
#: read, each with what it is and what to write instead. "Did you mean" never offers
#: another command for one of these.
UNSUPPORTED_COMMANDS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "newcommand": (r"\newcommand defines a macro, which a program cannot", _MACRO),
        "renewcommand": (r"\renewcommand defines a macro, which a program cannot", _MACRO),
        "providecommand": (r"\providecommand defines a macro, which a program cannot", _MACRO),
        "def": (r"\def defines a macro, which a program cannot", _MACRO),
        "let": (r"\let defines a macro, which a program cannot", _MACRO),
        "abs": (r"\abs, the physics package's absolute value, is not supported",
                r"write |x|, or \left| x \right|"),
        "norm": (r"\norm, the physics package's norm, is not supported", r"write \| x \|"),
        "qty": (r"\qty, the physics package's brackets, is not supported",
                r"write \left( ... \right)"),
        "dd": (r"\dd, the physics package's differential, is not supported",
               r"write d x, or \mathrm{d} x"),
        "tr": (r"\tr, the physics package's trace, is not supported",
               r"write \operatorname{tr}(A), or contract the indices: A^{a}_{a}"),
        "Tr": (r"\Tr, the physics package's trace, is not supported",
               r"write \operatorname{Tr}(A), or contract the indices: A^{a}_{a}"),
        "vb": (r"\vb, the physics package's bold vector, is not supported", r"write \mathbf{v}"),
        "va": (r"\va, the physics package's vector arrow, is not supported", r"write \vec{v}"),
        "order": (r"\order, the physics package's order, is not supported",
                  r"write the terms out, or \mathcal{O}(x^2) as a symbol"),
        "eval": (r"\eval, the physics package's evaluation bar, is not supported",
                 r"write \left. f \right|_{x=0}"),
        "liminf": (r"\liminf is not supported", r"write \lim_{x \to a} for a limit"),
        "limsup": (r"\limsup is not supported", r"write \lim_{x \to a} for a limit"),
        "forall": (r"the quantifier \forall is not supported", r"write \text{for all}"),
        "exists": (r"the quantifier \exists is not supported", r"write \text{there is}"),
        "not": (r"\not is not supported", r"write \neq for not equal, or \notin"),
    }
)  # fmt: skip
