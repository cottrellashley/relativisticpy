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

#: Variant letterforms that denote the same symbol as the plain letter.
GREEK_VARIANTS: Mapping[str, str] = MappingProxyType(
    {
        "varepsilon": "epsilon",
        "vartheta": "theta",
        "varphi": "phi",
        "varrho": "rho",
        "varsigma": "sigma",
        "varkappa": "kappa",
    }
)

#: Commands that stand for a symbol, mapped to the name of that symbol.
SYMBOL_COMMANDS: Mapping[str, str] = MappingProxyType(
    {
        **{letter: letter for letter in GREEK},
        **GREEK_VARIANTS,
        "varpi": "varpi",
        "ell": "ell",
        "hbar": "hbar",
        "infty": "infty",
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
        "Re": "re", "Im": "im", "sgn": "sign",
    }
)  # fmt: skip

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
        "mid": TokenKind.BAR,
    }
)

#: Commands whose argument is prose, not mathematics. The argument becomes a
#: single name (``\text{otherwise}``) or, if it contains spaces, a string.
TEXT_COMMANDS: frozenset[str] = frozenset(
    {"text", "textrm", "textit", "textbf", "textsf", "texttt", "textup", "mbox", "operatorname"}
)

#: Commands that only change the typeface of their argument. The argument is
#: lexed as ordinary mathematics: ``\mathrm{d}x`` reads exactly like ``d x``.
STYLE_COMMANDS: frozenset[str] = frozenset(
    {
        "mathrm", "mathit", "mathbf", "mathsf", "mathtt", "mathcal", "mathbb",
        "mathfrak", "mathscr", "mathnormal", "boldsymbol", "bm", "pmb",
    }
)  # fmt: skip

#: Typesetting commands without arguments; they are ignored.
LAYOUT_COMMANDS: frozenset[str] = frozenset(
    {
        "displaystyle", "textstyle", "scriptstyle", "scriptscriptstyle",
        "limits", "nolimits", "nonumber", "notag", "allowbreak", "displaybreak",
        "quad", "qquad", "enspace", "thinspace", "medspace", "thickspace",
        "negthinspace", "negmedspace", "negthickspace", "hfill", "hline",
        "noindent", "centering",
    }
)  # fmt: skip

#: The layout commands that insert horizontal space.
SPACING_COMMANDS: frozenset[str] = frozenset(
    {
        "quad", "qquad", "enspace", "thinspace", "medspace", "thickspace",
        "negthinspace", "negmedspace", "negthickspace",
    }
)  # fmt: skip

#: Typesetting commands with one braced argument (``\label{eq:1}``); ignored.
LAYOUT_COMMANDS_WITH_ARGUMENT: frozenset[str] = frozenset(
    {"label", "tag", "vspace", "hspace", "phantom", "hphantom", "vphantom"}
)

#: Commands that end a line, and therefore a statement.
LINE_BREAK_COMMANDS: frozenset[str] = frozenset({"newline", "linebreak", "cr"})

#: Delimiter sizing commands (``\big(``, ``\Bigl[`` ...).
SIZE_COMMANDS: frozenset[str] = frozenset(
    size + suffix for size in ("big", "Big", "bigg", "Bigg") for suffix in ("", "l", "r", "m")
)

# --------------------------------------------------------------------------
# Environments
# --------------------------------------------------------------------------

#: Environments that only lay out equations. Their contents are statements,
#: ``\\`` separates statements and ``&`` alignment markers are ignored.
ALIGNMENT_ENVIRONMENTS: frozenset[str] = frozenset(
    {
        "align", "align*", "aligned", "alignat", "alignat*", "alignedat",
        "equation", "equation*", "gather", "gather*", "gathered", "split",
        "multline", "multline*", "eqnarray", "eqnarray*", "displaymath", "math",
    }
)  # fmt: skip

#: Matrix environments. ``vmatrix`` denotes the determinant of its entries.
MATRIX_ENVIRONMENTS: frozenset[str] = frozenset(
    {"matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix", "smallmatrix", "array"}
)

#: Piecewise definitions.
CASES_ENVIRONMENTS: frozenset[str] = frozenset({"cases", "dcases"})

#: Environments followed by a mandatory argument that is not mathematics,
#: such as the column specification of ``\begin{array}{cc}``.
ENVIRONMENTS_WITH_ARGUMENT: frozenset[str] = frozenset(
    {"array", "alignat", "alignat*", "alignedat"}
)

# --------------------------------------------------------------------------
# Unicode
# --------------------------------------------------------------------------

#: Unicode letters, mapped to the name of the symbol they denote.
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
        "∞": "infty", "ℏ": "hbar", "ℓ": "ell",
    }
)  # fmt: skip

#: Unicode mathematical operators that have a LaTeX command equivalent.
UNICODE_COMMANDS: Mapping[str, str] = MappingProxyType(
    {
        "∂": "partial", "∇": "nabla", "√": "sqrt", "∑": "sum", "∏": "prod",
        "∫": "int", "∬": "iint", "∭": "iiint", "∮": "oint",
    }
)  # fmt: skip

#: Unicode operators that lex directly as an operator token.
UNICODE_OPERATORS: Mapping[str, TokenKind] = MappingProxyType(
    {
        "·": TokenKind.STAR,
        "⋅": TokenKind.STAR,
        "×": TokenKind.STAR,
        "÷": TokenKind.SLASH,
        "−": TokenKind.MINUS,
        "≤": TokenKind.LESS_EQ,
        "≥": TokenKind.GREATER_EQ,
        "≠": TokenKind.NOT_EQ,
        "→": TokenKind.ARROW,
        "↦": TokenKind.ARROW,
        "≡": TokenKind.DEFINE,
        "≔": TokenKind.DEFINE,
        "′": TokenKind.PRIME,
    }
)

#: Unicode superscript characters (``r²``) and their plain equivalents.
UNICODE_SUPERSCRIPTS: Mapping[str, str] = MappingProxyType(
    dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿ", "0123456789+-n", strict=True))
)

#: Unicode subscript characters (``x₀``) and their plain equivalents.
UNICODE_SUBSCRIPTS: Mapping[str, str] = MappingProxyType(
    dict(zip("₀₁₂₃₄₅₆₇₈₉₊₋", "0123456789+-", strict=True))
)
