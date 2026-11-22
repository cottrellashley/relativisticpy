r"""How results look: one printer for every surface.

Every value RelativisticPy gives back is shown in one of four forms, so that a
result reads the same at the command line, at the interactive prompt, in a
notebook, in the report of a check and in machine-readable output:

- :func:`unicode`: mathematics for a terminal or a ``repr``;
- :func:`text`: plain ASCII, in the language's own syntax;
- :func:`latex`: LaTeX, for notebooks and papers;
- :func:`data`: a JSON-serialisable description, for programs and agents.

    >>> import sympy
    >>> M, r = sympy.symbols("M r")
    >>> unicode(48 * M**2 / r**6)
    '48M²/r⁶'
    >>> text(48 * M**2 / r**6)
    '48*M^2/r^6'
    >>> latex(48 * M**2 / r**6)
    '\\frac{48 M^{2}}{r^{6}}'

The text and LaTeX forms read back: the language parses them as the value they show.
So does the Unicode form, wherever the language reads Unicode. Euler's number is
``e`` and the imaginary unit ``i``, so that a symbol ``E`` is never mistaken for
either; numbers are shown as written, and an integer too long to read is shortened,
with its number of digits:

    >>> E = sympy.Symbol("E")
    >>> unicode(E + sympy.E), unicode(sympy.Float(0.1) * 3), unicode(sympy.factorial(100))
    ('E + e', '0.30000000000000004', '9.332621544394415…×10¹⁵⁷ (an integer of 158 digits)')

Tensors are shown by their name and indices, then a line for each non-zero
component, labelled by coordinate; equations say whether they hold and, if not,
where; numerical data is summarised, never dumped.

Notebooks show RelativisticPy's values through this module too: Jupyter's
``_repr_latex_`` and ``_repr_html_`` give :func:`jupyter_latex` and
:func:`jupyter_html`, and marimo's ``_display_`` gives :func:`marimo`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal

import sympy
from sympy.core.function import AppliedUndef
from sympy.printing.latex import LatexPrinter
from sympy.printing.precedence import PRECEDENCE, precedence
from sympy.printing.str import StrPrinter

from relativisticpy.language.vocabulary import GREEK

__all__ = ["data", "html", "jupyter_html", "jupyter_latex", "latex", "marimo", "text", "unicode"]

#: The forms a value is shown in.
Form = Literal["unicode", "text", "latex"]

# -- the vocabulary of names ------------------------------------

#: Greek letters, by name, as Unicode writes them.
GREEK_LETTERS: Mapping[str, str] = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "zeta": "ζ",
    "eta": "η", "theta": "θ", "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ",
    "nu": "ν", "xi": "ξ", "omicron": "ο", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ",
    "upsilon": "υ", "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π",
    "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}  # fmt: skip

#: Other names that stand for a symbol a LaTeX command writes, and their Unicode.
SYMBOL_NAMES: Mapping[str, str] = {
    "varpi": "ϖ", "hbar": "ℏ", "ell": "ℓ", "odot": "⊙", "star": "⋆", "infty": "∞",
}  # fmt: skip

#: Accents, as a symbol's name ends with them (``adot`` is ``\dot{a}``), and the
#: combining character Unicode puts over a letter for each.
ACCENTS: Mapping[str, str] = {
    "ddot": "̈", "dot": "̇", "hat": "̂", "bar": "̄",
    "tilde": "̃", "vec": "⃗", "check": "̌", "breve": "̆",
}  # fmt: skip

_SUPERSCRIPTS = str.maketrans("0123456789-+()=n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺⁽⁾⁼ⁿ")
_SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_MINUS = "−"  # U+2212, which the language reads as a minus

#: An integer with more digits than this is shortened.
LONG_INTEGER = 80

#: The floats that are not numbers, as SymPy names them.
_SPECIAL_FLOATS = {"inf": sympy.oo, "-inf": -sympy.oo, "nan": sympy.nan}


def superscript(text: str) -> str:
    """Digits and signs as Unicode superscripts: ``-12`` → ``⁻¹²``."""
    return text.replace(_MINUS, "-").translate(_SUPERSCRIPTS)


def _is_double_struck(name: str) -> bool:
    return len(name) == 1 and not name.isascii() and name.isalnum() and name not in "αβγ"


def _double_struck_letter(name: str) -> str:
    """The letter a double-struck character such as ℝ is written with in ``\\mathbb``."""
    special = {"ℂ": "C", "ℍ": "H", "ℕ": "N", "ℙ": "P", "ℚ": "Q", "ℝ": "R", "ℤ": "Z"}
    if name in special:
        return special[name]
    code = ord(name)
    for start, first in ((0x1D538, "A"), (0x1D552, "a"), (0x1D7D8, "0")):
        if start <= code < start + (10 if first == "0" else 26):
            return chr(ord(first) + code - start)
    return name


class Name:
    """A symbol's name taken apart, as the language reads it from LaTeX: ``rho_vac`` is
    ρ with the subscript vac, ``adot`` is a with a dot, ``x'`` is x primed."""

    __slots__ = ("accent", "base", "primes", "subscripts")

    def __init__(self, name: str) -> None:
        primes = len(name) - len(name.rstrip("'"))
        core = name[: len(name) - primes] if primes else name
        base, *subscripts = core.split("_") if core and not core.startswith("_") else [core]
        accent = None
        if base not in GREEK and base not in SYMBOL_NAMES:
            for suffix in ACCENTS:  # the longest first: ddot before dot
                if base.endswith(suffix) and len(base) > len(suffix):
                    accent, base = suffix, base[: -len(suffix)]
                    break
        self.base, self.subscripts, self.accent, self.primes = base, subscripts, accent, primes

    @property
    def simple(self) -> bool:
        """Whether the name is one plain letter, perhaps with digits after it."""
        return bool(re.fullmatch(r"[A-Za-z]\d*", self.base)) and not self.subscripts


def _unicode_word(word: str) -> str:
    """A name or a subscript in Unicode: a Greek letter as itself."""
    if word in GREEK_LETTERS:
        return GREEK_LETTERS[word]
    if word in SYMBOL_NAMES and word != "star":  # ⋆ is not read
        return SYMBOL_NAMES[word]
    return word


def unicode_name(name: str) -> str:
    """A symbol's name in Unicode: ``theta`` → ``θ``, ``x_0`` → ``x₀``, ``adot`` → ``ȧ``,
    ``M_odot`` → ``M_⊙``, and a name of two, the change in x, ``Delta x`` → ``Δx``."""
    head, space, rest = name.partition(" ")
    if space:
        return unicode_name(head) + unicode_name(rest)
    if name in GREEK_LETTERS:
        return GREEK_LETTERS[name]
    if name[:1] == "d" and name[1:] in GREEK_LETTERS:  # the differential of a coordinate
        return "d" + GREEK_LETTERS[name[1:]]
    parts = Name(name)
    base = _unicode_word(parts.base)
    if parts.accent is not None:
        base += ACCENTS[parts.accent]
    for subscript in parts.subscripts:
        if subscript.isdigit():
            base += subscript.translate(_SUBSCRIPTS)
        else:
            base += "_" + _unicode_word(subscript)
    return base + "'" * parts.primes


def _latex_word(word: str, *, upright: bool) -> str:
    """A name or a subscript in LaTeX: Greek letters and the symbols of commands as their
    command, a word of several letters upright, which the language reads as one name."""
    if word in GREEK or word in ("varpi", "hbar", "ell", "odot", "star", "infty"):
        return "\\" + word
    if _is_double_struck(word):
        return f"\\mathbb{{{_double_struck_letter(word)}}}"
    if len(word) > 1 and not word.isdigit() and not re.fullmatch(r"[A-Za-z]\d+", word):
        return f"\\mathrm{{{word}}}" if upright else word
    return word


def latex_name(name: str) -> str:
    r"""A symbol's name in LaTeX that the language reads back as that name:
    ``rho_vac`` → ``\rho_{\mathrm{vac}}``, ``M_odot`` → ``M_{\odot}``, ``adot`` →
    ``\dot{a}``, ``Kretschmann`` → ``\mathrm{Kretschmann}``, ``Delta x`` → ``\Delta x``."""
    head, space, rest = name.partition(" ")
    if space:
        return f"{latex_name(head)} {latex_name(rest)}"
    if name[:1] == "d" and name[1:] in GREEK:  # the differential of a coordinate, dtheta
        return "d\\" + name[1:]
    parts = Name(name)
    base = _latex_word(parts.base, upright=True)
    if parts.accent is not None:
        base = f"\\{parts.accent}{{{base}}}"
    if parts.subscripts:
        subscripts = "".join(_latex_word(s, upright=True) for s in parts.subscripts[:1])
        base += "_{" + subscripts + "}"
        for extra in parts.subscripts[1:]:  # x_a_b, as rare as it is ambiguous
            base = "{" + base + "}_{" + _latex_word(extra, upright=True) + "}"
    return base + "'" * parts.primes


def text_name(name: str, *, latex_mode: bool = False) -> str:
    r"""A symbol's name as plain text that the language reads back as that name. A Greek
    letter is its LaTeX command, ``\theta``; any other name is written as it is, except
    that in a statement read as LaTeX, where ``Mr`` is M times r, a name of several
    letters is ``\mathrm{Mr}``; a name of two, as ``Delta x``, is the two, ``\Delta x``."""
    head, space, rest = name.partition(" ")
    if space:
        latex = latex_mode or head in GREEK
        return f"{text_name(head, latex_mode=latex)} {text_name(rest, latex_mode=latex)}"
    if name in GREEK:
        return "\\" + name
    if name in ("hbar", "ell", "varpi"):
        return "\\" + name
    if _is_double_struck(name):
        return f"\\mathbb{{{_double_struck_letter(name)}}}"
    if latex_mode and re.fullmatch(r"[A-Za-z]{2,}\d*'*", name):
        return f"\\mathrm{{{name}}}"
    return name


# -- numbers ------------------------------------


def digits(n: int) -> int:
    """The number of decimal digits of an integer, without converting it to a string,
    which Python refuses beyond a few thousand digits."""
    n = abs(n)
    if n < 10:
        return 1
    count = max(1, int(n.bit_length() * 0.30102999566398114))
    power = 10 ** (count - 1)
    while power > n:
        count, power = count - 1, power // 10
    while power * 10 <= n:
        count, power = count + 1, power * 10
    return count


def _long_integer(n: int) -> tuple[str, str, int] | None:
    """A long integer as its first digits and its exponent: ``("9.332621544394415", "157",
    158)`` for 100!; ``None`` for an integer short enough to show."""
    count = digits(n)
    if count <= LONG_INTEGER:
        return None
    head = str(abs(n) // 10 ** (count - 16))
    sign = "-" if n < 0 else ""
    return f"{sign}{head[0]}.{head[1:]}", str(count - 1), count


def _float_text(value: Any) -> str:
    """A floating-point number as the shortest text that reads back as it: ``0.3``,
    ``6.02214076e+23``. A SymPy float of more than double precision keeps its digits."""
    number = sympy.Float(value)
    if number._prec <= 53:  # a double: Python finds the shortest form that reads back
        written = repr(float(number))
    else:
        written = sympy.StrPrinter({"full_prec": True}).doprint(number)
    return written


def _float_parts(value: Any) -> tuple[str, str | None]:
    """A float as its mantissa and decimal exponent, the mantissa with a decimal point so
    that it reads back as a float: ``("6.02214076", "23")``, ``("0.3", None)``."""
    written = _float_text(value)
    if written in ("inf", "-inf", "nan"):
        return written, None
    mantissa, _, exponent = written.lower().partition("e")
    if not exponent:
        return mantissa, None
    if "." not in mantissa:
        mantissa += ".0"
    return mantissa, str(int(exponent))


# -- the order of terms ------------------------------------


def _ordered_terms(terms: list[Any]) -> list[Any]:
    """Terms with a positive one first, as written by hand: ``r - 2M``, not ``-2M + r``,
    and derivatives first, as a differential equation is: ``t''(τ) + …``."""
    derivatives = [term for term in terms if isinstance(term, sympy.Derivative)]
    if derivatives:
        terms = derivatives + [term for term in terms if not isinstance(term, sympy.Derivative)]
    if terms and terms[0].could_extract_minus_sign():
        for position, term in enumerate(terms):
            if not term.could_extract_minus_sign():
                return [term, *terms[:position], *terms[position + 1 :]]
    return terms


def _is_unknown_function(expr: Any) -> bool:
    return isinstance(expr, AppliedUndef)


def _book(
    expr: Any, form: Literal["latex", "unicode"], show: Callable[[Any], str]
) -> tuple[str, bool] | None:
    """A call of a capability an integration declares written forms for, as the first of
    them in ``form`` writes it, as ``J_{0}\\left(x\\right)``, with whether it is written
    with scripts; ``None`` for anything else (see :mod:`relativisticpy.integrations`)."""
    if not isinstance(expr, sympy.Function) or isinstance(expr, AppliedUndef):
        return None
    from relativisticpy.integrations import _forms

    return _forms.printed(expr, form, show)


def _lagrange(expr: Any) -> tuple[Any, int] | None:
    """The function and the order of a derivative written with primes, ``f''(x)``: an
    unknown function of one variable, differentiated by it at most three times."""
    function = expr.expr
    if not (_is_unknown_function(function) and len(function.args) == 1):
        return None
    variable = function.args[0]
    if not isinstance(variable, sympy.Symbol):
        return None
    if len(expr.variable_count) != 1 or expr.variable_count[0][0] != variable:
        return None
    order = int(expr.variable_count[0][1])
    return (function, order) if 1 <= order <= 3 else None


# -- the plain printers: Unicode and text ------------------------------------


class _PlainPrinter(StrPrinter):  # type: ignore[misc]
    """One line of mathematics in the language's syntax: in Unicode (``48M²/r⁶``) or in
    ASCII (``48*M^2/r^6``)."""

    printmethod = "_relativisticpy_plain"

    def __init__(
        self,
        *,
        unicode: bool,
        latex_mode: bool = False,
        names: Mapping[Any, str] | None = None,
    ) -> None:
        super().__init__({"order": None})
        self.unicode = unicode
        self.latex_mode = latex_mode  # text read as LaTeX: protect names of several letters
        self.names = dict(names or {})  # symbols that stand for something already written

    # -- names --

    def _name(self, name: str) -> str:
        return unicode_name(name) if self.unicode else text_name(name, latex_mode=self.latex_mode)

    def _print(self, expr: Any, **settings: Any) -> str:
        """A call of a capability in its Unicode form, as ``Γ(x)``; anything else as the
        printer's own methods print it."""
        if self.unicode and (book := _book(expr, "unicode", self._print)) is not None:
            return book[0]
        return str(super()._print(expr, **settings))

    def _print_Symbol(self, expr: Any) -> str:
        if expr in self.names:
            return str(self.names[expr])
        return self._name(expr.name)

    _print_Dummy = _print_Symbol

    def _print_Exp1(self, expr: Any) -> str:
        return "e"

    def _print_ImaginaryUnit(self, expr: Any) -> str:
        return "i"

    def _print_Pi(self, expr: Any) -> str:
        return "π" if self.unicode else "pi"

    def _print_Infinity(self, expr: Any) -> str:
        return "∞" if self.unicode else "oo"

    def _print_NegativeInfinity(self, expr: Any) -> str:
        return f"{_MINUS}∞" if self.unicode else "-oo"

    def _print_ComplexInfinity(self, expr: Any) -> str:
        return "complex ∞" if self.unicode else "zoo"

    def _print_NaN(self, expr: Any) -> str:
        return "undefined" if self.unicode else "nan"

    def _print_BooleanTrue(self, expr: Any) -> str:
        return "True"

    def _print_BooleanFalse(self, expr: Any) -> str:
        return "False"

    # -- numbers --

    def _minus(self, text: str) -> str:
        return text.replace("-", _MINUS) if self.unicode else text

    def _print_Integer(self, expr: Any) -> str:
        return self._integer(int(expr.p))

    def _integer(self, n: int) -> str:
        long = _long_integer(n)
        if long is None:
            return self._minus(str(n))
        mantissa, exponent, _ = long
        if self.unicode:
            return f"{self._minus(mantissa)}…×10{superscript(exponent)}"
        return f"{mantissa}...*10^{exponent}"

    def _print_Rational(self, expr: Any) -> str:
        return f"{self._integer(int(expr.p))}/{self._integer(int(expr.q))}"

    def _print_Float(self, expr: Any) -> str:
        mantissa, exponent = _float_parts(expr)
        if mantissa in ("inf", "-inf", "nan"):
            return str(self._print(_SPECIAL_FLOATS[mantissa]))
        if exponent is None:
            return self._minus(mantissa)
        if self.unicode:
            return f"{self._minus(mantissa)}×10{superscript(exponent)}"
        return f"{mantissa}e{'+' if not exponent.startswith('-') else ''}{exponent}"

    # -- sums, products and powers --

    def _as_ordered_terms(self, expr: Any, order: Any = None) -> list[Any]:
        return _ordered_terms(list(super()._as_ordered_terms(expr, order=order)))

    def _print_Add(self, expr: Any, order: Any = None) -> str:
        terms = self._as_ordered_terms(expr, order=order)
        pieces: list[str] = []
        for term in terms:
            negative = term.could_extract_minus_sign() and not term.is_Add
            shown = self._print(-term) if negative else self._print(term)
            if precedence(term) < PRECEDENCE["Add"] or term.is_Add:
                shown = f"({shown})"
            if pieces:
                pieces.append(f" {_MINUS if self.unicode else '-'} " if negative else " + ")
            elif negative:
                pieces.append(_MINUS if self.unicode else "-")
            pieces.append(shown)
        return "".join(pieces)

    def _factor(self, expr: Any) -> str:
        """A factor of a product, in brackets if it is a sum or a signed number."""
        shown = self._print(expr)
        if precedence(expr) < PRECEDENCE["Mul"] or (
            expr.is_Number and expr.could_extract_minus_sign()
        ):
            return f"({shown})"
        if expr.is_Rational and not expr.is_Integer:
            return f"({shown})"
        return str(shown)

    def _join(self, factors: Sequence[tuple[Any, str]]) -> str:
        """Factors multiplied: by juxtaposition in Unicode, where that cannot be misread
        (``2M``, ``M r``, ``2 sin(θ)``), and with ``*`` in text."""
        if not self.unicode:
            return "*".join(shown for _, shown in factors)
        result = ""
        previous: Any = None
        for expr, shown in factors:
            if previous is None:
                result = shown
            elif previous.is_Number and expr.is_Number:
                result += "·" + shown
            elif previous.is_Number and _attaches(expr, shown):
                result += shown
            else:
                result += " " + shown
            previous = expr
        return result

    def _print_Mul(self, expr: Any) -> str:
        coefficient, _ = expr.as_coeff_Mul()
        sign = ""
        if coefficient.is_Number and coefficient.is_extended_negative:
            expr = -expr
            sign = _MINUS if self.unicode else "-"
            if not expr.is_Mul:
                return sign + self._factor(expr) if expr.is_Add else sign + self._print(expr)
        numerator: list[Any] = []
        denominator: list[Any] = []
        for factor in expr.as_ordered_factors():
            if factor.is_Pow and factor.exp.is_number and factor.exp.is_extended_negative:
                exponent = -factor.exp
                denominator.append(
                    factor.base
                    if exponent == 1
                    else sympy.Pow(factor.base, exponent, evaluate=False)
                )
            elif factor.is_Rational and not factor.is_Integer:
                if factor.p != 1:
                    numerator.append(sympy.Integer(factor.p))
                denominator.append(sympy.Integer(factor.q))
            else:
                numerator.append(factor)
        top = self._join([(f, self._factor(f)) for f in numerator]) if numerator else "1"
        if not denominator:
            return sign + top
        bottom = self._join([(f, self._factor(f)) for f in denominator])
        if len(denominator) > 1:
            bottom = f"({bottom})"
        return f"{sign}{top}/{bottom}"  # a b/c is (a b)/c, as / and * bind alike

    def _base(self, base: Any) -> str:
        """The base of a power, in brackets unless it is a symbol, a positive integer or
        a function's value."""
        shown = self._print(base)
        book = _book(base, "unicode", self._print) if self.unicode else None
        bare = (
            base.is_Symbol
            or (base.is_Integer and base >= 0)
            or base in (sympy.pi, sympy.E, sympy.I)
            or (
                isinstance(base, sympy.Function)
                and not isinstance(base, sympy.factorial)
                and not (book is not None and book[1])  # J_{1/2}(x), whose script binds first
            )
            or (isinstance(base, sympy.Derivative) and _lagrange(base) is not None)
        )
        return shown if bare else f"({shown})"

    def _exponent(self, exponent: Any) -> str:
        shown = self._print(exponent)
        if (
            exponent.is_Symbol
            or (exponent.is_Integer and exponent >= 0)
            or exponent
            in (
                sympy.pi,
                sympy.E,
                sympy.I,
            )
        ):
            return str(shown)
        if exponent.is_Integer and not self.unicode:
            return str(shown)  # x^-1
        return f"({shown})"

    def _print_Pow(self, expr: Any, rational: bool = False) -> str:
        base, exponent = expr.base, expr.exp
        if base is sympy.E:
            return f"e^{self._exponent(exponent)}"
        if exponent is sympy.S.Half:
            return self._root(base)
        if exponent.is_Rational and exponent.is_extended_negative:
            inverse = sympy.Pow(base, -exponent, evaluate=False)
            return f"1/{self._print(inverse) if inverse.exp != 1 else self._factor(base)}"
        if self.unicode and exponent.is_Integer:
            return self._base(base) + superscript(str(exponent))
        return f"{self._base(base)}^{self._exponent(exponent)}"

    def _root(self, base: Any) -> str:
        if not self.unicode:
            return f"sqrt({self._print(base)})"
        shown = self._print(base)
        if base.is_Symbol or (base.is_Integer and base >= 0):
            return f"√{shown}"
        return f"√({shown})"

    def _print_exp(self, expr: Any) -> str:
        return f"e^{self._exponent(expr.args[0])}"

    # -- functions --

    def _call(self, name: str, args: Iterable[Any]) -> str:
        return f"{name}({', '.join(self._print(a) for a in args)})"

    def _print_Function(self, expr: Any) -> str:
        if _is_unknown_function(expr):
            name = expr.func.__name__
            if self.unicode:
                shown = unicode_name(name)
            elif name in GREEK:  # \gamma(r), the unknown function, not gamma(r), Euler's
                shown = "\\" + name
            else:
                shown = text_name(name, latex_mode=self.latex_mode)
            return self._call(shown, expr.args)
        return self._call(expr.func.__name__, expr.args)

    def _print_Abs(self, expr: Any) -> str:
        if self.unicode:
            return f"|{self._print(expr.args[0])}|"
        return self._call("abs", expr.args)

    def _print_factorial(self, expr: Any) -> str:
        argument = expr.args[0]
        shown = self._print(argument)
        if not (argument.is_Symbol or (argument.is_Integer and argument >= 0)):
            shown = f"({shown})"
        return f"{shown}!"

    def _print_gamma(self, expr: Any) -> str:
        return self._call("gamma", expr.args)

    def _print_Max(self, expr: Any) -> str:
        return self._call("max", expr.args)

    def _print_Min(self, expr: Any) -> str:
        return self._call("min", expr.args)

    def _print_Derivative(self, expr: Any) -> str:
        lagrange = _lagrange(expr)
        if lagrange is not None:
            function, order = lagrange
            return str(self._print(function)).replace("(", "'" * order + "(", 1)
        function = expr.expr
        if self.unicode:
            operators = []
            for variable, count in expr.variable_count:
                name = self._print(variable)
                name = f"{{{name}}}" if len(name) > 1 else name
                operators.append(f"∂_{name}" + (superscript(str(count)) if count > 1 else ""))
            shown = self._print(function)
            if not (_is_unknown_function(function) or function.is_Symbol):
                shown = f"({shown})"
            return " ".join(operators) + " " + str(shown)
        variables = []
        for variable, count in expr.variable_count:
            variables.append(self._print(variable))
            if count > 1:
                variables.append(str(count))
        return f"diff({self._print(function)}, {', '.join(variables)})"

    def _limits(self, limits: Any) -> list[str]:
        return [self._print(part) for part in limits]

    def _print_Integral(self, expr: Any) -> str:
        if self.unicode:
            signs, differentials = [], []
            for limit in expr.limits:
                variable, *bounds = limit
                sign = "∫"
                if len(bounds) == 2:
                    low, high = (self._script(self._print(b)) for b in bounds)
                    sign += f"_{low}^{high}"
                signs.append(sign)
                differentials.append("d" + self._print(variable))
            body = self._print(expr.function)
            if precedence(expr.function) < PRECEDENCE["Mul"]:
                body = f"({body})"
            return " ".join(signs) + f" {body} " + " ".join(differentials)
        parts = [self._print(expr.function)]
        for limit in expr.limits:
            if len(limit) == 1:
                parts.append(self._print(limit[0]))
            else:
                parts.append(f"({', '.join(self._print(p) for p in limit)})")
        return f"integrate({', '.join(parts)})"

    def _script(self, shown: str) -> str:
        return shown if len(shown) == 1 else f"{{{shown}}}"

    def _big_operator(self, sign: str, command: str, expr: Any) -> str:
        pieces = []
        for variable, low, high in expr.limits:
            if self.unicode:
                lower = self._script(f"{self._print(variable)}={self._print(low)}")
                pieces.append(f"{sign}_{lower}^{self._script(self._print(high))}")
            else:
                pieces.append(
                    f"\\{command}_{{{self._print(variable)}={self._print(low)}}}"
                    f"^{{{self._print(high)}}}"
                )
        body = self._print(expr.function)
        if precedence(expr.function) <= PRECEDENCE["Add"]:
            body = f"({body})"
        return " ".join(pieces) + " " + str(body)

    def _print_Sum(self, expr: Any) -> str:
        return self._big_operator("∑", "sum", expr)

    def _print_Product(self, expr: Any) -> str:
        return self._big_operator("∏", "prod", expr)

    def _print_Limit(self, expr: Any) -> str:
        # As the language reads it in either form: lim_{x→0⁺} is not read, and text in
        # quotes is no value, so a limit from both sides is written as LaTeX is.
        function, variable, point, direction = expr.args
        arguments = [self._print(function), self._print(variable), self._print(point)]
        if str(direction) == "+":
            return f"limit({', '.join(arguments)})"
        side = {"-": "^-", "+-": ""}.get(str(direction), "")
        return rf"\lim_{{{arguments[1]} \to {arguments[2]}{side}}} {arguments[0]}"

    def _print_Piecewise(self, expr: Any) -> str:
        pieces = []
        for value, condition in expr.args:
            shown = "otherwise" if condition is sympy.true else self._print(condition)
            pieces.append(f"{self._print(value)}, {shown}")
        return "{ " + "; ".join(pieces) + " }"

    # -- relations and logic --

    _RELATIONS: Mapping[str, tuple[str, str]] = {
        "==": ("=", "="),
        "!=": ("≠", "!="),
        "<": ("<", "<"),
        ">": (">", ">"),
        "<=": ("≤", "<="),
        ">=": ("≥", ">="),
    }

    def _print_Relational(self, expr: Any) -> str:
        unicode_op, text_op = self._RELATIONS[expr.rel_op]
        lhs, rhs = self._print(expr.lhs), self._print(expr.rhs)
        operator = unicode_op if self.unicode else text_op
        if not self.unicode and expr.rel_op == "==" and _definable(expr.lhs):
            operator = "=="  # x = y would define x
        return f"{lhs} {operator} {rhs}"

    def _logic(self, expr: Any, word: str) -> str:
        pieces = []
        for argument in expr.args:
            shown = self._print(argument)
            if isinstance(argument, sympy.And | sympy.Or):
                shown = f"({shown})"
            pieces.append(shown)
        return f" {word} ".join(pieces)

    def _print_And(self, expr: Any) -> str:
        return self._logic(expr, "and")

    def _print_Or(self, expr: Any) -> str:
        return self._logic(expr, "or")

    def _print_Not(self, expr: Any) -> str:
        shown = self._print(expr.args[0])
        return (
            f"not ({shown})" if isinstance(expr.args[0], sympy.And | sympy.Or) else f"not {shown}"
        )

    # -- containers --

    def _print_list(self, expr: Any) -> str:
        return "[" + ", ".join(self._print(item) for item in expr) + "]"

    def _print_tuple(self, expr: Any) -> str:
        if len(expr) == 1:
            return f"({self._print(expr[0])},)"
        return "(" + ", ".join(self._print(item) for item in expr) + ")"

    _print_Tuple = _print_tuple

    def _print_NDimArray(self, expr: Any) -> str:
        return str(self._print(expr.tolist() if expr.rank() else expr[()]))

    _print_ImmutableDenseNDimArray = _print_NDimArray
    _print_MutableDenseNDimArray = _print_NDimArray
    _print_ImmutableSparseNDimArray = _print_NDimArray

    def _print_MatrixBase(self, expr: Any) -> str:
        return str(self._print(expr.tolist()))

    _print_ImmutableDenseMatrix = _print_MatrixBase
    _print_MutableDenseMatrix = _print_MatrixBase

    def _print_dict(self, expr: Any) -> str:
        items = (f"{self._print(key)}: {self._print(value)}" for key, value in expr.items())
        return "{" + ", ".join(items) + "}"

    def _print_str(self, expr: str) -> str:
        return '"' + expr.replace("\\", "\\\\").replace('"', '\\"') + '"'

    def emptyPrinter(self, expr: Any) -> str:
        return str(expr)


def _attaches(expr: Any, shown: str) -> bool:
    """Whether a factor can follow a number without a space and still be read as the
    product: ``2M``, ``2π``, ``2(x + 1)``, ``2√x``, but not ``2 sin(θ)``."""
    if shown.startswith(("(", "√")):
        return True
    if expr.is_Symbol or (expr.is_Pow and expr.base.is_Symbol and expr.exp.is_Integer):
        return True
    return expr in (sympy.pi, sympy.I, sympy.E) or (expr.is_Pow and expr.base in (sympy.pi,))


def _definable(expr: Any) -> bool:
    """Whether ``expr = ...`` would define ``expr`` rather than claim an equation: a plain
    name has no meaning yet in a new session."""
    return bool(expr.is_Symbol)


# -- LaTeX ------------------------------------


class _LatexPrinter(LatexPrinter):  # type: ignore[misc]
    """SymPy's LaTeX, but in the forms the language reads back: ``f(x)`` for an unknown
    function, ``M_{\\odot}`` and ``\\rho_{\\mathrm{vac}}`` for names, floats as written,
    long integers shortened, and terms with a positive one first."""

    printmethod = "_latex"

    def __init__(self, settings: Mapping[str, Any] | None = None) -> None:
        super().__init__(dict(settings or {}))

    def _print_Symbol(self, expr: Any, style: str = "plain") -> str:
        names = self._settings.get("symbol_names") or {}
        if expr in names:
            return str(names[expr])
        return latex_name(expr.name)

    _print_Dummy = _print_Symbol

    def _print_Integer(self, expr: Any) -> str:
        return self._integer(int(expr.p))

    def _integer(self, n: int) -> str:
        long = _long_integer(n)
        if long is None:
            return str(n)
        mantissa, exponent, _ = long
        return f"{mantissa}\\ldots \\times 10^{{{exponent}}}"

    def _print_Rational(self, expr: Any) -> str:
        if expr.q == 1:
            return self._integer(int(expr.p))
        sign = "- " if expr.p < 0 else ""
        return f"{sign}\\frac{{{self._integer(abs(int(expr.p)))}}}{{{self._integer(int(expr.q))}}}"

    def _print_Float(self, expr: Any) -> str:
        mantissa, exponent = _float_parts(expr)
        if mantissa in ("inf", "-inf", "nan"):
            return str(self._print(_SPECIAL_FLOATS[mantissa]))
        if exponent is None:
            return mantissa
        return f"{mantissa} \\times 10^{{{exponent}}}"

    def _print_BooleanTrue(self, expr: Any) -> str:
        return "\\text{True}"

    def _print_BooleanFalse(self, expr: Any) -> str:
        return "\\text{False}"

    def _as_ordered_terms(self, expr: Any, order: Any = None) -> list[Any]:
        return _ordered_terms(list(super()._as_ordered_terms(expr, order=order)))

    def _unknown(self, expr: Any, exp: Any = None) -> str:
        name = expr.func.__name__
        head = latex_name(name)
        if not Name(name).simple and name not in GREEK and not Name(name).subscripts:
            head = f"\\operatorname{{{name}}}"
        arguments = ", ".join(self._print(a) for a in expr.args)
        shown = f"{head}\\left({arguments}\\right)"
        return f"{shown}^{{{exp}}}" if exp is not None else shown

    def _print_Function(self, expr: Any, exp: Any = None) -> str:
        if _is_unknown_function(expr):
            return self._unknown(expr, exp)
        return str(super()._print_Function(expr, exp))

    def _print(self, expr: Any, **settings: Any) -> str:
        """A call of a capability in its LaTeX form, as ``J_{0}\\left(x\\right)``, in
        brackets before a power where it is written with scripts; anything else as SymPy's
        printer, with this one's methods, prints it."""
        book = _book(expr, "latex", self._print)
        if book is None:
            return str(super()._print(expr, **settings))
        shown, scripted = book
        power = settings.get("exp")
        if power is None:
            return shown
        return f"\\left({shown}\\right)^{{{power}}}" if scripted else f"{shown}^{{{power}}}"

    def _print_atan2(self, expr: Any, exp: Any = None) -> str:
        arguments = ", ".join(self._print(a) for a in expr.args)
        shown = f"\\operatorname{{atan2}}\\left({arguments}\\right)"
        return f"{shown}^{{{exp}}}" if exp is not None else shown

    def _print_Derivative(self, expr: Any) -> str:
        lagrange = _lagrange(expr)
        if lagrange is not None:
            function, order = lagrange
            shown = self._print(function)
            return str(shown).replace("\\left(", "'" * order + "\\left(", 1)
        return str(super()._print_Derivative(expr))

    def _print_list(self, expr: Any) -> str:
        return "[" + ", ".join(self._print(item) for item in expr) + "]"

    def _print_tuple(self, expr: Any) -> str:
        if len(expr) == 1:
            return f"\\left({self._print(expr[0])},\\right)"
        return "\\left(" + ", ".join(self._print(item) for item in expr) + "\\right)"

    _print_Tuple = _print_tuple

    def _print_str(self, expr: str) -> str:
        return "\\text{" + _latex_text(expr) + "}"

    def _print_NDimArray(self, expr: Any) -> str:
        if expr.rank() == 0:
            return str(self._print(expr[()]))
        if expr.rank() == 1:
            return str(self._print(sympy.Matrix([expr.tolist()])))
        if expr.rank() == 2:
            return str(self._print(sympy.Matrix(expr.tolist())))
        return str(self._print(expr.tolist()))

    _print_ImmutableDenseNDimArray = _print_NDimArray
    _print_MutableDenseNDimArray = _print_NDimArray

    def _print_Relational(self, expr: Any) -> str:
        operators = {"==": "=", "!=": "\\neq", "<": "<", ">": ">", "<=": "\\leq", ">=": "\\geq"}
        return f"{self._print(expr.lhs)} {operators[expr.rel_op]} {self._print(expr.rhs)}"


def _latex_text(text: str) -> str:
    """Text for ``\\text{...}``, its special characters escaped."""
    escapes = {"\\": "\\textbackslash{}", "{": "\\{", "}": "\\}", "$": "\\$", "&": "\\&",
               "#": "\\#", "%": "\\%", "_": "\\_", "^": "\\^{}", "~": "\\~{}"}  # fmt: skip
    return "".join(escapes.get(character, character) for character in text)


# -- scalars ------------------------------------


def _scalar(value: Any, form: Form) -> str:
    """A SymPy value, or a Python number, in one form."""
    if isinstance(value, bool):
        value = sympy.true if value else sympy.false
    if form == "latex":
        return str(_LatexPrinter().doprint(value))
    if form == "unicode":
        return str(_PlainPrinter(unicode=True).doprint(value))
    shown = str(_PlainPrinter(unicode=False).doprint(value))
    if "\\" in shown or not shown.isascii():  # read as LaTeX: names of several letters split
        shown = str(_PlainPrinter(unicode=False, latex_mode=True).doprint(value))
    return shown


def _sympified(value: Any) -> Any:
    """A Python or NumPy number as SymPy's, which the printers know."""
    if isinstance(value, bool):
        return sympy.true if value else sympy.false
    if isinstance(value, int | float | complex):
        return sympy.sympify(value)
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "shape", None) == ():
        return _sympified(item())
    return value


# -- index notation: names, labels and heads ------------------------------------


def _index_label(name: str, form: Form) -> str:
    from relativisticpy.tensors.indices import latex_label, text_label, unicode_label

    return {"unicode": unicode_label, "text": text_label, "latex": latex_label}[form](name)


def symbol_name(name: str, form: Form) -> str:
    """A symbol's name, such as a coordinate's, in a form: ``theta`` is ``θ``,
    ``\\theta`` or ``\\theta``."""
    if form == "unicode":
        return unicode_name(name)
    if form == "latex":
        return latex_name(name)
    return text_name(name)


def _tensor_name(name: str, form: Form) -> str:
    """A tensor's name before its indices: ``Γ``, ``\\Gamma``; a name with a subscript of
    its own is set apart, as in ``{A_{0}}_{a}``, so that its indices are not a second
    subscript."""
    if form == "unicode":
        return unicode_name(name)
    if form == "latex":
        shown = latex_name(name)
        return f"{{{shown}}}" if "_" in shown else shown
    shown = text_name(name)
    return f"{shown}{{}}" if "_" in shown else shown


def component_label(tensor: Any, axis: int, position: int, form: Form) -> str:
    """The name of the component at ``position`` along an axis of a tensor, in a form:
    its coordinate (``t``, ``θ``), its element of a basis (``a:AAPL`` as the language
    writes it, ``\\text{AAPL}`` in LaTeX) or else its position."""
    space = tensor.spaces[axis] if tensor.spaces is not None else None
    coordinates = tensor.coordinates
    if coordinates and (space is None or space.spacetime) and position < len(coordinates):
        return symbol_name(str(coordinates[position]), form)
    if space is not None and space.basis is not None:
        return _element(space.basis[position], tensor.indices[axis].name, form)
    return str(position)


def _element(element: str, label: str, form: Form) -> str:
    """An element of a basis as a component's label: ``a:AAPL`` as the language writes it,
    ``\\text{AAPL}`` in LaTeX."""
    if form == "latex":
        return f"\\text{{{_latex_text(element)}}}"
    return f"{_index_label(label, form)}:{element}"


def tensor_head(tensor: Any, form: Form, labels: Sequence[str] | None = None) -> str | None:
    """How a tensor is written, ``Γ^a_{bc}``, or one of its components, with ``labels``
    (already in the form) for its free indices: ``Γ^t_{tr}``. A tensor computed from an
    expression is written as that expression, and one with neither a name nor an
    expression as its indices alone; ``None`` for such a tensor's own head."""
    if tensor.source:
        if labels is not None and not _writes(tensor.source, tensor.indices.names):
            # spec(H^i{}_j) has an index k of its own: name its components after it
            whole = _source(tensor.source, tensor.indices.names, form, None)
            opening, closing = ("\\left(", "\\right)") if form == "latex" else ("(", ")")
            return opening + whole + closing + _indices_in(tensor.indices, form, labels)
        return _source(tensor.source, tensor.indices.names, form, labels)
    if tensor.name:
        return _tensor_name(tensor.name, form) + _indices_in(tensor.indices, form, labels)
    if labels is None:
        return None
    return ("{}" if form == "latex" else "") + _indices_in(tensor.indices, form, labels)


def _writes(source: str, labels: Sequence[str]) -> bool:
    """Whether an expression as written has each of the labels as an index."""
    from relativisticpy.errors import RelativisticPyError
    from relativisticpy.language.parser import parse_expression

    try:
        node = parse_expression(source)
    except RelativisticPyError:
        return False
    written = {str(item.label) for item in _index_items(node) if item.label is not None}
    return set(labels) <= written


def _indices_in(indices: Any, form: Form, labels: Sequence[str] | None = None) -> str:
    if form == "unicode":
        return str(indices.unicode(labels))
    if form == "latex":
        return str(indices.latex(labels))
    return str(indices.text(labels))


_COMMAND = re.compile(r"\\([A-Za-z]+)")
_UPRIGHT = re.compile(r"\\(?:operatorname|mathrm|text|textrm|mathit)\{([^{}]*)\}")


def _source(
    source: str, free: Sequence[str], form: Form, labels: Sequence[str] | None = None
) -> str:
    r"""An expression as written, ``\Gamma^{r}_{tb}`` or ``2 R_{\mu\nu}``, in a form; with
    ``labels``, the free indices ``free`` replaced by them, to name one component:
    ``\Gamma^{r}_{tt}``, ``2 R_{tr}``."""
    from relativisticpy.errors import RelativisticPyError
    from relativisticpy.language.parser import parse_expression
    from relativisticpy.tensors.indices import join_labels

    substitution = {}
    if labels is not None and len(labels) == len(free):
        substitution = dict(zip(free, labels, strict=True))
    try:
        node = parse_expression(source)
    except RelativisticPyError:
        node = None
    replacements: list[tuple[int, int, str]] = []
    for item in _index_items(node):
        if item.label is None or item.value is not None:
            continue
        label = str(item.label)
        shown = substitution.get(label, _index_label(label, form))
        replacements.append((item.span.start, item.span.end, shown))
    pieces = []
    done = 0
    replacements.sort()
    for number, (start, end, shown) in enumerate(replacements):
        pieces.append(_converted(source[done:start], form))
        pieces.append(shown)
        done = end
        following = replacements[number + 1] if number + 1 < len(replacements) else None
        if following is not None and following[0] == end:  # neighbours in one group
            pair = join_labels([shown, following[2]], form)
            pieces.append(pair[len(shown) : len(pair) - len(following[2])])
        elif end < len(source) and source[end].isalpha() and re.search(r"\\[A-Za-z]+$", shown):
            pieces.append(" ")
    pieces.append(_converted(source[done:], form))
    shown = "".join(pieces).strip()
    return _unbraced(shown) if form == "unicode" else shown


_SCRIPT = re.compile(r"([_^])\{([^{}]+)\}")


def _unbraced(written: str) -> str:
    """Unicode scripts of one letter without their braces, ``Γ^r_{tb}``, except a letter
    after another unbraced one, which would join it: ``x^a_{b}``."""
    pieces: list[str] = []
    done = 0
    for match in _SCRIPT.finditer(written):
        before = "".join(pieces) + written[done : match.start()]
        content = match.group(2)
        single = len(content) == 1 or (len(content) == 2 and content[1] == "′")
        joins = bool(re.search(r"[_^][A-Za-z0-9]$", before)) and content[0].isascii()
        pieces.append(written[done : match.start()])
        if single and not joins:
            pieces.append(match.group(1) + content)
        else:
            pieces.append(match.group(0))
        done = match.end()
    pieces.append(written[done:])
    return "".join(pieces)


def _index_items(node: Any) -> Iterable[Any]:
    """The index items of a syntax tree, in order."""
    import dataclasses

    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        if type(node).__name__ == "IndexItem":
            yield node
            return
        for field in dataclasses.fields(node):
            yield from _index_items(getattr(node, field.name))
    elif isinstance(node, tuple | list):
        for child in node:
            yield from _index_items(child)


_UNICODE_COMMANDS = {
    "partial": "∂", "nabla": "∇", "cdot": "·", "times": "×", "infty": "∞", "hbar": "ℏ",
    "ell": "ℓ", "odot": "⊙", "pm": "±", "mp": "∓", "neq": "≠", "ne": "≠", "leq": "≤",
    "geq": "≥", "le": "≤", "ge": "≥", "to": "→", "sum": "∑", "prod": "∏", "int": "∫",
    "sqrt": "√",
}  # fmt: skip


def _converted(written: str, form: Form) -> str:
    """Text written in any of the language's notations, outside its indices, in a form:
    Greek letters and the symbols of commands as Unicode, or as commands."""
    from relativisticpy.language.vocabulary import UNICODE_COMMANDS, UNICODE_LETTERS

    if form == "unicode":

        def letter(match: re.Match[str]) -> str:
            name = match.group(1)
            if name in GREEK_LETTERS:
                return GREEK_LETTERS[name]
            if name in ("left", "right", "mathrm", "mathit", "text", "operatorname"):
                return ""
            return _UNICODE_COMMANDS.get(name, match.group(0))

        written = _UPRIGHT.sub(r"\1", written)  # \operatorname{spec} is spec
        shown = _COMMAND.sub(letter, written).replace("\\,", " ").replace("\\;", " ")
        return shown.replace("\\!", "").replace("-", _MINUS).replace("*", "·")
    pieces = []
    for character in written:
        name = UNICODE_LETTERS.get(character) or UNICODE_COMMANDS.get(character)
        if name is not None and not character.isascii():
            pieces.append("\\" + name + " ")
        elif character == _MINUS:
            pieces.append("-")
        elif character in "·×":
            pieces.append(" \\cdot " if form == "latex" else "*")
        else:
            pieces.append(character)
    shown = "".join(pieces)
    return re.sub(r"(\\[A-Za-z]+) (?=[^A-Za-z]|$)", r"\1", shown)


# -- components, grouped by symmetry ------------------------------------


def _same(one: Any, other: Any) -> bool:
    """Whether two components are equal: as written, or after cancelling."""
    if one == other:
        return True
    if not (isinstance(one, sympy.Expr) and isinstance(other, sympy.Expr)):
        return False
    difference = one - other
    if sympy.count_ops(difference) > 400:
        return False
    try:
        return bool(sympy.cancel(sympy.together(difference)) == 0)
    except (sympy.PolynomialError, TypeError, ValueError):
        return False


def _permutations(index: tuple[int, ...], classes: Sequence[Any]) -> Iterable[tuple[int, ...]]:
    """The indices made from ``index`` by exchanging axes of one class: axes in the same
    position, over the same space."""
    from itertools import permutations

    groups: dict[Any, list[int]] = {}
    for axis, kind in enumerate(classes):
        groups.setdefault(kind, []).append(axis)
    movable = [axes for axes in groups.values() if len(axes) > 1]
    if not movable:
        return
    choices = [list(permutations(axes)) for axes in movable]
    from itertools import product as cartesian

    for combination in cartesian(*choices):
        other = list(index)
        for axes, order in zip(movable, combination, strict=True):
            for target, source in zip(axes, order, strict=True):
                other[target] = index[source]
        candidate = tuple(other)
        if candidate != index:
            yield candidate


def component_groups(tensor: Any) -> list[tuple[list[tuple[int, tuple[int, ...]]], Any]]:
    """The non-zero components of a symbolic tensor, those equal, or opposite, by an
    exchange of indices in one place together: ``[([(1, (0, 0, 1)), (1, (0, 1, 0))],
    M/(r(r - 2M)))]`` for :math:`\\Gamma^t{}_{tr} = \\Gamma^t{}_{rt}`."""
    shape = tensor.shape
    entries = {}
    for index in _product(shape):
        value = tensor.components[index]
        if value != 0:
            entries[index] = value
    classes = [
        (index.upper, tensor.spaces[axis] if tensor.spaces is not None else None, shape[axis])
        for axis, index in enumerate(tensor.indices)
    ]
    seen: set[tuple[int, ...]] = set()
    groups = []
    for index, value in entries.items():
        if index in seen:
            continue
        seen.add(index)
        members = [(1, index)]
        for other in _permutations(index, classes):
            if other in seen or other not in entries:
                continue
            if _same(entries[other], value):
                members.append((1, other))
                seen.add(other)
            elif _same(entries[other], -value):
                members.append((-1, other))
                seen.add(other)
        groups.append((members, value))
    return groups


def _product(shape: Sequence[int]) -> Iterable[tuple[int, ...]]:
    from itertools import product as cartesian

    return cartesian(*(range(n) for n in shape))


def _labels_of(tensor: Any, index: tuple[int, ...], form: Form) -> list[str]:
    return [component_label(tensor, axis, i, form) for axis, i in enumerate(index)]


def _component(tensor: Any, index: tuple[int, ...], form: Form) -> str:
    """The head of one component: ``Γ^t_{tr}``."""
    head = tensor_head(tensor, form, _labels_of(tensor, index, form))
    assert head is not None
    return head


def _signed(sign: int, head: str, form: Form) -> str:
    if sign > 0:
        return head
    return {"unicode": _MINUS, "text": "-", "latex": "- "}[form] + head


#: The most components listed as lines, and as rows of LaTeX.
LINES = 200
LATEX_ROWS = 64


def _where(tensor: Any, form: Form) -> str:
    """Where a tensor lives: ``in (t, r, θ, φ)``, or over its spaces."""
    if tensor.spaces is not None and not all(space.spacetime for space in tensor.spaces):
        names = []
        for space in tensor.spaces:
            if space.name not in names:
                names.append(space.name)
        return " over " + (" × " if form == "unicode" else " x ").join(names)
    if tensor.coordinates:
        coordinates = ", ".join(symbol_name(str(c), form) for c in tensor.coordinates)
        return f" in ({coordinates})"
    return ""


def _count(number: int, noun: str) -> str:
    return f"{number:,} {noun}{'' if number == 1 else 's'}"


def _symbolic_tensor(tensor: Any, form: Form) -> str:
    groups = component_groups(tensor)
    head = tensor_head(tensor, form)
    if not groups:
        zero = head if head is not None else _indices_in(tensor.indices, form)
        return f"{zero} = 0"
    rows = []
    for members, value in groups:
        heads = [_signed(sign, _component(tensor, index, form), form) for sign, index in members]
        rows.append((heads, _scalar(value, form)))
    nonzero = sum(len(members) for members, _ in groups)
    if form == "latex":
        shown = [" = ".join(heads) + " &= " + value for heads, value in rows[:LATEX_ROWS]]
        if len(rows) > LATEX_ROWS:
            shown.append(f"&\\vdots \\quad \\text{{{len(rows) - LATEX_ROWS} more}}")
        return "\\begin{aligned}\n" + " \\\\\n".join(shown) + "\n\\end{aligned}"
    title = head if head is not None else "tensor " + _indices_in(tensor.indices, form)
    summary = f"{_count(nonzero, 'component')} non-zero"
    if nonzero != tensor.size:
        summary = f"{nonzero:,} of {tensor.size:,} components non-zero"
    lines = [f"{title}{_where(tensor, form)}: {summary}"]
    lines += [f"  {' = '.join(heads)} = {value}" for heads, value in rows[:LINES]]
    if len(rows) > LINES:
        lines.append(f"  … and {len(rows) - LINES} more: .nonzero() lists every component")
    return "\n".join(lines)


def _tensor(tensor: Any, form: Form) -> str:
    if tensor.rank == 0:
        return _scalar(tensor.components, form)
    if tensor.numerical:
        return _numerical_tensor(tensor, form)
    return _symbolic_tensor(tensor, form)


# -- numerical data ------------------------------------


def _number(value: Any, form: Form, digits_: int = 6) -> str:
    """A number of data, to ``digits_`` significant figures: ``0.126``, ``−1.3e+05``."""
    if isinstance(value, complex) or getattr(value, "imag", False):
        real, imag = float(value.real), float(value.imag)
        sign = "-" if imag < 0 else "+"
        shown = f"{real:.{digits_}g} {sign} {abs(imag):.{digits_}g}i"
    elif isinstance(value, bool):
        shown = str(value)
    elif isinstance(value, int) or (hasattr(value, "dtype") and value.dtype.kind in "iu"):
        shown = str(int(value))
    else:
        shown = f"{float(value):.{digits_}g}"
    if form == "unicode":
        return shown.replace("-", _MINUS)
    return shown


def _backend(components: Any) -> str:
    from relativisticpy.tensors import numerical

    if numerical.is_jax(components):
        return "JAX"
    if numerical.is_sparse(components):
        return "SciPy sparse"
    return "NumPy"


def _dense(components: Any) -> Any:
    from relativisticpy.tensors import numerical

    np = numerical.numpy()
    return np.asarray(numerical.dense(components))


def _statistics(array: Any, form: Form) -> str:
    np = __import__("numpy")
    if array.size == 0:
        return "no components"
    if np.iscomplexobj(array):
        magnitude = np.abs(array)
        return (
            f"|min| {_number(magnitude.min(), form, 4)}, |max| {_number(magnitude.max(), form, 4)}"
        )
    finite = array[np.isfinite(array)] if array.dtype.kind == "f" else array
    if finite.size == 0:
        return "no finite components"
    pieces = [
        f"min {_number(finite.min(), form, 4)}",
        f"max {_number(finite.max(), form, 4)}",
        f"mean {_number(finite.mean(), form, 4)}",
        f"std {_number(finite.std(), form, 4)}",
    ]
    return ", ".join(pieces)


#: Data with at most this many rows and columns is shown whole.
SMALL = 8


def _table(
    array: Any, rows: Sequence[Any] | None, columns: Sequence[Any] | None, form: Form
) -> list[str]:
    """A preview of a vector or a matrix as aligned text, labelled by its bases, its
    corners only where it is large."""
    if array.ndim == 1:
        array = array.reshape(-1, 1)
        columns = None
        vector = True
    else:
        vector = False
    height, width = array.shape
    row_picks = _picks(height)
    column_picks = _picks(width)
    ellipsis = "…" if form == "unicode" else "..."
    cells = [
        [ellipsis if i is None or j is None else _number(array[i, j], form) for j in column_picks]
        for i in row_picks
    ]
    row_labels = [
        ellipsis if i is None else str(rows[i]) if rows is not None else "" for i in row_picks
    ]
    column_labels = [
        ellipsis if j is None else str(columns[j]) if columns is not None else ""
        for j in column_picks
    ]
    widths = [
        max([len(cells[r][c]) for r in range(len(cells))] + [len(column_labels[c])])
        for c in range(len(column_picks))
    ]
    label_width = max((len(label) for label in row_labels), default=0)
    lines = []
    if columns is not None and not vector:
        header = (
            " " * label_width
            + "  "
            + "  ".join(label.rjust(widths[c]) for c, label in enumerate(column_labels))
        )
        lines.append("  " + header.rstrip())
    for r, row in enumerate(cells):
        label = row_labels[r].ljust(label_width)
        body = "  ".join(cell.rjust(widths[c]) for c, cell in enumerate(row))
        lines.append(("  " + (label + "  " if label_width else "") + body).rstrip())
    return lines


def _picks(size: int) -> list[int | None]:
    """The positions a preview shows: all of a small axis, the ends of a large one."""
    if size <= SMALL:
        return list(range(size))
    return [0, 1, 2, None, size - 2, size - 1]


def _spaces_line(tensor: Any, form: Form) -> str | None:
    """What each index ranges over: ``t ∈ days (d1, d2, …); a ∈ assets (AAPL, …)``."""
    if tensor.spaces is None:
        return None
    member = " ∈ " if form == "unicode" else " in "
    groups: dict[Any, list[str]] = {}
    for index, space in zip(tensor.indices, tensor.spaces, strict=True):
        groups.setdefault(space, []).append(_index_label(index.name, form))
    pieces = []
    for space, labels in groups.items():
        described = f"{', '.join(labels)}{member}{space.name}"
        if space.basis is not None:
            shown = ", ".join(space.basis[:6]) + (", …" if len(space.basis) > 6 else "")
            described += f" ({shown})"
        else:
            described += f" ({space.dim:,})"
        pieces.append(described)
    return "; ".join(pieces)


def _numerical_head(tensor: Any, form: Form) -> str:
    from relativisticpy.tensors import numerical

    head = tensor_head(tensor, form) or "tensor " + _indices_in(tensor.indices, form)
    times = " × " if form == "unicode" else " x "
    components = tensor.components
    kind = f"{times.join(str(n) for n in tensor.shape)} {components.dtype}"
    if numerical.is_sparse(components):
        stored = components.nnz
        share = stored / max(1, tensor.size)
        kind += f", sparse ({components.format.upper()}), {stored:,} stored ({share:.1%})"
    elif numerical.is_jax(components):
        kind += " (JAX)"
    return f"{head}: {kind}"


def _numerical_tensor(tensor: Any, form: Form) -> str:
    if form == "latex":
        return _numerical_latex(tensor)
    lines = [_numerical_head(tensor, form)]
    spaces = _spaces_line(tensor, form)
    if spaces:
        lines.append("  " + spaces)
    lines.extend(_preview(tensor, form))
    return "\n".join(lines)


def _preview(tensor: Any, form: Form) -> list[str]:
    from relativisticpy.tensors import numerical

    components = tensor.components
    if numerical.is_sparse(components):
        found = components.tocoo()
        shown = []
        for i, j, value in list(zip(found.row, found.col, found.data, strict=True))[:5]:
            labels = (tensor.labels(0)[int(i)], tensor.labels(1)[int(j)])
            shown.append(f"({labels[0]}, {labels[1]}) {_number(value, form)}")
        more = ", …" if components.nnz > 5 else ""
        return ["  " + ", ".join(shown) + more] if shown else ["  no stored components"]
    array = _dense(components)
    if tensor.rank <= 2:
        rows = tensor.labels(0)
        columns = tensor.labels(1) if tensor.rank == 2 else None
        lines = _table(array, rows, columns, form)
        if array.size > SMALL * SMALL or max(array.shape) > SMALL:
            lines.insert(0, "  " + _statistics(array, form))
        return lines
    flat = array.reshape(-1)
    head = ", ".join(_number(value, form) for value in flat[:6])
    return ["  " + _statistics(array, form), f"  [{head}{', …' if flat.size > 6 else ''}]"]


def _numerical_latex(tensor: Any) -> str:
    import numpy as np

    head = tensor_head(tensor, "latex") or _indices_in(tensor.indices, "latex")
    array = _dense(tensor.components)
    if tensor.rank <= 2 and max(tensor.shape) <= SMALL:
        rows = array.reshape(1, -1) if tensor.rank == 1 else array
        body = " \\\\ ".join(" & ".join(_latex_number(v) for v in row) for row in rows)
        return f"{head} = \\begin{{bmatrix}} {body} \\end{{bmatrix}}"
    field = "\\mathbb{C}" if np.iscomplexobj(array) else "\\mathbb{R}"
    sizes = " \\times ".join(str(n) for n in tensor.shape)
    shown = f"{head} \\in {field}^{{{sizes}}}"
    if array.size and not np.iscomplexobj(array) and np.isfinite(array).all():
        shown += (
            f",\\quad (\\min {_latex_number(array.min(), 3)},\\ \\max "
            f"{_latex_number(array.max(), 3)},\\ \\text{{mean }} {_latex_number(array.mean(), 3)})"
        )
    return shown


def _latex_number(value: Any, digits_: int = 6) -> str:
    shown = _number(value, "text", digits_)
    if "e" in shown and "i" not in shown:
        mantissa, exponent = shown.split("e")
        return f"{mantissa} \\times 10^{{{int(exponent)}}}"
    return shown.replace("i", "\\,i") if shown.endswith("i") else shown


def _array(value: Any, form: Form) -> str:
    """Numerical data that is not a tensor: a NumPy, JAX or SciPy array, or pandas data."""
    module = type(value).__module__ or ""
    rows: list[Any] | None = None
    columns: list[Any] | None = None
    if module.startswith("pandas"):
        array = value.to_numpy()
        rows = list(value.index)
        columns = list(value.columns) if value.ndim == 2 else None
    else:
        array = _dense(value)
    if getattr(array, "shape", None) == ():
        return _scalar(sympy.sympify(array.item()), form)
    if form == "latex":
        if array.ndim <= 2 and max(array.shape, default=0) <= SMALL:
            matrix = array.reshape(1, -1) if array.ndim == 1 else array
            body = " \\\\ ".join(" & ".join(_latex_number(v) for v in row) for row in matrix)
            return f"\\begin{{bmatrix}} {body} \\end{{bmatrix}}"
        sizes = " \\times ".join(str(n) for n in array.shape)
        return f"\\text{{array of }} {sizes}"
    times = " × " if form == "unicode" else " x "
    head = f"{times.join(str(n) for n in array.shape)} {array.dtype} ({_kind_of(value)})"
    lines = [head]
    if array.ndim <= 2:
        if array.size > SMALL * SMALL or max(array.shape) > SMALL:
            lines.append("  " + _statistics(array, form))
        lines.extend(_table(array, rows, columns, form))
    else:
        lines.append("  " + _statistics(array, form))
    return "\n".join(lines)


def _kind_of(value: Any) -> str:
    module = type(value).__module__ or ""
    if module.startswith("pandas"):
        return f"pandas {type(value).__name__}"
    return _backend(value)


def _is_data(value: Any) -> bool:
    from relativisticpy.tensors import numerical

    module = type(value).__module__ or ""
    return numerical.is_numerical(value) or module.startswith("pandas")


def _is_tensor(value: Any) -> bool:
    """Whether a value is a :class:`~relativisticpy.tensors.Tensor`, by what it has."""
    return all(hasattr(value, name) for name in ("components", "indices", "labels", "spaces"))


def tensor_form(tensor: Any, form: Form) -> str:
    """A tensor, or a metric, in a form: what ``Tensor._display_unicode`` and the like
    give."""
    if callable(getattr(tensor, "line_element_text", None)):
        return _metric(tensor, form)
    return _tensor(tensor, form)


def tensor_data(tensor: Any) -> dict[str, Any]:
    """A tensor, or a metric, as data."""
    described = _tensor_data(tensor)
    if callable(getattr(tensor, "line_element_text", None)):
        described["kind"] = "metric"
        forms: tuple[Form, ...] = ("text", "unicode", "latex")
        described["line_element"] = {form: line_element(tensor, form) for form in forms}
    return described


def equation_form(equation: Any, form: Form) -> str:
    """An equation of tensors in a form."""
    return _equation(equation, form)


def equation_data(equation: Any) -> dict[str, Any]:
    """An equation of tensors as data."""
    return _equation_data(equation)


def space_form(space: Any, form: Form) -> str:
    """A space in a form."""
    return _space(space, form)


def space_data(space: Any) -> dict[str, Any]:
    """A space as data."""
    return {"kind": "space", "name": space.name, "dim": space.dim,
            "basis": list(space.basis) if space.basis is not None else None,
            "field": space.field, "spacetime": space.spacetime,
            "inferred": getattr(space, "inferred", False)}  # fmt: skip


def _html_table(tensor_or_array: Any, header: str) -> str:
    """Numerical data as an HTML table, labelled by its bases, its corners only."""
    import html as escape

    if _is_tensor(tensor_or_array):
        array = _dense(tensor_or_array.components)
        rows = tensor_or_array.labels(0) if array.ndim >= 1 else None
        columns = tensor_or_array.labels(1) if array.ndim == 2 else None
    else:
        module = type(tensor_or_array).__module__ or ""
        if module.startswith("pandas"):
            array = tensor_or_array.to_numpy()
            rows = list(tensor_or_array.index)
            columns = list(tensor_or_array.columns) if tensor_or_array.ndim == 2 else None
        else:
            array = _dense(tensor_or_array)
            rows = columns = None
    parts = [f"<div><pre>{escape.escape(header)}</pre>"]
    if array.ndim in (1, 2):
        matrix = array.reshape(-1, 1) if array.ndim == 1 else array
        row_picks, column_picks = _picks(matrix.shape[0]), _picks(matrix.shape[1])
        parts.append("<table>")
        if columns is not None:
            cells = "".join(
                f"<th>{'…' if j is None else escape.escape(str(columns[j]))}</th>"
                for j in column_picks
            )
            parts.append(f"<thead><tr><th></th>{cells}</tr></thead>")
        parts.append("<tbody>")
        for i in row_picks:
            label = "…" if i is None else escape.escape(str(rows[i])) if rows is not None else ""
            cells = "".join(
                "<td>…</td>"
                if i is None or j is None
                else f"<td>{_number(matrix[i, j], 'text')}</td>"
                for j in column_picks
            )
            parts.append(f"<tr><th>{label}</th>{cells}</tr>")
        parts.append("</tbody></table>")
    parts.append("</div>")
    return "".join(parts)


# -- spaces, metrics, geodesics ------------------------------------


def _space(space: Any, form: Form) -> str:
    if form == "latex":
        name = _latex_text(space.name)
        field = "\\mathbb{C}" if space.field == "complex" else "\\mathbb{R}"
        return f"\\text{{{name}}} \\cong {field}^{{{space.dim}}}"
    field = (
        ("ℂ" if space.field == "complex" else "ℝ")
        if form == "unicode"
        else ("C" if space.field == "complex" else "R")
    )
    power = superscript(str(space.dim)) if form == "unicode" else f"^{space.dim}"
    described = f"space {space.name}: {field}{power}"
    if space.basis is not None:
        shown = ", ".join(space.basis[:8]) + (", …" if len(space.basis) > 8 else "")
        described += f", basis ({shown})"
    if space.spacetime:
        described += ", spacetime"
    if getattr(space, "inferred", False):
        described += ", inferred from data"
    return described


def _metric(metric: Any, form: Form) -> str:
    """A metric by its line element: ``g_{μν} in (t, r, θ, φ): ds² = …``; in LaTeX the
    line element alone, which the language reads back as the metric."""
    if form == "latex":
        return f"ds^{{2}} = {line_element(metric, form)}"
    head = tensor_head(metric, form) or "g"
    square = "ds²" if form == "unicode" else "ds^2"
    return f"{head}{_where(metric, form)}: {square} = {line_element(metric, form)}"


def _negative(coefficient: Any) -> bool:
    """Whether a coefficient reads best with its sign taken out: ``−(1 − 2M/r)``."""
    if coefficient.could_extract_minus_sign():
        return True
    if coefficient.is_Add:
        constant = coefficient.as_coeff_Add()[0]
        return bool(constant.is_negative)
    return False


def line_element(metric: Any, form: Form = "unicode") -> str:
    r"""A metric's line element as a paper writes it, a term for each pair of coordinates
    in their order, the differentials last: ``−(1 − 2M/r) dt² + r dr²/(r − 2M) + …``,
    ``-\left(1 - \frac{2 M}{r}\right) dt^{2} + \ldots``."""
    coordinates = [str(c) for c in metric.coordinates]
    matrix = metric.matrix
    terms: list[tuple[Any, str]] = []
    for i, first in enumerate(coordinates):
        for j in range(i, len(coordinates)):
            coefficient = matrix[i, j] if i == j else 2 * matrix[i, j]
            if coefficient == 0:
                continue
            terms.append((coefficient, _differentials(first, coordinates[j], form)))
    if not terms:
        return "0"
    pieces = []
    for number, (coefficient, differential) in enumerate(terms):
        negative = _negative(coefficient)
        shown = _term(-coefficient if negative else coefficient, differential, form)
        if number == 0:
            pieces.append(
                ({"unicode": _MINUS, "text": "-", "latex": "- "}[form] if negative else "") + shown
            )
        else:
            pieces.append((f" {_MINUS} " if form == "unicode" else " - ") if negative else " + ")
            pieces.append(shown)
    return "".join(pieces)


def _differentials(first: str, second: str, form: Form) -> str:
    """``dt²``, ``dt dr``; ``d\\theta^{2}`` in LaTeX."""

    def one(name: str) -> str:
        return "d" + symbol_name(name, form)

    if first == second:
        if form == "unicode":
            return one(first) + "²"
        return one(first) + ("^{2}" if form == "latex" else "^2")
    joiner = "*" if form == "text" else "\\, " if form == "latex" else " "
    return one(first) + joiner + one(second)


def _term(coefficient: Any, differential: str, form: Form) -> str:
    """A term of a line element: its coefficient, then its differentials, which a
    fraction puts in its numerator: ``r dr²/(r − 2M)``, ``r^{2}\\, d\\theta^{2}``."""
    if coefficient == 1:
        return differential
    numerator, denominator = sympy.fraction(coefficient)
    if form == "latex":
        if denominator != 1:
            top = "" if numerator == 1 else _scalar(numerator, form) + "\\, "
            return f"\\frac{{{top}{differential}}}{{{_scalar(denominator, form)}}}"
        shown = _scalar(numerator, form)
        if numerator.is_Add:
            shown = f"\\left({shown}\\right)"
        return f"{shown}\\, {differential}"
    joiner = "*" if form == "text" else " "
    top = _scalar(numerator, form)
    if numerator.is_Add:
        top = f"({top})"
    top = differential if numerator == 1 else f"{top}{joiner}{differential}"
    if denominator == 1:
        return top
    bottom = _scalar(denominator, form)
    if denominator.is_Add or denominator.is_Mul:
        bottom = f"({bottom})"
    return f"{top}/{bottom}"


# -- equations of tensors ------------------------------------


def _equation_sides(source: str | None) -> tuple[str, str] | None:
    """The two sides of an equation as written: ``("R_{\\mu\\nu}", "g_{\\mu\\nu}")``."""
    if not source:
        return None
    from relativisticpy.errors import RelativisticPyError
    from relativisticpy.language import Source, parse
    from relativisticpy.language import syntax as ast

    try:
        program = parse(Source(source))
    except RelativisticPyError:
        return None
    if len(program.statements) != 1:
        return None
    statement = program.statements[0]
    if isinstance(statement, ast.Evaluate) and isinstance(statement.expr, ast.Equation):
        lhs, rhs = statement.expr.lhs, statement.expr.rhs
    elif isinstance(statement, ast.Define) and len(statement.targets) == 1:
        lhs, rhs = statement.targets[0], statement.value
    else:
        return None
    return lhs.span.text, rhs.span.text


def _side(written: str, free: Sequence[str], form: Form, labels: Sequence[str] | None) -> str:
    return _source(written, free, form, labels)


def _constant(written: str) -> bool:
    """Whether a side, as written, has no indices: ``0``, ``\\Lambda``."""
    return "_" not in written and "^" not in written


def _free(equation: Any, sides: tuple[str, str] | None) -> tuple[str, ...]:
    """The free indices of an equation: its own, or else those its left side is written
    with, each once."""
    if equation.indices is not None:
        return tuple(equation.indices.names)
    if sides is None:
        return ()
    from relativisticpy.errors import RelativisticPyError
    from relativisticpy.language.parser import parse_expression

    try:
        node = parse_expression(sides[0])
    except RelativisticPyError:
        return ()
    labels = [str(i.label) for i in _index_items(node) if i.label is not None and i.value is None]
    return tuple(label for label in dict.fromkeys(labels) if labels.count(label) == 1)


def _equation_title(equation: Any, sides: tuple[str, str] | None, form: Form) -> str:
    free = _free(equation, sides)
    if sides is not None:
        return " = ".join(_side(side, free, form, None) for side in sides)
    word = "\\text{equation}" if form == "latex" else "equation"
    if equation.indices is not None:
        joiner = "\\ {}" if form == "latex" else " "
        return word + joiner + _indices_in(equation.indices, form)
    return word


def _equation_label(equation: Any, index: tuple[int, ...], form: Form) -> list[str]:
    labels = []
    for axis, position in enumerate(index):
        name = equation.labels(axis)[position]
        space = equation.spaces[axis] if equation.spaces is not None else None
        if isinstance(name, str) and space is not None and not space.spacetime:
            label = equation.indices[axis].name if equation.indices is not None else "a"
            labels.append(_element(name, label, form))
        elif isinstance(name, str):
            labels.append(symbol_name(name, form))
        else:
            labels.append(str(name))
    return labels


def _failure_line(
    equation: Any, failure: Any, sides: tuple[str, str] | None, form: Form, numbers: bool
) -> str:
    """One component, or components equal by symmetry, where the sides differ:
    ``R_{tt} = 0 but g_{tt} = 2M/r − 1``."""
    free = _free(equation, sides)

    def value(side: Any) -> str:
        if numbers:
            return _number(side, form, 5) if form != "latex" else _latex_number(side, 5)
        return _scalar(side, form)

    lhs_value, rhs_value = value(failure.lhs), value(failure.rhs)
    if sides is None:
        labels = _equation_label(equation, failure.index, form)
        where = (
            ("{}" if form == "latex" else "") + _indices_in(equation.indices, form, labels)
            if equation.indices is not None
            else "(" + ", ".join(labels) + ")"
        )
        relation = {"unicode": "≠", "text": "!=", "latex": "\\neq"}[form]
        if failure.differs is not True:
            relation = {"unicode": "≟", "text": "=?", "latex": "\\stackrel{?}{=}"}[form]
        if form == "latex":
            return f"{where} &: {lhs_value} {relation} {rhs_value}"
        return f"{where}: {lhs_value} {relation} {rhs_value}"

    def heads(written: str) -> str:
        shown = []
        for sign, index in failure.members:
            labels = _equation_label(equation, index, form)
            shown.append(_signed(sign, _side(written, free, form, labels), form))
        return " = ".join(shown)

    lhs_written, rhs_written = sides
    left = None if _constant(lhs_written) else heads(lhs_written)
    right = None if _constant(rhs_written) else heads(rhs_written)
    if form == "latex":
        relation = "\\neq" if failure.differs is True else "\\stackrel{?}{=}"
        if left is None:
            return f"{lhs_value} &{relation} {right} = {rhs_value}"
        if right is None:
            return f"{left} &= {lhs_value} {relation} {rhs_value}"
        return f"{left} &= {lhs_value} {relation} {right} = {rhs_value}"
    if failure.differs is not True:
        left_part = f"{left} = {lhs_value}" if left else lhs_value
        right_part = f"{right} = {rhs_value}" if right else rhs_value
        return f"{left_part} and {right_part}: not shown equal"
    if left is None:
        return f"{right} = {rhs_value}, not {lhs_value}"
    if right is None:
        return f"{left} = {lhs_value}, not {rhs_value}"
    return f"{left} = {lhs_value} but {right} = {rhs_value}"


#: The most components where an equation fails that it lists.
FAILURES = 20


def _equation(equation: Any, form: Form) -> str:
    analysis = equation.analysis()
    sides = _equation_sides(equation.source)
    title = _equation_title(equation, sides, form)
    total = analysis.total
    if analysis.holds is True:
        verdict = (
            ("holds" if total == 1 else f"holds in all {total:,} components")
            if equation.rank
            else "holds"
        )
        if form == "latex":
            return f"{title} \\tag*{{{verdict}}}"
        return f"{title}: {verdict}"
    numbers = equation.numerical
    if analysis.differing:
        verdict = f"differs in {analysis.differing:,} of {total:,} components"
        undecided = analysis.failing - analysis.differing
        if undecided:
            verdict += f", and {undecided:,} {'is' if undecided == 1 else 'are'} not decided"
    else:
        verdict = (
            f"not decided in {analysis.failing:,} of {total:,} components: SymPy could not "
            "show the sides equal, nor numbers that they differ"
        )
    where = ""
    if numbers and analysis.largest is not None:
        index, size = analysis.largest
        labels = _equation_label(equation, index, form)
        if sides is not None and equation.indices is not None and not _constant(sides[0]):
            where = _side(sides[0], equation.indices.names, form, labels)
        elif equation.indices is not None:
            where = ("{}" if form == "latex" else "") + _indices_in(equation.indices, form, labels)
        else:
            where = "(" + ", ".join(labels) + ")"
        shown = _number(size, form, 3) if form != "latex" else _latex_number(size, 3)
        verdict += f"; largest difference {shown}, at "
    failures = analysis.failures[:FAILURES]
    lines = [_failure_line(equation, failure, sides, form, numbers) for failure in failures]
    more = analysis.failing - len(failures)
    if form == "latex":
        said = f"\\text{{{_latex_text(verdict)}}}{where}\\text{{:}}"
        rows = [f"&{title}\\ {said}", *lines]
        if more > 0:
            rows.append(f"&\\vdots \\quad \\text{{{more:,} more}}")
        return "\\begin{aligned}\n" + " \\\\\n".join(rows) + "\n\\end{aligned}"
    result = [f"{title}: {verdict}{where}"] + [f"  {line}" for line in lines]
    if more > 0:
        result.append(f"  … and {more:,} more: .failures() lists them all")
    return "\n".join(result)


def _equation_data(equation: Any) -> dict[str, Any]:
    analysis = equation.analysis()
    sides = _equation_sides(equation.source)
    described: dict[str, Any] = {
        "kind": "tensor equation",
        "holds": analysis.holds,
        "lhs": sides[0] if sides else None,
        "rhs": sides[1] if sides else None,
        "indices": equation.indices.text() if equation.indices is not None else None,
        "shape": list(equation.shape),
        "components": analysis.total,
        "failing": analysis.failing,
        "differing": analysis.differing,
    }
    numbers = equation.numerical
    failures = []
    for failure in analysis.failures[:FAILURES]:
        labels = [_jsonable(equation.labels(axis)[i]) for axis, i in enumerate(failure.index)]
        failures.append(
            {
                "component": labels,
                "lhs": _jsonable(failure.lhs) if numbers else _scalar(failure.lhs, "text"),
                "rhs": _jsonable(failure.rhs) if numbers else _scalar(failure.rhs, "text"),
                "differs": failure.differs,
            }
        )
    described["failures"] = failures
    if analysis.largest is not None:
        index, size = analysis.largest
        described["largest"] = {"component": list(index), "difference": size}
    return described


# -- the four forms ------------------------------------


def _hook(value: Any, form: str) -> Callable[[], Any] | None:
    """A value's own way of showing itself, ``_display_unicode`` and the like, for the
    results other modules define, such as the report of a check."""
    method = getattr(value, f"_display_{form}", None)
    return method if callable(method) else None


def _render(value: Any, form: Form) -> str:
    if value is None:
        return {"unicode": "None", "text": "None", "latex": "\\text{None}"}[form]
    if isinstance(value, str):
        if form == "latex":
            return "\\text{" + _latex_text(value) + "}"
        return str(_PlainPrinter(unicode=form == "unicode").doprint(value))
    try:
        hook = _hook(value, form)
        special = str(hook()) if hook is not None else _special(value, form)
    except Exception as error:  # a failure of RelativisticPy's own: show what is known
        name = type(value).__name__
        reason = f"{type(error).__name__}: {error}"
        if form == "latex":
            return f"\\text{{{_latex_text(f'{name} (not shown: {reason})')}}}"
        return f"<{name}: internal error while showing it, {reason}>"
    if special is not None:
        return special
    if _is_data(value) and getattr(value, "shape", ()) != ():
        return _array(value, form)
    value = _sympified(value)
    if isinstance(value, list | tuple):
        return _container(value, form)
    if isinstance(value, dict):
        return _mapping(value, form)
    if isinstance(value, sympy.Basic | sympy.MatrixBase | sympy.NDimArray):
        if isinstance(value, sympy.Integer) and form != "latex" and _long_integer(int(value)):
            return _scalar(value, form) + f" (an integer of {digits(int(value)):,} digits)"
        return _scalar(value, form)
    if form == "latex":
        method = getattr(value, "latex", None)
        if callable(method):
            return str(method())
        return "\\text{" + _latex_text(str(value)) + "}"
    return str(value)


def _special(value: Any, form: Form) -> str | None:
    """The forms of RelativisticPy's own values that do not show themselves: tensors,
    equations and spaces, where they come without their hooks."""
    from relativisticpy.tensors import Space, Tensor
    from relativisticpy.tensors.equation import TensorEquation

    if isinstance(value, Tensor):
        return tensor_form(value, form)
    if isinstance(value, TensorEquation):
        return _equation(value, form)
    if isinstance(value, Space):
        return _space(value, form)
    return None


def _container(value: Sequence[Any], form: Form) -> str:
    items = [_render(item, form) for item in value]
    if isinstance(value, tuple):
        inner = ", ".join(items) + ("," if len(items) == 1 else "")
        return f"\\left({inner}\\right)" if form == "latex" else f"({inner})"
    return "[" + ", ".join(items) + "]"


def _mapping(value: Mapping[Any, Any], form: Form) -> str:
    items = [f"{_render(key, form)}: {_render(item, form)}" for key, item in value.items()]
    if form == "latex":
        return "\\left\\{" + ", ".join(items) + "\\right\\}"
    return "{" + ", ".join(items) + "}"


def unicode(value: Any) -> str:
    """``value`` as mathematics for a terminal: ``48M²/r⁶``.

    >>> import sympy
    >>> theta = sympy.Symbol("theta")
    >>> unicode(-sympy.sin(theta) * sympy.cos(theta))
    '−sin(θ) cos(θ)'
    """
    return _render(value, "unicode")


def text(value: Any) -> str:
    r"""``value`` as plain ASCII text that the language reads back as the same value.

    >>> import sympy
    >>> theta = sympy.Symbol("theta")
    >>> text(-sympy.sin(theta) * sympy.cos(theta))
    '-sin(\\theta)*cos(\\theta)'
    """
    return _render(value, "text")


def latex(value: Any) -> str:
    r"""``value`` typeset in LaTeX, which the language reads back as the same value.

    >>> import sympy
    >>> latex(sympy.Symbol("rho_vac") + sympy.Symbol("M_odot"))
    'M_{\\odot} + \\rho_{\\mathrm{vac}}'
    """
    return _render(value, "latex")


def html(value: Any) -> str:
    """``value`` as HTML, for notebooks: numerical data as a table, with the labels of
    its bases; anything else typeset, for MathJax or KaTeX."""
    if _is_tensor(value) and value.numerical and value.rank > 0:
        header = _numerical_head(value, "unicode")
        spaces = _spaces_line(value, "unicode")
        if spaces:
            header += "\n  " + spaces
        if value.rank > 2 or (value.size > SMALL * SMALL or max(value.shape) > SMALL):
            header += "\n  " + _statistics(_dense(value.components), "unicode")
        return _html_table(value, header)
    if _is_data(value) and getattr(value, "shape", ()) != ():
        return _html_table(value, unicode(value).splitlines()[0])
    import html as escape

    return f"<div>\\[{escape.escape(latex(value))}\\]</div>"


def _tabulated(value: Any) -> bool:
    """Whether :func:`html` shows ``value`` as a table of numbers."""
    if _is_tensor(value):
        return bool(value.numerical and value.rank > 0)
    return _is_data(value) and getattr(value, "shape", ()) != ()


# -- notebooks: the hooks Jupyter and marimo call ------------------------------------


def jupyter_latex(value: Any) -> str:
    r"""``value`` typeset for Jupyter, as its ``_repr_latex_`` gives it: the LaTeX of
    :func:`latex` between dollar signs, in display style, and in display mode where a tag
    needs it.

    >>> jupyter_latex(sympy.Symbol("x") ** 2)
    '$\\displaystyle x^{2}$'
    """
    shown = latex(value)
    if "\\tag" in shown:
        return f"$$ {shown} $$"
    return f"$\\displaystyle {shown}$"


def jupyter_html(value: Any) -> str | None:
    """Numerical data as an HTML table for Jupyter, as its ``_repr_html_`` gives it, from
    :func:`html`; ``None`` for anything else, which Jupyter typesets from its LaTeX."""
    return html(value) if _tabulated(value) else None


def marimo(value: Any) -> Any:
    """``value`` as marimo shows it, as its ``_display_`` gives it: numerical data as the
    table of :func:`html`, and anything else typeset from :func:`latex`, by marimo's own
    KaTeX. Only marimo calls it, so marimo is there to import."""
    import marimo as mo

    if _tabulated(value):
        return mo.Html(html(value))
    return mo.md("\\[" + latex(value) + "\\]")


def data(value: Any) -> dict[str, Any]:
    """``value`` described as JSON-serialisable data: its kind, its three forms and its
    structure.

    >>> data(sympy.Rational(1, 2))["kind"], data(True)["value"]
    ('number', True)
    """
    hook = _hook(value, "data")
    described = dict(hook()) if hook is not None else _describe(value)
    described.setdefault("text", text(value))
    described.setdefault("unicode", unicode(value))
    described.setdefault("latex", latex(value))
    return described


def _describe(value: Any) -> dict[str, Any]:
    """The kind and structure of a value, for :func:`data`."""
    from relativisticpy.tensors import Space, Tensor
    from relativisticpy.tensors.equation import TensorEquation

    if value is None:
        return {"kind": "none"}
    if isinstance(value, str):
        return {"kind": "string", "value": value}
    if isinstance(value, Tensor):
        return tensor_data(value)
    if isinstance(value, TensorEquation):
        return _equation_data(value)
    if isinstance(value, Space):
        return space_data(value)
    if _is_data(value) and getattr(value, "shape", ()) != ():
        return _array_data(value)
    value = _sympified(value)
    if value is sympy.true or value is sympy.false:
        return {"kind": "boolean", "value": value is sympy.true}
    if isinstance(value, list | tuple):
        return {"kind": "list" if isinstance(value, list) else "tuple",
                "items": [data(item) for item in value]}  # fmt: skip
    if isinstance(value, sympy.Number):
        return _number_data(value)
    if isinstance(value, sympy.Basic) and value.is_Relational:
        from relativisticpy.tensors.equation import holds

        return {"kind": "relation", "relation": value.rel_op, "holds": holds(value),
                "lhs": text(value.lhs), "rhs": text(value.rhs)}  # fmt: skip
    if isinstance(value, sympy.Basic):
        symbols = sorted(str(s) for s in getattr(value, "free_symbols", ()))
        return {"kind": "expression", "symbols": symbols}
    return {"kind": type(value).__name__.lower()}


def _number_data(value: Any) -> dict[str, Any]:
    described: dict[str, Any] = {"kind": "number"}
    if value.is_Integer:
        count = digits(int(value))
        described["digits"] = count
        if count <= LONG_INTEGER:
            described["value"] = int(value)
    elif value.is_Float and math.isfinite(float(value)):
        described["value"] = float(value)
    elif value.is_Rational:
        described["value"] = float(value)
        described["exact"] = text(value)
    return described


def _labels_key(labels: Sequence[Any]) -> str:
    return ",".join(str(label) for label in labels)


def _tensor_data(tensor: Any) -> dict[str, Any]:
    described: dict[str, Any] = {
        "kind": "tensor",
        "name": tensor.name,
        "indices": {
            form: _indices_in(tensor.indices, form) for form in ("text", "unicode", "latex")
        },
        "head": {form: tensor_head(tensor, form) for form in ("text", "unicode", "latex")},
        "rank": tensor.rank,
        "shape": list(tensor.shape),
        "coordinates": [str(c) for c in tensor.coordinates],
        "spaces": [space.name for space in tensor.spaces] if tensor.spaces is not None else None,
    }
    if tensor.numerical:
        described.update(_array_data(tensor.components, tensor))
        described["kind"] = "tensor"
        return described
    if tensor.rank == 0:
        described["value"] = {"text": text(tensor.components), "latex": latex(tensor.components)}
        return described
    components = {}
    for members, value in component_groups(tensor):
        shown = {"text": _scalar(value, "text"), "latex": _scalar(value, "latex")}
        for sign, index in members:
            names = [tensor.labels(axis)[i] for axis, i in enumerate(index)]
            if sign > 0:
                components[_labels_key(names)] = shown
            else:
                negated = -value
                components[_labels_key(names)] = {
                    "text": _scalar(negated, "text"),
                    "latex": _scalar(negated, "latex"),
                }
    described["components"] = components
    described["nonzero"] = len(components)
    return described


def _array_data(value: Any, tensor: Any = None) -> dict[str, Any]:
    from relativisticpy.tensors import numerical

    module = type(value).__module__ or ""
    array = value.to_numpy() if module.startswith("pandas") else _dense(value)
    described: dict[str, Any] = {
        "kind": "array",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "backend": _kind_of(value),
    }
    if numerical.is_sparse(value):
        described["stored"] = int(value.nnz)
    rows: Sequence[Any] | None = None
    columns: Sequence[Any] | None = None
    if tensor is not None:
        rows = tensor.labels(0) if tensor.rank >= 1 else None
        columns = tensor.labels(1) if tensor.rank >= 2 else None
    elif module.startswith("pandas"):
        rows = [str(x) for x in value.index]
        columns = [str(x) for x in value.columns] if value.ndim == 2 else None
    preview = array[tuple(slice(0, SMALL) for _ in array.shape)] if array.ndim else array
    described["preview"] = _jsonable(preview.tolist())
    if rows is not None:
        described["rows"] = [_jsonable(r) for r in list(rows)[:SMALL]]
    if columns is not None:
        described["columns"] = [_jsonable(c) for c in list(columns)[:SMALL]]
    if array.size and array.dtype.kind in "iuf":
        described["statistics"] = {
            "min": float(array.min()), "max": float(array.max()),
            "mean": float(array.mean()), "std": float(array.std()),
        }  # fmt: skip
    return described


def _jsonable(value: Any) -> Any:
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    item = getattr(value, "item", None)
    return _jsonable(item()) if callable(item) else str(value)
