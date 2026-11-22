r"""Combinatorics.

The Stirling numbers in Karamata's notation, as Knuth writes them, ``{n \brace k}`` and
``{n \brack k}``; the Catalan, Fibonacci, Bell and Bernoulli numbers; the harmonic numbers
``H_n``; the partitions of a number, ``p(n)``; and the multinomial coefficient. A letter is
the number only where its argument is an integer: ``p(n)`` of an integer, and ``B_n`` and
``H_n`` of a name the program assumes one, so that ``H_0`` stays Hubble's constant and
``H_n(x)`` Hermite's polynomial (design/notation.md, 32, 34 and 35). Exact with SymPy.
OEIS is the On-Line Encyclopedia of Integer Sequences; Graham, Knuth and Patashnik (1994)
is *Concrete Mathematics*, 2nd edition.
"""

from __future__ import annotations

from typing import Any

import sympy

from relativisticpy.errors import EvaluationError
from relativisticpy.integrations import Example, function
from relativisticpy.integrations.number_theory import elementwise

n, k = sympy.symbols("n k", integer=True, nonnegative=True)
_COUNTS = [(0,), (1,), (5,), (10,)]  # whole numbers, where the numbers are defined
_PAIRS = [(5, 2), (6, 3), (7, 0), (4, 4)]  # n and k


def counted(value: Any, name: str) -> Any:
    """A count as a function of whole numbers takes it: a whole number given as a decimal,
    as data gives it, made an integer; a symbol or an expression kept as it is, for a value
    later; and a number that is no whole number refused, saying so."""
    value = sympy.sympify(value)
    if value.is_Float and float(value).is_integer():
        value = sympy.Integer(int(value))
    if value.is_number and not (value.is_Integer and value >= 0):
        raise EvaluationError(
            f"{name} takes whole numbers, not {value}",
            hint="it counts, of whole numbers from 0, as books define it",
        )
    return value


class stirling2(sympy.Function):  # type: ignore[misc]  # named as its calls print
    """``{n \\brace k}``, the Stirling number of the second kind, kept as it is where n or k
    is no number."""

    nargs = 2

    @classmethod
    def eval(cls, n: Any, k: Any) -> Any:
        if n.is_Integer and k.is_Integer:
            from sympy.functions.combinatorial.numbers import stirling

            return sympy.Integer(stirling(n, k, kind=2))
        return None

    def _eval_is_integer(self) -> bool | None:
        return True if all(arg.is_integer for arg in self.args) else None

    def _latex(self, printer: Any) -> str:
        n, k = (printer._print(arg) for arg in self.args)
        return f"{{{n} \\brace {k}}}"


class stirling1(sympy.Function):  # type: ignore[misc]  # named as its calls print
    """``{n \\brack k}``, the Stirling number of the first kind, unsigned, kept as it is
    where n or k is no number."""

    nargs = 2

    @classmethod
    def eval(cls, n: Any, k: Any) -> Any:
        if n.is_Integer and k.is_Integer:
            from sympy.functions.combinatorial.numbers import stirling

            return sympy.Integer(stirling(n, k, kind=1))
        return None

    def _eval_is_integer(self) -> bool | None:
        return True if all(arg.is_integer for arg in self.args) else None

    def _latex(self, printer: Any) -> str:
        n, k = (printer._print(arg) for arg in self.args)
        return f"{{{n} \\brack {k}}}"


@function(
    "stirling2",
    summary=r"the Stirling number of the second kind, `{n \brace k}`",
    examples=[
        Example(r"\sum_{k=0}^{5} {5 \brace k}", 52, "OEIS A000110: the Bell number B_5"),
        Example("stirling2(6, 3)", 90, "OEIS A008277"),
    ],
    points=_PAIRS,
)
class StirlingSecondKind:
    def exact(self, n: Any, k: Any) -> Any:
        return stirling2(counted(n, "stirling2"), counted(k, "stirling2"))

    def numeric(self, xp: Any, n: Any, k: Any) -> Any:
        return elementwise(self.exact, n, k)


@function(
    "stirling1",
    summary=r"the Stirling number of the first kind, `{n \brack k}`, unsigned",
    examples=[
        Example(r"\sum_{k=0}^{5} {5 \brack k}", 120, "Graham, Knuth and Patashnik (1994), "
                "(6.9): the sum is n!"),
        Example("stirling1(5, 2)", 50, "OEIS A132393"),
    ],
    points=_PAIRS,
)  # fmt: skip
class StirlingFirstKind:
    def exact(self, n: Any, k: Any) -> Any:
        return stirling1(counted(n, "stirling1"), counted(k, "stirling1"))

    def numeric(self, xp: Any, n: Any, k: Any) -> Any:
        return elementwise(self.exact, n, k)


@function(
    "catalan",
    integers=["n"],
    summary="the Catalan number",
    examples=[
        Example("catalan(10)", 16796, "OEIS A000108"),
        Example("catalan(4)", 14, "OEIS A000108"),
    ],
    points=_COUNTS,
)
class Catalan:
    def exact(self, n: Any) -> Any:
        return sympy.catalan(counted(n, "catalan"))

    def numeric(self, xp: Any, n: Any) -> Any:
        return elementwise(self.exact, n)


@function(
    "fibonacci",
    integers=["n"],
    summary="the Fibonacci number, from F_0 = 0 and F_1 = 1",
    examples=[
        Example("fibonacci(10)", 55, "OEIS A000045"),
        Example("fibonacci(20)", 6765, "OEIS A000045"),
    ],
    points=_COUNTS,
)
class Fibonacci:
    def exact(self, n: Any) -> Any:
        return sympy.fibonacci(counted(n, "fibonacci"))

    def numeric(self, xp: Any, n: Any) -> Any:
        return elementwise(self.exact, n)


@function(
    "bell",
    integers=["n"],
    summary="the Bell number",
    examples=[
        Example("bell(5)", 52, "OEIS A000110"),
        Example("bell(10)", 115975, "OEIS A000110"),
    ],
    points=_COUNTS,
)
class Bell:
    def exact(self, n: Any) -> Any:
        return sympy.bell(counted(n, "bell"))

    def numeric(self, xp: Any, n: Any) -> Any:
        return elementwise(self.exact, n)


@function(
    "bernoulli",
    latex="B_n",
    integers=["n"],
    summary="the Bernoulli number; B_1 = 1/2, where DLMF takes -1/2",
    examples=[
        Example("bernoulli(10)", sympy.Rational(5, 66), "DLMF 24.2, Table 24.2.1"),
        Example(r"n \in \mathbb{N}" "\n" r"\left. B_{2n} \right|_{n=3}", sympy.Rational(1, 42),
                "DLMF 24.2, Table 24.2.1"),
    ],
    points=_COUNTS,
)  # fmt: skip
class Bernoulli:
    def exact(self, n: Any) -> Any:
        return sympy.bernoulli(counted(n, "bernoulli"))

    def numeric(self, xp: Any, n: Any) -> Any:
        return elementwise(self.exact, n)


@function(
    "harmonic",
    latex="H_n",
    integers=["n"],
    summary="the harmonic number; 1 + 1/2 + ... + 1/n",
    examples=[
        Example("harmonic(4)", sympy.Rational(25, 12), "OEIS A001008 and A002805"),
        Example(r"n \in \mathbb{N}" "\n" "H_{n+1} - H_n", 1 / (n + 1),
                "Graham, Knuth and Patashnik (1994), (6.11): H_n = 1 + 1/2 + ... + 1/n"),
    ],
    points=_COUNTS,
)  # fmt: skip
class Harmonic:
    def exact(self, n: Any) -> Any:
        n = counted(n, "harmonic")
        value = sympy.harmonic(n)
        offset = n.as_coeff_Add()[0] if isinstance(n, sympy.Add) else sympy.S.Zero
        return sympy.expand_func(value) if offset.is_Integer and offset != 0 else value

    def numeric(self, xp: Any, n: Any) -> Any:
        return elementwise(self.exact, n)


@function(
    "partition",
    latex="p(n)",
    integers=["n"],
    summary="the number of partitions of n",
    examples=[
        Example("p(5)", 7, "OEIS A000041"),
        Example("partition(100)", 190569292, "OEIS A000041"),
    ],
    points=_COUNTS,
)
class Partition:
    def exact(self, n: Any) -> Any:
        return sympy.partition(counted(n, "partition"))

    def numeric(self, xp: Any, n: Any) -> Any:
        return elementwise(self.exact, n)


@function(
    "multinomial",
    summary="the multinomial coefficient; (k_1 + ... + k_m)!/(k_1! ... k_m!)",
    examples=[
        Example("multinomial(2, 3, 4)", 1260, "sympy.multinomial_coefficients(3, 9)[(2, 3, 4)]"),
        Example("multinomial(1, 1, 1, 1)", 24, "the 4! orderings of four objects"),
    ],
    points=[(2, 3), (1, 1), (0, 4)],
    lifts=False,
)
class Multinomial:
    def exact(self, k: Any, *rest: Any) -> Any:
        counts = [counted(count, "multinomial") for count in (k, *rest)]
        whole = sympy.factorial(sympy.Add(*counts))
        return whole / sympy.Mul(*(sympy.factorial(count) for count in counts))
