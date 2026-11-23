r"""Number theory.

The remainder, ``a \bmod m``, and the arithmetic functions by the letters books give them:
Euler's totient ``\varphi(n)``, the Möbius function ``\mu(n)``, the prime-counting function
``\pi(x)``, the divisor functions ``\sigma_k(n)`` and ``d(n)``, the numbers of prime
factors ``\omega(n)`` and ``\Omega(n)``, and von Mangoldt's ``\Lambda(n)``. Each letter is
the function only where its argument is an integer, so that ``\varphi(x)`` of a real x and
``\mu(t)`` stay a program's own, and π only of a name assumed an integer, so that
``\pi(x)`` stays π times x (design/notation.md, 32 and 33). Exact with SymPy, which keeps a
call of a symbol as it is until the symbol has a value, as the index of a sum gets one.
OEIS is the On-Line Encyclopedia of Integer Sequences; Apostol (1976) is *Introduction to
Analytic Number Theory*; Hardy and Wright (2008) is *An Introduction to the Theory of
Numbers*, 6th edition; Graham, Knuth and Patashnik (1994) is *Concrete Mathematics*.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import sympy

from relativisticpy.errors import EvaluationError
from relativisticpy.integrations import Example, function

_WHOLE = [(1,), (12,), (36,), (97,)]  # positive integers, where the functions are defined


def whole(n: Any, name: str) -> Any:
    """``n`` as a function of integers takes it: a whole number given as a decimal, as data
    gives it, made an integer; a symbol or an expression kept as it is, for a value later;
    and a number that is no positive integer refused, saying so."""
    n = sympy.sympify(n)
    if n.is_Float and float(n).is_integer():
        n = sympy.Integer(int(n))
    if n.is_number and not (n.is_Integer and n > 0):
        raise EvaluationError(
            f"{name} takes positive integers, not {n}",
            hint="it counts, of a whole number above 0, as books define it",
        )
    return n


def elementwise(compute: Callable[..., Any], *arrays: Any) -> Any:
    """A function of integers computed at each number of arrays, which NumPy has none of:
    each as it is, a complex number too, which the function refuses, never its real part."""
    import numpy as np

    each = np.vectorize(lambda *values: float(compute(*values)), otypes=[float])
    return each(*(np.asarray(array) for array in arrays))


@function(
    "mod",
    aliases=["Mod"],
    summary=r"the remainder of a on division by m, `a \bmod m`",
    examples=[
        Example(r"12! \bmod 13", 12, "Hardy and Wright (2008), theorem 80: Wilson's theorem"),
        Example("mod(-7, 3)", 2, "Graham, Knuth and Patashnik (1994), (3.21)"),
    ],
    points=[(17, 5), (-7, 3), (7.5, 2)],
)
class Remainder:
    def exact(self, a: Any, m: Any) -> Any:
        return sympy.Mod(a, m)

    def numeric(self, xp: Any, a: Any, m: Any) -> Any:
        return xp.mod(a, m)


@function(
    "totient",
    latex=(r"\varphi(n)", r"\phi(n)"),
    unicode="φ(n)",
    integers=["n"],
    summary="Euler's totient function",
    examples=[
        Example(r"\varphi(36)", 12, "OEIS A000010"),
        Example("totient(97)", 96, "OEIS A000010: p - 1 of a prime p"),
    ],
    points=_WHOLE,
)
class Totient:
    def exact(self, n: Any) -> Any:
        return sympy.totient(whole(n, "totient"))

    def numeric(self, xp: Any, n: Any) -> Any:
        return elementwise(self.exact, n)


@function(
    "mobius",
    latex=r"\mu(n)",
    unicode="μ(n)",
    integers=["n"],
    summary="the Möbius function",
    examples=[
        Example(r"\mu(30)", -1, "OEIS A008683"),
        Example("mobius(12)", 0, "OEIS A008683"),
    ],
    points=_WHOLE,
)
class Mobius:
    def exact(self, n: Any) -> Any:
        return sympy.mobius(whole(n, "mobius"))

    def numeric(self, xp: Any, n: Any) -> Any:
        return elementwise(self.exact, n)


@function(
    "primepi",
    latex=r"\pi(x)",
    unicode="π(x)",
    integers=["x"],
    summary="the prime-counting function",
    examples=[
        Example("primepi(100)", 25, "OEIS A000720"),
        Example(r"n \in \mathbb{N}" "\n" r"\left. \pi(n) \right|_{n=1000}", 168, "OEIS A000720"),
    ],
    points=[(2,), (10,), (100,), (30.5,)],
)
class PrimeCounting:
    def exact(self, x: Any) -> Any:
        x = sympy.sympify(x)
        if x.is_number and x.is_extended_real is not True:
            raise EvaluationError(f"primepi takes real numbers, not {x}")
        return sympy.primepi(x)

    def numeric(self, xp: Any, x: Any) -> Any:
        return elementwise(self.exact, x)


@function(
    "divisor_sigma",
    latex=r"\sigma_k(n)",
    integers=["n"],
    summary="the sum of the kth powers of the divisors of n",
    examples=[
        Example(r"\sigma_1(28)", 56, "OEIS A000203: 28 is perfect"),
        Example("divisor_sigma(12, 2)", 210, "OEIS A001157"),
    ],
    points=[(1, 1), (12, 0), (28, 1), (36, 2)],
)
class DivisorSigma:
    def exact(self, n: Any, k: Any) -> Any:
        return sympy.divisor_sigma(whole(n, "divisor_sigma"), k)

    def numeric(self, xp: Any, n: Any, k: Any) -> Any:
        return elementwise(self.exact, n, k)


class divisor_count(sympy.Function):  # type: ignore[misc]  # named as its calls print
    """d(n), the number of divisors of n, kept as it is where n is no number."""

    nargs = 1

    @classmethod
    def eval(cls, n: Any) -> Any:
        if n.is_Integer and n > 0:
            return sympy.Integer(sympy.divisor_count(n))
        return None

    def _eval_is_integer(self) -> bool | None:
        return True if self.args[0].is_integer else None

    def _eval_is_positive(self) -> bool | None:
        return True if self.args[0].is_positive and self.args[0].is_integer else None


@function(
    "divisor_count",
    latex="d(n)",
    integers=["n"],
    summary="the number of divisors of n",
    examples=[
        Example("d(28)", 6, "OEIS A000005"),
        Example("divisor_count(36)", 9, "OEIS A000005"),
    ],
    points=_WHOLE,
)
class DivisorCount:
    def exact(self, n: Any) -> Any:
        return divisor_count(whole(n, "divisor_count"))

    def numeric(self, xp: Any, n: Any) -> Any:
        return elementwise(self.exact, n)


@function(
    "primenu",
    latex=r"\omega(n)",
    unicode="ω(n)",
    integers=["n"],
    summary="the number of distinct primes that divide n",
    examples=[
        Example(r"\omega(30)", 3, "OEIS A001221"),
        Example("primenu(1024)", 1, "OEIS A001221"),
    ],
    points=_WHOLE,
)
class DistinctPrimeFactors:
    def exact(self, n: Any) -> Any:
        return sympy.primenu(whole(n, "primenu"))

    def numeric(self, xp: Any, n: Any) -> Any:
        return elementwise(self.exact, n)


@function(
    "primeomega",
    latex=r"\Omega(n)",
    unicode="Ω(n)",
    integers=["n"],
    summary="the number of prime factors of n, with multiplicity",
    examples=[
        Example(r"\Omega(1024)", 10, "OEIS A001222"),
        Example("primeomega(30)", 3, "OEIS A001222"),
    ],
    points=_WHOLE,
)
class PrimeFactors:
    def exact(self, n: Any) -> Any:
        return sympy.primeomega(whole(n, "primeomega"))

    def numeric(self, xp: Any, n: Any) -> Any:
        return elementwise(self.exact, n)


class mangoldt(sympy.Function):  # type: ignore[misc]  # named as its calls print
    """Λ(n), von Mangoldt's function, kept as it is where n is no number."""

    nargs = 1

    @classmethod
    def eval(cls, n: Any) -> Any:
        if not (n.is_Integer and n > 0):
            return None
        factors = sympy.factorint(n)
        if len(factors) != 1:
            return sympy.S.Zero
        (prime,) = factors
        return sympy.log(prime)

    def _eval_is_real(self) -> bool | None:
        return True if self.args[0].is_integer else None


@function(
    "mangoldt",
    latex=r"\Lambda(n)",
    unicode="Λ(n)",
    integers=["n"],
    summary="von Mangoldt's function",
    examples=[
        Example(r"\Lambda(8)", sympy.log(2), "Apostol (1976), section 2.8"),
        Example("mangoldt(6)", 0, "Apostol (1976), section 2.8"),
    ],
    points=_WHOLE,
)
class VonMangoldt:
    def exact(self, n: Any) -> Any:
        return mangoldt(whole(n, "mangoldt"))

    def numeric(self, xp: Any, n: Any) -> Any:
        return elementwise(self.exact, n)
