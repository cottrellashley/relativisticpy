r"""What an integration declares: capabilities, each of a kind, and the book names no
integration provides yet.

A capability is a class whose methods compute it, decorated with its kind, which says how
it is written, what it means and how it is checked::

    @function(
        "besselj",
        latex=r"J_{\nu}(x)",
        summary=r"the Bessel function of the first kind, of order \nu",
        requires="scipy",
        examples=[
            Example("J_0(0)", 1, "DLMF 10.8.1"),
            Example(r"J_{1/2}(\pi/2)", 2 / sympy.pi, "DLMF 10.16.1"),
        ],
    )
    class BesselJ:
        def exact(self, nu, x):
            return sympy.besselj(nu, x)

        def numeric(self, xp, nu, x):
            from scipy.special import jv

            return jv(nu, x)

``exact`` computes with SymPy, and its signature is the declaration of the parameters:
how many arguments a call takes, and the names its written forms use. ``numeric``, where
there is one, computes with arrays: ``xp`` is the array module of the data, NumPy's or
JAX's. A capability is checked by the conformance suite, ``tests/integrations``.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar, TypeVar

_Kind = TypeVar("_Kind", bound="Capability")


@dataclass(frozen=True)
class Example:
    """A program that uses a capability, and the value of its last output, known from
    outside the integration: ``source`` says where it comes from, a closed form's place in a
    book, as ``"DLMF 10.8.1"``, or the library called directly, as
    ``"scipy.special.jv(0, 1.5)"``. Numbers agree to ``tolerance``, relatively. ``given``
    makes the Python objects the program reads by name, as ``\\texttt{prices}``, for an
    object integration's example."""

    program: str
    answer: Any
    source: str
    tolerance: float = 1e-12
    given: Callable[[], Mapping[str, Any]] | None = None


def near(found: complex, expected: complex, tolerance: float) -> bool:
    """Whether a number is the one expected, to ``tolerance``, as every check compares
    numbers: relative to the size of the one expected, or to 1 where that is smaller, so
    that 1e-15 is near 0. A number that is not finite is near none."""
    if not (math.isfinite(found.real) and math.isfinite(found.imag)):
        return False
    return abs(found - expected) <= tolerance * max(1.0, abs(expected))


@dataclass(frozen=True, eq=False)
class Capability:
    """What the language can write and compute, as an integration declares it: called by
    ``name`` and its ``aliases``, written in books in its ``latex`` and ``unicode`` forms,
    and computed by the methods of ``implementation``, an instance of the class the
    declaration decorates. ``requires`` names the library it needs beyond SymPy, as pip
    installs it; ``points`` are arguments it is checked at."""

    #: The kind of capability, as the conformance suite and the reference name it.
    kind: ClassVar[str] = "capability"
    #: What a call takes, as the error that refuses another kind of argument says it.
    takes: ClassVar[str] = "its arguments"

    implementation: Any
    name: str
    summary: str
    examples: tuple[Example, ...]
    aliases: tuple[str, ...] = ()
    latex: tuple[str, ...] = ()
    unicode: tuple[str, ...] = ()
    requires: str | None = None
    points: tuple[tuple[Any, ...], ...] = ()
    #: Whether a call gives a scalar where its arguments are scalars.
    scalar: bool = False
    #: Whether it applies to each component of a tensor given as its first argument; of a
    #: function of a matrix, whether it acts on a map written with its indices, over the
    #: same spaces.
    lifts: bool = False
    #: Whether it reads text in quotes as names, as ``symbols('x y')`` does, never as Python.
    takes_names: bool = False
    #: The parameters a form written with a letter reads as it only where they are
    #: integers, as φ(n) is Euler's totient only where n is one (design/notation.md, 32).
    integers: tuple[str, ...] = ()
    #: Whether its value is a form of its argument, as factor's is the factorised one: shown
    #: as it is computed, never tidied into another, and an example's answer is that form,
    #: not any expression equal to it.
    gives_form: bool = False

    @property
    def module(self) -> str:
        """The module that declares it."""
        return type(self.implementation).__module__

    @property
    def names(self) -> tuple[str, ...]:
        """Every name it is called by: its own, then its aliases."""
        return (self.name, *self.aliases)

    @property
    def brief(self) -> str:
        """What it is, as a note says it: its summary up to a semicolon, without the marks
        of code the reference shows."""
        return self.summary.split(";")[0].replace("`", "").strip()

    @property
    def exact(self) -> Callable[..., Any]:
        """What computes it with SymPy."""
        method: Callable[..., Any] = self.implementation.exact
        return method

    @property
    def numeric(self) -> Callable[..., Any] | None:
        """What computes it with arrays, given the array module first; ``None`` where it has
        nothing of its own, and the engines translate :attr:`exact` into NumPy and SciPy."""
        method: Callable[..., Any] | None = getattr(self.implementation, "numeric", None)
        return method

    @cached_property
    def signature(self) -> inspect.Signature:
        """The signature of :attr:`exact`: its parameters, in order."""
        return inspect.signature(self.exact)

    @property
    def parameters(self) -> tuple[str, ...]:
        """The names of the arguments a call takes, in order."""
        return tuple(p.name for p in self.signature.parameters.values() if _positional(p))

    @property
    def keywords(self) -> tuple[str, ...]:
        """The parameters a written form gives by name, known when a program is read, as
        the counts p and q of {}_pF_q."""
        kinds = inspect.Parameter.KEYWORD_ONLY
        return tuple(p.name for p in self.signature.parameters.values() if p.kind is kinds)

    @property
    def arity(self) -> tuple[int, int | None]:
        """How many arguments a call takes: at least the first, and at most the second, or
        any number where it is ``None``."""
        parameters = list(self.signature.parameters.values())
        positional = [p for p in parameters if _positional(p)]
        needed = sum(1 for p in positional if p.default is p.empty)
        if any(p.kind is p.VAR_POSITIONAL for p in parameters):
            return needed, None
        return needed, len(positional)

    def text(self) -> str:
        """How it is called in plain text: its name, with its parameters."""
        variadic = any(p.kind is p.VAR_POSITIONAL for p in self.signature.parameters.values())
        parameters = [*self.parameters, "..."] if variadic else list(self.parameters)
        return f"{self.name}({', '.join(parameters)})"


class Function(Capability):
    """Scalar in, scalar out, as the Bessel functions are: given a tensor, it applies to
    each component, and given data, to each number."""

    kind = "function"
    takes = "numbers and expressions"


class MapFunction(Capability):
    """A function of a matrix, as the determinant and the inverse are: given a map written
    with its indices, as A^a{}_b, it acts on the map, and its value is over the same
    spaces."""

    kind = "map function"
    takes = "a matrix"


class Solver(Capability):
    """Equations in, solutions out, as ``solve`` and ``dsolve``."""

    kind = "solver"
    takes = "equations"


class Constructor(Capability):
    """What builds an object the language works with: a spacetime, an equation, symbols."""

    kind = "constructor"
    takes = "its parameters"


def function(
    name: str,
    *,
    summary: str,
    examples: Sequence[Example],
    aliases: Sequence[str] = (),
    latex: str | Sequence[str] = (),
    unicode: str | Sequence[str] = (),
    requires: str | None = None,
    points: Sequence[Sequence[Any]] = (),
    lifts: bool = True,
    scalar: bool = True,
    integers: Sequence[str] = (),
    takes_names: bool = False,
    gives_form: bool = False,
) -> Callable[[type], Function]:
    """Declare a function: scalar in, scalar out. It ``lifts`` to tensors and data, applied
    to each component, unless it takes its arguments whole, as ``min`` and ``max`` do, and
    gives a ``scalar``, unless it gives another value, as ``latex`` gives text. A form
    written with a letter, as ``\\varphi(n)``, reads as it only where the parameters it
    names as ``integers`` are integers, and prints so only there. Where it ``takes_names``,
    it is given text in quotes as it is written, as ``limit`` is the side of a limit, '-',
    and never reads it as Python. One that ``gives_form``, as ``factor`` does, is shown as
    it computes its value, and its examples are held to the form of their answers."""
    return declares(
        Function,
        name,
        summary=summary,
        examples=examples,
        aliases=aliases,
        latex=latex,
        unicode=unicode,
        requires=requires,
        points=points,
        lifts=lifts,
        scalar=scalar,
        integers=integers,
        takes_names=takes_names,
        gives_form=gives_form,
    )


def map_function(
    name: str,
    *,
    summary: str,
    examples: Sequence[Example],
    aliases: Sequence[str] = (),
    latex: str | Sequence[str] = (),
    unicode: str | Sequence[str] = (),
    requires: str | None = None,
    points: Sequence[Sequence[Any]] = (),
    lifts: bool = True,
    scalar: bool = False,
) -> Callable[[type], MapFunction]:
    """Declare a function of a matrix, which gives a ``scalar``, as the trace does, or a
    matrix. It ``lifts`` to a map written with its indices, acting on it, unless its value
    is over other spaces, as a Kronecker product's is. ``points`` are the arguments it is
    checked at, matrices written as lists of rows."""
    return declares(
        MapFunction,
        name,
        summary=summary,
        examples=examples,
        aliases=aliases,
        latex=latex,
        unicode=unicode,
        requires=requires,
        points=points,
        lifts=lifts,
        scalar=scalar,
    )


def solver(
    name: str,
    *,
    summary: str,
    examples: Sequence[Example],
    aliases: Sequence[str] = (),
    requires: str | None = None,
) -> Callable[[type], Solver]:
    """Declare a solver: equations in, solutions out."""
    return declares(
        Solver, name, summary=summary, examples=examples, aliases=aliases, requires=requires
    )


def constructor(
    name: str,
    *,
    summary: str,
    examples: Sequence[Example],
    aliases: Sequence[str] = (),
    requires: str | None = None,
    takes_names: bool = False,
) -> Callable[[type], Constructor]:
    """Declare what builds an object: a spacetime, whose ``exact`` declares its coordinates
    (:func:`relativisticpy.spacetimes.coordinates_of`), an equation, or symbols, from names
    given in quotes where it ``takes_names``."""
    return declares(
        Constructor,
        name,
        summary=summary,
        examples=examples,
        aliases=aliases,
        requires=requires,
        takes_names=takes_names,
    )


def declares(kind: type[_Kind], name: str, **declared: Any) -> Callable[[type], _Kind]:
    """The decorator that makes a class the implementation of a capability of ``kind``."""
    for written in ("aliases", "latex", "unicode", "integers"):
        if written in declared:
            value = declared[written]
            declared[written] = (value,) if isinstance(value, str) else tuple(value)
    declared["examples"] = tuple(declared["examples"])
    if "points" in declared:
        declared["points"] = tuple(tuple(point) for point in declared["points"])

    def decorate(cls: type) -> _Kind:
        return kind(_implementation(cls, name), name, **declared)

    return decorate


def _implementation(cls: type, name: str) -> Any:
    """An instance of the class that implements the capability ``name``, which needs an
    ``exact`` method and an ``__init__`` without arguments."""
    if not callable(getattr(cls, "exact", None)):
        raise TypeError(
            f"{name}: {cls.__name__} has no exact method; declare how it is computed with "
            f"SymPy, as def exact(self, x): return sympy.{name}(x)"
        )
    try:
        return cls()
    except TypeError as error:
        raise TypeError(
            f"{name}: {cls.__name__} is made without arguments, so its __init__ takes none"
        ) from error


def _positional(parameter: inspect.Parameter) -> bool:
    return parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)


@dataclass(frozen=True)
class Reservation:
    """A name books write for something no integration provides yet: ``what`` it is, the
    forms it is ``written`` in, and what to write ``instead``, if anything. A program that
    writes it is refused, saying what it is, rather than reading it as an unknown function
    or a symbol. A capability with its name retires it."""

    name: str
    what: str
    written: tuple[str, ...]
    instead: str | None = None


def reserved(
    name: str, *, what: str, written: str | Sequence[str], instead: str | None = None
) -> Reservation:
    """Reserve a book name until an integration provides it."""
    forms = (written,) if isinstance(written, str) else tuple(written)
    return Reservation(name, what, forms, instead)
