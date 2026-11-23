r"""Python objects: what an object becomes when a program reads it by name, as
``\texttt{prices}``, from the Python scope a workbook is given.

``\texttt{name}`` is the object called ``name`` in the scope: ``Workbook(scope=globals())``,
the objects passed to :func:`relativisticpy.live.show`, or the scope of an editor.
Attributes, keys and positions follow, as in ``\texttt{table.x}``,
``\texttt{prices["NVDA"]}`` and ``\texttt{data[0]}``. A reference is read by a small grammar
of its own (:func:`reference`), and is never Python to run.

What an object becomes is decided by the object integrations, one for each family of
types, each declared with :func:`python_object` and held to the pattern by the conformance
suite, as every capability is. An object becomes one of four things:

- a :class:`Scalar`: a number, or a SymPy expression;
- :class:`Data`: an array of numbers, each axis labelled where the object labels it, as the
  index and columns of a DataFrame are; data from two objects meets by label, never by
  position;
- an :class:`Array`: a vector or a matrix, as the language's own lists are;
- a :class:`PythonFunction`: called in a program, on numbers, and elementwise on data.

    >>> reference('prices["NVDA"].values[0]').steps
    (('key', 'NVDA'), ('attribute', 'values'), ('key', 0))
    >>> PythonScope({"G": 6.6743e-11}).read(reference("G"))
    Scalar(value=6.67430000000000e-11)

What an editor's program reads is read where the editor is, in the notebook's process, and
crosses to the worker that computes it as data (:class:`Carried`): never the scope itself.
A Python function crosses by name, where another process can import it; one defined in the
notebook cannot, and the editor then computes in the notebook's own process, saying why.
"""

from __future__ import annotations

import contextlib
import difflib
import functools
import importlib.metadata
import inspect
import numbers
import pickle
import re
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol

import sympy

from relativisticpy.errors import (
    CompileError,
    DataError,
    EvaluationError,
    RelativisticPyError,
    UsageError,
    translate,
)
from relativisticpy.integrations._declarations import Example
from relativisticpy.language.source import Span

# == what an object becomes ==================================================================


@dataclass(frozen=True, eq=False)
class Scalar:
    """A number or a SymPy expression: what a Python number, a NumPy number or a SymPy
    expression becomes."""

    value: Any


@dataclass(frozen=True, eq=False)
class Data:
    """An array of numbers, as a NumPy, SciPy sparse or JAX array, with, for each axis, the
    names of its elements where the object gives them, its ``bases``, as a DataFrame's index
    does, and the axis's own name, as a pandas axis or an xarray dimension has one. Data from
    two objects meets by these names, never by position."""

    components: Any
    bases: tuple[tuple[str, ...] | None, ...] = ()
    names: tuple[str | None, ...] = ()

    def __post_init__(self) -> None:
        shape = self.shape
        object.__setattr__(self, "bases", self.bases or (None,) * len(shape))
        object.__setattr__(self, "names", self.names or (None,) * len(shape))
        if len(self.bases) != len(shape) or len(self.names) != len(shape):
            raise ValueError(
                f"data of {len(shape)} axes is given {len(self.bases)} bases and "
                f"{len(self.names)} names: give one of each for every axis, None where it has none"
            )
        for size, basis in zip(shape, self.bases, strict=True):
            if basis is not None and len(basis) != size:
                raise ValueError(f"an axis of {size} elements is given {len(basis)} names")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(getattr(self.components, "shape", ()))

    def __repr__(self) -> str:
        labelled = any(basis is not None for basis in self.bases)
        return f"Data({' × '.join(map(str, self.shape))}{', labelled' if labelled else ''})"


@dataclass(frozen=True, eq=False)
class Array:
    """A vector or a matrix, as the language's own lists are: what a Python list of numbers,
    or of lists of them, becomes. Its entries are SymPy numbers, exact where Python's are."""

    value: Any


@dataclass(frozen=True, eq=False)
class PythonFunction:
    r"""A Python callable, as ``\texttt{hann}`` names one: called in a program, as
    ``\texttt{hann}(n, N)``, on numbers, and elementwise on data (see :func:`call`). ``name``
    is how the program writes it. It crosses to another process by name, where that process
    can import it; one defined in a notebook cannot be pickled for another process."""

    func: Callable[..., Any]
    name: str

    def __str__(self) -> str:
        return f"the Python function {self.name}"

    def __reduce__(self) -> tuple[Any, ...]:
        why = _stays(self.func)
        if why is not None:
            raise pickle.PicklingError(f"the Python function {self.name} {why}")
        return (PythonFunction, (self.func, self.name))


#: What an object becomes.
Becomes = Scalar | Data | Array | PythonFunction


def describe(made: Becomes) -> str:
    """What an object became, in words, as a reading says it."""
    if isinstance(made, Scalar):
        return "a SymPy expression" if _symbolic(made.value) else "a number"
    if isinstance(made, Data):
        return "data of " + " × ".join(map(str, made.shape)) + " numbers"
    if isinstance(made, Array):
        return "a matrix" if getattr(made.value, "rank", lambda: 1)() == 2 else "a vector"
    return "a Python function"


def _symbolic(value: Any) -> bool:
    return isinstance(value, sympy.Basic) and not value.is_number


# == the declaration of an object integration ==================================================


@dataclass(frozen=True, eq=False)
class ObjectConversion:
    r"""What a family of Python objects becomes, as an object integration declares it: the
    types it ``takes``, each by the name of its class, as ``pandas.DataFrame``, which takes
    its subclasses too, or by a method a protocol gives its objects, as ``__array__``; and
    the ``becomes`` method of ``implementation``, an instance of the class the declaration
    decorates, which makes an object a :class:`Scalar`, :class:`Data`, an :class:`Array` or
    a :class:`PythonFunction`. ``requires`` names the library whose types it takes, which it
    never imports: no object of a library's type is made without the library."""

    #: The kind of capability, as the conformance suite and the reference name it.
    kind: ClassVar[str] = "object"

    implementation: Any
    name: str
    takes: tuple[str, ...]
    summary: str
    examples: tuple[Example, ...]
    requires: str | None = None

    @property
    def module(self) -> str:
        """The module that declares it."""
        return type(self.implementation).__module__

    @property
    def types(self) -> tuple[str, ...]:
        """The classes it takes, by their qualified names."""
        return tuple(taken for taken in self.takes if not taken.startswith("_"))

    @property
    def protocols(self) -> tuple[str, ...]:
        """The methods by which it takes objects of a protocol, as ``__array__``."""
        return tuple(taken for taken in self.takes if taken.startswith("_"))

    def becomes(self, value: Any) -> Becomes:
        """What ``value`` becomes."""
        made: Becomes = self.implementation.becomes(value)
        return made


def python_object(
    name: str,
    *,
    takes: str | Sequence[str],
    summary: str,
    examples: Sequence[Example],
    requires: str | None = None,
) -> Callable[[type], ObjectConversion]:
    r"""Declare what a family of Python objects becomes when a program reads one, as
    ``\texttt{prices}`` reads a DataFrame: the types it ``takes``, by the qualified names of
    their classes, ``"pandas.DataFrame"``, or by the method a protocol gives its objects,
    ``"__array__"``; a ``summary``; and two or more ``examples``, each a program that reads
    the objects it is ``given``, with its answer and where the answer comes from::

        @python_object(
            "pandas",
            takes=["pandas.DataFrame", "pandas.Series"],
            summary="labelled data: its index and columns are the bases of spaces",
            requires="pandas",
            examples=[...],
        )
        class Tables:
            def becomes(self, value):
                return Data(value.to_numpy(), bases=..., names=...)
    """
    taken = (takes,) if isinstance(takes, str) else tuple(takes)

    def decorate(cls: type) -> ObjectConversion:
        if not callable(getattr(cls, "becomes", None)):
            raise TypeError(
                f"{name}: {cls.__name__} has no becomes method; declare what an object "
                "becomes, as def becomes(self, value): return Data(numpy.asarray(value))"
            )
        try:
            implementation = cls()
        except TypeError as error:
            raise TypeError(
                f"{name}: {cls.__name__} is made without arguments, so its __init__ takes none"
            ) from error
        return ObjectConversion(implementation, name, taken, summary, tuple(examples), requires)

    return decorate


# == which integration takes an object ==========================================================


def conversion_of(value: Any) -> ObjectConversion | None:
    """The object integration that takes ``value``: the one that takes its most specific
    class, or, if none takes its class, one whose protocol it keeps, a function last;
    ``None`` where none does. A library's classes are found among the modules already
    imported, so that nothing is imported to tell: an object of a library's type is never
    made without it."""
    from relativisticpy.integrations._discovery import catalogue

    conversions = catalogue().objects
    mro = type(value).__mro__
    best: ObjectConversion | None = None
    rank = len(mro) + 1
    for conversion in conversions:
        for taken in conversion.types:
            cls = _imported(taken)
            if cls is None or not isinstance(value, cls):
                continue
            position = mro.index(cls) if cls in mro else len(mro)
            if position < rank:
                best, rank = conversion, position
    if best is not None:
        return best
    for conversion in conversions:  # a protocol of data first, as __array__
        if any(_keeps(value, method) for method in conversion.protocols if method != "__call__"):
            return conversion
    for conversion in conversions:  # then anything called
        if "__call__" in conversion.protocols and callable(value):
            return conversion
    return None


def _keeps(value: Any, method: str) -> bool:
    """Whether ``value`` keeps a protocol: its class has the protocol's method."""
    return hasattr(type(value), method)


def _imported(qualified: str) -> type | None:
    """The class called ``qualified``, as ``pandas.DataFrame``, if its module is imported."""
    parts = qualified.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        found: Any = sys.modules.get(".".join(parts[:cut]))
        if found is None:
            continue
        for attribute in parts[cut:]:
            found = getattr(found, attribute, None)
        return found if isinstance(found, type) else None
    return None


def becomes(value: Any, reference: Reference | None = None) -> Becomes:
    """What ``value`` becomes, by the object integration that takes it: refused, naming what
    is read, where none takes it, and in RelativisticPy's words where the one that takes it
    cannot read it. ``reference`` is how the program names it, for the messages."""
    written = f"\\texttt{{{reference.written}}}" if reference is not None else "the object"
    conversion = conversion_of(value)
    if conversion is None:
        raise DataError(
            f"{written} is {_kind(value)}, which RelativisticPy does not read",
            notes=[f"it reads {_what_is_read()}"],
            hint=_instead(value, reference),
        )
    try:
        made = conversion.becomes(value)
    except RelativisticPyError as error:
        error.message = f"{written}: {error.message}"
        raise
    except Exception as error:  # the object's own library failed on it
        raise DataError(
            f"{written}, {_kind(value)}, cannot be read: {type(error).__name__}: {error}"
        ) from error
    if not isinstance(made, Scalar | Data | Array | PythonFunction):
        raise TypeError(
            f"{conversion.name}: becomes gives {type(made).__name__}; give a Scalar, Data, an "
            "Array or a PythonFunction"
        )
    if isinstance(made, Data):
        return Data(own(made.components, written), made.bases, made.names)
    if isinstance(made, PythonFunction) and reference is not None:
        return PythonFunction(made.func, written)
    if isinstance(made, Scalar):
        return Scalar(_scalar_value(made.value, conversion))
    return made


def _scalar_value(value: Any, conversion: ObjectConversion) -> Any:
    """A scalar as the language keeps one: a SymPy value, or a NumPy number."""
    from relativisticpy.tensors import numerical

    if isinstance(value, sympy.Basic) or numerical.is_numerical(value):
        return value
    if isinstance(value, numbers.Number):
        return sympy.sympify(value, strict=True)
    raise TypeError(
        f"{conversion.name}: becomes gives a Scalar of {_kind(value)}; give a number or a SymPy "
        "expression"
    )


def own(components: Any, written: str) -> Any:
    """Numerical components of one's own: a copy, so that the object changed in place
    afterwards changes nothing read from it, with small integers widened, so that arithmetic
    cannot overflow them; refused where they are text."""
    from relativisticpy.tensors import numerical

    dtype = getattr(components, "dtype", None)
    if dtype is not None and dtype.kind in "USaO" and _has_text(components):
        raise DataError(f"the data of {written} is text: read numbers")
    if numerical.is_jax(components):
        return components
    if numerical.is_sparse(components):
        return numerical.widened(components.copy())
    return numerical.widened(numerical.numpy().array(components))


def _has_text(components: Any) -> bool:
    dtype = components.dtype
    if dtype.kind in "USa":
        return True
    return any(isinstance(item, str | bytes) for item in components.ravel().tolist())


def _kind(value: Any) -> str:
    """What a value is, as a message names it: ``a dict``, ``a collections.deque``."""
    cls = type(value)
    named = cls.__name__ if cls.__module__ == "builtins" else f"{cls.__module__}.{cls.__qualname__}"
    return f"{'an' if named[:1].lower() in 'aeiou' else 'a'} {named}"


def _what_is_read() -> str:
    """What the object integrations read, by the types and protocols they take."""
    from relativisticpy.integrations._discovery import catalogue

    families = []
    for conversion in catalogue().objects:
        shown = [shown_type(taken) for taken in conversion.takes]
        families.append(", ".join(shown))
    return "; ".join(families)


def shown_type(taken: str) -> str:
    """A type an object integration takes, as messages and the reference show it."""
    if taken == "__call__":
        return "anything callable"
    if taken.startswith("_"):
        return f"anything with {taken}"
    return taken.removeprefix("builtins.")


def _instead(value: Any, reference: Reference | None) -> str | None:
    """What to read instead of a value no integration takes, where there is something."""
    if isinstance(value, str | bytes):
        return "text is not a value to compute with: write the expression in the program"
    if isinstance(value, Mapping):
        named = f'\\texttt{{{reference.written}["key"]}}' if reference is not None else ""
        return f"read one of its values by its key, as {named}, or make it a pandas Series"
    return None


# == references ===============================================================================

#: A name of Python's: letters, digits and underscores, not starting with a digit.
_NAME = r"[A-Za-z_][A-Za-z0-9_]*"
_FIRST = re.compile(rf"({_NAME})")
_STEP = re.compile(rf"""\s*(?:\.\s*({_NAME})|\[\s*(?:"([^"]*)"|'([^']*)'|(-?\d+))\s*\])""")
#: What LaTeX escapes in text, as the underscore of ``\texttt{M\_sun}``.
_ESCAPED = re.compile(r"\\([_#$%&{}])")

#: The kinds of step a reference takes into an object.
StepKind = Literal["attribute", "key"]


@dataclass(frozen=True)
class Reference:
    r"""A Python object as a program names it: a ``name`` in the scope, then each attribute,
    key and position of its ``steps``, as ``prices["NVDA"]`` takes the key ``"NVDA"`` of
    ``prices``; ``written`` as the program writes it, inside ``\texttt{...}``."""

    name: str
    steps: tuple[tuple[StepKind, str | int], ...]
    written: str

    def shown(self, upto: int | None = None) -> str:
        """The reference as Python writes it, up to its step ``upto``."""
        parts = [self.name]
        for kind, value in self.steps[:upto]:
            parts.append(f".{value}" if kind == "attribute" else f"[{value!r}]")
        return "".join(parts).replace("'", '"')


def reference(written: str, span: Span | None = None) -> Reference:
    r"""The reference ``\texttt{written}`` makes: a name, then attributes, ``.x``, keys in
    quotes, ``["NVDA"]``, and positions, ``[0]``. Anything else is refused, saying what may
    be written: a reference is never Python to run. LaTeX's escapes are read as what they
    escape, so ``M\_sun`` is ``M_sun``."""
    text = _ESCAPED.sub(r"\1", written).strip()
    first = _FIRST.match(text)
    if first is None:
        raise _unreadable(written, text, 0, span)
    name, position = first.group(1), first.end()
    steps: list[tuple[StepKind, str | int]] = []
    while position < len(text):
        step = _STEP.match(text, position)
        if step is None:
            raise _unreadable(written, text, position, span)
        attribute, double, single, integer = step.groups()
        if attribute is not None:
            steps.append(("attribute", attribute))
        elif integer is not None:
            steps.append(("key", int(integer)))
        else:
            steps.append(("key", double if double is not None else single))
        position = step.end()
    for part in (name, *(value for kind, value in steps if kind == "attribute")):
        if isinstance(part, str) and part.startswith("__") and part.endswith("__"):
            raise CompileError(
                f"\\texttt{{{written}}} reads {part}, which is Python's own, not data",
                span=span,
                hint="a reference reads the objects of the scope, their attributes and keys, "
                "never Python's internals, as __class__",
            )
    return Reference(name, tuple(steps), written)


def _unreadable(written: str, text: str, position: int, span: Span | None) -> CompileError:
    where = f"after {text[:position]}, at {text[position:]!r}" if position else f"at {text!r}"
    return CompileError(
        f"\\texttt{{{written}}} is not a reference to a Python object: {where}",
        span=span,
        notes=["a reference is a name, then attributes, keys in quotes and positions"],
        hint=r'write it as \texttt{prices}, \texttt{table.x}, \texttt{prices["NVDA"]} or '
        r"\texttt{data[0]}: it is read, never run as Python",
    )


# == scopes ===================================================================================


class ObjectReader(Protocol):
    """What reads the objects a program names: a Python scope, or what one read, carried to
    another process. ``backend`` is the array library data read from one is kept in."""

    backend: str

    def read(self, reference: Reference) -> Becomes: ...


class PythonScope:
    r"""The Python scope a workbook is given, in which ``\texttt{name}`` names an object: a
    mapping, as ``globals()`` or a dict, or anything whose attributes are read, as a module.
    ``then`` is read where it has no such name, as a workbook's own scope is under the
    objects passed to :func:`~relativisticpy.live.show`. It is read anew each time a program
    names an object, so that a program run again reads the objects as they are then; and it
    stays in its process: what an editor's program reads crosses to another as data
    (:class:`Carried`)."""

    def __init__(
        self, scope: Any, *, then: ObjectReader | None = None, backend: str = "numpy"
    ) -> None:
        if scope is None or isinstance(
            scope, str | bytes | numbers.Number | list | tuple | set | frozenset
        ):
            raise UsageError(
                f"a scope is a mapping, as globals() or a dict, or a module or object whose "
                f"attributes are read, not {_kind(scope)}",
                hint="give the objects by name: scope={'prices': prices}",
            )
        self.scope = scope
        self.then = then
        self.backend = backend

    def read(self, reference: Reference) -> Becomes:
        """What the object ``reference`` names becomes."""
        found, value = self._lookup(reference.name)
        if not found:
            if self.then is not None:
                return self.then.read(reference)
            raise self._missing(reference)
        for number, (kind, step) in enumerate(reference.steps):
            value = _step(value, kind, step, reference, number)
        return becomes(value, reference)

    def names(self) -> list[str]:
        """The names of the scope, but Python's own."""
        if isinstance(self.scope, Mapping):
            keys = [key for key in self.scope if isinstance(key, str)]
        else:
            keys = dir(self.scope)
        return [key for key in keys if not key.startswith("_")]

    def _lookup(self, name: str) -> tuple[bool, Any]:
        if isinstance(self.scope, Mapping):
            return (True, self.scope[name]) if name in self.scope else (False, None)
        try:
            return True, getattr(self.scope, name)
        except AttributeError:
            return False, None

    def _missing(self, reference: Reference) -> CompileError:
        name = reference.name
        close = difflib.get_close_matches(name, self.names(), n=1)
        return CompileError(
            f"\\texttt{{{reference.written}}} names {name}, which is not in the Python scope",
            notes=[f"did you mean {close[0]}?"] if close else [],
            hint=f"define {name} in Python first, or give it, as in show(source, {name}={name})",
        )

    def __reduce__(self) -> tuple[Any, ...]:
        raise TypeError(
            "a Python scope stays in its process: what a program reads from it goes to "
            "another process as data"
        )


def _step(value: Any, kind: StepKind, step: str | int, reference: Reference, number: int) -> Any:
    """The attribute or key ``step`` of ``value``, the object ``reference`` names up to its
    step ``number``."""
    owner = reference.shown(number)
    try:
        if kind == "attribute":
            assert isinstance(step, str)
            return getattr(value, step)
        return value[step]
    except AttributeError:
        public = [name for name in dir(value) if not name.startswith("_")]
        close = difflib.get_close_matches(str(step), public, n=1)
        raise CompileError(
            f"\\texttt{{{reference.written}}}: {owner} has no attribute {step}",
            notes=[f"did you mean {owner}.{close[0]}?"] if close else [],
        ) from None
    except (KeyError, IndexError):
        at = f"[{step}]" if isinstance(step, int) else f'["{step}"]'
        raise CompileError(
            f"\\texttt{{{reference.written}}}: {owner} has nothing at {at}",
            notes=_keys(value),
        ) from None
    except Exception as error:  # what the object does when it is read is its own
        raise CompileError(
            f"\\texttt{{{reference.written}}}: reading {owner} failed: "
            f"{type(error).__name__}: {error}"
        ) from None


def _keys(value: Any) -> list[str]:
    """A note of the keys an object has, where it has a few that are named."""
    keys = getattr(value, "keys", None)
    try:
        names = [str(key) for key in keys()] if callable(keys) else []
    except Exception:  # an object whose keys cannot be listed says nothing of them
        return []
    if not names:
        length = getattr(value, "__len__", None)
        return [f"it has {length()} positions, from 0"] if callable(length) else []
    shown = ", ".join(names[:8]) + (f" and {len(names) - 8} more" if len(names) > 8 else "")
    return [f"its keys are {shown}"]


@dataclass(frozen=True)
class Refusal:
    """Why an object could not be read, carried to another process to be said there."""

    kind: type[RelativisticPyError]
    message: str
    hint: str | None
    notes: tuple[str, ...]

    def error(self) -> RelativisticPyError:
        return self.kind(self.message, hint=self.hint, notes=self.notes)


class Carried:
    """What a program reads from a Python scope, read where the scope is and carried to the
    process that computes the program: what each reference, as it is written, became, or
    why it could not be read. A reference not read where the scope is is refused."""

    def __init__(self, read: Mapping[str, Becomes | Refusal], *, backend: str = "numpy") -> None:
        self.objects = dict(read)
        self.backend = backend

    @classmethod
    def reading(cls, reader: ObjectReader, references: Sequence[Reference]) -> Carried:
        """What ``reader`` makes of each of ``references``, to carry to another process."""
        read: dict[str, Becomes | Refusal] = {}
        for each in references:
            try:
                read[each.written] = reader.read(each)
            except Exception as error:  # said where the program is compiled, as it would be here
                failure = translate(error)
                notes = tuple(failure.notes)
                read[each.written] = Refusal(type(failure), failure.message, failure.hint, notes)
        return cls(read, backend=reader.backend)

    def read(self, reference: Reference) -> Becomes:
        found = self.objects.get(reference.written)
        if found is None:
            raise CompileError(
                f"\\texttt{{{reference.written}}} was not read where the Python scope is",
                hint="run the program again, so that it is read",
            )
        if isinstance(found, Refusal):
            raise found.error()
        return found

    def same(self, other: Carried | None) -> bool:
        """Whether ``other`` carries the same objects, as they are now."""
        if other is None or self.objects.keys() != other.objects.keys():
            return False
        return all(_same(self.objects[key], other.objects[key]) for key in self.objects)


def _same(one: Becomes | Refusal, other: Becomes | Refusal) -> bool:
    """Whether two objects read are the same, value for value."""
    if type(one) is not type(other):
        return False
    if isinstance(one, Refusal):
        return one == other
    if isinstance(one, PythonFunction):
        assert isinstance(other, PythonFunction)
        return one.func is other.func
    if isinstance(one, Data):
        assert isinstance(other, Data)
        return (
            one.bases == other.bases and one.names == other.names
            and _same_components(one.components, other.components)
        )  # fmt: skip
    assert isinstance(one, Scalar | Array)
    assert isinstance(other, Scalar | Array)
    try:
        return bool(type(one.value) is type(other.value) and one.value == other.value)
    except Exception:  # values that cannot be compared are taken to differ
        return False


def _same_components(one: Any, other: Any) -> bool:
    from relativisticpy.tensors import numerical

    if type(one) is not type(other) or getattr(one, "shape", None) != getattr(other, "shape", None):
        return False
    np = numerical.numpy()
    return bool(np.array_equal(numerical.dense(one), numerical.dense(other), equal_nan=True))


# == calling a Python function ===============================================================


def call(function: PythonFunction, *args: Any) -> Any:
    r"""A Python function called on the values of a program, as ``\texttt{hann}(n, N)``: on
    numbers, as Python's numbers; on expressions, as SymPy's; and on data elementwise, given
    whole arrays where it takes them, as a function written with NumPy does, and one number
    at a time where it does not. What it gives is a number, an expression, or, on data, an
    array of the data's shape."""
    from relativisticpy.tensors import numerical

    arrays = [arg for arg in args if _is_array(arg)]
    if not arrays:
        return _scalar(function, _called(function, [_plain(arg) for arg in args]))
    if not all(numerical.is_numerical(array) for array in arrays):
        return _symbolic_elementwise(function, args)
    np = numerical.numpy()
    given = [np.asarray(numerical.dense(arg)) if _is_array(arg) else _plain(arg) for arg in args]
    shape = np.broadcast_shapes(*(np.shape(arg) for arg in given if _is_array(arg)))
    try:
        whole = np.asarray(function.func(*given))
    except Exception:  # it takes numbers, not arrays: it is given one at a time below
        whole = None
    if whole is not None and whole.shape == shape and whole.dtype.kind in "biufc":
        return numerical.widened(whole)
    spread = np.broadcast_arrays(*(arg if _is_array(arg) else np.asarray(arg) for arg in given))
    results = [
        _number(function, _called(function, [part[index].item() for part in spread]))
        for index in np.ndindex(shape)
    ]
    kind = complex if any(isinstance(result, complex) for result in results) else float
    return np.asarray(results, dtype=kind).reshape(shape)


def _is_array(value: Any) -> bool:
    from relativisticpy.tensors import numerical

    if isinstance(value, sympy.NDimArray):
        return True
    return numerical.is_numerical(value) and getattr(value, "shape", ()) != ()


def _plain(value: Any) -> Any:
    """A value as a Python function is given it: a number as Python's, an expression as
    SymPy's, an array or a list as it is."""
    if isinstance(value, sympy.Basic) and value.is_number:
        if value.is_Integer:
            return int(value)
        number = complex(value)
        return number.real if number.imag == 0 else number
    if isinstance(value, sympy.logic.boolalg.BooleanAtom):
        return bool(value)
    return value


def _called(function: PythonFunction, args: Sequence[Any]) -> Any:
    try:
        return function.func(*args)
    except Exception as error:
        shown = ", ".join(_shown(arg) for arg in args)
        raise EvaluationError(
            f"the Python function {function.name}({shown}) raised {type(error).__name__}: {error}",
            hint="the error is the function's own, not RelativisticPy's: call it in Python "
            "with these arguments to see why",
        ) from error


def _shown(value: Any) -> str:
    text = str(value)
    return text if len(text) <= 24 else text[:21] + "..."


def _scalar(function: PythonFunction, result: Any) -> Any:
    """What a function gave for numbers, as the language keeps a number or an expression."""
    from relativisticpy.tensors import numerical

    if numerical.is_numerical(result) and getattr(result, "shape", ()) == ():
        return result
    if isinstance(result, sympy.Basic):
        return result
    if isinstance(result, bool):  # a truth value, as a claim that holds is
        return sympy.true if result else sympy.false
    if isinstance(result, numbers.Number):
        return sympy.sympify(result, strict=True)
    raise _not_a_number(function, result)


def _number(function: PythonFunction, result: Any) -> Any:
    """What a function gave for one number of data, as a Python number."""
    value = _scalar(function, result)
    try:
        number = complex(value)
    except TypeError:
        raise EvaluationError(
            f"the Python function {function.name} gave {_shown(value)} for a number of data, "
            "which is no number",
        ) from None
    return number.real if number.imag == 0 else number


def _not_a_number(function: PythonFunction, result: Any) -> EvaluationError:
    return EvaluationError(
        f"the Python function {function.name} gave {_kind(result)}, not a number",
        hint="a Python function called in a program gives a number, or, given data, an "
        "array of the data's shape",
    )


def _symbolic_elementwise(function: PythonFunction, args: Sequence[Any]) -> Any:
    """A function applied to each component of tensors with symbolic components."""
    shaped = [_symbolic_array(arg) if _is_array(arg) else arg for arg in args]
    shape = next(arg.shape for arg in shaped if isinstance(arg, sympy.NDimArray))
    flat = [
        list(sympy.flatten(arg.tolist())) if isinstance(arg, sympy.NDimArray) else None
        for arg in shaped
    ]
    size = len(next(items for items in flat if items is not None))
    results = []
    for position in range(size):
        given = [
            _plain(items[position]) if items is not None else _plain(arg)
            for items, arg in zip(flat, shaped, strict=True)
        ]
        results.append(_scalar(function, _called(function, given)))
    return sympy.ImmutableDenseNDimArray(results, shape)


def _symbolic_array(value: Any) -> Any:
    """Components as a SymPy array, numbers among them as SymPy's."""
    from relativisticpy.tensors import arrays, numerical

    if numerical.is_numerical(value):
        listed = numerical.numpy().asarray(numerical.dense(value)).tolist()
        return sympy.ImmutableDenseNDimArray(listed)
    return arrays.to_array(value)


# == what crosses to another process ==========================================================


def _stays(func: Callable[..., Any]) -> str | None:
    """Why a Python function cannot go to another process, which imports it by name: it was
    defined in the notebook, or in a module no other process may import; ``None`` where it
    can go."""
    if isinstance(func, functools.partial):
        return _stays(func.func)
    module = getattr(func, "__module__", None)
    if not isinstance(module, str) or module.startswith("__"):
        return "is defined in the notebook, so no other process can import it"
    top = module.split(".")[0]
    if top not in sys.stdlib_module_names and top not in _installed():
        return (
            f"is defined in {module}, which another process may not import, as it is not installed"
        )
    try:
        pickle.dumps(func)
    except Exception as error:
        return f"cannot be pickled ({type(error).__name__}: {error})"
    return None


@functools.cache
def _installed() -> frozenset[str]:
    """The modules of the libraries installed, by the names they are imported by."""
    return frozenset(importlib.metadata.packages_distributions())


# == the conformance suite, for object integrations ===========================================


def check(conversion: ObjectConversion, names: Sequence[str] | None = None) -> dict[str, list[str]]:
    """The failures of an object integration in each check of its kind, or in those
    ``names``: an empty list for each it passes."""
    return {
        name: runs(conversion) for name, runs in CHECKS.items() if names is None or name in names
    }


@contextlib.contextmanager
def usable(conversion: ObjectConversion) -> Iterator[None]:
    """The language with ``conversion`` in it while the block runs: as it is, if an
    integration declares it, and installed for the block otherwise."""
    from relativisticpy.integrations._discovery import catalogue, installed

    if any(conversion is each for each in catalogue().objects):
        yield
    else:
        with installed(conversion):
            yield


def declaration(conversion: ObjectConversion) -> list[str]:
    """A name of its own, a summary, the types it takes, examples that give the objects
    their programs read, with sources, and the library it takes the types of, never imported
    when its module is."""
    from relativisticpy.integrations._discovery import catalogue

    name, failures = conversion.name, []
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        failures.append(
            f"{name!r}: name an object integration with one word in lower case, as pandas"
        )
    other = next((o for o in catalogue().objects if o.name == name and o is not conversion), None)
    if other is not None:
        failures.append(f"{name}: {other.module} declares an object integration of this name")
    summary = conversion.summary.strip()
    if not summary or "\n" in summary or summary.endswith(".") or "|" in summary:
        failures.append(
            f"{name}: give a summary, one phrase without a full stop or a bar, as the reference "
            'lists it: summary="a DataFrame or Series: labelled data"'
        )
    if not conversion.takes:
        failures.append(
            f"{name}: declare the types it takes, by their classes, as "
            "takes=['pandas.DataFrame'], or by a protocol's method, as takes=['__array__']"
        )
    for taken in conversion.takes:
        if not re.fullmatch(r"_\w+|[A-Za-z_]\w*(\.[A-Za-z_]\w*)+", taken):
            failures.append(
                f"{name}: it takes {taken!r}, which is neither a class by its module and name, "
                "as pandas.DataFrame, nor a protocol's method, as __array__"
            )
    if len(conversion.examples) < 2:
        failures.append(
            f"{name} has {len(conversion.examples)} example(s): give at least two, each a "
            r"program that reads an object, as Example(r'P_{ta} := \texttt{prices}', ..., "
            "given=lambda: {'prices': ...})"
        )
    for example in conversion.examples:
        program = " ".join(example.program.split())
        if not example.source.strip():
            failures.append(f"{name}: the example {program} says not where its answer comes from")
        if not callable(example.given):
            failures.append(
                f"{name}: the example {program} gives no objects: give them, as "
                "given=lambda: {'prices': prices}"
            )
    try:
        inspect.signature(conversion.implementation.becomes).bind(None)
    except TypeError:
        failures.append(f"{name}: becomes takes the object alone, as def becomes(self, value)")
    failures += _imports(conversion)
    return failures


def _imports(conversion: ObjectConversion) -> list[str]:
    """That a module of object integrations imports no optional library when it is imported:
    an object of the library's types is never made without it."""
    from relativisticpy.integrations._conformance import distribution_of, optional_libraries

    module = sys.modules.get(conversion.module)
    for value in vars(module).values() if module is not None else ():
        owner = getattr(value, "__name__", "") if isinstance(value, type(sys)) else ""
        library = (owner or getattr(value, "__module__", None) or "").split(".")[0]
        if library in optional_libraries():
            return [
                f"{conversion.name}: its module imports {library} when it is imported; import "
                "it inside becomes, or inside what makes the examples' objects, and declare "
                f'requires="{distribution_of(library)}"'
            ]
    return []


def examples(conversion: ObjectConversion) -> list[str]:
    """Each example, given its objects, reads one this integration takes, and gives its
    answer. An example whose objects need a library not installed is checked where it is."""
    from relativisticpy import Workbook
    from relativisticpy.integrations._conformance import as_shown, from_source, same

    name, failures = conversion.name, []
    with usable(conversion):
        for example in conversion.examples:
            program = " ".join(example.program.split())
            objects = _given(example)
            if objects is None:
                continue
            if not any(conversion_of(value) is conversion for value in objects.values()):
                failures.append(f"{name}'s example {program} is given no object {name} takes")
            if "\\texttt" not in example.program:
                failures.append(
                    f"{name}'s example {program} reads no object: name one, as \\texttt{{prices}}"
                )
            try:
                with Workbook(scope=objects) as wb:
                    values = wb.exe(example.program)
            except RelativisticPyError as error:
                failures.append(f"{name}'s example {program} fails: {error.kind}: {error.message}")
                continue
            if not values:
                failures.append(f"{name}'s example {program} shows nothing")
            elif not same(_listed(values[-1]), example.answer, example.tolerance):
                failures.append(
                    f"{name}'s example {program} gives {as_shown(values[-1])}, not "
                    f"{as_shown(example.answer)}{from_source(example)}: fix becomes, or the "
                    "example's "
                    "answer"
                )
    return failures


def _listed(value: Any) -> Any:
    """A value as an example's answer is written: data as nested lists of its numbers."""
    from relativisticpy.tensors import Tensor, numerical

    components = value.components if isinstance(value, Tensor) else value
    if numerical.is_numerical(components) and getattr(components, "shape", ()) != ():
        return numerical.numpy().asarray(numerical.dense(components)).tolist()
    return value


def _given(example: Example) -> dict[str, Any] | None:
    """The objects an example gives, or ``None`` where it gives none, or they need a library
    that is not installed."""
    if example.given is None:
        return None
    try:
        return dict(example.given())
    except ImportError:
        return None


def errors(conversion: ObjectConversion) -> list[str]:
    """What its objects become meets the rest of the language as any value does: called,
    added and given to a function, it is computed or refused, saying why, and never fails
    inside RelativisticPy."""
    from relativisticpy import Workbook
    from relativisticpy.errors import InternalError

    failures = []
    with usable(conversion):
        for example in conversion.examples:
            objects = _given(example) or {}
            for key, value in objects.items():
                if conversion_of(value) is not conversion:
                    continue
                written = f"\\texttt{{{key}}}"
                for program in (f"{written}(1)", f"{written} + {written}", f"\\sin({written})"):
                    try:
                        with Workbook(scope=objects) as wb:
                            wb.exe(program)
                    except InternalError as error:
                        failures.append(
                            f"{conversion.name}: {program} fails inside RelativisticPy, with "
                            f"{key} given ({'; '.join(error.notes)}): make becomes give what "
                            "the language computes with, a Scalar, Data, an Array or a "
                            "PythonFunction"
                        )
                    except RelativisticPyError:
                        pass  # refused, saying why
    return failures


def in_reference(conversion: ObjectConversion) -> list[str]:
    """The language reference says what its objects become, from its declaration."""
    from relativisticpy.reference import markdown

    with usable(conversion):
        page = markdown()
    wanted = [*(f"`{shown_type(taken)}`" for taken in conversion.takes), conversion.summary]
    missing = [item for item in wanted if item not in page]
    if not missing:
        return []
    return [
        f"the reference does not describe {conversion.name}: it lacks {', '.join(missing)}; "
        "write it again with uv run python -m relativisticpy.reference --write docs/reference.md"
    ]


#: The checks of an object integration, by name.
CHECKS: dict[str, Callable[[ObjectConversion], list[str]]] = {
    "declaration": declaration,
    "examples": examples,
    "errors": errors,
    "reference": in_reference,
}
