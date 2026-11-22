"""Finding the integrations, and computing what they declare.

Every module of this package is an integration, but for those whose name starts with an
underscore, which make the framework; adding the file is enough. Packages outside
RelativisticPy contribute through the entry-point group ``relativisticpy.integrations``,
Python's standard mechanism: each entry point names a module, a capability, or a list of
them. The catalogue is found once, the first time it is asked for.

The registry of implementations names what computes a capability by the capability's name
(:class:`Exactly`, :class:`Numerically`), so that a worker process finds it again, and an
optional library is imported only when a capability is first used: a missing one is an
error that says what to install.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import inspect
import pkgutil
import re
import types
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import sympy

from relativisticpy.errors import EvaluationError, RelativisticPyError, _known_failure
from relativisticpy.integrations._declarations import Capability, Function, Reservation

#: The entry-point group through which packages outside RelativisticPy contribute
#: integrations.
GROUP = "relativisticpy.integrations"


class Catalogue:
    """The capabilities every integration declares, by each of their names, and the book
    names still reserved: those no capability provides by its name."""

    def __init__(
        self, capabilities: Iterable[Capability], reservations: Iterable[Reservation] = ()
    ) -> None:
        self.capabilities = tuple(capabilities)
        self._by_name: dict[str, Capability] = {}
        for capability in self.capabilities:
            for name in capability.names:
                other = self._by_name.get(name)
                if other is not None:
                    raise ValueError(
                        f"two integrations declare {name!r}: {other.module} and "
                        f"{capability.module}; rename one of them"
                    )
                self._by_name[name] = capability
        self.declared = tuple(reservations)
        self.reservations = tuple(r for r in self.declared if r.name not in self._by_name)

    def find(self, name: str) -> Capability:
        """The capability called ``name``, by its name or an alias."""
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(f"no integration declares {name!r}") from None

    def get(self, name: str) -> Capability | None:
        return self._by_name.get(name)


_catalogue: Catalogue | None = None
_listeners: list[Callable[[], None]] = []


def catalogue() -> Catalogue:
    """The capabilities and reservations every integration declares, found once."""
    global _catalogue
    if _catalogue is None:
        _catalogue = _discover()
    return _catalogue


def capabilities() -> tuple[Capability, ...]:
    """Every capability, in the order of the modules that declare them, by name, and then
    of their declarations."""
    return catalogue().capabilities


def reservations() -> tuple[Reservation, ...]:
    """The book names reserved and not yet provided."""
    return catalogue().reservations


def find(name: str) -> Capability:
    """The capability called ``name``."""
    return catalogue().find(name)


def on_change(listener: Callable[[], None]) -> None:
    """Call ``listener`` whenever the capabilities change, as :func:`installed` changes
    them: the tables built from them are built again."""
    _listeners.append(listener)


@contextlib.contextmanager
def installed(*added: Capability) -> Iterator[None]:
    """The language with the capabilities ``added`` besides those the integrations
    declare, while the block runs: to check one before it is part of an integration."""
    global _catalogue
    before = catalogue()
    _catalogue = Catalogue([*before.capabilities, *added], before.declared)
    _changed()
    try:
        yield
    finally:
        _catalogue = before
        _changed()


def _changed() -> None:
    for listener in _listeners:
        listener()


def _discover() -> Catalogue:
    found: list[Capability] = []
    reserved: list[Reservation] = []
    for declarations in _declarations():
        for value in declarations:
            if isinstance(value, Capability) and not any(value is c for c in found):
                found.append(value)
            elif isinstance(value, Reservation) and value not in reserved:
                reserved.append(value)
    return Catalogue(found, reserved)


def _declarations() -> Iterator[Iterable[Any]]:
    """What each integration declares: the values of the modules of this package, then
    what the entry points name."""
    package = importlib.import_module(__package__ or "relativisticpy.integrations")
    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda info: info.name):
        if not info.ispkg and not info.name.startswith("_"):
            module = importlib.import_module(f"{package.__name__}.{info.name}")
            yield list(vars(module).values())
    for entry in importlib.metadata.entry_points(group=GROUP):
        try:
            loaded = entry.load()
        except Exception as error:  # one broken package must not stop the language
            warnings.warn(
                f"the integration {entry.name} ({entry.value}) could not be loaded: "
                f"{type(error).__name__}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if isinstance(loaded, types.ModuleType):
            yield list(vars(loaded).values())
        elif isinstance(loaded, Capability | Reservation):
            yield [loaded]
        else:
            yield list(loaded)


# -- computing what a capability declares ----------------------------------------------


@dataclass(frozen=True)
class Exactly:
    """What computes the capability called ``name`` with SymPy, found by that name: so it
    is the same in a worker process, and a library it needs is imported only now."""

    name: str

    def __call__(self, *args: Any, **attributes: Any) -> Any:
        capability = find(self.name)
        try:
            return capability.exact(*args, **attributes)
        except RelativisticPyError:
            raise
        except ImportError as error:
            raise missing(self.name, capability, error) from None
        except (TypeError, AttributeError) as error:
            refused = wrong_kind(self.name, capability, args)
            if refused is None or _known_failure(error) is not None:
                raise  # said in RelativisticPy's words where the error is translated
            raise refused from error

    @property
    def __signature__(self) -> inspect.Signature:
        """The parameters of the capability's ``exact``, as calls are checked against."""
        return find(self.name).signature


@dataclass(frozen=True)
class Numerically:
    """What computes the capability called ``name`` with arrays, given the array module and
    the arguments, found by that name, as :class:`Exactly` is."""

    name: str

    def __call__(self, xp: Any, *args: Any, **attributes: Any) -> Any:
        capability = find(self.name)
        numeric = capability.numeric
        assert numeric is not None
        try:
            return numeric(xp, *args, **attributes)
        except RelativisticPyError:
            raise
        except ImportError as error:
            raise missing(self.name, capability, error) from None

    def takes(self, count: int) -> bool:
        """Whether it computes a call of ``count`` arguments."""
        numeric = find(self.name).numeric
        if numeric is None:
            return False
        try:
            inspect.signature(numeric).bind(None, *([None] * count))
        except TypeError:
            return False
        return True


def missing(name: str, capability: Capability, error: ImportError) -> EvaluationError:
    """The error for a library a capability needs that is not installed, which says how to
    install it: the library it declares it requires, or else the module it imported."""
    library = capability.requires or (error.name or "").split(".")[0] or "a library"
    extra = extra_for(library)
    install = f'pip install "relativisticpy[{extra}]"' if extra else f"pip install {library}"
    return EvaluationError(f"{name} needs {library}, which is not installed", hint=install)


def extra_for(library: str) -> str | None:
    """The smallest extra of RelativisticPy that installs ``library``, as ``numeric`` for
    SciPy; ``None`` where none does."""
    try:
        requirements = importlib.metadata.requires("relativisticpy") or []
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - a source tree
        return None
    extras: dict[str, set[str]] = {}
    for requirement in requirements:
        named = re.match(r"[A-Za-z0-9._-]+", requirement)
        extra = re.search(r"""extra\s*==\s*["']([^"']+)["']""", requirement)
        if named is not None and extra is not None:
            extras.setdefault(extra.group(1), set()).add(_normal(named.group(0)))
    wanted = _normal(library)
    holding = [extra for extra, libraries in extras.items() if wanted in libraries]
    return min(holding, key=lambda extra: (len(extras[extra]), extra)) if holding else None


def _normal(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def wrong_kind(
    name: str, capability: Capability, args: Sequence[Any]
) -> RelativisticPyError | None:
    """The error for a function given what it cannot take, as ``sin([1, 2])``: a list, a
    matrix or a tensor where it takes numbers and expressions; ``None`` for any other
    failure, or another kind of capability, which takes what it declares."""
    if not isinstance(capability, Function):
        return None
    for value in args:
        kind = _kind_of(value)
        if kind is None:
            continue
        hint = None
        if capability.lifts:
            hint = (
                f"{name} applies to each component of a tensor: write the entries as one, "
                f"as in v_a := [1, 2], then {name}(v_a)"
            )
        return EvaluationError(f"{name} takes {capability.takes}, not {kind}", hint=hint)
    return None


def _kind_of(value: Any) -> str | None:
    """What a value that is no number or expression is, as an error names it."""
    from relativisticpy.tensors import Tensor
    from relativisticpy.tensors.equation import TensorEquation

    kinds: Mapping[type, str] = {
        list: "a list",
        sympy.NDimArray: "an array",
        sympy.MatrixBase: "a matrix",
        Tensor: "a tensor",
        TensorEquation: "an equation between tensors",
    }
    return next((said for kind, said in kinds.items() if isinstance(value, kind)), None)
