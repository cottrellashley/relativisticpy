r"""The conformance suite: what every capability must do, checked, and said in words that
say what to fix.

There are seven checks, each a function that takes a capability and gives its failures,
none where it passes:

1. **declaration**: names that are the capability's alone, a summary, two examples with
   their sources, the library it needs declared, and written forms that read as it;
2. **examples**: each example runs in the language and gives its known answer;
3. **agreement**: the exact and numeric implementations agree at points, with NumPy and
   with JAX;
4. **lifting**: applied to a tensor, or to data, it gives its results component by
   component, over the same spaces;
5. **errors**: the wrong number of arguments, the wrong kind of argument and a library
   that is missing are each refused, saying what to do;
6. **display**: a call prints in every form, and each form reads back;
7. **reference**: the language reference describes it, from its declaration.

``tests/integrations`` runs them for every capability the integrations declare;
:func:`check` runs them for one, as an agent checks a capability it writes before it
joins an integration. A check that does not apply to a kind, as agreement does not to a
solver, is not run for it (:func:`applies`).
"""

from __future__ import annotations

import contextlib
import functools
import importlib.metadata
import json
import math
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import sympy

from relativisticpy.errors import InternalError, RelativisticPyError
from relativisticpy.integrations._declarations import Capability, Example, Function
from relativisticpy.integrations._discovery import (
    Exactly,
    Numerically,
    catalogue,
    extra_for,
    installed,
)

#: The points a function is checked at when it declares none, each argument the same.
DEFAULT_POINTS = (0.3, 0.7, 1.6)

#: How far the exact and the numeric value may differ, relative to their size.
AGREEMENT = 1e-9


@dataclass(frozen=True)
class Check:
    """One part of the suite: its ``name``, what it holds a capability to, the kinds of
    capability it ``applies`` to, and the function that ``runs`` it."""

    name: str
    what: str
    applies: Callable[[Capability], bool]
    runs: Callable[[Capability], list[str]]


def check(capability: Capability, names: Sequence[str] | None = None) -> dict[str, list[str]]:
    """The failures of ``capability`` in each check that applies to it, or in those
    ``names``: an empty list for each it passes."""
    return {
        each.name: each.runs(capability)
        for each in CHECKS
        if (names is None or each.name in names) and each.applies(capability)
    }


def applies(name: str, capability: Capability) -> bool:
    """Whether the check called ``name`` applies to ``capability``."""
    return next(each for each in CHECKS if each.name == name).applies(capability)


@contextlib.contextmanager
def usable(capability: Capability) -> Iterator[None]:
    """The language with ``capability`` in it, while the block runs: as it is, if an
    integration declares it, and installed for the block otherwise."""
    if catalogue().get(capability.name) is capability:
        yield
    else:
        with installed(capability):
            yield


# -- 1. declaration ----------------------------------------------------------------------


def declaration(capability: Capability) -> list[str]:
    """Names of its own, a summary, examples with sources, the library it needs declared
    and imported only when used, and the plain text it is called with."""
    from relativisticpy.compiler.builtins import CONSTANTS
    from relativisticpy.reading.names import _DECLARATIONS

    name, failures = capability.name, []
    for written in capability.names:
        if not _one_name(written):
            failures.append(
                f"{name}: {written!r} is not one name in the language; a name is letters, "
                "digits and underscores, starting with a letter, as besselj or erfc"
            )
        other = catalogue().get(written)
        if other is not None and other is not capability:
            failures.append(
                f"{name}: {written!r} is already declared by {other.module}; choose another "
                "name, or extend that capability"
            )
        if written in CONSTANTS or written in _DECLARATIONS:
            failures.append(
                f"{name}: {written!r} is a name the language gives a meaning of its own; "
                "choose another name"
            )
    summary = capability.summary.strip()
    if not summary or "\n" in summary or summary.endswith("."):
        failures.append(
            f"{name}: give a summary, one phrase without a full stop, as the reference "
            f'lists it: summary="the Bessel function of the first kind"'
        )
    if len(capability.examples) < 2:
        failures.append(
            f"{name} has {len(capability.examples)} example(s): give at least two, each a "
            "program with its answer and where the answer comes from, as "
            'Example("J_0(0)", 1, "DLMF 10.8.1")'
        )
    for example in capability.examples:
        if not isinstance(example, Example) or not example.source.strip():
            failures.append(
                f"{name}: the example {_shown(example)} says not where its answer comes from: "
                'give its source, a book and equation, "DLMF 10.8.1", or the library called '
                'directly, "scipy.special.jv(0, 1.5)"'
            )
    failures += _dependencies(capability)
    failures += _forms(capability)
    return failures


def _one_name(written: str) -> bool:
    from relativisticpy.language import parse
    from relativisticpy.language import syntax as ast

    try:
        statements = parse(written).statements
    except RelativisticPyError:
        return False
    expr = _expression(statements)
    return isinstance(expr, ast.Name) and expr.id == written


def _expression(statements: Sequence[Any]) -> Any:
    """The expression of a program of one statement that evaluates one; ``None`` else."""
    from relativisticpy.language import syntax as ast

    if len(statements) == 1 and isinstance(statements[0], ast.Evaluate):
        return statements[0].expr
    return None


def _dependencies(capability: Capability) -> list[str]:
    """The library a capability needs beyond SymPy: declared, installed by an extra, and
    imported only when the capability is used, so that the language runs without it."""
    import numpy as np

    name, failures = capability.name, []
    requires = capability.requires
    if requires is not None and extra_for(requires) is None:
        failures.append(
            f"{name} requires {requires}, which no extra of RelativisticPy installs: add it "
            "to one in pyproject.toml, or require a library that one does"
        )
    declared = _imported_as(requires) if requires is not None else set()
    module = sys.modules.get(capability.module)
    for value in vars(module).values() if module is not None else ():
        owner = getattr(value, "__name__", "") if isinstance(value, type(sys)) else ""
        library = (owner or getattr(value, "__module__", None) or "").split(".")[0]
        if library in _optional():
            failures.append(
                f"{name}: its module imports {library} when it is imported; import it inside "
                f'exact or numeric, and declare requires="{_distribution(library)}", so that '
                "the language runs without it"
            )
            break
    point = _points(capability)[:1]
    calls: list[tuple[str, Callable[[], Any]]] = []
    if point:
        calls.append(("exact", lambda: capability.exact(*point[0])))
    numeric = capability.numeric
    if point and numeric is not None:
        calls.append(("numeric", lambda: numeric(np, *map(np.asarray, point[0]))))
    with _without(*(library for library in _optional() if library not in declared)):
        for method, call in calls:
            try:
                call()
            except ImportError as error:
                library = (error.name or "").split(".")[0]
                failures.append(
                    f"{name}: {method} imports {library}, which is optional: declare "
                    f'requires="{_distribution(library)}", so that a missing one is said'
                )
            except Exception:
                pass  # only a missing library is this check's
    return failures


def _forms(capability: Capability) -> list[str]:
    """The plain text a capability is called with reads as a call of it."""
    from relativisticpy.language import parse
    from relativisticpy.language import syntax as ast

    text = capability.text().replace(", ...)", ")")
    try:
        expr = _expression(parse(text).statements)
    except RelativisticPyError as error:
        return [f"{capability.name}: its call, {text}, does not parse: {error.message}"]
    if not (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name)):
        return [
            f"{capability.name}: its call, {text}, reads as something else; name its "
            "parameters after letters or words, as def exact(self, nu, x)"
        ]
    return []


# -- 2. examples -------------------------------------------------------------------------


def examples(capability: Capability) -> list[str]:
    """Each example runs in the language, uses the capability, and gives its answer."""
    from relativisticpy import Workbook

    failures = []
    with usable(capability):
        for example in capability.examples:
            shown = _shown(example)
            try:
                with Workbook() as wb:
                    used = _uses(wb.ir(example.program), capability)
                    values = wb.exe(example.program)
            except RelativisticPyError as error:
                failures.append(
                    f"{capability.name}'s example {shown} fails: {error.kind}: "
                    f"{error.message}; fix the program, or what it needs"
                )
                continue
            if not used:
                failures.append(
                    f"{capability.name}'s example {shown} does not use {capability.name}: "
                    f"write a program that calls it, as {capability.text()}"
                )
            if not values:
                failures.append(f"{capability.name}'s example {shown} shows nothing")
                continue
            if not _same(values[-1], example.answer, example.tolerance):
                failures.append(
                    f"{capability.name}'s example {shown} gives {_value(values[-1])}, not "
                    f"{_value(example.answer)}{_from(example)}: fix exact, or the example's "
                    "answer"
                )
    return failures


def _uses(ir: str, capability: Capability) -> bool:
    return any(f"call {name}" in ir or f"map {name}" in ir for name in capability.names)


# -- 3. agreement ------------------------------------------------------------------------


def agreement(capability: Capability) -> list[str]:
    """Exact and numeric agree at each point, numeric takes arrays of the points at once,
    and JAX agrees with NumPy where it is installed."""
    import numpy as np

    name, failures = capability.name, []
    points = _points(capability)
    computed = []
    for point in points:
        try:
            exact = complex(sympy.N(_exactly(capability, point), 30))
        except (TypeError, ValueError) as error:
            failures.append(
                f"{name}: exact at {_at(capability, point)} is no number ({error}); declare "
                "points where it is one, as points=[(0, 1.5), (1, 2.5)]"
            )
            continue
        numeric = complex(np.asarray(_numerically(capability, point)).item())
        computed.append(numeric)
        if not _close(numeric, exact):
            failures.append(
                f"{name}: numeric and exact differ at {_at(capability, point)}: numeric gives "
                f"{_number(numeric)}, exact {_number(exact)}; fix numeric, or, if the point is "
                "outside where it is defined, declare points=[...] inside"
            )
    if failures or not points:
        return failures
    columns = [np.asarray([point[k] for point in points]) for k in range(len(points[0]))]
    together = np.asarray(_numerically(capability, tuple(columns)))
    if together.shape != (len(points),) or not all(
        _close(complex(a), b) for a, b in zip(together.ravel(), computed, strict=False)
    ):
        failures.append(
            f"{name}: numeric given arrays of the points at once does not give an array of "
            "its values at each; compute with xp's functions, which take arrays"
        )
    failures += _jax(capability, columns, together)
    return failures


def _jax(capability: Capability, columns: list[Any], expected: Any) -> list[str]:
    try:
        import jax.numpy as jnp
    except ImportError:  # pragma: no cover - JAX is a development dependency
        return []
    import numpy as np

    try:
        found = np.asarray(_numerically(capability, tuple(map(jnp.asarray, columns)), jnp))
    except Exception as error:
        return [
            f"{capability.name}: numeric fails with JAX arrays ({type(error).__name__}: "
            f"{error}); compute with xp, the array module it is given"
        ]
    if found.shape != expected.shape or not np.allclose(found, expected, rtol=1e-5):
        return [f"{capability.name}: numeric gives other values with JAX than with NumPy"]
    return []


# -- 4. lifting --------------------------------------------------------------------------


def lifting(capability: Capability) -> list[str]:
    """Applied to a tensor, and to data, it gives its results component by component,
    over the space of the tensor."""
    from relativisticpy import Workbook
    from relativisticpy import display as show
    from relativisticpy.tensors import Tensor

    name, failures = capability.name, []
    points = _points(capability)
    lifted = [point[0] for point in points]
    rest = list(points[0][1:])
    expected = [_exactly(capability, (value, *rest)) for value in lifted]
    after = "".join(f", {show.text(value)}" for value in rest)
    space = f"\\mathcal{{V}} := \\mathbb{{R}}^{{{len(lifted)}}}\nl \\in \\mathcal{{V}}\n"
    components = ", ".join(show.text(value) for value in lifted)
    program = f"{space}v^{{l}} := [{components}]\n{name}(v^{{l}}{after})"
    with usable(capability):
        try:
            with Workbook() as wb:
                result = wb.exe(program)[-1]
        except RelativisticPyError as error:
            return [f"{name} applied to a tensor fails: {error.kind}: {error.message}"]
        failures += _components(capability, result, expected, "a tensor", "V")
        if not all(_numeric_value(value) for value in lifted):
            return failures
        import numpy as np

        try:
            with Workbook() as wb:
                wb["v^l"] = np.asarray([complex(v).real for v in lifted])
                result = wb.expr(f"{name}(v^{{l}}{after})")
                over = wb["v"].spaces[0].name
        except RelativisticPyError as error:
            return [*failures, f"{name} applied to data fails: {error.kind}: {error.message}"]
        if not isinstance(result, Tensor):
            return [*failures, f"{name} applied to data gives no tensor: {_value(result)}"]
        failures += _components(capability, result, expected, "data", over)
    return failures


def _components(
    capability: Capability, result: Any, expected: list[Any], what: str, space: str
) -> list[str]:
    from relativisticpy.tensors import Tensor

    name = capability.name
    if not isinstance(result, Tensor):
        return [
            f"{name} applied to {what} gives {_value(result)}, not a tensor of its values at "
            "each component; declare it lifts=True, as a function of numbers does"
        ]
    if [s.name for s in result.spaces or ()] != [space]:
        return [f"{name} applied to {what} loses its space: its result is over {result.spaces}"]
    values = (
        list(sympy.flatten(sympy.Array(result.to_numpy()).tolist()))
        if _data(result)
        else [result.components[k] for k in range(len(expected))]
    )
    if len(values) != len(expected) or not all(
        _same(a, b, 1e-9) for a, b in zip(values, expected, strict=False)
    ):
        shown = ", ".join(_value(_evaluated(v)) for v in values)
        wanted = ", ".join(_value(_evaluated(v)) for v in expected)
        return [
            f"{name} applied to {what} gives [{shown}], not its value at each component, "
            f"[{wanted}]; {_FROM_DATA if what == 'data' else _FROM_TENSOR}"
        ]
    return []


def _data(tensor: Any) -> bool:
    from relativisticpy.tensors.numerical import is_numerical

    return bool(is_numerical(tensor.components))


# -- 5. errors ---------------------------------------------------------------------------


def errors(capability: Capability) -> list[str]:
    """The wrong number of arguments and a missing library are each refused, saying what to
    do, and the wrong kind of argument, a list given to a function or text in quotes to
    anything else, is computed or refused so, never a failure of RelativisticPy's own."""
    name, failures = capability.name, []
    low, high = capability.arity
    counts = ([low - 1] if low > 0 else []) + ([high + 1] if high is not None else [])
    with usable(capability):
        for count in counts:
            program = f"{name}({', '.join(['1'] * count)})"
            error = _raised(program)
            if error is None or isinstance(error, InternalError) or "argument" not in error.message:
                failures.append(
                    f"{program} is not refused as the wrong number of arguments "
                    f"({_said(error)}); declare the parameters in exact's signature"
                )
        wrong = "[1, 2]" if isinstance(capability, Function) else "'text'"
        wrong = "1" if capability.takes_names else wrong
        rest = (_points(capability) or [(1,) * low])[0][1:]
        program = f"{name}({', '.join([wrong, *map(_written_value, rest)])})"
        error = _raised(program)
        if isinstance(error, InternalError):
            failures.append(
                f"{program} fails inside {name}, as a failure of RelativisticPy's own "
                f"({'; '.join(error.notes)}): refuse what it cannot take with an "
                f"EvaluationError in exact that says what {name} takes"
            )
    if capability.requires is not None:
        failures += _missing(capability)
    return failures


def _missing(capability: Capability) -> list[str]:
    """A capability whose library is not installed says what to install."""
    name, requires = capability.name, capability.requires or ""
    point = _points(capability)[:1]
    call = _numerically if capability.numeric is not None else _exactly
    with _without(*_imported_as(requires)):
        try:
            call(capability, point[0] if point else ())
        except RelativisticPyError as error:
            if "pip install" in (error.hint or "") and "not installed" in error.message:
                return []
            return [f"{name} without {requires} says {error.message!r}, not what to install"]
        except ImportError:
            return [
                f"{name} without {requires} fails with ImportError: import {requires} inside "
                "exact or numeric, which the integrations turn into an error that says what "
                "to install"
            ]
        except Exception:
            pass  # only a missing library is this check's
    return [f"{name} computes without {requires}: does it need it? declare requires only then"]


def _raised(program: str) -> RelativisticPyError | None:
    from relativisticpy import Workbook

    try:
        with Workbook() as wb:
            wb.exe(program)
    except RelativisticPyError as error:
        return error
    return None


def _said(error: RelativisticPyError | None) -> str:
    if error is None:
        return "it computes"
    return f"it says {error.kind}: {error.message}"


# -- 6. display --------------------------------------------------------------------------


def display(capability: Capability) -> list[str]:
    """A call prints in the four forms, text, Unicode, LaTeX and data, and each of the
    first three reads back as the same value."""
    from relativisticpy import Workbook
    from relativisticpy import display as show

    name, failures = capability.name, []
    value = _call(capability)
    with usable(capability):
        for form in ("text", "unicode", "latex"):
            shown = getattr(show, form)(value)
            try:
                with Workbook() as wb:
                    back = wb.expr(shown)
            except RelativisticPyError as error:
                failures.append(
                    f"{name}: {_value(value)} prints in {form} as {shown}, which does not read "
                    f"back: {error.message}; declare a {form} form the language reads"
                )
                continue
            if not _same(back, value, 1e-12):
                failures.append(
                    f"{name}: {_value(value)} prints in {form} as {shown}, which reads back as "
                    f"{_read_as(back)}; declare a {form} form the language reads as {name}"
                )
        described = show.data(value)
        try:
            json.dumps(described)
        except TypeError as error:
            failures.append(f"{name}: the data form of {_value(value)} is not JSON: {error}")
    return failures


def _call(capability: Capability) -> Any:
    """A call of a capability at symbols named after its parameters, but for a name the
    language gives a value, as e, or at its first point where it has no value there."""
    from relativisticpy.compiler.builtins import CONSTANTS

    low, _ = capability.arity
    symbols = [
        sympy.Symbol(f"{parameter}_1" if parameter in CONSTANTS else parameter)
        for parameter in capability.parameters[:low]
    ]
    try:
        return capability.exact(*symbols)
    except Exception:
        return _exactly(capability, _points(capability)[0])


# -- 7. reference ------------------------------------------------------------------------


def reference(capability: Capability) -> list[str]:
    """The language reference describes it: how it is called, its other names, and its
    summary."""
    from relativisticpy.reference import markdown

    try:
        with usable(capability):
            page = markdown()
    except ValueError as error:
        return [
            f"the reference cannot be written with {capability.name}: {error}; write its "
            "summary without a bar, |, which a table of the reference cannot hold"
        ]
    wanted = [f"`{capability.text()}`", *(f"`{alias}`" for alias in capability.aliases)]
    wanted.append(capability.summary)
    missing = [item for item in wanted if item not in page]
    if not missing:
        return []
    return [
        f"the reference does not describe {capability.name}: it lacks {', '.join(missing)}; "
        "write it again with uv run python -m relativisticpy.reference --write docs/reference.md"
    ]


def _function(capability: Capability) -> bool:
    return isinstance(capability, Function)


CHECKS: tuple[Check, ...] = (
    Check("declaration", "names, a summary, examples, its library, its forms",
          lambda capability: True, declaration),
    Check("examples", "each example gives its known answer", lambda capability: True, examples),
    Check("agreement", "exact and numeric agree at points",
          lambda capability: _function(capability) and capability.numeric is not None,
          agreement),
    Check("lifting", "applied to a tensor or data, component by component",
          lambda capability: _function(capability) and capability.lifts, lifting),
    Check("errors", "wrong arguments and a missing library are refused, saying what to do",
          lambda capability: True, errors),
    Check("display", "a call prints in every form, and each reads back",
          lambda capability: _function(capability) and capability.scalar, display),
    Check("reference", "the reference describes it", lambda capability: True, reference),
)  # fmt: skip


# -- what the checks share -------------------------------------------------------------------


def _points(capability: Capability) -> list[tuple[Any, ...]]:
    """The arguments a capability is checked at: those it declares, or else each of the
    default points for every argument it needs."""
    if capability.points:
        return [tuple(point) for point in capability.points]
    low, _ = capability.arity
    return [(value,) * low for value in DEFAULT_POINTS] if low else []


def _exactly(capability: Capability, point: tuple[Any, ...]) -> Any:
    with usable(capability):
        return Exactly(capability.name)(*point)


def _numerically(capability: Capability, point: tuple[Any, ...], xp: Any = None) -> Any:
    import numpy as np

    with usable(capability):
        arrays = [np.asarray(value, dtype=float) if xp is None else value for value in point]
        return Numerically(capability.name)(np if xp is None else xp, *arrays)


def _close(numeric: complex, exact: complex) -> bool:
    if not (math.isfinite(numeric.real) and math.isfinite(numeric.imag)):
        return False
    return abs(numeric - exact) <= AGREEMENT * max(1.0, abs(exact))


def _numeric_value(value: Any) -> bool:
    try:
        return bool(sympy.sympify(value).is_number)
    except (sympy.SympifyError, TypeError):
        return False


def _at(capability: Capability, point: tuple[Any, ...]) -> str:
    names = capability.parameters
    return "(" + ", ".join(f"{n}={v}" for n, v in zip(names, point, strict=False)) + ")"


def _same(value: Any, answer: Any, tolerance: float) -> bool:
    """Whether the language's value is the answer: a truth value as an equation that holds
    or not; text as text; lists, arrays and matrices item by item; numbers to
    ``tolerance``, relatively; and expressions where their difference simplifies to 0."""
    if isinstance(answer, bool) or answer is sympy.true or answer is sympy.false:
        return _holds(value) is bool(answer)
    if isinstance(answer, str) or isinstance(value, str):
        return bool(value == answer)
    mine, theirs = _items(value), _items(answer)
    if mine is not None or theirs is not None:
        return (
            mine is not None
            and theirs is not None
            and len(mine) == len(theirs)
            and all(_same(a, b, tolerance) for a, b in zip(mine, theirs, strict=True))
        )
    if isinstance(answer, sympy.Equality) and isinstance(value, sympy.Equality):
        return _same(value.lhs, answer.lhs, tolerance) and _same(value.rhs, answer.rhs, tolerance)
    try:
        value, answer = sympy.sympify(value), sympy.sympify(answer)
    except (sympy.SympifyError, TypeError):
        return bool(value == answer)
    if value == answer:
        return True
    if value.is_number and answer.is_number:
        a, b = complex(sympy.N(value, 30)), complex(sympy.N(answer, 30))
        return abs(a - b) <= tolerance * max(1.0, abs(b))
    try:
        return bool(sympy.simplify(value - answer) == 0)
    except (TypeError, AttributeError):
        return False


def _items(value: Any) -> list[Any] | None:
    from relativisticpy.tensors import Tensor

    if isinstance(value, Tensor):
        value = value.components
    if isinstance(value, list | tuple):
        return list(value)
    if isinstance(value, sympy.NDimArray | sympy.MatrixBase):
        return list(value.tolist())
    return None


def _holds(value: Any) -> bool | None:
    holds = getattr(value, "holds", None)
    if callable(holds):
        result = holds()
        return None if result is None else bool(result)
    if value is sympy.true or value is sympy.false or isinstance(value, bool):
        return bool(value)
    return None


def _shown(example: Any) -> str:
    program = getattr(example, "program", example)
    return " ".join(str(program).split())


_FROM_DATA = (
    "data is computed by numeric where there is one, and by exact translated into NumPy "
    "otherwise: fix the one that differs"
)
_FROM_TENSOR = "each component is computed by exact: fix exact, or declare lifts=False"


def _from(example: Example) -> str:
    return f" ({example.source})" if example.source.strip() else ""


def _number(value: complex) -> str:
    return f"{value.real:.12g}" if value.imag == 0 else f"{value:.12g}"


def _evaluated(value: Any) -> Any:
    """A number as a decimal, so that a message shows its value."""
    try:
        return sympy.N(value, 12) if sympy.sympify(value).is_number else value
    except (sympy.SympifyError, TypeError):
        return value


def _read_as(value: Any) -> str:
    """What a value read back is, said so that it is told apart from what was printed."""
    from sympy.core.function import AppliedUndef

    unknown = sorted(
        {f.func.__name__ for f in getattr(value, "atoms", lambda *_: set())(AppliedUndef)}
    )
    if unknown:
        named = ", ".join(unknown)
        return f"{_value(value)}, in which {named} is a function the language does not know"
    return _value(value)


def _value(value: Any) -> str:
    from relativisticpy import display as show

    try:
        return show.unicode(value)
    except Exception:
        return repr(value)


def _written_value(value: Any) -> str:
    from relativisticpy import display as show

    return show.text(value)


@functools.cache
def _optional() -> tuple[str, ...]:
    """The libraries an extra of RelativisticPy installs, by the names they are imported
    by, but NumPy, which the engines compute numbers with."""
    try:
        requirements = importlib.metadata.requires("relativisticpy") or []
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - a source tree
        return ()
    names: set[str] = set()
    for requirement in requirements:
        if "extra ==" in requirement:
            distribution = requirement.split(";")[0].split(">")[0].split("=")[0].strip()
            names.update(_imported_as(distribution))
    return tuple(sorted(names - {"numpy"}))


@functools.cache
def _imported_as(distribution: str) -> frozenset[str]:
    """The names a library is imported by, as scikit-learn is by sklearn."""
    names = {distribution.replace("-", "_")}
    for module, distributions in importlib.metadata.packages_distributions().items():
        if any(d.lower() == distribution.lower() for d in distributions):
            names.add(module)
    return frozenset(names)


@functools.cache
def _distribution(module: str) -> str:
    """The library, as pip installs it, that a module is imported from."""
    found = importlib.metadata.packages_distributions().get(module)
    return found[0] if found else module


@contextlib.contextmanager
def _without(*libraries: str) -> Iterator[None]:
    """As if ``libraries`` were not installed: importing one of them, or what is in it,
    fails while the block runs."""
    hidden = {
        name: module
        for name, module in sys.modules.items()
        if any(name == library or name.startswith(library + ".") for library in libraries)
    }
    try:
        for name in [*hidden, *libraries]:
            sys.modules[name] = None  # type: ignore[assignment]
        yield
    finally:
        for name in libraries:
            if sys.modules.get(name, False) is None:
                del sys.modules[name]
        sys.modules.update(hidden)
