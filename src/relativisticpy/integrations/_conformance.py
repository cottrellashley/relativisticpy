r"""The conformance suite: what every capability must do, checked, and said in words that
say what to fix.

There are thirteen checks, each a function that takes a capability and gives its
failures, none where it passes:

1. **declaration**: names that are the capability's alone, a summary, two examples with
   their sources, the library it needs declared, and written forms that read as it;
2. **examples**: each example runs in the language and gives its known answer, in the
   form of the answer where the capability gives a form, as ``factor`` does;
3. **agreement**: the exact and numeric implementations agree at points, with NumPy and
   with JAX; of an optimizer, the optimizers of the other kind, exact or numeric, find the
   same optimum of the problems it is checked at; of a distribution, SymPy's density and
   distribution function are SciPy's;
4. **lifting**: applied to a tensor, or to data, it gives its results component by
   component, over the same spaces;
5. **errors**: the wrong number of arguments, the wrong kind of argument and a library
   that is missing are each refused, saying what to do;
6. **display**: a call prints in every form, and each form reads back;
7. **reference**: the language reference indexes it, and its help describes it in full,
   both from its declaration;
8. **transform**: a transform agrees with the integral or the sum it stands for, its inverse
   undoes it, and, of data, it keeps Parseval's relation and gives data over a space of the
   size and the labels it declares;
9. **rewrite**: a rewrite gives the numbers of what it rewrites, computed as written;
10. **feasibility**: the optimum an optimizer reports keeps every constraint of its problem;
11. **optimality**: the conditions of optimality hold at it, stationarity or the KKT
    conditions, or, over finitely many points, no other point is better;
12. **density**: a distribution's density integrates, or sums, to 1;
13. **moments**: a distribution's mean and variance, by SymPy and by SciPy, are those it
    declares.

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

from relativisticpy.errors import InternalError, RelativisticPyError, translate
from relativisticpy.integrations import _distributions, _objects, _optimizers, _transforms
from relativisticpy.integrations._declarations import (
    Capability,
    Example,
    Function,
    MapFunction,
    near,
)
from relativisticpy.integrations._discovery import (
    Exactly,
    Numerically,
    catalogue,
    extra_for,
    installed,
)
from relativisticpy.integrations._objects import ObjectConversion

#: The points a function is checked at when it declares none, each argument the same.
DEFAULT_POINTS = (0.3, 0.7, 1.6)

#: How far the exact and the numeric value may differ, relative to their size.
AGREEMENT = 1e-9

#: How far beyond its points a function is checked on data: where a degree or an order is
#: large enough for a series summed in double precision to have lost its digits.
FAR = 40


@dataclass(frozen=True)
class Check:
    """One part of the suite: its ``name``, what it holds a capability to, the kinds of
    capability it ``applies`` to, and the function that ``runs`` it."""

    name: str
    what: str
    applies: Callable[[Capability], bool]
    runs: Callable[[Capability], list[str]]


def check(
    capability: Capability | ObjectConversion, names: Sequence[str] | None = None
) -> dict[str, list[str]]:
    """The failures of ``capability`` in each check that applies to it, or in those
    ``names``: an empty list for each it passes. An object integration is held to the
    checks of its kind (:data:`relativisticpy.integrations._objects.CHECKS`)."""
    if isinstance(capability, ObjectConversion):
        return _objects.check(capability, names)
    return {
        each.name: each.runs(capability)
        for each in CHECKS
        if (names is None or each.name in names) and each.applies(capability)
    }


def applies(name: str, capability: Capability | ObjectConversion) -> bool:
    """Whether the check called ``name`` applies to ``capability``."""
    if isinstance(capability, ObjectConversion):
        return name in _objects.CHECKS
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
            f"{name}: give a summary, one phrase without a full stop, with which its help "
            f'begins: summary="the Bessel function of the first kind"'
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
    failures += _written(capability)
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


def _written(capability: Capability) -> list[str]:
    """The plain text a capability is called with reads as a call of it; each written form
    parses, by the language's own parser, into a call written with the parameters of
    exact, under a name that is no other capability's, and reads as this capability and
    no other."""
    from relativisticpy.integrations import _forms
    from relativisticpy.language import parse
    from relativisticpy.language import syntax as ast

    name, failures = capability.name, []
    if isinstance(capability, _transforms.Rewrite):
        with usable(capability):
            return _transforms.sum_failures(capability)
    if isinstance(capability, _optimizers.Optimizer):
        return []  # never called: an optimum, as books write it, is found by the optimizers
    if isinstance(capability, _distributions.Distribution):  # written after \sim
        with usable(capability):
            return _distributions.form_failures(capability)
    text = capability.text().replace(", ...)", ")")
    try:
        expr = _expression(parse(text).statements)
    except RelativisticPyError as error:
        return [f"{name}: its call, {text}, does not parse: {error.message}"]
    if not (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name)):
        failures.append(
            f"{name}: its call, {text}, reads as something else; name its parameters after "
            "letters or words, as def exact(self, nu, x)"
        )
    parameters = {*capability.parameters, *capability.keywords}
    unknown = [each for each in capability.integers if each not in parameters]
    if unknown:
        failures.append(
            f"{name}: it reads {', '.join(unknown)} as integers, which exact takes no parameter "
            "of; name the parameters of exact that are integers, as integers=['n']"
        )
    low, _ = capability.arity
    with usable(capability):
        for form, written in (("latex", capability.latex), ("unicode", capability.unicode)):
            for each in written:
                if isinstance(capability, _transforms.Transform) and _transforms.form_head(each):
                    failures += _transforms.form_failures(capability, each)  # \mathcal{L}\{f\}
                    continue
                pattern = _forms.parse_form(capability, each, form)  # type: ignore[arg-type]
                if pattern is None:
                    failures.append(
                        f"{name}: the {form} form {each} does not parse as a call, as "
                        r"J_{\nu}(x) or \operatorname{erfc}(x) does; write it as books do"
                    )
                    continue
                named = catalogue().get(pattern.shape.head)
                if named is not None and named is not capability:
                    failures.append(
                        f"{name}: the form {each} is written with {pattern.shape.head}, a name "
                        f"of {named.name}, so that a call of {named.name} would read as {name}; "
                        "write it with a name of its own"
                    )
                strange = [v for v in pattern.variables() if v not in parameters]
                if strange:
                    failures.append(
                        f"{name}: the form {each} writes {', '.join(strange)}, which "
                        f"{'is' if len(strange) == 1 else 'are'} no parameter of exact; "
                        "name the variables of a form as exact names its parameters, as "
                        f"{capability.text()}"
                    )
                needed = [p for p in capability.parameters[:low] if p not in pattern.variables()]
                if needed:
                    failures.append(
                        f"{name}: the form {each} leaves out {', '.join(needed)}, which exact "
                        "needs; write every parameter in it"
                    )
                found = _forms.matching(pattern.shape)
                if found is None or found.pattern.owner is not capability:
                    other = getattr(found.pattern.owner, "name", "nothing") if found else "nothing"
                    failures.append(
                        f"{name}: the form {each} reads as {other}, not as {name}; write it so "
                        "that it is this capability's alone"
                    )
    return failures


# -- 2. examples -------------------------------------------------------------------------


def examples(capability: Capability) -> list[str]:
    """Each example runs in the language, uses the capability, and gives its answer."""
    from relativisticpy import Workbook

    failures = []
    with usable(capability):
        for example in capability.examples:
            shown = _shown(example)
            try:
                with Workbook() as wb, _optimizers.answering() as answers:
                    used = _uses(wb.ir(example.program), capability)
                    values = wb.exe(example.program)
                    if isinstance(capability, _optimizers.Optimizer):
                        used = any(found.by == capability.name for found in answers)
            except RelativisticPyError as error:
                failures.append(
                    f"{capability.name}'s example {shown} fails: {error.kind}: "
                    f"{error.message}; fix the program, or what it needs"
                )
                continue
            if not used and isinstance(capability, _optimizers.Optimizer):
                failures.append(
                    f"{capability.name}'s example {shown} is not found by {capability.name}: "
                    "write an optimum it takes, and the optimizers before it do not"
                )
            elif not used:
                failures.append(
                    f"{capability.name}'s example {shown} does not use {capability.name}: "
                    f"write a program that calls it, as {capability.text()}"
                )
            if not values:
                failures.append(f"{capability.name}'s example {shown} shows nothing")
                continue
            if capability.gives_form and not _alike(values[-1], example.answer):
                failures.append(
                    f"{capability.name}'s example {shown} gives {_value(values[-1])}, not "
                    f"{_value(example.answer)} as it is written{_from(example)}: it gives a "
                    "form, which is shown as it is computed; fix exact, or the example's answer"
                )
            elif not _same(values[-1], example.answer, example.tolerance):
                failures.append(
                    f"{capability.name}'s example {shown} gives {_value(values[-1])}, not "
                    f"{_value(example.answer)}{_from(example)}: fix exact, or the example's "
                    "answer"
                )
    return failures


def _alike(value: Any, answer: Any) -> bool:
    """Whether the language's value is the answer as it is written: the same expression,
    term for term and factor for factor, as SymPy holds both, and not only one equal to
    it, as x^2 - 1 is to (x - 1)(x + 1); lists item by item."""
    mine, theirs = _items(value), _items(answer)
    if mine is not None or theirs is not None:
        return (
            mine is not None
            and theirs is not None
            and len(mine) == len(theirs)
            and all(_alike(a, b) for a, b in zip(mine, theirs, strict=True))
        )
    try:
        return bool(sympy.sympify(value) == sympy.sympify(answer))
    except (sympy.SympifyError, TypeError):
        return bool(value == answer)


def _uses(ir: str, capability: Capability) -> bool:
    """Whether a program's IR calls the capability, or, of a rewrite, what computes it."""
    names = capability.names
    if isinstance(capability, _transforms.Rewrite):
        names = (capability.computes,)
    return any(f"call {name}" in ir or f"map {name}" in ir for name in names)


# -- 3. agreement ------------------------------------------------------------------------


def agreement(capability: Capability) -> list[str]:
    """Exact and numeric agree at each point, numeric takes arrays of the points at once,
    and JAX agrees with NumPy where it is installed; for a function of a matrix, entry by
    entry, at matrices."""
    import numpy as np

    if isinstance(capability, MapFunction):
        return _map_agreement(capability)
    if isinstance(capability, _optimizers.Optimizer):
        return _optimizers.agreement(capability)
    if isinstance(capability, _distributions.Distribution):
        return _distributions.agreement(capability)
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
        if not near(numeric, exact, AGREEMENT):
            failures.append(
                f"{name}: numeric and exact differ at {_at(capability, point)}: numeric gives "
                f"{_number(numeric)}, exact {_number(exact)}; fix numeric, or, if the point is "
                "outside where it is defined, declare points=[...] inside"
            )
    if failures or not points or not all(_scalar(value) for point in points for value in point):
        return failures  # lists, as the parameters of {}_pF_q, are checked a point at once
    columns = [np.asarray([point[k] for point in points]) for k in range(len(points[0]))]
    together = np.asarray(_numerically(capability, tuple(columns)))
    if together.shape != (len(points),) or not all(
        near(complex(a), b, AGREEMENT) for a, b in zip(together.ravel(), computed, strict=False)
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
    """Applied to a tensor, and to data, in the place of any of its arguments that is a
    number or an expression at its points, it gives its results component by component,
    over the space of the tensor, and so it does applied to data past its points, where it
    may be complex or not defined (:func:`_past`); a function of a matrix, applied to a
    map, acts on it."""
    if isinstance(capability, MapFunction):
        return _map_lifting(capability)
    points = _points(capability)
    places = [k for k in range(len(points[0])) if all(_scalar(point[k]) for point in points)]
    return [
        failure
        for place in places
        for failure in _lifted(capability, points, place) or _past(capability, points, place)
    ]


def _lifted(capability: Capability, points: list[tuple[Any, ...]], place: int) -> list[str]:
    """The failures of a capability applied to a tensor, and to data, as its argument at
    ``place``, the others those of its first point."""
    from relativisticpy import Workbook
    from relativisticpy import display as show
    from relativisticpy.tensors import Tensor

    name, failures = capability.name, []
    lifted = [point[place] for point in points]
    first = list(points[0])
    expected = [_exactly(capability, (*first[:place], v, *first[place + 1 :])) for v in lifted]
    written = [show.text(value) for value in first]
    written[place] = "v^{l}"
    call = f"{name}({', '.join(written)})"
    what = "a tensor" if place == 0 else f"a tensor, as its argument {place + 1},"
    space = f"\\mathcal{{V}} := \\mathbb{{R}}^{{{len(lifted)}}}\nl \\in \\mathcal{{V}}\n"
    components = ", ".join(show.text(value) for value in lifted)
    with usable(capability):
        try:
            with Workbook() as wb:
                result = wb.exe(f"{space}v^{{l}} := [{components}]\n{call}")[-1]
        except RelativisticPyError as error:
            return [f"{name} applied to {what} fails: {error.kind}: {error.message}"]
        failures += _components(capability, result, expected, what, "V")
        if not all(_numbers(value) for value in [*lifted, *first]):  # data is numbers
            return failures
        import numpy as np

        data = what.replace("a tensor", "data")
        try:
            with Workbook() as wb:
                wb["v^l"] = np.asarray([complex(v).real for v in lifted])
                result = wb.expr(call)
                over = wb["v"].spaces[0].name
        except RelativisticPyError as error:
            return [*failures, f"{name} applied to {data} fails: {error.kind}: {error.message}"]
        if not isinstance(result, Tensor):
            return [*failures, f"{name} applied to {data} gives no tensor: {_value(result)}"]
        failures += _components(capability, result, expected, data, over)
    return failures


def _past(capability: Capability, points: list[tuple[Any, ...]], place: int) -> list[str]:
    """The failures of a function applied to data past its points, as its argument at
    ``place``, the others those of each point: below 0, beyond the points, and far beyond
    them, where an order or a degree is large, between whole numbers and off the real line,
    where it may be complex, or not defined. It gives exact's value at each, complex where
    that is, or refuses where exact refuses; never nan, a real part alone, or a value where
    exact has none. The first that fails says what to fix."""
    for point in points:
        if not all(_numbers(value) for value in point):
            continue
        value = complex(point[place]).real
        if not math.isfinite(value):
            continue
        below = -value if value else -1.5
        real = [below, value + 2, -(value + 2), value + 0.5, -(value + 0.5), value + FAR]
        off = [complex(value, 0.75), complex(below, -0.75)]
        for values in (real, off):
            failures = _past_at(capability, point, place, list(dict.fromkeys(values)))
            if failures:
                return failures
        far = tuple(each + FAR if _scalar(each) else each for each in point)
        failures = _past_at(capability, far, place, [value + FAR])  # far in every argument
        if failures:
            return failures
    return []


def _past_at(
    capability: Capability, point: tuple[Any, ...], place: int, values: list[Any]
) -> list[str]:
    """The failures of a function applied to data of ``values``, as its argument at
    ``place``, the others those of ``point``, against exact's value at each."""
    import numpy as np

    name = capability.name
    computed, refused = [], []
    for value in values:
        at = (*point[:place], sympy.sympify(value), *point[place + 1 :])
        try:
            exact = complex(sympy.N(_exactly(capability, at), 30))
        except Exception as error:  # refused, as the language refuses it, or no number
            if not isinstance(translate(error), InternalError):
                refused.append(value)
            continue
        if math.isfinite(exact.real) and math.isfinite(exact.imag):
            computed.append((value, exact))
    if computed:
        found = _of_data(capability, point, place, [value for value, _ in computed])
        if isinstance(found, str):
            return [f"{name} applied to data past its points {found}"]
        for (value, exact), number in zip(computed, found, strict=True):
            if not near(complex(number), exact, AGREEMENT):
                return [
                    f"{name} applied to data past its points gives {_number(complex(number))} "
                    f"at {_at(capability, (*point[:place], value, *point[place + 1 :]))}, where "
                    f"exact gives {_number(exact)}: {_FROM_DATA}, and give a complex value where "
                    "exact has one, or compute there with exact"
                ]
    for value in refused:
        found = _of_data(capability, point, place, [value])
        if not isinstance(found, str) or "is refused" not in found:
            given = np.asarray(found).ravel().tolist() if not isinstance(found, str) else found
            return [
                f"{name} applied to data past its points is not refused at "
                f"{_at(capability, (*point[:place], value, *point[place + 1 :]))}, where exact "
                f"refuses: it gives {given}; refuse there in numeric as exact does"
            ]
    return []


def _of_data(capability: Capability, point: tuple[Any, ...], place: int, values: list[Any]) -> Any:
    """The values of a function applied to data of ``values`` as its argument at ``place``,
    the others those of ``point``; what is wrong, as text, where it gives none."""
    import numpy as np

    from relativisticpy import Workbook
    from relativisticpy import display as show

    written = [show.text(value) for value in point]
    written[place] = "v^{l}"
    call = f"{capability.name}({', '.join(written)})"
    with usable(capability):
        try:
            with Workbook() as wb:
                wb["v^l"] = np.asarray(values)
                result = wb.expr(call)
        except InternalError as error:
            return f"fails inside, at {values} ({'; '.join(error.notes)})"
        except RelativisticPyError as error:
            return f"is refused at {values}: {error.message}"
    if not _data(result):
        return f"gives no data at {values}: {_value(result)}"
    return np.asarray(result.to_numpy(), dtype=complex).ravel()


def _scalar(value: Any) -> bool:
    """Whether a value is a number or an expression, rather than a list of them."""
    return not isinstance(value, list | tuple | sympy.Tuple | sympy.NDimArray)


def _numbers(value: Any) -> bool:
    """Whether a value is a number, or a list of them, as the parameters of {}_pF_q are."""
    if isinstance(value, list | tuple):
        return all(_numbers(each) for each in value)
    return _numeric_value(value)


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
            f"[{wanted}]; {_FROM_DATA if what.startswith('data') else _FROM_TENSOR}"
        ]
    return []


def _data(tensor: Any) -> bool:
    from relativisticpy.tensors.numerical import is_numerical

    return bool(is_numerical(tensor.components))


# -- 3 and 4, of a function of a matrix ---------------------------------------------------


def _map_agreement(capability: Capability) -> list[str]:
    """Exact and numeric agree at each point, entry by entry, with NumPy and with JAX: a
    function of a matrix takes each matrix whole."""
    import numpy as np

    name, failures = capability.name, []
    for point in _points(capability):
        exact = _entries(_exactly(capability, point))
        if exact is None:
            failures.append(
                f"{name}: exact at {_at(capability, point)} is no number; declare points where "
                "it is one, as points=[([[2, 1], [1, 3]],)]"
            )
            continue
        numeric = np.asarray(_numerically(capability, point), dtype=complex)
        if numeric.shape != exact.shape or not _close_entries(numeric, exact, AGREEMENT):
            failures.append(
                f"{name}: numeric and exact differ at {_at(capability, point)}: numeric gives "
                f"{_shown_entries(numeric)}, exact {_shown_entries(exact)}; fix numeric, or, if "
                "the point is outside where it is defined, declare points=[...] inside"
            )
            continue
        failures += _map_jax(capability, point, numeric)
    return failures


def _map_jax(capability: Capability, point: tuple[Any, ...], expected: Any) -> list[str]:
    """JAX agrees with NumPy at a point, in 64 bits, as the language computes with JAX."""
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:  # pragma: no cover - JAX is a development dependency
        return []
    import numpy as np

    from relativisticpy.tensors.numerical import double_precision

    double_precision(jax)
    arrays = tuple(jnp.asarray(np.asarray(value, dtype=float)) for value in point)
    try:
        found = np.asarray(_numerically(capability, arrays, jnp), dtype=complex)
    except Exception as error:
        return [
            f"{capability.name}: numeric fails with JAX arrays ({type(error).__name__}: "
            f"{error}); compute with xp, the array module it is given"
        ]
    if found.shape != expected.shape or not _close_entries(found, expected, 1e-5):
        return [f"{capability.name}: numeric gives other values with JAX than with NumPy"]
    return []


def _map_lifting(capability: Capability) -> list[str]:
    """Applied to a map, A^{a}{}_{b} over a space, written or bound as data, a function of
    a matrix acts on its matrix of components: its value is the number exact gives, or a
    map over the same space."""
    import numpy as np

    from relativisticpy import Workbook
    from relativisticpy import display as show

    name = capability.name
    point = next((p for p in _points(capability) if _square_place(p) is not None), None)
    if point is None:
        return [
            f"{name} acts on a map, but none of its points is a square matrix: declare one, "
            "as points=[([[2, 1], [1, 3]],)], or lifts=False where its value is over other spaces"
        ]
    place = _square_place(point)
    assert place is not None
    matrix = point[place]
    expected = _exactly(capability, point)
    written = [show.text(value) for value in point]
    written[place] = "A^{a}{}_{b}"
    call = f"{name}({', '.join(written)})"
    space = f"\\mathcal{{V}} := \\mathbb{{R}}^{{{len(matrix)}}}\na, b \\in \\mathcal{{V}}"
    failures: list[str] = []
    with usable(capability):
        for what in ("a map", "data"):
            try:
                with Workbook() as wb:
                    wb.exe(space)
                    if what == "data":
                        wb["A^{a}{}_{b}"] = np.asarray(matrix, dtype=float)
                    else:
                        wb.exe(f"A^{{a}}{{}}_{{b}} := {show.text(matrix)}")
                    result = wb.expr(call)
            except RelativisticPyError as error:
                failures.append(f"{name} applied to {what} fails: {error.kind}: {error.message}")
                continue
            failures += _over_the_map(capability, result, expected, what)
    return failures


def _over_the_map(capability: Capability, result: Any, expected: Any, what: str) -> list[str]:
    """The failures of the value of a function of a matrix applied to a map: a number where
    it gives one, and otherwise a map over the same space, with exact's entries."""
    from relativisticpy.tensors import Tensor

    name = capability.name
    if not capability.scalar:
        if not isinstance(result, Tensor):
            return [f"{name} applied to {what} gives {_value(result)}, not a map over its space"]
        if [space.name for space in result.spaces or ()] != ["V", "V"]:
            return [f"{name} applied to {what} loses its space: its value is over {result.spaces}"]
        result = result.to_numpy() if _data(result) else result.components
    found, wanted = _entries(result), _entries(expected)
    if found is None or wanted is None or found.shape != wanted.shape:
        return [f"{name} applied to {what} gives {_value(result)}, not {_value(expected)}"]
    if not _close_entries(found, wanted, 1e-9):
        return [
            f"{name} applied to {what} gives {_shown_entries(found)}, not its value at the map's "
            f"matrix, {_shown_entries(wanted)}; {_FROM_DATA if what == 'data' else _FROM_TENSOR}"
        ]
    return []


def _square_place(point: tuple[Any, ...]) -> int | None:
    """Which argument of a point is a square matrix, written as a list of its rows."""
    for place, value in enumerate(point):
        rows = value if isinstance(value, list) else None
        if rows and all(isinstance(row, list) and len(row) == len(rows) for row in rows):
            return place
    return None


def _entries(value: Any) -> Any:
    """A value as an array of complex numbers: a number, or the entries of a matrix or an
    array; ``None`` where it holds no number."""
    import numpy as np

    arrays = (sympy.NDimArray, sympy.MatrixBase, np.ndarray)
    jax = hasattr(value, "tolist") and not isinstance(value, sympy.Basic)
    if isinstance(value, arrays) or jax:
        shape = tuple(np.shape(value)) if not isinstance(value, sympy.NDimArray) else value.shape
        flat = list(sympy.flatten(value.tolist())) if shape else [value.tolist()]
    elif isinstance(value, list | tuple):
        shape = tuple(np.shape(value))
        flat = list(sympy.flatten(value))
    else:
        shape, flat = (), [value]
    try:
        numbers = [complex(sympy.N(entry, 30)) for entry in flat]
    except (TypeError, ValueError):
        return None
    return np.asarray(numbers, dtype=complex).reshape(shape)


def _close_entries(found: Any, expected: Any, tolerance: float) -> bool:
    """Whether the entries found are near those expected, as :func:`near` compares numbers,
    relative to the largest of them."""
    import numpy as np

    if not np.all(np.isfinite(found)):
        return False
    scale = max(1.0, float(np.abs(expected).max(initial=0.0)))
    return bool(np.allclose(found, expected, rtol=tolerance, atol=tolerance * scale))


def _shown_entries(values: Any) -> str:
    """Entries as a failure shows them, on one line, real where they are."""
    import numpy as np

    shown = np.real_if_close(values, tol=1000)
    return np.array2string(np.round(shown, 12), separator=", ").replace("\n", "")


# -- 5. errors ---------------------------------------------------------------------------


def errors(capability: Capability) -> list[str]:
    """The wrong number of arguments and a missing library are each refused, saying what to
    do, and the wrong kind of argument, a list given to a function or text in quotes to
    anything else, is computed or refused so, never a failure of RelativisticPy's own."""
    name, failures = capability.name, []
    if isinstance(capability, _optimizers.Optimizer):  # never called: only its library
        return _missing(capability) if capability.requires is not None else []
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
                    f"back: {error.message}; {_to_read(capability, value, form)}"
                )
                continue
            if not _same(back, value, 1e-12):
                failures.append(
                    f"{name}: {_value(value)} prints in {form} as {shown}, which reads back as "
                    f"{_read_as(back)}; {_to_read(capability, value, form)}"
                )
        described = show.data(value)
        try:
            json.dumps(described)
        except TypeError as error:
            failures.append(f"{name}: the data form of {_value(value)} is not JSON: {error}")
    return failures


def _to_read(capability: Capability, value: Any, form: str) -> str:
    """What makes a call print as the language reads it back: in text, the name it prints
    with, called; in LaTeX and Unicode, a form of the capability's own."""
    if form == "text":
        printed = type(value).__name__ if isinstance(value, sympy.Function) else None
        if printed and printed != capability.name:
            return f"name the capability {printed}, as its calls print, or give it that alias"
        return f"make exact give a call that prints as {capability.text()}"
    example = r"J_{\nu}(x)" if form == "latex" else "J_ν(x)"
    return f'declare a {form} form the language reads as {capability.name}, as {form}="{example}"'


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
    """The language reference indexes it, by its names and the forms books write it in,
    and its help, ``relativisticpy help NAME``, describes it in full, from its summary."""
    from relativisticpy import reference as written

    with usable(capability):
        page, described = written.markdown(), written.capability(capability.name)
    named = (*capability.aliases, *capability.latex)
    if isinstance(capability, _transforms.Rewrite):  # never called: written, and computed
        named = (*named, capability.computes)
    wanted = [f"`{name}`" for name in named]
    if f"`{capability.name}`" not in page and f"`{capability.text()}`" not in page:
        wanted.insert(0, f"`{capability.name}`")
    missing = [item for item in wanted if item not in page]
    summary = capability.summary[:1].upper() + capability.summary[1:]
    if not described.startswith(f"# {capability.name}\n") or summary not in described:
        missing.append(f"its help, {summary}")
    if not missing:
        return []
    return [
        f"the reference does not describe {capability.name}: it lacks {', '.join(missing)}; "
        "write it again with uv run python -m relativisticpy.reference --write docs/reference.md"
    ]


def _function(capability: Capability) -> bool:
    return isinstance(capability, Function)


def _computed(capability: Capability) -> bool:
    """Whether a capability computes values of numbers, as a function and a function of a
    matrix do."""
    return isinstance(capability, Function | MapFunction)


CHECKS: tuple[Check, ...] = (
    Check("declaration", "names, a summary, examples, its library, its forms",
          lambda capability: True, declaration),
    Check("examples", "each example gives its known answer", lambda capability: True, examples),
    Check("agreement", "exact and numeric agree at points",
          lambda capability: (_computed(capability) and capability.numeric is not None)
          or isinstance(capability, _optimizers.Optimizer | _distributions.Distribution),
          agreement),
    Check("lifting", "applied to a tensor or data, component by component, or to a map",
          lambda capability: _computed(capability) and capability.lifts, lifting),
    Check("errors", "wrong arguments and a missing library are refused, saying what to do",
          lambda capability: not isinstance(capability, _transforms.Rewrite), errors),
    Check("display", "a call prints in every form, and each reads back",
          lambda capability: _function(capability) and capability.scalar, display),
    Check(
        "reference",
        "the reference indexes it, and its help describes it",
        lambda capability: True,
        reference,
    ),
    Check("transform", "against its integral or sum, its inverse, Parseval and its space",
          lambda capability: isinstance(capability, _transforms.Transform),
          _transforms.transform_checks),
    Check("rewrite", "the numbers of what it rewrites, as written",
          lambda capability: isinstance(capability, _transforms.Rewrite),
          _transforms.rewrite_checks),
    Check("feasibility", "the optimum it reports keeps every constraint",
          lambda capability: isinstance(capability, _optimizers.Optimizer),
          _optimizers.feasibility),
    Check("optimality", "stationarity or the KKT conditions hold at its optimum",
          lambda capability: isinstance(capability, _optimizers.Optimizer),
          _optimizers.optimality),
    Check("density", "the density integrates, or sums, to 1",
          lambda capability: isinstance(capability, _distributions.Distribution),
          _distributions.density_checks),
    Check("moments", "the mean and variance are those declared, by SymPy and by SciPy",
          lambda capability: isinstance(capability, _distributions.Distribution),
          _distributions.moment_checks),
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


def _numeric_value(value: Any) -> bool:
    if not _scalar(value):
        return False
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
        return near(complex(sympy.N(value, 30)), complex(sympy.N(answer, 30)), tolerance)
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
