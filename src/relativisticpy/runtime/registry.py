"""The registry of implementations: what computes each operation the IR names.

An operation of the IR that calls a function, :class:`~relativisticpy.ir.ops.Call` or
:class:`~relativisticpy.ir.ops.Map`, holds a name and attributes, never a Python
function. The engines look the name up here. Its :class:`Implementation` says how many
arguments it takes, whether it applies to each component of a tensor, and what kind of
value it gives.

The names are of two kinds:

- a builtin is registered under the name the language calls it by: ``sin``, ``solve``,
  ``Schwarzschild``;
- what the compiler emits for notation is registered in a namespace: ``core.sum`` for
  :math:`\\sum`, ``core.subs`` for an evaluation bar, ``core.diff`` for a derivative,
  ``linalg.solve`` for an inverse times a vector, ``stats.ledoit_wolf``, ``optimum.find``
  for an optimum, ``prob.expectation`` for an expectation of random variables, and
  ``notation.solve`` for the algebra of index notation.

The builtins are the capabilities the integrations declare
(:mod:`relativisticpy.integrations`), registered under each of their names; what the
compiler emits for notation is the core, here, with its implementation in
:mod:`relativisticpy.runtime.core`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from relativisticpy import assumptions, integrations, spacetimes
from relativisticpy.diffgeom import geodesic_equations
from relativisticpy.integrations import _distributions as distributions
from relativisticpy.integrations import _objects as objects
from relativisticpy.integrations import _optimizers as optima
from relativisticpy.runtime import (
    arithmetic,
    core,
    notation,
    quantum,
    recurrences,
    sequences,
    vectors,
)
from relativisticpy.tensors import linalg

#: What an operation gives: ``"scalar"``, a scalar where its arguments are scalars (a
#: value otherwise, as ``sin`` of a list is); ``"value"``, any other value, as a list, an
#: equation or a matrix; ``"argument"``, a value of the type of its first argument, as a
#: derivative of each component of a tensor is; ``"tensor"``, a tensor, whose indices and
#: spaces the compiler works out; ``"notation"``, a value of index notation, whose free
#: indices notation's rules give (:mod:`relativisticpy.typing.notation`).
Result = Literal["scalar", "value", "argument", "tensor", "notation"]


@dataclass(frozen=True)
class Implementation:
    """An operation the IR names, and the function that computes it.

    ``func`` is given the values of the operation's arguments, in order, then its
    attributes, by name. It takes from ``min_args`` to ``max_args`` arguments (``None``:
    any number). A ``componentwise`` operation given a tensor applies to every component,
    as ``simplify(R_{abcd})`` simplifies each, in parallel when the workbook has workers.
    ``result`` is the kind of value it gives (see :data:`Result`). A ``builtin`` is
    called by its name in the language, and ``parameters`` names its parameters in
    messages, as a spacetime's. Text an operation that ``takes_names`` is given is names,
    as for ``symbols('x y')``, which never runs it; any other refuses text written in
    quotes, which SymPy would run as Python.

    Two operations are staged, as lowering notation to components is: one that
    ``references`` its arguments after the first is given those as the nodes they are,
    not computed, to make IR that uses them; one that ``evaluates`` its argument, the IR
    another made, gives that IR's value.

    A capability of an integration may compute with arrays itself: ``numeric`` is then
    given the array module of the data, NumPy's or JAX's, and the arguments.
    """

    name: str
    func: Callable[..., Any]
    result: Result = "value"
    min_args: int = 0
    max_args: int | None = None
    componentwise: bool = False
    builtin: bool = False
    takes_names: bool = False
    parameters: tuple[str, ...] = ()
    references: bool = False
    evaluates: bool = False
    numeric: integrations.Numerically | None = None

    @property
    def scalar(self) -> bool:
        """Whether it gives a scalar for scalar arguments."""
        return self.result == "scalar"

    def check_arity(self, count: int) -> str | None:
        """An error message if ``count`` arguments are wrong for this function:
        ``Schwarzschild(M, G, c) takes at most 3 arguments, not 4``."""
        low, high = self.min_args, self.max_args
        if low <= count and (high is None or count <= high):
            return None
        if high == low:
            expected = f"{low}"
        elif high is None:
            expected = f"at least {low}"
        elif low == 0:
            expected = f"at most {high}"
        elif high == low + 1:
            expected = f"{low} or {high}"
        else:
            expected = f"{low} to {high}"
        plural = "" if expected in ("1", "at least 1", "at most 1") else "s"
        return f"{self.signature()} takes {expected} argument{plural}, not {count}"

    def signature(self) -> str:
        """How the function is called: ``Schwarzschild(M, G, c)``, or ``solve()``."""
        return f"{self.name}({', '.join(self.parameters)})"


def lookup(name: str) -> Implementation:
    """The implementation registered as ``name``."""
    try:
        return _ENTRIES[name]
    except KeyError:
        raise KeyError(f"no operation is registered as {name!r}") from None


# -- the builtins ----------------------------------------------------------------------


def _builtins() -> Iterator[Implementation]:
    """The functions a program calls by name: every capability the integrations declare,
    under its name and each alias. A spacetime names its parameters in messages. A rewrite
    is no function: what it rewrites is computed by the capability it names; nor is an
    optimizer: an optimum, as books write it, is found by the optimizers in turn."""
    for capability in integrations.capabilities():
        if isinstance(capability, integrations.Rewrite | integrations.Optimizer):
            continue
        low, high = capability.arity
        result: Result = "scalar" if capability.scalar else "value"
        if isinstance(capability, integrations.Transform) and capability.on_data:
            result = "tensor"  # data over the space the transform names
        spacetime = isinstance(capability, integrations.Constructor) and spacetimes.declares(
            capability.exact
        )
        for name in capability.names:
            yield Implementation(
                name,
                integrations.Exactly(name),
                result,
                low,
                high,
                componentwise=capability.lifts and isinstance(capability, integrations.Function),
                builtin=True,
                takes_names=capability.takes_names,
                parameters=capability.parameters if spacetime else (),
                numeric=integrations.Numerically(name) if capability.numeric else None,
            )


# -- what the compiler emits for notation ---------------------------------------------------


def _core() -> Iterator[Implementation]:
    def entry(
        name: str,
        func: Callable[..., Any],
        result: Result,
        low: int = 0,
        high: int | None = None,
        *,
        componentwise: bool = False,
        takes_names: bool = False,
    ) -> Implementation:
        return Implementation(
            name, func, result, low, high, componentwise=componentwise, takes_names=takes_names
        )

    # collections
    yield entry("core.list", core.make_list, "value")
    yield entry("core.tuple", core.make_tuple, "value")
    yield entry("core.item", core.item, "value", 2, 2, takes_names=True)
    yield entry("core.part", core.item, "tensor", 2, 2, takes_names=True)
    yield entry("core.piecewise", core.piecewise, "scalar")
    # calculus, each applied to every component of a tensor
    yield entry("core.limits", core.limits, "value", 3, 3)
    yield entry("core.sum", core.summation, "argument", 4, 4, componentwise=True)
    yield entry("core.prod", core.product, "argument", 4, 4, componentwise=True)
    yield entry("core.limit", core.limit, "scalar", 4, 4, componentwise=True)
    yield entry("core.diff", core.differentiate, "argument", 2, componentwise=True)
    yield entry("core.derivative_at", core.derivative_at, "scalar", 4, 4)
    yield entry("core.gradient", core.gradient, "tensor", 2, 2)
    yield entry("core.subs", core.substitute, "argument", 1, componentwise=True)
    yield entry("core.time_derivative", core.time_derivative, "argument", 2, 2, componentwise=True)
    # functions, relations and logic
    yield entry("core.function", core.function, "scalar")
    yield entry("core.recurrence", recurrences.member, "argument", 3)
    yield entry("core.lt", core.less, "scalar", 2, 2)
    yield entry("core.gt", core.greater, "scalar", 2, 2)
    yield entry("core.le", core.less_equal, "scalar", 2, 2)
    yield entry("core.ge", core.greater_equal, "scalar", 2, 2)
    yield entry("core.eq", core.equal, "scalar", 2, 2)
    yield entry("core.ne", core.unequal, "scalar", 2, 2)
    yield entry("core.not", core.logical_not, "scalar", 1, 1)
    yield entry("core.and", core.logical_and, "scalar", 2)
    yield entry("core.or", core.logical_or, "scalar", 2)
    # the metric
    yield entry("core.where", core.where, "value", 3, 3)
    yield entry("core.where_function", core.where_function, "value", 3, 3)
    yield entry("core.line_element", core.line_element, "scalar", 2, 2)
    yield entry("core.components", core.components, "value", 1, 1)
    yield entry("core.metric_diagnosis", core.metric_diagnosis, "value", 1, 1)
    yield entry("core.geodesic_equations", geodesic_equations, "value", 3, 3)
    # what was defined with a symbol an assumption has since given facts
    yield entry("core.assume", assumptions.substituted, "argument", 1, 1)
    # values for Python
    yield entry("core.tensor", core.tensor, "value", 1, 1)
    yield entry("core.tensor_equation", core.tensor_equation, "value", 1, 1)
    # spaces
    yield entry("core.identity", core.identity, "tensor", 1, 1)
    yield entry("core.ones", core.ones, "tensor", 1, 1)
    yield entry("core.uniform", core.uniform, "tensor", 1, 1)
    yield entry("core.positions", core.positions, "tensor", 2, 2)
    yield entry("core.power", core.power, "argument", 2, 2)
    yield entry("core.space_metric", core.space_metric, "tensor", 2, 2)
    yield entry("core.hadamard", core.elementwise_product, "value", 2, 2)
    yield entry("core.conjugate", core.conjugated, "argument", 2, 2)
    yield entry("core.squared_modulus", core.squared_modulus, "argument", 2, 2)
    yield entry("core.real", core.real, "scalar", 1, 1)
    # linear algebra and statistics
    yield entry("linalg.solve", linalg.solve, "tensor", 3, 3)
    yield entry("linalg.hadamard", linalg.hadamard, "argument", 2, 2)
    yield entry("linalg.diag", linalg.diag, "tensor", 1, 1)
    yield entry("linalg.eig", linalg.eig, "value", 4, 4)
    yield entry("linalg.svd", linalg.svd, "value", 2, 2)
    yield entry("linalg.det", linalg.det, "scalar", 2, 2)
    yield entry("linalg.trace", linalg.trace, "scalar", 2, 2)
    yield entry("linalg.function", core.matrix_function, "tensor", 3, 3)
    yield entry("linalg.of_a_map", core.of_a_map, "tensor", 1)
    yield entry("linalg.number_of_a_map", core.of_a_map, "scalar", 1)
    yield entry("linalg.identity", core.identity_beside, "value", 1, 1)
    yield entry("linalg.ones", core.ones_beside, "value", 1, 1)
    yield entry("linalg.entry", core.entry, "scalar", 3, 3)
    yield entry("stats.ledoit_wolf", core.ledoit_wolf, "tensor", 1, 1)
    # Python's functions, called on the values of a program
    yield entry("python.call", objects.call, "value", 1)
    # optima: a problem, found by the optimizers in turn, as a number, where it is, or a
    # tensor; the extremes of data along an index, its label there, and along one of several
    yield entry("optimum.find", optima.optimum, "scalar", 1)
    yield entry("optimum.point", optima.optimum, "value", 1)
    yield entry("optimum.components", optima.optimum, "tensor", 1)
    yield entry("optimum.extreme", optima.extreme, "scalar", 1, 1)
    yield entry("optimum.label", optima.extreme, "value", 1, 1)
    yield entry("optimum.along", optima.extreme, "tensor", 1, 1)
    yield entry("optimum.order", optima.order, "scalar", 2, 2)
    # number theory: over the values a condition gives an index, and remainders
    yield entry("arith.over", arithmetic.over, "argument", 5, componentwise=True)
    yield entry("arith.divides", arithmetic.divides, "scalar", 2, 2)
    yield entry("arith.power_mod", arithmetic.power_mod, "scalar", 3, 3)
    yield entry("arith.coefficient", arithmetic.coefficient, "scalar", 3, 3)
    yield entry("arith.solve", arithmetic.solve_congruences, "value", 3)
    # quantum mechanics without a representation, and its brackets
    yield entry("quantum.ket", quantum.ket, "value", 1)
    yield entry("quantum.dagger", quantum.dagger, "value", 1, 1)
    yield entry("quantum.braket", quantum.braket, "value", 2)
    yield entry("quantum.commutator", quantum.commutator, "value", 3, 3)
    yield entry("quantum.rule", quantum.rule, "value", 2, 3)
    yield entry("quantum.reduced", quantum.reduced, "scalar", 1)
    yield entry("quantum.ordered", quantum.ordered, "scalar", 1)
    yield entry("quantum.delta", quantum.delta, "scalar", 2, 2)
    yield entry("quantum.clebsch_gordan", quantum.clebsch_gordan, "scalar", 6, 6)
    yield entry("quantum.conjugate", quantum.conjugate, "argument", 1)
    yield entry("quantum.modulus", quantum.modulus, "scalar", 1)
    yield entry("quantum.inner", quantum.inner, "scalar", 2, 2)
    # sequences: sums over their index, difference equations at rest, and their transforms
    yield entry("seq.sum", sequences.summation, "argument", 4, 4, componentwise=True)
    yield entry("seq.recurrence", sequences.member, "argument", 3, 3)
    yield entry("seq.solve", sequences.solved, "value", 2, 2)
    yield entry("seq.transform", sequences.transformed, "scalar", 2, 2)
    # random variables: one of a law, or of laws given others, and what is computed of them
    yield entry("prob.variable", distributions.variable, "scalar", 2)
    yield entry("prob.sample", distributions.sample, "value", 3, 3)
    yield entry("prob.member", distributions.member, "scalar", 2, 2)
    yield entry("prob.probability", distributions.probability, "scalar", 1)
    yield entry("prob.expectation", distributions.expectation, "scalar", 1)
    yield entry("prob.variance", distributions.variance, "scalar", 1)
    yield entry("prob.covariance", distributions.covariance, "scalar", 2)
    yield entry("prob.density", distributions.density, "scalar", 2, 2)
    yield entry("prob.cdf", distributions.cdf, "scalar", 2, 2)
    yield entry("prob.quantile", distributions.quantile, "scalar", 2, 2)
    yield entry("prob.mgf", distributions.generating, "scalar", 2, 2)
    yield entry("prob.cf", distributions.characteristic, "scalar", 2, 2)
    # vector calculus, from the metric of the coordinates it is written in
    yield entry("vectors.gradient", vectors.gradient, "value", 1, 2)
    yield entry("vectors.divergence", vectors.divergence, "scalar", 1, 2)
    yield entry("vectors.curl", vectors.curl, "value", 1, 2)
    yield entry("vectors.laplacian", vectors.laplacian, "scalar", 1, 2)
    yield entry("vectors.directional", vectors.directional, "scalar", 2, 3)
    yield entry("vectors.hessian", vectors.hessian, "value", 1, 1)
    yield entry("vectors.component", vectors.component, "scalar", 2, 2)
    yield entry("vectors.levi_civita", vectors.levi_civita, "tensor", 1, 1)
    yield entry("vectors.along", vectors.along, "scalar", 4, 4)
    yield entry("vectors.norm", vectors.norm, "scalar", 1, 1)
    yield entry("vectors.induced_metric", vectors.induced_metric, "value", 1, 1)


# -- the algebra of index notation ----------------------------------------------------------


def _notation() -> Iterator[Implementation]:
    """The operations of abstract index notation (:mod:`relativisticpy.runtime.notation`),
    each giving a value of notation, but for the algebra they are written in and the
    components of notation, which are staged."""

    def entry(name: str, func: Callable[..., Any], low: int, high: int | None) -> Implementation:
        return Implementation(f"notation.{name}", func, "notation", low, high)

    yield Implementation("notation.algebra", notation.algebra, "value", 0, 0)
    yield entry("tensor", notation.tensor, 1, 1)
    yield entry("ricci_scalar", notation.ricci_scalar, 1, 1)
    yield entry("scalar", notation.scalar, 1, 1)
    yield entry("add", notation.add, 1, None)
    yield entry("mul", notation.mul, 1, None)
    yield entry("denominator", notation.denominator, 1, 1)
    yield entry("div", notation.divide, 2, 2)
    yield entry("neg", notation.negative, 1, 1)
    yield entry("value", notation.value, 1, 1)
    yield entry("expression", notation.expression, 2, 2)
    yield entry("equation", notation.equation, 3, 3)
    yield entry("unknown", notation.unknown, 2, 2)
    yield entry("solve", notation.solve, 2, 2)
    yield entry("lhs", notation.lhs, 1, 1)
    yield entry("rhs", notation.rhs, 1, 1)
    yield entry("subs", notation.substitute, 2, 2)
    yield entry("trace", notation.trace, 1, 1)
    yield entry("simplify", notation.simplify, 1, 1)
    yield entry("rebase", notation.rebase, 2, 2)
    yield Implementation("notation.lower", notation.lower, "value", 1, None, references=True)
    yield Implementation("notation.components", _staged, "tensor", 1, 1, evaluates=True)


def _staged(lowered: Any) -> Any:  # pragma: no cover - the executor computes it itself
    raise TypeError("the components of notation are the value of the IR lowered for them")


_ENTRIES: dict[str, Implementation] = {}


def _register() -> None:
    """Register every operation, again whenever the integrations change."""
    entries: dict[str, Implementation] = {}
    for entry in (*_builtins(), *_core(), *_notation()):
        if entry.name in entries:  # pragma: no cover - a mistake in the tables above
            raise ValueError(f"two operations are registered as {entry.name!r}")
        entries[entry.name] = entry
    _ENTRIES.clear()
    _ENTRIES.update(entries)


_register()
integrations.on_change(_register)

#: Every operation, by its name: the builtins first, in the order the integrations declare
#: them, then what the compiler emits for notation, then the algebra of index notation.
REGISTRY = MappingProxyType(_ENTRIES)
