"""Kernels: what each IR operation computes, on SymPy values."""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce, singledispatch
from typing import TYPE_CHECKING, Any

import sympy
from sympy import ImmutableDenseNDimArray, ImmutableMatrix

from relativisticpy.diffgeom import curvature
from relativisticpy.diffgeom.metric import metric_from_line_element
from relativisticpy.ir import ops
from relativisticpy.tensors import arrays
from relativisticpy.tensors.equation import TensorEquation

if TYPE_CHECKING:
    from relativisticpy.runtime.executor import Executor


@singledispatch
def execute(op: ops.Op, args: list[Any], executor: Executor) -> Any:
    raise NotImplementedError(f"no kernel for {type(op).__name__}")  # pragma: no cover


@execute.register(ops.Const)
def _const(op: ops.Const, args: list[Any], executor: Executor) -> Any:
    return op.value


@execute.register(ops.Call)
def _call(op: ops.Call, args: list[Any], executor: Executor) -> Any:
    return op.func(*args)


@execute.register(ops.Add)
def _add(op: ops.Add, args: list[Any], executor: Executor) -> Any:
    total = args[0] if op.signs[0] > 0 else -args[0]
    for sign, value in zip(op.signs[1:], args[1:], strict=True):
        total = total + value if sign > 0 else total - value
    return total


@execute.register(ops.Mul)
def _mul(op: ops.Mul, args: list[Any], executor: Executor) -> Any:
    return reduce(operator.mul, args)


@execute.register(ops.Pow)
def _pow(op: ops.Pow, args: list[Any], executor: Executor) -> Any:
    return args[0] ** args[1]


@execute.register(ops.Contract)
def _contract(op: ops.Contract, args: list[Any], executor: Executor) -> Any:
    return arrays.contract(list(zip(args, op.labels, strict=True)), op.output, op.dim)


@execute.register(ops.Permute)
def _permute(op: ops.Permute, args: list[Any], executor: Executor) -> Any:
    return arrays.permute(args[0], op.axes)


@execute.register(ops.Select)
def _select(op: ops.Select, args: list[Any], executor: Executor) -> Any:
    return arrays.select(args[0], dict(op.fixed))


@execute.register(ops.Partial)
def _partial(op: ops.Partial, args: list[Any], executor: Executor) -> Any:
    return arrays.partial(args[0], op.coordinates)


@execute.register(ops.Symmetrize)
def _symmetrize(op: ops.Symmetrize, args: list[Any], executor: Executor) -> Any:
    return arrays.symmetrize(args[0], op.axes, antisymmetric=op.antisymmetric)


@dataclass(frozen=True)
class ComponentCall:
    """``func(component, *extra)``: picklable, so it can run in a worker process."""

    func: Callable[..., Any]
    extra: tuple[Any, ...]

    def __call__(self, component: Any) -> Any:
        return self.func(component, *self.extra)


@execute.register(ops.Map)
def _map(op: ops.Map, args: list[Any], executor: Executor) -> Any:
    target, extra = args[0], tuple(args[1:])
    call = ComponentCall(op.func, extra)
    if not isinstance(target, sympy.NDimArray):
        return call(target)
    results = executor.map(call, arrays.flatten(target))
    return ImmutableDenseNDimArray(results, target.shape)


@execute.register(ops.MakeArray)
def _make_array(op: ops.MakeArray, args: list[Any], executor: Executor) -> Any:
    return ImmutableDenseNDimArray([sympy.sympify(a) for a in args], op.shape)


@execute.register(ops.ToArray)
def _to_array(op: ops.ToArray, args: list[Any], executor: Executor) -> Any:
    value = args[0]
    if isinstance(value, sympy.NDimArray | sympy.MatrixBase | list | tuple):
        array = arrays.to_array(value)
        if array.shape != op.shape and len(array) == len(ImmutableDenseNDimArray.zeros(*op.shape)):
            array = array.reshape(*op.shape)  # a 1×n matrix for a vector
        if array.shape == op.shape:
            return array
        raise ValueError(f"expected components of shape {op.shape}, got {array.shape}")
    if value == 0:
        return arrays.zeros(op.shape[0], len(op.shape))
    raise ValueError(f"expected components of shape {op.shape}, got {value}")


@execute.register(ops.Equation)
def _equation(op: ops.Equation, args: list[Any], executor: Executor) -> Any:
    lhs, rhs = args
    if isinstance(lhs, sympy.NDimArray) or isinstance(rhs, sympy.NDimArray):
        array = lhs if isinstance(lhs, sympy.NDimArray) else rhs
        lhs = (
            lhs if isinstance(lhs, sympy.NDimArray) else arrays.zeros(array.shape[0], array.rank())
        )
        rhs = (
            rhs if isinstance(rhs, sympy.NDimArray) else arrays.zeros(array.shape[0], array.rank())
        )
        return TensorEquation(lhs, rhs)
    return sympy.Eq(lhs, rhs)


@execute.register(ops.LineElement)
def _line_element(op: ops.LineElement, args: list[Any], executor: Executor) -> Any:
    return ImmutableDenseNDimArray(metric_from_line_element(args[0], op.coordinates).tolist())


@execute.register(ops.Geometry)
def _geometry(op: ops.Geometry, args: list[Any], executor: Executor) -> Any:
    simplify = executor.simplify_map
    coordinates = op.coordinates
    quantity = op.quantity
    if quantity == "kronecker":
        return curvature.kronecker(len(coordinates))
    if quantity == "metric":
        matrix = curvature.metric_matrix(args[0])
        if matrix.rows != len(coordinates):
            raise ValueError(
                f"a metric in {len(coordinates)} coordinates must be {len(coordinates)}×"
                f"{len(coordinates)}, not {matrix.rows}×{matrix.cols}"
            )
        curvature.check_metric(matrix)
        return ImmutableDenseNDimArray(matrix.tolist())
    matrices = [ImmutableMatrix(a.tolist()) if _is_matrix(a) else a for a in args]
    if quantity == "inverse":
        return _array(curvature.inverse(matrices[0], simplify))
    if quantity == "determinant":
        return curvature.determinant(matrices[0])
    if quantity == "christoffel":
        return curvature.christoffel(matrices[0], matrices[1], coordinates, simplify)
    if quantity == "riemann":
        return curvature.riemann(args[0], coordinates, simplify)
    if quantity == "ricci":
        return curvature.ricci(args[0], coordinates, simplify)
    if quantity == "ricci_scalar":
        return curvature.ricci_scalar(args[0], matrices[1])
    if quantity == "einstein":
        return curvature.einstein(args[0], args[1], matrices[2], simplify)
    if quantity == "riemann_lower":
        return curvature.lower_riemann(args[0], matrices[1])
    if quantity == "weyl":
        return curvature.weyl(args[0], args[1], args[2], matrices[3], simplify)
    if quantity == "kretschmann":
        return curvature.kretschmann(args[0], matrices[1], matrices[2])
    raise NotImplementedError(quantity)  # pragma: no cover


def _is_matrix(value: Any) -> bool:
    return isinstance(value, sympy.NDimArray) and value.rank() == 2


def _array(matrix: ImmutableMatrix) -> ImmutableDenseNDimArray:
    return ImmutableDenseNDimArray(matrix.tolist())
