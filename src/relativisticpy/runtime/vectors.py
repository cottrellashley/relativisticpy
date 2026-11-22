r"""Vector calculus, computed from the metric of the coordinates it is written in.

Each operator is the one index notation gives, taken from the metric the frame has, with
no formula of its own for each system of coordinates (Arfken, Weber and Harris (2013),
*Mathematical Methods for Physicists*, sections 3.8 to 3.10). A vector's components are
those along the unit vectors of its coordinates, an orthonormal basis, so that the
coordinates must be orthogonal: their metric diagonal, with the scale factors
:math:`h_i = \sqrt{g_{ii}}` along it. Then

- the gradient is :math:`g^{ij}\partial_j f`, and its component along a unit vector
  :math:`h_i g^{ij}\partial_j f`;
- the divergence is :math:`\frac{1}{\sqrt g}\partial_i(\sqrt g\, v^i)`, of the components
  :math:`v^i = v_{\hat\imath}/h_i` in the coordinates' own basis;
- the curl is :math:`\frac{1}{\sqrt g}\epsilon^{ijk}\partial_j v_k`, of the components
  :math:`v_k = h_k v_{\hat k}`, and along a unit vector :math:`h_i` times it;
- the Laplacian is :math:`\frac{1}{\sqrt g}\partial_i(\sqrt g\, g^{ij}\partial_j f)`.

Without a metric, the coordinates are Cartesian, each of scale factor 1. Each function is
registered by :mod:`relativisticpy.runtime.registry` as ``vectors.`` and its name.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from typing import Any

import sympy
from sympy import ImmutableDenseNDimArray, ImmutableMatrix, Symbol

from relativisticpy import display
from relativisticpy.diffgeom import curvature
from relativisticpy.errors import EvaluationError
from relativisticpy.tensors import arrays
from relativisticpy.tensors.simplification import is_zero, simplify_component, tidy


def _metric(metric: Sequence[Any], coordinates: Sequence[Symbol]) -> ImmutableMatrix:
    """The metric given, or the identity of Cartesian coordinates where none is."""
    if not metric:
        return ImmutableMatrix.eye(len(coordinates))
    return curvature.metric_matrix(metric[0])


def scale_factors(g: ImmutableMatrix, coordinates: Sequence[Symbol]) -> list[sympy.Expr]:
    """The scale factors of orthogonal coordinates, :math:`\\sqrt{g_{ii}}`, each positive,
    as r sin θ is of φ where θ lies between 0 and π; refused where the coordinates are not
    orthogonal, or not of a space, whose lengths are positive."""
    for i, j in product(range(g.rows), repeat=2):
        if i < j and not is_zero(g[i, j]):
            pair = f"{display.unicode(coordinates[i])}{display.unicode(coordinates[j])}"
            raise EvaluationError(
                f"the coordinates are not orthogonal: g_{pair} = {display.unicode(g[i, j])}, "
                "so their unit vectors, along which a vector's components are taken, are "
                "not defined by scale factors",
                hint="write the vector and its derivatives with indices, as \\nabla_\\mu V^\\mu",
            )
    factors = []
    for i, coordinate in enumerate(coordinates):
        if g[i, i].is_extended_negative:
            raise EvaluationError(
                f"g_{display.unicode(coordinate)}{display.unicode(coordinate)} = "
                f"{display.unicode(g[i, i])} is negative, so these coordinates are not of a "
                "space with lengths, as vector calculus needs",
                hint="for a spacetime, write derivatives with indices, as \\nabla_\\mu V^\\mu",
            )
        factors.append(positive_root(g[i, i]))
    return factors


def positive_root(square: Any) -> sympy.Expr:
    """The square root of ``square`` that books write for a length, as r sin θ of
    r² sin² θ: each factor raised to an even power comes out of the root as it is, as if
    positive, as sin θ is for θ between 0 and π, without the absolute values SymPy puts on
    real ones."""
    coefficient, factors = sympy.factor(sympy.sympify(square)).as_coeff_mul()
    outside: list[sympy.Expr] = []
    inside: list[sympy.Expr] = [coefficient]
    for factor in factors:
        base, exponent = factor.as_base_exp()
        if exponent.is_Integer and exponent % 2 == 0:
            outside.append(base ** (exponent // 2))
        else:
            inside.append(factor)
    return sympy.Mul(*outside) * sympy.sqrt(sympy.Mul(*inside))


def _components(vector: Any, size: int) -> list[sympy.Expr]:
    values = [sympy.sympify(value) for value in arrays.flatten(arrays.to_array(vector))]
    if len(values) != size:
        raise EvaluationError(
            f"a vector of {len(values)} components is not of these {size} coordinates"
        )
    return values


def gradient(f: Any, *metric: Any, coordinates: tuple[Symbol, ...]) -> list[sympy.Expr]:
    """The gradient of ``f``: :math:`h_i g^{ij}\\partial_j f` along each unit vector."""
    g = _metric(metric, coordinates)
    h = scale_factors(g, coordinates)
    inverse = curvature.inverse(g)
    n = len(coordinates)
    return [
        tidy(h[i] * sum(inverse[i, j] * sympy.diff(f, coordinates[j]) for j in range(n)))
        for i in range(n)
    ]


def divergence(vector: Any, *metric: Any, coordinates: tuple[Symbol, ...]) -> sympy.Expr:
    """The divergence of a vector given along the unit vectors:
    :math:`\\frac{1}{\\sqrt g}\\partial_i(\\sqrt g\\, v^i)`."""
    g = _metric(metric, coordinates)
    h = scale_factors(g, coordinates)
    v = _components(vector, len(coordinates))
    root = sympy.Mul(*h)
    total = sum(
        sympy.diff(root * v[i] / h[i], coordinate) for i, coordinate in enumerate(coordinates)
    )
    return tidy(total / root)


def curl(vector: Any, *metric: Any, coordinates: tuple[Symbol, ...]) -> list[sympy.Expr]:
    """The curl of a vector of three components given along the unit vectors:
    :math:`\\frac{h_i}{\\sqrt g}\\epsilon^{ijk}\\partial_j v_k`, with :math:`v_k = h_k
    v_{\\hat k}`."""
    g = _metric(metric, coordinates)
    h = scale_factors(g, coordinates)
    v = _components(vector, 3)
    lowered = [h[k] * v[k] for k in range(3)]
    root = sympy.Mul(*h)
    result = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        twist = sympy.diff(lowered[k], coordinates[j]) - sympy.diff(lowered[j], coordinates[k])
        result.append(tidy(h[i] * twist / root))
    return result


def laplacian(f: Any, *metric: Any, coordinates: tuple[Symbol, ...]) -> sympy.Expr:
    """The Laplacian of ``f``: :math:`\\frac{1}{\\sqrt g}\\partial_i(\\sqrt g\\, g^{ij}
    \\partial_j f)`."""
    g = _metric(metric, coordinates)
    root = sympy.Mul(*scale_factors(g, coordinates))
    inverse = curvature.inverse(g)
    n = len(coordinates)
    total = sum(
        sympy.diff(
            root * sum(inverse[i, j] * sympy.diff(f, coordinates[j]) for j in range(n)),
            coordinates[i],
        )
        for i in range(n)
    )
    return tidy(total / root)


def directional(along: Any, f: Any, *metric: Any, coordinates: tuple[Symbol, ...]) -> Any:
    """:math:`(\\mathbf{a}\\cdot\\nabla) f`, the derivative of a function ``f`` along
    ``a``: the dot product of a with its gradient."""
    a = _components(along, len(coordinates))
    each = gradient(f, *metric, coordinates=coordinates)
    return tidy(sum(component * part for component, part in zip(a, each, strict=True)))


def hessian(f: Any, *, coordinates: tuple[Symbol, ...]) -> ImmutableDenseNDimArray:
    """The Hessian of ``f``, the matrix of its second derivatives by ``coordinates``."""
    return ImmutableDenseNDimArray(
        [[sympy.diff(f, row, column) for column in coordinates] for row in coordinates]
    )


def component(vector: Any, position: int) -> sympy.Expr:
    """A vector's component at ``position``, counted from 0, of a list or a column."""
    return sympy.sympify(arrays.flatten(arrays.to_array(vector))[position])


def norm(vector: Any) -> sympy.Expr:
    """The length of a vector, the square root of the sum of the squares of its components,
    simplified, as books take a vector's components to be real."""
    values = [sympy.sympify(value) for value in arrays.flatten(arrays.to_array(vector))]
    return sympy.sqrt(simplify_component(sum(value**2 for value in values)))


def induced_metric(cartesian: Any, *, coordinates: tuple[Symbol, ...]) -> ImmutableDenseNDimArray:
    """The metric of coordinates that the Cartesian coordinates are written in:
    :math:`g_{ij} = \\sum_k \\partial_i x_k \\, \\partial_j x_k`."""
    values = [sympy.sympify(value) for value in cartesian]
    rows = [
        [
            simplify_component(sum(sympy.diff(x, row) * sympy.diff(x, column) for x in values))
            for column in coordinates
        ]
        for row in coordinates
    ]
    return ImmutableDenseNDimArray(rows)
