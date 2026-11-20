r"""Curvature kernels: the geometry of a metric, component by component.

Each function takes component arrays and returns component arrays. They are
shared by :class:`~relativisticpy.diffgeom.Metric` and by the IR executor.

Simplification is the expensive part of symbolic geometry, so every kernel
gathers the components it computes into one batch and hands the batch to a
``simplify`` map. The default simplifies in turn; the executor passes a map
that spreads the batch over worker processes. Symmetries are used so that
each independent component is computed once: Christoffel symbols for
:math:`b \le c`, Riemann components for :math:`c < d`.

Conventions (those of Misner, Thorne & Wheeler and of Carroll):

.. math::

    \Gamma^a{}_{bc} = \tfrac12 g^{ad}(\partial_b g_{dc} + \partial_c g_{db} - \partial_d g_{bc})

    R^a{}_{bcd} = \partial_c \Gamma^a{}_{db} - \partial_d \Gamma^a{}_{cb}
                + \Gamma^a{}_{ce}\Gamma^e{}_{db} - \Gamma^a{}_{de}\Gamma^e{}_{cb}

    R_{bd} = R^a{}_{bad}, \qquad R = g^{bd} R_{bd}, \qquad
    G_{ab} = R_{ab} - \tfrac12 R g_{ab}
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import product

import sympy
from sympy import Expr, ImmutableDenseNDimArray, ImmutableMatrix, Rational, Symbol

from relativisticpy.tensors import arrays
from relativisticpy.tensors.arrays import Array
from relativisticpy.tensors.simplification import simplify_component

#: Maps a batch of expressions to their simplified forms, in order.
SimplifyMap = Callable[[Sequence[Expr]], list[Expr]]


def serial(items: Sequence[Expr]) -> list[Expr]:
    """The default :data:`SimplifyMap`: simplify each component in turn."""
    return [simplify_component(item) for item in items]


def _batch(
    keys: Sequence[tuple[int, ...]], values: Sequence[Expr], simplify: SimplifyMap
) -> dict[tuple[int, ...], Expr]:
    return dict(zip(keys, simplify(list(values)), strict=True))


def metric_matrix(components: object) -> ImmutableMatrix:
    """A metric's components as a matrix, checking that they form one."""
    matrix = ImmutableMatrix(arrays.to_array(components).tolist())
    if not matrix.is_square:
        raise ValueError(
            f"a metric must be a square matrix, not {matrix.shape[0]}×{matrix.shape[1]}"
        )
    return matrix


def check_metric(g: ImmutableMatrix) -> None:
    """Raise ``ValueError`` unless ``g`` is symmetric and non-degenerate."""
    for i, j in product(range(g.rows), repeat=2):
        if i < j and sympy.simplify(g[i, j] - g[j, i]) != 0:
            raise ValueError(f"a metric must be symmetric, but g[{i}, {j}] ≠ g[{j}, {i}]")
    if sympy.simplify(g.det()) == 0:
        raise ValueError("a metric must be non-degenerate, but its determinant is zero")


def inverse(g: ImmutableMatrix, simplify: SimplifyMap = serial) -> ImmutableMatrix:
    r"""The inverse metric :math:`g^{ab}`."""
    if g.is_diagonal():
        return ImmutableMatrix.diag(*simplify([1 / g[i, i] for i in range(g.rows)]))
    raw = g.inv()
    keys = [(i, j) for i in range(g.rows) for j in range(i, g.rows)]
    values = _batch(keys, [raw[k] for k in keys], simplify)
    return ImmutableMatrix(g.rows, g.cols, lambda i, j: values[(min(i, j), max(i, j))])


def determinant(g: ImmutableMatrix) -> Expr:
    return simplify_component(g.det())


def christoffel(
    g: ImmutableMatrix,
    g_inv: ImmutableMatrix,
    coordinates: Sequence[Symbol],
    simplify: SimplifyMap = serial,
) -> Array:
    r"""The Christoffel symbols :math:`\Gamma^a{}_{bc}` of the Levi-Civita connection."""
    n = len(coordinates)
    dg = arrays.partial(ImmutableDenseNDimArray(g.tolist()), coordinates)  # dg[a, b, c] = ∂_a g_bc
    keys, values = [], []
    for a, b in product(range(n), repeat=2):
        for c in range(b, n):
            terms = [
                g_inv[a, d] * (dg[b, d, c] + dg[c, d, b] - dg[d, b, c])
                for d in range(n)
                if g_inv[a, d] != 0
            ]
            keys.append((a, b, c))
            values.append(Rational(1, 2) * sympy.Add(*terms))
    found = _batch(keys, values, simplify)
    return _fill(n, 3, lambda a, b, c: found[(a, min(b, c), max(b, c))])


def riemann(gamma: Array, coordinates: Sequence[Symbol], simplify: SimplifyMap = serial) -> Array:
    r"""The Riemann tensor :math:`R^a{}_{bcd}`, antisymmetric in its last two indices."""
    n = len(coordinates)
    dgamma = arrays.partial(gamma, coordinates)  # dgamma[c, a, d, b] = ∂_c Γ^a_db
    keys, values = [], []
    for a, b in product(range(n), repeat=2):
        for c in range(n):
            for d in range(c + 1, n):
                value = dgamma[c, a, d, b] - dgamma[d, a, c, b]
                value += sympy.Add(
                    *(
                        gamma[a, c, e] * gamma[e, d, b] - gamma[a, d, e] * gamma[e, c, b]
                        for e in range(n)
                    )
                )
                keys.append((a, b, c, d))
                values.append(value)
    found = _batch(keys, values, simplify)

    def component(a: int, b: int, c: int, d: int) -> Expr:
        if c == d:
            return sympy.S.Zero
        return found[(a, b, c, d)] if c < d else -found[(a, b, d, c)]

    return _fill(n, 4, component)


def ricci(gamma: Array, coordinates: Sequence[Symbol], simplify: SimplifyMap = serial) -> Array:
    r"""The Ricci tensor :math:`R_{bd} = R^a{}_{bad}`, computed straight from the connection."""
    n = len(coordinates)
    dgamma = arrays.partial(gamma, coordinates)
    keys, values = [], []
    for b in range(n):
        for d in range(b, n):
            value = sympy.Add(*(dgamma[a, a, d, b] - dgamma[d, a, a, b] for a in range(n)))
            value += sympy.Add(
                *(
                    gamma[a, a, e] * gamma[e, d, b] - gamma[a, d, e] * gamma[e, a, b]
                    for a in range(n)
                    for e in range(n)
                )
            )
            keys.append((b, d))
            values.append(value)
    found = _batch(keys, values, simplify)
    return _fill(n, 2, lambda b, d: found[(min(b, d), max(b, d))])


def ricci_scalar(ricci_tensor: Array, g_inv: ImmutableMatrix) -> Expr:
    r"""The Ricci scalar :math:`R = g^{ab} R_{ab}`."""
    n = g_inv.rows
    total = sympy.Add(*(g_inv[a, b] * ricci_tensor[a, b] for a, b in product(range(n), repeat=2)))
    return simplify_component(total)


def einstein(
    ricci_tensor: Array, scalar: Expr, g: ImmutableMatrix, simplify: SimplifyMap = serial
) -> Array:
    r"""The Einstein tensor :math:`G_{ab} = R_{ab} - \tfrac12 R g_{ab}`."""
    n = g.rows
    keys = [(a, b) for a in range(n) for b in range(a, n)]
    values = [ricci_tensor[a, b] - Rational(1, 2) * scalar * g[a, b] for a, b in keys]
    found = _batch(keys, values, simplify)
    return _fill(n, 2, lambda a, b: found[(min(a, b), max(a, b))])


def lower_riemann(riemann_tensor: Array, g: ImmutableMatrix) -> Array:
    r""":math:`R_{abcd} = g_{ae} R^e{}_{bcd}`."""
    n = g.rows
    metric = ImmutableDenseNDimArray(g.tolist())
    result = arrays.contract([(metric, "ae"), (riemann_tensor, "ebcd")], "abcd", n)
    lowered: Array = result.applyfunc(simplify_component)
    return lowered


def kretschmann(riemann_tensor: Array, g: ImmutableMatrix, g_inv: ImmutableMatrix) -> Expr:
    r"""The Kretschmann scalar :math:`R_{abcd} R^{abcd}`."""
    n = g.rows
    metric, inverse_metric = (
        ImmutableDenseNDimArray(g.tolist()),
        ImmutableDenseNDimArray(g_inv.tolist()),
    )
    total = arrays.contract(
        [
            (riemann_tensor, "abcd"),
            (riemann_tensor, "efgh"),
            (metric, "ae"),
            (inverse_metric, "bf"),
            (inverse_metric, "cg"),
            (inverse_metric, "dh"),
        ],
        "",
        n,
    )
    return simplify_component(total)


def weyl(
    riemann_lower: Array,
    ricci_tensor: Array,
    scalar: Expr,
    g: ImmutableMatrix,
    simplify: SimplifyMap = serial,
) -> Array:
    r"""The Weyl tensor :math:`C_{abcd}`, the trace-free part of the Riemann tensor.

    Defined in three or more dimensions; it vanishes identically in three.
    """
    n = g.rows
    if n < 3:
        raise ValueError("the Weyl tensor is defined in three or more dimensions")
    keys, values = [], []
    for a, b, c, d in product(range(n), repeat=4):
        if a < b and c < d:
            ricci_part = (
                g[a, c] * ricci_tensor[b, d]
                - g[a, d] * ricci_tensor[b, c]
                - g[b, c] * ricci_tensor[a, d]
                + g[b, d] * ricci_tensor[a, c]
            )
            scalar_part = g[a, c] * g[b, d] - g[a, d] * g[b, c]
            value = (
                riemann_lower[a, b, c, d]
                - ricci_part / (n - 2)
                + scalar * scalar_part / ((n - 1) * (n - 2))
            )
            keys.append((a, b, c, d))
            values.append(value)
    found = _batch(keys, values, simplify)

    def component(a: int, b: int, c: int, d: int) -> Expr:
        if a == b or c == d:
            return sympy.S.Zero
        sign = (1 if a < b else -1) * (1 if c < d else -1)
        return sign * found[(min(a, b), max(a, b), min(c, d), max(c, d))]

    return _fill(n, 4, component)


def kronecker(n: int) -> Array:
    r"""The Kronecker delta :math:`\delta^a{}_b`."""
    identity: Array = ImmutableDenseNDimArray(sympy.eye(n).tolist())
    return identity


def _fill(n: int, rank: int, component: Callable[..., Expr]) -> Array:
    flat = [component(*index) for index in product(range(n), repeat=rank)]
    filled: Array = ImmutableDenseNDimArray(flat, (n,) * rank)
    return filled
