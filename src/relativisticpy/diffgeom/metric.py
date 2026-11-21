"""The :class:`Metric`: a metric tensor that knows its own geometry."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from itertools import product
from typing import Any

import sympy
from sympy import Eq, Expr, Function, ImmutableMatrix, Symbol
from sympy.core.function import AppliedUndef

from relativisticpy.diffgeom import curvature
from relativisticpy.diffgeom.curvature import SimplifyMap
from relativisticpy.tensors import Indices, Tensor
from relativisticpy.tensors.indices import Index, latex_label


class Metric(Tensor):
    r"""A metric :math:`g_{\mu\nu}` on a coordinate patch.

    Every geometric quantity is computed on first use and cached::

        >>> from relativisticpy import spacetimes
        >>> g = spacetimes.schwarzschild()
        >>> g.christoffel[1, 0, 0]
        M*(-2*M + r)/r**3
        >>> g.ricci_scalar
        0

    ``simplify`` maps a batch of components to their simplified forms; pass a
    parallel map to spread the work of simplifying over processes.
    """

    def __init__(
        self,
        components: Any,
        coordinates: Sequence[Symbol | str],
        *,
        name: str = "g",
        simplify: SimplifyMap = curvature.serial,
        check: bool = True,
    ) -> None:
        coordinates = tuple(sympy.Symbol(c) if isinstance(c, str) else c for c in coordinates)
        matrix = curvature.metric_matrix(components)
        if matrix.rows != len(coordinates):
            raise ValueError(
                f"a {matrix.rows}×{matrix.rows} metric needs {matrix.rows} coordinates, "
                f"not {len(coordinates)}"
            )
        if check:
            curvature.check_metric(matrix)
        indices = Indices([Index("mu"), Index("nu")])
        super().__init__(matrix.tolist(), indices, coordinates, name=name)
        self._simplify = simplify

    @classmethod
    def from_line_element(cls, line_element: Expr, coordinates: Sequence[Symbol | str]) -> Metric:
        r"""Build a metric from :math:`ds^2`, a quadratic form in the differentials.

        The differential of a coordinate ``x`` is the symbol ``dx`` (see
        :func:`differential`)::

            >>> t, r = sympy.symbols("t r")
            >>> dt, dr = differential(t), differential(r)
            >>> Metric.from_line_element(-dt**2 + dr**2, (t, r))
            Metric(...)
        """
        symbols = tuple(sympy.Symbol(c) if isinstance(c, str) else c for c in coordinates)
        return cls(metric_from_line_element(line_element, symbols), symbols)

    def __repr__(self) -> str:
        return f"Metric({self.components})"

    # -- the metric itself -------------------------------------------------------

    @cached_property
    def matrix(self) -> ImmutableMatrix:
        return ImmutableMatrix(self.components.tolist())

    @cached_property
    def inverse_matrix(self) -> ImmutableMatrix:
        return curvature.inverse(self.matrix, self._simplify)

    @property
    def inverse(self) -> Tensor:
        r"""The inverse metric :math:`g^{\mu\nu}`."""
        return self._tensor(self.inverse_matrix.tolist(), "^{mu nu}")

    @cached_property
    def determinant(self) -> Expr:
        """The determinant of the metric, the ``g`` of :math:`\\sqrt{-g}`."""
        return curvature.determinant(self.matrix)

    @property
    def line_element(self) -> Expr:
        r"""The line element :math:`ds^2 = g_{\mu\nu}\,dx^\mu dx^\nu`."""
        differentials = [differential(c) for c in self.coordinates]
        return sympy.Add(
            *(
                self.matrix[i, j] * differentials[i] * differentials[j]
                for i, j in product(range(self.dim), repeat=2)
            )
        )

    # -- curvature ---------------------------------------------------------------

    @cached_property
    def christoffel(self) -> Tensor:
        r"""The Christoffel symbols :math:`\Gamma^a{}_{bc}`."""
        components = curvature.christoffel(
            self.matrix, self.inverse_matrix, self.coordinates, self._simplify
        )
        return self._tensor(components, "^{a}_{b c}", "Gamma")

    @cached_property
    def riemann(self) -> Tensor:
        r"""The Riemann tensor :math:`R^a{}_{bcd}`."""
        components = curvature.riemann(
            self.christoffel.components, self.coordinates, self._simplify
        )
        return self._tensor(components, "^{a}_{b c d}", "R")

    @cached_property
    def ricci(self) -> Tensor:
        r"""The Ricci tensor :math:`R_{ab}`."""
        components = curvature.ricci(self.christoffel.components, self.coordinates, self._simplify)
        return self._tensor(components, "_{a b}", "R")

    @cached_property
    def ricci_scalar(self) -> Expr:
        """The Ricci scalar :math:`R`."""
        return curvature.ricci_scalar(self.ricci.components, self.inverse_matrix)

    @cached_property
    def einstein(self) -> Tensor:
        r"""The Einstein tensor :math:`G_{ab} = R_{ab} - \frac12 R g_{ab}`."""
        components = curvature.einstein(
            self.ricci.components, self.ricci_scalar, self.matrix, self._simplify
        )
        return self._tensor(components, "_{a b}", "G")

    @cached_property
    def weyl(self) -> Tensor:
        r"""The Weyl tensor :math:`C_{abcd}`."""
        lowered = curvature.lower_riemann(self.riemann.components, self.matrix)
        components = curvature.weyl(
            lowered, self.ricci.components, self.ricci_scalar, self.matrix, self._simplify
        )
        return self._tensor(components, "_{a b c d}", "C")

    @cached_property
    def kretschmann(self) -> Expr:
        r"""The Kretschmann scalar :math:`R_{abcd} R^{abcd}`."""
        return curvature.kretschmann(self.riemann.components, self.matrix, self.inverse_matrix)

    def geodesic_equations(self, parameter: Symbol | str = "tau") -> list[Eq]:
        r"""The geodesic equations
        :math:`\ddot x^a + \Gamma^a{}_{bc}\,\dot x^b \dot x^c = 0`,
        with each coordinate a function of the affine ``parameter``."""
        return geodesic_equations(self.christoffel.components, self.coordinates, parameter)

    def _tensor(self, components: Any, indices: str, name: str = "g") -> Tensor:
        return Tensor(components, indices, self.coordinates, name=name)


def differential(coordinate: Symbol) -> Symbol:
    r"""The symbol standing for the differential of a coordinate: ``theta`` → ``d\theta``."""
    return sympy.Symbol("d" + latex_label(coordinate.name))


def metric_from_line_element(line_element: Expr, coordinates: Sequence[Symbol]) -> ImmutableMatrix:
    r"""Read the metric components off a line element.

    ``line_element`` must be a quadratic form in the differentials
    (:func:`differential`) of ``coordinates``: :math:`g_{\mu\mu}` is the
    coefficient of :math:`(dx^\mu)^2` and :math:`g_{\mu\nu}` half that of
    :math:`dx^\mu dx^\nu`.
    """
    differentials = [differential(c) for c in coordinates]
    polynomial = sympy.Poly(sympy.expand(line_element), *differentials)
    n = len(coordinates)
    g = sympy.zeros(n, n)
    for powers, coefficient in polynomial.terms():
        if sum(powers) != 2:
            raise ValueError(
                "a line element must be a quadratic form in the differentials, "
                f"but it has a term of degree {sum(powers)}"
            )
        pair = [i for i, power in enumerate(powers) for _ in range(power)]
        i, j = pair
        if i == j:
            g[i, i] = coefficient
        else:
            g[i, j] = g[j, i] = coefficient / 2
    return ImmutableMatrix(g)


def geodesic_equations(
    christoffel_symbols: Any, coordinates: Sequence[Symbol], parameter: Symbol | str = "tau"
) -> list[Eq]:
    """The geodesic equations for a connection, one per coordinate.

    The affine parameter needs a name of its own: a coordinate's would make the
    coordinate a function of itself, and a name in the metric would be read as both.
    """
    tau = sympy.Symbol(parameter) if isinstance(parameter, str) else parameter
    if tau.name in {c.name for c in coordinates}:
        raise ValueError(f"the parameter {tau} is a coordinate: give the parameter another name")
    names = {s.name for s in christoffel_symbols.free_symbols}
    names |= {f.func.__name__ for f in christoffel_symbols.atoms(AppliedUndef)}
    if tau.name in names:
        raise ValueError(f"{tau} already appears in the metric: give the parameter another name")
    paths = [Function(c.name)(tau) for c in coordinates]
    along = dict(zip(coordinates, paths, strict=True))
    n = len(coordinates)
    equations = []
    for a in range(n):
        acceleration = sympy.Add(
            *(
                christoffel_symbols[a, b, c].subs(along) * paths[b].diff(tau) * paths[c].diff(tau)
                for b in range(n)
                for c in range(n)
                if christoffel_symbols[a, b, c] != 0
            )
        )
        equations.append(Eq(paths[a].diff(tau, 2) + acceleration, 0))
    return equations
