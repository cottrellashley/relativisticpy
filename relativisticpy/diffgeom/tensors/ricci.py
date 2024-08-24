# Standard Library
from functools import singledispatchmethod
from itertools import product

from loguru import logger

from relativisticpy.algebras import Indices, EinsumArray
# External Modules
from relativisticpy.diffgeom.manifold import CoordIndices
from relativisticpy.diffgeom.tensor import Tensor
from relativisticpy.diffgeom.metric import Metric
from relativisticpy.symengine import SymbolArray, Rational, zeros, diff, simplify

# This Module
from relativisticpy.diffgeom.connection import LeviCivitaConnection
from relativisticpy.diffgeom.tensors.riemann import Riemann


class Ricci(Tensor):
    """
     The Ricci Curvature Tensor class.

     - **Type:** Tensor.
     - **Role:** A contraction of the Riemann tensor, used in the Einstein field equations to relate spacetime curvature to matter content.
     - **Properties:** Symmetric, R_{mu nu} = R_{nu mu}
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @singledispatchmethod
    @classmethod
    def _new(cls, operand, indices):
        logger.debug(f"[Ricci] Handling init from tensor: {operand.__class__.__name__}")

    @_new.register
    @classmethod
    def _(cls, operand: Metric, indices: Indices) -> Tensor:
        components = cls.components_from_metric(operand)
        return cls(indices, components)

    @_new.register
    @classmethod
    def _(cls, operand: LeviCivitaConnection, indices: Indices) -> Tensor:
        components = cls.components_from_connection(operand)
        return cls(indices, components)

    @_new.register
    @classmethod
    def _(cls, operand: Riemann, indices: Indices) -> Tensor:
        components = cls.components_from_riemann(operand)
        return cls(indices, components)

    @staticmethod
    def components_from_metric(metric: Metric) -> SymbolArray:
        dim = metric.dimention
        wrt_array = metric.indices.basis
        gamma = LeviCivitaConnection.components_from_metric(metric)
        skeleton = SymbolArray(zeros(dim ** 2), (dim, dim))
        for j, p, i, d in product(range(dim), range(dim), range(dim), range(dim)):
            skeleton[j, p] += Rational(1, dim) * (
                    diff(gamma[i, p, j], wrt_array[i])
                    - diff(gamma[i, i, j], wrt_array[p])
            ) + (
                                      gamma[i, i, d] * gamma[d, p, j]
                                      - gamma[i, p, d] * gamma[d, i, j]
                              )
        return simplify(skeleton)

    @staticmethod
    def components_from_connection(connection: LeviCivitaConnection) -> SymbolArray:
        dim = connection.dimention
        wrt = connection.indices.basis
        gamma = connection.components
        skeleton = SymbolArray(zeros(dim ** 2), (dim, dim))
        for j, p, i, d in product(range(dim), range(dim), range(dim), range(dim)):
            skeleton[j, p] += Rational(1, dim) * (
                    diff(gamma[i, p, j], wrt[i])
                    - diff(gamma[i, i, j], wrt[p])
            ) + (
                               gamma[i, i, d] * gamma[d, p, j]
                               - gamma[i, p, d] * gamma[d, i, j]
                       )
        return simplify(skeleton)

    @staticmethod
    def components_from_riemann(riemann: Riemann) -> SymbolArray:
        raise NotImplementedError("Ricci tensor from Riemann tensor is not yet implemented.")
