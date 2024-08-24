from functools import singledispatchmethod
from itertools import product

from loguru import logger

from relativisticpy.algebras import EinsumArray, Indices
from relativisticpy.diffgeom.tensor import Tensor
from relativisticpy.diffgeom.manifold import CoordIndices
from relativisticpy.diffgeom.metric import Metric
from relativisticpy.diffgeom.connection import LeviCivitaConnection
from relativisticpy.diffgeom.tensors.ricci import Ricci
from relativisticpy.diffgeom.tensors.riemann import Riemann
from relativisticpy.symengine import SymbolArray, simplify, Rational, zeros, diff


class RicciScalar(Tensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @singledispatchmethod
    @classmethod
    def _new(cls, operand, indices):
        logger.debug(f"[MetricScalar] Handling init: {operand.__class__.__name__}")

    @_new.register
    @classmethod
    def _(cls, operand: Metric, indices: Indices) -> Tensor:
        dim = operand.dimention
        ric = Ricci.components_from_metric(operand)
        inv_metric = operand.uu_components
        skeleton = float()
        for i, j in product(range(dim), range(dim)):
            skeleton += inv_metric[i, j] * ric[i, j]
        components = simplify(skeleton)
        return cls(indices, components)

    @_new.register
    @classmethod
    def _(cls, operand: LeviCivitaConnection, indices: Indices) -> Tensor:
        metric = operand.metric
        ric = Ricci.components_from_connection(operand)
        ig = metric.uu_components
        skeleton = float()
        for i, j in product(range(operand.dimention), range(operand.dimention)):
            skeleton += ig[i, j] * ric[i, j]
        components = simplify(skeleton)

        return cls(indices, components)

    @_new.register
    @classmethod
    def _(cls, operand: Riemann, indices: Indices) -> Tensor:
        raise NotImplementedError("Ricci tensor from Riemann tensor is not yet implemented.")

    @staticmethod
    def components_from_metric(metric: Metric):
        dim = metric.dimention
        ric = Ricci.components_from_metric(metric)
        inv_metric = metric.uu_components
        skeleton = float()
        for i, j in product(range(dim), range(dim)):
            skeleton += inv_metric[i, j] * ric[i, j]
        return simplify(skeleton)

    @staticmethod
    def components_from_connection(connection: LeviCivitaConnection):
        metric = connection.metric
        ric = Ricci.components_from_connection(connection)
        ig = metric.uu_components
        skeleton = float()
        for i, j in product(range(connection.dimention), range(connection.dimention)):
            skeleton += ig[i, j] * ric[i, j]
        return simplify(skeleton)
