# Standard Library
from functools import singledispatchmethod
from typing import Union
from itertools import product

from loguru import logger

from relativisticpy.algebras import EinsumArray, Indices
# External Modules
from relativisticpy.diffgeom.manifold import CoordIndices
from relativisticpy.diffgeom.tensor import Tensor
from relativisticpy.diffgeom.metric import Metric
from relativisticpy.symengine import SymbolArray, Rational, zeros, diff, simplify

# This Module
from relativisticpy.diffgeom.connection import LeviCivitaConnection

# TODO: Add mechanism which detects self-contractions and map any self contraction to relevant tensor i.e. Riemann^{a}_{b}_{a}_{c} == Ricci_{b}_{c}


class Riemann(Tensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @singledispatchmethod
    @classmethod
    def _new(cls, operand, indices):
        logger.debug(f"[MetricScalar] Handling init: {operand.__class__.__name__}")

    @_new.register
    @classmethod
    def _(cls, operand: Metric, indices: Indices) -> Tensor:
        components = cls.components_from_metric(operand, 'ulll')
        return cls(indices, components)

    @_new.register
    @classmethod
    def _(cls, operand: LeviCivitaConnection, indices: Indices) -> Tensor:
        raise NotImplementedError("Riemann tensor from connection is not yet implemented.")

    @staticmethod
    def components_from_metric(metric: Metric, index_structure: str = None) -> SymbolArray:
        if index_structure == 'llll':
            return Riemann.llll_components_from_metric(metric)
        elif index_structure == 'ulll':
            return Riemann.ulll_components_from_metric(metric)
        else:
            raise NotImplementedError("Only 'llll' and 'ulll' index structures have currently been implemented.")

    @classmethod
    def ulll_components_from_metric(cls, metric: Metric):
        dim = metric.dimention
        wrt = metric.indices.basis
        gamma = LeviCivitaConnection.components_from_metric(metric)
        skeleton = SymbolArray(zeros(dim**4), (dim, dim, dim, dim))
        for i, j, k, p, d in product(range(dim), range(dim), range(dim), range(dim), range(dim)):
            skeleton[i, j, k, p] += Rational(1, dim) * (
                diff(gamma[i, p, j], wrt[k]) - diff(gamma[i, k, j], wrt[p])
            ) + (gamma[i, k, d] * gamma[d, p, j] - gamma[i, p, d] * gamma[d, k, j])
        return simplify(skeleton)

    @classmethod
    def llll_components_from_metric(cls, metric: Metric):
        dim = metric.dimention
        metric_comps = metric.components
        rie_comps = Riemann.ulll_components_from_metric(metric)
        skeleton = SymbolArray(zeros(dim**4), (dim, dim, dim, dim))
        for i, j, k, p, d in product(
            range(dim), range(dim), range(dim), range(dim), range(dim)
        ):
            skeleton[i, j, k, p] += metric_comps[i, d] * rie_comps[d, j, k, p]
        return simplify(skeleton)

    @classmethod
    def components_from_connection(cls, connection: LeviCivitaConnection) -> SymbolArray:
        raise NotImplementedError("Riemann tensor from connection is not yet implemented.")
