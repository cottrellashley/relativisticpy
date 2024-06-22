# Standard Library
from typing import Union
from itertools import product

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

    @classmethod
    def component_equations(cls):
        return (
            (SymbolArray, lambda arg: arg),
            (Metric, cls.components_from_metric),
            (LeviCivitaConnection, cls.components_from_connection),
            (Riemann, cls.components_from_riemann)
        )

    @classmethod
    def from_equation(cls, indices: CoordIndices, *args, **kwargs) -> 'Riemann':
        components = None
        metric = None

        # Categorize positional arguments
        for arg in args:
            if isinstance(arg, SymbolArray):
                components = arg
            elif isinstance(arg, cls):
                components = arg.reshape(indices).components
            elif isinstance(arg, Metric):
                metric = arg

        # Categorize keyword arguments
        for key, value in kwargs.items():
            if isinstance(value, SymbolArray):
                components = value
            elif isinstance(value, cls):
                components = value.reshape(indices).components
            elif isinstance(value, Metric):
                metric = value

        if components is None:
            if metric is not None:
                components = cls.components_from_metric(metric, "".join(["l" if idx.covariant else "u" for idx in indices.indices]))
            else:
                raise TypeError("Components or metric is required.")

        return cls(indices, components)

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

    def components(self, index_structure: str = None) -> SymbolArray:
        # TODO: Implement this method using the metric to raise and lower indices of this object.
        pass

