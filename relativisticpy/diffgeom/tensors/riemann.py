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
    def from_equation(cls, indices: CoordIndices, *args, **kwargs) -> 'LeviCivitaConnection':
        "Dynamic constructor for the inheriting classes."

        components = None
        metric = None

        # Categorize positional arguments
        for arg in args:
            if isinstance(arg, SymbolArray):
                components = arg
            elif isinstance(arg, Metric):
                metric = arg

        # Categorize keyword arguments
        for _, value in kwargs.items():
            if isinstance(value, SymbolArray):
                components = arg
            elif isinstance(value, Metric):
                metric = value

        if components is None:
            if metric is not None:
                components = cls.components_from_metric(metric)
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
        N = metric.dimention
        wrt = metric.basis
        C = LeviCivitaConnection.componens_from_metric(metric)
        A = SymbolArray(zeros(N**4), (N, N, N, N))
        for i, j, k, p, d in product(range(N), range(N), range(N), range(N), range(N)):
            A[i, j, k, p] += Rational(1, N) * (
                diff(C[i, p, j], wrt[k]) - diff(C[i, k, j], wrt[p])
            ) + (C[i, k, d] * C[d, p, j] - C[i, p, d] * C[d, k, j])
        return simplify(A)

    @classmethod
    def llll_components_from_metric(cls, metric: Metric):
        dim = metric.dimention
        metric_comps = metric.components
        rie_comps = Riemann.ulll_components_from_metric(metric)
        A = SymbolArray(zeros(dim**4), (dim, dim, dim, dim))
        for i, j, k, p, d in product(
            range(dim), range(dim), range(dim), range(dim), range(dim)
        ):
            A[i, j, k, p] += metric_comps[i, d] * rie_comps[d, j, k, p]
        return simplify(A)

    def from_connection(connection: LeviCivitaConnection) -> SymbolArray:
        # Setup Relevant quantities for computation
        N = connection.dimention
        wrt = connection.basis
        C = connection.components
        A = SymbolArray(zeros(N**4), (N, N, N, N))

        # Perform computation
        for i, j, k, p, d in product(range(N), range(N), range(N), range(N), range(N)):
            A[i, j, k, p] += Rational(1, N) * (
                diff(C[i, p, j], wrt[k]) - diff(C[i, k, j], wrt[p])
            ) + (C[i, k, d] * C[d, p, j] - C[i, p, d] * C[d, k, j])

        # Simplify object on before returning.
        return simplify(A)
    
    @property
    def ll_ricci(self):
        return self.components
    
    @property
    def uu_ricci(self):
        return self.uu_components

    @property
    def llll_components(self):
        return self.components
    
    def components(self, index_structure: str = 'llll') -> SymbolArray:
        if index_structure == 'llll':
            return self.llll_components
        elif index_structure == 'ulll':
            return self.ulll_components
        else:
            raise NotImplementedError("Only 'llll' and 'ulll' index structures have currently been implemented.")
