# Standard Library
from typing import Union
from itertools import product

# External Modules
from relativisticpy.core import Indices, MultiIndexObject, Metric, einstein_convention
from relativisticpy.providers import SymbolArray, Rational, zeros, diff, simplify

# This Module
from relativisticpy.gr.connection import Connection
from relativisticpy.gr.geometric_object import GeometricObject

@einstein_convention
class Riemann(GeometricObject):
    # __getitem__() => Riemann[Indices] => Riemann components corresponding to the Indices structure provided by the Indices.
    
    def __init__(self, indices: Indices, arg, basis: SymbolArray = None):
        super().__init__( symbols = arg, indices = indices, basis = basis)

    def from_components(self, components) -> SymbolArray: return components

    def from_metric(self, metric: Metric) -> SymbolArray:
        N = metric.dimention
        wrt = metric.basis
        C = Connection.from_metric(metric)
        A = SymbolArray(zeros(N**4),(N,N,N,N))
        for i, j, k, p, d in product(range(N), range(N), range(N), range(N), range(N)):
            A[i, j, k, p] += Rational(1, N)*(diff(C[i,p,j],wrt[k])-diff(C[i,k,j],wrt[p]))+(C[i,k,d]*C[d,p,j]-C[i,p,d]*C[d,k,j])
        return simplify(A)

    def from_connection(self, connection: Connection) -> SymbolArray:
        N = connection.dimention
        wrt = connection.basis
        C = connection.components
        A = SymbolArray(zeros(N**4),(N,N,N,N))
        for i, j, k, p, d in product(range(N), range(N), range(N), range(N), range(N)):
            A[i, j, k, p] += Rational(1, N)*(diff(C[i,p,j],wrt[k])-diff(C[i,k,j],wrt[p]))+(C[i,k,d]*C[d,p,j]-C[i,p,d]*C[d,k,j])
        return simplify(A)