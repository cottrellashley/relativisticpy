# Standard Library
from typing import Union
from itertools import product

# External Modules
from relativisticpy.core import Indices, MultiIndexObject, einstein_convention
from relativisticpy.providers import SymbolArray, Rational, zeros, diff, simplify

# This Module
from relativisticpy.gr.metric import Metric
from relativisticpy.gr.connection import Connection

@einstein_convention
class Riemann(MultiIndexObject):
    # __getitem__() => Riemann[Indices] => Riemann components corresponding to the Indices structure provided by the Indices.

    @classmethod
    def comps_from_metric(cls, metric: Metric) -> SymbolArray:
        N = metric.dimention
        wrt = metric.basis
        C = Connection.comps_from_metric(metric)
        A = SymbolArray(zeros(N**4),(N,N,N,N))
        for i, j, k, p, d in product(range(N), range(N), range(N), range(N), range(N)):
            A[i, j, k, p] += Rational(1, N)*(diff(C[i,p,j],wrt[k])-diff(C[i,k,j],wrt[p]))+(C[i,k,d]*C[d,p,j]-C[i,p,d]*C[d,k,j])
        return simplify(A)

    @classmethod
    def comps_from_connection(cls, connection: Connection) -> SymbolArray:
        N = connection.dimention
        wrt = connection.basis
        C = connection.components
        A = SymbolArray(zeros(N**4),(N,N,N,N))
        for i, j, k, p, d in product(range(N), range(N), range(N), range(N), range(N)):
            A[i, j, k, p] += Rational(1, N)*(diff(C[i,p,j],wrt[k])-diff(C[i,k,j],wrt[p]))+(C[i,k,d]*C[d,p,j]-C[i,p,d]*C[d,k,j])
        return simplify(A)
    
    def __init__(self, indices: Indices, arg: Union[Metric, Connection] = None, basis: SymbolArray = None):
        self.indices = indices

        if isinstance(arg, Metric):
            self.metric = arg

        if isinstance(arg, (Metric, Connection, SymbolArray)):
            self.comps = Riemann.comps_from_metric(arg) if isinstance(arg, Metric) else Riemann.comps_from_connection(arg)
            self.basis = basis if basis != None else arg.basis
            super().__init__(
                                components  =   self.comps, 
                                indices     =   self.indices,
                                basis       =   self.basis
                            )