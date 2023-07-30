# Standard Library
from typing import Union
from itertools import product

# External Modules
from relativisticpy.core import Indices, MultiIndexObject
from relativisticpy.providers import SymbolArray, Rational, zeros, diff, simplify

# This Module
from relativisticpy.gr.metric import Metric
from relativisticpy.gr.connection import Connection

class Riemann(MultiIndexObject):
    # __getitem__() => Riemann[Indices] => Riemann components corresponding to the Indices structure provided by the Indices.

    @classmethod
    def from_metric(metric: Metric) -> 'Riemann':
        connection_comps = Connection._compute_comps(metric)
        N = metric.dimention
        wrt = metric.basis
        C = connection_comps
        A = SymbolArray(zeros(N**4),(N,N,N,N))
        for i, j, k, p, d in product(range(N), range(N), range(N), range(N), range(N)):
            A[i, j, k, p] += Rational(1, N)*(diff(C[i,p,j],wrt[k])-diff(C[i,k,j],wrt[p]))+(C[i,k,d]*C[d,p,j]-C[i,p,d]*C[d,k,j])
        return simplify(A)

    @classmethod
    def from_connection(connection: Connection) -> 'Riemann':
        N = connection.dimention
        wrt = connection.basis
        C = connection.components
        A = SymbolArray(zeros(N**4),(N,N,N,N))
        for i, j, k, p, d in product(range(N), range(N), range(N), range(N), range(N)):
            A[i, j, k, p] += Rational(1, N)*(diff(C[i,p,j],wrt[k])-diff(C[i,k,j],wrt[p]))+(C[i,k,d]*C[d,p,j]-C[i,p,d]*C[d,k,j])
        return simplify(A)
    
    def __init__(self, indices: Indices, arg: Union[Metric, Connection, SymbolArray] = None, basis: SymbolArray = None):
        if isinstance(arg, (Metric, Connection, SymbolArray)):
            comps = arg if isinstance(arg, SymbolArray) else Riemann.from_metric(arg) if isinstance(arg, Metric) else Riemann.from_connection(arg)
            basis = basis if basis != None else arg.basis
            super().__init__(
                                components  =   comps, 
                                indices     =   indices,
                                basis       =   basis
                            )
