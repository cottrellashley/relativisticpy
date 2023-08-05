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
    def comps_from_metric(metric: Metric) -> SymbolArray:
        N = metric.dimention
        wrt = metric.basis
        C = Connection._compute_comps(metric)
        A = SymbolArray(zeros(N**4),(N,N,N,N))
        for i, j, k, p, d in product(range(N), range(N), range(N), range(N), range(N)):
            A[i, j, k, p] += Rational(1, N)*(diff(C[i,p,j],wrt[k])-diff(C[i,k,j],wrt[p]))+(C[i,k,d]*C[d,p,j]-C[i,p,d]*C[d,k,j])
        return simplify(A)

    @classmethod
    def comps_from_connection(connection: Connection) -> SymbolArray:
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
            self.comps = Riemann.from_metric(arg) if isinstance(arg, Metric) else Riemann.from_connection(arg)
            self.basis = basis if basis != None else arg.basis
            super().__init__(
                                components  =   self.comps, 
                                indices     =   self.indices,
                                basis       =   self.basis
                            )

    def __getitem__(self, idcs: Indices):
        # This should be implemented as follows:
        # 1. If the indices cov and contravarient indices structure matches the self.indices, then just return the current components
        # 2. If the indices does not match the self.indices, we must then perform a summation with the metric tensor in order to return the components
        #    which represent the indices structure given by the input parameter
        res = self.comps
        deltas = self.indices.covariance_delta(idcs) # { 'raise': [0,1], 'lower': [3] }
        for delta in deltas:
            res = getattr(self.metric, delta[0])(res, delta[1])
        return res[idcs.__index__()]