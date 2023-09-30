# Standard Library
from itertools import product
from typing import Union

# External Modules
from relativisticpy.core import Indices, MultiIndexObject, Metric
from relativisticpy.providers import SymbolArray, Rational, zeros, diff, simplify

# This Module
from relativisticpy.gr.connection import Connection
from relativisticpy.gr.riemann import Riemann

class Ricci(MultiIndexObject):

    def from_metric(metric: Metric) -> 'Ricci':
        N = metric.dimention
        ig = metric.inv.components
        CR = Riemann.comps_from_metric(metric)
        A = SymbolArray(zeros(N**2),(N,N))
        for i, j, d, s in product(range(N), range(N), range(N), range(N)):
            A[i,j] += ig[d,s]*CR[d,i,s,j]
        return simplify(A)

    def from_connection(connection: Connection) -> 'Ricci':
        return None

    def from_riemann(riemann: Riemann) -> 'Ricci':
        return None
    
    def __init__(self, indices : Indices, arg : Union[Metric, Connection]):
        super().__init__(indices = indices, components = Ricci.from_metric(arg), basis = arg.basis)
