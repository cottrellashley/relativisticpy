# Standard Library
from itertools import product

# External Modules
from relativisticpy.core import Indices, Metric
from relativisticpy.symengine import SymbolArray, Rational, zeros, diff, simplify

# This Module
from relativisticpy.gr.connection import Connection
from relativisticpy.gr.geometric_object import GeometricObject
from relativisticpy.gr.riemann import Riemann

class Ricci(GeometricObject):

    SYMBOL = "RicciSymbol"
    NAME = "Ricci"

    def __init__(self, indices : Indices, arg, basis: SymbolArray = None):
        super().__init__(indices = indices, symbols = arg, basis = basis)

    def from_connection(self, connection: Connection) -> SymbolArray:
        N = connection.dimention
        wrt = connection.basis
        Gamma = connection.components
        A = SymbolArray(zeros(N**2),(N,N))
        for mu, nu, lamda, sigma in product(range(N), range(N), range(N), range(N)):
            A[mu, nu] += Rational(1, N)*(diff(Gamma[lamda,mu,nu],wrt[lamda])-diff(Gamma[lamda,mu,lamda],wrt[nu]))+(Gamma[lamda,mu,sigma]*Gamma[sigma,nu,lamda]-Gamma[lamda,mu,nu]*Gamma[sigma,sigma,lamda])
        return simplify(A)

    def from_metric(self, metric: Metric) -> SymbolArray: 
        N = metric.dimention
        wrt = metric.basis
        Gamma = Connection.from_metric(metric)
        A = SymbolArray(zeros(N**2),(N,N))
        for mu, nu, lamda, sigma in product(range(N), range(N), range(N), range(N)):
            A[mu, nu] += Rational(1, N)*(diff(Gamma[lamda,mu,nu],wrt[lamda])-diff(Gamma[lamda,mu,lamda],wrt[nu]))+(Gamma[lamda,mu,sigma]*Gamma[sigma,nu,lamda]-Gamma[lamda,mu,nu]*Gamma[sigma,sigma,lamda])
        return simplify(A)

    def riemann0000(metric: Metric) -> SymbolArray:
        G = metric.components
        N = metric.dimention
        R = Riemann.comps_from_metric(metric)
        RL = SymbolArray(zeros(N**4),(N,N,N,N))
        for i, j, k, p, d in product(range(N), range(N), range(N), range(N), range(N)):
            RL[i,j,k,p] += G[i,d]*R[d,j,k,p]
        return RL
