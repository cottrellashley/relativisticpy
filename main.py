from relativisticpy.gr import Connection, Ricci, Riemann
from relativisticpy.core import Indices, Idx, MultiIndexObject, Metric
from relativisticpy.providers import Sympify


if __name__ == '__main__':

    basis = Sympify('[t, r]')

    test = Metric.from_string('_{a}_{b}','[[1,r],[0,r**2]]', '[r, theta]')

    idcs = Indices(-Idx('a'), Idx('b'), Idx('c'), Idx('d'))

    rie = Riemann(idcs, test, basis)

    print(rie[Indices(Idx('a'), Idx('b'), Idx('c'), Idx('d'))])

    print(rie)