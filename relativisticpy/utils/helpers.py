from typing import Tuple
from operator import itemgetter
from itertools import product
from typing import List
from relativisticpy.typing import MetricType

import sympy as smp
from sympy import tensorproduct
from sympy import diff, Rational, zeros, simplify
from sympy import MutableDenseNDimArray as SymbolArray


def transpose_list(l):
    """
    This will transpose a list.

    Input : [[1,2],
             [3,4],
             [5,6]]

    Output: [[1,3,5],
             [2,4,6]]
    """
    return list(map(list, zip(*[i for i in l if i != None])))


def tensor_trace_product(a: SymbolArray, b: SymbolArray, trace: List[List[int]]):
    """
    Performs the tensor product by:
        1. Performing the tensor product.
        2. Taking the trace.
    """
    if len(trace) == 0:
        return tensorproduct(a, b)

    shape_a = a.shape
    shape_b = b.shape
    new_shape = [
        shape_a[0] for i in range(len(shape_b) + len(shape_a) - 2 * len(trace))
    ]
    summed_index_locations = transpose_list(trace)
    all = (
        [
            (idxs[: len(a.shape)], idxs[len(a.shape) :])
            for idxs in list(product(*[range(i) for i in a.shape + b.shape]))
            if itemgetter(*summed_index_locations[0])(idxs[: len(a.shape)])
            == itemgetter(*summed_index_locations[1])(idxs[len(a.shape) :])
        ]
        if len(summed_index_locations) > 0
        else [
            (idxs[: len(a.shape)], idxs[len(a.shape) :])
            for idxs in list(product(*[range(i) for i in a.shape + b.shape]))
        ]
    )

    result_indices = [i for i in range(len(shape_a) + len(shape_b) - 2 * len(trace))]
    res_a_indices = [
        i for i in range(len(shape_a)) if i not in summed_index_locations[0]
    ]
    res_b_indices = [
        i for i in range(len(shape_b)) if i not in summed_index_locations[1]
    ]

    def generator(idx):
        if not len(res_a_indices) == 0 and not len(res_b_indices) == 0:
            return [
                (IndicesA, IndicesB)
                for (IndicesA, IndicesB) in all
                if itemgetter(*res_a_indices)(IndicesA)
                == itemgetter(*result_indices[: len(res_a_indices)])(idx)
                and itemgetter(*res_b_indices)(IndicesB)
                == itemgetter(*result_indices[len(res_a_indices) :])(idx)
            ]
        elif len(res_a_indices) == 0 and not len(res_b_indices) == 0:
            return [
                (IndicesA, IndicesB)
                for (IndicesA, IndicesB) in all
                if itemgetter(*res_b_indices)(IndicesB)
                == itemgetter(*result_indices[: len(res_b_indices)])(idx)
            ]
        elif len(res_b_indices) == 0 and not len(res_a_indices) == 0:
            return [
                (IndicesA, IndicesB)
                for (IndicesA, IndicesB) in all
                if itemgetter(*res_a_indices)(IndicesA)
                == itemgetter(*result_indices[: len(res_a_indices)])(idx)
            ]
        else:
            return [(IndicesA, IndicesB) for IndicesA, IndicesB in all]

    func = lambda idcs: sum([a[i] * b[j] for i, j in generator(idcs)])
    zeros = SymbolArray.zeros(*new_shape)

    for i in list(product(*[range(i) for i in new_shape])):
        zeros[i] = func(i)

    return zeros


def connection_components_from_metric(metric: MetricType):
    D = metric.dimention
    empty = SymbolArray.zeros(D, D, D)
    g = metric._.components
    ig = metric.inv.components
    wrt = metric.basis
    for i, j, k, d in product(range(D), range(D), range(D), range(D)):
        empty[i, j, k] += (
            Rational(1, 2)
            * (ig[d, i])
            * (diff(g[k, d], wrt[j]) + diff(g[d, j], wrt[k]) - diff(g[j, k], wrt[d]))
        )
    return simplify(empty)


def riemann1000_components_from_metric(metric: MetricType):
    N = metric.dimention
    wrt = metric.basis
    C = connection_components_from_metric(metric)
    A = SymbolArray(zeros(N**4), (N, N, N, N))
    for i, j, k, p, d in product(range(N), range(N), range(N), range(N), range(N)):
        A[i, j, k, p] += Rational(1, N) * (
            diff(C[i, p, j], wrt[k]) - diff(C[i, k, j], wrt[p])
        ) + (C[i, k, d] * C[d, p, j] - C[i, p, d] * C[d, k, j])
    return simplify(A)


def riemann0000_components_from_metric(metric: MetricType):
    dim = metric.dimention
    metric_comps = metric.components
    rie_comps = riemann1000_components_from_metric(metric)
    A = SymbolArray(zeros(dim**4), (dim, dim, dim, dim))
    for i, j, k, p, d in product(
        range(dim), range(dim), range(dim), range(dim), range(dim)
    ):
        A[i, j, k, p] += metric_comps[i, d] * rie_comps[d, j, k, p]
    return simplify(A)


def ricci_components_from_metric(metric: MetricType):
    N = metric.dimention
    ig = metric.inv.components
    CR = riemann0000_components_from_metric(metric)
    A = SymbolArray(zeros(N**2), (N, N))
    for i, j, d, s in product(range(N), range(N), range(N), range(N)):
        A[i, j] += ig[d, s] * CR[d, i, s, j]
    return simplify(A)


def ricci_scalar(metric: MetricType):
    N = metric.dimention
    R = ricci_components_from_metric(metric)
    ig = metric.inv.components
    A = float()
    for d in range(N):
        for s in range(N):
            A += ig[d, s] * R[d, s]
    return simplify(A)


def kscalar_from_metric(metric: MetricType):
    """
    Warning: Very Slow! Indirect claculation with too many for loops.
    """
    N = metric.dimention
    R = riemann0000_components_from_metric(metric)
    ig = metric.inv.components
    A = float()
    for i, j, k, p, d, n, s, t in product(
        range(N), range(N), range(N), range(N), range(N), range(N), range(N), range(N)
    ):
        A += ig[i, j] * ig[k, p] * ig[d, n] * ig[s, t] * R[i, k, d, s] * R[j, p, d, s]
    return simplify(A)


######### DEPRICATED ########


class GrComputations:
    """
    Warning: Depricated class. Will be removed when it's dependencies are fully removed.
    """

    def __init__(self, Metric, Basis):
        self.Metric = Metric
        self.Basis = Basis
        self.Dimention = len(Basis)

    def Derivative(self):
        N = self.Dimention
        A = smp.MutableDenseNDimArray(smp.zeros(N**3), (N, N, N))
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    A[i, j, k] = smp.diff(self.Metric[j, k], self.Basis[i])
        return smp.simplify(A)

    def Ginv(self):
        N = self.Dimention
        g_m = self.Metric.tomatrix()
        inv_g = g_m.inv()
        A = smp.MutableDenseNDimArray(smp.zeros(N**2), (N, N))
        for i in range(N):
            for j in range(N):
                A[i, j] = inv_g[i, j]
        return smp.simplify(A)

    def Gamma(self):
        N = self.Dimention
        A = smp.MutableDenseNDimArray(smp.zeros(N**3), (N, N, N))
        ig = self.Ginv()
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for d in range(N):
                        A[i, j, k] += (
                            smp.Rational(1, 2)
                            * (ig[d, i])
                            * (
                                smp.diff(self.Metric[k, d], self.Basis[j])
                                + smp.diff(self.Metric[d, j], self.Basis[k])
                                - smp.diff(self.Metric[j, k], self.Basis[d])
                            )
                        )
        return smp.simplify(A)

    def TDerivative(self, V):
        N = self.Dimention
        A = smp.MutableDenseNDimArray(smp.zeros(N**3), (N, N, N))
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    A[i, j, k] = smp.diff(V[j, k], self.Basis[i])
        return smp.simplify(A)

    def CovariantD10(self, V):
        N = self.Dimention
        C = self.Gamma()
        A = smp.MutableDenseNDimArray(smp.zeros(N**2), (N, N))
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    A[i, j] += (
                        smp.Rational(1, N) * smp.diff(V[j], self.Basis[i])
                        + C[j, i, k] * V[k]
                    )
        return smp.simplify(A)

    def CovariantD01(self, V):
        N = self.Dimention
        C = self.Gamma()
        A = smp.MutableDenseNDimArray(smp.zeros(N**2), (N, N))
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    A[i, j] += (
                        smp.Rational(1, N) * smp.diff(V[j], self.Basis[i])
                        - C[k, i, j] * V[k]
                    )
        return smp.simplify(A)

    def CovariantD20(self, T):
        N = self.Dimention
        C = self.Gamma()
        A = smp.MutableDenseNDimArray(smp.zeros(N**3), (N, N, N))
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for p in range(N):
                        A[i, j, k] += (
                            smp.Rational(1, N) * smp.diff(T[j, k], self.Basis[i])
                            + C[j, i, p] * T[p, k]
                            + C[k, i, p] * T[p, j]
                        )
        return smp.simplify(A)

    def CovariantD02(self, T):
        N = self.Dimention
        C = self.Gamma()
        A = smp.MutableDenseNDimArray(smp.zeros(N**3), (N, N, N))
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for p in range(N):
                        A[i, j, k] += (
                            smp.Rational(1, N) * smp.diff(T[j, k], self.Basis[i])
                            - C[p, i, j] * T[p, k]
                            - C[p, i, k] * T[p, j]
                        )
        return smp.simplify(A)

    def CovariantD11(self, T):
        N = self.Dimention
        C = self.Gamma()
        A = smp.MutableDenseNDimArray(smp.zeros(N**3), (N, N, N))
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for p in range(N):
                        A[i, j, k] += (
                            smp.Rational(1, N) * smp.diff(T[j, k], self.Basis[i])
                            + C[j, i, p] * T[p, k]
                            - C[p, i, k] * T[j, p]
                        )
        return smp.simplify(A)

    def Riemann1000(self):
        N = self.Dimention
        C = self.Gamma()
        A = smp.MutableDenseNDimArray(smp.zeros(N**4), (N, N, N, N))
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for p in range(N):
                        for d in range(N):
                            A[i, j, k, p] += smp.Rational(1, N) * (
                                smp.diff(C[i, p, j], self.Basis[k])
                                - smp.diff(C[i, k, j], self.Basis[p])
                            ) + (C[i, k, d] * C[d, p, j] - C[i, p, d] * C[d, k, j])
        return smp.simplify(A)

    def Riemann0000(self):
        N = self.Dimention
        R = self.Riemann1000()
        A = smp.MutableDenseNDimArray(smp.zeros(N**4), (N, N, N, N))
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for p in range(N):
                        for d in range(N):
                            A[i, j, k, p] += self.Metric[i, d] * R[d, j, k, p]
        return smp.simplify(A)

    def Riemann1100(self):
        N = self.Dimention
        R = self.Riemann1000()
        I = self.Ginv()
        A = smp.MutableDenseNDimArray(smp.zeros(N**4), (N, N, N, N))
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for p in range(N):
                        for d in range(N):
                            A[j, i, k, p] += I[i, d] * R[j, d, k, p]
        return smp.simplify(A)

    def Riemann1110(self):
        N = self.Dimention
        R = self.Riemann1000()
        I = self.Ginv()
        A = smp.MutableDenseNDimArray(smp.zeros(N**4), (N, N, N, N))
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for p in range(N):
                        for d in range(N):
                            for s in range(N):
                                A[j, i, s, p] += I[s, k] * I[i, d] * R[j, d, k, p]
        return smp.simplify(A)

    def Riemann1111(self):
        N = self.Dimention
        R = self.Riemann1000()
        I = self.Ginv()
        A = smp.MutableDenseNDimArray(smp.zeros(N**4), (N, N, N, N))
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for p in range(N):
                        for d in range(N):
                            for s in range(N):
                                for n in range(N):
                                    A[j, i, s, n] += (
                                        I[n, p] * I[s, k] * I[i, d] * R[j, d, k, p]
                                    )
        return smp.simplify(A)

    def Ricci(self):
        N = self.Dimention
        ig = self.Ginv()
        CR = self.Riemann0000()
        A = smp.MutableDenseNDimArray(smp.zeros(N**2), (N, N))
        for i in range(N):
            for j in range(N):
                for d in range(N):
                    for s in range(N):
                        A[i, j] += ig[d, s] * CR[d, i, s, j]
        return smp.simplify(A)

    def ricscalar(self):
        N = self.Dimention
        R = self.Ricci()
        ig = self.Ginv()
        A = float()
        for d in range(N):
            for s in range(N):
                A += ig[d, s] * R[d, s]
        return smp.simplify(A)

    def kscalar(self):
        N = self.Dimention
        R = self.Riemann0000()
        ig = self.Ginv()
        A = float()
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for p in range(N):
                        for d in range(N):
                            for n in range(N):
                                for s in range(N):
                                    for t in range(N):
                                        A += (
                                            ig[i, j]
                                            * ig[k, p]
                                            * ig[d, n]
                                            * ig[s, t]
                                            * R[i, k, d, s]
                                            * R[j, p, d, s]
                                        )
        return smp.simplify(A)

    def Geodesic(self, t):
        N = self.Dimention
        C = self.Gamma()
        A = smp.MutableDenseNDimArray(smp.zeros(N), (N))
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    A[[i]] += smp.Rational(1, N**2) * (
                        smp.diff(self.Basis[i], t, t)
                    ) + C[i, j, k] * (smp.diff(self.Basis[j], t)) * (
                        smp.diff(self.Basis[k], t)
                    )
        return smp.simplify(A)

    def Lagrangian(self, t):
        N = self.Dimention
        A = float()
        for i in range(N):
            for j in range(N):
                A += (
                    -self.Metric[i, j]
                    * smp.diff(self.Basis[i], t)
                    * smp.diff(self.Basis[j], t)
                )

        return smp.sqrt(A)

    def EulerLagrange(self, t):
        L = self.Lagrangian(t)
        N = self.Dimention
        A = smp.MutableDenseNDimArray(smp.zeros(N), (N))
        for i in range(N):
            A[[i]] = smp.diff(self.Basis[i], t)

        B = smp.MutableDenseNDimArray(smp.zeros(N), (N))
        C = smp.MutableDenseNDimArray(smp.zeros(N), (N))
        for i in range(N):
            C[[i]] = smp.diff(L, A[i])
        for i in range(N):
            B[[i]] = C[i].subs(L, 1)

        D = smp.MutableDenseNDimArray(smp.zeros(N), (N))
        E = smp.MutableDenseNDimArray(smp.zeros(N), (N))
        for i in range(N):
            D[[i]] = smp.diff(L, self.Basis[i])
        for i in range(N):
            E[[i]] = D[i].subs(L, 1)

        F = smp.MutableDenseNDimArray(smp.zeros(N), (N))
        for i in range(N):
            F[[i]] = E[i] - smp.diff(B[i], t)

        return F

    def Einstein(self, StressEnergy, Cosmological, SI_Units):
        G = self.Metric
        T = StressEnergy
        N = self.Dimention
        Lambda = smp.symbols("Lambda")
        Ric = self.Ricci()
        RScalar = self.RicciScalar()
        E = smp.MutableDenseNDimArray(smp.zeros(N**2), (N, N))

        if Cosmological and not SI_Units:
            for i in range(N):
                for j in range(N):
                    E[i, j] = (
                        Ric[i, j]
                        - smp.Rational(1, 2) * G[i, j] * RScalar
                        + (Lambda) * G[i, j]
                        - (8 * smp.pi) * T[i, j]
                    )
            return E

        if Cosmological and SI_Units:
            for i in range(N):
                for j in range(N):
                    E[i, j] = (
                        Ric[i, j]
                        - smp.Rational(1, 2) * G[i, j] * RScalar
                        + (Lambda) * G[i, j]
                        - (8 * smp.pi) * T[i, j]
                    )
            return E

        if not Cosmological and not SI_Units:
            for i in range(N):
                for j in range(N):
                    E[i, j] = (
                        Ric[i, j]
                        - smp.Rational(1, 2) * G[i, j] * RScalar
                        - (8 * smp.pi) * T[i, j]
                    )
            return E

        if not Cosmological and SI_Units:
            for i in range(N):
                for j in range(N):
                    E[i, j] = (
                        Ric[i, j]
                        - smp.Rational(1, 2) * G[i, j] * RScalar
                        - (8 * smp.pi) * T[i, j]
                    )
            return E
