import sympy as smp


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
