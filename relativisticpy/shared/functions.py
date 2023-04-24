from sympy import zeros, diff, simplify, Rational, sqrt
from sympy import MutableDenseNDimArray as Array
from relativisticpy.deserializers.sympify import Sympify

#################################################
########## Arbitrary Helper Methods #############
#################################################

def sympify(string: str):
    parsed_object = Sympify().parse(string)
    return parsed_object

def transpose_list(l):
    """
    This will transpose a list.

    Input : [[1,2],[3,4],[5,6]]
    Output: [[1,3,5],[2,4,6]]
    """
    return list(map(list, zip(*l)))

#################################################
###### General Relativity Helper Methods ########
#################################################

def derivative(Metric, Basis):
    N = len(Basis)
    A = Array(zeros(N**3),(N,N,N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                A[i,j,k] = diff(Metric[j,k],Basis[i])
    return simplify(A)


def invert_metric(Metric, Basis):
    N = len(Basis)
    g_m = Metric.tomatrix()
    inv_g = g_m.inv()
    A = Array(zeros(N**2),(N,N))
    for i in range(N):
        for j in range(N):
            A[i,j] = inv_g[i, j]
    return simplify(A)


def christoffel(Metric, Basis):
    N = len(Basis)
    A = Array(zeros(N**3),(N,N,N))
    ig = invert_metric(Metric, Basis)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for d in range(N):
                    A[i, j, k] += Rational(1, 2)*(ig[d,i])*(diff(Metric[k,d],Basis[j]) + diff(Metric[d,j],Basis[k]) - diff(Metric[j,k],Basis[d]))
    return simplify(A)

def Tderivative(Metric, Basis, V):
    N = len(Basis)
    A = Array(zeros(N**3),(N,N,N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                A[i,j,k] = diff(V[j,k],Basis[i])
    return simplify(A)

def covariantD10(Metric, Basis, V):
    N = len(Basis)
    C = christoffel()
    A = Array(zeros(N**2),(N,N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                A[i,j] += Rational(1, N)*diff(V[j],Basis[i]) + C[j,i,k]*V[k]
    return simplify(A)


def covariantD01(Metric, Basis, V):
    N = len(Basis)
    C = christoffel()
    A = Array(zeros(N**2),(N,N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                A[i,j] += Rational(1, N)*diff(V[j],Basis[i]) - C[k,i,j]*V[k]
    return simplify(A)


def covariantD20(Metric, Basis, T):
    N = len(Basis)
    C = christoffel()
    A = Array(zeros(N**3),(N,N,N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for p in range(N):
                    A[i,j,k] += Rational(1, N)*diff(T[j,k],Basis[i]) + C[j,i,p]*T[p,k] + C[k,i,p]*T[p,j]
    return simplify(A)


def covariantD02(Metric, Basis, T):
    N = len(Basis)
    C = christoffel()
    A = Array(zeros(N**3),(N,N,N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for p in range(N):
                    A[i,j,k] += Rational(1, N)*diff(T[j,k],Basis[i]) - C[p,i,j]*T[p,k] - C[p,i,k]*T[p,j]
    return simplify(A)


def covariantD11(Metric, Basis, T):
    N = len(Basis)
    C = christoffel()
    A = Array(zeros(N**3),(N,N,N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for p in range(N):
                    A[i,j,k] += Rational(1, N)*diff(T[j,k],Basis[i]) + C[j,i,p]*T[p,k] - C[p,i,k]*T[j,p]
    return simplify(A)

def riemann1000(Metric, Basis):
    N = len(Basis)
    C = christoffel(Metric, Basis)
    A = Array(zeros(N**4),(N,N,N,N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for p in range(N):
                    for d in range(N):
                        A[i, j, k, p] += Rational(1, N)*(diff(C[i,p,j],Basis[k]) - diff(C[i,k,j],Basis[p]))+(C[i,k,d]*C[d,p,j]-C[i,p,d]*C[d,k,j])
    return simplify(A)

def riemann0000(Metric, Basis):
    N = len(Basis)
    R = riemann1000(Metric, Basis)
    A = Array(zeros(N**4),(N,N,N,N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for p in range(N):
                    for d in range(N):
                        A[i,j,k,p] += Metric[i,d]*R[d,j,k,p]
    return simplify(A)

def riemann1100(Metric, Basis):
    N = len(Basis)
    R = riemann1000(Metric, Basis)
    I = invert_metric(Metric, Basis)
    A = Array(zeros(N**4),(N,N,N,N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for p in range(N):
                    for d in range(N):
                        A[j,i,k,p] += I[i,d]*R[j,d,k,p]
    return simplify(A)

def riemann1110(Metric, Basis):
    N = len(Basis)
    R = riemann1000(Metric, Basis)
    I = invert_metric(Metric, Basis)
    A = Array(zeros(N**4),(N,N,N,N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for p in range(N):
                    for d in range(N):
                        for s in range(N):
                            A[j,i,s,p] += I[s,k]*I[i,d]*R[j,d,k,p]
    return simplify(A)

def riemann1111(Metric, Basis):
    N = len(Basis)
    R = riemann1000(Metric, Basis)
    I = invert_metric(Metric, Basis)
    A = Array(zeros(N**4),(N,N,N,N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for p in range(N):
                    for d in range(N):
                        for s in range(N):
                            for n in range(N):
                                A[j,i,s,n] += I[n,p]*I[s,k]*I[i,d]*R[j,d,k,p]
    return simplify(A)

def ricci(Metric, Basis):
    N = len(Basis)
    ig = invert_metric(Metric, Basis)
    CR = riemann0000(Metric, Basis)
    A = Array(zeros(N**2),(N,N))
    for i in range(N):
        for j in range(N):
            for d in range(N):
                for s in range(N):
                    A[i,j] += ig[d,s]*CR[d,i,s,j]
    return simplify(A)


def ricci_scalar(Metric, Basis):
    N = len(Basis)
    R = ricci(Metric, Basis)
    ig = invert_metric(Metric, Basis)
    A = float()
    for d in range(N):
        for s in range(N):
            A += ig[d,s]*R[d,s]
    return simplify(A)


def kscalar(Metric, Basis):
    N = len(Basis)
    R = riemann0000(Metric, Basis)
    ig = invert_metric(Metric, Basis)
    A = float()
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for p in range(N):
                    for d in range(N):
                        for n in range(N):
                            for s in range(N):
                                for t in range(N):
                                    A += ig[i,j]*ig[k,p]*ig[d,n]*ig[s,t]*R[i,k,d,s]*R[j,p,d,s]
    return simplify(A)


def geodesic(Metric, Basis,t):
    N = len(Basis)
    C = christoffel(Metric, Basis)
    A = Array(zeros(N),(N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                A[[i]] += Rational(1, N**2)*(diff(Basis[i], t, t)) + C[i, j, k]*(diff(Basis[j], t))*(diff(Basis[k], t))
    return simplify(A)


def lagrangian(Metric, Basis, t):
    N = len(Basis)
    A = float()
    for i in range(N):
        for j in range(N):
            A += - Metric[i,j]*diff(Basis[i], t)*diff(Basis[j], t)

    return sqrt(A)


def EulerLagrange(Metric, Basis,t):
    L = lagrangian(t)
    N = len(Basis)
    A = Array(zeros(N),(N))
    for i in range(N):
        A[[i]] = diff(Basis[i],t)
    
    
    B = Array(zeros(N),(N))
    C = Array(zeros(N),(N))
    for i in range(N):
        C[[i]] = diff(L, A[i])
    for i in range(N):
        B[[i]] = C[i].subs(L, 1)


    D = Array(zeros(N),(N))
    E = Array(zeros(N),(N))
    for i in range(N):
        D[[i]] = diff(L, Basis[i])
    for i in range(N):
        E[[i]] = D[i].subs(L, 1)

    F = Array(zeros(N),(N))
    for i in range(N):
        F[[i]] = E[i] - diff(B[i],t)
    
    return F
    
# def Einstein(Metric, Basis, StressEnergy, Cosmological, SI_Units):
#     G = Metric
#     T = StressEnergy
#     N = len(Basis)
#     Lambda  = symbols('Lambda')
#     Ric = ricci()
#     RScalar = ricciScalar()
#     E = Array(zeros(N**2),(N,N))
    
#     if Cosmological and not SI_Units: 
#         for i in range(N):
#             for j in range(N):
#                 E[i,j] = Ric[i,j] - Rational(1,2)*G[i,j]*RScalar + (Lambda)*G[i,j] - (8*pi)*T[i,j]
#         return E
        
#     if Cosmological and SI_Units: 
#         for i in range(N):
#             for j in range(N):
#                 E[i,j] = Ric[i,j] - Rational(1,2)*G[i,j]*RScalar + (Lambda)*G[i,j] - (8*pi)*T[i,j]
#         return E
        
#     if not Cosmological and not SI_Units: 
#         for i in range(N):
#             for j in range(N):
#                 E[i,j] = Ric[i,j] - Rational(1,2)*G[i,j]*RScalar - (8*pi)*T[i,j]
#         return E
        
#     if not Cosmological and SI_Units: 
#         for i in range(N):
#             for j in range(N):
#                 E[i,j] = Ric[i,j] - Rational(1,2)*G[i,j]*RScalar - (8*pi)*T[i,j]
#         return E