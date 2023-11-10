from relativisticpy.symengine import simplify

def kscalar(dim, riemann0000, inverse_metric):
    N = dim
    R = riemann0000
    ig = inverse_metric
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