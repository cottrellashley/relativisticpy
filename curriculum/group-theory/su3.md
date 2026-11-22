# SU(3)

The group of the eightfold way and of colour: the dimensions of its representations from
their Dynkin labels, the eight Gell-Mann matrices \(\lambda_a\) with
\(\operatorname{tr}(\lambda_a \lambda_b) = 2\delta_{ab}\), the Casimir operators of the
fundamental and of the adjoint representation, and a product of representations named,
as physicists name them, by their dimensions.

## The dimension of the decuplet

**Level 1** · `group.su3.decuplet` · standard

The dimension of the representation of SU(3) with Dynkin labels \((p, q)\),
\(\frac12 (p+1)(q+1)(p+q+2)\), at \((3, 0)\): the ten baryons of the decuplet.

```latex
\left. \frac{1}{2}(p+1)(q+1)(p+q+2) \right|_{p=3,\, q=0}
```

```python
from itertools import product

expected = Rational(1, 2) * (3 + 1) * (0 + 1) * (3 + 0 + 2)
assert expected == 10
# (3, 0) is the symmetric product of three fundamentals: the monomials of degree 3 in 3 variables
assert len({tuple(sorted(w)) for w in product(range(3), repeat=3)}) == 10
```

## The Casimir operator of the fundamental representation

**Level 3** · `group.su3.casimir` · Georgi (1999), chapter 7

The eight Gell-Mann matrices, each named with a subscript, and a sum over the family:
\(\sum_a (\lambda_a / 2)^2 = C_F \mathbb{1}\), with \(C_F = \frac43\) for SU(3).

```latex
\lambda_1 := \begin{pmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}
\lambda_2 := \begin{pmatrix} 0 & -i & 0 \\ i & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}
\lambda_3 := \begin{pmatrix} 1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & 0 \end{pmatrix}
\lambda_4 := \begin{pmatrix} 0 & 0 & 1 \\ 0 & 0 & 0 \\ 1 & 0 & 0 \end{pmatrix}
\lambda_5 := \begin{pmatrix} 0 & 0 & -i \\ 0 & 0 & 0 \\ i & 0 & 0 \end{pmatrix}
\lambda_6 := \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & 1 \\ 0 & 1 & 0 \end{pmatrix}
\lambda_7 := \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & -i \\ 0 & i & 0 \end{pmatrix}
\lambda_8 := \frac{1}{\sqrt{3}} \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & -2 \end{pmatrix}
\sum_{a=1}^{8} \left(\frac{\lambda_a}{2}\right)^2
```

```python
lam = [
    Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
    Matrix([[0, -I, 0], [I, 0, 0], [0, 0, 0]]),
    Matrix([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
    Matrix([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
    Matrix([[0, 0, -I], [0, 0, 0], [I, 0, 0]]),
    Matrix([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
    Matrix([[0, 0, 0], [0, 0, -I], [0, I, 0]]),
    Matrix([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / sqrt(3),
]
assert all(m == m.H and m.trace() == 0 for m in lam)  # Hermitian and traceless
assert all(
    (lam[a] * lam[b]).trace() == 2 * KroneckerDelta(a, b) for a in range(8) for b in range(8)
)
expected = sum(((m / 2) ** 2 for m in lam), zeros(3, 3))
N_c = 3
assert expected == Rational(N_c**2 - 1, 2 * N_c) * eye(3) == Rational(4, 3) * eye(3)
```

## The Casimir operator of the adjoint representation

**Level 4** · `group.su3.adjoint-casimir` · standard

The structure constants from the Gell-Mann matrices by a trace,
\(f_{abc} = -\frac{i}{4} \operatorname{tr}([\lambda_a, \lambda_b] \lambda_c)\), then the
Casimir operator of the adjoint representation in index notation:
\(f_{acd} f_{bcd} = 3\delta_{ab}\), \(C_A = N = 3\). The index of \(\lambda_a\) counts the
matrices.

```latex
a, b, c, d = 1, \ldots, 8
\lambda_1 := \begin{pmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}
\lambda_2 := \begin{pmatrix} 0 & -i & 0 \\ i & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}
\lambda_3 := \begin{pmatrix} 1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & 0 \end{pmatrix}
\lambda_4 := \begin{pmatrix} 0 & 0 & 1 \\ 0 & 0 & 0 \\ 1 & 0 & 0 \end{pmatrix}
\lambda_5 := \begin{pmatrix} 0 & 0 & -i \\ 0 & 0 & 0 \\ i & 0 & 0 \end{pmatrix}
\lambda_6 := \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & 1 \\ 0 & 1 & 0 \end{pmatrix}
\lambda_7 := \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & -i \\ 0 & i & 0 \end{pmatrix}
\lambda_8 := \frac{1}{\sqrt{3}} \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & -2 \end{pmatrix}
f_{abc} := -\frac{i}{4} \operatorname{tr}\left([\lambda_a, \lambda_b] \lambda_c\right)
f_{acd} f_{bcd} = 3 \delta_{ab}
```

```python
from itertools import product

lam = [
    Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
    Matrix([[0, -I, 0], [I, 0, 0], [0, 0, 0]]),
    Matrix([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
    Matrix([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
    Matrix([[0, 0, -I], [0, 0, 0], [I, 0, 0]]),
    Matrix([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
    Matrix([[0, 0, 0], [0, 0, -I], [0, I, 0]]),
    Matrix([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / sqrt(3),
]
f = {
    (a, b, c): simplify(-I / 4 * ((lam[a] * lam[b] - lam[b] * lam[a]) * lam[c]).trace())
    for a, b, c in product(range(8), repeat=3)
}
# the standard table: f_123 = 1, f_147 = f_246 = f_257 = f_345 = 1/2,
# f_156 = f_367 = -1/2, f_458 = f_678 = sqrt(3)/2, and their antisymmetric permutations
table = {(1, 2, 3): 1, (1, 4, 7): S.Half, (2, 4, 6): S.Half, (2, 5, 7): S.Half}
table |= {(3, 4, 5): S.Half, (1, 5, 6): -S.Half, (3, 6, 7): -S.Half}
table |= {(4, 5, 8): sqrt(3) / 2, (6, 7, 8): sqrt(3) / 2}
for (a, b, c), value in table.items():
    for p, q, r in ((a, b, c), (b, c, a), (c, a, b)):
        assert f[p - 1, q - 1, r - 1] == value and f[q - 1, p - 1, r - 1] == -value
assert sum(1 for v in f.values() if v != 0) == 6 * len(table)  # nothing else
casimir = Matrix(
    8, 8, lambda a, b: sum(f[a, c, d] * f[b, c, d] for c in range(8) for d in range(8))
)
assert simplify(casimir) == 3 * eye(8)
expected = True
```

## A quark and an antiquark

**Level 5** · `group.su3.octet-singlet` · standard

Representations named by their dimensions in bold, and a claim about their product:
\(\mathbf{3} \otimes \bar{\mathbf{3}} = \mathbf{8} \oplus \mathbf{1}\) in SU(3), the meson
octet and singlet. No design of the language has representations as values.

```latex
\mathbf{3} \otimes \bar{\mathbf{3}} = \mathbf{8} \oplus \mathbf{1} \text{ in } SU(3)
```

```python
t_1, t_2 = symbols("t_1 t_2", nonzero=True)
g = diag(t_1, t_2, 1 / (t_1 * t_2))  # the maximal torus of SU(3)
lam = [
    Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
    Matrix([[0, -I, 0], [I, 0, 0], [0, 0, 0]]),
    Matrix([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
    Matrix([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
    Matrix([[0, 0, -I], [0, 0, 0], [I, 0, 0]]),
    Matrix([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
    Matrix([[0, 0, 0], [0, 0, -I], [0, I, 0]]),
    Matrix([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / sqrt(3),
]
fundamental = g.trace()
antifundamental = g.inv().trace()
# the adjoint: g acts on the Lie algebra by X -> g X g^-1, in the basis of Gell-Mann matrices
adjoint = Matrix(8, 8, lambda a, b: (lam[a] * g * lam[b] * g.inv()).trace() / 2).trace()
assert simplify(fundamental * antifundamental - adjoint - 1) == 0
assert adjoint.subs({t_1: 1, t_2: 1}) == 8  # the dimension
eigen = [t_1, t_2, 1 / (t_1 * t_2)]
sextet = sum(eigen[i] * eigen[j] for i in range(3) for j in range(i, 3))
assert simplify(fundamental * antifundamental - sextet - fundamental) != 0  # not 6 + 3
expected = True
```
