# Eigenvalues and eigenvectors

The characteristic equation, diagonalisation, the spectral theorem, the exponential of a
matrix, similarity and the Jordan form, and the determinant of an operator on functions.

## Eigenvalues from the characteristic equation

**Level 2** · `la.eigen.characteristic` · Strang (2016), section 6.1

The eigenvalues are the roots of \(\det(A - \lambda I) = 0\), with \(I\) the identity of
\(A\)'s size. This \(A\) is singular, so one of them is zero.

```latex
A := \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}
solve(\det(A - \lambda I) = 0, \lambda)
```

```python
lam = symbols("lambda")
expected = solve((Matrix([[1, 2], [2, 4]]) - lam * eye(2)).det(), lam)
assert expected == [0, 5]
```

## Powers by diagonalisation

**Level 3** · `la.eigen.powers` · Strang (2016), section 6.2

With the eigenvectors as the columns of \(X\), \(\Lambda = X^{-1} A X\) is diagonal, and
\(A^k = X \Lambda^k X^{-1}\) for every \(k\): a diagonal matrix raised to a symbolic power.

```latex
A := \begin{pmatrix} 1 & 5 \\ 0 & 6 \end{pmatrix}
X := \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}
\Lambda := X^{-1} A X
X \Lambda^k X^{-1}
```

```python
k = symbols("k")
A = Matrix([[1, 5], [0, 6]])
X = Matrix([[1, 1], [0, 1]])
assert X.inv() * A * X == diag(1, 6)
expected = X * diag(1, 6**k) * X.inv()
assert expected == Matrix([[1, 6**k - 1], [0, 6**k]])
assert expected.subs(k, 3) == A**3
```

## The spectral theorem

**Level 2** · `la.eigen.spectral` · Strang (2016), section 6.4

A symmetric matrix is \(Q \Lambda Q^T\), with its orthonormal eigenvectors as the columns
of \(Q\) and its eigenvalues on the diagonal of \(\Lambda\).

```latex
S := \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}
Q := \frac{1}{\sqrt{5}} \begin{pmatrix} 2 & 1 \\ -1 & 2 \end{pmatrix}
\Lambda := \begin{pmatrix} 0 & 0 \\ 0 & 5 \end{pmatrix}
Q \Lambda Q^T = S
```

```python
S = Matrix([[1, 2], [2, 4]])
Q = Matrix([[2, 1], [-1, 2]]) / sqrt(5)
assert simplify(Q.T * Q) == eye(2)
assert all(simplify(S * Q[:, i] - value * Q[:, i]) == zeros(2, 1) for i, value in enumerate([0, 5]))
assert simplify(Q * diag(0, 5) * Q.T) == S
expected = True
```

## The exponential of a matrix

**Level 2** · `la.eigen.exponential` · Strang (2016), section 6.3

\(e^{At}\) for the generator of rotations, the sum of its power series: a rotation through
the angle \(t\).

```latex
A := \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}
e^{At}
```

```python
t = symbols("t")
expected = simplify((Matrix([[0, 1], [-1, 0]]) * t).exp())
assert expected == Matrix([[cos(t), sin(t)], [-sin(t), cos(t)]])
```

## Similarity and the Jordan form

**Level 4** · `la.eigen.jordan` · Horn and Johnson (2013), section 3.1

Two matrices are similar, \(A \sim B\), when they have the same Jordan form. This \(A\)
has the double eigenvalue 2 but only one independent eigenvector, so it is similar to the
Jordan block \(J_2(2)\) and not to \(2I\), though all three have the same eigenvalues.

```latex
A := \begin{pmatrix} 1 & 1 \\ -1 & 3 \end{pmatrix}
A \sim \begin{pmatrix} 2 & 1 \\ 0 & 2 \end{pmatrix}
A \sim \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix}
```

```python
A = Matrix([[1, 1], [-1, 3]])
P, J = A.jordan_form()
assert J == Matrix([[2, 1], [0, 2]]) and P * J * P.inv() == A
assert (2 * eye(2)).jordan_form()[1] == 2 * eye(2) != J
outputs = [True, False]
```

## The determinant of a differential operator

**Level 5** · `la.eigen.zeta-determinant` · Dunne (2008), J. Phys. A 41, 304006

The determinant of \(-d^2/dx^2\) on \([0, L]\) with Dirichlet conditions, as quantum field
theory defines it. The product of its eigenvalues \(n^2\pi^2/L^2\) diverges; its
zeta-function regularisation, \(e^{-\zeta'(0)}\) with \(\zeta(s) = \sum_n
\lambda_n^{-s}\), is \(2L\).

```latex
-y'' = \lambda y, \quad y(0) = 0, \quad y(L) = 0
\det\left(-\frac{d^2}{dx^2}\right)
```

```python
L = symbols("L")
# zeta(s) = (L/pi)^(2s) zeta_R(2s), so zeta'(0) = 2 log(L/pi) zeta_R(0) + 2 zeta_R'(0)
h = Float("1e-12", 40)
riemann_prime_0 = (N(zeta(h), 40) - N(zeta(-h), 40)) / (2 * h)
assert zeta(0) == Rational(-1, 2) and abs(riemann_prime_0 + log(2 * pi).evalf(30) / 2) < 1e-15
for length in (1, 2, Rational(7, 3)):
    determinant = exp(-(2 * log(length / pi) * zeta(0) + 2 * riemann_prime_0))
    assert abs(N(determinant, 30) - 2 * length) < 1e-12
expected = 2 * L
```
