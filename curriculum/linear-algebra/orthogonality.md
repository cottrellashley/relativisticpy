# Orthogonality

Least squares, the Gram–Schmidt process in \(\mathbb{R}^n\) and in a space of functions
with an inner product, and the QR and singular value decompositions.

## Least squares

**Level 2** · `la.orthogonality.least-squares` · Strang (2016), section 4.3

The line \(C + Dt\) closest to the points \((0, 6)\), \((1, 0)\) and \((2, 0)\), from the
normal equations \(A^T A \hat{x} = A^T b\): the line \(5 - 3t\).

```latex
A := \begin{pmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{pmatrix}
b := \begin{pmatrix} 6 \\ 0 \\ 0 \end{pmatrix}
(A^T A)^{-1} A^T b
```

```python
A = Matrix([[1, 0], [1, 1], [1, 2]])
b = Matrix([6, 0, 0])
expected = (A.T * A).LUsolve(A.T * b)
assert expected == Matrix([5, -3])
```

## Gram–Schmidt and A = QR

**Level 2** · `la.orthogonality.qr` · Strang (2016), section 4.4

With the orthonormal vectors Gram–Schmidt makes from the columns of \(A\) as the columns of
\(Q\), \(R = Q^T A\) is upper triangular, and \(A = QR\).

```latex
A := \begin{pmatrix} 1 & 2 & 3 \\ -1 & 0 & -3 \\ 0 & -2 & 3 \end{pmatrix}
Q := \begin{pmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}} & \frac{1}{\sqrt{3}} \\ -\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}} & \frac{1}{\sqrt{3}} \\ 0 & -\frac{2}{\sqrt{6}} & \frac{1}{\sqrt{3}} \end{pmatrix}
Q^T A
```

```python
A = Matrix([[1, 2, 3], [-1, 0, -3], [0, -2, 3]])
Q, R = A.QRdecomposition()
columns = [Matrix([1, -1, 0]) / sqrt(2), Matrix([1, 1, -2]) / sqrt(6), Matrix([1, 1, 1]) / sqrt(3)]
assert simplify(Q - Matrix.hstack(*columns)) == zeros(3, 3)
expected = R
assert R == Matrix([[sqrt(2), sqrt(2), sqrt(18)], [0, sqrt(6), -sqrt(6)], [0, 0, sqrt(3)]])
```

## The singular value decomposition

**Level 2** · `la.orthogonality.svd` · Strang (2016), section 7.2

\(A = U \Sigma V^T\), with orthogonal \(U\) and \(V\), and the singular values
\(\sqrt{45}\) and \(\sqrt{5}\), the square roots of the eigenvalues of \(A^T A\), on the
diagonal of \(\Sigma\).

```latex
A := \begin{pmatrix} 3 & 0 \\ 4 & 5 \end{pmatrix}
U := \frac{1}{\sqrt{10}} \begin{pmatrix} 1 & -3 \\ 3 & 1 \end{pmatrix}
\Sigma := \begin{pmatrix} \sqrt{45} & 0 \\ 0 & \sqrt{5} \end{pmatrix}
V := \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}
U \Sigma V^T = A
```

```python
A = Matrix([[3, 0], [4, 5]])
assert (A.T * A).eigenvals() == {45: 1, 5: 1}
U = Matrix([[1, -3], [3, 1]]) / sqrt(10)
V = Matrix([[1, -1], [1, 1]]) / sqrt(2)
assert simplify(U.T * U) == eye(2) and simplify(V.T * V) == eye(2)
assert simplify(U * diag(sqrt(45), sqrt(5)) * V.T) == A
expected = True
```

## Gram–Schmidt on polynomials

**Level 3** · `la.orthogonality.gram-schmidt` · Axler (2015), example 6.33

The Gram–Schmidt process in an inner product space of functions: from \(1, x, x^2\), with
\(\langle p, q \rangle = \int_{-1}^{1} p(x) q(x)\,dx\), an orthonormal basis of the
polynomials of degree at most two, whose third member is \(\sqrt{45/8}\,(x^2 - 1/3)\).

```latex
\langle p, q \rangle := \int_{-1}^{1} p(x)\, q(x)\, dx
\|p\| := \sqrt{\langle p, p \rangle}
e_1 := \frac{1}{\|1\|}
e_2 := \frac{x - \langle x, e_1 \rangle e_1}{\|x - \langle x, e_1 \rangle e_1\|}
e_3 := \frac{x^2 - \langle x^2, e_1 \rangle e_1 - \langle x^2, e_2 \rangle e_2}{\|x^2 - \langle x^2, e_1 \rangle e_1 - \langle x^2, e_2 \rangle e_2\|}
e_3
```

```python
x = symbols("x")
inner = lambda p, q: integrate(p * q, (x, -1, 1))
norm = lambda p: sqrt(inner(p, p))
e1 = 1 / norm(Integer(1))
v2 = x - inner(x, e1) * e1
e2 = v2 / norm(v2)
v3 = x**2 - inner(x**2, e1) * e1 - inner(x**2, e2) * e2
expected = simplify(v3 / norm(v3))
assert simplify(expected - sqrt(Rational(45, 8)) * (x**2 - Rational(1, 3))) == 0
```
