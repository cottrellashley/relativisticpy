# Linear systems and eigenvalues

Norms of matrices and the condition number of a linear system, the spectral radius that
decides whether an iteration converges, and the power method for the dominant
eigenvalue: written as a limit, and as the loop an algorithm is.

## The infinity norm of a matrix

**Level 1** · `num.linear.infinity-norm` · Burden and Faires (2011), section 7.1

\(\|A\|_\infty\), the norm a book writes with a subscript on the double bars: the largest
sum of the absolute values along a row, \(\max(4, 4, 7) = 7\).

```latex
\left\| \begin{pmatrix} 1 & 2 & -1 \\ 0 & 3 & -1 \\ 5 & -1 & 1 \end{pmatrix} \right\|_\infty
```

```python
A = Matrix([[1, 2, -1], [0, 3, -1], [5, -1, 1]])
expected = max(sum(abs(entry) for entry in A.row(i)) for i in range(3))
assert expected == 7
assert np.linalg.norm(np.array(A.tolist(), float), np.inf) == 7
```

## The condition number of a matrix

**Level 2** · `num.linear.condition-number` · Burden and Faires (2011), section 7.5

\(K(A) = \|A\|_\infty \|A^{-1}\|_\infty\) for a matrix whose rows are nearly parallel: 60002,
so a small change in the data of \(A\mathbf{x} = \mathbf{b}\) can change the solution
sixty thousand times as much.

```latex
A := \begin{pmatrix} 1 & 2 \\ 1.0001 & 2 \end{pmatrix}
\|A\|_\infty \, \|A^{-1}\|_\infty
```

```python
A = Matrix([[1, 2], [Rational(10001, 10000), 2]])
norm = lambda M: max(sum(abs(entry) for entry in M.row(i)) for i in range(M.rows))
assert A.inv() == Matrix([[-10000, 10000], [Rational(10001, 2), -5000]])
expected = norm(A) * norm(A.inv())
assert expected == 60002  # the book's value
assert abs(np.linalg.cond(np.array([[1, 2], [1.0001, 2]]), np.inf) - 60002) < 1e-6
```

## The spectral radius of the Jacobi iteration

**Level 3** · `num.linear.jacobi-spectral-radius` · Burden and Faires (2011), section 7.3

Splitting \(A = D - L - U\), the Jacobi iteration multiplies the error by
\(T_j = D^{-1}(L + U)\), and converges because its spectral radius,
\(\rho(T_j) = \max|\lambda|\), the book's notation, is \(\frac{\sqrt{2}}{4} < 1\).

```latex
A := \begin{pmatrix} 4 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 4 \end{pmatrix}
D := \operatorname{diag}(4, 4, 4)
T_j := D^{-1}(D - A)
\rho(T_j)
```

```python
A = Matrix([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])
D = diag(4, 4, 4)
T = D.inv() * (D - A)
assert T.eigenvals() == {0: 1, sqrt(2) / 4: 1, -sqrt(2) / 4: 1}
expected = sqrt(2) / 4
assert abs(max(abs(np.linalg.eigvals(np.array(T.tolist(), float)))) - float(expected)) < 1e-12
```

## The power method, as a limit

**Level 4** · `num.eigen.power-limit` · Burden and Faires (2011), section 9.3

The iterates \(\mathbf{x}^{(k)} = A^k \mathbf{x}^{(0)}\), labelled by superscripts in
parentheses, turn towards the dominant eigenvector, and their Rayleigh quotients tend to
the dominant eigenvalue, 3. The language must raise the matrix to a symbolic power and
take the limit.

```latex
A := \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}
\mathbf{x}^{(0)} := \begin{pmatrix} 1 \\ 0 \end{pmatrix}
\mathbf{x}^{(k)} := A^k \mathbf{x}^{(0)}
\lim_{k \to \infty} \frac{(\mathbf{x}^{(k)})^T A \mathbf{x}^{(k)}}{(\mathbf{x}^{(k)})^T \mathbf{x}^{(k)}}
```

```python
k = symbols("k", positive=True, integer=True)
A = Matrix([[2, 1], [1, 2]])
x = Matrix([(3**k + 1) / 2, (3**k - 1) / 2])  # A^k x^(0), by diagonalising A
assert all(simplify(x.subs(k, n) - A**n * Matrix([1, 0])) == zeros(2, 1) for n in range(5))
expected = limit((x.T * A * x)[0] / (x.T * x)[0], k, oo)
assert expected == 3 == max(A.eigenvals())
```

## Power iteration, as an algorithm

**Level 5** · `num.eigen.power-iteration` · Trefethen and Bau (1997), lecture 27

The same method as Trefethen and Bau write it, a loop: apply \(A\), normalise, and take
the Rayleigh quotient. After five steps the estimate is \(\lambda^{(5)} = 88574/29525 =
2.99996613\), not yet 3. No design of the language reads an algorithm: a loop over
\(k\), and names given new values as it runs.

```latex
A := \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}
v^{(0)} := \begin{pmatrix} 1 \\ 0 \end{pmatrix}
\textbf{for } k = 1, \ldots, 5
    w := A v^{(k-1)}
    v^{(k)} := w / \|w\|
    \lambda^{(k)} := (v^{(k)})^T A v^{(k)}
\lambda^{(5)}
```

```python
A = Matrix([[2, 1], [1, 2]])
v = Matrix([1, 0])
for _ in range(5):
    w = A * v
    v = w / sqrt((w.T * w)[0])
    estimate = simplify((v.T * A * v)[0])
expected = estimate
assert expected == Rational(88574, 29525)
vector = np.array([1.0, 0.0])
for _ in range(5):
    vector = np.array([[2, 1], [1, 2]]) @ vector
    vector /= np.linalg.norm(vector)
assert abs(vector @ np.array([[2, 1], [1, 2]]) @ vector - float(expected)) < 1e-12
```
