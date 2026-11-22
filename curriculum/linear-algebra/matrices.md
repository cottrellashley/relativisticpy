# Matrices and linear systems

Products, inverses, determinants, rank and trace, elimination as a factorisation, and the
solution of a square system, as a first course in linear algebra writes them.

## A matrix product

**Level 1** · `la.matrices.product` · Strang (2016), section 2.4

Two matrices written side by side multiply, rows into columns: a 2 × 3 matrix by a 3 × 2.

```latex
\begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix} \begin{pmatrix} 7 & 8 \\ 9 & 10 \\ 11 & 12 \end{pmatrix}
```

```python
expected = Matrix([[1, 2, 3], [4, 5, 6]]) * Matrix([[7, 8], [9, 10], [11, 12]])
assert expected == Matrix([[58, 64], [139, 154]])
```

## The inverse of a 2 × 2 matrix

**Level 1** · `la.matrices.inverse` · Strang (2016), section 2.5

The power \(-1\) of a matrix with symbolic entries: one over the determinant, times the
matrix with its diagonal entries swapped and its other entries negated.

```latex
\begin{pmatrix} a & b \\ c & d \end{pmatrix}^{-1}
```

```python
a, b, c, d = symbols("a b c d")
expected = Matrix([[a, b], [c, d]]).inv()
assert simplify(expected - Matrix([[d, -b], [-c, a]]) / (a * d - b * c)) == zeros(2, 2)
```

## The trace

**Level 1** · `la.matrices.trace` · Strang (2016), section 6.1

The trace of a matrix, the sum of its diagonal entries, written as an operator on the
matrix's name.

```latex
A := \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix}
\operatorname{tr} A
```

```python
expected = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).trace()
assert expected == 15
```

## The rank

**Level 2** · `la.matrices.rank` · Strang (2016), section 3.2

The rank of a matrix whose third row is twice its second minus its first: the number of
pivots elimination leaves.

```latex
\operatorname{rank} \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix}
```

```python
expected = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).rank()
assert expected == 2
```

## Elimination as a factorisation

**Level 1** · `la.matrices.lu` · Strang (2016), section 2.6

\(A = LU\) for Strang's matrix with 2 on the diagonal and 1 beside it: \(L\) is lower
triangular and holds the multipliers \(\tfrac12\) and \(\tfrac23\), and \(U\) is the upper
triangular matrix elimination leaves, with the pivots \(2, \tfrac32, \tfrac43\). An
equation between matrices, which holds.

```latex
\begin{pmatrix} 2 & 1 & 0 \\ 1 & 2 & 1 \\ 0 & 1 & 2 \end{pmatrix} = \begin{pmatrix} 1 & 0 & 0 \\ \frac{1}{2} & 1 & 0 \\ 0 & \frac{2}{3} & 1 \end{pmatrix} \begin{pmatrix} 2 & 1 & 0 \\ 0 & \frac{3}{2} & 1 \\ 0 & 0 & \frac{4}{3} \end{pmatrix}
```

```python
L, U, _ = Matrix([[2, 1, 0], [1, 2, 1], [0, 1, 2]]).LUdecomposition()
assert L == Matrix([[1, 0, 0], [Rational(1, 2), 1, 0], [0, Rational(2, 3), 1]])
assert U == Matrix([[2, 1, 0], [0, Rational(3, 2), 1], [0, 0, Rational(4, 3)]])
expected = True
```

## A linear system

**Level 2** · `la.matrices.system` · Strang (2016), section 2.2

Three equations in three unknowns, \(A\mathbf{x} = \mathbf{b}\), solved as
\(\mathbf{x} = A^{-1}\mathbf{b}\); elimination and back substitution give
\((x, y, z) = (-1, 2, 2)\).

```latex
A := \begin{pmatrix} 2 & 4 & -2 \\ 4 & 9 & -3 \\ -2 & -3 & 7 \end{pmatrix}
\mathbf{b} := \begin{pmatrix} 2 \\ 8 \\ 10 \end{pmatrix}
A^{-1} \mathbf{b}
```

```python
A = Matrix([[2, 4, -2], [4, 9, -3], [-2, -3, 7]])
expected = A.LUsolve(Matrix([2, 8, 10]))
assert expected == Matrix([-1, 2, 2])
```
