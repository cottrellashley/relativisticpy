# Inverses

The inverse of a square matrix, \(A^{-1}\), with \(A A^{-1} = A^{-1} A = I\), written as the
power −1 of a name or of a matrix written out, as a call, and as Unicode's superscripts
write it. A matrix that is singular, or not square, has none, and must be refused. Strang
(2016) is *Introduction to Linear Algebra*, 5th edition; Riley, Hobson and Bence (2006) is
*Mathematical Methods for Physics and Engineering*, 3rd edition; Lamport (1994) is *LaTeX: A
Document Preparation System*, 2nd edition; the amsmath package is described in its *User's
Guide* (American Mathematical Society, 2020); AsciiMath is the plain text notation of
asciimath.org; the Unicode Standard is version 15.0.

`«A»` is the rows of the matrix: in LaTeX, the body of a matrix environment; in plain text
and Unicode, a list of rows.

## Names

```python
a, b, c, d, t = symbols("a b c d t")
```

## Operands

| operand | written in | `A` | separates |
|---|---|---|---|
| `numbers` | latex | `1 & 2 \\ 3 & 4` | the inverse from the reciprocal of each entry |
| | text, unicode | `[[1, 2], [3, 4]]` | |
| | python | `[[1, 2], [3, 4]]` | |
| `negative` | latex | `2 & -1 \\ 1 & 1` | entries with a sign |
| | text | `[[2, -1], [1, 1]]` | |
| | unicode | `[[2, −1], [1, 1]]` | |
| | python | `[[2, -1], [1, 1]]` | |
| `symbols` | latex | `a & b \\ c & d` | entries that are letters, divided by the determinant ad − bc |
| | text, unicode | `[[a, b], [c, d]]` | |
| | python | `[[a, b], [c, d]]` | |
| `diagonal` | latex | `2 & 0 \\ 0 & 5` | zeros that stay zeros |
| | text, unicode | `[[2, 0], [0, 5]]` | |
| | python | `[[2, 0], [0, 5]]` | |
| `three-by-three` | latex | `2 & 0 & 1 \\ 1 & 1 & 0 \\ 0 & 1 & 1` | a matrix of three rows |
| | text, unicode | `[[2, 0, 1], [1, 1, 0], [0, 1, 1]]` | |
| | python | `[[2, 0, 1], [1, 1, 0], [0, 1, 1]]` | |
| `rotation` | latex | `\cos t & -\sin t \\ \sin t & \cos t` | a rotation, whose inverse is its transpose once cos²t + sin²t is 1 |
| | text | `[[cos(t), -sin(t)], [sin(t), cos(t)]]` | |
| | unicode | `[[cos t, −sin t], [sin t, cos t]]` | |
| | python | `[[cos(t), -sin(t)], [sin(t), cos(t)]]` | |
| `singular` | latex | `1 & 2 \\ 2 & 4` | a singular matrix, which has no inverse, from one given anyway |
| | text, unicode | `[[1, 2], [2, 4]]` | |
| | python | `[[1, 2], [2, 4]]` | |
| `rectangular` | latex | `1 & 2 & 3 \\ 4 & 5 & 6` | a matrix that is not square, which has no inverse |
| | text, unicode | `[[1, 2, 3], [4, 5, 6]]` | |
| | python | `[[1, 2, 3], [4, 5, 6]]` | |

## Oracle

```python
M = Matrix(A)
if not M.is_square or simplify(M.det()) == 0:
    error = "RelativisticPyError"
else:
    expected = M.inv()
```

## Second route

The inverse times the matrix, on either side, is the identity; a matrix without one is not
square or has a rank below its size.

```python
if not M.is_square or simplify(M.det()) == 0:
    assert error == "RelativisticPyError"
    assert not M.is_square or M.rank() < M.rows
else:
    assert simplify(M * expected) == eye(M.rows)
    assert simplify(expected * M) == eye(M.rows)
```

## Forms

### The power −1

**Form** · `power` · Strang (2016), §2.5, Inverse Matrices: `A^{-1}`

```latex
A := \begin{bmatrix} «A» \end{bmatrix}
A^{-1}
```

### The power −1 of a bold capital

**Form** · `bold-power` · Riley, Hobson and Bence (2006), §8.10, The inverse of a matrix: `\mathbf{A}^{-1}`, of a matrix written out in round brackets

```latex
\mathbf{A} := \begin{pmatrix} «A» \end{pmatrix}
\mathbf{A}^{-1}
```

### The power −1 of a matrix written out

**Form** · `written-out-power` · Strang (2016), §2.5, Inverse Matrices: the inverse of a matrix written out in square brackets, `\begin{bmatrix} a & b \\ c & d \end{bmatrix}^{-1}`

```latex
\begin{bmatrix} «A» \end{bmatrix}^{-1}
```

### The power −1 of a matrix in round brackets

**Form** · `round-power` · docs/matrices.md, Arithmetic: `\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}^{-1}`

```latex
\begin{pmatrix} «A» \end{pmatrix}^{-1}
```

### The power −1 of brackets that grow

**Form** · `left-right-power` · the amsmath package, Matrices: `matrix`, without delimiters, set in Lamport's (1994) `\left(` and `\right)`, §3.3.4, Delimiters

```latex
\left( \begin{matrix} «A» \end{matrix} \right)^{-1}
```

### A call of inverse

**Form** · `inverse-call` · docs/reference.md, Functions, Matrices: `inverse`

```text
A := «A»
inverse(A)
```

### A call of inv

**Form** · `inv-call` · MATLAB's `inv(A)`; docs/reference.md, Functions, Matrices: `inv`

```text
A := «A»
inv(A)
```

### A caret in plain text

**Form** · `caret` · AsciiMath: `A^(-1)`, the brackets round the exponent dropped as it is set

```text
A := «A»
A^(-1)
```

### Two stars

**Form** · `double-star` · SymPy's `A**-1`, the power −1 of a matrix, which is its inverse

```text
A := «A»
A**-1
```

### Superscripts in Unicode

**Form** · `unicode-superscript` · the Unicode Standard, U+207B SUPERSCRIPT MINUS and U+00B9 SUPERSCRIPT ONE: `A⁻¹`

```unicode
A := «A»
A⁻¹
```

### A caret in Unicode

**Form** · `unicode-caret` · as typed, with Unicode's minus sign, U+2212

```unicode
A := «A»
A^(−1)
```
