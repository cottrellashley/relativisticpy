# Transposes

The transpose of a matrix, \(A^T\), its rows written as columns, with a letter T raised
three ways, a small t, a prime, a tilde, and as each language of computing writes it. Strang
(2016) is *Introduction to Linear Algebra*, 5th edition; Lamport (1994) is *LaTeX: A
Document Preparation System*, 2nd edition; ISO 80000-2:2019 is the standard for
mathematical signs and symbols; Goodfellow, Bengio and Courville (2016) is *Deep Learning*;
Artin (2011) is *Algebra*, 2nd edition; Axler (2015) is *Linear Algebra Done Right*, 3rd
edition; Lang (1987) is *Linear Algebra*, 3rd edition; Greene (2012) is *Econometric
Analysis*, 7th edition; Griffiths (2005) is *Introduction to Quantum Mechanics*, 2nd
edition; AsciiMath is the plain text notation of asciimath.org; the Unicode Standard is
version 15.0.

`«A»` is the rows of the matrix: in LaTeX, the body of a matrix environment; in plain text
and Unicode, a list of rows. A transpose does not conjugate: the conjugate transpose is
another operation, \(A^\dagger\).

## Names

```python
a, b, c, d = symbols("a b c d")
```

## Operands

| operand | written in | `A` | separates |
|---|---|---|---|
| `square` | latex | `1 & 2 \\ 3 & 4` | the rows written as columns, from the matrix as it is |
| | text, unicode | `[[1, 2], [3, 4]]` | |
| | python | `[[1, 2], [3, 4]]` | |
| `negative` | latex | `0 & -1 \\ 5 & 2` | entries with a sign |
| | text | `[[0, -1], [5, 2]]` | |
| | unicode | `[[0, −1], [5, 2]]` | |
| | python | `[[0, -1], [5, 2]]` | |
| `rectangular` | latex | `1 & 2 & 3 \\ 4 & 5 & 6` | a 2 × 3 matrix, whose transpose is 3 × 2 |
| | text, unicode | `[[1, 2, 3], [4, 5, 6]]` | |
| | python | `[[1, 2, 3], [4, 5, 6]]` | |
| `symbols` | latex | `a & b \\ c & d` | entries that are letters |
| | text, unicode | `[[a, b], [c, d]]` | |
| | python | `[[a, b], [c, d]]` | |
| `complex` | latex | `1 & 2i \\ 3 & 4` | a transpose, which keeps 2i, from the conjugate transpose, which makes it −2i |
| | text | `[[1, 2*i], [3, 4]]` | |
| | unicode | `[[1, 2i], [3, 4]]` | |
| | python | `[[1, 2*I], [3, 4]]` | |
| `three-by-three` | latex | `1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9` | a matrix of three rows |
| | text, unicode | `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]` | |
| | python | `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]` | |

## Oracle

```python
expected = Matrix(A).T
```

## Second route

The entry in row i and column j of the transpose is the entry in row j and column i of the
matrix, and the transpose of the transpose is the matrix.

```python
M = Matrix(A)
assert expected == Matrix(M.cols, M.rows, lambda i, j: M[j, i])
assert expected.T == M
```

## Forms

### A superscript T

**Form** · `T` · Strang (2016), §2.7, Transposes and Permutations: `A^T`

```latex
A := \begin{bmatrix} «A» \end{bmatrix}
A^T
```

### A superscript T in braces

**Form** · `T-braces` · Strang's `A^T`, with the superscript in braces, as Lamport (1994), §3.3.1, writes one

```latex
A := \begin{bmatrix} «A» \end{bmatrix}
A^{T}
```

### An upright T

**Form** · `T-upright` · ISO 80000-2:2019, matrices: `A^T`, the T upright, as it is no variable

```latex
A := \begin{pmatrix} «A» \end{pmatrix}
A^{\mathrm{T}}
```

### A raised tack

**Form** · `top` · Goodfellow, Bengio and Courville (2016), §2.1: the transpose `A^\top`

```latex
A := \begin{bmatrix} «A» \end{bmatrix}
A^\top
```

### The symbol intercal

**Form** · `intercal` · docs/language.md, Matrices: `A^\intercal`

```latex
A := \begin{pmatrix} «A» \end{pmatrix}
A^\intercal
```

### A small t

**Form** · `t` · Artin (2011), chapter 1, Matrices, and Axler (2015), §3.F, Duality: the transpose `A^t`

```latex
A := \begin{bmatrix} «A» \end{bmatrix}
A^t
```

### A t on the left

**Form** · `left-t` · Lang (1987), chapter II, Matrices: the transpose `{}^tA`, its t raised before the letter

```latex
A := \begin{pmatrix} «A» \end{pmatrix}
{}^{t}A
```

### A prime

**Form** · `prime` · Greene (2012), Appendix A, Matrix Algebra: the transpose `\mathbf{A}'`, of a matrix set in bold

```latex
\mathbf{A} := \begin{bmatrix} «A» \end{bmatrix}
\mathbf{A}'
```

**Not of** · `complex` · the book's matrices are real

### A tilde

**Form** · `tilde` · Griffiths (2005), Appendix A.3, Matrices: the transpose `\tilde{T}`, written with a tilde

```latex
T := \begin{pmatrix} «A» \end{pmatrix}
\tilde{T}
```

### A call of transpose

**Form** · `transpose-call` · SymPy's and MATLAB's `transpose(A)`

```text
A := «A»
transpose(A)
```

### An attribute T

**Form** · `attribute-T` · NumPy's and SymPy's `A.T`

```text
A := «A»
A.T
```

### A caret T in plain text

**Form** · `caret-T` · AsciiMath: `A^T`

```text
A := «A»
A^T
```

### MATLAB's dot and prime

**Form** · `matlab-dot-prime` · MATLAB's `A.'`, the transpose, which does not conjugate

```text
A := «A»
A.'
```

### MATLAB's prime

**Form** · `matlab-prime` · MATLAB's `A'`, the transpose of a real matrix

```text
A := «A»
A'
```

**Not of** · `complex` · of a complex matrix, MATLAB's prime is the conjugate transpose

### A superscript T in Unicode

**Form** · `unicode-T` · the Unicode Standard, U+1D40 MODIFIER LETTER CAPITAL T; docs/reference.md, Unicode: `Aᵀ` is `A^{T}`

```unicode
A := «A»
Aᵀ
```

### A raised tack in Unicode

**Form** · `unicode-top` · the Unicode Standard, U+22A4 DOWN TACK, the ⊤ of `A^\top`, as typed

```unicode
A := «A»
A^⊤
```

### A prime in Unicode

**Form** · `unicode-prime` · the Unicode Standard, U+2032 PRIME, Greene's prime as typed

```unicode
A := «A»
A′
```

**Not of** · `complex` · the book's matrices are real
