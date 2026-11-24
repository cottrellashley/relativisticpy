# Matrix products

The product of two matrices, \(AB\), whose entry in row i and column j is row i of A times
column j of B, written side by side, with a dot, and as each language of computing writes
it. Strang (2016) is *Introduction to Linear Algebra*, 5th edition; Riley, Hobson and Bence
(2006) is *Mathematical Methods for Physics and Engineering*, 3rd edition; Griffiths (2005)
is *Introduction to Quantum Mechanics*, 2nd edition; Gibbs and Wilson (1901) is *Vector
Analysis*; PEP 465 (2014) is Python's proposal of a dedicated infix operator for matrix
multiplication; the Unicode Standard is version 15.0.

`«A»` and `«B»` are the rows of the two matrices: in LaTeX, the body of a matrix
environment; in plain text and Unicode, a list of rows. A row times a column is a 1 × 1
matrix, which books take for the number it holds, as Strang (2016) takes a row times a
column for their dot product.

## Names

```python
a, b, c, d, x, y = symbols("a b c d x y")
```

## Operands

| operand | written in | `A` | `B` | separates |
|---|---|---|---|---|
| `numbers` | latex | `1 & 2 \\ 3 & 4` | `0 & 1 \\ 1 & 0` | the order of the factors, BA, and the product entry by entry |
| | text, unicode | `[[1, 2], [3, 4]]` | `[[0, 1], [1, 0]]` | |
| | python | `[[1, 2], [3, 4]]` | `[[0, 1], [1, 0]]` | |
| `negative` | latex | `2 & -1 \\ 0 & 3` | `1 & 4 \\ -2 & 5` | entries with a sign |
| | text | `[[2, -1], [0, 3]]` | `[[1, 4], [-2, 5]]` | |
| | unicode | `[[2, −1], [0, 3]]` | `[[1, 4], [−2, 5]]` | |
| | python | `[[2, -1], [0, 3]]` | `[[1, 4], [-2, 5]]` | |
| `symbols` | latex | `a & b \\ c & d` | `x & 0 \\ 0 & y` | a diagonal factor on the right, which scales the columns, from one on the left, which scales the rows |
| | text, unicode | `[[a, b], [c, d]]` | `[[x, 0], [0, y]]` | |
| | python | `[[a, b], [c, d]]` | `[[x, 0], [0, y]]` | |
| `rectangular` | latex | `1 & 2 & 3 \\ 4 & 5 & 6` | `1 & 0 \\ 0 & 1 \\ 1 & 1` | a 2 × 3 times a 3 × 2, which is 2 × 2, where the other order is 3 × 3 |
| | text, unicode | `[[1, 2, 3], [4, 5, 6]]` | `[[1, 0], [0, 1], [1, 1]]` | |
| | python | `[[1, 2, 3], [4, 5, 6]]` | `[[1, 0], [0, 1], [1, 1]]` | |
| `column` | latex | `1 & 2 \\ 3 & 4` | `5 \\ 6` | a matrix times a column, which is a column |
| | text, unicode | `[[1, 2], [3, 4]]` | `[[5], [6]]` | |
| | python | `[[1, 2], [3, 4]]` | `[[5], [6]]` | |
| `row-by-column` | latex | `1 & 2 & 3` | `4 \\ 5 \\ 6` | a row times a column, which is a number |
| | text, unicode | `[[1, 2, 3]]` | `[[4], [5], [6]]` | |
| | python | `[[1, 2, 3]]` | `[[4], [5], [6]]` | |
| `column-by-row` | latex | `1 \\ 2` | `3 & 4` | a column times a row, which is a matrix, from their dot product |
| | text, unicode | `[[1], [2]]` | `[[3, 4]]` | |
| | python | `[[1], [2]]` | `[[3, 4]]` | |
| `nonconforming` | latex | `1 & 2 & 3 \\ 4 & 5 & 6` | `1 & 2 \\ 3 & 4` | a 2 × 3 times a 2 × 2, which has no product, from one taken anyway |
| | text, unicode | `[[1, 2, 3], [4, 5, 6]]` | `[[1, 2], [3, 4]]` | |
| | python | `[[1, 2, 3], [4, 5, 6]]` | `[[1, 2], [3, 4]]` | |

## Oracle

```python
left, right = Matrix(A), Matrix(B)
if left.cols != right.rows:
    error = "RelativisticPyError"
else:
    product = left * right
    expected = product[0, 0] if product.shape == (1, 1) else product
```

## Second route

Each entry as the sum over k of the entry of A in row i and column k times that of B in row
k and column j.

```python
if left.cols != right.rows:
    assert error == "RelativisticPyError"
else:
    by_entries = Matrix(
        left.rows,
        right.cols,
        lambda i, j: sum(left[i, k] * right[k, j] for k in range(left.cols)),
    )
    assert by_entries == (Matrix([expected]) if by_entries.shape == (1, 1) else expected)
```

## Forms

### Side by side

**Form** · `juxtaposition` · Strang (2016), §2.4, Rules for Matrix Operations: `AB`, the matrices in square brackets

```latex
A := \begin{bmatrix} «A» \end{bmatrix}
B := \begin{bmatrix} «B» \end{bmatrix}
AB
```

### Bold capitals side by side

**Form** · `bold` · Riley, Hobson and Bence (2006), §8.4, Basic matrix algebra: the product of matrices set in bold, `\mathbf{A}\mathbf{B}`, written out in round brackets

```latex
\mathbf{A} := \begin{pmatrix} «A» \end{pmatrix}
\mathbf{B} := \begin{pmatrix} «B» \end{pmatrix}
\mathbf{A}\mathbf{B}
```

### A dot between them

**Form** · `cdot` · Gibbs and Wilson (1901), chapter V, Linear Vector Functions: the direct product of two dyadics, `\Phi \cdot \Psi`

```latex
A := \begin{pmatrix} «A» \end{pmatrix}
B := \begin{pmatrix} «B» \end{pmatrix}
A \cdot B
```

**Not of** · `column-by-row` · between a column and a row, which are vectors, a dot is their dot product

### Square brackets side by side

**Form** · `brackets-side-by-side` · Strang (2016), §2.4, Rules for Matrix Operations: the matrices written out in square brackets, one after the other

```latex
\begin{bmatrix} «A» \end{bmatrix} \begin{bmatrix} «B» \end{bmatrix}
```

### Round brackets side by side

**Form** · `round-side-by-side` · Griffiths (2005), Appendix A.3, Matrices: matrices written out in round brackets, one after the other

```latex
\begin{pmatrix} «A» \end{pmatrix} \begin{pmatrix} «B» \end{pmatrix}
```

### A star in plain text

**Form** · `star` · SymPy's `Matrix` and MATLAB's `mtimes`: `A*B` is the matrix product

```text
A := «A»
B := «B»
A*B
```

### Side by side in plain text

**Form** · `text-juxtaposition` · docs/language.md, Numbers and arithmetic: juxtaposition is multiplication; docs/matrices.md, Arithmetic: matrices multiply as matrices

```text
A := «A»
B := «B»
A B
```

### Lists of rows side by side

**Form** · `lists` · docs/matrices.md, Writing a matrix: lists of rows are matrices, and multiply as matrices

```text
«A» «B»
```

### Python's operator

**Form** · `matmul` · PEP 465 (2014): `A @ B`, the matrix product, as NumPy reads it

```text
A := «A»
B := «B»
A @ B
```

### The Wolfram Language's dot

**Form** · `wolfram-dot` · the Wolfram Language's `Dot`: `a.b` of two matrices is their product

```text
A := «A»
B := «B»
A.B
```

### The dot operator in Unicode

**Form** · `unicode-dot-operator` · the Unicode Standard, U+22C5 DOT OPERATOR, the dot of `A \cdot B`, as typed

```unicode
A := «A»
B := «B»
A⋅B
```

**Not of** · `column-by-row` · between a column and a row, which are vectors, a dot is their dot product

### The middle dot in Unicode

**Form** · `unicode-middle-dot` · the Unicode Standard, U+00B7 MIDDLE DOT, as typed for a product

```unicode
A := «A»
B := «B»
A·B
```

**Not of** · `column-by-row` · between a column and a row, which are vectors, a dot is their dot product

### Bold capitals in Unicode

**Form** · `unicode-bold` · the Unicode Standard, Mathematical Alphanumeric Symbols, U+1D400 and U+1D401, MATHEMATICAL BOLD CAPITAL A and B

```unicode
𝐀 := «A»
𝐁 := «B»
𝐀𝐁
```
