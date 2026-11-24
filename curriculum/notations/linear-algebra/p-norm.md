# Norms of an order

The norms of an order, of vectors and of matrices, \(\|x\|_p\): of a vector, the sum of the
moduli, the length, and the greatest modulus; of a matrix, the greatest sum of a column,
the greatest sum of a row, the root of the sum of the squares, and the greatest singular
value. Golub and Van Loan (2013) is *Matrix Computations*, 4th edition; Lamport (1994) is
*LaTeX: A Document Preparation System*, 2nd edition; the amsmath package is described in
its *User's Guide* (American Mathematical Society, 2020); the Unicode Standard is version
15.0.

`«name»` is the vector's name or the matrix's, `«value»` what it is defined as, `«p»` the
order, and `«lowered»` the order as Unicode writes it beneath the bars.

## Operands

| operand | written in | `name` | `value` | `p` | `lowered` | separates |
|---|---|---|---|---|---|---|
| `vector-1` | latex | `x` | `(3, -4)` | `1` | | the sum of the moduli, 7, from the sum, −1 |
| | text | `x` | `[3, -4]` | `1` | | |
| | unicode | `x` | `(3, −4)` | | `₁` | |
| | python | | `[3, -4]` | `1` | | |
| `vector-2` | latex | `x` | `(3, -4)` | `2` | | the length, 5 |
| | text | `x` | `[3, -4]` | `2` | | |
| | unicode | `x` | `(3, −4)` | | `₂` | |
| | python | | `[3, -4]` | `2` | | |
| `vector-infinity` | latex | `x` | `(3, -4)` | `\infty` | | the greatest modulus, 4, from the greatest component, 3 |
| | text | `x` | `[3, -4]` | `Inf` | | |
| | unicode | `x` | `(3, −4)` | | `_∞` | |
| | python | | `[3, -4]` | `oo` | | |
| `matrix-1` | latex | `A` | `\begin{pmatrix} 1 & -2 \\ 3 & 4 \end{pmatrix}` | `1` | | the greatest sum of the moduli of a column, 6 |
| | text | `A` | `[[1, -2], [3, 4]]` | `1` | | |
| | unicode | `A` | `[[1, −2], [3, 4]]` | | `₁` | |
| | python | | `[[1, -2], [3, 4]]` | `1` | | |
| `matrix-infinity` | latex | `A` | `\begin{pmatrix} 1 & -2 \\ 3 & 4 \end{pmatrix}` | `\infty` | | the greatest sum of the moduli of a row, 7 |
| | text | `A` | `[[1, -2], [3, 4]]` | `Inf` | | |
| | unicode | `A` | `[[1, −2], [3, 4]]` | | `_∞` | |
| | python | | `[[1, -2], [3, 4]]` | `oo` | | |
| `matrix-frobenius` | latex | `A` | `\begin{pmatrix} 1 & -2 \\ 3 & 4 \end{pmatrix}` | `F` | | the root of the sum of the squares, √30 |
| | text | `A` | `[[1, -2], [3, 4]]` | `'fro'` | | |
| | unicode | `A` | `[[1, −2], [3, 4]]` | | `_F` | |
| | python | | `[[1, -2], [3, 4]]` | `"fro"` | | |
| `matrix-2` | latex | `A` | `\begin{pmatrix} 1 & -2 \\ 3 & 4 \end{pmatrix}` | `2` | | the greatest singular value, from the root of the sum of the squares |
| | text | `A` | `[[1, -2], [3, 4]]` | `2` | | |
| | unicode | `A` | `[[1, −2], [3, 4]]` | | `₂` | |
| | python | | `[[1, -2], [3, 4]]` | `2` | | |

## Oracle

```python
expected = Matrix(value).norm(p)
```

## Second route

NumPy's norm of the same order.

```python
order = np.inf if p == oo else p
by_numpy = np.linalg.norm(np.array(value, dtype=float), order)
assert abs(float(expected) - by_numpy) < 1e-9
```

## Forms

### Double bars and the order beneath

**Form** · `subscript` · Golub and Van Loan (2013), §2.2, Vector Norms, and §2.3, Matrix Norms: `\|x\|_p` and `\|A\|_F`

```latex
«name» := «value»
\|«name»\|_{«p»}
```

### The order beneath, without braces

**Form** · `subscript-unbraced` · docs/matrices.md, The notation of linear algebra: `\|A\|_2`, `\|A\|_F`, `\|A\|_\infty`, `\|x\|_1`

```latex
«name» := «value»
\|«name»\|_«p»
```

### Double bars, opening and closing

**Form** · `lVert` · the amsmath package, Delimiters: `\lVert` and `\rVert`, with the order beneath

```latex
«name» := «value»
\lVert «name» \rVert_{«p»}
```

### Double bars that grow

**Form** · `left-right` · Lamport (1994), §3.3.4, Delimiters: `\left\|` and `\right\|`, with the order beneath

```latex
«name» := «value»
\left\| «name» \right\|_{«p»}
```

### A call of norm

**Form** · `norm-call` · MATLAB's `norm(X, p)`, with the orders 1, 2, `Inf` and `'fro'`

```text
«name» := «value»
norm(«name», «p»)
```

### The order beneath, in Unicode

**Form** · `unicode-subscript` · the Unicode Standard, U+2016 DOUBLE VERTICAL LINE, U+2081 SUBSCRIPT ONE and U+2082 SUBSCRIPT TWO: `‖x‖₁`; an order with no subscript character after `_`

```unicode
«name» := «value»
‖«name»‖«lowered»
```
