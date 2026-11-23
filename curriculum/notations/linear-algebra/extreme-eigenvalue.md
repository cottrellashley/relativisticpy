# The greatest and the least eigenvalue

The greatest and the least eigenvalue of a real symmetric matrix, \(\lambda_{\max}(A)\) and
\(\lambda_{\min}(A)\), whose eigenvalues are real: written with a subscript, as the last and
the first when they are counted from the smallest, and as each language of computing asks
for them. Horn and Johnson (2013) is *Matrix Analysis*, 2nd edition, whose chapter 4 counts
the eigenvalues of a Hermitian matrix from the smallest,
\(\lambda_1 \le \lambda_2 \le \cdots \le \lambda_n\); the Unicode Standard is version 15.0.

`«A»` is the rows of the matrix: in LaTeX, the body of a matrix environment; in plain text
and Unicode, a list of rows. `«end»` is `max` or `min`; `«k»` is the place of that
eigenvalue counted from the smallest, n or 1; `«lowered»` is the subscript in Unicode's
subscript letters.

## Operands

| operand | written in | `A` | `end` | `k` | `lowered` | separates |
|---|---|---|---|---|---|---|
| `max` | latex | `2 & 1 \\ 1 & 2` | `\max` | `2` | | the greatest of 1 and 3 |
| | text | `[[2, 1], [1, 2]]` | `max` | | | |
| | unicode | `[[2, 1], [1, 2]]` | `max` | | `ₘₐₓ` | |
| | python | `[[2, 1], [1, 2]]` | `max` | | | |
| `min` | latex | `2 & 1 \\ 1 & 2` | `\min` | `1` | | the least of 1 and 3 |
| | text | `[[2, 1], [1, 2]]` | `min` | | | |
| | unicode | `[[2, 1], [1, 2]]` | `min` | | `ₘᵢₙ` | |
| | python | `[[2, 1], [1, 2]]` | `min` | | | |
| `max-negative` | latex | `-3 & 1 \\ 1 & -3` | `\max` | `2` | | the greatest of −4 and −2, which is −2, from the greatest modulus, 4 |
| | text | `[[-3, 1], [1, -3]]` | `max` | | | |
| | unicode | `[[−3, 1], [1, −3]]` | `max` | | `ₘₐₓ` | |
| | python | `[[-3, 1], [1, -3]]` | `max` | | | |
| `min-negative` | latex | `-3 & 1 \\ 1 & -3` | `\min` | `1` | | the least of −4 and −2 |
| | text | `[[-3, 1], [1, -3]]` | `min` | | | |
| | unicode | `[[−3, 1], [1, −3]]` | `min` | | `ₘᵢₙ` | |
| | python | `[[-3, 1], [1, -3]]` | `min` | | | |
| `max-three` | latex | `2 & 0 & 0 \\ 0 & 3 & 4 \\ 0 & 4 & 9` | `\max` | `3` | | a matrix of three rows: the greatest of 1, 2 and 11 |
| | text | `[[2, 0, 0], [0, 3, 4], [0, 4, 9]]` | `max` | | | |
| | unicode | `[[2, 0, 0], [0, 3, 4], [0, 4, 9]]` | `max` | | `ₘₐₓ` | |
| | python | `[[2, 0, 0], [0, 3, 4], [0, 4, 9]]` | `max` | | | |
| `min-three` | latex | `2 & 0 & 0 \\ 0 & 3 & 4 \\ 0 & 4 & 9` | `\min` | `1` | | the least of 1, 2 and 11, which is not on the diagonal |
| | text | `[[2, 0, 0], [0, 3, 4], [0, 4, 9]]` | `min` | | | |
| | unicode | `[[2, 0, 0], [0, 3, 4], [0, 4, 9]]` | `min` | | `ₘᵢₙ` | |
| | python | `[[2, 0, 0], [0, 3, 4], [0, 4, 9]]` | `min` | | | |
| `max-irrational` | latex | `1 & 1 \\ 1 & 0` | `\max` | `2` | | eigenvalues that are roots: the greatest is (1 + √5)/2 |
| | text | `[[1, 1], [1, 0]]` | `max` | | | |
| | unicode | `[[1, 1], [1, 0]]` | `max` | | `ₘₐₓ` | |
| | python | `[[1, 1], [1, 0]]` | `max` | | | |
| `min-irrational` | latex | `1 & 1 \\ 1 & 0` | `\min` | `1` | | eigenvalues that are roots: the least is (1 − √5)/2 |
| | text | `[[1, 1], [1, 0]]` | `min` | | | |
| | unicode | `[[1, 1], [1, 0]]` | `min` | | `ₘᵢₙ` | |
| | python | `[[1, 1], [1, 0]]` | `min` | | | |
| `min-singular` | latex | `4 & -2 \\ -2 & 1` | `\min` | `1` | | a least eigenvalue of 0, which exists, of a singular matrix |
| | text | `[[4, -2], [-2, 1]]` | `min` | | | |
| | unicode | `[[4, −2], [−2, 1]]` | `min` | | `ₘᵢₙ` | |
| | python | `[[4, -2], [-2, 1]]` | `min` | | | |

## Oracle

```python
M = Matrix(A)
expected = end(M.eigenvals())
```

## Second route

NumPy's eigenvalues of a symmetric matrix, in increasing order, the last or the first.

```python
increasing = np.linalg.eigvalsh(np.array(M.tolist(), dtype=float))
by_numpy = increasing[-1] if end is max else increasing[0]
assert abs(float(expected) - by_numpy) < 1e-9
```

## Forms

### A subscript max or min

**Form** · `subscript` · Horn and Johnson (2013), chapter 4, Hermitian matrices, symmetric matrices, and congruences: `\lambda_{\max}(A)` and `\lambda_{\min}(A)`

```latex
A := \begin{bmatrix} «A» \end{bmatrix}
\lambda_{«end»}(A)
```

### A subscript, without brackets

**Form** · `subscript-without-brackets` · docs/reference.md, Functions, Matrices: `\lambda_{\max} A`

```latex
A := \begin{pmatrix} «A» \end{pmatrix}
\lambda_{«end»} A
```

### Counted from the smallest

**Form** · `counted` · Horn and Johnson (2013), chapter 4: `\lambda_1 \le \lambda_2 \le \cdots \le \lambda_n`, so that the greatest is `\lambda_n(A)` and the least `\lambda_1(A)`

```latex
A := \begin{bmatrix} «A» \end{bmatrix}
\lambda_{«k»}(A)
```

### A call of lambda_max or lambda_min

**Form** · `lambda-call` · docs/reference.md, Functions, Matrices: `lambda_max`, `lambda_min`

```text
A := «A»
lambda_«end»(A)
```

### The greatest or the least of eig

**Form** · `max-of-eig` · MATLAB's `max(eig(A))` and `min(eig(A))`

```text
A := «A»
«end»(eig(A))
```

### A subscript in Unicode

**Form** · `unicode-subscript` · the Unicode Standard, U+03BB GREEK SMALL LETTER LAMDA: `λ_max(A)`, as typed

```unicode
A := «A»
λ_«end»(A)
```

### Subscript letters in Unicode

**Form** · `unicode-subscript-letters` · the Unicode Standard, Superscripts and Subscripts and Phonetic Extensions: `λₘₐₓ(A)` with U+2098, U+2090 and U+2093, and `λₘᵢₙ(A)` with U+2098, U+1D62 and U+2099

```unicode
A := «A»
λ«lowered»(A)
```
