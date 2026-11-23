# Determinants

The determinant of a square matrix, \(\det A\), written with the operator before a name or
a matrix written out, between bars, as a call, and as a line of plain text writes it. A
matrix that is not square has none, and must be refused. Strang (2016) is *Introduction to
Linear Algebra*, 5th edition; Lay, Lay and McDonald (2016) is *Linear Algebra and Its
Applications*, 5th edition; Riley, Hobson and Bence (2006) is *Mathematical Methods for
Physics and Engineering*, 3rd edition; Griffiths (2005) is *Introduction to Quantum
Mechanics*, 2nd edition; Lamport (1994) is *LaTeX: A Document Preparation System*, 2nd
edition; the amsmath package is described in its *User's Guide* (American Mathematical
Society, 2020); AsciiMath is the plain text notation of asciimath.org; the Unicode Standard
is version 15.0.

`«A»` is the rows of the matrix: in LaTeX, the body of a matrix environment; in plain text
and Unicode, a list of rows. Bars round a matrix are its determinant, which may be
negative, never an absolute value.

## Names

```python
a, b, c, d = symbols("a b c d")
```

## Operands

| operand | written in | `A` | separates |
|---|---|---|---|
| `numbers` | latex | `1 & 2 \\ 3 & 4` | a determinant of −2, from its absolute value 2, and the permanent 10 |
| | text, unicode | `[[1, 2], [3, 4]]` | |
| | python | `[[1, 2], [3, 4]]` | |
| `swap` | latex | `0 & 1 \\ 1 & 0` | a determinant of −1, from its absolute value 1 |
| | text, unicode | `[[0, 1], [1, 0]]` | |
| | python | `[[0, 1], [1, 0]]` | |
| `negative` | latex | `3 & -2 \\ 4 & 1` | entries with a sign |
| | text | `[[3, -2], [4, 1]]` | |
| | unicode | `[[3, −2], [4, 1]]` | |
| | python | `[[3, -2], [4, 1]]` | |
| `symbols` | latex | `a & b \\ c & d` | entries that are letters: ad − bc |
| | text, unicode | `[[a, b], [c, d]]` | |
| | python | `[[a, b], [c, d]]` | |
| `three-by-three` | latex | `2 & -1 & 0 \\ 1 & 3 & 2 \\ 0 & 1 & 4` | a matrix of three rows, expanded by cofactors |
| | text | `[[2, -1, 0], [1, 3, 2], [0, 1, 4]]` | |
| | unicode | `[[2, −1, 0], [1, 3, 2], [0, 1, 4]]` | |
| | python | `[[2, -1, 0], [1, 3, 2], [0, 1, 4]]` | |
| `triangular` | latex | `-2 & 5 & 7 \\ 0 & 3 & 1 \\ 0 & 0 & 4` | a triangular matrix, whose determinant is the product of its diagonal |
| | text | `[[-2, 5, 7], [0, 3, 1], [0, 0, 4]]` | |
| | unicode | `[[−2, 5, 7], [0, 3, 1], [0, 0, 4]]` | |
| | python | `[[-2, 5, 7], [0, 3, 1], [0, 0, 4]]` | |
| `singular` | latex | `1 & 2 \\ 2 & 4` | a singular matrix, whose determinant is 0, which exists, from a refusal |
| | text, unicode | `[[1, 2], [2, 4]]` | |
| | python | `[[1, 2], [2, 4]]` | |
| `rectangular` | latex | `1 & 2 & 3 \\ 4 & 5 & 6` | a matrix that is not square, which has no determinant |
| | text, unicode | `[[1, 2, 3], [4, 5, 6]]` | |
| | python | `[[1, 2, 3], [4, 5, 6]]` | |

## Oracle

```python
M = Matrix(A)
if not M.is_square:
    error = "RelativisticPyError"
else:
    expected = M.det()
```

## Second route

Leibniz's formula: the sum, over every permutation of the columns, of the product of the
entries it picks, with the sign of the permutation.

```python
from itertools import permutations


def sign(p):
    return (-1) ** sum(1 for i in range(len(p)) for j in range(i + 1, len(p)) if p[i] > p[j])


if not M.is_square:
    assert error == "RelativisticPyError"
else:
    n = M.rows
    leibniz = sum(sign(p) * prod(M[i, p[i]] for i in range(n)) for p in permutations(range(n)))
    assert expand(leibniz - expected) == 0
```

## Forms

### The operator before a name

**Form** · `det` · Strang (2016), §5.1, The Properties of Determinants: `\det A`

```latex
A := \begin{bmatrix} «A» \end{bmatrix}
\det A
```

### The operator before brackets

**Form** · `det-brackets` · Griffiths (2005), Appendix A.3, Matrices: the determinant `\det(T)`, written with brackets, of a matrix in round brackets

```latex
A := \begin{pmatrix} «A» \end{pmatrix}
\det(A)
```

### The operator before a matrix written out

**Form** · `det-written-out` · Lay, Lay and McDonald (2016), §3.1, Introduction to Determinants: `\det` before a matrix written out in square brackets

```latex
\det \begin{bmatrix} «A» \end{bmatrix}
```

### Bars round a matrix written out

**Form** · `vmatrix` · the amsmath package, Matrices: `vmatrix`; Strang (2016), §5.1, writes a determinant between bars

```latex
\begin{vmatrix} «A» \end{vmatrix}
```

### Bars round a name

**Form** · `bars` · Riley, Hobson and Bence (2006), §8.9, The determinant of a matrix: `|\mathbf{A}|`, of a matrix in round brackets

```latex
\mathbf{A} := \begin{pmatrix} «A» \end{pmatrix}
|\mathbf{A}|
```

### Bars that grow

**Form** · `left-right-bars` · the amsmath package, Matrices: `matrix`, without delimiters, set in Lamport's (1994) `\left|` and `\right|`, §3.3.4, Delimiters

```latex
\left| \begin{matrix} «A» \end{matrix} \right|
```

### The operator's name, typed

**Form** · `operatorname` · the amsmath package, Operator names: `\operatorname{det}`, set as `\det` is

```latex
A := \begin{pmatrix} «A» \end{pmatrix}
\operatorname{det} A
```

### A call of det

**Form** · `det-call` · MATLAB's and SymPy's `det(A)`; docs/reference.md, Functions, Matrices: `det`

```text
A := «A»
det(A)
```

### The operator in plain text

**Form** · `text-det` · docs/language.md, Functions you can call: a function without parentheses applies to what follows it

```text
A := «A»
det A
```

### Bars in plain text

**Form** · `text-bars` · AsciiMath: `|A|`

```text
A := «A»
|A|
```

### The operator in Unicode

**Form** · `unicode-det` · as typed, with Unicode's minus sign, U+2212

```unicode
A := «A»
det A
```
