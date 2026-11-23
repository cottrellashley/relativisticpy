# Eigenvalues

The eigenvalues of a square matrix, the roots of its characteristic polynomial
\(\det(A - \lambda I)\), each as often as it is a root, in any order: written as the
spectrum, as the roots of the characteristic equation, and as each language of computing
asks for them. A matrix that is not square has none, and must be refused. Strang (2016) is
*Introduction to Linear Algebra*, 5th edition; Horn and Johnson (2013) is *Matrix
Analysis*, 2nd edition; the Unicode Standard is version 15.0.

`«A»` is the rows of the matrix: in LaTeX, the body of a matrix environment; in plain text
and Unicode, a list of rows. The spectrum \(\sigma(A)\) of Horn and Johnson is a set, and
the solutions of an equation are its distinct roots: each holds a repeated eigenvalue once.

## Names

```python
a, b, lam = symbols("a b lambda")
```

## Operands

| operand | written in | `A` | separates |
|---|---|---|---|
| `symmetric` | latex | `2 & 1 \\ 1 & 2` | eigenvalues, 1 and 3, from the entries of the diagonal |
| | text, unicode | `[[2, 1], [1, 2]]` | |
| | python | `[[2, 1], [1, 2]]` | |
| `triangular` | latex | `2 & 5 \\ 0 & -1` | a triangular matrix, whose eigenvalues are its diagonal, one negative |
| | text | `[[2, 5], [0, -1]]` | |
| | unicode | `[[2, 5], [0, −1]]` | |
| | python | `[[2, 5], [0, -1]]` | |
| `repeated` | latex | `2 & 1 \\ 0 & 2` | an eigenvalue that is a double root, counted twice |
| | text, unicode | `[[2, 1], [0, 2]]` | |
| | python | `[[2, 1], [0, 2]]` | |
| `rotation` | latex | `0 & -1 \\ 1 & 0` | a real matrix whose eigenvalues are i and −i |
| | text | `[[0, -1], [1, 0]]` | |
| | unicode | `[[0, −1], [1, 0]]` | |
| | python | `[[0, -1], [1, 0]]` | |
| `three-by-three` | latex | `2 & 0 & 0 \\ 0 & 3 & 4 \\ 0 & 4 & 9` | a matrix of three rows: 1, 2 and 11 |
| | text, unicode | `[[2, 0, 0], [0, 3, 4], [0, 4, 9]]` | |
| | python | `[[2, 0, 0], [0, 3, 4], [0, 4, 9]]` | |
| `symbols` | latex | `a & b \\ b & a` | entries that are letters: a − b and a + b |
| | text, unicode | `[[a, b], [b, a]]` | |
| | python | `[[a, b], [b, a]]` | |
| `irrational` | latex | `1 & 1 \\ 1 & 0` | eigenvalues that are roots: (1 ± √5)/2 |
| | text, unicode | `[[1, 1], [1, 0]]` | |
| | python | `[[1, 1], [1, 0]]` | |
| `rectangular` | latex | `1 & 2 & 3 \\ 4 & 5 & 6` | a matrix that is not square, which has no eigenvalues |
| | text, unicode | `[[1, 2, 3], [4, 5, 6]]` | |
| | python | `[[1, 2, 3], [4, 5, 6]]` | |

## Oracle

```python
M = Matrix(A)
if not M.is_square:
    error = "RelativisticPyError"
else:
    expected = [value for value, times in M.eigenvals().items() for _ in range(times)]
    in_any_order = True
```

## Second route

The product of λ minus each eigenvalue is the characteristic polynomial, det(λI − A).

```python
if not M.is_square:
    assert error == "RelativisticPyError"
else:
    characteristic = (lam * eye(M.rows) - M).det()
    assert expand(prod(lam - value for value in expected) - characteristic) == 0
```

## Forms

### The spectrum, by name

**Form** · `spec` · docs/spaces.md, Linear algebra: `\operatorname{spec}(A)`, the eigenvalues

```latex
A := \begin{pmatrix} «A» \end{pmatrix}
\operatorname{spec}(A)
```

### The spectrum, as a set

**Form** · `sigma` · Horn and Johnson (2013), chapter 1, Eigenvalues, eigenvectors, and similarity: the spectrum `\sigma(A)`, the set of the eigenvalues of A

```latex
A := \begin{bmatrix} «A» \end{bmatrix}
\sigma(A)
```

**Not of** · `repeated` · a set holds a repeated eigenvalue once

### The roots of the characteristic equation

**Form** · `characteristic` · Strang (2016), §6.1, Introduction to Eigenvalues: the eigenvalues solve `\det(A - \lambda I) = 0`; docs/language.md, Equations: `solve(x^2 = 4, x)`

```latex
A := \begin{bmatrix} «A» \end{bmatrix}
solve(\det(A - \lambda I) = 0, \lambda)
```

**Not of** · `repeated` · the solutions of an equation are its distinct roots

### A call of spec

**Form** · `spec-call` · docs/reference.md, Functions, Matrices: `spec`

```text
A := «A»
spec(A)
```

### A call of eig

**Form** · `eig-call` · MATLAB's `eig(A)`, the eigenvalues, each as often as it is a root

```text
A := «A»
eig(A)
```

### SymPy's method

**Form** · `sympy-eigenvals` · SymPy's `A.eigenvals()`, each eigenvalue with the number of times it is a root

```text
A := «A»
A.eigenvals()
```

### The characteristic equation in plain text

**Form** · `text-characteristic` · Strang (2016), §6.1: `det(A - lambda I) = 0`, typed in plain text; docs/language.md, Equations: `solve`

```text
A := «A»
solve(det(A - lambda I) = 0, lambda)
```

**Not of** · `repeated` · the solutions of an equation are its distinct roots

### The spectrum in Unicode

**Form** · `unicode-sigma` · the Unicode Standard, U+03C3 GREEK SMALL LETTER SIGMA: Horn and Johnson's `σ(A)`, as typed

```unicode
A := «A»
σ(A)
```

**Not of** · `repeated` · a set holds a repeated eigenvalue once

### The characteristic equation in Unicode

**Form** · `unicode-characteristic` · the Unicode Standard, U+03BB GREEK SMALL LETTER LAMDA and U+2212 MINUS SIGN: Strang's equation, as typed

```unicode
A := «A»
solve(det(A − λI) = 0, λ)
```

**Not of** · `repeated` · the solutions of an equation are its distinct roots
