# Traces

The trace of a square matrix, \(\operatorname{tr} A\), the sum of its diagonal, written
with the operator's name in either case, with and without brackets, as a sum, and as each
language of computing writes it. A matrix that is not square has none, and must be
refused. Horn and Johnson (2013) is *Matrix Analysis*, 2nd edition; Goodfellow, Bengio and
Courville (2016) is *Deep Learning*; Riley, Hobson and Bence (2006) is *Mathematical Methods
for Physics and Engineering*, 3rd edition; ISO 80000-2:2019 is the standard for
mathematical signs and symbols; the physics package is the LaTeX package of that name (de
la Barrera, 2012); the Unicode Standard is version 15.0.

`«A»` is the rows of the matrix: in LaTeX, the body of a matrix environment; in plain text
and Unicode, a list of rows. `«n»` is its size.

## Names

```python
a, b, c, d = symbols("a b c d")
```

## Operands

| operand | written in | `A` | `n` | separates |
|---|---|---|---|---|
| `numbers` | latex | `1 & 2 \\ 3 & 4` | `2` | the sum of the diagonal, 5, from the sum of every entry, 10, and the determinant |
| | text, unicode | `[[1, 2], [3, 4]]` | | |
| | python | `[[1, 2], [3, 4]]` | | |
| `negative` | latex | `-2 & 1 \\ 0 & 3` | `2` | entries with a sign |
| | text | `[[-2, 1], [0, 3]]` | | |
| | unicode | `[[−2, 1], [0, 3]]` | | |
| | python | `[[-2, 1], [0, 3]]` | | |
| `symbols` | latex | `a & b \\ c & d` | `2` | entries that are letters: a + d |
| | text, unicode | `[[a, b], [c, d]]` | | |
| | python | `[[a, b], [c, d]]` | | |
| `three-by-three` | latex | `1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9` | `3` | a matrix of three rows |
| | text, unicode | `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]` | | |
| | python | `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]` | | |
| `complex` | latex | `i & 1 \\ 2 & 3` | `2` | a complex entry on the diagonal |
| | text | `[[i, 1], [2, 3]]` | | |
| | unicode | `[[i, 1], [2, 3]]` | | |
| | python | `[[I, 1], [2, 3]]` | | |
| `rectangular` | latex | `1 & 2 & 3 \\ 4 & 5 & 6` | | a matrix that is not square, which has no trace |
| | text, unicode | `[[1, 2, 3], [4, 5, 6]]` | | |
| | python | `[[1, 2, 3], [4, 5, 6]]` | | |

## Oracle

```python
M = Matrix(A)
if not M.is_square:
    error = "RelativisticPyError"
else:
    expected = M.trace()
```

## Second route

The trace is the sum of the eigenvalues, each as often as it is a root of the
characteristic polynomial.

```python
if not M.is_square:
    assert error == "RelativisticPyError"
else:
    by_eigenvalues = sum(value * times for value, times in M.eigenvals().items())
    assert simplify(by_eigenvalues - expected) == 0
```

## Forms

### The operator before a name

**Form** · `tr` · Horn and Johnson (2013), chapter 0, Review and miscellanea: `\operatorname{tr} A`; docs/language.md, Matrices

```latex
A := \begin{pmatrix} «A» \end{pmatrix}
\operatorname{tr} A
```

### The operator before brackets

**Form** · `tr-brackets` · docs/reference.md, Functions, Matrices: `\operatorname{tr}(A)`

```latex
A := \begin{pmatrix} «A» \end{pmatrix}
\operatorname{tr}(A)
```

### A capital T, before a name

**Form** · `Tr` · Riley, Hobson and Bence (2006), §8.8, The trace of a matrix: `\operatorname{Tr} \mathbf{A}`, of a matrix in round brackets

```latex
\mathbf{A} := \begin{pmatrix} «A» \end{pmatrix}
\operatorname{Tr} \mathbf{A}
```

### A capital T, before brackets

**Form** · `Tr-brackets` · Goodfellow, Bengio and Courville (2016), §2.10, The Trace Operator: `\operatorname{Tr}(A)`, of a matrix in square brackets

```latex
A := \begin{bmatrix} «A» \end{bmatrix}
\operatorname{Tr}(A)
```

### An upright name

**Form** · `mathrm` · ISO 80000-2:2019, matrices: `tr A`, the name set upright

```latex
A := \begin{pmatrix} «A» \end{pmatrix}
\mathrm{tr}\, A
```

### The sum of the diagonal

**Form** · `sum` · Riley, Hobson and Bence (2006), §8.8, The trace of a matrix: `\operatorname{Tr} A = \sum_i A_{ii}`

```latex
A := \begin{pmatrix} «A» \end{pmatrix}
\sum_{i=1}^{«n»} A_{ii}
```

### The physics package's trace

**Form** · `physics-tr` · the physics package, Operators, trace and rank: `\tr`

```latex
A := \begin{pmatrix} «A» \end{pmatrix}
\tr A
```

**Refused** · `ParseError` · docs/reference.md, Not supported: `\tr` is an error that says to write `\operatorname{tr}(A)`

### The physics package's trace, with a capital

**Form** · `physics-Tr` · the physics package, Operators, trace and rank: `\Tr`

```latex
A := \begin{pmatrix} «A» \end{pmatrix}
\Tr A
```

**Refused** · `ParseError` · docs/reference.md, Not supported: `\Tr` is an error that says to write `\operatorname{Tr}(A)`

### A call of trace

**Form** · `trace-call` · SymPy's and MATLAB's `trace(A)`

```text
A := «A»
trace(A)
```

### A call of tr

**Form** · `tr-call` · docs/reference.md, Functions, Matrices: `tr`

```text
A := «A»
tr(A)
```

### A call of Tr

**Form** · `Tr-call` · docs/reference.md, Functions, Matrices: `Tr`

```text
A := «A»
Tr(A)
```

### The Wolfram Language's call

**Form** · `wolfram-Tr` · the Wolfram Language's `Tr[m]`, its argument in square brackets

```text
A := «A»
Tr[A]
```

### The operator in Unicode

**Form** · `unicode-tr` · as typed, with Unicode's minus sign, U+2212

```unicode
A := «A»
tr A
```
