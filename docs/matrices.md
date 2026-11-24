# Matrices

> **For:** anyone who computes with matrices. **You will learn:** how to write a matrix,
> and the notation of linear algebra as books write it. **Before this:**
> [The language](language.md).

```python
>>> from relativisticpy import Workbook
>>> wb = Workbook()

```

## Writing a matrix

Matrices are written as LaTeX environments, or as lists of rows, and `vmatrix` is a
determinant:

```python
>>> wb.exe(r"""
...     \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}
...     \begin{vmatrix} a & b \\ c & d \end{vmatrix}
... """)
[[[1, 2], [3, 4]], a*d - b*c]

```

## Arithmetic

In arithmetic, lists and matrices are vectors and matrices: a number scales
them, arrays of one shape add, matrices multiply as matrices, and a square
matrix has integer powers, so \(A^{-1}\) is its inverse. A matrix written in blocks,
as `\begin{pmatrix} B & AB \end{pmatrix}`, is one matrix. Anything else, such as
adding a number to a list, is an error. NumPy's `A @ B` is the product `A B` of matrices,
with a note, and of anything else it is refused.

```python
>>> wb.exe(r"""
...     2 [1, 2] + [0, 1]
...     \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} u \\ v \end{pmatrix}
...     \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}^{-1}
... """)
[[2, 5], [[u + 2*v], [3*u + 4*v]], [[-2, 1], [3/2, -1/2]]]

```

## The notation of linear algebra

The rest of the notation of matrices is read as books write it, by what it is written
on: a superscript T is the transpose of a matrix and a power of a number, and
\(\rho(A)\) is the spectral radius of a matrix, while \(\rho(1 + 3w)\) is not. Each
reading is recorded, as `Workbook.explain` shows, with a note where another was
possible, and a definition of the program's own always wins. The functions of a matrix
are in the [reference](reference.md#functions), and `relativisticpy help kron` describes
one in full.

| Written | Read as |
|---|---|
| `A^T`, `A^\top`, `A^\intercal`, `Aᵀ` | the transpose, where T has no value |
| `A^t`, `{}^tA`, `\tilde{A}` | the transpose too, with a note: where t means nothing, `A^t` is no power |
| `A^\dagger`, `A^*`, `A^{-1}`, `A^+` | the conjugate transpose, the inverse and the pseudoinverse |
| `\operatorname{tr} A`, `\det A`, `\operatorname{rank} A` | the trace, the determinant and the rank; without brackets, of the factors that follow, so `\operatorname{tr} A B` is tr(AB) |
| `e^{A t}`, `\exp(A)`, `\log A`, `\sqrt{A}`, `A^{1/2}`, `A^k` | functions of the matrix, not of each entry |
| `\rho(A)`, `\kappa(A)`, `\lambda_{\max}(A)`, `\lambda_k(A)` | the spectral radius, the condition number and the eigenvalues, counted from the smallest |
| `\operatorname{spec}(A)`, `\sigma(A)` | the spectrum: every eigenvalue, from the smallest, as often as it is one; `\sigma` of what is no matrix is what it was |
| `\|A\|_2`, `\|A\|_F`, `\|A\|_\infty`, `\|x\|_1` | norms, of orders 2, Frobenius's, \(\infty\) and 1 |
| `\operatorname{adj} A`, `\ker A`, `\dim \ker A`, `\operatorname{vec} A`, `(A^3)_{12}` | the adjugate, the null space and its dimension, the columns stacked, and an entry, counted from 1 |
| `A \otimes B`, `A \oplus B` | the Kronecker product and the direct sum; between tensors, `\otimes` is their product |
| `I`, `\mathbb{1}`, `𝟙`, `\mathbf{1}` | added to a matrix, the identity of its size; `\mathbf{1}` in a product, the vector of ones of the size it meets |

A number added to a square matrix is that multiple of the identity, and a product that is
a 1 × 1 matrix, as a row times a matrix times a column, is the number it holds, however it
is written. A function of a matrix of a name nothing defines, as `\det M` or
`\operatorname{tr}(\ln M)`, is an unknown value, det(M) or tr(log M), since M may be a
matrix of any size: a claim about it relates, and is not proven by its 1 × 1 case. Each
says so in a note. In plain text, two bars side by side are a norm, as `\|` writes one:
`||x||_1` is `\|x\|_1`.

```python
>>> wb.exe(r"""
...     A := \begin{pmatrix} 0 & 1 \\ -2 & -3 \end{pmatrix}
...     B := \begin{pmatrix} 0 \\ 1 \end{pmatrix}
...     C := \begin{pmatrix} 1 & 0 \end{pmatrix}
...     \left(\operatorname{tr} A\right)^2 - 4 \det A
...     \operatorname{rank} \begin{pmatrix} B & AB \end{pmatrix}
...     C (s I - A)^{-1} B
...     \lambda_{\max}(A) + \|A\|_\infty
...     C \cdot B
...     \sigma(A)
... """)
[1, 2, 1/(s**2 + 3*s + 2), 4, 0, [-2, -1]]
>>> wb.exe(r"""
...     R := \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}
...     e^{R t}
...     R^T R + \rho(R)
... """)
[[[cos(t), sin(t)], [-sin(t), cos(t)]], [[2, 0], [0, 2]]]
>>> print(wb.explain(r"R^T - \lambda I"))
line 1: R^T - \lambda I
  matrix or number: R^T is the transpose of R, since R is a matrix, and T has no value
  matrix or number: I is the identity of the size of R^T, since nothing defines I, and it is added to the matrix R^T
  R: a value, defined on line 2
  \lambda: a symbol
  I: the identity
  note: I is read as the identity of the size of R^T, as it is added to it; to name it, define I

```

A power SymPy leaves undone, as \(A^n\) of some singular matrices, is refused, saying so.
`eig(A)` of a matrix written without indices is its eigenvalues, as MATLAB gives them and
`spec(A)` writes them, with a note; of a map written with its indices, as
`eig(A^{a}{}_{b})`, it is the eigenvalues and the eigenvectors, as NumPy's `eig` gives them:

```python
>>> wb.expr("eig(A)")
[-2, -1]

```

## Maps written with indices

Given a map written with its indices, as `A^{a}{}_{b}`, a function of a matrix acts on
its matrix of components, and its value is a number or a map of the same space (see
[Beyond spacetime](spaces.md#linear-algebra)).

**Next:** [Vectors](vectors.md), or [Beyond spacetime](spaces.md), for linear algebra in
index notation over data.
