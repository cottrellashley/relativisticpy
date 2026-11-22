# Laplacians and spanning trees

The Laplacian \(L = D - A\) of a graph and its spectrum: the algebraic connectivity of a
cycle, the components counted by the kernel, the eigenvectors of a cycle, the
normalized Laplacian; and Kirchhoff's matrix-tree theorem, which counts spanning trees
by a cofactor of \(L\) or by its nonzero eigenvalues.

## The largest eigenvalue of a regular graph

**Level 1** · `graph.spectrum.regular` · Biggs (1993), chapter 3

A graph in which every vertex has degree \(k\) has \(k\) as its largest adjacency eigenvalue:
\(\lambda_{\max} = 3\) for the complete graph on four vertices.

```latex
\lambda_{\max}\begin{pmatrix} 0 & 1 & 1 & 1 \\ 1 & 0 & 1 & 1 \\ 1 & 1 & 0 & 1 \\ 1 & 1 & 1 & 0 \end{pmatrix}
```

```python
A = ones(4, 4) - eye(4)
expected = max(A.eigenvals())
assert expected == 3
assert A * ones(4, 1) == 3 * ones(4, 1)  # the vector of ones is its eigenvector
```

## The determinant of a reduced Laplacian

**Level 1** · `graph.trees.reduced-laplacian` · Biggs (1993), chapter 6

The Laplacian of the complete graph on four vertices with its first row and column
removed: its determinant, 16, is the number of spanning trees, \(4^{4-2}\).

```latex
\det\begin{pmatrix} 3 & -1 & -1 \\ -1 & 3 & -1 \\ -1 & -1 & 3 \end{pmatrix}
```

```python
expected = Matrix([[3, -1, -1], [-1, 3, -1], [-1, -1, 3]]).det()
assert expected == 16 == 4 ** (4 - 2)
```

## The algebraic connectivity of a cycle

**Level 2** · `graph.laplacian.algebraic-connectivity` · Fiedler (1973)

The second-smallest eigenvalue of the Laplacian, \(\lambda_2(L)\), with the eigenvalues in
increasing order: positive exactly when the graph is connected. For the five-cycle it is
\(2 - 2\cos\frac{2\pi}{5} = \frac{5 - \sqrt{5}}{2}\).

```latex
A := \begin{pmatrix} 0 & 1 & 0 & 0 & 1 \\ 1 & 0 & 1 & 0 & 0 \\ 0 & 1 & 0 & 1 & 0 \\ 0 & 0 & 1 & 0 & 1 \\ 1 & 0 & 0 & 1 & 0 \end{pmatrix}
D := \operatorname{diag}(2, 2, 2, 2, 2)
L := D - A
\lambda_2(L)
```

```python
A = Matrix(5, 5, lambda i, j: 1 if (i - j) % 5 in (1, 4) else 0)
L = 2 * eye(5) - A
eigenvalues = sorted(L.eigenvals(multiple=True), key=lambda v: float(v))
expected = eigenvalues[1]
assert simplify(expected - (5 - sqrt(5)) / 2) == 0
assert simplify(expected - (2 - 2 * cos(2 * pi / 5))) == 0
assert abs(np.sort(np.linalg.eigvalsh(np.array(L.tolist(), float)))[1] - float(expected)) < 1e-12
```

## Counting components

**Level 2** · `graph.laplacian.components` · standard

The multiplicity of the eigenvalue 0 of the Laplacian is the number of connected
components: a triangle and a separate edge have \(\dim \ker L = 2\).

```latex
A := \begin{pmatrix} 0 & 1 & 1 & 0 & 0 \\ 1 & 0 & 1 & 0 & 0 \\ 1 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 1 & 0 \end{pmatrix}
D := \operatorname{diag}(2, 2, 2, 1, 1)
L := D - A
\dim \ker L
```

```python
from scipy.sparse.csgraph import connected_components

A = Matrix([[0, 1, 1, 0, 0], [1, 0, 1, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0]])
L = diag(2, 2, 2, 1, 1) - A
expected = len(L.nullspace())
assert expected == 2
assert connected_components(np.array(A.tolist(), float), directed=False)[0] == 2
```

## Spanning trees from the adjugate

**Level 2** · `graph.trees.adjugate` · Biggs (1993), chapter 6

Kirchhoff's theorem: every cofactor of the Laplacian is the number of spanning trees, so
the adjugate of \(L\) is \(\tau(G)\) in every entry. For the complete graph on four vertices,
\((\operatorname{adj} L)_{11} = 16\).

```latex
L := \begin{pmatrix} 3 & -1 & -1 & -1 \\ -1 & 3 & -1 & -1 \\ -1 & -1 & 3 & -1 \\ -1 & -1 & -1 & 3 \end{pmatrix}
\left(\operatorname{adj} L\right)_{11}
```

```python
L = 4 * eye(4) - ones(4, 4)
assert L.adjugate() == 16 * ones(4, 4)
expected = L.adjugate()[0, 0]
assert expected == 16 == 4 ** (4 - 2)  # Cayley's formula
```

## The Laplacian as a quadratic form

**Level 2** · `graph.laplacian.quadratic-form` · standard

\(\mathbf{x}^T L \mathbf{x} = \sum_{\{i, j\} \in E} (x_i - x_j)^2\): for the path on three
vertices, a claim for every \(\mathbf{x}\), which holds, and shows \(L\) is positive
semidefinite.

```latex
L := \begin{pmatrix} 1 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 1 \end{pmatrix}
\mathbf{x} := \begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix}
\mathbf{x}^T L \mathbf{x} = (x_1 - x_2)^2 + (x_2 - x_3)^2
```

```python
x = Matrix(symbols("x_1 x_2 x_3"))
L = Matrix([[1, -1, 0], [-1, 2, -1], [0, -1, 1]])
assert expand((x.T * L * x)[0] - (x[0] - x[1]) ** 2 - (x[1] - x[2]) ** 2) == 0
expected = True
```

## Spanning trees from the spectrum

**Level 3** · `graph.trees.spectral` · Biggs (1993), chapter 6

The matrix-tree theorem in spectral form, \(\tau(G) = \frac{1}{n}\prod_{\lambda \neq 0}
\lambda\), the product over the nonzero eigenvalues of \(L\). The path on four vertices is a
tree, so the product \((2 - \sqrt{2}) \cdot 2 \cdot (2 + \sqrt{2}) = 4\), divided by 4, is 1.

```latex
L := \begin{pmatrix} 1 & -1 & 0 & 0 \\ -1 & 2 & -1 & 0 \\ 0 & -1 & 2 & -1 \\ 0 & 0 & -1 & 1 \end{pmatrix}
\frac{1}{4} \prod_{\substack{\lambda \in \operatorname{spec}(L) \\ \lambda \neq 0}} \lambda
```

```python
L = Matrix([[1, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 1]])
assert L.eigenvals() == {0: 1, 2 - sqrt(2): 1, 2: 1, 2 + sqrt(2): 1}
expected = simplify(Rational(1, 4) * (2 - sqrt(2)) * 2 * (2 + sqrt(2)))
assert expected == 1 == L[1:, 1:].det()
```

## The eigenvectors of a cycle

**Level 3** · `graph.laplacian.cycle-eigenvector` · Brouwer and Haemers (2012), chapter 1

The Laplacian of a cycle is circulant, so the Fourier vectors \((1, \omega, \omega^2,
\ldots)\), with \(\omega = e^{2\pi i/n}\), are its eigenvectors, with the eigenvalues
\(2 - 2\cos\frac{2\pi k}{n}\). For the six-cycle and \(k = 1\) the eigenvalue is 1.

```latex
A := \begin{pmatrix} 0 & 1 & 0 & 0 & 0 & 1 \\ 1 & 0 & 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 & 0 & 1 \\ 1 & 0 & 0 & 0 & 1 & 0 \end{pmatrix}
L := \operatorname{diag}(2, 2, 2, 2, 2, 2) - A
\omega := e^{2\pi i/6}
\mathbf{v} := \begin{pmatrix} 1 \\ \omega \\ \omega^2 \\ \omega^3 \\ \omega^4 \\ \omega^5 \end{pmatrix}
L \mathbf{v} = \left(2 - 2\cos\frac{2\pi}{6}\right) \mathbf{v}
```

```python
A = Matrix(6, 6, lambda i, j: 1 if (i - j) % 6 in (1, 5) else 0)
L = 2 * eye(6) - A
omega = exp(2 * pi * I / 6)
v = Matrix([omega**j for j in range(6)])
residual = L * v - (2 - 2 * cos(2 * pi / 6)) * v
assert residual.applyfunc(lambda entry: simplify(expand_complex(entry))) == zeros(6, 1)
assert 2 - 2 * cos(2 * pi / 6) == 1
expected = True
```

## The normalized Laplacian of a star

**Level 3** · `graph.laplacian.normalized` · Chung (1997), chapter 1

Chung's normalized Laplacian \(\mathcal{L} = I - D^{-1/2} A D^{-1/2}\), with a matrix raised
to the power \(-\frac{1}{2}\). Its eigenvalues lie in \([0, 2]\), and 2 is among them
exactly when the graph is bipartite, as the star with three leaves is.

```latex
A := \begin{pmatrix} 0 & 1 & 1 & 1 \\ 1 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 \end{pmatrix}
D := \operatorname{diag}(3, 1, 1, 1)
\mathcal{L} := I - D^{-1/2} A D^{-1/2}
\lambda_{\max}(\mathcal{L})
```

```python
A = Matrix([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
root = diag(1 / sqrt(3), 1, 1, 1)
normalized = eye(4) - root * A * root
assert normalized.eigenvals() == {0: 1, 1: 2, 2: 1}
expected = 2
assert abs(max(np.linalg.eigvalsh(np.array(normalized.evalf().tolist(), float))) - 2) < 1e-12
```

## Cayley's formula

**Level 4** · `graph.trees.cayley` · Biggs (1993), chapter 6

The reduced Laplacian of the complete graph on six vertices is \(6I_5 - J_5\), with \(I_5\)
the identity and \(J_5\) the matrix of ones of size five, as graph theory writes them.
Its determinant is the number of spanning trees, \(6^{6-2}\): a claim, which holds.

```latex
\det\left(6 I_5 - J_5\right) = 6^{6-2}
```

```python
determinant = (6 * eye(5) - ones(5, 5)).det()
assert determinant == 1296 == 6**4
assert abs(np.linalg.det(6 * np.eye(5) - np.ones((5, 5))) - 1296) < 1e-8
expected = True
```
