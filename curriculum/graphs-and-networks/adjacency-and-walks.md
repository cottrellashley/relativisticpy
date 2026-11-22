# Adjacency and walks

A graph through its adjacency matrix \(A\): walks of length \(k\) are the entries of
\(A^k\), triangles its trace, edges half the sum of its entries; and, beyond what a
matrix says, a graph given as a pair of sets and its chromatic number.

## Counting walks

**Level 1** · `graph.walks.count` · Biggs (1993), chapter 2

The number of walks of length 3 from one vertex of a triangle to another is the entry
\((A^3)_{12}\) of the cube of its adjacency matrix: 3, one round the triangle each way and
one back and forth.

```latex
\left(\begin{pmatrix} 0 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 0 \end{pmatrix}^3\right)_{12}
```

```python
A = Matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
expected = (A**3)[0, 1]
assert expected == 3
assert sum(A[0, a] * A[a, b] * A[b, 1] for a in range(3) for b in range(3)) == 3  # 0, a, b, 1
```

## Counting triangles

**Level 1** · `graph.walks.triangles` · standard

Each triangle gives six closed walks of length 3, so \(\frac{1}{6}\operatorname{tr}(A^3)\)
counts triangles: two in the bow tie, two triangles sharing a vertex.

```latex
\frac{1}{6} \operatorname{tr}\left(\begin{pmatrix} 0 & 1 & 1 & 0 & 0 \\ 1 & 0 & 1 & 0 & 0 \\ 1 & 1 & 0 & 1 & 1 \\ 0 & 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 1 & 0 \end{pmatrix}^3\right)
```

```python
from itertools import combinations

A = Matrix([[0, 1, 1, 0, 0], [1, 0, 1, 0, 0], [1, 1, 0, 1, 1], [0, 0, 1, 0, 1], [0, 0, 1, 1, 0]])
expected = (A**3).trace() / 6
assert expected == 2
assert sum(1 for a, b, c in combinations(range(5), 3) if A[a, b] and A[b, c] and A[a, c]) == 2
```

## Counting edges

**Level 2** · `graph.adjacency.edges` · Newman (2010), chapter 6

Each edge is counted twice in the sum of the degrees, so the bow tie has
\(\frac{1}{2}\mathbf{1}^T A \mathbf{1} = 6\) edges, with \(\mathbf{1}\) the vector of ones, as
network books write it.

```latex
A := \begin{pmatrix} 0 & 1 & 1 & 0 & 0 \\ 1 & 0 & 1 & 0 & 0 \\ 1 & 1 & 0 & 1 & 1 \\ 0 & 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 1 & 0 \end{pmatrix}
\frac{1}{2} \mathbf{1}^T A \mathbf{1}
```

```python
A = Matrix([[0, 1, 1, 0, 0], [1, 0, 1, 0, 0], [1, 1, 0, 1, 1], [0, 0, 1, 0, 1], [0, 0, 1, 1, 0]])
expected = (ones(1, 5) * A * ones(5, 1))[0] / 2
assert expected == 6
assert sum(1 for i in range(5) for j in range(i + 1, 5) if A[i, j]) == 6
```

## Closed walks in a complete graph

**Level 3** · `graph.walks.closed` · Biggs (1993), chapter 2

The number of closed walks of length \(k\) is \(\operatorname{tr}(A^k) = \sum_i \lambda_i^k\).
For the complete graph on four vertices, with eigenvalues 3 and \(-1\) three times, it is
\(3^k + 3(-1)^k\): a matrix raised to a symbolic power.

```latex
k \geq 1
A := \begin{pmatrix} 0 & 1 & 1 & 1 \\ 1 & 0 & 1 & 1 \\ 1 & 1 & 0 & 1 \\ 1 & 1 & 1 & 0 \end{pmatrix}
\operatorname{tr}\left(A^k\right)
```

```python
k = symbols("k", integer=True, positive=True)
A = ones(4, 4) - eye(4)
assert A.eigenvals() == {3: 1, -1: 3}
expected = 3**k + 3 * (-1) ** k
assert all(expected.subs(k, n) == (A**n).trace() for n in range(1, 8))
```

## The chromatic number of a cycle

**Level 5** · `graph.invariants.chromatic-number` · Diestel (2017), chapter 5

A graph given as a pair of sets, \(G = (V, E)\), and its chromatic number \(\chi(G)\), the
fewest colours that give adjacent vertices different colours: an odd cycle needs three.
A graph as an object, and an invariant defined by a minimum over all colourings, are
beyond any design of the language yet.

```latex
V := \{1, 2, 3, 4, 5\}
E := \{\{1, 2\}, \{2, 3\}, \{3, 4\}, \{4, 5\}, \{5, 1\}\}
G := (V, E)
\chi(G)
```

```python
from itertools import product

edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)]


def colourable(colours):
    for colouring in product(range(colours), repeat=5):
        if all(colouring[a - 1] != colouring[b - 1] for a, b in edges):
            return True
    return False


expected = min(c for c in range(1, 6) if colourable(c))
assert expected == 3 and not colourable(2)
```
