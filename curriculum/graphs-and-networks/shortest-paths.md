# Distances and shortest paths

The distance between two vertices: in an unweighted graph, the first power of \(A\) with
a nonzero entry; in a weighted one, by the min-plus recurrence of the all-pairs
algorithm, and, as the books define it, the least weight over all paths. The weighted
graph is Cormen, Leiserson, Rivest and Stein's example for Dijkstra's algorithm, with
the vertices \(s, t, x, y, z\) in that order.

## Distance from powers of the adjacency matrix

**Level 4** · `graph.paths.distance-from-powers` · Biggs (1993), chapter 2

The distance between two vertices is the least \(k\) for which some walk of length \(k\)
joins them, \(\min\{k \geq 1 : (A^k)_{uv} > 0\}\): 3 between opposite vertices of a
six-cycle.

```latex
A := \begin{pmatrix} 0 & 1 & 0 & 0 & 0 & 1 \\ 1 & 0 & 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 & 0 & 1 \\ 1 & 0 & 0 & 0 & 1 & 0 \end{pmatrix}
\min\left\{ k \geq 1 : \left(A^k\right)_{14} > 0 \right\}
```

```python
from scipy.sparse.csgraph import shortest_path

A = Matrix(6, 6, lambda i, j: 1 if (i - j) % 6 in (1, 5) else 0)
expected = min(k for k in range(1, 7) if (A**k)[0, 3] > 0)
assert expected == 3
assert shortest_path(np.array(A.tolist(), float), unweighted=True)[0, 3] == 3
```

## All-pairs distances by the min-plus product

**Level 4** · `graph.paths.min-plus` · Cormen, Leiserson, Rivest and Stein (2009), section 25.1

\(l^{(m)}_{ij}\), the least weight of a path from \(i\) to \(j\) with at most \(m\) edges, obeys
\(l^{(m)}_{ij} = \min_k (l^{(m-1)}_{ik} + w_{kj})\): matrix multiplication with \(\min\) for
sum and \(+\) for product. The weights come from Python, with \(\infty\) where there is no
edge; with five vertices, four edges suffice, and \(l^{(4)}_{13} = \delta(s, x) = 9\).

```python-scope
import numpy as np

weights = np.array(
    [
        [0, 10, np.inf, 5, np.inf],
        [np.inf, 0, 1, 2, np.inf],
        [np.inf, np.inf, 0, np.inf, 4],
        [np.inf, 3, 9, 0, 2],
        [7, np.inf, 6, np.inf, 0],
    ]
)
```

```latex
w_{ij} := \texttt{weights}
l^{(1)}_{ij} := w_{ij}
l^{(m)}_{ij} := \min_{k} \left(l^{(m-1)}_{ik} + w_{kj}\right)
l^{(4)}_{13}
```

```python
from scipy.sparse.csgraph import shortest_path

inf = np.inf
w = np.array(
    [
        [0, 10, inf, 5, inf],
        [inf, 0, 1, 2, inf],
        [inf, inf, 0, inf, 4],
        [inf, 3, 9, 0, 2],
        [7, inf, 6, inf, 0],
    ]
)
lengths = w.copy()
for _ in range(3):  # l^(2), l^(3), l^(4)
    lengths = np.min(lengths[:, :, None] + w[None, :, :], axis=1)
assert list(lengths[0]) == [0, 8, 9, 5, 7]  # the book's distances from s
dijkstra = shortest_path(np.where(np.isinf(w), 0, w), method="D")
assert np.allclose(dijkstra, lengths)
expected = 9
```

## The shortest-path weight

**Level 5** · `graph.paths.shortest` · Cormen, Leiserson, Rivest and Stein (2009), chapter 24

The book's definition: the weight of a path is the sum of the weights of its edges, and
\(\delta(u, v)\) is the least weight over all paths from \(u\) to \(v\). A minimum over a set
of paths, objects the language has no design for, with the edge weights given in Python
as a dictionary.

```python-scope
weights = {
    ("s", "t"): 10,
    ("s", "y"): 5,
    ("t", "x"): 1,
    ("t", "y"): 2,
    ("x", "z"): 4,
    ("y", "t"): 3,
    ("y", "x"): 9,
    ("y", "z"): 2,
    ("z", "s"): 7,
    ("z", "x"): 6,
}
```

```latex
V := \{s, t, x, y, z\}
w := \texttt{weights}
w(p) := \sum_{i=1}^{k} w(v_{i-1}, v_i), \quad p = \langle v_0, v_1, \ldots, v_k \rangle
\delta(u, v) := \min\left\{ w(p) : u \overset{p}{\leadsto} v \right\}
\delta(s, x)
```

```python
from itertools import permutations

weights = {
    ("s", "t"): 10,
    ("s", "y"): 5,
    ("t", "x"): 1,
    ("t", "y"): 2,
    ("x", "z"): 4,
    ("y", "t"): 3,
    ("y", "x"): 9,
    ("y", "z"): 2,
    ("z", "s"): 7,
    ("z", "x"): 6,
}
middle = ["t", "y", "z"]
paths = [["s", *p, "x"] for r in range(4) for p in permutations(middle, r)]  # simple paths
weight = lambda p: sum(weights[e] for e in zip(p, p[1:]))
expected = min(weight(p) for p in paths if all(e in weights for e in zip(p, p[1:])))
assert expected == 9  # s, y, t, x
```
