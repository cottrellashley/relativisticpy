# Random walks and centrality

A random walk steps to a uniformly chosen neighbour, with the transition matrix
\(P = D^{-1}A\); its stationary distribution is proportional to the degrees. PageRank is
the stationary distribution of a walk that sometimes jumps at random. Katz centrality
counts walks of every length, and the effective resistance comes from the
pseudoinverse of the Laplacian.

## Katz centrality

**Level 2** · `graph.centrality.katz` · Newman (2010), chapter 7

Katz centrality \((I - \alpha A)^{-1}\mathbf{1}\) counts walks of every length ending at a
vertex, those of length \(k\) weighted by \(\alpha^k\), which converges for
\(\alpha < 1/\lambda_{\max} = 1/\sqrt{2}\). On the path of three vertices with
\(\alpha = \frac{1}{2}\) the middle vertex scores 4 and the ends 3.

```latex
A := \begin{pmatrix} 0 & 1 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 0 \end{pmatrix}
\alpha := \frac{1}{2}
(I - \alpha A)^{-1} \mathbf{1}
```

```python
A = Matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
assert max(A.eigenvals()) == sqrt(2)
expected = (eye(3) - A / 2).inv() * ones(3, 1)
assert expected == Matrix([3, 4, 3])
series = sum(np.linalg.matrix_power(np.array(A.tolist(), float) / 2, k) for k in range(200))
assert np.allclose(series @ np.ones(3), [3, 4, 3])  # the walks, counted
```

## The stationary distribution of a random walk

**Level 3** · `graph.walk.stationary` · Lovász (1993)

The adjacency matrix of the bow tie comes from Python. The walk's stationary
distribution solves \(\boldsymbol{\pi}^T P = \boldsymbol{\pi}^T\) with
\(\boldsymbol{\pi}^T\mathbf{1} = 1\); it is \(\pi_v = d_v/2m\), so the vertex shared by the two
triangles, of degree 4 among 6 edges, has \(\pi_3 = \frac{1}{3}\).

```python-scope
import numpy as np

adjacency = np.array(
    [[0, 1, 1, 0, 0], [1, 0, 1, 0, 0], [1, 1, 0, 1, 1], [0, 0, 1, 0, 1], [0, 0, 1, 1, 0]]
)
```

```latex
A := \texttt{adjacency}
D := \operatorname{diag}(A \mathbf{1})
P := D^{-1} A
\boldsymbol{\pi}^T P = \boldsymbol{\pi}^T, \quad \boldsymbol{\pi}^T \mathbf{1} = 1
\pi_3
```

```python
A = Matrix([[0, 1, 1, 0, 0], [1, 0, 1, 0, 0], [1, 1, 0, 1, 1], [0, 0, 1, 0, 1], [0, 0, 1, 1, 0]])
degrees = [sum(A.row(i)) for i in range(5)]
P = diag(*[Rational(1, d) for d in degrees]) * A
p = symbols("p_1:6")
solution = solve([*(Matrix([p]) * P - Matrix([p])), sum(p) - 1], p)
assert [solution[v] for v in p] == [Rational(d, sum(degrees)) for d in degrees]
expected = solution[p[2]]
assert expected == Rational(1, 3)
walk = np.linalg.matrix_power(np.array(P.tolist(), float), 200)
assert np.allclose(walk[0], [float(solution[v]) for v in p])  # the walk forgets where it began
```

## PageRank

**Level 3** · `graph.pagerank` · Langville and Meyer (2006), chapter 4

Four pages: the first links to the second and third, the second and fourth to the third,
the third to the first. With the links from Python as a column-stochastic matrix \(M\) and
the damping \(d = 0.85\), the PageRank vector solves
\(\mathbf{r} = \frac{1 - d}{N}\mathbf{1} + dM\mathbf{r}\): the third page, which three pages
link to, ranks highest.

```python-scope
import numpy as np

links = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [1, 1, 0, 1], [0, 0, 0, 0]])  # j -> i
transition = links / links.sum(axis=0)
```

```latex
M := \texttt{transition}
d := 0.85
N := 4
(I - d M)^{-1} \frac{1 - d}{N} \mathbf{1}
```

```python
M = Matrix([[0, 0, 1, 0], [Rational(1, 2), 0, 0, 0], [Rational(1, 2), 1, 0, 1], [0, 0, 0, 0]])
d = Rational(17, 20)
exact = (eye(4) - d * M).inv() * ones(4, 1) * (1 - d) / 4
assert sum(exact) == 1
rank = np.ones(4) / 4
for _ in range(300):  # the power method on the Google matrix
    rank = 0.15 / 4 + 0.85 * np.array(M.tolist(), float) @ rank
assert np.allclose(rank, [float(v) for v in exact], atol=1e-12)
assert np.argmax(rank) == 2 and abs(rank[3] - 0.0375) < 1e-12
expected = [float(v) for v in exact]
```

## The effective resistance

**Level 4** · `graph.resistance.effective` · Klein and Randić (1993)

With unit resistors on the edges of a four-cycle, the resistance between two adjacent
vertices is one ohm in parallel with three, \(\frac{3}{4}\). It is
\((\mathbf{e}_1 - \mathbf{e}_2)^T L^{+} (\mathbf{e}_1 - \mathbf{e}_2)\), with \(L^{+}\) the
pseudoinverse of the Laplacian and \(\mathbf{e}_i\) the standard basis vectors.

```latex
L := \begin{pmatrix} 2 & -1 & 0 & -1 \\ -1 & 2 & -1 & 0 \\ 0 & -1 & 2 & -1 \\ -1 & 0 & -1 & 2 \end{pmatrix}
(\mathbf{e}_1 - \mathbf{e}_2)^T L^{+} (\mathbf{e}_1 - \mathbf{e}_2)
```

```python
L = Matrix([[2, -1, 0, -1], [-1, 2, -1, 0], [0, -1, 2, -1], [-1, 0, -1, 2]])
difference = Matrix([1, -1, 0, 0])
expected = (difference.T * L.pinv() * difference)[0]
assert expected == Rational(3, 4) == 1 * 3 / Integer(1 + 3)  # one ohm parallel to three
grounded = np.array(L.tolist(), float)[1:, 1:]  # ground vertex 1, inject one ampere at 2
assert abs(np.linalg.solve(grounded, [1, 0, 0])[0] - 0.75) < 1e-12
```
