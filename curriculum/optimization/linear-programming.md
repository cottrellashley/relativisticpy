# Linear programming

Linear objectives under linear constraints: the Wyndor Glass problem of Hillier and
Lieberman and its dual, the general form Boyd and Vandenberghe display, a knapsack whose
variables must be 0 or 1, and the transport problem between two distributions.

## The Wyndor Glass Company

**Level 2** · `opt.linear.wyndor` · Hillier and Lieberman (2015), chapter 3

The prototype linear program: profit \(3x_1 + 5x_2\) under three capacity constraints.
The optimum is at the vertex \((2, 6)\), with profit 36.

```latex
\max_{x_1, x_2} \; 3x_1 + 5x_2 \quad \text{subject to} \quad x_1 \leq 4, \quad 2x_2 \leq 12, \quad 3x_1 + 2x_2 \leq 18, \quad x_1, x_2 \geq 0
```

```python
from scipy.optimize import linprog

found = linprog([-3, -5], A_ub=[[1, 0], [0, 2], [3, 2]], b_ub=[4, 12, 18], bounds=[(0, None)] * 2)
assert np.allclose(found.x, [2, 6])
vertices = [(0, 0), (4, 0), (4, 3), (2, 6), (0, 6)]  # the corners of the feasible region
expected = max(3 * a + 5 * b for a, b in vertices)
assert expected == 36 and abs(-found.fun - expected) < 1e-9
```

## The dual of a linear program

**Level 3** · `opt.linear.duality` · Hillier and Lieberman (2015), chapter 6

The dual of the Wyndor problem, written from its data in matrix form: minimise
\(b^T y\) subject to \(A^T y \geq c\) and \(y \geq 0\). By strong duality its optimum is the
primal's, 36, at \(y = (0, \frac{3}{2}, 1)\), the shadow prices of the three
capacities.

```latex
A := \begin{pmatrix} 1 & 0 \\ 0 & 2 \\ 3 & 2 \end{pmatrix}
b := \begin{pmatrix} 4 \\ 12 \\ 18 \end{pmatrix}
c := \begin{pmatrix} 3 \\ 5 \end{pmatrix}
\min_{y} \; b^T y \quad \text{subject to} \quad A^T y \geq c, \quad y \geq 0
```

```python
from scipy.optimize import linprog

A = np.array([[1, 0], [0, 2], [3, 2]])
found = linprog([4, 12, 18], A_ub=-A.T, b_ub=[-3, -5], bounds=[(0, None)] * 3)
assert np.allclose(found.x, [0, 1.5, 1])
y = [0, Rational(3, 2), 1]
assert 1 * y[0] + 3 * y[2] >= 3 and 2 * y[1] + 2 * y[2] >= 5  # feasible
expected = 4 * y[0] + 12 * y[1] + 18 * y[2]
assert expected == 36 and abs(found.fun - expected) < 1e-9
```

## A linear program in general form

**Level 4** · `opt.linear.general-form` · Boyd and Vandenberghe (2004), section 4.3

The display of the book: "minimize" and "subject to" in an array, with \(\preceq\) for
the componentwise inequality between vectors. Here \(x \succeq 0\), \(x_2 \leq 3\) and
\(x_1 + x_2 + x_3 = 4\); the optimum is \(-7\), at \((1, 3, 0)\).

```latex
c := \begin{pmatrix} -1 \\ -2 \\ 0 \end{pmatrix}
G := \begin{pmatrix} -1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & -1 \\ 0 & 1 & 0 \end{pmatrix}
h := \begin{pmatrix} 0 \\ 0 \\ 0 \\ 3 \end{pmatrix}
A := \begin{pmatrix} 1 & 1 & 1 \end{pmatrix}
b := 4
\begin{array}{ll} \text{minimize} & c^T x \\ \text{subject to} & G x \preceq h \\ & A x = b \end{array}
```

```python
from scipy.optimize import linprog

G = [[-1, 0, 0], [0, -1, 0], [0, 0, -1], [0, 1, 0]]
found = linprog(
    [-1, -2, 0], A_ub=G, b_ub=[0, 0, 0, 3], A_eq=[[1, 1, 1]], b_eq=[4], bounds=[(None, None)] * 3
)
assert np.allclose(found.x, [1, 3, 0])
expected = -1 * 1 - 2 * 3 + 0 * 0
assert expected == -7 and abs(found.fun - expected) < 1e-9
```

## A knapsack

**Level 4** · `opt.integer.knapsack` · Cormen, Leiserson, Rivest and Stein (2009), section 16.2

Items worth 60, 100 and 120 dollars, weighing 10, 20 and 30 pounds, and a knapsack that
holds 50 pounds. Each item is taken whole or not at all, \(x \in \{0, 1\}^3\): the best
load is the second and third items, worth 220, while taking a fraction of an item, as
the linear relaxation may, would give 240.

```latex
\max_{x \in \{0, 1\}^3} \; 60x_1 + 100x_2 + 120x_3 \quad \text{subject to} \quad 10x_1 + 20x_2 + 30x_3 \leq 50
```

```python
from itertools import product

from scipy.optimize import linprog

loads = [x for x in product([0, 1], repeat=3) if 10 * x[0] + 20 * x[1] + 30 * x[2] <= 50]
expected = max(60 * x[0] + 100 * x[1] + 120 * x[2] for x in loads)
assert expected == 220
relaxed = linprog([-60, -100, -120], A_ub=[[10, 20, 30]], b_ub=[50], bounds=[(0, 1)] * 3)
assert abs(-relaxed.fun - 240) < 1e-9  # the fractional knapsack is worth more
```

## The transport problem

**Level 5** · `opt.transport.wasserstein` · Villani (2003), chapter 1

Kantorovich's problem between two distributions on the line: the least mean distance
over the couplings \(\gamma \in \Pi(\mu, \nu)\), the joint distributions whose marginals
are \(\mu\) and \(\nu\). It is the Wasserstein distance \(W_1(\mu, \nu) = \frac{3}{4}\). Point
masses, couplings and an infimum over a set of measures are beyond any design of the
language yet.

```latex
\mu := \frac{1}{4}\delta_0 + \frac{1}{2}\delta_1 + \frac{1}{4}\delta_3
\nu := \frac{1}{2}\delta_1 + \frac{1}{2}\delta_2
\inf_{\gamma \in \Pi(\mu, \nu)} \int |x - y| \, d\gamma(x, y)
```

```python
from scipy.optimize import linprog
from scipy.stats import wasserstein_distance

xs, mu = [0, 1, 3], [0.25, 0.5, 0.25]
ys, nu = [1, 2], [0.5, 0.5]
cost = [abs(a - b) for a in xs for b in ys]  # the coupling, row by row
rows = [[1 if i == r else 0 for i in range(3) for _ in range(2)] for r in range(3)]
columns = [[1 if j == c else 0 for _ in range(3) for j in range(2)] for c in range(2)]
found = linprog(cost, A_eq=rows + columns, b_eq=mu + nu, bounds=[(0, None)] * 6)
expected = Rational(3, 4)
assert abs(found.fun - float(expected)) < 1e-9
assert abs(wasserstein_distance(xs, ys, mu, nu) - float(expected)) < 1e-12
```
