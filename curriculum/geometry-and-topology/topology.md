# Topology

What does not change when a space is bent: the Euler characteristic of a triangulated
surface, the linking number of two closed curves by Gauss's integral, the homology of the
Klein bottle, and the classification of closed surfaces by orientability and Euler
characteristic. Hatcher (2002) is *Algebraic Topology*; Massey (1967) is *Algebraic
Topology: An Introduction*.

## A triangulated torus

**Level 1** · `geom.topology.torus-triangulation` · standard

The smallest triangulation of the torus has 7 vertices, 21 edges and 14 triangles: its Euler
characteristic, \(V - E + F\), is 0.

```latex
\left. V - E + F \right|_{V = 7, E = 21, F = 14}
```

```python
from itertools import combinations

expected = 7 - 21 + 14
# The seven-vertex torus: the triangles {i, i+1, i+3} and {i, i+2, i+3}, modulo 7.
faces = {frozenset({i, (i + 1) % 7, (i + 3) % 7}) for i in range(7)}
faces |= {frozenset({i, (i + 2) % 7, (i + 3) % 7}) for i in range(7)}
edges = {frozenset(pair) for face in faces for pair in combinations(sorted(face), 2)}
assert (len(faces), len(edges)) == (14, 21)
assert all(sum(edge <= face for face in faces) == 2 for edge in edges)  # a closed surface
# Its Betti numbers are the torus's, 1, 2 and 1, from the ranks of the boundary maps.
edges, faces = sorted(sorted(e) for e in edges), sorted(sorted(f) for f in faces)
index = {tuple(e): j for j, e in enumerate(edges)}
d1 = Matrix(7, 21, lambda i, j: 1 if i == edges[j][1] else -1 if i == edges[j][0] else 0)
d2 = zeros(21, 14)
for j, (p, q, r) in enumerate(faces):  # the boundary of [p, q, r] is [q, r] - [p, r] + [p, q]
    d2[index[(q, r)], j], d2[index[(p, r)], j], d2[index[(p, q)], j] = 1, -1, 1
assert d1 * d2 == zeros(7, 14)
betti = (7 - d1.rank(), 21 - d1.rank() - d2.rank(), 14 - d2.rank())
assert betti == (1, 2, 1)
assert betti[0] - betti[1] + betti[2] == expected == 0
```

## The linking number of the Hopf link

**Level 4** · `geom.topology.hopf-link` · Gauss (1833)

Two unit circles, each through the centre of the other: Gauss's double integral over the
two closed curves, with their differentials as vectors, counts how often one winds around
the other, \(-1\) with these orientations. The value is an integer, but the integrand has no
elementary antiderivative. Gauss's own note takes the difference of the two points the
other way round, which changes the sign; the integrand here, with
\(\mathbf{r}_1 - \mathbf{r}_2\), is the form in use today.

```latex
\mathbf{r}_1(s) := (\cos s, \sin s, 0), \quad 0 \le s \le 2\pi
\mathbf{r}_2(t) := (1 + \cos t, 0, \sin t), \quad 0 \le t \le 2\pi
\frac{1}{4\pi} \oint_{\mathbf{r}_1} \oint_{\mathbf{r}_2} \frac{(\mathbf{r}_1 - \mathbf{r}_2) \cdot (d\mathbf{r}_1 \times d\mathbf{r}_2)}{\left| \mathbf{r}_1 - \mathbf{r}_2 \right|^3}
```

```python
from scipy.integrate import dblquad


def integrand(t, s):
    r1, r2 = np.array([np.cos(s), np.sin(s), 0.0]), np.array([1 + np.cos(t), 0.0, np.sin(t)])
    dr1, dr2 = np.array([-np.sin(s), np.cos(s), 0.0]), np.array([-np.sin(t), 0.0, np.cos(t)])
    return (r1 - r2).dot(np.cross(dr1, dr2)) / np.linalg.norm(r1 - r2) ** 3


value = dblquad(integrand, 0, 2 * np.pi, 0, 2 * np.pi, epsabs=1e-11, epsrel=1e-11)[0]
assert abs(value / (4 * np.pi) + 1) < 1e-9
expected = -1
tolerance = 1e-6
# The second circle crosses the plane z = 0 at (2, 0, 0), outside the unit disk, and at the
# origin, inside it: it passes once through the disk that the first circle bounds.
# By Ampere's law the integral is the circulation of the second circle about a current along
# the first, by the right-hand rule. The second turns about (r_2 - c) x r_2' = (0, -1, 0) at
# t = 0, while the first runs through its centre c = (1, 0, 0) along +y: linked negatively.
axis = np.cross(np.array([2.0, 0.0, 0.0]) - np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
assert axis.dot(np.array([0.0, 1.0, 0.0])) < 0
```

## The homology of the Klein bottle

**Level 5** · `geom.topology.klein-bottle` · Hatcher (2002), sections 1.2 and 2.A

The fundamental group of the Klein bottle by a presentation, and the claim that its first
homology group, the fundamental group made abelian, is \(\mathbb{Z} \oplus \mathbb{Z}_2\):
groups as values, given by generators and relations, and isomorphism between them. No design
of the language covers them yet.

```latex
\pi_1(K) = \langle a, b \mid a b a^{-1} b \rangle
H_1(K) \cong \mathbb{Z} \oplus \mathbb{Z}_2
```

```python
from sympy.matrices.normalforms import smith_normal_form

# H_1 is Z^2, on a and b, modulo the exponent sums of the relator a b a^-1 b, which are (0, 2).
assert smith_normal_form(Matrix([[0, 2]]), domain=ZZ) == Matrix([[2, 0]])  # Z_2 and a free Z
# Again from a Delta-complex: one vertex, edges a, b, c (all loops, so the boundary map on edges
# is 0), and two triangles with boundaries a + b - c and a - b - c, the vertical sides of the
# square glued in opposite directions: H_1 = Z^3 / (Z + 2Z) = Z + Z_2, and H_2 = 0.
assert smith_normal_form(Matrix([[1, 1], [1, -1], [-1, -1]]), domain=ZZ) == Matrix(
    [[1, 0], [0, 2], [0, 0]]
)
# The torus, glued the same way round on both sides, gives Z^2 instead.
assert smith_normal_form(Matrix([[1, 1], [1, 1], [-1, -1]]), domain=ZZ) == Matrix(
    [[1, 0], [0, 0], [0, 0]]
)
expected = True
```

## Dyck's theorem

**Level 5** · `geom.topology.dyck` · Massey (1967), chapter 1

The connected sum of a torus and a projective plane is the connected sum of three
projective planes: a claim of homeomorphism between surfaces named in the notation of
topology, decided by the classification of closed surfaces, since both are non-orientable
with Euler characteristic \(-1\). Spaces as values are not designed yet.

```latex
T^2 \# \mathbb{RP}^2 \cong \mathbb{RP}^2 \# \mathbb{RP}^2 \# \mathbb{RP}^2
```

```python
from sympy.matrices.normalforms import smith_normal_form


def chi(*summands):  # chi(M # N) = chi(M) + chi(N) - 2
    return sum(summands) - 2 * (len(summands) - 1)


torus, projective_plane = 0, 1
assert chi(torus, projective_plane) == chi(projective_plane, projective_plane, projective_plane)
assert chi(torus, projective_plane) == -1
# Both are non-orientable, having a projective plane as a summand, and a closed surface is
# determined by its orientability and its Euler characteristic. Again by H_1, from the
# polygons a b a^-1 b^-1 c c and a a b b c c: the exponent sums of their one relator.
left = smith_normal_form(Matrix([[0, 0, 2]]), domain=ZZ)
right = smith_normal_form(Matrix([[2, 2, 2]]), domain=ZZ)
assert left == right == Matrix([[2, 0, 0]])  # Z^2 + Z_2 for both
expected = True
```
