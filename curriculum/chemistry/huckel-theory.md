# Hückel theory

The π electrons of conjugated hydrocarbons in Hückel's approximation: each carbon
contributes one p orbital, with the Coulomb integral \(\alpha\) on the diagonal and the
resonance integral \(\beta < 0\) between bonded neighbours. The roots of the secular
determinant are the orbital energies; McQuarrie and Simon (1997), *Physical Chemistry: A
Molecular Approach*, write them in \(x = (\alpha - E)/\beta\).

## The secular determinant of ethylene

**Level 1** · `chem.huckel.ethylene` · McQuarrie and Simon (1997), chapter 10

Two p orbitals: the determinant
\(\begin{vmatrix} \alpha - E & \beta \\ \beta & \alpha - E \end{vmatrix}\), whose roots
are the orbital energies \(E = \alpha \pm \beta\).

```latex
\begin{vmatrix} \alpha - E & \beta \\ \beta & \alpha - E \end{vmatrix}
```

```python
alpha, beta = symbols("alpha beta")
E = symbols("E")  # an energy, not Euler's number
expected = (alpha - E) ** 2 - beta**2
assert Matrix([[alpha - E, beta], [beta, alpha - E]]).det() == expand(expected)
assert set(solve(expected, E)) == {alpha + beta, alpha - beta}
```

## The orbital energies of butadiene

**Level 2** · `chem.huckel.butadiene` · McQuarrie and Simon (1997), chapter 10

Four carbons in a chain: divided by \(\beta\), the secular determinant has \(x =
(\alpha - E)/\beta\) on its diagonal and 1 between neighbours. Its roots, \(x = \pm 1.618\)
and \(\pm 0.618\), give the orbital energies \(E = \alpha - x\beta\).

```latex
solve(\begin{vmatrix} x & 1 & 0 & 0 \\ 1 & x & 1 & 0 \\ 0 & 1 & x & 1 \\ 0 & 0 & 1 & x \end{vmatrix} = 0, x)
```

```python
x = symbols("x")
chain = Matrix(4, 4, lambda i, j: x if i == j else (1 if abs(i - j) == 1 else 0))
assert chain.det() == x**4 - 3 * x**2 + 1
expected = solve(chain.det(), x)  # the four roots, in the order SymPy gives them
golden = (1 + sqrt(5)) / 2
assert set(expected) == {golden, -golden, golden - 1, 1 - golden}  # 1.618 and 0.618
# the same numbers as the eigenvalues of the chain's adjacency matrix
adjacency = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
found = np.sort(np.linalg.eigvalsh(adjacency))
assert np.allclose(found, np.sort([float(root) for root in expected]))
```

## The delocalisation energy of benzene

**Level 3** · `chem.huckel.benzene-delocalisation` · Hückel (1931), Z. Phys. 70, 204

The orbital energies of a ring of six carbons are \(E_k = \alpha + 2\beta\cos(2\pi k/6)\).
The six π electrons fill \(k = 0\) and \(k = \pm 1\), so \(E_\pi = 6\alpha + 8\beta\);
three isolated double bonds would give \(6(\alpha + \beta)\), and the ring is lower in
energy by \(2|\beta|\): its delocalisation energy is \(2\beta\), negative as \(\beta\) is.
The energies are a family, \(E_k\).

```latex
E_k := \alpha + 2\beta \cos\frac{2\pi k}{6}
E_\pi := 2E_0 + 2E_1 + 2E_{-1}
E_\pi - 6(\alpha + \beta)
```

```python
alpha, beta = symbols("alpha beta")
E = lambda k: alpha + 2 * beta * cos(2 * pi * k / 6)
expected = simplify(2 * E(0) + 2 * E(1) + 2 * E(-1) - 6 * (alpha + beta))
assert expected == 2 * beta
# the same from the Hueckel matrix of the ring, in units of |beta| (alpha = 0, beta = -1)
ring = Matrix(6, 6, lambda i, j: -1 if (i - j) % 6 in (1, 5) else 0)
assert ring.eigenvals() == {-2: 1, -1: 2, 1: 2, 2: 1}
occupied = [-2, -1, -1]  # the three lowest orbitals, two electrons each
assert 2 * sum(occupied) - 6 * (0 + -1) == expected.subs(beta, -1)
```
