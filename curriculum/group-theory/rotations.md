# Rotations

The rotation group SO(3) and its Lie algebra. The generators are written as real
antisymmetric matrices, \((J_i)_{jk} = -\epsilon_{ijk}\), so that
\([J_i, J_j] = \epsilon_{ijk} J_k\) and \(e^{\theta J_z}\) turns the \(xy\)-plane through
\(\theta\): the exponential map and its inverse, commutators, a reducible representation
as a direct sum, how the generators themselves transform, and the product of two vector
representations.

## A rotation from its generator

**Level 1** · `group.rotations.exponential` · standard

The exponential map, written as a book writes it, about an axis that is not a coordinate
axis: the generator about the unit vector \(\mathbf{n} = (1, 1, 1)/\sqrt{3}\) is
\(\mathbf{n} \cdot \mathbf{J}\), and the exponential of \(2\pi/3\) times it, a third of a turn
about the diagonal, permutes the axes, \(x \to y \to z \to x\).

```latex
\exp\left(\frac{2\pi}{3\sqrt{3}} \begin{pmatrix} 0 & -1 & 1 \\ 1 & 0 & -1 \\ -1 & 1 & 0 \end{pmatrix}\right)
```

```python
from scipy.linalg import expm

J = [Matrix(3, 3, lambda j, k: -LeviCivita(i, j, k)) for i in range(3)]  # (J_i)_jk = -eps_ijk
n = Matrix([1, 1, 1]) / sqrt(3)
generator = sum((n[i] * J[i] for i in range(3)), zeros(3, 3))
assert sqrt(3) * generator == Matrix([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
expected = Matrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]])  # x -> y -> z -> x
exponential = (2 * pi / 3 * generator).exp()  # in roots of -1, which expand to it
assert (exponential - expected).applyfunc(expand_complex) == zeros(3, 3)
# Rodrigues's formula for a unit axis: exp(t K) = 1 + sin(t) K + (1 - cos(t)) K^2
angle = 2 * pi / 3
rodrigues = eye(3) + sin(angle) * generator + (1 - cos(angle)) * generator**2
assert simplify(rodrigues - expected) == zeros(3, 3)
assert expected * Matrix([1, 0, 0]) == Matrix([0, 1, 0])  # x goes to y
written = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
numeric = expm(2 * np.pi / (3 * np.sqrt(3)) * written)  # and by SciPy
assert np.allclose(numeric, np.array(expected, dtype=float))
```

## The commutator of two generators

**Level 1** · `group.rotations.commutator` · standard

The commutator bracket \([A, B] = AB - BA\) of two matrices written out: the generators
of rotations about \(x\) and about \(y\) commute to the generator about \(z\),
\([J_x, J_y] = J_z\).

```latex
\left[ \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & -1 \\ 0 & 1 & 0 \end{pmatrix}, \begin{pmatrix} 0 & 0 & 1 \\ 0 & 0 & 0 \\ -1 & 0 & 0 \end{pmatrix} \right]
```

```python
J = [Matrix(3, 3, lambda j, k: -LeviCivita(i, j, k)) for i in range(3)]  # (J_i)_jk = -eps_ijk
A = Matrix([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
B = Matrix([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
assert (A, B) == (J[0], J[1])
expected = A * B - B * A
assert expected == J[2] == Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
```

## A reducible representation

**Level 1** · `group.rotations.direct-sum` · standard

The direct sum of two representations is block diagonal: the rotation of the plane
\(\oplus\) the trivial representation is the rotation of space about the \(z\)-axis.
\(\oplus\) between matrices.

```latex
\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \oplus \begin{pmatrix} 1 \end{pmatrix}
```

```python
theta = symbols("theta", real=True)
expected = diag(Matrix([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]), Matrix([[1]]))
assert expected.shape == (3, 3)
J_z = Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
assert simplify((theta * J_z).exp() - expected) == zeros(3, 3)  # the rotation about z
```

## The generators transform as a vector

**Level 2** · `group.rotations.adjoint-action` · standard

A quarter turn about the \(y\)-axis takes the \(z\)-axis to the \(x\)-axis, and conjugating
by it takes the generator about \(z\) to the generator about \(x\):
\(R J_z R^{-1} = J_x\). The adjoint action of SO(3) on its generators is the vector
representation.

```latex
J_x := \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & -1 \\ 0 & 1 & 0 \end{pmatrix}
J_z := \begin{pmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}
R := \begin{pmatrix} 0 & 0 & 1 \\ 0 & 1 & 0 \\ -1 & 0 & 0 \end{pmatrix}
R J_z R^{-1} = J_x
```

```python
J = [Matrix(3, 3, lambda j, k: -LeviCivita(i, j, k)) for i in range(3)]
R = (pi / 2 * J[1]).exp()  # a quarter turn about y
assert R == Matrix([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
assert R * Matrix([0, 0, 1]) == Matrix([1, 0, 0])  # z goes to x
assert R * J[2] * R.inv() == J[0]
# In general R (n.J) R^-1 = (Rn).J: the generator about the rotated axis.
n = Matrix(symbols("n_1:4"))
Rn = R * n
generator = lambda v: sum((v[i] * J[i] for i in range(3)), zeros(3, 3))
assert expand(R * generator(n) * R.inv() - generator(Rn)) == zeros(3, 3)
expected = True
```

## A one-parameter subgroup and its generator

**Level 3** · `group.rotations.one-parameter-subgroup` · Hall (2015), chapter 2

The generator is the derivative of the rotations at the identity, and the exponential map
gives them back: \(J = dR/d\theta\) at \(\theta = 0\), then \(e^{\theta J} = R(\theta)\)
for every \(\theta\). A function whose values are matrices, differentiated and
exponentiated. This is the mathematicians' convention, as in Hall (2015), *Lie Groups, Lie
Algebras, and Representations*, 2nd edition; physicists such as Georgi (1999) take the
Hermitian generator \(-iJ\) instead.

```latex
R(\theta) := \begin{pmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{pmatrix}
J := \left. \frac{d}{d\theta} R(\theta) \right|_{\theta = 0}
e^{\theta J} = R(\theta)
```

```python
from scipy.linalg import expm

theta = symbols("theta", real=True)
R = Matrix([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])
J = R.diff(theta).subs(theta, 0)
assert J == Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
assert simplify((theta * J).exp() - R) == zeros(3, 3)
numeric = expm(1.3 * np.array(J, dtype=float))  # and by SciPy, at theta = 1.3
assert np.allclose(numeric, np.array(R.subs(theta, 1.3), dtype=float))
expected = True
```

## Two vectors make a scalar, a vector and a symmetric tensor

**Level 3** · `group.rotations.tensor-square` · standard

The character of a tensor product is the product of the characters. For two vectors,
\(\operatorname{tr}(R \otimes R) = (1 + 2\cos\theta)^2\) is the sum of the characters of
spin 0, 1 and 2: a tensor of rank two splits into its trace, its antisymmetric part and
its symmetric traceless part, \(1 \otimes 1 = 0 \oplus 1 \oplus 2\).

```latex
R := \begin{pmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{pmatrix}
\operatorname{tr}(R \otimes R) = 1 + (1 + 2\cos\theta) + (1 + 2\cos\theta + 2\cos 2\theta)
```

```python
from sympy.physics.quantum import TensorProduct

theta = symbols("theta", real=True)
R = Matrix([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])
left = TensorProduct(R, R).trace()
assert simplify(left - R.trace() ** 2) == 0
m = symbols("m", integer=True)
chi = lambda l: Sum(exp(I * m * theta), (m, -l, l)).doit()  # the character of spin l
written = [1, 1 + 2 * cos(theta), 1 + 2 * cos(theta) + 2 * cos(2 * theta)]
assert all(simplify((chi(l) - c).rewrite(exp)) == 0 for l, c in enumerate(written))
assert simplify(left - sum(written)) == 0
assert left.subs(theta, 0) == 9  # the dimensions: 9 = 1 + 3 + 5
expected = True
```
