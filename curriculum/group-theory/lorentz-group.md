# The Lorentz group

The generators of the Lorentz group acting on four-vectors \((t, x, y, z)\): Jackson's real
matrices \(S_i\) for rotations and \(K_i\) for boosts, with
\([S_i, S_j] = \epsilon_{ijk} S_k\), \([S_i, K_j] = \epsilon_{ijk} K_k\) and
\([K_i, K_j] = -\epsilon_{ijk} S_k\); the split of the algebra into two commuting copies of
su(2); and the generators \(\mathcal{J}^{\mu\nu}\) in index notation, as Peskin and
Schroeder write them, with \(g_{\mu\nu} = \operatorname{diag}(1, -1, -1, -1)\).

## The commutator of two boosts

**Level 2** · `group.lorentz.boost-commutator` · Jackson (1999), section 11.7

The commutator bracket of two matrices defined by name: boosts along \(x\) and along \(y\)
do not commute, and their commutator is minus the generator of rotations about \(z\),
\([K_1, K_2] = -S_3\), the root of Thomas precession.

```latex
K_1 := \begin{pmatrix} 0 & 1 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}
K_2 := \begin{pmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}
S_3 := \begin{pmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}
[K_1, K_2] = -S_3
```

```python
K_1 = Matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
K_2 = Matrix([[0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
S_3 = Matrix([[0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
assert K_1 * K_2 - K_2 * K_1 == -S_3
eta = diag(1, -1, -1, -1)
assert all(X.T * eta + eta * X == zeros(4, 4) for X in (K_1, K_2, S_3))  # all in so(3, 1)
zeta, theta = symbols("zeta theta", real=True)
boost = Matrix(
    [[cosh(zeta), sinh(zeta), 0, 0], [sinh(zeta), cosh(zeta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
)
assert simplify((zeta * K_1).exp() - boost) == zeros(4, 4)  # K_1 generates boosts along x
rotation = diag(1, Matrix([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]), 1)
assert simplify((theta * S_3).exp() - rotation) == zeros(4, 4)  # S_3 rotates x into y
expected = True
```

## Two commuting copies of su(2)

**Level 3** · `group.lorentz.two-su2` · Weinberg (1995), chapter 5

Complex combinations of the generators, \(A_i = \frac12 (S_i + iK_i)\) and
\(B_i = \frac12 (S_i - iK_i)\), commute with each other, so the Lorentz algebra is two
copies of su(2), whose pairs of spins \((A, B)\) label its representations. One of the
nine commutators, \([A_1, B_2]\), in which the rotation and the boost about \(z\) cancel.

```latex
S_1 := \begin{pmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & -1 \\ 0 & 0 & 1 & 0 \end{pmatrix}
S_2 := \begin{pmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \end{pmatrix}
K_1 := \begin{pmatrix} 0 & 1 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}
K_2 := \begin{pmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}
A_1 := \frac{1}{2}\left(S_1 + i K_1\right)
B_2 := \frac{1}{2}\left(S_2 - i K_2\right)
[A_1, B_2] = 0
```

```python
from itertools import product

S = [
    Matrix([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]]),
    Matrix([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, -1, 0, 0]]),
    Matrix([[0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 0]]),
]
K = [
    Matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
    Matrix([[0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]),
    Matrix([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]),
]


def bracket(X, Y):
    return X * Y - Y * X


def combination(coefficients, basis):
    return sum((c * X for c, X in zip(coefficients, basis, strict=True)), zeros(4, 4))


for i, j in product(range(3), repeat=2):  # Jackson's algebra
    eps = [LeviCivita(i, j, k) for k in range(3)]
    assert bracket(S[i], S[j]) == combination(eps, S)
    assert bracket(S[i], K[j]) == combination(eps, K)
    assert bracket(K[i], K[j]) == -combination(eps, S)
A = [(S[i] + I * K[i]) / 2 for i in range(3)]
B = [(S[i] - I * K[i]) / 2 for i in range(3)]
for i, j in product(range(3), repeat=2):
    eps = [LeviCivita(i, j, k) for k in range(3)]
    assert bracket(A[i], B[j]) == zeros(4, 4)
    assert bracket(A[i], A[j]) == combination(eps, A)  # each is su(2)
    assert bracket(B[i], B[j]) == combination(eps, B)
assert bracket(S[0], S[1]) != zeros(4, 4)  # the parts commute, their pieces do not
expected = True
```

## The Lorentz algebra in index notation

**Level 4** · `group.lorentz.index-form` · Peskin and Schroeder (1995), section 3.1

The six generators as one antisymmetric pair of indices, each a matrix acting on vectors
\(V^\beta\): \((\mathcal{J}^{\mu\nu})^\alpha{}_\beta = i(g^{\mu\alpha}\delta^\nu_\beta -
g^{\nu\alpha}\delta^\mu_\beta)\), the vector representation (Peskin and Schroeder write it
with \(\alpha\) lowered). Their commutators, a tensor whose components are matrices, close
on the generators.

```latex
g_{\mu\nu} = \operatorname{diag}(1, -1, -1, -1)
(\mathcal{J}^{\mu\nu})^\alpha{}_\beta := i\left(g^{\mu\alpha} \delta^\nu_\beta - g^{\nu\alpha} \delta^\mu_\beta\right)
[\mathcal{J}^{\mu\nu}, \mathcal{J}^{\rho\sigma}] = i\left(g^{\nu\rho} \mathcal{J}^{\mu\sigma} - g^{\mu\rho} \mathcal{J}^{\nu\sigma} - g^{\nu\sigma} \mathcal{J}^{\mu\rho} + g^{\mu\sigma} \mathcal{J}^{\nu\rho}\right)
```

```python
from itertools import product

g = diag(1, -1, -1, -1)  # g^{mu nu} = g_{mu nu}
delta = eye(4)


def J(mu, nu):  # the matrix (J^{mu nu})^alpha_beta, alpha the row and beta the column
    return Matrix(4, 4, lambda al, be: I * (g[mu, al] * delta[nu, be] - g[nu, al] * delta[mu, be]))


for mu, nu, rho, sg in product(range(4), repeat=4):
    left = J(mu, nu) * J(rho, sg) - J(rho, sg) * J(mu, nu)
    right = I * (
        g[nu, rho] * J(mu, sg)
        - g[mu, rho] * J(nu, sg)
        - g[nu, sg] * J(mu, rho)
        + g[mu, sg] * J(nu, rho)
    )
    assert left == right
# They are the rotations and boosts: J^{12} = i S_3 and J^{01} = i K_1, so that
# Lambda = exp(-i omega_{mu nu} J^{mu nu} / 2) is exp(theta S_3) for
# omega_{12} = -omega_{21} = theta, and exp(zeta K_1) for omega_{01} = -omega_{10} = zeta.
S_3 = Matrix([[0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
K_1 = Matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
assert J(1, 2) == I * S_3 and J(0, 1) == I * K_1
zeta = symbols("zeta", real=True)
generator = -I / 2 * (zeta * J(0, 1) - zeta * J(1, 0))
assert simplify(generator.exp() - (zeta * K_1).exp()) == zeros(4, 4)
expected = True
```
