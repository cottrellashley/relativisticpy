# Small oscillations and rigid bodies

The period of a pendulum, the normal modes of coupled oscillators, and the inertia
tensor of a rigid body: the angular momentum it gives, written with Cartesian tensors,
and its principal moments.

## The period of a pendulum

**Level 1** · `cm.oscillations.pendulum-period` · standard (Huygens, 1673)

The period of small oscillations of a pendulum one metre long, \(2\pi\sqrt{l/g}\), with
an evaluation bar: about two seconds.

```latex
2\pi \sqrt{\frac{l}{g}} \Big|_{l = 1,\ g = 9.81}
```

```python
expected = 2 * np.pi * np.sqrt(1 / 9.81)
assert abs(expected - 2.006) < 1e-3  # seconds
```

## A free index that does not match

**Level 1** · `cm.oscillations.mismatched-index` · Goldstein (2002), section 5.3

Cartesian tensors, their indices running over the three axes of space. The angular
momentum of a rigid body is \(L_i = I_{ij} \omega_j\); written with \(\omega_i\)
instead, the right side is free in \(j\) and the left in \(i\). The notation is wrong
and must be refused, not used to define \(L_i\).

```latex
\mathcal{E} := \mathbb{R}^{3}
i, j \in \mathcal{E}
I_{ij} := \frac{M a^2}{12} \begin{pmatrix} 8 & -3 & -3 \\ -3 & 8 & -3 \\ -3 & -3 & 8 \end{pmatrix}
\omega_j := (\Omega, 0, 0)
L_i = I_{ij} \omega_i
```

```python
error = "IndexNotationError"
message = "free indices"
```

## The angular momentum of a spinning cube

**Level 2** · `cm.oscillations.angular-momentum` · Taylor (2005), section 10.3

The same Cartesian tensors, contracted correctly: a cube spinning about an edge through
the corner it turns about has an angular momentum \(L_i = I_{ij} \omega_j\) that is not
parallel to its angular velocity.

```latex
\mathcal{E} := \mathbb{R}^{3}
i, j \in \mathcal{E}
I_{ij} := \frac{M a^2}{12} \begin{pmatrix} 8 & -3 & -3 \\ -3 & 8 & -3 \\ -3 & -3 & 8 \end{pmatrix}
\omega_j := (\Omega, 0, 0)
I_{ij} \omega_j
```

```python
M, a, Omega = symbols("M a Omega")
inertia = M * a**2 / 12 * Matrix([[8, -3, -3], [-3, 8, -3], [-3, -3, 8]])
expected = list(inertia * Matrix([Omega, 0, 0]))
assert expected == [2 * M * a**2 * Omega / 3, -M * a**2 * Omega / 4, -M * a**2 * Omega / 4]
```

## Two carts and three springs

**Level 2** · `cm.oscillations.two-carts` · Taylor (2005), section 11.2

Two equal masses joined by three equal springs. The normal frequencies are the positive
roots of \(\det(K - \omega^2 M) = 0\): \(\sqrt{k/m}\) and \(\sqrt{3k/m}\). The masses,
the spring constant and the frequency are positive, as the book says in words.

```latex
Positive := [k, m, \omega]
M := \begin{pmatrix} m & 0 \\ 0 & m \end{pmatrix}
K := \begin{pmatrix} 2k & -k \\ -k & 2k \end{pmatrix}
solve(\det(K - \omega^2 M) = 0, \omega)
```

```python
k, m, omega = symbols("k m omega", positive=True)
M = Matrix([[m, 0], [0, m]])
K = Matrix([[2 * k, -k], [-k, 2 * k]])
expected = solve(Eq((K - omega**2 * M).det(), 0), omega)
assert expected == [sqrt(k / m), sqrt(3 * k / m)]
```

## The principal moments of a cube about a corner

**Level 2** · `cm.oscillations.principal-moments` · Taylor (2005), section 10.5

The inertia tensor of a uniform cube of side \(a\) about one corner, and the
characteristic equation \(\det(\mathbf{I} - \lambda \mathbf{1}) = 0\), with
\(\mathbf{1}\) the unit matrix. The principal moments are \(M a^2 / 6\), about the
diagonal, and \(11 M a^2 / 12\), twice.

```latex
\mathbf{I} = \frac{M a^2}{12} \begin{pmatrix} 8 & -3 & -3 \\ -3 & 8 & -3 \\ -3 & -3 & 8 \end{pmatrix}
solve(\det(\mathbf{I} - \lambda \mathbf{1}) = 0, \lambda)
```

```python
M, a, lam = symbols("M a lambda")
x, y, z = symbols("x y z")
inertia = M * a**2 / 12 * Matrix([[8, -3, -3], [-3, 8, -3], [-3, -3, 8]])
# the tensor from its definition, I_jk = integral of rho (r^2 delta_jk - x_j x_k) over the cube
X = (x, y, z)
density = M / a**3


def element(j, k):
    integrand = density * ((x**2 + y**2 + z**2) * (1 if j == k else 0) - X[j] * X[k])
    return integrate(integrand, (x, 0, a), (y, 0, a), (z, 0, a))


assert simplify(Matrix(3, 3, element) - inertia) == zeros(3, 3)
expected = solve(Eq((inertia - lam * eye(3)).det(), 0), lam)
assert expected == [M * a**2 / 6, 11 * M * a**2 / 12]
```
