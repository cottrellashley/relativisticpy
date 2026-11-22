# The Dirac equation and the gamma matrices

The Dirac field, in natural units with the metric \((+, -, -, -)\). The gamma matrices
satisfy \(\{\gamma^\mu, \gamma^\nu\} = 2g^{\mu\nu}\), are written in a representation as
blocks of Pauli matrices or used without one, through their algebra, their traces and
Feynman's slash, which contracts them with a four-vector. A plane-wave spinor solves
\((\gamma^\mu p_\mu - m)u(p) = 0\).

## A plane-wave solution

**Level 2** · `qft.dirac.plane-wave` · Griffiths (2008), section 7.2

The positive-energy spinor for a momentum \(p\) along \(z\), in the Dirac representation:
\(u = (1, 0, p/(E + m), 0)\) solves \((\gamma^\mu p_\mu - m)u = 0\) when
\(E = \sqrt{p^2 + m^2}\). Matrices named with superscripts, and a number standing for
that multiple of the identity.

```latex
\gamma^0 := \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & -1 \end{pmatrix}
\gamma^3 := \begin{pmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \\ -1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{pmatrix}
E := \sqrt{p^2 + m^2}
u := \begin{pmatrix} 1 \\ 0 \\ \frac{p}{E + m} \\ 0 \end{pmatrix}
\left(\gamma^0 E - \gamma^3 p - m\right) u = 0
```

```python
p, m = symbols("p m", positive=True)
E = sqrt(p**2 + m**2)
sigma_z = Matrix([[1, 0], [0, -1]])
gamma_0 = diag(1, 1, -1, -1)
gamma_3 = Matrix(BlockMatrix([[zeros(2, 2), sigma_z], [-sigma_z, zeros(2, 2)]]))
assert gamma_3 == Matrix([[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]])
u = Matrix([1, 0, p / (E + m), 0])  # Griffiths (2008), u^(1) with p_x = p_y = 0
assert simplify((gamma_0 * E - gamma_3 * p - m * eye(4)) * u) == zeros(4, 1)
expected = True
```

## The Clifford algebra

**Level 4** · `qft.dirac.clifford` · Peskin and Schroeder (1995), eqs. (3.22) and (3.25)

The Weyl representation, written as Peskin and Schroeder write it, in \(2\times2\) blocks
of Pauli matrices, satisfies the defining relation of the gamma matrices, an equation
with two free spacetime indices.

```latex
\gamma^0 := \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
\gamma^i := \begin{pmatrix} 0 & \sigma^i \\ -\sigma^i & 0 \end{pmatrix}
\{\gamma^\mu, \gamma^\nu\} = 2g^{\mu\nu}
```

```python
sigma = [Matrix([[0, 1], [1, 0]]), Matrix([[0, -I], [I, 0]]), Matrix([[1, 0], [0, -1]])]
one, zero = eye(2), zeros(2, 2)
gamma_matrices = [Matrix(BlockMatrix([[zero, one], [one, zero]]))]
gamma_matrices += [Matrix(BlockMatrix([[zero, s], [-s, zero]])) for s in sigma]
g = diag(1, -1, -1, -1)
for mu in range(4):
    for nu in range(4):
        anticommutator = gamma_matrices[mu] * gamma_matrices[nu] + gamma_matrices[nu] * gamma_matrices[mu]
        assert anticommutator == 2 * g[mu, nu] * eye(4)
expected = True
```

## A trace of four gamma matrices

**Level 4** · `qft.dirac.trace` · Peskin and Schroeder (1995), section 5.1

The trace theorem for four gamma matrices, an identity with four free indices that
follows from their algebra alone, with no representation written down.

```latex
\operatorname{tr}\left(\gamma^\mu\gamma^\nu\gamma^\rho\gamma^\sigma\right) = 4\left(g^{\mu\nu}g^{\rho\sigma} - g^{\mu\rho}g^{\nu\sigma} + g^{\mu\sigma}g^{\nu\rho}\right)
```

```python
import itertools

# Any representation will do: the Weyl representation.
sigma = [Matrix([[0, 1], [1, 0]]), Matrix([[0, -I], [I, 0]]), Matrix([[1, 0], [0, -1]])]
one, zero = eye(2), zeros(2, 2)
gamma_matrices = [Matrix(BlockMatrix([[zero, one], [one, zero]]))]
gamma_matrices += [Matrix(BlockMatrix([[zero, s], [-s, zero]])) for s in sigma]
g = diag(1, -1, -1, -1)
for mu, nu, rho, sig in itertools.product(range(4), repeat=4):
    product = gamma_matrices[mu] * gamma_matrices[nu] * gamma_matrices[rho] * gamma_matrices[sig]
    stated = 4 * (g[mu, nu] * g[rho, sig] - g[mu, rho] * g[nu, sig] + g[mu, sig] * g[nu, rho])
    assert product.trace() == stated
expected = True
```

## A slashed momentum squared

**Level 4** · `qft.dirac.slash` · standard

Feynman's slash, written with the `\slashed` of the slashed package, contracts the gamma
matrices with a momentum, and \(p^2 = p^\mu p_\mu\) is the square of a four-vector rather
than a power: the slashed momentum squared is \(p^2\).

```latex
\slashed{p} := \gamma^\mu p_\mu
\slashed{p}\slashed{p} = p^2
```

```python
sigma = [Matrix([[0, 1], [1, 0]]), Matrix([[0, -I], [I, 0]]), Matrix([[1, 0], [0, -1]])]
one, zero = eye(2), zeros(2, 2)
gamma_matrices = [Matrix(BlockMatrix([[zero, one], [one, zero]]))]
gamma_matrices += [Matrix(BlockMatrix([[zero, s], [-s, zero]])) for s in sigma]
g = diag(1, -1, -1, -1)
p_up = Matrix(symbols("p0:4", real=True))  # p^mu
p_down = g * p_up  # p_mu
p_slash = sum((gamma_matrices[mu] * p_down[mu] for mu in range(4)), zeros(4, 4))
p_squared = (p_up.T * g * p_up)[0]
assert expand(p_slash * p_slash) == expand(p_squared * eye(4))
expected = True
```
