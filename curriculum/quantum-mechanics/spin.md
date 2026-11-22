# Spin one-half

The Pauli matrices, their algebra, and expectation values, in the notations of
quantum mechanics.

## A commutator of Pauli matrices

**Level 2** · `qm.spin.pauli-commutator` · Sakurai (2017), eq. (3.2.33)

\([\sigma_x, \sigma_y] = 2 i \sigma_z\), with the matrices written out.

```latex
\sigma_x := \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
\sigma_y := \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}
\sigma_z := \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
\sigma_x \sigma_y - \sigma_y \sigma_x = 2 i \sigma_z
```

```python
expected = True
```

## An expectation value in Dirac notation

**Level 4** · `qm.spin.bra-ket-expectation` · Sakurai (2017), section 1.4

The expectation of \(\sigma_z\) in the state \(|+\rangle\), written with a bra and a ket.

```latex
|+\rangle := \begin{pmatrix} 1 \\ 0 \end{pmatrix}
\sigma_z := \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
\langle + | \sigma_z | + \rangle
```

```python
expected = 1
```

## A rotation of a spinor

**Level 2** · `qm.spin.rotation` · Sakurai (2017), section 3.2

The rotation operator \(\exp(-i\theta\sigma_y/2)\), the exponential of a matrix: about the
\(y\)-axis it is \(\cos\frac{\theta}{2} - i\sigma_y \sin\frac{\theta}{2}\), a real rotation by
half the angle.

```latex
\sigma_y := \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}
\exp\left(-\frac{i\theta}{2} \sigma_y\right)
```

```python
theta = symbols("theta", real=True)
sigma_y = Matrix([[0, -I], [I, 0]])
computed = simplify((-I * theta / 2 * sigma_y).exp())
stated = cos(theta / 2) * eye(2) - I * sigma_y * sin(theta / 2)  # Sakurai, section 3.2
assert simplify(computed - stated) == zeros(2, 2)
assert stated.subs(theta, 2 * pi) == -eye(2)  # a rotation by 2 pi changes the sign of a spinor
expected = stated
```

## Spin one

**Level 2** · `qm.spin.spin-one` · Sakurai (2017), section 3.5

The spin matrices of spin one, written out:
\(S^2 = S_x^2 + S_y^2 + S_z^2 = s(s+1)\hbar^2 = 2\hbar^2\) times the identity.

```latex
S_x := \frac{\hbar}{\sqrt{2}} \begin{pmatrix} 0 & 1 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 0 \end{pmatrix}
S_y := \frac{\hbar}{\sqrt{2}} \begin{pmatrix} 0 & -i & 0 \\ i & 0 & -i \\ 0 & i & 0 \end{pmatrix}
S_z := \hbar \begin{pmatrix} 1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & -1 \end{pmatrix}
S_x^2 + S_y^2 + S_z^2 = 2\hbar^2 \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}
```

```python
hbar = symbols("hbar", positive=True)
s, ms = 1, [1, 0, -1]


def raising(r, c):  # <m + 1|S_+|m> = hbar sqrt(s(s + 1) - m(m + 1)), basis m = 1, 0, -1
    return hbar * sqrt(s * (s + 1) - ms[c] * (ms[c] + 1)) if ms[r] == ms[c] + 1 else 0


S_plus = Matrix(3, 3, raising)
S_minus = S_plus.T
S_x = (S_plus + S_minus) / 2
S_y = (S_plus - S_minus) / (2 * I)
S_z = hbar * diag(*ms)
assert S_x == hbar / sqrt(2) * Matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
assert simplify(S_y - hbar / sqrt(2) * Matrix([[0, -I, 0], [I, 0, -I], [0, I, 0]])) == zeros(3, 3)
assert simplify(S_x * S_y - S_y * S_x - I * hbar * S_z) == zeros(3, 3)
assert simplify(S_x**2 + S_y**2 + S_z**2) == s * (s + 1) * hbar**2 * eye(3)
expected = True
```
