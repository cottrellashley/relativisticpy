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
