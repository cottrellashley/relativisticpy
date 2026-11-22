# Quantum information

Qubits as Nielsen and Chuang write them: systems of several qubits are tensor products,
mixed states are density matrices, a subsystem is described by a partial trace, and
the von Neumann entropy measures how mixed a state is.

## The tensor product of two Pauli matrices

**Level 1** · `qm.information.tensor-product` · Nielsen and Chuang (2010), section 2.1.7

\(X \otimes Y\), the Kronecker product of two \(2\times2\) matrices: a \(4\times4\)
matrix of blocks, not their matrix product.

```latex
\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \otimes \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}
```

```python
from sympy.physics.quantum import TensorProduct

X = Matrix([[0, 1], [1, 0]])
Y = Matrix([[0, -I], [I, 0]])
expected = TensorProduct(X, Y)
assert expected == Matrix([[0, 0, 0, -I], [0, 0, I, 0], [0, -I, 0, 0], [I, 0, 0, 0]])
```

## The purity of a qubit

**Level 2** · `qm.information.purity` · Nielsen and Chuang (2010), Exercise 2.72

A qubit's density matrix in terms of its Bloch vector, \(\rho = (I + \vec r\cdot\vec\sigma)/2\):
\(\operatorname{tr}\rho^2 = (1 + |\vec r|^2)/2\), which is one only for a pure state, on the
surface of the Bloch sphere.

```latex
I := \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}
X := \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
Y := \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}
Z := \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
\rho := \frac{1}{2}\left(I + r_x X + r_y Y + r_z Z\right)
\operatorname{tr}(\rho^2)
```

```python
r_x, r_y, r_z = symbols("r_x r_y r_z", real=True)
X = Matrix([[0, 1], [1, 0]])
Y = Matrix([[0, -I], [I, 0]])
Z = Matrix([[1, 0], [0, -1]])
rho = (eye(2) + r_x * X + r_y * Y + r_z * Z) / 2
assert rho.trace() == 1 and rho.H == rho
expected = expand((rho**2).trace())
assert simplify(expected - (1 + r_x**2 + r_y**2 + r_z**2) / 2) == 0
```

## The von Neumann entropy

**Level 2** · `qm.information.entropy` · Nielsen and Chuang (2010), section 11.3

\(S(\rho) = -\operatorname{tr}(\rho\log_2\rho)\), with the logarithm of a matrix that is not
diagonal: for eigenvalues \(3/4\) and \(1/4\), \(S = 2 - \tfrac{3}{4}\log_2 3 \approx 0.811\)
bits.

```latex
\rho := \begin{pmatrix} \frac{1}{2} & \frac{1}{4} \\ \frac{1}{4} & \frac{1}{2} \end{pmatrix}
-\operatorname{tr}(\rho \log_2 \rho)
```

```python
rho = Matrix([[Rational(1, 2), Rational(1, 4)], [Rational(1, 4), Rational(1, 2)]])
P, D = rho.diagonalize()
log2_rho = P * diag(*[log(p, 2) for p in D.diagonal()]) * P.inv()
computed = simplify(-(rho * log2_rho).trace())
# The entropy of the eigenvalues, as a probability distribution.
assert simplify(computed + sum(n * p * log(p, 2) for p, n in rho.eigenvals().items())) == 0
stated = 2 - Rational(3, 4) * log(3, 2)
assert simplify(computed - stated) == 0
expected = stated
```

## The reduced state of a Bell pair

**Level 4** · `qm.information.partial-trace` · Nielsen and Chuang (2010), section 2.4.3

The Bell state \((|00\rangle + |11\rangle)/\sqrt2\), with \(|00\rangle\) the tensor product
\(|0\rangle|0\rangle\), as a density operator: tracing out the second qubit leaves the
maximally mixed state \(I/2\). The pair is entangled.

```latex
|0\rangle := \begin{pmatrix} 1 \\ 0 \end{pmatrix}
|1\rangle := \begin{pmatrix} 0 \\ 1 \end{pmatrix}
|\Phi^+\rangle := \frac{|00\rangle + |11\rangle}{\sqrt{2}}
\rho^{AB} := |\Phi^+\rangle\langle\Phi^+|
\operatorname{tr}_B(\rho^{AB})
```

```python
from sympy.physics.quantum import TensorProduct

ket_0, ket_1 = Matrix([1, 0]), Matrix([0, 1])
phi = (TensorProduct(ket_0, ket_0) + TensorProduct(ket_1, ket_1)) / sqrt(2)
rho = phi * phi.H
assert rho**2 == rho  # the pair is in a pure state
# tr_B: the sum over the second qubit's basis of <j|_B rho |j>_B.
expected = zeros(2, 2)
for ket_j in (ket_0, ket_1):
    embed = TensorProduct(eye(2), ket_j)  # |i> -> |i>|j>
    expected += embed.H * rho * embed
assert expected == eye(2) / 2
assert (expected**2).trace() == Rational(1, 2)  # its half is maximally mixed
```
