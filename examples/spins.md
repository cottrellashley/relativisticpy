# Two spins

Quantum mechanics in index notation. The state of a spin is a vector in a complex
space of two dimensions, $\mathcal{H} = \mathbb{C}^2$, and an operator on it is a
tensor with one index up and one down:

```bash
relativisticpy examples/spins.md --render
```

$$
\begin{aligned}
\mathcal{H} &:= \mathbb{C}^{2} \\
i, j, k, l &\in \mathcal{H}
\end{aligned}
$$

The Pauli matrices:

$$
\begin{aligned}
X^{i}{}_{j} &:= [[0, 1], [1, 0]] \\
Y^{i}{}_{j} &:= [[0, -i], [i, 0]] \\
Z^{i}{}_{j} &:= [[1, 0], [0, -1]]
\end{aligned}
$$

Two spins with a Heisenberg coupling $J$: an operator on $\mathcal{H} \otimes
\mathcal{H}$ is simply a tensor with two indices of each kind,
$H = J\, \vec\sigma_1 \cdot \vec\sigma_2$:

$$
H^{ik}{}_{jl} := J \left( X^i{}_j X^k{}_l + Y^i{}_j Y^k{}_l + Z^i{}_j Z^k{}_l \right)
$$

Its spectrum, exactly: the singlet at $-3J$ and the triplet at $J$, three times.

$$
\operatorname{spec}(H^{ik}{}_{jl})
$$

The trace of $H$ vanishes, as it must for a sum of traceless products:

$$
\operatorname{tr}(H^{ik}{}_{jl})
$$

A single spin in a magnetic field precesses: the operator of its evolution over a time
$t$ is the exponential of the map $Z$, written without indices as a matrix is.

$$
U := \exp\left(-\frac{i \omega t}{2} Z\right)
$$

$$
U^{i}{}_{j}
$$
