# Index notation

Einstein's summation convention over the three dimensions of Euclidean space, the
Kronecker delta and the Levi-Civita symbol, as books on Cartesian tensors write them:
every index down, which is allowed where the metric is the identity. Notation that breaks
the convention must be refused.

## The trace of the Kronecker delta

**Level 1** · `la.index.delta-trace` · Riley, Hobson and Bence (2006), section 26.8

A repeated index is summed over the range the book gives it: in three dimensions,
\(\delta_{ii} = 3\).

```latex
i, j = 1, 2, 3
\delta_{ii}
```

```python
expected = sum(KroneckerDelta(i, i) for i in range(1, 4))
assert expected == 3
```

## The epsilon–delta identity

**Level 2** · `la.index.epsilon-delta` · Riley, Hobson and Bence (2006), section 26.8

Two Levi-Civita symbols contracted over one index: an identity for every value of the
four free indices.

```latex
V := \mathbb{R}^{3}
i, j, k, m, n \in V
\epsilon_{ijk} \epsilon_{imn} = \delta_{jm} \delta_{kn} - \delta_{jn} \delta_{km}
```

```python
R = range(3)
assert all(
    sum(LeviCivita(i, j, k) * LeviCivita(i, m, n) for i in R)
    == KroneckerDelta(j, m) * KroneckerDelta(k, n) - KroneckerDelta(j, n) * KroneckerDelta(k, m)
    for j in R
    for k in R
    for m in R
    for n in R
)
expected = True
```

## The triple vector product, derived

**Level 3** · `la.index.bac-cab` · Griffiths (2017), section 1.1.3

The BAC–CAB rule, \(\mathbf{a} \times (\mathbf{b} \times \mathbf{c}) =
\mathbf{b}(\mathbf{a} \cdot \mathbf{c}) - \mathbf{c}(\mathbf{a} \cdot \mathbf{b})\), derived
in index notation for vectors that have no components: the epsilon–delta identity, put
into the product, and the deltas contracted.

```latex
V := \mathbb{R}^{3}
i, j, k, l, m \in V
\epsilon_{ijk} \epsilon_{klm} = \delta_{il} \delta_{jm} - \delta_{im} \delta_{jl}
\epsilon_{ijk} a_j \epsilon_{klm} b_l c_m = (\delta_{il} \delta_{jm} - \delta_{im} \delta_{jl}) a_j b_l c_m
(\delta_{il} \delta_{jm} - \delta_{im} \delta_{jl}) a_j b_l c_m = b_i a_j c_j - c_i a_j b_j
```

```python
R = range(3)
a, b, c = symbols("a1:4"), symbols("b1:4"), symbols("c1:4")
eps, delta = LeviCivita, KroneckerDelta
assert all(
    sum(eps(i, j, k) * eps(k, l, m) for k in R)
    == delta(i, l) * delta(j, m) - delta(i, m) * delta(j, l)
    for i in R
    for j in R
    for l in R
    for m in R
)
for i in R:
    triple = sum(
        eps(i, j, k) * a[j] * eps(k, l, m) * b[l] * c[m] for j in R for k in R for l in R for m in R
    )
    deltas = sum(
        (delta(i, l) * delta(j, m) - delta(i, m) * delta(j, l)) * a[j] * b[l] * c[m]
        for j in R
        for l in R
        for m in R
    )
    rule = b[i] * sum(a[j] * c[j] for j in R) - c[i] * sum(a[j] * b[j] for j in R)
    assert expand(triple - deltas) == 0 and expand(deltas - rule) == 0
outputs = [True, True, True]
```

## An index three times

**Level 1** · `la.index.thrice` · standard

An index appears in a term once, free, or twice, summed. \(a_i b_i c_i\) is not index
notation, and must be refused.

```latex
V := \mathbb{R}^{3}
i \in V
a_{i} := [1, 2, 3]
b_{i} := [4, 5, 6]
c_{i} := [7, 8, 9]
a_i b_i c_i
```

```python
error = "IndexNotationError"
```

## Free indices that differ

**Level 1** · `la.index.free-indices` · standard

Every term of an equation has the same free indices. \(A_{ij} x_j = b_j\), a slip for
\(A_{ij} x_j = b_i\), is free in \(i\) on the left and in \(j\) on the right, and must be
refused, though these components satisfy the equation the slip stands for.

```latex
V := \mathbb{R}^{3}
i, j \in V
A_{ij} := \begin{pmatrix} 2 & 4 & -2 \\ 4 & 9 & -3 \\ -2 & -3 & 7 \end{pmatrix}
x_{j} := [-1, 2, 2]
b_{j} := [2, 8, 10]
A_{ij} x_j = b_j
```

```python
error = "IndexNotationError"
```
