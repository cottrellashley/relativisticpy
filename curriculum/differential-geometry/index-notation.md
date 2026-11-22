# Index notation

Einstein's summation convention and its rules: an index repeated once up and once down is
summed, every term has the same free indices, and nothing else is allowed. Notation that
breaks the rules is not a computation to guess at; it must be refused.

## The trace of the Kronecker delta

**Level 1** · `dg.index-notation.kronecker-trace` · standard

The Kronecker delta contracted with itself counts the dimensions.

```latex
Coordinates := [t, x, y, z]
\delta^\mu{}_\mu
```

```python
expected = eye(4).trace()
```

## An index repeated three times

**Level 1** · `dg.index-notation.repeated-thrice` · standard

\(A_\mu B^\mu A_\mu\) uses \(\mu\) three times: it is not a sum, and means nothing.

```latex
ds^2 = d\theta^2 + \sin^2\theta\, d\phi^2
A_\mu := [1, \cos\theta]
B^\mu := [\sin\theta, 1]
A_\mu B^\mu A_\mu
```

```python
error = "IndexNotationError"
```

## Free indices that do not match

**Level 1** · `dg.index-notation.free-indices` · standard

A conservation law with a slip: the divergence has the free index \(\nu\), the source
\(\mu\), which is summed on the left.

```latex
ds^2 = -dt^2 + dx^2 + dy^2 + dz^2
T^{\mu\nu} := diag(\rho, p, p, p)
J^\mu := [\rho, 0, 0, 0]
\partial_\mu T^{\mu\nu} = J^\mu
```

```python
error = "IndexNotationError"
```

## Free indices in different positions

**Level 1** · `dg.index-notation.index-position` · standard

Einstein's equation with its source written contravariant: the free indices have the same
names on both sides, but are down on the left and up on the right.

```latex
G_{\mu\nu} + \Lambda g_{\mu\nu} = 8 \pi G T^{\mu\nu}
```

```python
error = "IndexNotationError"
```

## The contracted epsilon identity

**Level 2** · `dg.index-notation.epsilon-delta` · standard

The Levi-Civita symbol of three-dimensional Euclidean space, contracted on one index:
\(\epsilon_{ijk} \epsilon^{imn} = \delta_j^m \delta_k^n - \delta_j^n \delta_k^m\).

```latex
ds^2 = dx^2 + dy^2 + dz^2
\epsilon_{ijk} \epsilon^{imn} = \delta_j^m \delta_k^n - \delta_j^n \delta_k^m
```

```python
d = lambda a, b: 1 if a == b else 0
expected = all(
    sum(LeviCivita(i, j, k) * LeviCivita(i, m, n) for i in range(3))
    == d(j, m) * d(k, n) - d(j, n) * d(k, m)
    for j in range(3)
    for k in range(3)
    for m in range(3)
    for n in range(3)
)
assert expected
```
