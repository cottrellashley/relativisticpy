# The trace of a tensor

A tensor of two indices contracted with itself, one index up and one down,
\(T = T^\mu{}_\mu = g^{\mu\nu} T_{\mu\nu}\): its trace. Carroll (1997) is *Lecture Notes on
General Relativity*, arXiv:gr-qc/9712019; Schutz (2009) is *A First Course in General
Relativity*, 2nd edition.

Each operand gives a metric by its line element, `«line»`, and a tensor by its components,
`«T»`, with its indices written as `«indices»`: both down, both up, or one of each.
`«last»` is the number of the last coordinate, counted from 0.

## Names

```python
t, x, y, z, r, theta, phi = symbols("t x y z r theta phi")
rho, p, a, b, c = symbols("rho p a b c")
```

## Operands

| operand | written in | `line` | `indices` | `T` | `last` | `position` | separates |
|---|---|---|---|---|---|---|---|
| `mixed` | latex | `-dt^2 + dx^2 + dy^2 + dz^2` | `^{\mu}{}_{\nu}` | `[[1, 2, 0, 0], [0, 3, 0, 0], [0, 0, 5, 0], [0, 0, 0, 7]]` | `3` | | a tensor given with one index up and one down, whose trace needs no metric: 16 |
| | text | `-dt^2 + dx^2 + dy^2 + dz^2` | `^{mu}_{nu}` | `[[1, 2, 0, 0], [0, 3, 0, 0], [0, 0, 5, 0], [0, 0, 0, 7]]` | | | |
| | unicode | `−dt² + dx² + dy² + dz²` | `^μ_ν` | `[[1, 2, 0, 0], [0, 3, 0, 0], [0, 0, 5, 0], [0, 0, 0, 7]]` | | | |
| | python | `diag(-1, 1, 1, 1)` | | `[[1, 2, 0, 0], [0, 3, 0, 0], [0, 0, 5, 0], [0, 0, 0, 7]]` | | `"mixed"` | |
| `lower` | latex | `-dt^2 + dx^2 + dy^2 + dz^2` | `_{\mu\nu}` | `[[\rho, 0, 0, 0], [0, p, 0, 0], [0, 0, p, 0], [0, 0, 0, p]]` | `3` | | a perfect fluid at rest, given with both indices down: −ρ + 3p, where the sum of the diagonal is ρ + 3p |
| | text | `-dt^2 + dx^2 + dy^2 + dz^2` | `_{mu nu}` | `[[rho, 0, 0, 0], [0, p, 0, 0], [0, 0, p, 0], [0, 0, 0, p]]` | | | |
| | unicode | `−dt² + dx² + dy² + dz²` | `_{μν}` | `[[ρ, 0, 0, 0], [0, p, 0, 0], [0, 0, p, 0], [0, 0, 0, p]]` | | | |
| | python | `diag(-1, 1, 1, 1)` | | `diag(rho, p, p, p)` | | `"lower"` | |
| `upper` | latex | `-dt^2 + dx^2 + dy^2 + dz^2` | `^{\mu\nu}` | `[[\rho, 0, 0, 0], [0, p, 0, 0], [0, 0, p, 0], [0, 0, 0, p]]` | `3` | | the same fluid with both indices up, as Carroll (1997), eq. (1.110), gives it: −ρ + 3p, eq. (8.19) |
| | text | `-dt^2 + dx^2 + dy^2 + dz^2` | `^{mu nu}` | `[[rho, 0, 0, 0], [0, p, 0, 0], [0, 0, p, 0], [0, 0, 0, p]]` | | | |
| | unicode | `−dt² + dx² + dy² + dz²` | `^{μν}` | `[[ρ, 0, 0, 0], [0, p, 0, 0], [0, 0, p, 0], [0, 0, 0, p]]` | | | |
| | python | `diag(-1, 1, 1, 1)` | | `diag(rho, p, p, p)` | | `"upper"` | |
| `polar` | latex | `dr^2 + r^2 d\theta^2` | `_{\mu\nu}` | `[[1, 0], [0, 1]]` | `1` | | a metric with a coordinate in it: 1 + 1/r², where the sum of the diagonal is 2 |
| | text | `dr^2 + r^2*dtheta^2` | `_{mu nu}` | `[[1, 0], [0, 1]]` | | | |
| | unicode | `dr² + r² dθ²` | `_{μν}` | `[[1, 0], [0, 1]]` | | | |
| | python | `diag(1, r**2)` | | `[[1, 0], [0, 1]]` | | `"lower"` | |
| `metric` | latex | `dr^2 + r^2 d\theta^2` | `_{\mu\nu}` | `[[1, 0], [0, r^2]]` | `1` | | the metric's own components, whose trace is the dimension, 2 |
| | text | `dr^2 + r^2*dtheta^2` | `_{mu nu}` | `[[1, 0], [0, r^2]]` | | | |
| | unicode | `dr² + r² dθ²` | `_{μν}` | `[[1, 0], [0, r²]]` | | | |
| | python | `diag(1, r**2)` | | `diag(1, r**2)` | | `"lower"` | |
| `sphere` | latex | `d\theta^2 + \sin^2\theta \, d\phi^2` | `_{\mu\nu}` | `[[a, b], [b, c]]` | `1` | | a metric with a function of a coordinate in it, and a tensor with a part off the diagonal: a + c/sin²θ |
| | text | `dtheta^2 + sin(theta)^2*dphi^2` | `_{mu nu}` | `[[a, b], [b, c]]` | | | |
| | unicode | `dθ² + sin²θ dφ²` | `_{μν}` | `[[a, b], [b, c]]` | | | |
| | python | `diag(1, sin(theta)**2)` | | `[[a, b], [b, c]]` | | `"lower"` | |

## Setup

```latex
ds^2 = «line»
T«indices» := «T»
```

```text
ds^2 = «line»
T«indices» := «T»
```

```unicode
ds² = «line»
T«indices» := «T»
```

## Oracle

The line is written as the metric's components in Python, and `position` says where the
indices of the components are.

```python
metric = Matrix(line)
components = Matrix(T)
if position == "mixed":
    mixed = components
elif position == "lower":
    mixed = metric.inv() * components
else:
    mixed = components * metric
expected = simplify(mixed.trace())
```

## Second route

The sum of the eigenvalues of the tensor taken as a map, its index raised, each as often
as it is repeated.

```python
eigenvalues = mixed.eigenvals()
assert simplify(sum(value * count for value, count in eigenvalues.items()) - expected) == 0
```

## Forms

### Indices staggered

**Form** · `staggered` · Carroll (1997), eq. (3.92): `R = R^\mu{}_\mu`, and eq. (8.19): `T = T^\mu{}_\mu = -\rho + 3p`

```latex
T^\mu{}_\mu
```

### Indices one above the other

**Form** · `stacked` · Carroll (1997), eq. (3.27): `\Gamma^\mu_{\mu\lambda}`, a contraction written with its indices one above the other

```latex
T^\mu_\mu
```

### Indices staggered, in braces

**Form** · `braces` · docs/tensors.md, Defining tensors: `T^{\mu}{}_{\mu}`

```latex
T^{\mu}{}_{\mu}
```

### The lower index first

**Form** · `lower-first` · Carroll (1997), eq. (1.61): an index lowered keeps its place, as in `T_\mu{}^\beta{}_{\gamma\delta}`

```latex
T_\mu{}^\mu
```

### The inverse metric written out

**Form** · `inverse-metric` · Carroll (1997), eq. (3.92): `R = g^{\mu\nu} R_{\mu\nu}`

```latex
g^{\mu\nu} T_{\mu\nu}
```

### An explicit sum

**Form** · `sum` · Schutz (2009), §2.1: a sum over an index written out, `\sum_{\beta=0}^{3}`, before the summation convention leaves it out

```latex
\sum_{\mu=0}^{«last»} T^\mu{}_\mu
```

### The trace of the map

**Form** · `operatorname-tr` · docs/spaces.md, Linear algebra: `\operatorname{tr}` of a map, `A^a{}_b`, or of a matrix of components

```latex
\operatorname{tr}(T^\mu{}_\nu)
```

### Indices in plain text

**Form** · `text` · docs/tensors.md, Index notation: Greek names spelled out, `_{mu nu}`

```text
T^{mu}_{mu}
```

### The inverse metric written out, in plain text

**Form** · `text-inverse-metric` · docs/tensors.md, Index notation: Greek names spelled out, `_{mu nu}`

```text
g^{mu nu} T_{mu nu}
```

### Indices in Unicode

**Form** · `unicode` · docs/reference.md, Unicode: a Greek letter is its command, so `T^μ_μ` is `T^\mu_\mu`

```unicode
T^μ_μ
```

### The inverse metric written out, in Unicode

**Form** · `unicode-inverse-metric` · docs/reference.md, Unicode: a Greek letter is its command

```unicode
g^{μν} T_{μν}
```
