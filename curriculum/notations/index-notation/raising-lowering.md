# Raising and lowering an index

An index lowered or raised by the metric, \(V_\mu = g_{\mu\nu} V^\nu\) and
\(\omega^\mu = g^{\mu\nu} \omega_\nu\), written with the same letter in the other position, with
the metric written out, and component by component. Carroll (1997) is *Lecture Notes on
General Relativity*, arXiv:gr-qc/9712019; Landau and Lifshitz (1975) is *The Classical
Theory of Fields*, 4th edition, which writes the metric with the signature (+ − − −).

Each operand gives a metric by its line element, `«line»`, and the components of a vector
`«V»` to lower, of a covector `«omega»` to raise, or of a tensor `«F»` with both indices down
to raise. `«named»` and `«numbered»` are the components asked for, listed by the names of
the coordinates and by their numbers, counted from 0; `«eta»` is the metric of flat
spacetime as a matrix.

## Names

```python
t, x, y, z, r, theta, phi = symbols("t x y z r theta phi")
a, b, E, B, M = symbols("a b E B M")
```

## Operands

| operand | written in | `line` | `V` | `omega` | `F` | `named` | `numbered` | `eta` | `kind` | `components` | separates |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `minkowski` | latex | `-dt^2 + dx^2 + dy^2 + dz^2` | `[5, 1, 2, 3]` | | | `[V_t, V_x, V_y, V_z]` | `[V_0, V_1, V_2, V_3]` | `\begin{pmatrix} -1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}` | | | lowering by the metric of flat spacetime: the time component changes sign |
| | text | `-dt^2 + dx^2 + dy^2 + dz^2` | `[5, 1, 2, 3]` | | | | | | | | |
| | unicode | `−dt² + dx² + dy² + dz²` | `[5, 1, 2, 3]` | | | | | | | | |
| | python | `diag(-1, 1, 1, 1)` | | | | | | | `"lower"` | `[5, 1, 2, 3]` | |
| `mostly-minus` | latex | `dt^2 - dx^2 - dy^2 - dz^2` | `[5, 1, 2, 3]` | | | `[V_t, V_x, V_y, V_z]` | `[V_0, V_1, V_2, V_3]` | `\begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & -1 \end{pmatrix}` | | | the signature of Landau and Lifshitz (1975): the space components change sign |
| | text | `dt^2 - dx^2 - dy^2 - dz^2` | `[5, 1, 2, 3]` | | | | | | | | |
| | unicode | `dt² − dx² − dy² − dz²` | `[5, 1, 2, 3]` | | | | | | | | |
| | python | `diag(1, -1, -1, -1)` | | | | | | | `"lower"` | `[5, 1, 2, 3]` | |
| `polar` | latex | `dr^2 + r^2 d\theta^2` | `[a, b]` | | | `[V_r, V_\theta]` | `[V_0, V_1]` | | | | lowering by a metric with a coordinate in it: (a, r²b) |
| | text | `dr^2 + r^2*dtheta^2` | `[a, b]` | | | | | | | | |
| | unicode | `dr² + r² dθ²` | `[a, b]` | | | | | | | | |
| | python | `diag(1, r**2)` | | | | | | | `"lower"` | `[a, b]` | |
| `schwarzschild` | latex | `-\left(1 - \frac{2M}{r}\right) dt^2 + \frac{dr^2}{1 - \frac{2M}{r}} + r^2 d\Omega^2` | `[\frac{1}{\sqrt{1 - \frac{2M}{r}}}, 0, 0, 0]` | | | `[V_t, V_r, V_\theta, V_\phi]` | `[V_0, V_1, V_2, V_3]` | | | | the four-velocity of an observer at rest in a curved metric |
| | text | `-(1 - 2*M/r)*dt^2 + dr^2/(1 - 2*M/r) + r^2*(dtheta^2 + sin(theta)^2*dphi^2)` | `[1/sqrt(1 - 2*M/r), 0, 0, 0]` | | | | | | | | |
| | unicode | `−(1 − 2M/r) dt² + dr²/(1 − 2M/r) + r² dΩ²` | `[1/√(1 − 2M/r), 0, 0, 0]` | | | | | | | | |
| | python | `diag(-(1 - 2*M/r), 1/(1 - 2*M/r), r**2, r**2*sin(theta)**2)` | | | | | | | `"lower"` | `[1/sqrt(1 - 2*M/r), 0, 0, 0]` | |
| `covector` | latex | `-dt^2 + dx^2 + dy^2 + dz^2` | | `[5, 1, 2, 3]` | | `[\omega^t, \omega^x, \omega^y, \omega^z]` | `[\omega^{0}, \omega^{1}, \omega^{2}, \omega^{3}]` | `\begin{pmatrix} -1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}` | | | raising by the inverse metric of flat spacetime, Carroll (1997), eq. (1.63) |
| | text | `-dt^2 + dx^2 + dy^2 + dz^2` | | `[5, 1, 2, 3]` | | | | | | | |
| | unicode | `−dt² + dx² + dy² + dz²` | | `[5, 1, 2, 3]` | | | | | | | |
| | python | `diag(-1, 1, 1, 1)` | | | | | | | `"raise"` | `[5, 1, 2, 3]` | |
| `polar-covector` | latex | `dr^2 + r^2 d\theta^2` | | `[a, b]` | | `[\omega^r, \omega^\theta]` | `[\omega^{0}, \omega^{1}]` | | | | raising by the inverse of a metric with a coordinate in it: (a, b/r²), where lowering gives (a, r²b) |
| | text | `dr^2 + r^2*dtheta^2` | | `[a, b]` | | | | | | | |
| | unicode | `dr² + r² dθ²` | | `[a, b]` | | | | | | | |
| | python | `diag(1, r**2)` | | | | | | | `"raise"` | `[a, b]` | |
| `sphere-covector` | latex | `d\theta^2 + \sin^2\theta \, d\phi^2` | | `[a, b]` | | `[\omega^\theta, \omega^\phi]` | `[\omega^{0}, \omega^{1}]` | | | | raising by the inverse of a metric with a function of a coordinate in it: (a, b/sin²θ) |
| | text | `dtheta^2 + sin(theta)^2*dphi^2` | | `[a, b]` | | | | | | | |
| | unicode | `dθ² + sin²θ dφ²` | | `[a, b]` | | | | | | | |
| | python | `diag(1, sin(theta)**2)` | | | | | | | `"raise"` | `[a, b]` | |
| `field-strength` | latex | `-dt^2 + dx^2 + dy^2 + dz^2` | | | `[[0, -E, 0, 0], [E, 0, 0, 0], [0, 0, 0, B], [0, 0, -B, 0]]` | | | | | | both indices of an antisymmetric tensor raised: its time row and column change sign, the rest do not |
| | text | `-dt^2 + dx^2 + dy^2 + dz^2` | | | `[[0, -E, 0, 0], [E, 0, 0, 0], [0, 0, 0, B], [0, 0, -B, 0]]` | | | | | | |
| | unicode | `−dt² + dx² + dy² + dz²` | | | `[[0, −E, 0, 0], [E, 0, 0, 0], [0, 0, 0, B], [0, 0, −B, 0]]` | | | | | | |
| | python | `diag(-1, 1, 1, 1)` | | | | | | | `"raise both"` | `[[0, -E, 0, 0], [E, 0, 0, 0], [0, 0, 0, B], [0, 0, -B, 0]]` | |

## Setup

```latex
ds^2 = «line»
```

```text
ds^2 = «line»
```

```unicode
ds² = «line»
```

## Oracle

The line is written as the metric's components in Python; `kind` says what is done to the
components.

```python
metric = Matrix(line)
inverse = metric.inv()
given = Matrix(components)
if kind == "lower":
    found = metric * given
elif kind == "raise":
    found = inverse * given
else:
    found = inverse * given * inverse
expected = [[simplify(each) for each in row] for row in found.tolist()]
if kind != "raise both":
    expected = [row[0] for row in expected]
```

## Second route

The other operation undoes it: what is lowered, raised again, and what is raised, lowered
again, are what was given.

```python
back = Matrix(expected)
if kind == "lower":
    back = inverse * back
elif kind == "raise":
    back = metric * back
else:
    back = metric * back * metric
assert all(simplify(each) == 0 for each in back - given)
```

## Forms

### The same letter, its index down

**Form** · `lower` · Carroll (1997), eq. (1.61): a tensor with an index lowered by the metric is denoted by the same letter

```latex
V^\mu := «V»
V_\mu
```

### The metric written out

**Form** · `lower-by-metric` · Carroll (1997), eq. (1.62): `V_\mu = \eta_{\mu\nu} V^\nu`, and eq. (2.29), where `g_{\mu\nu}` takes the place of `\eta_{\mu\nu}`

```latex
V^\mu := «V»
g_{\mu\nu} V^\nu
```

### The metric of flat spacetime, as a matrix

**Form** · `lower-by-eta` · Carroll (1997), eq. (1.8): `\eta_{\mu\nu}` as a matrix, and eq. (1.62): `V_\mu = \eta_{\mu\nu} V^\nu`

```latex
\eta_{\mu\nu} := «eta»
V^\mu := «V»
\eta_{\mu\nu} V^\nu
```

### Each component, by the coordinate's name

**Form** · `lower-by-name` · Carroll (1997), eq. (3.103): an index lowered, component by component, `R_{\theta\phi\theta\phi} = g_{\theta\lambda} R^\lambda{}_{\phi\theta\phi}`; docs/tensors.md, Components: a coordinate's name as an index fixes the component

```latex
V^\mu := «V»
«named»
```

### Each component, by number

**Form** · `lower-by-number` · Carroll (1997), eq. (1.63): components numbered from 0, `(-\omega_0, \omega_1, \omega_2, \omega_3)`; docs/tensors.md, Components: `g_{01}`

```latex
V^\mu := «V»
«numbered»
```

### The same letter, its index up

**Form** · `raise` · Carroll (1997), eq. (1.61): a tensor with an index raised by the inverse metric is denoted by the same letter

```latex
\omega_\mu := «omega»
\omega^\mu
```

### The inverse metric written out

**Form** · `raise-by-metric` · Carroll (1997), eq. (1.62): `\omega^\mu = \eta^{\mu\nu} \omega_\nu`, with `g^{\mu\nu}` in the place of `\eta^{\mu\nu}`

```latex
\omega_\mu := «omega»
g^{\mu\nu} \omega_\nu
```

### The inverse metric of flat spacetime, as a matrix

**Form** · `raise-by-eta` · Carroll (1997), eq. (1.62): `\omega^\mu = \eta^{\mu\nu} \omega_\nu`, with `\eta^{\mu\nu}` written as the matrix of eq. (1.8), which is its own inverse

```latex
\eta^{\mu\nu} := «eta»
\omega_\mu := «omega»
\eta^{\mu\nu} \omega_\nu
```

### Each raised component, by the coordinate's name

**Form** · `raise-by-name` · docs/tensors.md, Components: a coordinate's name as an index fixes the component

```latex
\omega_\mu := «omega»
«named»
```

### Each raised component, by number in braces

**Form** · `raise-by-number` · docs/tensors.md, Derivatives: on a tensor with one index, `k^{3}`, in braces, is its component

```latex
\omega_\mu := «omega»
«numbered»
```

### Both indices up, the same letter

**Form** · `raise-both` · Carroll (1997), eq. (1.61): `T_{\mu\nu}{}^{\rho\sigma}`, indices moved by the metric under the same letter

```latex
F_{\mu\nu} := «F»
F^{\mu\nu}
```

### Both indices up, the inverse metric written out

**Form** · `raise-both-by-metric` · Carroll (1997), eq. (1.61): `\eta_{\mu\alpha}\eta_{\nu\beta}\eta^{\rho\gamma}\eta^{\sigma\delta} T^{\alpha\beta}{}_{\gamma\delta}`, a metric for each index moved

```latex
F_{\mu\nu} := «F»
g^{\mu\alpha} g^{\nu\beta} F_{\alpha\beta}
```

### The index down, in plain text

**Form** · `text-lower` · docs/tensors.md, Index notation: Greek names spelled out, `_{mu nu}`

```text
V^{mu} := «V»
V_{mu}
```

### The metric written out, in plain text

**Form** · `text-lower-by-metric` · docs/tensors.md, Index notation: Greek names spelled out, `_{mu nu}`

```text
V^{mu} := «V»
g_{mu nu} V^{nu}
```

### The index up, in plain text

**Form** · `text-raise` · docs/tensors.md, Index notation: Greek names spelled out

```text
omega_{mu} := «omega»
omega^{mu}
```

### The inverse metric written out, in plain text

**Form** · `text-raise-by-metric` · docs/tensors.md, Index notation: Greek names spelled out

```text
omega_{mu} := «omega»
g^{mu nu} omega_{nu}
```

### Both indices up, in plain text

**Form** · `text-raise-both` · docs/tensors.md, Index notation: Greek names spelled out

```text
F_{mu nu} := «F»
F^{mu nu}
```

### The index down, in Unicode

**Form** · `unicode-lower` · docs/reference.md, Unicode: a Greek letter is its command

```unicode
V^μ := «V»
V_μ
```

### The metric written out, in Unicode

**Form** · `unicode-lower-by-metric` · docs/reference.md, Unicode: a Greek letter is its command

```unicode
V^μ := «V»
g_{μν} V^ν
```

### The index up, in Unicode

**Form** · `unicode-raise` · docs/reference.md, Unicode: a Greek letter is its command

```unicode
ω_μ := «omega»
ω^μ
```

### The inverse metric written out, in Unicode

**Form** · `unicode-raise-by-metric` · docs/reference.md, Unicode: a Greek letter is its command

```unicode
ω_μ := «omega»
g^{μν} ω_ν
```

### Both indices up, in Unicode

**Form** · `unicode-raise-both` · docs/reference.md, Unicode: a Greek letter is its command

```unicode
F_{μν} := «F»
F^{μν}
```
