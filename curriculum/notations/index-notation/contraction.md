# A vector contracted with itself

A vector contracted with itself by the metric, \(g_{\mu\nu} V^\mu V^\nu = V_\mu V^\mu\), the
norm of the vector in Carroll's word, written with its index lowered, with the metric
written out, with an explicit sum and as a dot. Carroll (1997) is *Lecture Notes on
General Relativity*, arXiv:gr-qc/9712019; Schutz (2009) is *A First Course in General
Relativity*, 2nd edition; Landau and Lifshitz (1975) is *The Classical Theory of Fields*,
4th edition, which writes the metric with the signature (+ − − −).

Each operand gives a metric by its line element, `«line»`, and the components of a vector,
`«V»`; `«eta»` is the metric of flat spacetime as a matrix, and `«last»` the number of the
last coordinate, counted from 0.

## Names

```python
t, x, y, z, r, theta, phi = symbols("t x y z r theta phi")
v, omega, a, b, M = symbols("v omega a b M")
```

## Operands

| operand | written in | `line` | `V` | `eta` | `last` | separates |
|---|---|---|---|---|---|---|
| `four-velocity` | latex | `-dt^2 + dx^2 + dy^2 + dz^2` | `[\frac{1}{\sqrt{1 - v^2}}, \frac{v}{\sqrt{1 - v^2}}, 0, 0]` | `\begin{pmatrix} -1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}` | `3` | a timelike vector, normalised to −1: its time part counts negative |
| | text | `-dt^2 + dx^2 + dy^2 + dz^2` | `[1/sqrt(1 - v^2), v/sqrt(1 - v^2), 0, 0]` | | | |
| | unicode | `−dt² + dx² + dy² + dz²` | `[1/√(1 − v²), v/√(1 − v²), 0, 0]` | | | |
| | python | `diag(-1, 1, 1, 1)` | `[1/sqrt(1 - v**2), v/sqrt(1 - v**2), 0, 0]` | | | |
| `null` | latex | `-dt^2 + dx^2 + dy^2 + dz^2` | `[\omega, 0, 0, \omega]` | `\begin{pmatrix} -1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}` | `3` | a null vector, of norm 0: without the metric, 2ω² |
| | text | `-dt^2 + dx^2 + dy^2 + dz^2` | `[omega, 0, 0, omega]` | | | |
| | unicode | `−dt² + dx² + dy² + dz²` | `[ω, 0, 0, ω]` | | | |
| | python | `diag(-1, 1, 1, 1)` | `[omega, 0, 0, omega]` | | | |
| `mostly-minus` | latex | `dt^2 - dx^2 - dy^2 - dz^2` | `[3, 1, 1, 1]` | `\begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & -1 \end{pmatrix}` | `3` | the signature of Landau and Lifshitz (1975), whose interval is c²dt² − dx² − dy² − dz²: 6, where the other gives −6 |
| | text | `dt^2 - dx^2 - dy^2 - dz^2` | `[3, 1, 1, 1]` | | | |
| | unicode | `dt² − dx² − dy² − dz²` | `[3, 1, 1, 1]` | | | |
| | python | `diag(1, -1, -1, -1)` | `[3, 1, 1, 1]` | | | |
| `polar` | latex | `dr^2 + r^2 d\theta^2` | `[a, b]` | | `1` | a metric with a coordinate in it, of the plane in polar coordinates: without it, a² + b² |
| | text | `dr^2 + r^2*dtheta^2` | `[a, b]` | | | |
| | unicode | `dr² + r² dθ²` | `[a, b]` | | | |
| | python | `diag(1, r**2)` | `[a, b]` | | | |
| `sphere` | latex | `d\theta^2 + \sin^2\theta \, d\phi^2` | `[1, 1]` | | `1` | a metric with a function of a coordinate in it, of the unit sphere |
| | text | `dtheta^2 + sin(theta)^2*dphi^2` | `[1, 1]` | | | |
| | unicode | `dθ² + sin²θ dφ²` | `[1, 1]` | | | |
| | python | `diag(1, sin(theta)**2)` | `[1, 1]` | | | |
| `schwarzschild` | latex | `-\left(1 - \frac{2M}{r}\right) dt^2 + \frac{dr^2}{1 - \frac{2M}{r}} + r^2 d\Omega^2` | `[\frac{1}{1 - \frac{2M}{r}}, 1, 0, 0]` | | `3` | a radial null vector of a curved metric |
| | text | `-(1 - 2*M/r)*dt^2 + dr^2/(1 - 2*M/r) + r^2*(dtheta^2 + sin(theta)^2*dphi^2)` | `[1/(1 - 2*M/r), 1, 0, 0]` | | | |
| | unicode | `−(1 − 2M/r) dt² + dr²/(1 − 2M/r) + r² dΩ²` | `[1/(1 − 2M/r), 1, 0, 0]` | | | |
| | python | `diag(-(1 - 2*M/r), 1/(1 - 2*M/r), r**2, r**2*sin(theta)**2)` | `[1/(1 - 2*M/r), 1, 0, 0]` | | | |

## Setup

```latex
ds^2 = «line»
V^\mu := «V»
```

```text
ds^2 = «line»
V^{mu} := «V»
```

```unicode
ds² = «line»
V^μ := «V»
```

## Oracle

The line is written as the metric's components in Python.

```python
metric = Matrix(line)
column = Matrix(V)
expected = simplify((column.T * metric * column)[0, 0])
```

## Second route

The covector the metric lowers the vector to, contracted with itself by the inverse metric.

```python
covector = metric * column
assert simplify((covector.T * metric.inv() * covector)[0, 0] - expected) == 0
```

## Forms

### The lower index first

**Form** · `lower-upper` · Carroll (1997), eq. (1.35): `\omega_\mu V^\mu`, a lower index summed with an upper one

```latex
V_\mu V^\mu
```

### The upper index first

**Form** · `upper-lower` · Carroll (1997), eq. (7.123): `l^\mu l_\mu = 0`, a null vector contracted with itself

```latex
V^\mu V_\mu
```

### Indices in braces

**Form** · `braces` · docs/tensors.md, Defining tensors: `u_{\mu} u^{\mu}`

```latex
V_{\mu} V^{\mu}
```

### The metric written out

**Form** · `metric` · Carroll (1997), §1: the norm of a vector, `\eta_{\mu\nu} V^\mu V^\nu`, and eq. (2.29), where `g_{\mu\nu}` takes the place of `\eta_{\mu\nu}`

```latex
g_{\mu\nu} V^\mu V^\nu
```

### The metric of flat spacetime, as a matrix

**Form** · `eta` · Carroll (1997), eq. (1.8): `\eta_{\mu\nu}` as a matrix, and eq. (1.99): `\eta_{\mu\nu} U^\mu U^\nu = -1`

```latex
\eta_{\mu\nu} := «eta»
\eta_{\mu\nu} V^\mu V^\nu
```

### A dot

**Form** · `dot` · Carroll (1997), eq. (1.55): `\eta(V, W) = \eta_{\mu\nu} V^\mu W^\nu = V \cdot W`

```latex
V \cdot V
```

### An explicit sum

**Form** · `sum` · Schutz (2009), §2.1: a sum over an index written out, `\sum_{\beta=0}^{3}`, before the summation convention leaves it out

```latex
\sum_{\mu=0}^{«last»} V_\mu V^\mu
```

### Indices in plain text

**Form** · `text` · docs/tensors.md, Index notation: Greek names spelled out, `_{mu nu}`

```text
V_{mu} V^{mu}
```

### The metric written out, in plain text

**Form** · `text-metric` · docs/tensors.md, Index notation: the plain-text style `_{a}_{b}`

```text
g_{mu}_{nu} V^{mu} V^{nu}
```

### Indices in Unicode

**Form** · `unicode` · docs/reference.md, Unicode: a Greek letter is its command, so `V_μ V^μ` is `V_\mu V^\mu`

```unicode
V_μ V^μ
```

### The metric written out, in Unicode

**Form** · `unicode-metric` · docs/reference.md, Unicode: a Greek letter is its command

```unicode
g_{μν} V^μ V^ν
```

### A dot in Unicode

**Form** · `unicode-dot` · Carroll (1997), eq. (1.55): `V \cdot W`, typed with U+00B7 MIDDLE DOT, which docs/reference.md reads as `\cdot`

```unicode
V·V
```
