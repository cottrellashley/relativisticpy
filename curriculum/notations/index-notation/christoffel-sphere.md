# The Christoffel symbols of a sphere

The connection coefficients of the two-sphere of radius a, whose metric is
\(ds^2 = a^2(d\theta^2 + \sin^2\theta\, d\phi^2)\): \(\Gamma^\theta_{\phi\phi} = -\sin\theta\cos\theta\)
and \(\Gamma^\phi_{\theta\phi} = \Gamma^\phi_{\phi\theta} = \cot\theta\), the rest 0. The metric is
given by its line element and by its components; a symbol is asked for by the names of the
coordinates, by their numbers, or by its definition, and all of them at once by free
indices. Carroll (1997) is *Lecture Notes on General Relativity*, arXiv:gr-qc/9712019,
which works this example in §3, eqs. (3.100) and (3.101); Knuth (1984) is *The TeXbook*.

`«radius»` is the radius. A symbol \(\Gamma^{u}{}_{fs}\) is asked for by its upper index,
`«up»`, and its two lower ones, `«first»` and `«second»`: as the names of coordinates, or
as letters free to run over them, which ask for every symbol. `«up-n»`, `«first-n»` and
`«second-n»` are the same indices as the numbers of the coordinates, θ being 0 and φ 1.

## Names

```python
theta, phi = symbols("theta phi")
a, R = symbols("a R", positive=True)
```

## Operands

| operand | written in | `radius` | `up` | `first` | `second` | `up-n` | `first-n` | `second-n` | `component` | separates |
|---|---|---|---|---|---|---|---|---|---|---|
| `theta-phi-phi` | latex | `a` | `\theta` | `\phi` | `\phi` | `0` | `1` | `1` | | the one symbol with an upper θ that is not 0, both lower indices alike |
| | text | `a` | `theta` | `phi` | `phi` | | | | | |
| | unicode | `a` | `θ` | `φ` | `φ` | | | | | |
| | python | `a` | | | | | | | `(0, 1, 1)` | |
| `phi-theta-phi` | latex | `a` | `\phi` | `\theta` | `\phi` | `1` | `0` | `1` | | a symbol whose lower indices differ |
| | text | `a` | `phi` | `theta` | `phi` | | | | | |
| | unicode | `a` | `φ` | `θ` | `φ` | | | | | |
| | python | `a` | | | | | | | `(1, 0, 1)` | |
| `phi-phi-theta` | latex | `a` | `\phi` | `\phi` | `\theta` | `1` | `1` | `0` | | the same symbol, its lower indices the other way round, which the connection's symmetry makes equal |
| | text | `a` | `phi` | `phi` | `theta` | | | | | |
| | unicode | `a` | `φ` | `φ` | `θ` | | | | | |
| | python | `a` | | | | | | | `(1, 1, 0)` | |
| `theta-theta-theta` | latex | `a` | `\theta` | `\theta` | `\theta` | `0` | `0` | `0` | | a symbol that is 0 |
| | text | `a` | `theta` | `theta` | `theta` | | | | | |
| | unicode | `a` | `θ` | `θ` | `θ` | | | | | |
| | python | `a` | | | | | | | `(0, 0, 0)` | |
| `phi-phi-phi` | latex | `a` | `\phi` | `\phi` | `\phi` | `1` | `1` | `1` | | another symbol that is 0, of the coordinate the metric does not depend on |
| | text | `a` | `phi` | `phi` | `phi` | | | | | |
| | unicode | `a` | `φ` | `φ` | `φ` | | | | | |
| | python | `a` | | | | | | | `(1, 1, 1)` | |
| `every-symbol` | latex | `a` | `\sigma` | `\mu` | `\nu` | | | | | every symbol at once, by indices free to run over both coordinates |
| | text | `a` | `sigma` | `mu` | `nu` | | | | | |
| | unicode | `a` | `σ` | `μ` | `ν` | | | | | |
| | python | `a` | | | | | | | `None` | |
| `radius-R` | latex | `R` | `\theta` | `\phi` | `\phi` | `0` | `1` | `1` | | a radius named R, which names the Ricci scalar where no metric is written with it |
| | text | `R` | `theta` | `phi` | `phi` | | | | | |
| | unicode | `R` | `θ` | `φ` | `φ` | | | | | |
| | python | `R` | | | | | | | `(0, 1, 1)` | |

## Oracle

The symbols from their definition, \(\Gamma^\sigma{}_{\mu\nu} = \frac{1}{2} g^{\sigma\rho}(\partial_\mu g_{\nu\rho} + \partial_\nu g_{\rho\mu} - \partial_\rho g_{\mu\nu})\),
with the metric's components; `component` is the symbol asked for, or `None` for all.

```python
coordinates = [theta, phi]
metric = diag(radius**2, radius**2 * sin(theta) ** 2)
inverse = metric.inv()


def christoffel(s, m, n):
    terms = [
        inverse[s, k]
        * (
            diff(metric[n, k], coordinates[m])
            + diff(metric[k, m], coordinates[n])
            - diff(metric[m, n], coordinates[k])
        )
        for k in range(2)
    ]
    return simplify(sum(terms) / 2)


if component is None:
    expected = [[[christoffel(s, m, n) for n in range(2)] for m in range(2)] for s in range(2)]
else:
    expected = christoffel(*component)
```

## Second route

The symbols as Carroll (1997), eq. (3.101), gives them.

```python
given = [
    [[0, 0], [0, -sin(theta) * cos(theta)]],
    [[0, cot(theta)], [cot(theta), 0]],
]
if component is None:
    pairs = [
        (expected[s][m][n], given[s][m][n]) for s in range(2) for m in range(2) for n in range(2)
    ]
else:
    s, m, n = component
    pairs = [(expected, given[s][m][n])]
assert all(simplify(mine - theirs) == 0 for mine, theirs in pairs)
```

## Forms

### The line element, the symbol by the coordinates' names

**Form** · `line-element` · Carroll (1997), eq. (3.100): `ds^2 = a^2(d\theta^2 + \sin^2\theta ~ d\phi^2)`, and eq. (3.101): `\Gamma^\theta_{\phi\phi} = -\sin\theta \cos\theta`

```latex
ds^2 = «radius»^2(d\theta^2 + \sin^2\theta ~ d\phi^2)
\Gamma^«up»_{«first»«second»}
```

### The solid angle

**Form** · `solid-angle` · docs/reference.md, Built-in tensors: in a line element, `d\Omega^2` is `d\theta^2 + \sin^2\theta\, d\phi^2`

```latex
ds^2 = «radius»^2 \, d\Omega^2
\Gamma^«up»_{«first»«second»}
```

### The metric's components, as diag

**Form** · `diag` · docs/tensors.md, Coordinates and the metric: give the coordinates, then the metric's components, with `diag(...)`

```latex
Coordinates := [\theta, \phi]
g_{\mu\nu} := diag(«radius»^2, «radius»^2 \sin^2\theta)
\Gamma^«up»_{«first»«second»}
```

### The metric's components, as a matrix

**Form** · `pmatrix` · docs/tensors.md, Coordinates and the metric: the components as a `pmatrix`

```latex
Coordinates := [\theta, \phi]
g_{\mu\nu} := \begin{pmatrix} «radius»^2 & 0 \\ 0 & «radius»^2 \sin^2\theta \end{pmatrix}
\Gamma^«up»_{«first»«second»}
```

### Indices staggered

**Form** · `staggered` · docs/reference.md, Built-in tensors: the Christoffel symbols, `\Gamma^a{}_{bc}`

```latex
ds^2 = «radius»^2(d\theta^2 + \sin^2\theta ~ d\phi^2)
\Gamma^{«up»}{}_{«first»«second»}
```

### Indices in braces

**Form** · `braces` · docs/tensors.md, The built-in tensors: `\Gamma^{a}_{bc}`

```latex
ds^2 = «radius»^2(d\theta^2 + \sin^2\theta ~ d\phi^2)
\Gamma^{«up»}_{«first»«second»}
```

### The letter C

**Form** · `C` · docs/tensors.md, The built-in tensors: `C^{a}_{bc}`

```latex
ds^2 = «radius»^2(d\theta^2 + \sin^2\theta ~ d\phi^2)
C^{«up»}_{«first»«second»}
```

### By the coordinates' numbers

**Form** · `by-number` · Carroll (1997), eq. (8.12): symbols by the numbers of the coordinates, counted from 0, as `\Gamma^2_{33} = -\sin\theta\,\cos\theta`; docs/tensors.md, Components

```latex
ds^2 = «radius»^2(d\theta^2 + \sin^2\theta ~ d\phi^2)
\Gamma^«up-n»_{«first-n»«second-n»}
```

### By number, each index named

**Form** · `by-number-equals` · docs/tensors.md, Components: `\Gamma^{a=1}_{b=2 c=2}`

```latex
ds^2 = «radius»^2(d\theta^2 + \sin^2\theta ~ d\phi^2)
\Gamma^{a=«up-n»}_{b=«first-n» c=«second-n»}
```

### By number, after a colon

**Form** · `by-number-colon` · docs/tensors.md, Components: `C^{a:1}_{b:2 c:2}`

```latex
ds^2 = «radius»^2(d\theta^2 + \sin^2\theta ~ d\phi^2)
C^{a:«up-n»}_{b:«first-n» c:«second-n»}
```

### By the definition

**Form** · `definition` · Carroll (1997), eq. (3.21): `\Gamma^\sigma_{\mu\nu} = {1\over 2} g^{\sigma\rho}(\partial_\mu g_{\nu\rho} + \partial_\nu g_{\rho\mu} - \partial_\rho g_{\mu\nu})`, plain TeX's `\over` as Knuth (1984), chapter 17, sets a fraction

```latex
ds^2 = «radius»^2(d\theta^2 + \sin^2\theta ~ d\phi^2)
{1\over 2} g^{«up»\rho}(\partial_{«first»} g_{«second»\rho} + \partial_{«second»} g_{\rho«first»} - \partial_\rho g_{«first»«second»})
```

### In plain text

**Form** · `text` · docs/tensors.md, Index notation: Greek names spelled out, `_{mu nu}`; Coordinates and the metric: `diag(...)`

```text
Coordinates := [theta, phi]
g_{mu nu} := diag(«radius»^2, «radius»^2*sin(theta)^2)
C^{«up»}_{«first» «second»}
```

### The line element in plain text

**Form** · `text-line-element` · docs/tensors.md, Coordinates and the metric: in a line element, `dt` is the differential of the coordinate t; docs/language.md, Names and symbols: a Greek letter spelled out, as `theta`

```text
ds^2 = «radius»^2*(dtheta^2 + sin(theta)^2*dphi^2)
Gamma^{«up»}_{«first» «second»}
```

### In Unicode

**Form** · `unicode` · docs/reference.md, Unicode: a Greek letter is its command, and superscripts are scripts

```unicode
ds² = «radius»²(dθ² + sin²θ dφ²)
Γ^«up»_{«first»«second»}
```

### The metric's components in Unicode

**Form** · `unicode-diag` · docs/reference.md, Unicode: a Greek letter is its command, and superscripts are scripts

```unicode
Coordinates := [θ, φ]
g_{μν} := diag(«radius»², «radius»² sin²θ)
Γ^«up»_{«first»«second»}
```
