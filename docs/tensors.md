# Tensors and geometry

```python
>>> from relativisticpy import Workbook
>>> wb = Workbook()

```

## Coordinates and the metric

Index notation needs coordinates; geometry needs a metric. Give the
coordinates, then the metric's components with both indices down:

```python
>>> wb.expr(r"""
...     Coordinates := [t, r, \theta, \phi]
...     g_{\mu\nu} := \begin{pmatrix}
...         -(1 - \frac{2M}{r}) & 0 & 0 & 0 \\
...         0 & \frac{1}{1 - \frac{2M}{r}} & 0 & 0 \\
...         0 & 0 & r^2 & 0 \\
...         0 & 0 & 0 & r^2 \sin^2\theta
...     \end{pmatrix}
... """)

```

A nested list works as well as a matrix, and so does `diag(...)`. A metric given
with both indices *up*, `g^{\mu\nu} := ...`, is inverted for you. Or write the
line element:

```python
>>> wb.expr(r"ds^2 = -(1 - \frac{2M}{r}) dt^2 + \frac{dr^2}{1 - \frac{2M}{r}} + r^2 d\Omega^2")

```

In a line element, `dt` or `d t` is the differential of the coordinate `t`, and
`d\Omega^2` is the solid angle element \(d\theta^2 + \sin^2\theta\, d\phi^2\).
A [spacetime](#spacetimes) brings its own coordinates: `g_{\mu\nu} := Kerr(M, a)`.

Without declared coordinates, a line element names them, in the order their
differentials first appear: `ds^2 = r^2 d\Omega^2 - dt^2 + dr^2` is in
\((\theta, \phi, t, r)\), so `g_{00}` is \(g_{\theta\theta}\). Declare
`Coordinates := [...]` first to fix the order. Written alone, `ds^2` is the line
element of the metric in force:

```python
>>> sphere = Workbook()
>>> sphere.exe(r"ds^2 = r^2 d\Omega^2 - dt^2 + dr^2" "\n" "Coordinates" "\n" "ds^2")
[[theta, phi, t, r], d\phi**2*r**2*sin(theta)**2 + d\theta**2*r**2 + dr**2 - dt**2]

```

A metric is checked where it is defined. One with the wrong number of components,
one that is not symmetric, or one whose determinant is zero, as when a line element
leaves out a coordinate's differential, is refused there, with the reason; the
metric in force, if there is one, stays:

```python
>>> flat = Workbook()
>>> flat.exe(r"Coordinates := [t, r]" "\n" r"ds^2 = -dt^2")
Traceback (most recent call last):
  ...
relativisticpy.errors.EvaluationError: the metric is degenerate: its determinant is zero
 --> <input>:2:1
  |
2 | ds^2 = -dt^2
  | ^^^^^^^^^^^^
note: no term of it has dr
hint: give it a term with dr

```

`ds^2 = 0`, which says that a path is null, is a condition, read but not computed:
it defines no metric.

### Names defined after the metric

Papers often write a metric first and say what it is written with afterwards:
"\(ds^2 = -\Delta\, dt^2 + \Delta^{-1} dr^2 + r^2 d\Omega^2\), where
\(\Delta = 1 - 2M/r\)". A name or function the metric in force is written with, and
defined after it, takes that value in the metric from then on, so curvature computed
afterwards is that of the metric the paper means:

```python
>>> rn = Workbook()
>>> rn.exe(r"""
...     ds^2 = -\Delta dt^2 + \Delta^{-1} dr^2 + r^2 d\Omega^2
...     R_{abcd} R^{abcd}
...     \Delta = 1 - \frac{2M}{r}
...     R_{abcd} R^{abcd}
... """)
[(4*Delta**2 - 8*Delta + 4)/r**4, 48*M**2/r**6]

```

The same holds for a function, as \(f\) in `ds^2 = -f(r) dt^2 + \frac{dr^2}{f(r)} + ...`
followed by `f(r) = 1 - \frac{2M}{r}`, and for a definition written with `:=`. What
was computed before the definition stays as it was.

## Index notation

Indices go up with `^` and down with `_`. Inside braces each letter is its own
index, as in LaTeX, while Greek names stay whole: `_{ab}` is \(a, b\) and
`_{\mu\nu}` (or `_{mu nu}`) is \(\mu, \nu\). The plain-text style `_{a}_{b}` works
too.

An index repeated once up and once down is summed over (Einstein's
convention); the others are the result's free indices:

```python
>>> wb.expr(r"g_{ab} g^{bc}")
g_{ab} g^{bc} in (t, r, θ, φ): 4 of 16 components non-zero
  g_{tb} g^{bt} = 1
  g_{rb} g^{br} = 1
  g_{θb} g^{bθ} = 1
  g_{φb} g^{bφ} = 1
>>> wb.expr(r"g_{ab} g^{ab}")
4

```

Indices are raised and lowered with the metric automatically: ask for any
tensor with its indices wherever you need them. Before anything is computed,
the rules of index notation are checked, and a mistake is reported where it was
written:

```python
>>> wb.expr(r"R_{ab} R_{ab}")
Traceback (most recent call last):
  ...
relativisticpy.errors.IndexNotationError: index a appears twice as a lower index; a summed index must appear once up and once down
 --> <input>:1:1
  |
1 | R_{ab} R_{ab}
  | ^^^^^^^^^^^^^

```

## The built-in tensors

Once a metric is defined, these are available by name, recognised by name *and*
number of indices:

| Tensor | Written | Indices |
|---|---|---|
| metric | `g_{ab}`, `g^{ab}` | 2 |
| Christoffel symbols | `C^{a}_{bc}`, `\Gamma^{a}_{bc}` | 3 |
| Riemann tensor | `R^{a}_{bcd}` | 4 |
| Ricci tensor | `Ric_{ab}`, `R_{ab}` | 2 |
| Einstein tensor | `G_{ab}` | 2 |
| Weyl tensor | `W_{abcd}`, `C_{abcd}` | 4 |
| Kronecker delta | `\delta^{a}_{b}` | 2 |
| metric determinant | `g` | 0 |
| Ricci scalar | `Ric`, `R` | 0 |
| Kretschmann scalar | `Kretschmann` | 0 |

The conventions are those of Misner, Thorne & Wheeler and of Carroll:

$$
\Gamma^a{}_{bc} = \tfrac12 g^{ad}\left(\partial_b g_{dc} + \partial_c g_{db} - \partial_d g_{bc}\right)
$$

$$
R^a{}_{bcd} = \partial_c \Gamma^a{}_{db} - \partial_d \Gamma^a{}_{cb}
            + \Gamma^a{}_{ce}\Gamma^e{}_{db} - \Gamma^a{}_{de}\Gamma^e{}_{cb}
$$

$$
R_{bd} = R^a{}_{bad}, \qquad R = g^{bd} R_{bd}, \qquad G_{ab} = R_{ab} - \tfrac12 R\, g_{ab}
$$

```python
>>> wb.exe("R\nKretschmann\nG_{tt}")
[0, 48*M**2/r**6, 0]

```

`R` is the Ricci scalar unless your metric uses `R` as a symbol (the radius of a
sphere, say), in which case `R` stays a symbol, and a note says so wherever it is
read that way; `Ric` is always the Ricci scalar. Likewise `g` is the metric
determinant unless the metric uses `g`, when `\det(g_{\mu\nu})` still is. Spelled
out, as `RicciScalar` or `Riemann`, a built-in is no name of the language: it is a
symbol, with a note saying what to write. To rename a built-in, write
`MetricSymbol := G`; likewise
`ConnectionSymbol`, `RiemannSymbol`, `RicciSymbol`, `EinsteinSymbol`,
`WeylSymbol`, `DerivativeSymbol` and `CovariantDerivativeSymbol`.

With a metric in force, an equation about a built-in is a claim to check, never
a new definition: `R = 0` asks whether the Ricci scalar vanishes, and
`R_{\mu\nu} \equiv R^\alpha{}_{\mu\alpha\nu}`, as papers define the Ricci tensor,
checks that it is the contraction of the Riemann tensor:

```python
>>> wb.exe(r"""
...     R = 0
...     R_{\mu\nu} \equiv R^\alpha{}_{\mu\alpha\nu}
... """)
[True, R_{μν} = R^α{}_{μαν}: holds in all 10 components]

```

To give a built-in's name a meaning of your own, use `:=`.

## Components

An index can be fixed to a component, by number or by coordinate name. Writing a
coordinate's name as an index does the same:

```python
>>> wb.exe(r"""
...     \Gamma^{r}_{\theta\theta}
...     \Gamma^{a=1}_{b=2 c=2}
...     C^{a:1}_{b:2 c:2}
...     g_{01}
... """)
[2*M - r, 2*M - r, 2*M - r, 0]

```

Fixing some indices and not others gives the rest as a tensor:

```python
>>> print(wb.expr(r"\Gamma^{r}_{t b}"))
Γ^r_{t b} in (t, r, θ, φ): 1 of 4 components non-zero
  Γ^r_{t t} = M (r − 2M)/r³

```

A sum with whole-number limits is carried out, component by component for a
tensor, so `\sum_{n=0}^{2} R_{\mu\nu}` is zero here; one to infinity, or to a
symbol, is kept as written, and `doit(...)` carries it out where SymPy can.

## Defining tensors

Define a tensor from its components, or from an expression in other tensors.
The free indices on the right must match the left, in any order:

```python
>>> wb.expr(r"""
...     u^{\mu} := [1/\sqrt{1 - 2M/r}, 0, 0, 0]
...     u_{\mu} u^{\mu}
... """)
-1
>>> wb.expr(r"""
...     T^{\mu\nu} := (\rho + p) u^\mu u^\nu + p g^{\mu\nu}
...     T^{\mu}{}_{\mu}
... """)
3*p - rho

```

## What `=` means

Papers use `=` both to define and to claim, so RelativisticPy decides by what the
left side already means. `:=` (or `\equiv`, `\coloneqq`) always defines, or
redefines. `=` defines only what has no meaning yet:

| You write | When | It means |
|---|---|---|
| `x = 2`, `f(x) = x^2`, `T_{\mu\nu} = \ldots` | the name has no meaning yet | a definition |
| `ds^2 = \ldots` | always | the metric, from its line element |
| `\Delta = \ldots` | the metric in force is written with \(\Delta\) | a definition, which the metric takes ([above](#names-defined-after-the-metric)) |
| `K = \ldots`, `u^\mu = \ldots`, `f(x) = \ldots` | it already has a value | a claim, checked |
| `R = 0`, `R_{\mu\nu} = 0`, `g_{\mu\nu} = \ldots` | it is built in, or the metric in force | a claim, checked |
| `g_{tt} = \ldots`, `f(2) = 4` | a component, or a value | a claim, checked |
| `\pi = 3.14159` | a constant | a claim, checked |
| `R = \frac{2}{R^2}` | the name is in its own value | an equation, never a definition |
| `t = 0`, `ds^2 = 0`, `(k = 0)` | a coordinate in force, a null path, a case in brackets | a condition, read but not computed |

A claim where a definition could have been meant says so, in a note on its step
(see [Python](python.md)):

```python
>>> claims = Workbook()
>>> claims.exe("K := 48\nK = 47")
[False]
>>> [step.notes for step in claims.each("K = 48")]
[['K already has a value, so = compares it; to give K a new value, write K := ...']]
>>> claims.exe("K := 47\nK")
[47]

```

So retyping a corrected tensor, `u^\mu = [2, 0]` after `u^\mu = [1, 0]`, compares
the two rather than overwriting the first; an equation about a component, such as
`g_{tt} = -(1 - 2M/r)`, neither replaces the metric nor defines a new tensor's every
component; and `R = 0` written before any metric is the equation \(R = 0\) of
[index notation](notation.md), not a new meaning for \(R\).

A letter followed by brackets could be a function or a product. A constant, or a
name the statement also writes as a factor, as \(G\) is in
`\frac{2GM}{r} + G(p^2 + q^2)`, where in LaTeX \(2GM\) is \(2 \cdot G \cdot M\),
multiplies what is in the brackets; so does a name written as a factor before, where
the brackets hold more than names, as \(\rho\) does in `\rho(1 + 3w)` after
`p = w\rho`. Otherwise a letter that means nothing is an unknown function: of
names, as `\alpha(t, r)`, without a word; of anything else, with a note saying how
to write the product:

```python
>>> claims.expr(r"\frac{2GM}{r} + G(p^2 + q^2)")
2*G*M/r + G*(p**2 + q**2)
>>> [step.notes for step in claims.each(r"\gamma(t - v x)")]
[['\\gamma is not defined, so \\gamma(t - v x) is an unknown function applied to t - v x; for a product, write \\gamma \\cdot (t - v x)']]

```

## Derivatives

`\partial_\mu` (or `d_{\mu}`) is the partial derivative and `\nabla_\mu` (or
`D_{\mu}`) the covariant derivative. Each acts on the factor written after it,
so \(\partial_\mu \phi\, \partial^\mu \phi\) means
\((\partial_\mu\phi)(\partial^\mu\phi)\); use brackets for more. The comma and
semicolon notations mean the same: `T^{\mu\nu}{}_{;\nu}` is \(\nabla_\nu T^{\mu\nu}\).
They apply to a scalar too, one you define, a built-in one such as `R`, or a
function of the coordinates: `\phi_{,\mu}` is \(\partial_\mu \phi\) and
`\phi_{;\mu}{}^{;\mu}` is \(\Box\phi\). A name with no value has no derivative, so
that is an error.

```python
>>> [tensor.nonzero() for tensor in wb.exe(r"""
...     \nabla_\lambda g_{\mu\nu}
...     \nabla_\mu G^{\mu\nu}
... """)]
[{}, {}]

```

(No nonzero components: the metric is covariantly constant, and the Einstein
tensor is conserved.)

Components are taken after the derivative: \(\nabla_\mu T^{\mu t}\) is the \(t\)
component of \(\nabla_\mu T^{\mu\nu}\), which involves every component of
\(T\). A component chosen inside the operand, as in `\nabla_\mu (A^{\mu} B^{t})`,
is an error rather than a guess; define the operand as a tensor first.

A Leibniz derivative such as `\frac{d}{dx}` acts on the rest of the term:
`\frac{d}{dx} x^2 y` is \(\frac{d}{dx}(x^2 y)\).

A number on `\partial` or `\nabla` is the order of the derivative, never an
index: `\partial_t^2` and `\partial^2_t` are \(\partial_t \partial_t\). With no
index, `\partial^2` is \(\partial^\mu \partial_\mu\), and `\nabla^2` is
\(\nabla^\mu \nabla_\mu\), the same as `\Box`: the Laplacian of the metric in
force, which in spacetime is the d'Alembertian. (Some texts write
\(\nabla^2\) for the Laplacian of space alone, as in \(\Box = -\partial_t^2 +
\nabla^2\); RelativisticPy's is that of all the coordinates.)

```python
>>> flat = Workbook()
>>> flat.exe(r"""
...     ds^2 = -dt^2 + dx^2 + dy^2 + dz^2
...     \partial^2_t (t^3) = \partial_t \partial_t (t^3)
...     \nabla^2 (x^2 + y^2 + z^2 - t^2)
... """)
[True, 8]

```

On a tensor with one index, a number written bare above it, as in `k^3`, could be
a component or a power, so it is refused; in braces, `k^{3}` is the component,
and `k_{a} k^{a}` is \(k\) contracted with itself. (`k^0` and `k^1` are
components: neither is a power anyone writes.)

```python
>>> flat.exe(r"""
...     k^\mu := [\omega, 0, 0, \omega]
...     k^{3}
...     k_{a} k^{a}
... """)
[omega, 0]
>>> flat.exe("k^3")
Traceback (most recent call last):
    ...
relativisticpy.errors.IndexNotationError: k is a tensor, so k^3 could be its component with upper index 3 or a power of it
...

```

## Symmetrisation

Round brackets symmetrise and square brackets antisymmetrise, as in
`T_{(ab)}` and `F_{[\mu\nu]}`. A bracket takes the indices at the level it opens
at: `T_{(a}{}^{b}{}_{c)}` symmetrises \(a\) and \(c\), and leaves \(b\) alone. The
brackets may span several tensors, which is how the second Bianchi identity is
written:

```python
>>> wb.expr(r"\nabla_{[\lambda} R_{\rho\sigma]\mu\nu} = 0")
∇_{[λ} R_{ρσ]μν} = 0: holds in all 200 components

```

## Tensor equations

With a metric in force, an equation between tensors stands for one equation per
component. Those that do not hold identically are listed, ready to solve. (To
rearrange and solve an equation between tensors as an equation, before there
are components, see [Solving tensor equations](notation.md).) Here are the vacuum field
equations for a general static, spherically symmetric metric:

```python
>>> fresh = Workbook()
>>> equations = fresh.expr(r"""
...     Coordinates := [t, r, \theta, \phi]
...     g_{\mu\nu} := diag(-e^{2 \alpha(r)}, e^{2 \beta(r)}, r^2, r^2 \sin^2\theta)
...     R_{\mu\nu} = 0
... """)
>>> len(equations.equations())
4
>>> fresh.expr(r"dsolve(R_{tt} + e^{2 \alpha(r) - 2 \beta(r)} R_{rr} = 0, \beta(r))")
Eq(beta(r), C1 - alpha(r))

```

## Spacetimes

Each of these is a function returning a metric, used as
`g_{\mu\nu} := Schwarzschild(M)`. Parameters are symbols or numbers, and a
decimal is the exact fraction it denotes, so `Schwarzschild(1.7)` is exactly a
vacuum. Units are geometric (\(G = c = 1\)) unless `G` and `c` follow a
spacetime's own parameters, as in `Schwarzschild(M, G, c)`. A parameter left out
is the symbol the table names it with: `Kerr(M)` is `Kerr(M, a)`, and
`Schwarzschild()` is `Schwarzschild(M)`. The scale factor of `FLRW(k, a)` may be
named, and is then a function of \(t\), or given as an expression of \(t\). A name
that is not a spacetime is refused where the metric is defined, with the closest
spacetime's name:

```python
>>> Workbook().exe(r"g_{\mu\nu} := Schwarzchild(M)")
Traceback (most recent call last):
  ...
relativisticpy.errors.CompileError: unknown spacetime Schwarzchild
 --> <input>:1:15
  |
1 | g_{\mu\nu} := Schwarzchild(M)
  |               ^^^^^^^^^^^^
note: the spacetimes are Minkowski, Schwarzschild, ReissnerNordstrom, Kerr, KerrNewman, FLRW, DeSitter and AntiDeSitter
hint: did you mean Schwarzschild(M)?

```

| Function | Spacetime |
|---|---|
| `Minkowski()` | flat spacetime, \((t, x, y, z)\) |
| `Schwarzschild(M)` | static black hole |
| `ReissnerNordstrom(M, Q)` | charged black hole |
| `Kerr(M, a)` | rotating black hole, Boyer–Lindquist coordinates |
| `KerrNewman(M, a, Q)` | rotating, charged black hole |
| `FLRW(k, a)` | homogeneous, isotropic universe with scale factor \(a(t)\) |
| `DeSitter(L)`, `AntiDeSitter(L)` | constant curvature, radius \(L\) |

```python
>>> cosmos = Workbook()
>>> cosmos.expr(r"g_{\mu\nu} := FLRW(k)" "\n" r"G^{t}{}_{t}")
-3*(k + Derivative(a(t), t)**2)/a(t)**2

```

That is the first Friedmann equation, \(3(\dot a^2 + k)/a^2 = 8\pi G \rho\),
waiting for its right-hand side.

## Geodesics

`geodesic_equations(\tau)` gives the geodesic equations for the metric in
force, with each coordinate a function of the affine parameter \(\tau\). To
integrate them numerically, which is done from Python, with
`relativisticpy.numeric.geodesic`, see [Numerics](numerics.md); `geodesic(...)` in
the language says so.

```python
>>> wb.expr(r"geodesic_equations(\tau)")[2]
Eq(-sin(theta(tau))*cos(theta(tau))*Derivative(phi(tau), tau)**2 + Derivative(theta(tau), (tau, 2)) + 2*Derivative(r(tau), tau)*Derivative(theta(tau), tau)/r(tau), 0)

```

The same equations come from a Lagrangian. A derivative with respect to a
vector, \(\partial L / \partial \dot x^\mu\), has an index of its own:

```python
>>> lagrangian = Workbook()
>>> euler_lagrange = lagrangian.expr(r"""
...     g_{\mu\nu} := Schwarzschild(M)
...     x^{\mu} := [t(\tau), r(\tau), \theta(\tau), \phi(\tau)]
...     L := g_{\mu\nu} \dot{x}^\mu \dot{x}^\nu |_{t=t(\tau), r=r(\tau), \theta=\theta(\tau), \phi=\phi(\tau)}
...     \frac{d}{d\tau} \frac{\partial L}{\partial \dot{x}^\mu} - \frac{\partial L}{\partial x^\mu}
... """)
>>> euler_lagrange.indices
Indices('_{\\mu}')

```

## Performance

Geometric quantities are computed once per metric and cached, using their
symmetries (Christoffel symbols for \(b \le c\), Riemann components for
\(c < d\)). Simplifying components is the expensive part, and
`Workbook(workers=N)` (or `relativisticpy -j N`) spreads it over processes.
Results that vanish are recognised exactly by algebra alone (see
[How it works](architecture.md#simplification)), which is why checking that the
Kerr metric solves the vacuum equations takes seconds.

Some times, measured on an eight-core laptop processor (AMD Ryzen 7 8745HS):

| | One worker | Eight workers |
|---|---|---|
| A sum of 1,000 fractions, and its derivative | 0.4 s, 1.1 s | |
| Christoffel symbols to the Kretschmann scalar, for Schwarzschild, Reissner–Nordström, FLRW or a general spherically symmetric metric | 0.2 to 0.6 s | |
| The Christoffel symbols of Kerr | 3.5 s | 1.6 s |
| `R_{\mu\nu} = 0` for Kerr, a vacuum solution | 14 s | 5.4 s |
| The Ricci tensor of Kerr–Newman | 47 s | 49 s |
| Carroll's lecture notes, checked whole | 4 s | |

Some quantities are far harder. The full Riemann tensor of Kerr, simplified
component by component, takes more than a quarter of an hour even with eight
workers, because a few of its components take that long on their own. Ask for
what you need: the vacuum equations need only the Ricci tensor, which is
computed straight from the connection.
