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

## Index notation

Indices go up with `^` and down with `_`. Inside braces each letter is its own
index, as in LaTeX, while Greek names stay whole: `_{ab}` is \(a, b\) and
`_{\mu\nu}` (or `_{mu nu}`) is \(\mu, \nu\). The plain-text style `_{a}_{b}` works
too.

An index repeated once up and once down is summed over (Einstein's
convention); the others are the result's free indices:

```python
>>> wb.expr(r"g_{ab} g^{bc}")
Tensor('_{a}^{c}', [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
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
sphere, say), in which case `R` stays a symbol; `Ric` is always the Ricci
scalar. Likewise `g` is the metric determinant unless the metric uses `g`, when
`\det(g_{\mu\nu})` still is. To rename a built-in, write `MetricSymbol := G`; likewise
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
[True, TensorEquation(_{mu}_{nu}, 0 equations)]

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
[M*(-2*M + r)/r**3, 0, 0, 0]

```

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

`:=` always (re)defines. `=` defines a tensor the first time, but once a tensor
exists `=` states an equation about it rather than overwriting it. An equation
about a component, such as `g_{tt} = -(1 - 2M/r)`, whose indices name
coordinates, is always a claim: it neither replaces the metric nor defines a new
tensor's every component.

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

## Symmetrisation

Round brackets symmetrise and square brackets antisymmetrise, as in
`T_{(ab)}` and `F_{[\mu\nu]}`. A bracket takes the indices at the level it opens
at: `T_{(a}{}^{b}{}_{c)}` symmetrises \(a\) and \(c\), and leaves \(b\) alone. The
brackets may span several tensors, which is how the second Bianchi identity is
written:

```python
>>> wb.expr(r"\nabla_{[\lambda} R_{\rho\sigma]\mu\nu} = 0")
TensorEquation(_{lambda}_{rho}_{sigma}_{mu}_{nu}, 0 equations)

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
`g_{\mu\nu} := Schwarzschild(M)`. Parameters are symbols or numbers; units are
geometric (\(G = c = 1\)).

| Function | Spacetime |
|---|---|
| `Minkowski()` | flat spacetime, \((t, x, y, z)\) |
| `Schwarzschild(M)` | static black hole |
| `ReissnerNordstrom(M, Q)` | charged black hole |
| `Kerr(M, a)` | rotating black hole, Boyer–Lindquist coordinates |
| `KerrNewman(M, a, Q)` | rotating, charged black hole |
| `FLRW(k)` | homogeneous, isotropic universe with scale factor \(a(t)\) |
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
integrate them numerically see [Numerics](numerics.md).

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
Indices('_{mu}')

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
