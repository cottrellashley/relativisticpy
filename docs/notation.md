# Solving tensor equations

Components answer *what is this tensor here?* Index notation answers *what does
this equation say?* RelativisticPy works at both levels. An equation with a
tensor that has no components yet, such as the energy–momentum tensor
\(T_{\mu\nu}\), is kept as an equation between tensors. It can be rearranged,
traced, simplified and solved as on paper, and turned into components later,
once there is a metric.

## Einstein's equations, solved

```python
>>> from relativisticpy import Workbook
>>> wb = Workbook()
>>> wb.exe(r"EFE := G_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G T_{\mu\nu}")
[]
>>> wb.expr("EFE")
IndexEquation(G_{\mu\nu} + \Lambda g_{\mu\nu} = 8 \pi G T_{\mu\nu})
>>> wb.expr(r"solve(EFE, T_{\mu\nu})")
IndexEquation(T_{\mu\nu} = \frac{G_{\mu\nu} + \Lambda g_{\mu\nu}}{8 \pi G})
>>> wb.expr(r"solve(EFE, T^{\mu\nu})")
IndexEquation(T^{\mu\nu} = \frac{G^{\mu\nu} + \Lambda g^{\mu\nu}}{8 \pi G})

```

Solving for the Ricci tensor takes more than moving terms across, because the
Einstein tensor \(G_{\mu\nu} = R_{\mu\nu} - \frac{1}{2} R g_{\mu\nu}\) contains
its trace, \(R\). `solve` traces the equation, solves the trace for \(R\) and
substitutes it back. This is trace reversal:

```python
>>> wb.expr("trace(EFE)")
IndexEquation(4 \Lambda - R = 8 \pi G T)
>>> wb.expr(r"solve(EFE, R_{\mu\nu})")
IndexEquation(R_{\mu\nu} = 8 \pi G T_{\mu\nu} + \left(\Lambda - 4 \pi G T\right) g_{\mu\nu})

```

As in papers, \(T\) without indices is the trace \(T^\mu{}_\mu\). Equations
compare by what they say, so the solution equals the form in textbooks:

```python
>>> wb.expr(r"solve(EFE, R_{\mu\nu})") == wb.expr(
...     r"R_{\mu\nu} = \Lambda g_{\mu\nu} + 8\pi G \left(T_{\mu\nu} - \frac{1}{2} T g_{\mu\nu}\right)")
True

```

## The operations

| | |
|---|---|
| `solve(E, T_{\mu\nu})` | solve an equation for a tensor, with its indices up or down. The equation must be linear in the tensor; its trace may appear too |
| `trace(E)` | contract the two free indices of an equation, or an expression, with the metric |
| `simplify(E)` | the canonical form, under the tensors' symmetries and the identities below |
| `subs(E, rule)` | apply a rule: a definition, a contraction, or a value for a scalar such as `\Lambda = 0` |
| `LHS(E)`, `RHS(E)` | one side of an equation |

A rule given to `subs` applies whatever its indices are called, and in either
position. It may be a definition, such as that of the Einstein tensor:

```python
>>> wb.expr(r"subs(EFE, G_{ab} = R_{ab} - \frac{1}{2} R g_{ab})")
IndexEquation(R_{\mu\nu} + \left(\Lambda - \frac{1}{2} R\right) g_{\mu\nu} = 8 \pi G T_{\mu\nu})

```

A rule for a scalar puts its value into the coefficients:

```python
>>> wb.expr(r"subs(EFE, \Lambda = 0)")
IndexEquation(G_{\mu\nu} = 8 \pi G T_{\mu\nu})

```

It may also be a contraction. Here is the energy density that an observer moving
with a perfect fluid measures:

```python
>>> fluid = r"((\rho + p) u^\mu u^\nu + p g^{\mu\nu}) u_\nu"
>>> wb.expr(rf"subs({fluid}, u^\alpha u_\alpha = -1)")
IndexExpression(- \rho u^{\mu})

```

## What simplify knows

Canonical forms come from SymPy's implementation of the Butler–Portugal
algorithm. It knows each tensor's symmetries and renames dummy indices, so
\(T_{\mu\alpha} u^\alpha\) and \(T_{\mu\beta} u^\beta\) are the same term.
RelativisticPy adds the identities of General Relativity:

- the symmetries of the Riemann tensor, and the first Bianchi identity,
  \(R_{\rho[\sigma\mu\nu]} = 0\);
- its contractions: \(R^\alpha{}_{\mu\alpha\nu} = R_{\mu\nu}\),
  \(R^\mu{}_\mu = R\) and \(G^\mu{}_\mu = (1 - n/2) R\) in \(n\) dimensions;
- the Weyl tensor is traceless; \(g^\mu{}_\mu = n\), and the metric with one
  index up is \(\delta^\mu{}_\nu\);
- the trace of a tensor \(T_{\mu\nu}\) of your own is written \(T\).

```python
>>> wb.expr(r"simplify(R_{\rho\sigma\mu\nu} + R_{\rho\mu\nu\sigma} + R_{\rho\nu\sigma\mu})")
IndexExpression(0)
>>> wb.expr(r"simplify(R^{\alpha}{}_{\mu\alpha\nu} - R_{\nu\mu})")
IndexExpression(0)
>>> wb.expr(r"simplify(G^\mu{}_\mu)")
IndexExpression(- R)

```

## Identities

Without a metric there are no components, so an equation between tensors can
only be notation. `holds()` says whether it is an identity:

```python
>>> identities = Workbook()
>>> identities.expr(r"G_{\mu\nu} = R_{\mu\nu} - \frac{1}{2} R g_{\mu\nu}").holds()
True
>>> identities.expr(r"g^{\mu\nu} g_{\nu\sigma} = \delta^\mu_\sigma").holds()
True
>>> identities.expr(r"R_{\mu\nu} = 0").holds()
False

```

The last is a field equation, not an identity. When a paper is
[checked](checking.md), identities in index notation are verified this way.

Index notation has rules: an index appears once in a term, or twice to be
summed, once up and once down; and every term has the same free indices.
RelativisticPy checks them before anything is computed:

```python
>>> wb.expr(r"G_{\mu\nu} = 8\pi T_{\mu\nu\rho}")
Traceback (most recent call last):
  ...
relativisticpy.errors.IndexNotationError: the two sides have different free indices: _{\mu\nu} on the left, _{\mu\nu\rho} on the right
 --> <input>:1:1
  |
1 | G_{\mu\nu} = 8\pi T_{\mu\nu\rho}
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

```

An equation that breaks the rules may still be about components. Papers write
\(\Gamma^\theta_{r\theta} = 1/r\), with \(r\) and \(\theta\) naming
coordinates, so such an equation is read as a statement about components, which
needs a metric. So is any equation whose indices are the coordinates in force.

## From notation to components

Notation turns into components where it meets them. With a metric in force,
defining a tensor by notation gives its components. Here the field equations,
solved for \(T_{\mu\nu}\) in an FLRW universe, give its energy density:

```python
>>> cosmos = Workbook()
>>> cosmos.exe(r"""
...     ds^2 = -dt^2 + a(t)^2 \left(\frac{dr^2}{1 - k r^2} + r^2 d\Omega^2\right)
...     EFE := G_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G T_{\mu\nu}
...     T_{\mu\nu} := RHS(solve(EFE, T_{\mu\nu}))
... """)
[]
>>> cosmos.expr(r"T_{tt}")
-Lambda/(8*pi*G) + 3*(k + Derivative(a(t), t)**2)/(8*pi*G*a(t)**2)

```

That is the first Friedmann equation, \(8\pi G \rho = 3(\dot a^2 + k)/a^2 -
\Lambda\). The trace-reversed equations, solved as notation, then hold
component by component:

```python
>>> cosmos.expr(r"R_{\mu\nu} = RHS(solve(EFE, R_{\mu\nu}))")
TensorEquation(_{mu}_{nu}, 0 equations)

```

## Not yet

Derivatives are not yet part of the notation. \(\nabla_\mu T^{\mu\nu} = 0\)
works on components, with a metric, but cannot yet be rearranged as notation.
Nor can a tensor be defined abstractly, as
\(T_{\mu\nu} := (\rho + p) u_\mu u_\nu + p g_{\mu\nu}\) with a velocity \(u\)
that has no components. Both are on the [roadmap](roadmap.md).
