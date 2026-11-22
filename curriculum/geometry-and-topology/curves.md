# Curves

Curves in the plane and in space, as a course on curves writes them: arc length, the
curvature of a plane curve, and the curvature and torsion of a space curve, from any
parametrisation and from the Frenet–Serret frame. Stewart (2015) is *Calculus: Early
Transcendentals*, 8th edition; O'Neill (2006) is *Elementary Differential Geometry*, revised
2nd edition, whose Frenet formulas have \(\mathbf{B}' = -\tau \mathbf{N}\).

## The length of a catenary

**Level 1** · `geom.curves.catenary-length` · Stewart (2015), §8.1

The arc length \(\int \sqrt{1 + f'(x)^2}\,dx\) of the catenary \(y = \cosh x\) from 0 to 1.
On the real line the integrand is \(\cosh x\), and the length is \(\sinh 1\).

```latex
\int_0^1 \sqrt{1 + \sinh^2 x} \, dx
```

```python
from scipy.integrate import quad

x = symbols("x", real=True)
assert simplify(sqrt(1 + sinh(x) ** 2)) == cosh(x)  # for real x
expected = integrate(cosh(x), (x, 0, 1))
assert expected == sinh(1)
assert abs(quad(lambda t: np.sqrt(1 + np.sinh(t) ** 2), 0, 1)[0] - np.sinh(1)) < 1e-12
```

## The curvature of a helix

**Level 2** · `geom.curves.helix-curvature` · Stewart (2015), §13.3

The curvature of a space curve from any parametrisation,
\(\kappa = |\mathbf{r}' \times \mathbf{r}''| / |\mathbf{r}'|^3\), with the curve a triple of
functions: for the helix of radius \(a\) and pitch \(2\pi b\), \(a / (a^2 + b^2)\).

```latex
a, b > 0
t \in \mathbb{R}
\mathbf{r}(t) := (a \cos t, a \sin t, b t)
\kappa := \frac{\left| \mathbf{r}'(t) \times \mathbf{r}''(t) \right|}{\left| \mathbf{r}'(t) \right|^3}
\kappa
```

```python
t, s = symbols("t s", real=True)
a, b = symbols("a b", positive=True)
r = Matrix([a * cos(t), a * sin(t), b * t])
cross = r.diff(t).cross(r.diff(t, 2))
expected = simplify(
    sqrt(simplify(cross.dot(cross))) / sqrt(simplify(r.diff(t).dot(r.diff(t)))) ** 3
)
assert simplify(expected - a / (a**2 + b**2)) == 0
# By arc length, s = c t with c = sqrt(a^2 + b^2), the curvature is |d^2 r / ds^2|.
c = sqrt(a**2 + b**2)
q = r.subs(t, s / c)
assert simplify(q.diff(s).dot(q.diff(s))) == 1
assert simplify(sqrt(q.diff(s, 2).dot(q.diff(s, 2))) - expected) == 0
```

## The curvature of an ellipse at a vertex

**Level 2** · `geom.curves.ellipse-vertex` · Stewart (2015), §13.3

The curvature of a plane curve \((x(t), y(t))\),
\(|x'y'' - y'x''| / (x'^2 + y'^2)^{3/2}\), with Lagrange's primes on functions the program
defines: at the vertex \((a, 0)\) of the ellipse of semi-axes \(a, b > 0\), it is \(a / b^2\).

```latex
a, b > 0
x(t) := a \cos t
y(t) := b \sin t
\kappa(t) := \frac{\left| x'(t) y''(t) - y'(t) x''(t) \right|}{\left(x'(t)^2 + y'(t)^2\right)^{3/2}}
\kappa(0)
```

```python
t, Y = symbols("t Y", real=True)
a, b = symbols("a b", positive=True)
x, y = a * cos(t), b * sin(t)
numerator = Abs(diff(x, t) * diff(y, t, 2) - diff(y, t) * diff(x, t, 2))
kappa = numerator / (diff(x, t) ** 2 + diff(y, t) ** 2) ** Rational(3, 2)
expected = simplify(kappa.subs(t, 0))
assert expected == a / b**2
# Near (a, 0) the ellipse is the graph X = a sqrt(1 - Y^2/b^2), of curvature |X''|/(1 + X'^2)^(3/2).
X = a * sqrt(1 - Y**2 / b**2)
assert diff(X, Y).subs(Y, 0) == 0
assert Abs(diff(X, Y, 2).subs(Y, 0)) == expected
```

## The torsion of a helix from its Frenet frame

**Level 3** · `geom.curves.frenet-helix` · O'Neill (2006), chapter 2

The helix by arc length, with \(c = \sqrt{a^2 + b^2}\): its unit tangent
\(\mathbf{T} = d\mathbf{r}/ds\), curvature \(\kappa = \|d\mathbf{T}/ds\|\), principal normal
and binormal \(\mathbf{B} = \mathbf{T} \times \mathbf{N}\), and the torsion that the
Frenet–Serret equation \(d\mathbf{B}/ds = -\tau \mathbf{N}\) defines: \(b / (a^2 + b^2)\).

```latex
a, b > 0
s \in \mathbb{R}
c := \sqrt{a^2 + b^2}
\mathbf{r}(s) := \left(a \cos\frac{s}{c}, a \sin\frac{s}{c}, \frac{b s}{c}\right)
\mathbf{T} := \frac{d\mathbf{r}}{ds}
\kappa := \left\| \frac{d\mathbf{T}}{ds} \right\|
\mathbf{N} := \frac{1}{\kappa} \frac{d\mathbf{T}}{ds}
\mathbf{B} := \mathbf{T} \times \mathbf{N}
\tau := -\frac{d\mathbf{B}}{ds} \cdot \mathbf{N}
\tau
```

```python
s, t = symbols("s t", real=True)
a, b = symbols("a b", positive=True)
c = sqrt(a**2 + b**2)
r = Matrix([a * cos(s / c), a * sin(s / c), b * s / c])
T = r.diff(s)
assert simplify(T.dot(T)) == 1  # s is the arc length
kappa = simplify(sqrt(simplify(T.diff(s).dot(T.diff(s)))))
assert simplify(kappa - a / (a**2 + b**2)) == 0
N = simplify(T.diff(s) / kappa)
B = T.cross(N)
expected = simplify(-B.diff(s).dot(N))
assert simplify(expected - b / (a**2 + b**2)) == 0
# The other Frenet-Serret equation, dN/ds = -kappa T + tau B, holds with this torsion.
assert simplify(N.diff(s) + kappa * T - expected * B) == zeros(3, 1)
# And the torsion from any parametrisation, (r' x r'') . r''' / |r' x r''|^2, in t = s / c.
q = Matrix([a * cos(t), a * sin(t), b * t])
cross = q.diff(t).cross(q.diff(t, 2))
assert simplify(cross.dot(q.diff(t, 3)) / cross.dot(cross) - expected) == 0
```
