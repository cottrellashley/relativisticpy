# Integrals and the integral theorems

Volume and line integrals, and the divergence and Stokes theorems verified on Griffiths'
examples, each side computed from the field; Green's theorem in the plane, with a
differential form integrated over a curve; and the divergence of
\(\hat{\mathbf{r}}/r^2\), where the divergence theorem needs the Dirac delta.

## A volume integral

**Level 1** · `vc.integrals.volume` · Griffiths (2017), example 1.8

The integral of \(xyz^2\) over a prism, as nested integrals inside brackets, the inner
limits depending on the outer variable.

```latex
\int_0^3 z^2 \left\{ \int_0^1 x \left[ \int_0^{1 - x} y \, dy \right] dx \right\} dz
```

```python
x, y, z = symbols("x y z")
expected = integrate(z**2 * integrate(x * integrate(y, (y, 0, 1 - x)), (x, 0, 1)), (z, 0, 3))
assert expected == Rational(3, 8)
```

## A line integral

**Level 3** · `vc.integrals.line` · Griffiths (2017), example 1.6

The integral of \(\mathbf{v} \cdot d\mathbf{l}\) along the straight path from
\((1, 1, 0)\) to \((2, 2, 0)\), parametrised: the field evaluated on the path, dotted with
the path's tangent.

```latex
\mathbf{v} := y^2\,\hat{\mathbf{x}} + 2x(y + 1)\,\hat{\mathbf{y}}
\mathbf{r}(t) := (t, t, 0)
\int_1^2 \mathbf{v}(\mathbf{r}(t)) \cdot \mathbf{r}'(t) \, dt
```

```python
x, y, t = symbols("x y t")
v = Matrix([y**2, 2 * x * (y + 1), 0])
path = Matrix([t, t, 0])
expected = integrate(v.subs({x: t, y: t}).dot(path.diff(t)), (t, 1, 2))
assert expected == 10
```

## The divergence theorem

**Level 3** · `vc.integrals.divergence-theorem` · Griffiths (2017), example 1.10

The integral of \(\nabla \cdot \mathbf{v}\) over the unit cube equals the flux of
\(\mathbf{v}\) out through its six faces, taken in opposite pairs: both are 2.

```latex
\mathbf{v} := y^2\,\hat{\mathbf{x}} + (2xy + z^2)\,\hat{\mathbf{y}} + 2yz\,\hat{\mathbf{z}}
\int_0^1 \int_0^1 \int_0^1 \nabla \cdot \mathbf{v} \, dx \, dy \, dz = \int_0^1 \int_0^1 \left(\left. \mathbf{v} \right|_{x=1} - \left. \mathbf{v} \right|_{x=0}\right) \cdot \hat{\mathbf{x}} \, dy \, dz + \int_0^1 \int_0^1 \left(\left. \mathbf{v} \right|_{y=1} - \left. \mathbf{v} \right|_{y=0}\right) \cdot \hat{\mathbf{y}} \, dx \, dz + \int_0^1 \int_0^1 \left(\left. \mathbf{v} \right|_{z=1} - \left. \mathbf{v} \right|_{z=0}\right) \cdot \hat{\mathbf{z}} \, dx \, dy
```

```python
x, y, z = symbols("x y z")
v = [y**2, 2 * x * y + z**2, 2 * y * z]
divergence = diff(v[0], x) + diff(v[1], y) + diff(v[2], z)
volume = integrate(divergence, (x, 0, 1), (y, 0, 1), (z, 0, 1))
flux = (
    integrate(v[0].subs(x, 1) - v[0].subs(x, 0), (y, 0, 1), (z, 0, 1))
    + integrate(v[1].subs(y, 1) - v[1].subs(y, 0), (x, 0, 1), (z, 0, 1))
    + integrate(v[2].subs(z, 1) - v[2].subs(z, 0), (x, 0, 1), (y, 0, 1))
)
assert volume == flux == 2
expected = True
```

## Stokes' theorem

**Level 3** · `vc.integrals.stokes` · Griffiths (2017), example 1.11

The flux of \(\nabla \times \mathbf{v}\) through the unit square in the plane \(x = 0\)
equals the circulation of \(\mathbf{v}\) around its edge, traversed counterclockwise as
seen from \(+x\): both are \(4/3\).

```latex
\mathbf{v} := (2xz + 3y^2)\,\hat{\mathbf{y}} + 4yz^2\,\hat{\mathbf{z}}
\int_0^1 \int_0^1 \left. (\nabla \times \mathbf{v}) \cdot \hat{\mathbf{x}} \right|_{x=0} dy \, dz = \int_0^1 \left. \mathbf{v} \cdot \hat{\mathbf{y}} \right|_{x=0, z=0} dy + \int_0^1 \left. \mathbf{v} \cdot \hat{\mathbf{z}} \right|_{x=0, y=1} dz + \int_1^0 \left. \mathbf{v} \cdot \hat{\mathbf{y}} \right|_{x=0, z=1} dy + \int_1^0 \left. \mathbf{v} \cdot \hat{\mathbf{z}} \right|_{x=0, y=0} dz
```

```python
x, y, z = symbols("x y z")
vy, vz = 2 * x * z + 3 * y**2, 4 * y * z**2
curl_x = diff(vz, y) - diff(vy, z)
flux = integrate(curl_x.subs(x, 0), (y, 0, 1), (z, 0, 1))
circulation = (
    integrate(vy.subs({x: 0, z: 0}), (y, 0, 1))
    + integrate(vz.subs({x: 0, y: 1}), (z, 0, 1))
    + integrate(vy.subs({x: 0, z: 1}), (y, 1, 0))
    + integrate(vz.subs({x: 0, y: 0}), (z, 1, 0))
)
assert flux == circulation == Rational(4, 3)
expected = True
```

## Green's theorem with a differential form

**Level 4** · `vc.integrals.form` · Arfken, Weber and Harris (2013), section 4.8

The 1-form \(\omega = x\,dy - y\,dx\) integrated around a curve given by its equation, the
unit circle, positively oriented. By Green's theorem, the plane's case of
\(\oint_{\partial D} \omega = \int_D d\omega\), it is twice the area of the disk: \(2\pi\).

```latex
\omega := x\,dy - y\,dx
\oint_{x^2 + y^2 = 1} \omega
```

```python
t, r, theta = symbols("t r theta")
x, y = cos(t), sin(t)  # the unit circle, counterclockwise
expected = integrate(x * diff(y, t) - y * diff(x, t), (t, 0, 2 * pi))
assert expected == 2 * pi
assert (
    integrate(2 * r, (r, 0, 1), (theta, 0, 2 * pi)) == expected
)  # d(omega) = 2 dx dy over the disk
```

## The divergence of the inverse-square field

**Level 5** · `vc.integrals.delta` · Griffiths (2017), section 1.5.3

\(\hat{\mathbf{r}}/r^2\) has no divergence away from the origin, yet its flux through every
sphere about the origin is \(4\pi\): its divergence is \(4\pi\delta^3(\mathbf{r})\), an
equation between distributions.

```latex
ds^2 = dr^2 + r^2\, d\theta^2 + r^2 \sin^2\theta\, d\phi^2
\nabla \cdot \frac{\hat{\mathbf{r}}}{r^2} = 4\pi \delta^3(\mathbf{r})
```

```python
r, theta, phi = symbols("r theta phi")
R = symbols("R", positive=True)
assert simplify(diff(r**2 * (1 / r**2), r) / r**2) == 0  # zero wherever r > 0
flux = integrate(integrate((1 / R**2) * R**2 * sin(theta), (theta, 0, pi)), (phi, 0, 2 * pi))
assert flux == 4 * pi  # through the sphere of every radius R
expected = True
```
