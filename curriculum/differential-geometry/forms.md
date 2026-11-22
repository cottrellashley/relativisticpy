# Differential forms

Forms in the notation of Cartan: the volume form, the exterior derivative, the Hodge star
and Stokes' theorem. A form is written with its differentials, \(x\,dy - y\,dx\), not
with indices.

## The volume of the three-sphere

**Level 2** · `dg.forms.three-sphere-volume` · standard

The volume form \(\sqrt{g}\, d\chi \wedge d\theta \wedge d\phi\) of a three-sphere of
radius \(a\), integrated over it: \(2\pi^2 a^3\).

```latex
ds^2 = a^2 \left(d\chi^2 + \sin^2\chi\, d\Omega^2\right)
\int_0^{2\pi} \int_0^\pi \int_0^\pi \sqrt{g}\, d\chi\, d\theta\, d\phi
```

```python
a, chi, theta, phi = symbols("a chi theta phi", positive=True)
g = diag(a**2, a**2 * sin(chi) ** 2, a**2 * sin(chi) ** 2 * sin(theta) ** 2)
assert simplify(g.det() - (a**3 * sin(chi) ** 2 * sin(theta)) ** 2) == 0
expected = integrate(
    a**3 * sin(chi) ** 2 * sin(theta), (chi, 0, pi), (theta, 0, pi), (phi, 0, 2 * pi)
)
assert expected == 2 * pi**2 * a**3
```

## The exterior derivative of a one-form

**Level 4** · `dg.forms.exterior-derivative` · Carroll (2004), section 2.9

On the plane, \(d(x\,dy - y\,dx) = 2\, dx \wedge dy\): as an antisymmetric tensor, the
components \((d\omega)_{xy} = -(d\omega)_{yx} = 2\).

```latex
Coordinates := [x, y]
\omega := x\, dy - y\, dx
d\omega
```

```python
x, y = symbols("x y")
X = [x, y]
omega = [-y, x]  # the components omega_x, omega_y
expected = Matrix(2, 2, lambda i, j: diff(omega[j], X[i]) - diff(omega[i], X[j]))
assert expected == Matrix([[0, 2], [-2, 0]])
```

## The Hodge star in three dimensions

**Level 4** · `dg.forms.hodge-star` · Carroll (2004), section 2.9

In Euclidean space with the orientation \((x, y, z)\), the dual of the area element
\(dx \wedge dy\) is the one-form \(dz\), with components \((0, 0, 1)\).

```latex
ds^2 = dx^2 + dy^2 + dz^2
\star (dx \wedge dy)
```

```python
beta = Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])  # (dx ^ dy)_{ij}; indices raised by delta
expected = [
    sum(beta[i, j] * LeviCivita(i, j, k) for i in range(3) for j in range(3)) / 2 for k in range(3)
]
assert expected == [0, 0, 1]
```

## Stokes' theorem on the disk

**Level 5** · `dg.forms.stokes-theorem` · Carroll (2004), appendix E

\(\int_D d\omega = \int_{\partial D} \omega\): for \(\omega = x\,dy - y\,dx\) on the unit
disk both sides are \(2\pi\). The case asks for integrals of forms over a region and over
its boundary, given as sets.

```latex
Coordinates := [x, y]
\omega := x\, dy - y\, dx
D := \{(x, y) : x^2 + y^2 \leq 1\}
\int_D d\omega = \oint_{\partial D} \omega
```

```python
rho, s = symbols("rho s", positive=True)
# d omega = 2 dx ^ dy, over the disk in polar coordinates, dx dy = rho d rho d s
interior = integrate(2 * rho, (rho, 0, 1), (s, 0, 2 * pi))
# the boundary circle, x = cos s, y = sin s, counterclockwise
x, y = cos(s), sin(s)
boundary = integrate(x * diff(y, s) - y * diff(x, s), (s, 0, 2 * pi))
assert interior == boundary == 2 * pi
expected = True
```
