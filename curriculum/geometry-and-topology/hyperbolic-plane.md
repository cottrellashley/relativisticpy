# The hyperbolic plane

Distances, areas and lengths in the upper half-plane, where \(ds = |dz| / \operatorname{Im}
z\), and in the Poincaré disk, where \(ds = 2|dz| / (1 - |z|^2)\). Beardon (1983) is *The
Geometry of Discrete Groups*. The curvature of the half-plane, \(-1\), is
`dg.surfaces.hyperbolic-plane`.

## The distance between two points of the half-plane

**Level 1** · `geom.hyperbolic.distance` · Beardon (1983), chapter 7

\(d(z, w) = \operatorname{arccosh}\left(1 + |z - w|^2 / 2 \operatorname{Im} z \operatorname{Im}
w\right)\), with the inverse hyperbolic cosine by its book name: from \(1 + i\) to
\(-1 + i\) it is \(\operatorname{arccosh} 3 = 2 \ln(1 + \sqrt{2})\), less than the length 2
of the horizontal segment between them, which is not a geodesic.

```latex
\operatorname{arccosh}\left(1 + \frac{|z - w|^2}{2 \operatorname{Im} z \operatorname{Im} w}\right) \Big|_{z = 1 + i, w = -1 + i}
```

```python
z, w = 1 + I, -1 + I
expected = acosh(1 + Abs(z - w) ** 2 / (2 * im(z) * im(w)))
assert expected == acosh(3)
assert abs(N(expected - 2 * log(1 + sqrt(2)), 30)) < 1e-25
# Along the geodesic through both points, the semicircle |z| = sqrt(2), the length element
# |dz| / Im z is d(theta) / sin(theta), from theta = pi/4 to 3 pi/4.
theta = symbols("theta")
length = integrate(1 / sin(theta), (theta, pi / 4, 3 * pi / 4))
assert abs(N(length - expected, 30)) < 1e-25
assert N(expected) < 2  # the horizontal segment, at height 1, has length 2
```

## The area of the modular triangle

**Level 2** · `geom.hyperbolic.modular-domain` · Beardon (1983), chapter 7

The region \(|x| \le 1/2\), \(x^2 + y^2 \ge 1\) of the half-plane, the fundamental domain of
the modular group, is a triangle with angles \(\pi/3\), \(\pi/3\) and 0, one vertex at
infinity. Its area, \(\iint dx\,dy / y^2\), is its angle defect,
\(\pi - (\alpha + \beta + \gamma) = \pi/3\).

```latex
\int_{-1/2}^{1/2} \int_{\sqrt{1 - x^2}}^{\infty} \frac{1}{y^2} \, dy \, dx = \pi - \left(\frac{\pi}{3} + \frac{\pi}{3} + 0\right)
```

```python
from scipy.integrate import quad

x, y = symbols("x y", real=True)
area = integrate(1 / y**2, (y, sqrt(1 - x**2), oo), (x, -Rational(1, 2), Rational(1, 2)))
assert simplify(area - pi / 3) == 0
assert abs(quad(lambda t: 1 / np.sqrt(1 - t**2), -0.5, 0.5)[0] - np.pi / 3) < 1e-12
# The angle at e^(i pi/3), between the side x = 1/2 going up and the unit circle going left:
# the half-plane's angles are Euclidean ones, and the cosine is (0, 1) . (-sqrt(3)/2, 1/2).
assert acos(Matrix([0, 1]).dot(Matrix([-sqrt(3) / 2, Rational(1, 2)]))) == pi / 3
expected = True
```

## The circumference of a hyperbolic circle

**Level 3** · `geom.hyperbolic.circumference` · standard

In the Poincaré disk the circle \(|z| = r_0\) has hyperbolic radius
\(\rho = \int_0^{r_0} 2\,dr / (1 - r^2)\) and length
\(\int_0^{2\pi} 2 r_0\,d\theta / (1 - r_0^2)\), which is \(2\pi \sinh\rho\): more than the
Euclidean \(2\pi\rho\), and exponentially more for large circles.

```latex
0 < r_0 < 1
\rho := \int_0^{r_0} \frac{2}{1 - r^2} \, dr
L := \int_0^{2\pi} \frac{2 r_0}{1 - r_0^2} \, d\theta
L = 2\pi \sinh\rho
```

```python
from scipy.integrate import quad

theta = symbols("theta", real=True)
r0 = symbols("r_0", positive=True)  # 0 < r_0 < 1
radius = log((1 + r0) / (1 - r0))  # the integral of 2/(1 - r^2) from 0 to r_0
assert simplify(diff(radius, r0) - 2 / (1 - r0**2)) == 0 and radius.subs(r0, 0) == 0
length = integrate(2 * r0 / (1 - r0**2), (theta, 0, 2 * pi))
assert simplify((2 * pi * sinh(radius)).rewrite(exp) - length) == 0
for value in (0.1, 0.5, 0.9):  # and numerically
    rho = quad(lambda s: 2 / (1 - s**2), 0, value)[0]
    assert abs(2 * np.pi * np.sinh(rho) - 4 * np.pi * value / (1 - value**2)) < 1e-9
expected = True
```
