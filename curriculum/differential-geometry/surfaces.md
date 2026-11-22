# Curved surfaces

The geometry of two-dimensional surfaces: the Gaussian curvature of the torus and of the
hyperbolic plane, the metric a surface inherits from the space around it, Gauss's
Theorema Egregium and the Gauss–Bonnet theorem.

## The Gaussian curvature of the torus

**Level 2** · `dg.surfaces.torus-curvature` · do Carmo (1976), section 3-3

On a surface the Riemann tensor has one independent component, and the Gaussian
curvature is \(K = R_{\theta\phi\theta\phi} / \det g\). The torus of radii \(R > r\)
curves positively on the outside and negatively on the inside.

```latex
ds^2 = r^2 d\theta^2 + (R + r\cos\theta)^2 d\phi^2
K := \frac{R_{\theta\phi\theta\phi}}{\det(g_{\mu\nu})}
K
```

```python
theta, phi, r, R = symbols("theta phi r R", positive=True)
X = [theta, phi]
g = diag(r**2, (R + r * cos(theta)) ** 2)
gi = g.inv()
n = 2
Gam = [
    [
        [
            sum(
                gi[a, d] * (diff(g[d, b], X[c]) + diff(g[d, c], X[b]) - diff(g[b, c], X[d]))
                for d in range(n)
            )
            / 2
            for c in range(n)
        ]
        for b in range(n)
    ]
    for a in range(n)
]
up = (
    diff(Gam[0][1][1], X[0])
    - diff(Gam[0][0][1], X[1])  # R^theta_{phi theta phi}
    + sum(Gam[0][0][e] * Gam[e][1][1] - Gam[0][1][e] * Gam[e][0][1] for e in range(n))
)
expected = simplify(g[0, 0] * up / g.det())
assert simplify(expected - cos(theta) / (r * (R + r * cos(theta)))) == 0
```

## The hyperbolic plane

**Level 2** · `dg.surfaces.hyperbolic-plane` · standard

Poincaré's upper half-plane, \(y > 0\), has constant Gaussian curvature \(K = R/2 = -1\).

```latex
ds^2 = \frac{dx^2 + dy^2}{y^2}
K := \frac{R}{2}
K
```

```python
x, y = symbols("x y", positive=True)
X = [x, y]
g = diag(1 / y**2, 1 / y**2)
gi = g.inv()
n = 2
Gam = [
    [
        [
            sum(
                gi[a, d] * (diff(g[d, b], X[c]) + diff(g[d, c], X[b]) - diff(g[b, c], X[d]))
                for d in range(n)
            )
            / 2
            for c in range(n)
        ]
        for b in range(n)
    ]
    for a in range(n)
]
ricci = Matrix(
    n,
    n,
    lambda b, d: sum(
        diff(Gam[a][d][b], X[a])
        - diff(Gam[a][a][b], X[d])
        + sum(Gam[a][a][e] * Gam[e][d][b] - Gam[a][d][e] * Gam[e][a][b] for e in range(n))
        for a in range(n)
    ),
)
scalar = simplify(sum(gi[b, d] * ricci[b, d] for b in range(n) for d in range(n)))
expected = scalar / 2
assert expected == -1
```

## The metric induced on a sphere

**Level 3** · `dg.surfaces.induced-metric` · do Carmo (1976), section 2-5

The first fundamental form: a sphere of radius \(a\) embedded in Euclidean space, with
coordinates \((\theta, \phi)\) on it, inherits \(g_{ab} = \delta_{ij}\, \partial_a X^i
\partial_b X^j\). Two kinds of index, over the surface and over the space around it.

```latex
Coordinates := [\theta, \phi]
X^i := [a \sin\theta \cos\phi, a \sin\theta \sin\phi, a \cos\theta]
g_{ab} := \delta_{ij}\, \partial_a X^i \partial_b X^j
g_{ab}
```

```python
a, theta, phi = symbols("a theta phi")
X = Matrix([a * sin(theta) * cos(phi), a * sin(theta) * sin(phi), a * cos(theta)])
J = X.jacobian([theta, phi])
expected = simplify(J.T * J)
assert expected == diag(a**2, a**2 * sin(theta) ** 2)
```

## The Gauss–Bonnet theorem for the sphere

**Level 3** · `dg.surfaces.gauss-bonnet` · do Carmo (1976), section 4-5

The total curvature of a closed surface is \(2\pi\chi\): for the sphere, whatever its
radius, \(\int K\, dA = 4\pi\).

```latex
ds^2 = a^2 \left(d\theta^2 + \sin^2\theta\, d\phi^2\right)
K := \frac{R}{2}
\int_0^{2\pi} \int_0^\pi K \sqrt{g}\, d\theta\, d\phi
```

```python
a, theta, phi = symbols("a theta phi", positive=True)
K = 1 / a**2  # R = 2/a^2 for the sphere of radius a
area_element = a**2 * sin(theta)  # sqrt(det g), with sin(theta) >= 0 on [0, pi]
expected = integrate(K * area_element, (theta, 0, pi), (phi, 0, 2 * pi))
euler_characteristic = 2
assert expected == 2 * pi * euler_characteristic
```

## The Theorema Egregium

**Level 3** · `dg.surfaces.theorema-egregium` · do Carmo (1976), sections 3-3 and 4-3

The Gaussian curvature of the torus from how it bends in space, through the first and
second fundamental forms, \(K = (eg - f^2)/(EG - F^2)\), in do Carmo's letters. Gauss
showed it is the intrinsic curvature, \(\cos\theta / r(R + r\cos\theta)\).

```latex
\mathbf{x} := \left[(R + r\cos\theta)\cos\phi, (R + r\cos\theta)\sin\phi, r\sin\theta\right]
\mathbf{N} := \frac{\partial_\theta \mathbf{x} \times \partial_\phi \mathbf{x}}{\left\| \partial_\theta \mathbf{x} \times \partial_\phi \mathbf{x} \right\|}
E := \partial_\theta \mathbf{x} \cdot \partial_\theta \mathbf{x}
F := \partial_\theta \mathbf{x} \cdot \partial_\phi \mathbf{x}
G := \partial_\phi \mathbf{x} \cdot \partial_\phi \mathbf{x}
e := \mathbf{N} \cdot \partial_\theta \partial_\theta \mathbf{x}
f := \mathbf{N} \cdot \partial_\theta \partial_\phi \mathbf{x}
g := \mathbf{N} \cdot \partial_\phi \partial_\phi \mathbf{x}
\frac{e g - f^2}{E G - F^2}
```

```python
theta, phi, r, R = symbols("theta phi r R", positive=True)
x = Matrix([(R + r * cos(theta)) * cos(phi), (R + r * cos(theta)) * sin(phi), r * sin(theta)])
xt, xp = x.diff(theta), x.diff(phi)
normal = xt.cross(xp)
length = r * (R + r * cos(theta))  # |x_theta x x_phi|, with R > r
assert simplify(normal.dot(normal) - length**2) == 0
N = normal / length
E, F, G = xt.dot(xt), xt.dot(xp), xp.dot(xp)
e, f, g = N.dot(x.diff(theta, 2)), N.dot(x.diff(theta, phi)), N.dot(x.diff(phi, 2))
expected = simplify((e * g - f**2) / (E * G - F**2))
assert simplify(expected - cos(theta) / (r * (R + r * cos(theta)))) == 0
```
