# Geodesic curvature and the Gauss–Bonnet theorem

How a curve on a surface bends within it, and the Gauss–Bonnet theorem: the total
curvature of a closed surface is \(2\pi\chi\), of a geodesic triangle its angle excess, and
of a region with a boundary \(2\pi\) less the turning of the boundary. Chern's theorem takes
it to four dimensions. do Carmo (1976) is *Differential Geometry of Curves and Surfaces*.
The total curvature of the sphere, \(4\pi\), is `dg.surfaces.gauss-bonnet`.

## The geodesic curvature of a circle of latitude

**Level 2** · `geom.gauss-bonnet.latitude` · do Carmo (1976), section 4-4

The geodesic curvature of a unit-speed curve \(\boldsymbol{\alpha}(s)\) on a surface with
unit normal \(\mathbf{N}\) is \(\boldsymbol{\alpha}'' \cdot (\mathbf{N} \times
\boldsymbol{\alpha}')\). On the sphere of radius \(a\), with the outward normal, the circle
of colatitude \(\theta_0\) traversed eastward has \(k_g = \cot\theta_0 / a\): of the circles
of latitude, only the equator is a geodesic.

```latex
a > 0
0 < \theta_0 < \pi
s \in \mathbb{R}
\boldsymbol{\alpha}(s) := \left(a \sin\theta_0 \cos\frac{s}{a \sin\theta_0}, a \sin\theta_0 \sin\frac{s}{a \sin\theta_0}, a \cos\theta_0\right)
\mathbf{N}(s) := \frac{\boldsymbol{\alpha}(s)}{a}
k_g := \boldsymbol{\alpha}''(s) \cdot \left(\mathbf{N}(s) \times \boldsymbol{\alpha}'(s)\right)
k_g
```

```python
s = symbols("s", real=True)
a, theta0, theta = symbols("a theta_0 theta", positive=True)  # 0 < theta_0 < pi
rho = a * sin(theta0)  # the radius of the circle
alpha = Matrix([rho * cos(s / rho), rho * sin(s / rho), a * cos(theta0)])
assert simplify(alpha.diff(s).dot(alpha.diff(s))) == 1  # s is the arc length
N = alpha / a  # the outward normal of the sphere
expected = simplify(alpha.diff(s, 2).dot(N.cross(alpha.diff(s))))
assert simplify(expected - cot(theta0) / a) == 0
# The curvature 1/rho of the circle splits into normal and geodesic parts, k^2 = k_n^2 + k_g^2.
k_n = simplify(alpha.diff(s, 2).dot(N))
assert simplify(k_n + 1 / a) == 0
assert simplify(1 / rho**2 - k_n**2 - expected**2) == 0
# Liouville's formula for a parallel of ds^2 = E dtheta^2 + G dphi^2: G_theta / (2 G sqrt(E)).
G = a**2 * sin(theta) ** 2
assert simplify((diff(G, theta) / (2 * G * a)).subs(theta, theta0) - expected) == 0
```

## The total curvature of the torus

**Level 3** · `geom.gauss-bonnet.torus` · do Carmo (1976), section 4-5

The torus of radii \(a > r > 0\) curves positively on the outside and negatively on the
inside, and in total not at all: \(\int K\,dA = 2\pi\chi = 0\). Its area element,
\(\sqrt{g} = r(a + r\cos\theta)\), takes the condition \(a > r\) to simplify: that is what
this case adds to the sphere's `dg.surfaces.gauss-bonnet`, whose program has the same
shape.

```latex
a > r > 0
ds^2 = r^2 d\theta^2 + (a + r\cos\theta)^2 d\phi^2
K := \frac{R}{2}
\int_0^{2\pi} \int_0^{2\pi} K \sqrt{g} \, d\theta \, d\phi
```

```python
theta, phi = symbols("theta phi", real=True)
a, r = symbols("a r", positive=True)  # and a > r, so that a + r cos(theta) > 0
E, G = r**2, (a + r * cos(theta)) ** 2
root = r * (a + r * cos(theta))  # sqrt(E G), the area element
assert simplify(root**2 - E * G) == 0
# Gauss's formula for an orthogonal metric: K = -(d/dtheta (G_theta / sqrt(EG))) / (2 sqrt(EG)).
K = simplify(-diff(diff(G, theta) / root, theta) / (2 * root))
assert simplify(K - cos(theta) / (r * (a + r * cos(theta)))) == 0
expected = integrate(simplify(K * root), (theta, 0, 2 * pi), (phi, 0, 2 * pi))
assert expected == 0  # 2 pi chi, and chi = 0 for the torus
# The outer half, where K > 0, has total curvature 4 pi, and the inner half -4 pi.
assert integrate(simplify(K * root), (theta, -pi / 2, pi / 2), (phi, 0, 2 * pi)) == 4 * pi
```

## A geodesic triangle on the sphere

**Level 3** · `geom.gauss-bonnet.geodesic-triangle` · do Carmo (1976), section 4-5

The triangle on the unit sphere bounded by two meridians \(\lambda\) apart and the equator
has angles \(\lambda\), \(\pi/2\) and \(\pi/2\). Its total curvature, here its area, is its
angle excess, \(\alpha + \beta + \gamma - \pi\); the octant, \(\lambda = \pi/2\), is an
eighth of the sphere.

```latex
0 < \lambda < \pi
ds^2 = d\theta^2 + \sin^2\theta \, d\phi^2
K := \frac{R}{2}
\int_0^{\lambda} \int_0^{\pi/2} K \sqrt{g} \, d\theta \, d\phi = \lambda + \frac{\pi}{2} + \frac{\pi}{2} - \pi
```

```python
theta, phi = symbols("theta phi", real=True)
lam = symbols("lambda", positive=True)  # 0 < lambda < pi
g = diag(1, sin(theta) ** 2)
# K from the orthogonal metric: -(d/dtheta (G_theta / sqrt(G))) / (2 sqrt(G)), sqrt(G) = sin(theta).
K = simplify(-diff(diff(g[1, 1], theta) / sin(theta), theta) / (2 * sin(theta)))
assert K == 1
area = integrate(K * sin(theta), (theta, 0, pi / 2), (phi, 0, lam))
angles = lam + pi / 2 + pi / 2  # at the pole, and where the two meridians meet the equator
assert simplify(area - (angles - pi)) == 0
# The meridians meet the equator at right angles: their tangents (1, 0) and (0, 1) are orthogonal.
assert (Matrix([[1, 0]]) * g.subs(theta, pi / 2) * Matrix([0, 1]))[0] == 0
assert area.subs(lam, pi / 2) == 4 * pi / 8
expected = True
```

## A spherical cap

**Level 4** · `geom.gauss-bonnet.spherical-cap` · do Carmo (1976), section 4-5

The Gauss–Bonnet theorem for a region with a boundary, as the books write it, with the
region a set and integrals over it and over its boundary by the area and length elements
of the metric, \(\iint_D K\,dA + \oint_{\partial D} k_g\,ds = 2\pi\) for a disk. For the cap
of colatitude \(\theta_0\) on the sphere of radius \(a\), the total curvature
\(2\pi(1 - \cos\theta_0)\) and the turning \(2\pi\cos\theta_0\) of its boundary add to
\(2\pi\). The boundary is traversed eastward, with \(D\) on its left as seen from the
outward normal, the orientation in which its geodesic curvature is \(\cot\theta_0/a\), as
in `geom.gauss-bonnet.latitude`; beyond the equator it is negative.

```latex
a > 0
0 < \theta_0 < \pi
ds^2 = a^2 \left(d\theta^2 + \sin^2\theta \, d\phi^2\right)
D := \{(\theta, \phi) : \theta \le \theta_0\}
K := \frac{R}{2}
k_g := \frac{\cot\theta_0}{a}
\iint_D K \, dA + \oint_{\partial D} k_g \, ds = 2\pi
```

```python
theta, phi = symbols("theta phi", real=True)
a, theta0, t = symbols("a theta_0 t", positive=True)  # 0 < theta_0 < pi
K = 1 / a**2
interior = integrate(K * a**2 * sin(theta), (theta, 0, theta0), (phi, 0, 2 * pi))
assert simplify(interior - 2 * pi * (1 - cos(theta0))) == 0
# k_g of the boundary, by Liouville's formula G_theta / (2 G sqrt(E)), E = a^2, G = a^2 sin^2.
k_g = simplify((diff(a**2 * sin(t) ** 2, t) / (2 * a**2 * sin(t) ** 2 * a)).subs(t, theta0))
assert simplify(k_g - cot(theta0) / a) == 0
boundary = integrate(k_g * a * sin(theta0), (phi, 0, 2 * pi))  # ds = a sin(theta_0) dphi
assert simplify(interior + boundary - 2 * pi) == 0  # 2 pi chi(D), the cap a disk
# A hemisphere has a geodesic boundary and total curvature 2 pi.
assert interior.subs(theta0, pi / 2) == 2 * pi
assert boundary.subs(theta0, pi / 2) == 0
expected = True
```

## The Euler characteristic of the four-sphere

**Level 4** · `geom.gauss-bonnet.chern-four-sphere` · Chern (1944), Ann. Math. 45, 747

The Chern–Gauss–Bonnet theorem in four dimensions,
\(\chi = \frac{1}{32\pi^2} \int (R_{abcd} R^{abcd} - 4 R_{ab} R^{ab} + R^2) \sqrt{g}\,d^4x\),
for the unit four-sphere: the curvature terms add to 24 everywhere, the volume is
\(8\pi^2/3\), and \(\chi(S^4) = 2\).

```latex
ds^2 = d\alpha^2 + \sin^2\alpha \left(d\beta^2 + \sin^2\beta \left(d\theta^2 + \sin^2\theta \, d\phi^2\right)\right)
\frac{1}{32\pi^2} \int_0^{2\pi} \int_0^\pi \int_0^\pi \int_0^\pi \left(R_{abcd} R^{abcd} - 4 R_{ab} R^{ab} + R^2\right) \sqrt{g} \, d\alpha \, d\beta \, d\theta \, d\phi
```

```python
X = symbols("alpha beta theta phi", real=True)
al, be, th, ph = X
n = 4
g = diag(1, sin(al) ** 2, sin(al) ** 2 * sin(be) ** 2, sin(al) ** 2 * sin(be) ** 2 * sin(th) ** 2)
gi = g.inv()
Gam = [
    [
        [
            simplify(
                sum(
                    gi[A, d] * (diff(g[d, b], X[c]) + diff(g[d, c], X[b]) - diff(g[b, c], X[d]))
                    for d in range(n)
                )
                / 2
            )
            for c in range(n)
        ]
        for b in range(n)
    ]
    for A in range(n)
]
quadruples = [(A, b, c, d) for A in range(n) for b in range(n) for c in range(n) for d in range(n)]
up = {  # R^A_{bcd}
    (A, b, c, d): simplify(
        diff(Gam[A][d][b], X[c])
        - diff(Gam[A][c][b], X[d])
        + sum(Gam[A][c][e] * Gam[e][d][b] - Gam[A][d][e] * Gam[e][c][b] for e in range(n))
    )
    for A, b, c, d in quadruples
}
down = {(A, b, c, d): simplify(g[A, A] * up[A, b, c, d]) for A, b, c, d in quadruples}
# Constant curvature 1: R_abcd = g_ac g_bd - g_ad g_bc.
assert all(
    simplify(down[A, b, c, d] - (g[A, c] * g[b, d] - g[A, d] * g[b, c])) == 0
    for A, b, c, d in quadruples
)
riemann2 = simplify(
    sum(
        down[q] ** 2 * gi[q[0], q[0]] * gi[q[1], q[1]] * gi[q[2], q[2]] * gi[q[3], q[3]]
        for q in quadruples
    )
)
ricci = Matrix(n, n, lambda b, d: simplify(sum(up[A, b, A, d] for A in range(n))))
assert simplify(ricci - 3 * g) == zeros(n, n)
ricci2 = simplify(sum(ricci[b, d] ** 2 * gi[b, b] * gi[d, d] for b in range(n) for d in range(n)))
scalar = simplify(sum(gi[b, b] * ricci[b, b] for b in range(n)))
density = simplify(riemann2 - 4 * ricci2 + scalar**2)
assert (riemann2, ricci2, scalar, density) == (24, 36, 12, 24)
volume = integrate(
    sin(al) ** 3 * sin(be) ** 2 * sin(th), (al, 0, pi), (be, 0, pi), (th, 0, pi), (ph, 0, 2 * pi)
)
assert volume == 2 * pi ** Rational(5, 2) / gamma(Rational(5, 2)) == 8 * pi**2 / 3  # DLMF 5.19
expected = density * volume / (32 * pi**2)
assert expected == 2  # the Euler characteristic of an even-dimensional sphere
```
