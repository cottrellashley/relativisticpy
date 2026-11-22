# The fundamental forms of surfaces

The first and second fundamental forms of surfaces in space: the curvature of a graph; the
catenoid, a minimal surface; the helicoid, into which the catenoid can be bent without
stretching; and the Codazzi equations that every second fundamental form satisfies. do
Carmo (1976) is *Differential Geometry of Curves and Surfaces*, whose letters are \(E, F,
G\) for the first form and \(e, f, g\) for the second. The torus and the sphere are in
`differential-geometry/`.

## The curvature of a saddle

**Level 2** · `geom.surfaces.monge-saddle` · do Carmo (1976), section 3-3

The Gaussian curvature of a graph \(z = f(x, y)\),
\((f_{xx} f_{yy} - f_{xy}^2) / (1 + f_x^2 + f_y^2)^2\), with partial derivatives written as
subscripts: for the saddle \(z = xy\), \(-1 / (1 + x^2 + y^2)^2\).

```latex
f(x, y) := x y
K := \frac{f_{xx} f_{yy} - f_{xy}^2}{\left(1 + f_x^2 + f_y^2\right)^2}
K
```

```python
x, y = symbols("x y", real=True)
f = x * y
K = (diff(f, x, 2) * diff(f, y, 2) - diff(f, x, y) ** 2) / (
    1 + diff(f, x) ** 2 + diff(f, y) ** 2
) ** 2
expected = simplify(K)
assert simplify(expected + 1 / (1 + x**2 + y**2) ** 2) == 0
# The same from the fundamental forms of the patch (x, y, xy).
X = Matrix([x, y, x * y])
Xx, Xy = X.diff(x), X.diff(y)
normal = Xx.cross(Xy)
N = normal / sqrt(normal.dot(normal))
E, F, G = Xx.dot(Xx), Xx.dot(Xy), Xy.dot(Xy)
e, ff, g = N.dot(X.diff(x, 2)), N.dot(X.diff(x, y)), N.dot(X.diff(y, 2))
assert simplify((e * g - ff**2) / (E * G - F**2) - expected) == 0
```

## The catenoid is a minimal surface

**Level 3** · `geom.surfaces.catenoid-minimal` · do Carmo (1976), section 3-5

The mean curvature \(H = (eG - 2fF + gE) / 2(EG - F^2)\) from the two fundamental forms,
each coefficient a dot product of the partial derivatives, written as subscripts, of the
parametrisation and of its unit normal: the catenoid's vanishes everywhere, though its
Gaussian curvature does not.

```latex
a > 0
u, v \in \mathbb{R}
\mathbf{x}(u, v) := (a \cosh v \cos u, a \cosh v \sin u, a v)
\mathbf{N} := \frac{\mathbf{x}_u \times \mathbf{x}_v}{\left| \mathbf{x}_u \times \mathbf{x}_v \right|}
E := \mathbf{x}_u \cdot \mathbf{x}_u
F := \mathbf{x}_u \cdot \mathbf{x}_v
G := \mathbf{x}_v \cdot \mathbf{x}_v
e := \mathbf{N} \cdot \mathbf{x}_{uu}
f := \mathbf{N} \cdot \mathbf{x}_{uv}
g := \mathbf{N} \cdot \mathbf{x}_{vv}
H := \frac{e G - 2 f F + g E}{2 (E G - F^2)}
H
```

```python
u, v = symbols("u v", real=True)
a = symbols("a", positive=True)
x = Matrix([a * cosh(v) * cos(u), a * cosh(v) * sin(u), a * v])
xu, xv = x.diff(u), x.diff(v)
normal = xu.cross(xv)
length = a**2 * cosh(v) ** 2  # |x_u x x_v|
assert simplify(normal.dot(normal) - length**2) == 0
N = normal / length
E, F, G = xu.dot(xu), xu.dot(xv), xv.dot(xv)
e, f, g = N.dot(x.diff(u, 2)), N.dot(x.diff(u, v)), N.dot(x.diff(v, 2))
coefficients = [simplify(c) for c in (E, F, G, e, f, g)]
assert coefficients == [a**2 * cosh(v) ** 2, 0, a**2 * cosh(v) ** 2, -a, 0, a]
expected = simplify((e * G - 2 * f * F + g * E) / (2 * (E * G - F**2)))
assert expected == 0
# Minimal but not flat: the principal curvatures are -1/(a cosh^2 v) and 1/(a cosh^2 v).
assert simplify((e * g - f**2) / (E * G - F**2) + 1 / (a**2 * cosh(v) ** 4)) == 0
```

## The helicoid and the catenoid

**Level 3** · `geom.surfaces.helicoid-catenoid` · do Carmo (1976), section 4-2

A helicoid, \(\bar{\mathbf{x}}\), in the parameters \(u\) and \(v\) with its distance from
the axis \(a \sinh v\), has the first fundamental form of the catenoid \(\mathbf{x}\): the
two surfaces are locally isometric, and a piece of one can be bent into a piece of the
other without stretching.

```latex
a > 0
\mathbf{x}(u, v) := (a \cosh v \cos u, a \cosh v \sin u, a v)
\bar{\mathbf{x}}(u, v) := (a \sinh v \cos u, a \sinh v \sin u, a u)
E := \mathbf{x}_u \cdot \mathbf{x}_u
F := \mathbf{x}_u \cdot \mathbf{x}_v
G := \mathbf{x}_v \cdot \mathbf{x}_v
\bar{E} := \bar{\mathbf{x}}_u \cdot \bar{\mathbf{x}}_u
\bar{F} := \bar{\mathbf{x}}_u \cdot \bar{\mathbf{x}}_v
\bar{G} := \bar{\mathbf{x}}_v \cdot \bar{\mathbf{x}}_v
(E, F, G) = (\bar{E}, \bar{F}, \bar{G})
```

```python
u, v, w = symbols("u v w", real=True)
a = symbols("a", positive=True)


def first_form(p, s, t):  # E, F and G of a parametrised surface p(s, t)
    ps, pt = p.diff(s), p.diff(t)
    return [simplify(ps.dot(ps)), simplify(ps.dot(pt)), simplify(pt.dot(pt))]


catenoid = Matrix([a * cosh(v) * cos(u), a * cosh(v) * sin(u), a * v])
helicoid = Matrix([a * sinh(v) * cos(u), a * sinh(v) * sin(u), a * u])
assert first_form(catenoid, u, v) == [a**2 * cosh(v) ** 2, 0, a**2 * cosh(v) ** 2]
assert all(
    simplify(m - n) == 0
    for m, n in zip(first_form(catenoid, u, v), first_form(helicoid, u, v), strict=True)
)
# The helicoid in its usual coordinates, (w cos u, w sin u, a u), has E, F, G = w^2 + a^2, 0, 1;
# with w = a sinh v, dw = a cosh v dv, and (w^2 + a^2) du^2 + dw^2 = a^2 cosh^2 v (du^2 + dv^2).
usual = first_form(Matrix([w * cos(u), w * sin(u), a * u]), u, w)
assert usual == [a**2 + w**2, 0, 1]
assert simplify(usual[0].subs(w, a * sinh(v)) - a**2 * cosh(v) ** 2) == 0
assert simplify(diff(a * sinh(v), v) ** 2 - a**2 * cosh(v) ** 2) == 0
expected = True
```

## The Codazzi equations of the catenoid

**Level 4** · `geom.surfaces.codazzi` · do Carmo (1976), section 4-3

The catenoid's second fundamental form \(h_{ij}\), with its first fundamental form
\(a^2 \cosh^2 v\,(du^2 + dv^2)\) as the metric, satisfies the Codazzi–Mainardi equations,
\(\nabla_k h_{ij} = \nabla_j h_{ik}\): the covariant derivative of a tensor given by its
components, in index notation.

```latex
a > 0
ds^2 = a^2 \cosh^2 v \left(du^2 + dv^2\right)
h_{ij} := \begin{pmatrix} -a & 0 \\ 0 & a \end{pmatrix}
\nabla_k h_{ij} - \nabla_j h_{ik} = 0
```

```python
u, v = symbols("u v", real=True)
a = symbols("a", positive=True)
X = [u, v]
g = diag(a**2 * cosh(v) ** 2, a**2 * cosh(v) ** 2)
gi = g.inv()
Gam = [
    [
        [
            sum(
                gi[i, l] * (diff(g[l, j], X[k]) + diff(g[l, k], X[j]) - diff(g[j, k], X[l]))
                for l in range(2)
            )
            / 2
            for k in range(2)
        ]
        for j in range(2)
    ]
    for i in range(2)
]


def codazzi(h):  # whether nabla_k h_ij = nabla_j h_ik for every i, j and k
    def nabla(k, i, j):
        return diff(h[i, j], X[k]) - sum(
            Gam[l][k][i] * h[l, j] + Gam[l][k][j] * h[i, l] for l in range(2)
        )

    return all(
        simplify(nabla(k, i, j) - nabla(j, i, k)) == 0
        for i in range(2)
        for j in range(2)
        for k in range(2)
    )


# h and g are the catenoid's fundamental forms, from its embedding, with N = x_u x x_v / |...|.
x = Matrix([a * cosh(v) * cos(u), a * cosh(v) * sin(u), a * v])
N = x.diff(u).cross(x.diff(v)) / (a**2 * cosh(v) ** 2)
h = Matrix(2, 2, lambda i, j: simplify(N.dot(x.diff(X[i], X[j]))))
assert h == diag(-a, a)
assert simplify(Matrix(2, 2, lambda i, j: x.diff(X[i]).dot(x.diff(X[j]))) - g) == zeros(2, 2)
expected = codazzi(h)
assert expected
assert not codazzi(diag(-a, 2 * a))  # a form that no surface with this metric has
```
