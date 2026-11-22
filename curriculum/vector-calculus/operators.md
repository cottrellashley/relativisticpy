# Differential operators

Divergence, curl, gradient and the Laplacian, written with \(\nabla\) as books on vector
calculus write them, in Cartesian coordinates with the unit vectors
\(\hat{\mathbf{x}}, \hat{\mathbf{y}}, \hat{\mathbf{z}}\), and in cylindrical and spherical
coordinates given by their line element; the scale factors of curvilinear coordinates; and
two identities.

## The divergence

**Level 1** · `vc.operators.divergence` · Griffiths (2017), section 1.2.4

The divergence of a vector field written with unit vectors.

```latex
\mathbf{v} := xy\,\hat{\mathbf{x}} + 2yz\,\hat{\mathbf{y}} + 3zx\,\hat{\mathbf{z}}
\nabla \cdot \mathbf{v}
```

```python
x, y, z = symbols("x y z")
v = [x * y, 2 * y * z, 3 * z * x]
expected = sum(diff(component, coordinate) for component, coordinate in zip(v, (x, y, z)))
assert expected == 3 * x + y + 2 * z
```

## The curl

**Level 1** · `vc.operators.curl` · Griffiths (2017), section 1.2.5

The curl of the same field, a vector, given by its components along
\(\hat{\mathbf{x}}, \hat{\mathbf{y}}, \hat{\mathbf{z}}\).

```latex
\mathbf{v} := xy\,\hat{\mathbf{x}} + 2yz\,\hat{\mathbf{y}} + 3zx\,\hat{\mathbf{z}}
\nabla \times \mathbf{v}
```

```python
x, y, z = symbols("x y z")
vx, vy, vz = x * y, 2 * y * z, 3 * z * x
expected = [diff(vz, y) - diff(vy, z), diff(vx, z) - diff(vz, x), diff(vy, x) - diff(vx, y)]
assert expected == [-2 * y, -3 * z, -x]
```

## The curl of a gradient

**Level 2** · `vc.operators.curl-grad` · Griffiths (2017), section 1.2.7

\(\nabla \times (\nabla f) = 0\) for every function \(f\) of \(x\), \(y\) and \(z\):
mixed partial derivatives commute.

```latex
\nabla \times (\nabla f(x, y, z)) = \mathbf{0}
```

```python
x, y, z = symbols("x y z")
f = Function("f")(x, y, z)
gx, gy, gz = diff(f, x), diff(f, y), diff(f, z)
curl = [diff(gz, y) - diff(gy, z), diff(gx, z) - diff(gz, x), diff(gy, x) - diff(gx, y)]
assert all(simplify(component) == 0 for component in curl)
expected = True
```

## The Laplacian in spherical coordinates

**Level 2** · `vc.operators.laplacian-spherical` · Arfken, Weber and Harris (2013), section 3.10

In coordinates given by their line element, \(\nabla^2 r^n = n(n + 1) r^{n-2}\): the
Laplacian in spherical coordinates, with its scale factors.

```latex
ds^2 = dr^2 + r^2\, d\theta^2 + r^2 \sin^2\theta\, d\phi^2
\nabla^2 r^n
```

```python
r, theta, n = symbols("r theta n")
u = r**n
laplacian = diff(r**2 * diff(u, r), r) / r**2 + diff(sin(theta) * diff(u, theta), theta) / (
    r**2 * sin(theta)
)
expected = n * (n + 1) * r ** (n - 2)
assert simplify(laplacian - expected) == 0
```

## The curl in cylindrical coordinates

**Level 2** · `vc.operators.curl-cylindrical` · Arfken, Weber and Harris (2013), section 3.10

A rigid rotation about the \(z\) axis, \(\mathbf{v} = \omega\rho\,\hat{\boldsymbol\varphi}\):
its curl is twice the angular velocity, along the axis. Components along
\(\hat{\boldsymbol\rho}, \hat{\boldsymbol\varphi}, \hat{\mathbf{z}}\).

```latex
ds^2 = d\rho^2 + \rho^2\, d\varphi^2 + dz^2
\mathbf{v} := \omega \rho\, \hat{\boldsymbol{\varphi}}
\nabla \times \mathbf{v}
```

```python
rho, phi, z, omega = symbols("rho phi z omega")
v_rho, v_phi, v_z = 0, omega * rho, 0
expected = [
    diff(v_z, phi) / rho - diff(v_phi, z),
    diff(v_rho, z) - diff(v_z, rho),
    (diff(rho * v_phi, rho) - diff(v_rho, phi)) / rho,
]
assert expected == [0, 0, 2 * omega]
```

## A scale factor

**Level 2** · `vc.operators.scale-factor` · Arfken, Weber and Harris (2013), section 3.10

The scale factor of a curvilinear coordinate, \(h_i^2 = \sum_k (\partial x_k / \partial
q_i)^2\), from the Cartesian coordinates written in the spherical ones: \(h_\phi = r \sin
\theta\).

```latex
x := r \sin\theta \cos\phi
y := r \sin\theta \sin\phi
z := r \cos\theta
h_\phi := \sqrt{\left(\frac{\partial x}{\partial \phi}\right)^2 + \left(\frac{\partial y}{\partial \phi}\right)^2 + \left(\frac{\partial z}{\partial \phi}\right)^2}
h_\phi
```

```python
r, theta, phi = symbols("r theta phi")
point = [r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta)]
h = sqrt(sum(diff(coordinate, phi) ** 2 for coordinate in point))
expected = r * sin(theta)  # positive, for r > 0 and 0 < theta < pi
assert simplify(h**2 - expected**2) == 0
```

## The curl of a curl in index notation

**Level 4** · `vc.operators.curl-curl` · Arfken, Weber and Harris (2013), section 3.6

\(\nabla \times (\nabla \times \mathbf{A}) = \nabla(\nabla \cdot \mathbf{A}) -
\nabla^2 \mathbf{A}\), written in Cartesian index notation, with derivatives
\(\partial_j\) and the Levi-Civita symbol, for a field whose components are unknown
functions.

```latex
ds^2 = dx^2 + dy^2 + dz^2
A_{i} := [u(x, y, z), v(x, y, z), w(x, y, z)]
\epsilon_{ijk} \partial_j (\epsilon_{klm} \partial_l A_m) = \partial_i \partial_j A_j - \partial_j \partial_j A_i
```

```python
x, y, z = symbols("x y z")
X = (x, y, z)
A = [Function(name)(x, y, z) for name in ("u", "v", "w")]
R = range(3)
for i in R:
    left = sum(
        LeviCivita(i, j, k)
        * diff(sum(LeviCivita(k, l, m) * diff(A[m], X[l]) for l in R for m in R), X[j])
        for j in R
        for k in R
    )
    right = sum(diff(A[j], X[i], X[j]) for j in R) - sum(diff(A[i], X[j], X[j]) for j in R)
    assert simplify(left - right) == 0
expected = True
```
