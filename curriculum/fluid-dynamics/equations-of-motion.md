# The equations of motion

Conservation of mass, the continuity equation, which an incompressible flow satisfies as
\(\nabla \cdot \mathbf{u} = 0\), most easily through a stream function; the acceleration of
a fluid particle, the material derivative \(\partial \mathbf{u} / \partial t +
(\mathbf{u} \cdot \nabla) \mathbf{u}\); and the Navier–Stokes equations, in the index
notation of fluid mechanics. Batchelor (1967) is *An Introduction to Fluid Dynamics*.

## A stream function

**Level 2** · `fluid.motion.stream-function` · standard

Flow towards a wall, the stagnation-point flow of stream function \(\psi = kxy\). Velocities
taken from a stream function, \(u = \partial\psi/\partial y\) and
\(v = -\partial\psi/\partial x\), satisfy the continuity equation of an incompressible fluid.

```latex
\psi := k x y
u := \frac{\partial \psi}{\partial y}
v := -\frac{\partial \psi}{\partial x}
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
```

```python
x, y, k = symbols("x y k")
psi = k * x * y
u, v = diff(psi, y), -diff(psi, x)
assert (u, v) == (k * x, -k * y)
assert diff(u, x) + diff(v, y) == 0
# psi is constant along the flow, u dpsi/dx + v dpsi/dy = 0: the streamlines are xy = constant
assert simplify(u * diff(psi, x) + v * diff(psi, y)) == 0
expected = True
```

## The acceleration of a fluid particle

**Level 2** · `fluid.motion.material-derivative` · standard

In the steady flow \(\mathbf{u} = (kx, -ky)\) of the plane, in Cartesian coordinates
\(x, y\), nothing changes at a fixed point, yet every particle but the one at the
stagnation point accelerates: the material derivative \(\partial \mathbf{u} / \partial t +
(\mathbf{u} \cdot \nabla) \mathbf{u}\) is \((k^2 x, k^2 y)\). The operator
\(\mathbf{u} \cdot \nabla\) is made of a vector and \(\nabla\), and acts on each component.

```latex
ds^2 = dx^2 + dy^2
\mathbf{u} := (k x, -k y)
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u}
```

```python
x, y, t, k = symbols("x y t k")
u = Matrix([k * x, -k * y])
convective = Matrix([u[0] * diff(u[i], x) + u[1] * diff(u[i], y) for i in range(2)])
acceleration = u.diff(t) + convective
assert acceleration == Matrix([k**2 * x, k**2 * y])
# the particle that starts at (1, 1) moves on x = e^{kt}, y = e^{-kt}; its acceleration agrees
path = Matrix([exp(k * t), exp(-k * t)])
assert path.diff(t) == u.subs({x: path[0], y: path[1]})
assert path.diff(t, 2) == acceleration.subs({x: path[0], y: path[1]})
expected = [k**2 * x, k**2 * y]
```

## Plane Poiseuille flow solves the Navier–Stokes equations

**Level 4** · `fluid.motion.navier-stokes` · Batchelor (1967), chapter 3

The Navier–Stokes equations of an incompressible fluid in Cartesian index notation,
\(\rho(\partial_t u_i + u_j \partial_j u_i) = -\partial_i p + \mu\, \partial_j \partial_j
u_i\), with repeated lower indices summed, as fluid mechanics writes them, and with
\(\partial_t\), a derivative in a time that is not one of the coordinates of the line
element. Flow between plates at \(y = 0\) and \(y = h\), driven by a pressure that falls at
the rate \(G\), satisfies them in every component.

```latex
ds^2 = dx^2 + dy^2 + dz^2
u_i := \left[\frac{G}{2\mu} y (h - y), 0, 0\right]
p := p_0 - G x
\rho\left(\partial_t u_i + u_j \partial_j u_i\right) = -\partial_i p + \mu\, \partial_j \partial_j u_i
```

```python
x, y, z, t, G, mu, h, rho, p_0 = symbols("x y z t G mu h rho p_0")
X = (x, y, z)
u = [G / (2 * mu) * y * (h - y), 0, 0]
p = p_0 - G * x
for i in range(3):
    left = rho * (diff(u[i], t) + sum(u[j] * diff(u[i], X[j]) for j in range(3)))
    right = -diff(p, X[i]) + mu * sum(diff(u[i], X[j], 2) for j in range(3))
    assert simplify(left - right) == 0
# no slip on both plates, and G h^3 / 12 mu carried per unit width
assert u[0].subs(y, 0) == 0 and u[0].subs(y, h) == 0
assert simplify(integrate(u[0], (y, 0, h)) - G * h**3 / (12 * mu)) == 0
expected = True
```
