# Vorticity and circulation

The vorticity, the curl of the velocity, and the circulation round a closed curve, which
Stokes's theorem makes the flux of vorticity through it: a shear flow has vorticity though
its streamlines are straight, a line vortex has none away from its axis though its fluid
goes round, and the Rankine vortex keeps its vorticity in a core.

## The circulation of a line vortex

**Level 1** · `fluid.vorticity.line-vortex` · standard

The velocity \(v_\theta = \Gamma / 2\pi r\) of a line vortex, integrated round a circle of
radius \(r\): the circulation is \(\Gamma\), whatever the radius.

```latex
v_\theta := \frac{\Gamma}{2\pi r}
\int_0^{2\pi} v_\theta \, r \, d\theta
```

```python
Gamma, r, theta, t = symbols("Gamma r theta t")
R = symbols("R", positive=True)
expected = integrate(Gamma / (2 * pi * r) * r, (theta, 0, 2 * pi))
assert expected == Gamma
# the same in Cartesian components, u dx + v dy round the circle x = R cos t, y = R sin t
x, y = R * cos(t), R * sin(t)
u, v = -Gamma * y / (2 * pi * (x**2 + y**2)), Gamma * x / (2 * pi * (x**2 + y**2))
assert simplify(integrate(simplify(u * diff(x, t) + v * diff(y, t)), (t, 0, 2 * pi))) == Gamma
```

## The vorticity of a shear flow

**Level 2** · `fluid.vorticity.shear-flow` · standard

Plane Couette flow, \(u = Uy/h\), \(v = 0\), has straight streamlines but a uniform
vorticity, \(\omega = \partial v/\partial x - \partial u/\partial y = -U/h\): a fluid element
turns as it is sheared.

```latex
u := \frac{U y}{h}
v := 0
\frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}
```

```python
x, y, U, h = symbols("x y U h")
u, v = U * y / h, 0
expected = diff(v, x) - diff(u, y)
assert expected == -U / h
# by Stokes's theorem, the circulation counterclockwise round the unit square, of area 1
circulation = (
    integrate(u.subs(y, 0), (x, 0, 1))  # along the bottom
    + integrate(v, (y, 0, 1))  # up the right side
    - integrate(u.subs(y, 1), (x, 0, 1))  # back along the top
    - integrate(v, (y, 0, 1))  # down the left side
)
assert circulation == expected
```

## The Rankine vortex

**Level 3** · `fluid.vorticity.rankine` · standard

A core of radius \(a\) turning rigidly inside a line vortex: \(v_\theta\) is
\(\Gamma r / 2\pi a^2\) inside and \(\Gamma / 2\pi r\) outside. Its vorticity
\(\frac{1}{r}(r v_\theta)'\) is \(\Gamma / \pi a^2\) in the core and zero outside, and
integrated over the core it gives back the circulation \(\Gamma\), as Stokes's theorem says.
A function named with a subscript, defined by cases.

```latex
a > 0
v_\theta(r) := \begin{cases} \frac{\Gamma r}{2\pi a^2} & r < a \\ \frac{\Gamma}{2\pi r} & r \geq a \end{cases}
\omega(r) := \frac{1}{r} \frac{d}{dr}\left(r\, v_\theta(r)\right)
\int_0^a \omega(r)\, 2\pi r \, dr
```

```python
r, a = symbols("r a", positive=True)
Gamma = symbols("Gamma")
v_theta = Piecewise((Gamma * r / (2 * pi * a**2), r < a), (Gamma / (2 * pi * r), True))
omega = diff(r * v_theta, r) / r
expected = simplify(integrate(omega * 2 * pi * r, (r, 0, a)))
assert expected == Gamma
assert simplify(omega.subs(r, a / 2)) == Gamma / (pi * a**2)  # uniform in the core
assert simplify(omega.subs(r, 2 * a)) == 0  # irrotational outside
assert simplify(2 * pi * a * v_theta.subs(r, a)) == Gamma  # the circulation round the core
```
