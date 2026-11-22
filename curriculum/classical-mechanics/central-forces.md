# Central forces

Motion under a central force: the effective potential and its circular orbits, Kepler's
third law, and the Laplace–Runge–Lenz vector that the inverse-square force conserves.

## Kepler's third law for the Earth

**Level 1** · `cm.central.kepler-third-law` · Kepler (1619); IAU (2012) and (2015) values

The period of an orbit of one astronomical unit about the Sun, from
\(T = 2\pi \sqrt{a^3 / G M_\odot}\): a sidereal year, about 365.26 days. The evaluation
bar gives the product \(G M_\odot\) a value, as astronomers do.

```latex
2\pi \sqrt{\frac{a^3}{G M_\odot}} \Big|_{a = 1.495978707\times10^{11},\ G M_\odot = 1.32712440018\times10^{20}}
```

```python
expected = 2 * np.pi * np.sqrt(1.495978707e11**3 / 1.32712440018e20)
assert abs(expected / 86400 - 365.26) < 0.01  # seconds, as days
```

## The radius of a circular orbit

**Level 2** · `cm.central.circular-orbit` · Goldstein (2002), section 3.3

The effective potential of the Kepler problem, named with a subscript in text and
defined as a function of \(r\); its minimum is the circular orbit,
\(r = \ell^2 / \mu k\).

```latex
U_{\text{eff}}(r) = -\frac{k}{r} + \frac{\ell^2}{2 \mu r^2}
solve\left(\frac{d U_{\text{eff}}}{d r} = 0, r\right)
```

```python
r, k, ell, mu = symbols("r k ell mu", positive=True)
U_eff = -k / r + ell**2 / (2 * mu * r**2)
expected = solve(Eq(diff(U_eff, r), 0), r)
assert expected == [ell**2 / (k * mu)]
assert diff(U_eff, r, 2).subs(r, expected[0]) > 0  # a minimum: the orbit is stable
```

## The period of an elliptical orbit

**Level 3** · `cm.central.kepler-period` · Goldstein (2002), section 3.8

Kepler's third law, derived: the period is the area of the ellipse, \(\pi a b\), over
the constant areal velocity \(\ell / 2\mu\), with \(b = a\sqrt{1 - \epsilon^2}\) and
\(\ell^2 = \mu k a (1 - \epsilon^2)\). The eccentricity is \(\epsilon\), as in Taylor,
so that \(e\) stays Euler's number.

```latex
\ell := \sqrt{\mu k a (1 - \epsilon^2)}
b := a \sqrt{1 - \epsilon^2}
T := \frac{\pi a b}{\ell / (2\mu)}
T
```

```python
mu, k, a, epsilon = symbols("mu k a epsilon", positive=True)
ell = sqrt(mu * k * a * (1 - epsilon**2))
b = a * sqrt(1 - epsilon**2)
period = pi * a * b / (ell / (2 * mu))
expected = 2 * pi * sqrt(mu * a**3 / k)
assert simplify(period - expected) == 0
```

## The Laplace–Runge–Lenz vector is conserved

**Level 3** · `cm.central.runge-lenz` · Goldstein (2002), section 3.9

Vectors in phase space: \(\mathbf{A} = \mathbf{p} \times \mathbf{L} - m k\,
\mathbf{r}/r\), built with cross products, has a vanishing Poisson bracket with the
Kepler Hamiltonian, component by component.

```latex
\mathbf{r} = (x, y, z), \quad \mathbf{p} = (p_x, p_y, p_z), \quad r = |\mathbf{r}|
H = \frac{\mathbf{p}^2}{2m} - \frac{k}{r}
\mathbf{L} = \mathbf{r} \times \mathbf{p}
\mathbf{A} = \mathbf{p} \times \mathbf{L} - m k \frac{\mathbf{r}}{r}
\{\mathbf{A}, H\} = 0
```

```python
x, y, z, px, py, pz, m, k = symbols("x y z p_x p_y p_z m k")
position, momentum = Matrix([x, y, z]), Matrix([px, py, pz])
r = sqrt(x**2 + y**2 + z**2)
H = momentum.dot(momentum) / (2 * m) - k / r
A = momentum.cross(position.cross(momentum)) - m * k * position / r


def bracket(f, g):
    pairs = ((x, px), (y, py), (z, pz))
    return sum(diff(f, q) * diff(g, p) - diff(f, p) * diff(g, q) for q, p in pairs)


assert all(simplify(bracket(component, H)) == 0 for component in A)
expected = True
```
