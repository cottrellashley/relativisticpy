# Electromagnetic waves and their energy

The speed of light from the constants of electricity and magnetism, the energy carried by
a plane wave, the stress–energy tensor of the field and its conservation, and the
retarded Green's function of the wave equation.

## The speed of light from the constants of electromagnetism

**Level 1** · `em.waves.speed-of-light` · Maxwell (1865); Griffiths (2013), section 9.2.1

The speed of the waves of Maxwell's equations, \(1/\sqrt{\mu_0 \epsilon_0}\), evaluated
with the values of \(\mu_0\) and \(\epsilon_0\) in the SI before 2019, written in
scientific notation.

```latex
\frac{1}{\sqrt{\mu_0 \epsilon_0}} \Big|_{\mu_0 = 4\pi\times10^{-7},\ \epsilon_0 = 8.854187817\times10^{-12}}
```

```python
expected = 1 / np.sqrt(4 * np.pi * 1e-7 * 8.854187817e-12)
assert abs(expected - 299_792_458) / 299_792_458 < 1e-9
```

## The Poynting vector of a plane wave

**Level 2** · `em.waves.poynting` · Griffiths (2013), section 9.2.3

Vectors written with unit vectors, \(\hat{\mathbf{x}}\) and \(\hat{\mathbf{y}}\), and
their cross product: the energy flux of a plane wave points along its direction of
travel.

```latex
\mathbf{E} := E_0 \cos(k z - \omega t)\, \hat{\mathbf{x}}
\mathbf{B} := \frac{E_0}{c} \cos(k z - \omega t)\, \hat{\mathbf{y}}
\frac{1}{\mu_0} \mathbf{E} \times \mathbf{B}
```

```python
E0, k, z, omega, t, c, mu_0, epsilon_0 = symbols("E_0 k z omega t c mu_0 epsilon_0")
E = Matrix([E0 * cos(k * z - omega * t), 0, 0])
B = Matrix([0, E0 / c * cos(k * z - omega * t), 0])
S = E.cross(B) / mu_0
expected = list(S)
# Griffiths (9.62): S = c epsilon_0 E_0^2 cos^2(kz - omega t) z-hat, as mu_0 = 1/(epsilon_0 c^2)
flux = c * epsilon_0 * E0**2 * cos(k * z - omega * t) ** 2
assert simplify(S[2].subs(mu_0, 1 / (epsilon_0 * c**2)) - flux) == 0
assert S[0] == S[1] == 0
```

## The stress–energy tensor of the field is traceless

**Level 3** · `em.waves.stress-energy-traceless` · Carroll (2004), section 1.9

The stress–energy tensor built from the field tensor, with its indices raised and
lowered by the metric, and its trace (Heaviside–Lorentz units, \(c = 1\)).

```latex
ds^2 = -dt^2 + dx^2 + dy^2 + dz^2
F_{\mu\nu} := \begin{pmatrix} 0 & -E_x & -E_y & -E_z \\ E_x & 0 & B_z & -B_y \\ E_y & -B_z & 0 & B_x \\ E_z & B_y & -B_x & 0 \end{pmatrix}
T^{\mu\nu} := F^{\mu\alpha} F^{\nu}{}_{\alpha} - \frac{1}{4} g^{\mu\nu} F_{\alpha\beta} F^{\alpha\beta}
T^\mu{}_\mu
```

```python
Ex, Ey, Ez, Bx, By, Bz = symbols("E_x E_y E_z B_x B_y B_z")
F = Matrix([[0, -Ex, -Ey, -Ez], [Ex, 0, Bz, -By], [Ey, -Bz, 0, Bx], [Ez, By, -Bx, 0]])
eta = diag(-1, 1, 1, 1)
F_up = eta * F * eta
F_mixed = F_up * eta  # F^{mu}_{alpha}
invariant = sum(F[a, b] * F_up[a, b] for a in range(4) for b in range(4))
T = Matrix(
    4,
    4,
    lambda m, n: (
        sum(F_up[m, a] * F_mixed[n, a] for a in range(4)) - Rational(1, 4) * eta[m, n] * invariant
    ),
)
expected = expand(sum(T[m, n] * eta[m, n] for m in range(4) for n in range(4)))
assert expected == 0
# its energy density is (E^2 + B^2)/2, and its momentum density the Poynting vector
assert expand(T[0, 0] - (Ex**2 + Ey**2 + Ez**2 + Bx**2 + By**2 + Bz**2) / 2) == 0
assert expand(T[0, 1] - (Ey * Bz - Ez * By)) == 0
```

## Energy and momentum are conserved

**Level 4** · `em.waves.stress-energy-conservation` · Jackson (1999), section 12.10

A derivation in abstract index notation, with no components: from Maxwell's equations,
\(\partial_\mu F^{\nu\mu} = J^\nu\) and \(\partial_{[\mu} F_{\nu\lambda]} = 0\), the
divergence of the stress–energy tensor is the force density on the charges,
\(\partial_\mu T^{\mu\nu} = -F^{\nu\lambda} J_\lambda\).

```latex
T^{\mu\nu} := F^{\mu\lambda} F^{\nu}{}_{\lambda} - \frac{1}{4} g^{\mu\nu} F_{\lambda\sigma} F^{\lambda\sigma}
\partial_\mu F^{\nu\mu} = J^\nu, \quad \partial_{[\mu} F_{\nu\lambda]} = 0
\partial_\mu T^{\mu\nu} = -F^{\nu\lambda} J_\lambda
```

```python
X = symbols("t x y z")
A = [Function(f"A_{n}")(*X) for n in range(4)]  # any potential: F = dA has no monopoles
eta = diag(-1, 1, 1, 1)
F = Matrix(4, 4, lambda m, n: diff(A[n], X[m]) - diff(A[m], X[n]))
F_up = eta * F * eta
F_mixed = F_up * eta
invariant = sum(F[a, b] * F_up[a, b] for a in range(4) for b in range(4))
T = Matrix(
    4,
    4,
    lambda m, n: (
        sum(F_up[m, a] * F_mixed[n, a] for a in range(4)) - Rational(1, 4) * eta[m, n] * invariant
    ),
)
J = [sum(diff(F_up[n, m], X[m]) for m in range(4)) for n in range(4)]  # the current it needs
J_down = [eta[n, n] * J[n] for n in range(4)]
for n in range(4):
    divergence = sum(diff(T[m, n], X[m]) for m in range(4))
    force = -sum(F_up[n, l] * J_down[l] for l in range(4))
    assert expand(divergence - force) == 0
expected = True
```

## The retarded Green's function of the wave equation

**Level 5** · `em.waves.retarded-green` · Jackson (1999), section 6.4

A distribution: \(G = \delta(t - |\mathbf{x}|/c) / |\mathbf{x}|\) solves the wave
equation with a point source at the origin at \(t = 0\). Delta functions of a function
of the coordinates, and an equation that holds only as distributions, are beyond every
design of the language so far.

```latex
G(\mathbf{x}, t) := \frac{\delta(t - |\mathbf{x}|/c)}{|\mathbf{x}|}
\left(\nabla^2 - \frac{1}{c^2} \frac{\partial^2}{\partial t^2}\right) G(\mathbf{x}, t) = -4\pi\, \delta^3(\mathbf{x})\, \delta(t)
```

```python
from scipy.integrate import quad

r, t, c = symbols("r t c", positive=True)
test = exp(-(r**2) - 2 * t**2 + t)  # a smooth test function, spherically symmetric
box = diff(test, r, 2) + 2 / r * diff(test, r) - diff(test, t, 2) / c**2
# As distributions, <box G, f> = <G, box f>
#   = integral over space of box f(x, |x|/c) / |x| = 4 pi integral_0^oo r box f(r, r/c) dr,
# and the equation says it is -4 pi f(0, 0).
for speed in (1.0, 1.7, 3.0):
    integrand = lambdify(r, (r * box).subs(c, speed).subs(t, r / speed), "numpy")
    pairing = 4 * np.pi * quad(integrand, 0, np.inf)[0]
    assert abs(pairing + 4 * np.pi * float(test.subs({r: 0, t: 0}))) < 1e-8
expected = True
```
