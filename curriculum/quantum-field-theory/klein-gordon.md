# The Klein–Gordon field

The free scalar field, in natural units with the metric \((+, -, -, -)\): its plane
waves, which live on the mass shell, its Lagrangian density and the equation of motion
that follows from it, its Hamiltonian density, and its Feynman propagator with the
\(i\epsilon\) prescription.

## Plane waves on the mass shell

**Level 2** · `qft.klein-gordon.dispersion` · Peskin and Schroeder (1995), section 2.3

A plane wave \(e^{-i(Et - px)}\) solves the Klein–Gordon equation, written with partial
derivatives of a field, when \(E = \sqrt{p^2 + m^2}\).

```latex
E := \sqrt{p^2 + m^2}
\phi(t, x) := e^{-i(Et - px)}
\frac{\partial^2\phi}{\partial t^2} - \frac{\partial^2\phi}{\partial x^2} + m^2\phi = 0
```

```python
t, x, p, m = symbols("t x p m", real=True)
E = sqrt(p**2 + m**2)
phi = exp(-I * (E * t - p * x))
assert simplify(diff(phi, t, 2) - diff(phi, x, 2) + m**2 * phi) == 0
# Off the mass shell it is no solution: the energies allowed are +-E.
E_free = symbols("E")
wave = exp(-I * (E_free * t - p * x))
assert solve(simplify((diff(wave, t, 2) - diff(wave, x, 2) + m**2 * wave) / wave), E_free) == [
    -E,
    E,
]
expected = True
```

## The Hamiltonian density

**Level 3** · `qft.klein-gordon.hamiltonian` · Peskin and Schroeder (1995), section 2.2

From the Lagrangian density, the momentum conjugate to the field,
\(\pi = \partial\mathcal{L}/\partial\dot\phi\), and the Legendre transform
\(\mathcal{H} = \pi\dot\phi - \mathcal{L}\): the energy density is positive. Here \(\pi\)
is a field, not the constant.

```latex
\mathcal{L} := \frac{1}{2}\dot{\phi}^2 - \frac{1}{2}(\nabla\phi)^2 - \frac{1}{2}m^2\phi^2
\pi := \frac{\partial\mathcal{L}}{\partial\dot{\phi}}
\mathcal{H} := \pi\dot{\phi} - \mathcal{L}
\mathcal{H} = \frac{1}{2}\pi^2 + \frac{1}{2}(\nabla\phi)^2 + \frac{1}{2}m^2\phi^2
```

```python
phi_dot, grad_phi, phi, m = symbols("phi_dot grad_phi phi m", real=True)
L = phi_dot**2 / 2 - grad_phi**2 / 2 - m**2 * phi**2 / 2
momentum = diff(L, phi_dot)
assert momentum == phi_dot
H = momentum * phi_dot - L
assert simplify(H - (momentum**2 / 2 + grad_phi**2 / 2 + m**2 * phi**2 / 2)) == 0  # Peskin (2.8)
expected = True
```

## The Klein–Gordon equation from its Lagrangian

**Level 4** · `qft.klein-gordon.euler-lagrange` · Peskin and Schroeder (1995), section 2.2

The Euler–Lagrange equation in index notation, with a derivative of the Lagrangian
density with respect to a derivative of the field,
\(\partial_\mu\big(\partial\mathcal{L}/\partial(\partial_\mu\phi)\big) -
\partial\mathcal{L}/\partial\phi = 0\): for
\(\mathcal{L} = \frac{1}{2}\partial_\mu\phi\,\partial^\mu\phi - \frac{1}{2}m^2\phi^2\) it is
the Klein–Gordon equation.

```latex
\mathcal{L} := \frac{1}{2}\partial_\mu\phi\,\partial^\mu\phi - \frac{1}{2}m^2\phi^2
\partial_\mu\left(\frac{\partial\mathcal{L}}{\partial(\partial_\mu\phi)}\right) - \frac{\partial\mathcal{L}}{\partial\phi} = \partial_\mu\partial^\mu\phi + m^2\phi
```

```python
from sympy.calculus.euler import euler_equations

t, x, y, z, m = symbols("t x y z m", real=True)
phi = Function("phi")(t, x, y, z)
L = (
    diff(phi, t) ** 2 - diff(phi, x) ** 2 - diff(phi, y) ** 2 - diff(phi, z) ** 2
) / 2 - m**2 * phi**2 / 2
[equation] = euler_equations(L, phi, [t, x, y, z])
box = diff(phi, t, 2) - diff(phi, x, 2) - diff(phi, y, 2) - diff(phi, z, 2)  # d_mu d^mu
# SymPy writes dL/dphi - d_mu(dL/d(d_mu phi)) = 0, the negative of the program's left side.
assert equation.rhs == 0
assert simplify(-equation.lhs - (box + m**2 * phi)) == 0
expected = True
```

## The Feynman propagator

**Level 4** · `qft.klein-gordon.feynman-propagator` · Peskin and Schroeder (1995), section 2.4

The integral over \(p^0\) in the Feynman propagator, with the \(i\epsilon\) that moves its
poles off the real axis: for \(t > 0\) the contour closes below, around
\(p^0 = E_p\), and gives \(e^{-iE_pt}/2E_p\). Here \(p^0\) is a component, not a power.

```latex
t > 0
\lim_{\epsilon \to 0^+} \int_{-\infty}^{\infty} \frac{dp^0}{2\pi} \frac{i e^{-ip^0 t}}{(p^0)^2 - E_p^2 + i\epsilon}
```

```python
import numpy as np
from scipy.integrate import quad

t, E_p, epsilon = symbols("t E_p epsilon", positive=True)
p0 = symbols("p0")
pole = sqrt(E_p**2 - I * epsilon)  # just below the real axis
integrand = I * exp(-I * p0 * t) / (2 * pi * (p0**2 - E_p**2 + I * epsilon))
closed_below = -2 * pi * I * residue(integrand, p0, pole)  # clockwise, for t > 0
expected = limit(closed_below, epsilon, 0)
assert simplify(expected - exp(-I * E_p * t) / (2 * E_p)) == 0  # Peskin, section 2.4

# Check the contour numerically at a small epsilon: the integrand is even in p0 but for
# the phase, so the integral is (i/pi) times the cosine transform over p0 > 0.
E_v, t_v, eps_v = 1.3, 0.7, 0.05
re_f = lambda q: (q**2 - E_v**2) / ((q**2 - E_v**2) ** 2 + eps_v**2)
im_f = lambda q: -eps_v / ((q**2 - E_v**2) ** 2 + eps_v**2)
total = 0j
for f, unit in ((re_f, 1), (im_f, 1j)):
    near = quad(lambda q: f(q) * np.cos(q * t_v), 0, 4, points=[E_v], limit=500)[0]
    far = quad(f, 4, np.inf, weight="cos", wvar=t_v)[0]
    total += unit * (near + far)
numeric = 1j / np.pi * total
pole_v = np.sqrt(E_v**2 - 1j * eps_v)
assert abs(numeric - np.exp(-1j * pole_v * t_v) / (2 * pole_v)) < 1e-8
```
