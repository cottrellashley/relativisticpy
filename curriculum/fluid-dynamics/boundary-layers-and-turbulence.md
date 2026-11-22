# Boundary layers and turbulence

Two results beyond what the language can compute today: the Blasius boundary layer, a
boundary-value problem on a half-line with a condition at infinity, and Kolmogorov's
spectrum of turbulence, which follows from dimensions alone.

## The Blasius boundary layer

**Level 5** · `fluid.boundary-layer.blasius` · Blasius (1908), Z. Math. Phys. 56, 1

The boundary layer on a flat plate is self-similar: its stream function \(f(\eta)\) solves
\(f''' + \frac{1}{2} f f'' = 0\), with no slip at the plate and the free stream far away,
\(f'(\infty) = 1\). The friction on the plate is set by \(f''(0) \approx 0.33206\), a
number that only a numerical solution gives.

```latex
f''' + \frac{1}{2} f f'' = 0, \quad f(0) = f'(0) = 0, \quad f'(\infty) = 1
f''(0)
```

```python
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import brentq


def excess(curvature, end=20.0):  # f'(end) - 1, from f(0) = f'(0) = 0 and f''(0) = curvature
    def blasius(eta, f):
        return [f[1], f[2], -f[0] * f[2] / 2]

    solution = solve_ivp(blasius, [0, end], [0, 0, curvature], rtol=1e-12, atol=1e-14)
    return solution.y[1, -1] - 1


expected = brentq(excess, 0.1, 1.0, xtol=1e-14)  # by shooting
assert abs(expected - 0.33206) < 1e-5
assert abs(brentq(lambda s: excess(s, 30.0), 0.1, 1.0, xtol=1e-14) - expected) < 1e-9
# and by collocation, the condition at infinity imposed at eta = 20
eta = np.linspace(0, 20, 401)
start = np.vstack([eta - 1 + np.exp(-eta), 1 - np.exp(-eta), np.exp(-eta)])
collocation = solve_bvp(
    lambda e, f: np.vstack([f[1], f[2], -f[0] * f[2] / 2]),
    lambda f0, f1: np.array([f0[0], f0[1], f1[1] - 1]),
    eta,
    start,
    tol=1e-8,
    max_nodes=100000,
)
assert collocation.success and abs(collocation.sol(0)[2] - expected) < 1e-7
tolerance = 1e-6
```

## Kolmogorov's spectrum

**Level 5** · `fluid.turbulence.kolmogorov` · Pope (2000), chapter 6

In the inertial range of turbulence the energy spectrum \(E(k)\) depends only on the rate of
dissipation \(\varepsilon\) and the wavenumber \(k\). Written as a power law with a
dimensionless constant, its exponents follow from the dimensions, written in brackets, of
length \(L\) and time \(T\): \(E = C \varepsilon^{2/3} k^{-5/3}\). No design covers
dimensions.

```latex
[E(k)] = L^3 T^{-2}, \quad [\varepsilon] = L^2 T^{-3}, \quad [k] = L^{-1}, \quad [C] = 1
E(k) = C \varepsilon^{a} k^{b}
b
```

```python
a, b = symbols("a b")
L, T = symbols("L T", positive=True)
# the powers of length, 3 = 2a - b, and of time, -2 = -3a
solution = solve([Eq(2 * a - b, 3), Eq(-3 * a, -2)], [a, b])
assert solution == {a: Rational(2, 3), b: Rational(-5, 3)}
expected = solution[b]
assert simplify((L**2 / T**3) ** solution[a] * (1 / L) ** expected - L**3 / T**2) == 0
```
