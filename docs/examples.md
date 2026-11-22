# Examples

> **For:** anyone who learns from worked examples. **You will learn:** what
> RelativisticPy makes of whole derivations, in General Relativity, quantum mechanics and
> finance, with every result shown.

Each example is a file in the `examples` folder of the repository. A Markdown example is
a notebook: its displays are the program, and its results are typeset under each
display, as `relativisticpy examples/NAME.md --render` prints them. A Python example is
shown with what it prints. Every example runs in the tests, and this page is written
from them, by `tools/examples.py`.

| Example | File | What it shows |
|---|---|---|
| [The Schwarzschild solution](#the-schwarzschild-solution) | `schwarzschild.md` | The Schwarzschild metric, derived from Einstein's equations in vacuum |
| [The Kretschmann scalar, from scratch](#the-kretschmann-scalar-from-scratch) | `kretschmann.md` | The Kretschmann scalar by hand, index by index, checked against the built-in |
| [The Friedmann equations](#the-friedmann-equations) | `friedmann.md` | The Friedmann equations and the continuity equation of an expanding universe |
| [Geodesics from a Lagrangian](#geodesics-from-a-lagrangian) | `geodesics.md` | The geodesic equations from a Lagrangian, checked against the Christoffel symbols |
| [Light near a black hole](#light-near-a-black-hole) | `light.py` | Python: light bent and captured by a black hole, integrated numerically |
| [Two spins](#two-spins) | `spins.md` | Two spins: an operator on a product space, its exact spectrum, and the evolution of a spin |
| [A portfolio of 500 assets, in index notation](#a-portfolio-of-500-assets-in-index-notation) | `portfolio.py` | Python: 500 assets, the tangency portfolio, risk as a metric, and real factors told from noise |

## The Schwarzschild solution

Every equation on this page is both LaTeX, which GitHub typesets, and
RelativisticPy, which computes. To see the results typeset under each step:

```bash
relativisticpy examples/schwarzschild.md --render
```

### The ansatz

Outside a static, spherically symmetric star, spacetime can be written with two
unknown functions of the radius, $A(r)$ and $B(r)$:

$$
\begin{aligned}
\mathrm{Coordinates} &:= [t, r, \theta, \phi] \\
g_{\mu\nu} &:= \mathrm{diag}\left(-A(r),\; B(r),\; r^2,\; r^2 \sin^2\theta\right)
\end{aligned}
$$

### The field equations

In vacuum the Ricci tensor vanishes. Three of its components carry the
information we need:

$$
\begin{aligned}
E_0 &:= R_{tt} \\
E_1 &:= R_{rr} \\
E_2 &:= R_{\theta\theta}
\end{aligned}
$$

This combination of the first two involves only first derivatives, and fixes
$B$ in terms of $A$:

$$
\begin{aligned}
B_1 &:= \mathrm{RHS}\left(\mathrm{dsolve}\left((E_0 B(r) + E_1 A(r))\, r\, B(r),\; B(r)\right)\right) \\
B_1
\end{aligned}
$$

$$
\frac{C1}{A\left(r\right)}
$$

Substituting into the third leaves an equation for $A$ alone:

$$
\begin{aligned}
A_1 &:= \mathrm{RHS}\left(\mathrm{dsolve}\left(\mathrm{simplify}\left(E_2 \big|_{B(r) = B_1}\right),\; A(r)\right)\right) \\
A_1
\end{aligned}
$$

$$
C1 + \frac{C2}{r}
$$

### The integration constants

Far from the star, the radial acceleration of a particle at rest must be
Newton's, $-GM/r^2$. With $A = 1 + C/r$ and $B = 1/A$:

$$
\begin{aligned}
g_{\mu\nu} &:= \mathrm{diag}\left(-(1 + \tfrac{C}{r}),\; \frac{1}{1 + \tfrac{C}{r}},\; r^2,\; r^2 \sin^2\theta\right) \\
\mathrm{solve}\left(\lim_{r \to \infty} r^2\, \Gamma^{r}_{tt} - G M,\; C\right)
\end{aligned}
$$

$$
[- 2 G M]
$$

So $C = -2GM$, and spacetime around any spherical mass is the Schwarzschild
metric. Its curvature invariant shows that $r = 2GM$ is no singularity at all,
while $r = 0$ is:

$$
\begin{aligned}
g_{\mu\nu} &:= \mathrm{diag}\left(-(1 - \tfrac{2 G M}{r}),\; \frac{1}{1 - \tfrac{2 G M}{r}},\; r^2,\; r^2 \sin^2\theta\right) \\
R_{abcd} R^{abcd}
\end{aligned}
$$

$$
\frac{48 G^{2} M^{2}}{r^{6}}
$$

## The Kretschmann scalar, from scratch

Built-in tensors are a convenience, not a requirement. Here every step of the
calculation of $K = R_{abcd} R^{abcd}$ is written out by hand, in the
plain-text notation, and checked against the built-in result.

```bash
relativisticpy examples/kretschmann.md --render
```

Schwarzschild spacetime, with $G$ and $c$ kept explicit:

$$
\begin{aligned}
Coordinates &:= [t, r, \theta, \phi] \\
g_{\mu\nu} &:= \mathrm{diag}\left(-(1 - \frac{2 G M}{c^2 r}),\; \frac{1}{1 - \frac{2 G M}{c^2 r}},\; r^2,\; r^2 \sin^2\theta\right)
\end{aligned}
$$

The Christoffel symbols, from their definition:

$$
\Gamma^{a}_{c f} := \frac{1}{2} g^{a b} \left(\partial_{c} g_{b f} + \partial_{f} g_{b c} - \partial_{b} g_{c f}\right)
$$

The Riemann tensor, from the Christoffel symbols:

$$
Riemann^{a}_{m b n} := \partial_{b} \Gamma^{a}_{n m} - \partial_{n} \Gamma^{a}_{b m} + \Gamma^{a}_{b l} \Gamma^{l}_{n m} - \Gamma^{a}_{n l} \Gamma^{l}_{b m}
$$

Its indices raised and lowered by hand, then contracted:

$$
\begin{aligned}
Upper^{a f h s} &:= g^{s q} g^{h c} g^{f b} \, Riemann^{a}_{b c q} \\
Lower_{a f h s} &:= g_{a n} \, Riemann^{n}_{f h s} \\
K &:= \mathrm{simplify}\left(Upper^{a f h s} Lower_{a f h s}\right) \\
K
\end{aligned}
$$

$$
\frac{48 G^{2} M^{2}}{c^{4} r^{6}}
$$

The built-in Kretschmann scalar agrees:

$$
K - R_{abcd} R^{abcd}
$$

$$
0
$$

## The Friedmann equations

A homogeneous, isotropic universe has the FLRW metric, with spatial curvature
$k$ and a scale factor $a(t)$ to be found:

```bash
relativisticpy examples/friedmann.md --render
```

$$
\begin{aligned}
Coordinates &:= [t, r, \theta, \phi] \\
ds^2 &= -dt^2 + a(t)^2 \left(\frac{dr^2}{1 - k r^2} + r^2 d\Omega^2\right)
\end{aligned}
$$

Fill it with a perfect fluid at rest, of density $\rho(t)$ and pressure $p(t)$:

$$
\begin{aligned}
u^{\mu} &:= [1, 0, 0, 0] \\
T^{\mu\nu} &:= (\rho(t) + p(t))\, u^{\mu} u^{\nu} + p(t)\, g^{\mu\nu}
\end{aligned}
$$

Einstein's equations $G_{\mu\nu} = 8 \pi G\, T_{\mu\nu}$ then hold component by
component. The time component is the first Friedmann equation,
$\left(\dot a / a\right)^2 = \frac{8\pi G}{3}\rho - k/a^2$:

$$
\begin{aligned}
F_1 &:= G_{tt} = 8 \pi G\, T_{tt} \\
\mathrm{solve}(F_1, \rho(t))
\end{aligned}
$$

$$
[\frac{3 \left(k + \left(a'\left(t\right)\right)^{2}\right)}{8 \pi G a\left(t\right)^{2}}]
$$

and a radial component gives the second, for the acceleration $\ddot a$:

$$
\begin{aligned}
F_2 &:= G^{r}{}_{r} = 8 \pi G\, T^{r}{}_{r} \\
\mathrm{solve}(F_2, \ddot{a(t)})
\end{aligned}
$$

$$
[\frac{- 8 \pi G a\left(t\right)^{2} p\left(t\right) - k - \left(a'\left(t\right)\right)^{2}}{2 a\left(t\right)}]
$$

Energy conservation, $\nabla_\mu T^{\mu\nu} = 0$, is the continuity equation
$\dot\rho + 3 \frac{\dot a}{a}(\rho + p) = 0$:

$$
\nabla_{\mu} T^{\mu t}
$$

$$
\frac{a\left(t\right) \rho'\left(t\right) + 3 p\left(t\right) a'\left(t\right) + 3 \rho\left(t\right) a'\left(t\right)}{a\left(t\right)}
$$

## Geodesics from a Lagrangian

A free particle in curved spacetime extremises
$L = g_{\mu\nu} \dot x^\mu \dot x^\nu$. Its Euler-Lagrange equations are the
geodesic equations.

```bash
relativisticpy examples/geodesics.md --render
```

$$
\begin{aligned}
g_{\mu\nu} &:= \mathrm{Schwarzschild}(M) \\
x^{\mu} &:= [t(\tau), r(\tau), \theta(\tau), \phi(\tau)]
\end{aligned}
$$

The Lagrangian, along the path:

$$
L := g_{\mu\nu} \dot{x}^\mu \dot{x}^\nu \Big|_{t = t(\tau),\; r = r(\tau),\; \theta = \theta(\tau),\; \phi = \phi(\tau)}
$$

The Euler-Lagrange equations, one per coordinate:

$$
EL_{\mu} := \frac{d}{d\tau} \frac{\partial L}{\partial \dot{x}^\mu} - \frac{\partial L}{\partial x^\mu}
$$

The $\theta$ equation, for example:

$$
EL_{\theta}
$$

$$
2 r\left(\tau\right)^{2} \theta''\left(\tau\right) - 2 r\left(\tau\right)^{2} \sin{\left(\theta\left(\tau\right) \right)} \cos{\left(\theta\left(\tau\right) \right)} \left(\phi'\left(\tau\right)\right)^{2} + 4 r\left(\tau\right) r'\left(\tau\right) \theta'\left(\tau\right)
$$

Each is twice the geodesic equation
$\ddot x^\nu + \Gamma^\nu{}_{\alpha\beta}\, \dot x^\alpha \dot x^\beta = 0$
with its index lowered, which RelativisticPy confirms component by component:

$$
EL_{\mu} = \left. 2 g_{\mu\nu} \left( \ddot{x}^{\nu} + \Gamma^{\nu}_{\alpha\beta}\, \dot{x}^{\alpha} \dot{x}^{\beta} \right) \right|_{t = t(\tau),\; r = r(\tau),\; \theta = \theta(\tau),\; \phi = \phi(\tau)}
$$

$$
\text{equation}\ {}_{\mu} \tag*{holds in all 4 components}
$$

## Light near a black hole

```python
"""Light near a black hole.

Rays of light pass a Schwarzschild black hole of mass M (with G = c = 1). A ray
passing far away is bent by 4M/b, Einstein's deflection of starlight; a ray
passing closer than b = 3√3 M is captured. Both numbers come out of the
geodesic equation, integrated numerically.

    python examples/light.py

This needs NumPy and SciPy: pip install "relativisticpy[numeric]".
"""

import numpy as np

from relativisticpy import spacetimes
from relativisticpy.numeric import four_velocity, geodesic

M = 1.0
SCHWARZSCHILD = spacetimes.schwarzschild()


def ray(b, r0, length):
    """Light sent inwards from radius r0 with impact parameter b, until it falls in
    (reaches r = 2.5M, inside the photon sphere) or has travelled ``length``."""
    start = [0.0, r0, np.pi / 2, 0.0]  # t, r, θ, φ
    inward = -np.sqrt(1 - (1 - 2 * M / r0) * b**2 / r0**2)
    u = four_velocity(SCHWARZSCHILD, start, [inward, 0.0, b / r0**2], kind="null", M=M)
    return geodesic(SCHWARZSCHILD, start, u, (0.0, length), M=M, stop=lambda x, _: x[1] - 2.5 * M)


def deflection(b, r0=10_000.0):
    """How far a ray passing at impact parameter b is bent from a straight line."""
    path = ray(b, r0, length=2 * r0)
    r, phi = path["r"][-1], path["phi"][-1]
    straight = np.pi - np.arcsin(b / r0) - np.arcsin(b / r)  # the angle a straight line sweeps
    return phi - straight


def critical_impact_parameter(low=4.0, high=7.0, steps=30):
    """The impact parameter dividing captured rays from escaping ones, by bisection."""
    for _ in range(steps):
        middle = (low + high) / 2
        if ray(middle, r0=1000.0, length=3000.0).stopped:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def main():
    b = 100 * M
    einstein = 4 * M / b
    next_order = einstein + 15 * np.pi / 4 * (M / b) ** 2
    print(f"Light passing at b = {b:g}M is bent by {deflection(b):.5f} rad.")
    print(f"  Einstein's 4M/b:                      {einstein:.5f} rad")
    print(f"  with the next order, + 15π/4 (M/b)²:  {next_order:.5f} rad")

    critical = critical_impact_parameter()
    print(f"Light is captured below b = {critical:.5f}M,")
    print(f"  as the photon sphere predicts: 3√3 M = {3 * np.sqrt(3) * M:.5f}M.")


if __name__ == "__main__":
    main()
```

It prints:

```text
Light passing at b = 100M is bent by 0.04122 rad.
  Einstein's 4M/b:                      0.04000 rad
  with the next order, + 15π/4 (M/b)²:  0.04118 rad
Light is captured below b = 5.19615M,
  as the photon sphere predicts: 3√3 M = 5.19615M.
```

## Two spins

Quantum mechanics in index notation. The state of a spin is a vector in a complex
space of two dimensions, $\mathcal{H} = \mathbb{C}^2$, and an operator on it is a
tensor with one index up and one down:

```bash
relativisticpy examples/spins.md --render
```

$$
\begin{aligned}
\mathcal{H} &:= \mathbb{C}^{2} \\
i, j, k, l &\in \mathcal{H}
\end{aligned}
$$

The Pauli matrices:

$$
\begin{aligned}
X^{i}{}_{j} &:= [[0, 1], [1, 0]] \\
Y^{i}{}_{j} &:= [[0, -i], [i, 0]] \\
Z^{i}{}_{j} &:= [[1, 0], [0, -1]]
\end{aligned}
$$

Two spins with a Heisenberg coupling $J$: an operator on $\mathcal{H} \otimes
\mathcal{H}$ is simply a tensor with two indices of each kind,
$H = J\, \vec\sigma_1 \cdot \vec\sigma_2$:

$$
H^{ik}{}_{jl} := J \left( X^i{}_j X^k{}_l + Y^i{}_j Y^k{}_l + Z^i{}_j Z^k{}_l \right)
$$

Its spectrum, exactly: the singlet at $-3J$ and the triplet at $J$, three times.

$$
\operatorname{spec}(H^{ik}{}_{jl})
$$

$$
\begin{aligned}
\operatorname{spec}(H^{i0}{}_{jl}) &= - 3 J \\
\operatorname{spec}(H^{i1}{}_{jl}) &= J \\
\operatorname{spec}(H^{i2}{}_{jl}) &= J \\
\operatorname{spec}(H^{i3}{}_{jl}) &= J
\end{aligned}
$$

The trace of $H$ vanishes, as it must for a sum of traceless products:

$$
\operatorname{tr}(H^{ik}{}_{jl})
$$

$$
0
$$

A single spin in a magnetic field precesses: the operator of its evolution over a time
$t$ is the exponential of the map $Z$, written without indices as a matrix is.

$$
U := \exp\left(-\frac{i \omega t}{2} Z\right)
$$

$$
U^{i}{}_{j}
$$

$$
\begin{aligned}
U^{0}{}_{0} &= e^{- \frac{i \omega t}{2}} \\
U^{1}{}_{1} &= e^{\frac{i \omega t}{2}}
\end{aligned}
$$

## A portfolio of 500 assets, in index notation

```python
r"""A portfolio of 500 assets, in index notation.

Ten years of daily returns on 500 assets, driven by five hidden factors and noise,
are bound to the tensor R_{ta}: t ranges over 2,520 days and a over 500 assets. From
there everything is notation, as it would be written on a page:

- the expected returns and the covariance, as expectations over the days;
- the tangency portfolio of Markowitz, computed with linear solves, never an inverse;
- risk as geometry, with the covariance as the metric of the assets;
- the spectrum of the correlation matrix against the Marchenko–Pastur law of random
  matrices, which tells the five real factors from the noise.

    python examples/portfolio.py

This needs NumPy and SciPy: pip install "relativisticpy[numeric]".
"""

import numpy as np

from relativisticpy import Workbook

DAYS, ASSETS, FACTORS = 2520, 500, 5

rng = np.random.default_rng(7)
factors = rng.normal(0, 0.01, (DAYS, FACTORS))
loadings = rng.normal(0, 1, (FACTORS, ASSETS))
loadings /= np.linalg.norm(loadings, axis=0)  # every asset equally exposed
returns = 0.0004 + 0.6 * factors @ loadings + rng.normal(0, 0.01, (DAYS, ASSETS))

wb = Workbook()
wb["R_{ta}"] = returns
wb.exe(r"""
    \mu_a := \mathbb{E}_t[R_{ta}]
    \Sigma_{ab} := \mathbb{E}_t[(R_{ta} - \mu_a)(R_{tb} - \mu_b)]
    m^a := \frac{(\Sigma^{-1})^{ab} \mu_b}{1_c (\Sigma^{-1})^{cd} \mu_d}
""")
tangency = wb["m"].to_numpy()
print(f"the tangency portfolio's weights sum to {tangency.sum():.6f}")

# Signal and noise: the spectrum of the correlation matrix.
wb["Z_{ta}"] = (returns - returns.mean(axis=0)) / returns.std(axis=0)
wb.exe(r"C_{ab} := \mathbb{E}_t[Z_{ta} Z_{tb}]")
spectrum = wb.expr(r"\operatorname{spec}(C^a{}_b)").to_numpy()
edge = (1 + np.sqrt(ASSETS / DAYS)) ** 2  # the largest eigenvalue of pure noise
real = int((spectrum > edge).sum())
print(f"{real} eigenvalues stand above the Marchenko–Pastur edge of {edge:.2f}")

# Risk as geometry: with the covariance as the metric of the assets, w^a w_a is a
# variance. (Raising an index of an asset uses the metric from now on.)
wb["w^a"] = np.full(ASSETS, 1 / ASSETS)
wb.exe(r"g_{ab} := \Sigma_{ab}")
variance = wb.expr(r"w^a w_a")
print(f"the variance of the equally weighted portfolio is {variance:.3e}")
```

It prints:

```text
the tangency portfolio's weights sum to 1.000000
5 eigenvalues stand above the Marchenko–Pastur edge of 2.09
the variance of the equally weighted portfolio is 2.372e-07
```

**Next:** [Jupyter and notebooks](jupyter.md), to write notebooks of your own, or the
[guides](language.md) behind each step.
