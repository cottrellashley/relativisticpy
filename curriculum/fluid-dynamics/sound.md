# Sound

Small disturbances of a fluid at rest travel as sound: the linearised equations of mass and
momentum give waves of speed \(c\), with \(c^2 = (\partial p / \partial \rho)_s\), the
derivative at constant entropy, which in a perfect gas is \(\gamma p / \rho\). Landau and
Lifshitz (1987) is *Fluid Mechanics*, 2nd edition.

## The speed of sound in air

**Level 1** · `fluid.sound.speed-in-air` · Landau and Lifshitz (1987), §64

\(c = \sqrt{\gamma R T / M}\) for dry air at 20 °C, with \(\gamma = 1.4\) and a molar mass
of 28.97 g/mol: about 343 m/s.

```latex
\gamma := 1.4
R := 8.314
T := 293.15
M := 0.02897
\sqrt{\frac{\gamma R T}{M}}
```

```python
ratio, R, T, M = 1.4, 8.314, 293.15, 0.02897
expected = (ratio * R * T / M) ** 0.5
density = 101325 * M / (R * T)  # of air at one atmosphere, about 1.204 kg/m^3
assert abs((ratio * 101325 / density) ** 0.5 - expected) < 1e-9  # sqrt(gamma p / rho), the same
assert abs(expected - 343.2) < 0.05
```

## The dispersion relation of sound

**Level 3** · `fluid.sound.dispersion` · Landau and Lifshitz (1987), §64

A plane wave of the density perturbation, \(\rho' = R e^{i(kx - \omega t)}\), with the
velocity that the linearised continuity equation asks for, satisfies the linearised Euler
equation \(\rho_0 \partial u / \partial t + c^2 \partial \rho' / \partial x = 0\) only if
\(\omega = \pm ck\). The prime on the perturbation is part of its name, not a derivative.

```latex
\rho'(x, t) := R\, e^{i(k x - \omega t)}
u(x, t) := \frac{\omega R}{\rho_0 k} e^{i(k x - \omega t)}
\frac{\partial \rho'}{\partial t} + \rho_0 \frac{\partial u}{\partial x} = 0
solve\left(\rho_0 \frac{\partial u}{\partial t} + c^2 \frac{\partial \rho'}{\partial x} = 0, \omega\right)
```

```python
x, t, R, k, omega, c, rho_0 = symbols("x t R k omega c rho_0")
phase = exp(I * (k * x - omega * t))
density = R * phase
u = omega * R / (rho_0 * k) * phase
assert simplify(diff(density, t) + rho_0 * diff(u, x)) == 0  # continuity holds
expected = solve(Eq(rho_0 * diff(u, t) + c**2 * diff(density, x), 0), omega)  # in SymPy's order
assert expected == [-c * k, c * k]
# with omega = c k the perturbation solves the wave equation, travelling at the speed c
assert simplify((diff(density, t, 2) - c**2 * diff(density, x, 2)).subs(omega, c * k)) == 0
```

## The speed of sound at constant entropy

**Level 4** · `fluid.sound.adiabatic` · Landau and Lifshitz (1987), §64

The speed of sound is \(c^2 = (\partial p / \partial \rho)_s\), a derivative at constant
entropy written as thermodynamics writes it. Along an adiabat of a perfect gas,
\(p = p_0 (\rho / \rho_0)^\gamma\), it is \(\gamma p_0 / \rho_0\) at the density
\(\rho_0\).

```latex
p := p_0 \left(\frac{\rho}{\rho_0}\right)^\gamma
\left. \left(\frac{\partial p}{\partial \rho}\right)_s \right|_{\rho = \rho_0}
```

```python
p_0, rho_0, ratio, rho = symbols("p_0 rho_0 gamma rho", positive=True)
p = p_0 * (rho / rho_0) ** ratio
expected = simplify(diff(p, rho).subs(rho, rho_0))
assert expected == ratio * p_0 / rho_0
# Newton took the isothermal derivative, of p = p_0 rho / rho_0, smaller by the factor gamma
assert simplify(expected / diff(p_0 * rho / rho_0, rho)) == ratio
```
