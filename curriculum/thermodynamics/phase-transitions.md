# Phase transitions

Liquid and vapour in equilibrium: the Clapeyron equation for the slope of the
vapour-pressure curve, its integrated form with an ideal vapour, the latent heat read
from the curve, and Maxwell's equal-area construction for the van der Waals gas.
Schroeder (2000) is *An Introduction to Thermal Physics*; Callen (1985) is
*Thermodynamics and an Introduction to Thermostatistics*, 2nd edition; Pathria and Beale
(2011) is *Statistical Mechanics*, 3rd edition.

## The slope of the vapour-pressure curve

**Level 1** · `thermo.phases.clapeyron-slope` · Schroeder (2000), section 5.3

The Clapeyron equation, \(dp/dT = L/T\Delta v\), for water boiling at 373.15 K, with
\(L = 2.257 \times 10^6\) J/kg and \(\Delta v = 1.671\) m³/kg, the volume of a kilogram
of steam less that of the water: about 3.6 kPa per kelvin.

```latex
\left. \frac{L}{T \Delta v} \right|_{L = 2.257 \times 10^{6},\ T = 373.15,\ \Delta v = 1.671}
```

```python
expected = 2.257e6 / (373.15 * 1.671)
# second route: the slope of the saturation pressure of water in IAPWS-IF97 (region 4)
n = [0, 1167.0521452767, -724213.16703206, -17.073846940092, 12020.82470247, -3232555.0322333]
n += [14.91510861353, -4823.2657361591, 405113.40542057, -0.23855557567849, 650.17534844798]


def saturation(T):  # in pascals
    theta = T + n[9] / (T - n[10])
    A = theta**2 + n[1] * theta + n[2]
    B = n[3] * theta**2 + n[4] * theta + n[5]
    C = n[6] * theta**2 + n[7] * theta + n[8]
    return (2 * C / (-B + np.sqrt(B**2 - 4 * A * C))) ** 4 * 1e6


assert abs(saturation(300) - 3536.59) < 0.01  # the formulation's own check value
assert abs((saturation(373.16) - saturation(373.14)) / 0.02 / expected - 1) < 1e-3
```

## Water boiling on Everest

**Level 2** · `thermo.phases.boiling-on-everest` · Schroeder (2000), section 5.3

The Clausius–Clapeyron equation integrated with an ideal vapour, the liquid's volume
neglected beside it, and a constant latent heat, \(\ln(p_2/p_1) = -(L/R)(1/T_2 - 1/T_1)\), solved for the temperature at which water boils
under 33.7 kPa, the pressure on the summit of Everest: about 344 K, or 71 °C, with
\(L = 40.66\) kJ/mol.

```latex
L := 40.66 \times 10^{3}
R := 8.314
T_1 := 373.15
p_1 := 101.325
p_2 := 33.7
solve\left(\ln\frac{p_2}{p_1} = -\frac{L}{R}\left(\frac{1}{T_2} - \frac{1}{T_1}\right), T_2\right)
```

```python
from scipy.integrate import solve_ivp

L, R, T_1, p_1, p_2 = 40.66e3, 8.314, 373.15, 101.325, 33.7
expected = [1 / (1 / T_1 - R / L * np.log(p_2 / p_1))]
assert abs(expected[0] - 344.24) < 0.01
# second route: integrate dp/dT = L p / R T^2 down from the normal boiling point
# until the pressure falls to 33.7 kPa
reached = lambda T, p: p[0] - p_2
reached.terminal = True
descent = solve_ivp(
    lambda T, p: L * p / (R * T**2), [T_1, 300], [p_1], events=reached, rtol=1e-12, atol=1e-12
)
assert abs(descent.t_events[0][0] - expected[0]) < 1e-6
```

## The latent heat from the vapour-pressure curve

**Level 2** · `thermo.phases.latent-heat` · Schroeder (2000), section 5.3

With an ideal vapour, the liquid's volume neglected beside it, and a constant latent heat,
\(\ln p\) is linear in \(1/T\), with
slope \(-L/R\): \(L = -R\, d\ln p/d(1/T)\), a derivative with respect to \(1/T\), as the
curve is plotted. For \(p = p_0 e^{-B/T}\) it gives \(L = RB\).

```latex
p := p_0 e^{-B/T}
-R \frac{d \ln p}{d(1/T)}
```

```python
p_0, B, R, T, u = symbols("p_0 B R T u", positive=True)
p = p_0 * exp(-B / T)
# d/d(1/T) is d/du with u = 1/T
expected = simplify(-R * diff(log(p.subs(T, 1 / u)), u))
assert expected == R * B
# second route, by the chain rule: -R (d ln p/dT) / (d(1/T)/dT)
assert simplify(-R * diff(log(p), T) / diff(1 / T, T) - R * B) == 0
```

## Maxwell's equal-area construction

**Level 5** · `thermo.phases.maxwell-construction` · Callen (1985), chapter 9; Pathria and Beale (2011), section 12.2

Below the critical temperature a van der Waals isotherm has a loop, and the pressure
\(p_s\) at which liquid and vapour coexist cuts it into two equal areas. In reduced units,
at \(T_r = 9/10\), \(p_s \approx 0.647\). Three equations in \(V_l\), \(V_g\) and
\(p_s\), one of them an integral between two of the unknowns, whose trivial solutions,
\(V_l = V_g\), an inequality excludes.

```latex
T_r := \frac{9}{10}
p(V) := \frac{8 T_r}{3 V - 1} - \frac{3}{V^2}
p(V_l) = p_s, \quad p(V_g) = p_s, \quad \frac{1}{3} < V_l < V_g
\int_{V_l}^{V_g} p(V) \, dV = p_s (V_g - V_l)
p_s
```

```python
from scipy.integrate import quad
from scipy.optimize import fsolve

T_r = 0.9


def p(v):
    return 8 * T_r / (3 * v - 1) - 3 / v**2


def equal_areas(x):
    v_l, v_g, p_s = x
    return [p(v_l) - p_s, p(v_g) - p_s, quad(p, v_l, v_g)[0] - p_s * (v_g - v_l)]


v_l, v_g, p_s = fsolve(equal_areas, [0.6, 2.3, 0.65], xtol=1e-12)
assert 1 / 3 < v_l < v_g and max(abs(r) for r in equal_areas([v_l, v_g, p_s])) < 1e-10
expected = float(p_s)
# second route: equal Gibbs free energies, g = f + p v with f = -(8T/3) ln(3v - 1) - 3/v
g = lambda v: -(8 * T_r / 3) * np.log(3 * v - 1) - 3 / v + p(v) * v
liquid, gas = fsolve(lambda x: [p(x[0]) - p(x[1]), g(x[0]) - g(x[1])], [0.6, 2.3], xtol=1e-12)
assert abs(p(liquid) - expected) < 1e-9 and abs(gas - v_g) < 1e-8
assert abs(expected - 0.6470) < 1e-4
tolerance = 1e-6
```
