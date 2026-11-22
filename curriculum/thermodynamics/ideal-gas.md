# Ideal gases

The processes of an ideal gas as an introductory course writes them: adiabatic
compression, the entropy of mixing, and the first law along an adiabat, where the work
the gas does is the energy it loses. Schroeder (2000) is *An Introduction to Thermal
Physics*.

## The temperature after adiabatic compression

**Level 1** · `thermo.ideal-gas.adiabatic-temperature` · Schroeder (2000), section 1.5

Air at 300 K compressed adiabatically to a twentieth of its volume, as in a diesel
engine: \(T_2 = T_1 (V_1/V_2)^{\gamma - 1}\), with \(\gamma = 7/5\) for a diatomic gas,
about 994 K.

```latex
\left. T_1 \left(\frac{V_1}{V_2}\right)^{\gamma - 1} \right|_{T_1 = 300,\ V_1 = 20 V_2,\ \gamma = 7/5}
```

```python
from scipy.integrate import solve_ivp

expected = 300 * Integer(20) ** Rational(2, 5)
assert abs(float(expected) - 994.3) < 0.1
# the same from the first law, C_V dT = -p dV = -(R T / V) dV with C_V = 5R/2, integrated
adiabat = solve_ivp(lambda V, T: -T / (2.5 * V), [20, 1], [300], rtol=1e-12, atol=1e-12)
assert abs(adiabat.y[0, -1] - float(expected)) < 1e-6
```

## The entropy of mixing

**Level 1** · `thermo.ideal-gas.mixing-entropy` · Schroeder (2000), section 2.6

One mole each of two different ideal gases, mixed at the same temperature and pressure:
\(\Delta S = -R(n_A \ln x_A + n_B \ln x_B) = 2R \ln 2\), about 11.5 J/K, with
\(R = 8.314\) J/(mol K) and the mole fractions \(x_A = x_B = 1/2\).

```latex
\left. -R \left(n_A \ln x_A + n_B \ln x_B\right) \right|_{R = 8.314,\ n_A = 1,\ n_B = 1,\ x_A = 1/2,\ x_B = 1/2}
```

```python
R = Float("8.314")
expected = -R * (log(Rational(1, 2)) + log(Rational(1, 2)))
# second route: each gas expands at constant temperature into twice its volume,
# gaining n R ln(V_f / V_i) = R ln 2
assert abs(float(expected) - 2 * 8.314 * np.log(2)) < 1e-12
assert abs(float(expected) - 11.53) < 0.01
```

## The work of an adiabatic expansion

**Level 3** · `thermo.ideal-gas.adiabatic-work` · Schroeder (2000), section 1.5

Along an adiabat \(pV^\gamma\) is constant, with \(\gamma = 1 + nR/C_V\) and \(C_V\) the
heat capacity of the whole gas, all \(n\) moles, and \(TV^{\gamma - 1}\) is constant too. The work the gas does, \(\int p\,dV\), is what its energy loses,
\(C_V (T_1 - T_2)\): the first law with no heat. The program states, as books do, that the
quantities are positive.

```latex
n, R, C_V, T_1, V_1, V_2 > 0
\gamma := 1 + \frac{n R}{C_V}
p(V) := \frac{n R T_1}{V_1} \left(\frac{V_1}{V}\right)^{\gamma}
T_2 := T_1 \left(\frac{V_1}{V_2}\right)^{\gamma - 1}
\int_{V_1}^{V_2} p(V) \, dV = C_V (T_1 - T_2)
```

```python
from scipy.integrate import quad

n, R, C_V, T_1, V_1, V_2, V = symbols("n R C_V T_1 V_1 V_2 V", positive=True)
gamma = 1 + n * R / C_V
p = n * R * T_1 / V_1 * (V_1 / V) ** gamma
T_2 = T_1 * (V_1 / V_2) ** (gamma - 1)
work = integrate(p, (V, V_1, V_2), conds="none")
assert simplify(work - C_V * (T_1 - T_2)) == 0
# numerically: a mole of a diatomic gas expanding from 1 to 3 litres, starting at 300 K
values = {n: 1, R: 8.314, C_V: 2.5 * 8.314, T_1: 300, V_1: 1e-3, V_2: 3e-3}
pressure = lambdify(V, p.subs(values))
numeric = quad(pressure, 1e-3, 3e-3, epsabs=0, epsrel=1e-13)[0]
assert abs(numeric - float((C_V * (T_1 - T_2)).subs(values))) < 1e-9 * numeric
expected = True
```
