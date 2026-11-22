# Cycles and entropy

Heat engines and the entropy that limits them: the efficiencies of the Otto and Carnot
cycles, the entropy of a solid near absolute zero, and Clausius's equality, which makes
the entropy a function of the state where the heat is not. Schroeder (2000) is *An
Introduction to Thermal Physics*; Fermi (1937) is *Thermodynamics*.

## The efficiency of the Otto cycle

**Level 1** · `thermo.cycles.otto-efficiency` · Schroeder (2000), section 4.3

The idealised petrol engine, compressing air (\(\gamma = 7/5\)) by a ratio \(r = 8\):
its efficiency is \(1 - 1/r^{\gamma - 1}\), about 56%.

```latex
\left. 1 - \frac{1}{r^{\gamma - 1}} \right|_{r = 8,\ \gamma = 7/5}
```

```python
expected = 1 - 1 / Integer(8) ** Rational(2, 5)
assert abs(float(expected) - 0.5647) < 1e-4
# second route: the heats exchanged at constant volume, C_V times the temperature changes,
# between two adiabats on which T V^(gamma - 1) is constant
T_1, T_4 = 300.0, 500.0  # before the compression and before the exhaust
T_2, T_3 = T_1 * 8**0.4, T_4 * 8**0.4
assert abs(1 - (T_4 - T_1) / (T_3 - T_2) - float(expected)) < 1e-12
```

## The efficiency of the Carnot cycle

**Level 3** · `thermo.cycles.carnot` · Schroeder (2000), section 4.1

An ideal gas taken around two isotherms, at \(T_h\) and \(T_c\), joined by two adiabats.
The heats exchanged on the isotherms, with the volumes at the corners fixed by
\(TV^{\gamma - 1}\) being constant on the adiabats, give the efficiency
\(1 - Q_c/Q_h = 1 - T_c/T_h\), whatever the gas's \(\gamma\). The gas expands on the hot
isotherm, from \(V_a\) to \(V_b > V_a\), so that \(Q_h > 0\).

```latex
T_h, T_c, V_a, V_b, n, R > 0
T_h > T_c
V_b > V_a
\gamma > 1
V_c := V_b \left(\frac{T_h}{T_c}\right)^{1/(\gamma - 1)}
V_d := V_a \left(\frac{T_h}{T_c}\right)^{1/(\gamma - 1)}
Q_h := n R T_h \ln\frac{V_b}{V_a}
Q_c := n R T_c \ln\frac{V_c}{V_d}
1 - \frac{Q_c}{Q_h}
```

```python
from scipy.integrate import quad

T_h, T_c, V_a, V_b, n, R, gamma = symbols("T_h T_c V_a V_b n R gamma", positive=True)
V_c = V_b * (T_h / T_c) ** (1 / (gamma - 1))
V_d = V_a * (T_h / T_c) ** (1 / (gamma - 1))
Q_h = n * R * T_h * log(V_b / V_a)
Q_c = n * R * T_c * log(V_c / V_d)
expected = 1 - T_c / T_h
assert simplify(1 - Q_c / Q_h - expected) == 0
# second route: the work around the cycle, the integral of p dV on each leg, over Q_h, for a
# mole of monatomic gas (gamma = 5/3) between 500 K and 300 K, with V_a = 1 and V_b = 2
k = 8.314  # n R
corner_c, corner_d = 2 * (500 / 300) ** 1.5, (500 / 300) ** 1.5
work = (
    quad(lambda v: k * 500 / v, 1, 2)[0]  # the isotherm at T_h
    + quad(lambda v: k * 500 * 2 ** (2 / 3) / v ** (5 / 3), 2, corner_c)[0]  # p V^gamma fixed
    + quad(lambda v: k * 300 / v, corner_c, corner_d)[0]  # the isotherm at T_c
    + quad(lambda v: k * 300 * corner_d ** (2 / 3) / v ** (5 / 3), corner_d, 1)[0]
)
assert abs(work / (k * 500 * np.log(2)) - (1 - 300 / 500)) < 1e-9
```

## The entropy of a solid near absolute zero

**Level 2** · `thermo.entropy.third-law` · Schroeder (2000), sections 3.2 and 7.5

Near absolute zero the heat capacity of an insulating crystal follows Debye's law,
\(C_V = aT^3\). Its entropy, counted from \(S(0) = 0\) as the third law allows,
\(S(T) = \int_0^T C_V(T')/T'\,dT'\), is one third of its heat capacity.

```latex
C_V(T) := a T^3
S(T) := \int_0^T \frac{C_V(T')}{T'} \, dT'
\frac{S(T)}{C_V(T)}
```

```python
from scipy.integrate import quad

a, T, t = symbols("a T t", positive=True)
S = integrate(a * t**3 / t, (t, 0, T))
expected = simplify(S / (a * T**3))
assert expected == Rational(1, 3)
# numerically, for a = 2 and T = 5, say: the ratio depends on neither
assert abs(quad(lambda x: 2 * x**3 / x, 0, 5)[0] / (2 * 5**3) - 1 / 3) < 1e-12
```

## Clausius's equality

**Level 5** · `thermo.entropy.clausius-equality` · Fermi (1937), chapter IV

The heat a mole of ideal gas takes in reversibly, \(\delta Q = C_V\,dT + (RT/V)\,dV\),
is an inexact differential: around a cycle its integral is the net work done, in general
not zero. Divided by \(T\) it is exact, and its integral vanishes around every cycle:
Clausius's equality, which defines the entropy. An inexact differential, and a claim
about every closed path in the state space \(T, V > 0\).

```latex
T, V > 0
\delta Q := C_V \, dT + \frac{R T}{V} \, dV
\oint \frac{\delta Q}{T} = 0
```

```python
from scipy.integrate import quad

T, V, C_V, R = symbols("T V C_V R", positive=True)
P, Q = C_V, R * T / V  # delta Q = P dT + Q dV
assert simplify(diff(Q / T, T) - diff(P / T, V)) == 0  # closed, so exact on T, V > 0
assert simplify(diff(Q, T) - diff(P, V)) != 0  # but delta Q itself is not
S = C_V * log(T) + R * log(V)  # a potential: dS = delta Q / T
assert simplify(diff(S, T) - P / T) == 0 and simplify(diff(S, V) - Q / T) == 0
# numerically, around an ellipse in the (V, T) plane, for a monatomic gas, C_V = 3R/2
R_, C_ = 8.314, 1.5 * 8.314


def heat(s):  # delta Q along V = 0.02 + 0.01 cos s, T = 300 + 80 sin s
    v, t = 0.02 + 0.01 * np.cos(s), 300 + 80 * np.sin(s)
    return C_ * 80 * np.cos(s) + R_ * t / v * (-0.01 * np.sin(s))


assert abs(quad(heat, 0, 2 * np.pi, limit=200)[0]) > 100  # joules: heat is not a state function
entropy = quad(lambda s: heat(s) / (300 + 80 * np.sin(s)), 0, 2 * np.pi, limit=200)[0]
assert abs(entropy) < 1e-9
expected = True
```
