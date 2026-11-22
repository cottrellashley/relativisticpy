# Thermodynamics

Engines, equations of state and the Maxwell relations, in the notation of thermodynamics
books, where a partial derivative names the variable held fixed.

## The efficiency of a Carnot engine

**Level 1** · `statmech.thermo.carnot` · Callen (1985), section 4.6

The largest efficiency of an engine working between reservoirs at 500 K and 300 K,
\(1 - T_c / T_h\), with an evaluation bar.

```latex
\left. 1 - \frac{T_c}{T_h} \right|_{T_c = 300, T_h = 500}
```

```python
expected = 1 - Rational(300, 500)
assert expected == Rational(2, 5)
```

## Isothermal work of an ideal gas

**Level 1** · `statmech.thermo.isothermal-work` · standard

The work done by an ideal gas expanding at constant temperature, \(\int p \, dV\) with
\(p = nRT/V\): \(nRT \ln(V_2 / V_1)\).

```latex
\int_{V_1}^{V_2} \frac{n R T}{V} \, dV
```

```python
n, R, T, V, V_1, V_2 = symbols("n R T V V_1 V_2", positive=True)
expected = n * R * T * log(V_2 / V_1)
assert simplify(integrate(n * R * T / V, (V, V_1, V_2)) - expected) == 0
```

## The critical point of the van der Waals gas

**Level 3** · `statmech.thermo.van-der-waals` · Pathria and Beale (2011), section 12.2

The critical isotherm has an inflection with zero slope: \(\partial p / \partial V = 0\)
and \(\partial^2 p / \partial V^2 = 0\) together give \(V_c = 3b\) and
\(T_c = 8a / 27Rb\).

```latex
p := \frac{R T}{V - b} - \frac{a}{V^2}
solve([\frac{\partial p}{\partial V} = 0, \frac{\partial^2 p}{\partial V^2} = 0], [V, T])
```

```python
R, T, V, a, b = symbols("R T V a b", positive=True)
p = R * T / (V - b) - a / V**2
expected = solve([diff(p, V), diff(p, V, 2)], [V, T], dict=False)
assert expected == [(3 * b, 8 * a / (27 * R * b))]
assert simplify(p.subs({V: 3 * b, T: 8 * a / (27 * R * b)}) - a / (27 * b**2)) == 0  # p_c
```

## A Maxwell relation

**Level 4** · `statmech.thermo.maxwell-relation` · Callen (1985), section 7.1

\((\partial S / \partial V)_T = (\partial p / \partial T)_V\) for an ideal gas, in the
notation of thermodynamics: each derivative names the variable held constant.

```latex
S := C_V \ln T + n R \ln V
p := \frac{n R T}{V}
\left(\frac{\partial S}{\partial V}\right)_T = \left(\frac{\partial p}{\partial T}\right)_V
```

```python
C_V, n, R, T, V = symbols("C_V n R T V", positive=True)
S = C_V * log(T) + n * R * log(V)
p = n * R * T / V
assert simplify(diff(S, V) - diff(p, T)) == 0
expected = True
```
