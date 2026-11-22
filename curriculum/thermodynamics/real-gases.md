# Real gases

The van der Waals gas beyond the ideal gas: at its critical point, at low density, where
its second virial coefficient fixes the Boyle and inversion temperatures, and in units of
its critical constants, where every gas obeys the same equation. Schroeder (2000) is *An
Introduction to Thermal Physics*; Reif (1965) is *Fundamentals of Statistical and Thermal
Physics*; Pathria and Beale (2011) is *Statistical Mechanics*, 3rd edition.

## The critical compressibility factor

**Level 2** · `thermo.real-gases.critical-compressibility` · Schroeder (2000), section 5.3

At its critical point, \(V_c = 3b\) and \(T_c = 8a/27Rb\), a mole of van der Waals gas
has \(pV/RT = 3/8\), whatever \(a\) and \(b\): the same for every gas, where real fluids
have less, about 0.29 for argon and nitrogen.

```latex
p := \frac{R T}{V - b} - \frac{a}{V^2}
\left. \frac{p V}{R T} \right|_{V = 3b,\ T = \frac{8a}{27 R b}}
```

```python
R, T, V, a, b = symbols("R T V a b", positive=True)
p = R * T / (V - b) - a / V**2
V_c, T_c = 3 * b, 8 * a / (27 * R * b)
assert simplify(diff(p, V).subs({V: V_c, T: T_c})) == 0  # the critical point: an inflection
assert simplify(diff(p, V, 2).subs({V: V_c, T: T_c})) == 0  # with zero slope
expected = simplify((p * V / (R * T)).subs({V: V_c, T: T_c}))
assert expected == Rational(3, 8)
# second route: from the critical pressure, p_c = a/27b^2
assert simplify(a / (27 * b**2) * V_c / (R * T_c)) == Rational(3, 8)
```

## The critical point of the Dieterici gas

**Level 3** · `thermo.real-gases.dieterici-critical-point` · Pathria and Beale (2011), section 12.2

Dieterici's equation of state, \(p = \frac{RT}{V - b} e^{-a/RTV}\), has its critical
point where the isotherm is flat with an inflection: \(V_c = 2b\) and \(T_c = a/4Rb\).
Solved for both at once, the case is also one where SymPy's own check of a solution fails
under some orders of a set of symbols, so the answer must not depend on the run.

```latex
a, b, R > 0
p(V, T) := \frac{R T}{V - b} e^{-\frac{a}{R T V}}
solve\left([\frac{\partial p}{\partial V} = 0, \frac{\partial^2 p}{\partial V^2} = 0], [V, T]\right)
```

```python
a, b, R, V, T = symbols("a b R V T", positive=True)
p = R * T / (V - b) * exp(-a / (R * T * V))
V_c, T_c = 2 * b, a / (4 * R * b)
assert simplify(diff(p, V).subs({V: V_c, T: T_c})) == 0  # the isotherm is flat there
assert simplify(diff(p, V, 2).subs({V: V_c, T: T_c})) == 0  # with an inflection
expected = [(V_c, T_c)]
# second route: p_V = 0 gives R T V^2 = a (V - b), and with p_VV = 0 the volume is 2b
first = solve(Eq(R * T * V**2, a * (V - b)), T)[0]
assert simplify(first.subs(V, V_c) - T_c) == 0
# the critical pressure a / (4 e^2 b^2), and p_c V_c / R T_c = 2 / e^2, about 0.27
p_c = simplify(p.subs({V: V_c, T: T_c}))
assert simplify(p_c - a / (4 * E**2 * b**2)) == 0
assert abs(float(p_c * V_c / (R * T_c)) - 0.2707) < 1e-4
```

## The inversion temperature

**Level 2** · `thermo.real-gases.inversion-temperature` · Reif (1965), chapter 5

At low pressure a gas with second virial coefficient \(B(T)\) has
\(\mu_{JT} C_p = T\,dB/dT - B\), so throttling cools it below the temperature where this
vanishes. For van der Waals's \(B = b - a/RT\) that is \(2a/Rb\), \(27/4\) of the
critical temperature.

```latex
B(T) := b - \frac{a}{R T}
solve\left(T \frac{dB}{dT} - B = 0, T\right)
```

```python
T, V, a, b, R = symbols("T V a b R", positive=True)
B = b - a / (R * T)
expected = solve(Eq(T * diff(B, T) - B, 0), T)
assert expected == [2 * a / (R * b)]
assert expected[0] / (8 * a / (27 * R * b)) == Rational(27, 4)  # in units of T_c
# second route: for the full van der Waals gas, T (dV/dT)_p - V = -(T p_T + V p_V) / p_V,
# which tends to T B' - B as the volume grows
p = R * T / (V - b) - a / V**2
joule_thomson = -(T * diff(p, T) + V * diff(p, V)) / diff(p, V)
assert simplify(limit(joule_thomson, V, oo) - (T * diff(B, T) - B)) == 0
```

## The Boyle temperature

**Level 3** · `thermo.real-gases.boyle-temperature` · standard

The second virial coefficient as a limit, \(B = \lim_{V \to \infty} V(pV/RT - 1)\), the
first correction to the ideal gas at low density. Where it vanishes, at the Boyle
temperature, the gas is most nearly ideal: \(a/Rb\) for van der Waals.

```latex
p := \frac{R T}{V - b} - \frac{a}{V^2}
B := \lim_{V \to \infty} V \left(\frac{p V}{R T} - 1\right)
solve(B = 0, T)
```

```python
R, T, V, a, b, x = symbols("R T V a b x", positive=True)
p = R * T / (V - b) - a / V**2
B = limit(V * (p * V / (R * T) - 1), V, oo)
assert simplify(B - (b - a / (R * T))) == 0
expected = solve(Eq(B, 0), T)
assert expected == [a / (R * b)]
# second route: the coefficient of 1/V in the expansion of pV/RT in powers of x = 1/V
Z = (p * V / (R * T)).subs(V, 1 / x)
assert simplify(series(Z, x, 0, 2).removeO() - 1 - x * (b - a / (R * T))) == 0
```

## The law of corresponding states

**Level 3** · `thermo.real-gases.corresponding-states` · Pathria and Beale (2011), section 12.2

Measured in units of its critical constants, \(p_c = a/27b^2\), \(V_c = 3b\) and
\(T_c = 8a/27Rb\), the van der Waals equation loses \(a\) and \(b\):
\(p_r = 8T_r/(3V_r - 1) - 3/V_r^2\), the same equation for every gas.

```latex
p(V, T) := \frac{R T}{V - b} - \frac{a}{V^2}
p_c := \frac{a}{27 b^2}
V_c := 3 b
T_c := \frac{8 a}{27 R b}
\frac{p(V_r V_c, T_r T_c)}{p_c} = \frac{8 T_r}{3 V_r - 1} - \frac{3}{V_r^2}
```

```python
R, V, T, a, b, V_r, T_r = symbols("R V T a b V_r T_r", positive=True)
p = R * T / (V - b) - a / V**2
p_c, V_c, T_c = a / (27 * b**2), 3 * b, 8 * a / (27 * R * b)
reduced = p.subs({V: V_r * V_c, T: T_r * T_c}, simultaneous=True) / p_c
assert simplify(reduced - (8 * T_r / (3 * V_r - 1) - 3 / V_r**2)) == 0
# second route: two gases with different constants, nitrogen's and carbon dioxide's in SI
# units, give the same reduced pressure at the same reduced volume and temperature
gases = [(0.137, 3.87e-5), (0.364, 4.27e-5)]
for v_r, t_r in [(1.7, 1.1), (0.8, 0.95), (5.0, 2.0)]:
    found = [float(reduced.subs({R: 8.314, a: a_, b: b_, V_r: v_r, T_r: t_r})) for a_, b_ in gases]
    assert abs(found[0] - found[1]) < 1e-12 * abs(found[0])
    assert abs(found[0] - (8 * t_r / (3 * v_r - 1) - 3 / v_r**2)) < 1e-12 * abs(found[0])
expected = True
```
