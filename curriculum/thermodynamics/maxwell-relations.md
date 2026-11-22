# Maxwell relations and derivatives

Relations between the derivatives of thermodynamic quantities: the difference of the heat
capacities and the cooling of a van der Waals gas in a free expansion, then, in the
notation of thermodynamics, where a partial derivative names the variable held fixed, a
Maxwell relation, the cyclic relation and a Jacobian, each in variables other than those
the quantities are written in, so that the language must change variables. Callen (1985) is *Thermodynamics
and an Introduction to Thermostatistics*, 2nd edition; Reif (1965) is *Fundamentals of
Statistical and Thermal Physics*.

## The difference of the heat capacities

**Level 3** · `thermo.maxwell.heat-capacities` · Callen (1985), chapter 7

\(C_p - C_V = TV\alpha^2/\kappa_T\), with the compressibility \(\kappa_T\) and the
expansivity \(\alpha = \kappa_T\, \partial p/\partial T\) computed from the pressure: for
a mole of van der Waals gas, \(R / (1 - 2a(V - b)^2/RTV^3)\), which is \(R\) when
\(a = 0\).

```latex
p := \frac{R T}{V - b} - \frac{a}{V^2}
\kappa_T := -\frac{1}{V \frac{\partial p}{\partial V}}
\alpha := \kappa_T \frac{\partial p}{\partial T}
\frac{T V \alpha^2}{\kappa_T}
```

```python
from scipy.optimize import brentq

R, T, V, a, b = symbols("R T V a b", positive=True)
p = R * T / (V - b) - a / V**2
kappa_T = -1 / (V * diff(p, V))
alpha = kappa_T * diff(p, T)
expected = R / (1 - 2 * a * (V - b) ** 2 / (R * T * V**3))
assert simplify(T * V * alpha**2 / kappa_T - expected) == 0
assert expected.subs(a, 0) == R  # the ideal gas: C_p - C_V = R
# second route: alpha and kappa_T by finite differences of V(T, p), the root of the cubic,
# for nitrogen's constants in SI units, at 300 K and 1 MPa
R_, a_, b_ = 8.314, 0.137, 3.87e-5


def volume(t, q):
    return brentq(lambda v: R_ * t / (v - b_) - a_ / v**2 - q, 1e-4, 1e-1, xtol=1e-17)


v0 = volume(300, 1e6)
alpha_ = (volume(300 + 1e-3, 1e6) - volume(300 - 1e-3, 1e6)) / (2e-3 * v0)
kappa_ = -(volume(300, 1e6 + 1) - volume(300, 1e6 - 1)) / (2 * v0)
values = {R: R_, a: a_, b: b_, T: 300, V: v0}
assert abs(300 * v0 * alpha_**2 / kappa_ / float(expected.subs(values)) - 1) < 1e-6
```

## Cooling in a free expansion

**Level 3** · `thermo.maxwell.joule-expansion` · Reif (1965), chapter 5

In a free expansion the energy is constant, so the temperature changes at the rate
\((\partial T/\partial V)_U = -(T\, \partial p/\partial T - p)/C_V\). For a mole of van der
Waals gas with a constant \(C_V\), \(T\, \partial p/\partial T - p = a/V^2\) does not depend
on \(T\), so the change is an integral over the volume alone: expanding from \(V_1\) to
\(V_2\), its temperature changes by \(-(a/C_V)(1/V_1 - 1/V_2)\), a fall.

```latex
C_V > 0
V_2 > V_1 > b > 0
p := \frac{R T}{V - b} - \frac{a}{V^2}
-\frac{1}{C_V} \int_{V_1}^{V_2} \left(T \frac{\partial p}{\partial T} - p\right) dV
```

```python
R, T, V, V_1, V_2, a, b, C_V = symbols("R T V V_1 V_2 a b C_V", positive=True)
p = R * T / (V - b) - a / V**2
expected = -a / C_V * (1 / V_1 - 1 / V_2)
assert simplify(-integrate(T * diff(p, T) - p, (V, V_1, V_2)) / C_V - expected) == 0
# second route: the energy of the van der Waals gas, U = C_V T - a/V, does not change
T_1, T_2 = symbols("T_1 T_2", positive=True)
after = solve(Eq(C_V * T_2 - a / V_2, C_V * T_1 - a / V_1), T_2)[0]
assert simplify(after - T_1 - expected) == 0
```

## A Maxwell relation in other variables

**Level 4** · `thermo.maxwell.gibbs-relation` · Callen (1985), sections 3.5 and 7.1

A mole of van der Waals gas with a constant \(C_V\), its pressure and its entropy (up to a
constant) given as functions of \(T\) and \(V\), as Callen's van der Waals fluid gives them.
The Maxwell relation of the Gibbs free energy, \((\partial S/\partial p)_T =
-(\partial V/\partial T)_p\), holds \(T\) or \(p\) fixed, so each side asks for a change of
variables from \((T, V)\) to \((T, p)\), in which \(V\) is a root of a cubic that no one
writes out: \((\partial S/\partial p)_T = (\partial S/\partial V)_T / (\partial p/\partial
V)_T\) and \((\partial V/\partial T)_p = -(\partial p/\partial T)_V / (\partial p/\partial
V)_T\).

```latex
T, R, a, b, C_V > 0
V > b
p := \frac{R T}{V - b} - \frac{a}{V^2}
S := C_V \ln T + R \ln(V - b)
\left(\frac{\partial S}{\partial p}\right)_T = -\left(\frac{\partial V}{\partial T}\right)_p
```

```python
from scipy.optimize import brentq

T, V, R, a, b, C_V = symbols("T V R a b C_V", positive=True)
p = R * T / (V - b) - a / V**2
S = C_V * log(T) + R * log(V - b)
left = diff(S, V) / diff(p, V)  # (dS/dp)_T: at fixed T, S and p change through V alone
right = diff(p, T) / diff(p, V)  # -(dV/dT)_p, by implicit differentiation of p(T, V)
assert simplify(left - right) == 0
assert simplify(diff(S, V) - diff(p, T)) == 0  # because (dS/dV)_T = (dp/dT)_V, from F
# second route: finite differences in T and p, with V(T, p) a root of the cubic, for a mole
# of nitrogen in SI units at 300 K and 1 MPa, above its critical temperature
R_, a_, b_, C_ = 8.314, 0.137, 3.87e-5, 20.8


def volume(t, q):
    return brentq(lambda v: R_ * t / (v - b_) - a_ / v**2 - q, 1e-4, 1e-1, xtol=1e-18)


def entropy(t, q):
    return C_ * np.log(t) + R_ * np.log(volume(t, q) - b_)


t0, q0, h = 300.0, 1e6, 1e-6
dS_dp = (entropy(t0, q0 * (1 + h)) - entropy(t0, q0 * (1 - h))) / (2 * h * q0)
dV_dT = (volume(t0 * (1 + h), q0) - volume(t0 * (1 - h), q0)) / (2 * h * t0)
assert abs(dS_dp / -dV_dT - 1) < 1e-6
expected = True
```

## The cyclic relation

**Level 4** · `thermo.maxwell.triple-product` · standard

Three variables tied by an equation of state, here van der Waals's, which no one solves
for \(V\): the derivatives of each with respect to the next, the third held fixed,
multiply to \(-1\).

```latex
\left(p + \frac{a}{V^2}\right)(V - b) = R T
\left(\frac{\partial p}{\partial V}\right)_T \left(\frac{\partial V}{\partial T}\right)_p \left(\frac{\partial T}{\partial p}\right)_V = -1
```

```python
from scipy.optimize import brentq

p, V, T, a, b, R = symbols("p V T a b R", positive=True)
f = (p + a / V**2) * (V - b) - R * T  # the equation of state, f = 0
dp_dV = -diff(f, V) / diff(f, p)  # implicit differentiation, T fixed
dV_dT = -diff(f, T) / diff(f, V)  # p fixed
dT_dp = -diff(f, p) / diff(f, T)  # V fixed
assert simplify(dp_dV * dV_dT * dT_dp) == -1
# second route: each derivative by finite differences, for nitrogen's constants in SI units,
# at 300 K and 2 litres per mole, with V(T, p) found as a root of the cubic
R_, a_, b_ = 8.314, 0.137, 3.87e-5


def pressure(v, t):
    return R_ * t / (v - b_) - a_ / v**2


def volume(t, q):
    return brentq(lambda v: pressure(v, t) - q, 5e-5, 1.0, xtol=1e-17)


def temperature(q, v):
    return (q + a_ / v**2) * (v - b_) / R_


t0, v0, h = 300.0, 2e-3, 1e-6
q0 = pressure(v0, t0)
first = (pressure(v0 * (1 + h), t0) - pressure(v0 * (1 - h), t0)) / (2 * v0 * h)
second = (volume(t0 * (1 + h), q0) - volume(t0 * (1 - h), q0)) / (2 * t0 * h)
third = (temperature(q0 * (1 + h), v0) - temperature(q0 * (1 - h), v0)) / (2 * q0 * h)
assert abs(first * second * third + 1) < 1e-6
expected = True
```

## The Jacobian of temperature and entropy

**Level 4** · `thermo.maxwell.jacobian` · standard

Every Maxwell relation is one statement in the notation of Jacobians,
\(\partial(T, S)/\partial(p, V) = 1\): a cycle encloses the same area in the \(T\)–\(S\)
plane as in the \(p\)–\(V\) plane. Here for \(n\) moles of ideal gas, with the pressure
and the entropy (up to a constant) given, as books give them, as functions of \(T\) and
\(V\): the Jacobian is taken in variables they are not written in, which the chain rule
handles, \(\partial(T, S)/\partial(p, V) = [\partial(T, S)/\partial(T, V)] /
[\partial(p, V)/\partial(T, V)]\).

```latex
n, R, C_V, T, V > 0
p := \frac{n R T}{V}
S := n C_V \ln T + n R \ln V
\frac{\partial(T, S)}{\partial(p, V)} = 1
```

```python
P, n, R, C_V, T, V = symbols("P n R C_V T V", positive=True)
p = n * R * T / V
S = n * C_V * log(T) + n * R * log(V)


def jacobian(f, g, x, y):  # d(f, g)/d(x, y)
    return Matrix([[diff(f, x), diff(f, y)], [diff(g, x), diff(g, y)]]).det()


# through (T, V): d(T, S)/d(p, V) = [d(T, S)/d(T, V)] / [d(p, V)/d(T, V)] = (dS/dV)_T / (dp/dT)_V
assert simplify(jacobian(T, S, T, V) / jacobian(p, V, T, V)) == 1
# second route: S written in the pressure P and V, with T = PV/nR, and the determinant direct
T_of_P_V = P * V / (n * R)
assert simplify(jacobian(T_of_P_V, S.subs(T, T_of_P_V), P, V)) == 1
expected = True
```
