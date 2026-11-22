# Thermodynamic potentials

The energy and its Legendre transforms: the energy recovered from the Helmholtz free
energy, Euler's relation for an extensive fundamental relation, the free energy from the
energy by eliminating the entropy, and the thermodynamic identity that defines temperature
and pressure. Callen (1985) is *Thermodynamics and an Introduction to Thermostatistics*, 2nd
edition; Schroeder (2000) is *An Introduction to Thermal Physics*.

## The Gibbs–Helmholtz equation

**Level 2** · `thermo.potentials.gibbs-helmholtz` · standard

The energy from the Helmholtz free energy, \(U = -T^2\, \partial(F/T)/\partial T\). For a
monatomic ideal gas, whose \(F = -nRT \ln(cVT^{3/2})\) with \(c\) a constant, it is the
equipartition value \(\frac{3}{2} nRT\).

```latex
F := -n R T \ln\left(c V T^{3/2}\right)
-T^2 \frac{\partial}{\partial T}\left(\frac{F}{T}\right)
```

```python
n, R, T, V, c = symbols("n R T V c", positive=True)
F = -n * R * T * log(c * V * T ** Rational(3, 2))
expected = simplify(-(T**2) * diff(F / T, T))
assert expected == Rational(3, 2) * n * R * T
# second route: U = F + T S, with S = -dF/dT
assert simplify(F - T * diff(F, T) - expected) == 0
```

## Euler's relation

**Level 3** · `thermo.potentials.euler-relation` · Callen (1985), sections 3.1 and 3.4

The entropy of a simple ideal gas as Callen writes it, extensive in \(U\), \(V\) and
\(N\). The intensive parameters are its derivatives, \(1/T = \partial S/\partial U\),
\(p/T = \partial S/\partial V\) and \(\mu/T = -\partial S/\partial N\), and Euler's
relation \(S = U/T + pV/T - \mu N/T\) holds because \(S\) is homogeneous of the first
degree.

```latex
U, V, N, U_0, V_0, N_0 > 0
S := N s_0 + N R \ln\left[\left(\frac{U}{U_0}\right)^{c} \frac{V}{V_0} \left(\frac{N}{N_0}\right)^{-(c+1)}\right]
T := \left(\frac{\partial S}{\partial U}\right)^{-1}
p := T \frac{\partial S}{\partial V}
\mu := -T \frac{\partial S}{\partial N}
S = \frac{U}{T} + \frac{p V}{T} - \frac{\mu N}{T}
```

```python
U, V, N, U_0, V_0, N_0, R, c, s_0 = symbols("U V N U_0 V_0 N_0 R c s_0", positive=True)
S = N * s_0 + N * R * log((U / U_0) ** c * (V / V_0) * (N / N_0) ** (-(c + 1)))
T = 1 / diff(S, U)
p = T * diff(S, V)
mu = -T * diff(S, N)
assert simplify(S - (U / T + p * V / T - mu * N / T)) == 0
assert simplify(T - U / (c * N * R)) == 0 and simplify(p * V - N * R * T) == 0  # the gas laws
# second route: S is homogeneous of the first degree, which Euler's relation expresses
lam = symbols("lambda", positive=True)
scaled = S.subs({U: lam * U, V: lam * V, N: lam * N}, simultaneous=True)
assert simplify(expand_log(scaled - lam * S, force=True)) == 0
expected = True
```

## The free energy by a Legendre transform

**Level 4** · `thermo.potentials.legendre-transform` · Callen (1985), chapter 5

The Legendre transform as Callen performs it: from the energy of a monatomic ideal gas as a
function of its entropy and volume, \(U(S, V) = C V^{-2/3} e^{2S/3nR}\), the Helmholtz free
energy \(F = U - TS\), with \(S\) eliminated through \(T = \partial U/\partial S\), so that
\(F\) is a function of \(T\) and \(V\):
\(F = \frac{3}{2} nRT \left[1 - \ln(3nRT V^{2/3}/2C)\right]\), whose volume derivative
gives back \(p = nRT/V\). The equation \(T = \partial U/\partial S\) defines nothing: it ties
\(T\) to \(S\), and the arguments written on the left of \(F(T, V)\) say which variables
remain.

```latex
n, R, C, T, V > 0
U(S, V) := C V^{-2/3} e^{2 S / (3 n R)}
T = \frac{\partial U}{\partial S}
F(T, V) := U - T S
F(T, V)
```

```python
from scipy.optimize import brentq

n, R, C, T, V, S = symbols("n R C T V S", positive=True)
U = C * V ** Rational(-2, 3) * exp(2 * S / (3 * n * R))
(entropy,) = solve(Eq(T, diff(U, S)), S)  # S(T, V), from T = dU/dS
expected = Rational(3, 2) * n * R * T * (1 - log(3 * n * R * T * V ** Rational(2, 3) / (2 * C)))
assert simplify(expand_log((U - T * S).subs(S, entropy) - expected, force=True)) == 0
# the derivatives of F give back the variables eliminated: S = -dF/dT, and p = nRT/V
assert simplify(expand_log(-diff(expected, T) - entropy, force=True)) == 0
assert simplify(-diff(expected, V) - n * R * T / V) == 0
# second route: S found numerically from T = dU/dS, for n = 1, C = 2000, V = 0.02, T = 300
values = {n: 1, R: 8.314, C: 2000, V: Rational(1, 50)}
dU_dS = lambdify(S, diff(U, S).subs(values))
s_300 = brentq(lambda s: dU_dS(s) - 300, -200, 400)
u_300 = float(U.subs(values).subs(S, s_300))
assert abs(u_300 - 300 * s_300 - float(expected.subs(values).subs(T, 300))) < 1e-6 * u_300
```

## The thermodynamic identity

**Level 4** · `thermo.potentials.thermodynamic-identity` · Schroeder (2000), section 3.4

\(dU = T\,dS - p\,dV\), read as books read it: with the energy given as a function of
\(S\) and \(V\), the identity defines the temperature and the pressure as the
coefficients of \(dS\) and \(-dV\). For a monatomic ideal gas they satisfy
\(pV/T = nR\).

```latex
U := C V^{-2/3} e^{2 S / (3 n R)}
dU = T \, dS - p \, dV
\frac{p V}{T}
```

```python
n, R, C, S, V = symbols("n R C S V", positive=True)
U = C * V ** Rational(-2, 3) * exp(2 * S / (3 * n * R))
T = diff(U, S)  # the coefficient of dS
p = -diff(U, V)  # minus the coefficient of dV
expected = simplify(p * V / T)
assert expected == n * R
# second route: the energy is the equipartition value, U = 3nRT/2, and pV = 2U/3
assert simplify(U - Rational(3, 2) * n * R * T) == 0
assert simplify(p * V - Rational(2, 3) * U) == 0
```
