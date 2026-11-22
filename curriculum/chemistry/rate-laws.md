# Rate laws

How fast reactions go: first- and second-order decay and their half-lives, the Arrhenius
equation and the Eyring equation, in the notations chemists use for concentrations, a
bracket \([\mathrm{A}]\) in physical chemistry and \(C_\mathrm{A}\) in chemical
engineering. McQuarrie and Simon (1997) is *Physical Chemistry: A Molecular Approach*;
Levenspiel (1999) is *Chemical Reaction Engineering*, 3rd edition. Constants are CODATA
2018 values, with R rounded to ten figures.

## The age of a sample from its radiocarbon

**Level 1** · `chem.rates.radiocarbon-age` · Godwin (1962), Nature 195, 984

First-order decay: a sample whose carbon-14 activity has fallen to a quarter of that of
living matter is two half-lives old, \(t = (t_{1/2} / \ln 2) \ln(A_0 / A)\), 11 460 years
with Godwin's half-life of 5730 years. The half-life is a name with a fraction in its
subscript, \(t_{1/2}\).

```latex
t_{1/2} := 5730
\frac{t_{1/2}}{\ln 2} \ln 4
```

```python
half_life = 5730  # years, the Cambridge half-life of carbon-14
expected = 2 * half_life  # a quarter is left after two half-lives
assert simplify(half_life / log(2) * log(4) - expected) == 0
k = log(2) / half_life  # the first-order rate constant: exp(-k t) is a quarter at t = 11 460
assert abs(float(exp(-k * expected)) - 0.25) < 1e-15
```

## The exponential factor of the Arrhenius equation

**Level 1** · `chem.rates.arrhenius-factor` · Arrhenius (1889), Z. Phys. Chem. 4, 226

\(e^{-E_\mathrm{a}/RT}\) for an activation energy of 50 kJ/mol at 25 °C, about
\(1.7 \times 10^{-9}\), written with a solidus as chemists write it: \(E_\mathrm{a}/RT\)
is \(E_\mathrm{a}/(RT)\), the product below the line.

```latex
\left. e^{-E_\mathrm{a}/RT} \right|_{E_\mathrm{a} = 50\,000,\ R = 8.314462618,\ T = 298.15}
```

```python
E_a, R, T = 50000, Rational("8.314462618"), Rational("298.15")  # J/mol, J/(K mol), K
expected = exp(-E_a / (R * T))
assert abs(float(expected) - np.exp(-50000 / (8.314462618 * 298.15))) < 1e-22
assert abs(float(expected) / 1.739e-9 - 1) < 1e-3
```

## The activation energy from two temperatures

**Level 2** · `chem.rates.activation-energy` · McQuarrie and Simon (1997), chapter 28

The rule of thumb that a rate doubles for a rise of 10 K near room temperature: the
Arrhenius equation at two temperatures, \(\ln(k_2 / k_1) = (E_\mathrm{a} / R)(1/T_1 -
1/T_2)\), with \(k_2 = 2k_1\) between 298 K and 308 K, solved for the activation energy:
about 53 kJ/mol.

```latex
R := 8.314462618
solve\left(\ln 2 = \frac{E_\mathrm{a}}{R}\left(\frac{1}{298} - \frac{1}{308}\right), E_\mathrm{a}\right)
```

```python
E_a = symbols("E_a")
R = Rational("8.314462618")
expected = solve(Eq(log(2), E_a / R * (Rational(1, 298) - Rational(1, 308))), E_a)
assert len(expected) == 1 and abs(float(expected[0]) - 52896.46) < 0.01  # J/mol
k = lambda T: exp(-expected[0] / (R * T))  # the Arrhenius factor doubles from 298 K to 308 K
assert abs(float(k(308) / k(298)) - 2) < 1e-12
```

## The half-life of a second-order reaction

**Level 2** · `chem.rates.second-order-half-life` · Levenspiel (1999), chapter 3

For \(-dC_\mathrm{A}/dt = kC_\mathrm{A}^2\) the concentration falls as
\(C_{\mathrm{A}0} / (1 + kC_{\mathrm{A}0} t)\), in the notation of chemical engineering,
and halves at \(t = 1/kC_{\mathrm{A}0}\): unlike a first-order half-life, it depends on
the initial concentration.

```latex
k, C_{\mathrm{A}0} > 0
C_\mathrm{A}(t) := \frac{C_{\mathrm{A}0}}{1 + k C_{\mathrm{A}0} t}
solve\left(C_\mathrm{A}(t) = \frac{C_{\mathrm{A}0}}{2}, t\right)
```

```python
t = symbols("t")
k, C0 = symbols("k C_A0", positive=True)
C = C0 / (1 + k * C0 * t)
assert simplify(diff(C, t) + k * C**2) == 0  # it solves the rate law
expected = solve(Eq(C, C0 / 2), t)
assert expected == [1 / (k * C0)]
assert simplify(1 / C.subs(t, expected[0]) - 1 / C0 - k * expected[0]) == 0  # 1/C_A - 1/C_A0 = kt
```

## The second-order rate law in brackets

**Level 4** · `chem.rates.second-order-brackets` · McQuarrie and Simon (1997), chapter 28

Physical chemistry writes the concentration of A as \([\mathrm{A}]\) and its initial value
as \([\mathrm{A}]_0\): brackets that name a quantity, not a list. The integrated law
\([\mathrm{A}] = [\mathrm{A}]_0 / (1 + k[\mathrm{A}]_0 t)\) satisfies the rate law
\(\mathrm{d}[\mathrm{A}]/\mathrm{d}t = -k[\mathrm{A}]^2\).

```latex
[\mathrm{A}] := \frac{[\mathrm{A}]_0}{1 + k [\mathrm{A}]_0 t}
\frac{\mathrm{d}[\mathrm{A}]}{\mathrm{d}t} = -k [\mathrm{A}]^2
```

```python
from scipy.integrate import solve_ivp

t, k, A0 = symbols("t k A_0")
A = A0 / (1 + k * A0 * t)
assert simplify(diff(A, t) + k * A**2) == 0
assert A.subs(t, 0) == A0
# and numerically: the rate law integrated from [A]_0 = 2 with k = 0.7 reaches the formula
solution = solve_ivp(lambda s, y: [-0.7 * y[0] ** 2], (0, 3), [2.0], rtol=1e-12, atol=1e-14)
assert abs(solution.y[0][-1] - 2.0 / (1 + 0.7 * 2.0 * 3)) < 1e-9
expected = True
```

## The Eyring equation

**Level 4** · `chem.rates.eyring` · Eyring (1935), J. Chem. Phys. 3, 107

Transition-state theory marks the activated complex with a double dagger: the Gibbs
energy of activation is \(\Delta^\ddagger G\), one name. For 80 kJ/mol at 25 °C the rate
constant \((k_\mathrm{B} T / h) e^{-\Delta^\ddagger G / RT}\) is about 0.060 s⁻¹, a
half-life of about 12 s.

```latex
k_\mathrm{B} := 1.380649 \times 10^{-23}
h := 6.62607015 \times 10^{-34}
R := 8.314462618
T := 298.15
\Delta^\ddagger G := 80 \times 10^{3}
\frac{k_\mathrm{B} T}{h} \exp\left(-\frac{\Delta^\ddagger G}{R T}\right)
```

```python
k_B, h = Rational("1.380649e-23"), Rational("6.62607015e-34")  # CODATA 2018, exact
R, T, barrier = Rational("8.314462618"), Rational("298.15"), 80000
expected = k_B * T / h * exp(-barrier / (R * T))
assert abs(float(expected) - 0.05996) < 1e-5  # per second
# the same in logarithms, with k_B = R / N_A
N_A = 6.02214076e23
log_k = np.log(8.314462618 / N_A * 298.15 / 6.62607015e-34) - 80e3 / (8.314462618 * 298.15)
assert abs(np.log(float(expected)) - log_k) < 1e-8
```
