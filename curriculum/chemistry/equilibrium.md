# Chemical equilibrium

The equilibrium constant and the standard Gibbs energy of reaction, the pH of a weak acid
and of a buffer, and how an equilibrium shifts with temperature, in the notation of
chemistry: a reaction's quantities subscripted r, the standard state marked °, and the p
of pH. McQuarrie and Simon (1997) is *Physical Chemistry: A Molecular Approach*.
Constants are CODATA 2018 values, with R rounded to ten figures.

## The standard Gibbs energy of the autoprotolysis of water

**Level 2** · `chem.equilibrium.water-gibbs` · McQuarrie and Simon (1997), chapter 26

\(\Delta_\mathrm{r} G^\circ = -RT \ln K\), with \(K_\mathrm{w} = 1.0 \times 10^{-14}\) at
25 °C: about +79.9 kJ/mol. \(\Delta_\mathrm{r} G^\circ\) is one name, the Gibbs energy of
the reaction (r) in the standard state (°).

```latex
R := 8.314462618
T := 298.15
K_\mathrm{w} := 1.0 \times 10^{-14}
\Delta_\mathrm{r} G^\circ := -R T \ln K_\mathrm{w}
\Delta_\mathrm{r} G^\circ
```

```python
R, T, K_w = Rational("8.314462618"), Rational("298.15"), Rational(1, 10**14)
expected = -R * T * log(K_w)
assert abs(float(expected) - 79912.1) < 0.1  # J/mol
# from the standard Gibbs energies of formation of OH-(aq) and H2O(l), -157.24 and -237.13 kJ/mol
assert abs(float(expected) / 1000 - (-157.24 + 237.13)) < 0.05
```

## The pH of a buffer

**Level 2** · `chem.equilibrium.henderson-hasselbalch` · standard

A buffer of 0.10 M acetic acid and 0.20 M acetate, by the Henderson–Hasselbalch equation,
\(\mathrm{pH} = \mathrm{p}K_\mathrm{a} + \log_{10}([\mathrm{A^-}] / [\mathrm{HA}])\), with
the concentrations put in. The p of chemists is an operator, \(\mathrm{p}X = -\log_{10}
X\), here applied to the \(K_\mathrm{a}\) defined first: pH 5.05.

```latex
K_\mathrm{a} := 1.8 \times 10^{-5}
\mathrm{p}K_\mathrm{a} + \log_{10} \frac{0.20}{0.10}
```

```python
K_a = Rational(18, 10**6)
expected = -log(K_a, 10) + log(2, 10)
assert abs(float(expected) - 5.0458) < 1e-4
# the equilibrium itself, K_a = x (0.20 + x) / (0.10 - x) with x = [H3O+], agrees to 1e-4
x = symbols("x", positive=True)
(root,) = solve(Eq(K_a * (Rational(1, 10) - x), x * (Rational(2, 10) + x)), x)
assert abs(float(-log(root, 10) - expected)) < 1e-4
```

## The pH of a weak acid

**Level 3** · `chem.equilibrium.weak-acid` · standard

0.10 M acetic acid, \(K_\mathrm{a} = 1.8 \times 10^{-5}\). With \(x = [\mathrm{H_3O^+}]\)
from an ICE table, the equilibrium condition is a quadratic in \(x\); its positive root
gives pH 2.88, neglecting the autoprotolysis of water. The equation constrains \(x\) for
the line that follows.

```latex
x > 0
\frac{x^2}{0.10 - x} = 1.8 \times 10^{-5}
-\log_{10} x
```

```python
x = symbols("x", positive=True)
K_a, c = Rational(18, 10**6), Rational(1, 10)
(root,) = solve(Eq(x**2 / (c - x), K_a), x)  # the other root is negative
assert simplify(root - (-K_a + sqrt(K_a**2 + 4 * K_a * c)) / 2) == 0
expected = -log(root, 10)
assert abs(float(expected) - 2.8753) < 1e-4
assert abs(float(expected + log(sqrt(K_a * c), 10))) < 0.005  # x << c, so x = sqrt(K_a c)
```

## The van 't Hoff equation

**Level 3** · `chem.equilibrium.van-t-hoff` · McQuarrie and Simon (1997), chapter 26

How an equilibrium constant changes with temperature: from \(K = e^{-\Delta_\mathrm{r}
G^\circ / RT}\) and \(\Delta_\mathrm{r} G^\circ = \Delta_\mathrm{r} H^\circ - T
\Delta_\mathrm{r} S^\circ\), with \(\Delta_\mathrm{r} H^\circ\) and \(\Delta_\mathrm{r}
S^\circ\) independent of temperature, \(d \ln K / dT = \Delta_\mathrm{r} H^\circ / RT^2\).

```latex
\Delta_\mathrm{r} G^\circ := \Delta_\mathrm{r} H^\circ - T \Delta_\mathrm{r} S^\circ
K := \exp\left(-\frac{\Delta_\mathrm{r} G^\circ}{R T}\right)
\frac{d \ln K}{d T} = \frac{\Delta_\mathrm{r} H^\circ}{R T^2}
```

```python
T, R = symbols("T R", positive=True)
dH, dS = symbols("Delta_rH Delta_rS", real=True)
K = exp(-(dH - T * dS) / (R * T))
assert simplify(diff(log(K), T) - dH / (R * T**2)) == 0
# integrated between 400 K and 401 K: ln(K2/K1) = -(dH/R)(1/T2 - 1/T1), exactly
step = log(K).subs(T, 401) - log(K).subs(T, 400)
assert simplify(step + dH / R * (Rational(1, 401) - Rational(1, 400))) == 0
expected = True
```
