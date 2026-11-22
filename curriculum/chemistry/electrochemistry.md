# Electrochemistry

Cells and their potentials: the Nernst equation from the Gibbs energy of the cell
reaction, a concentration cell as general chemistry writes it at 25 °C, and the Daniell
cell with the concentrations of its ions. Standard states are marked ° or ⦵. Constants
are CODATA 2018 values, with R and F rounded to ten figures.

## A concentration cell at 25 °C

**Level 1** · `chem.electrochemistry.concentration-cell` · standard

A copper concentration cell, Cu²⁺ at 0.010 M in one half-cell and 1.0 M in the other.
General chemistry writes the Nernst equation at 25 °C with the common logarithm,
\(E = E^\circ - (0.0592/n)\log Q\); here \(E^\circ = 0\), \(n = 2\) and \(Q = 0.010/1.0\),
the dilute over the concentrated, so the cell gives \((0.0592/2)\log(1.0/0.010) = 0.0592\)
V. In chemistry log is \(\log_{10}\), and the program says so first, as a book's list of
symbols does: other fields write log for ln (`qft.interactions.running-coupling`), so the
reading must come from the program, and the user's definition wins.

```latex
\log x := \log_{10} x
\frac{0.0592}{2} \log \frac{1.0}{0.010}
```

```python
expected = Rational(592, 10**4) / 2 * log(100, 10)
assert expected == Rational(592, 10**4)  # volts
# 0.0592 V is (RT/F) ln 10 at 298.15 K, rounded
assert abs(8.314462618 * 298.15 / 96485.33212 * np.log(10) - 0.0592) < 1e-4
```

## The Nernst equation

**Level 3** · `chem.electrochemistry.nernst-equation` · standard

From the Gibbs energy of the cell reaction, \(\Delta_\mathrm{r} G = \Delta_\mathrm{r}
G^\ominus + RT \ln Q\), and the potential of the cell, \(E = -\Delta_\mathrm{r} G / nF\),
follows \(E = E^\ominus - (RT/nF) \ln Q\). Names that differ only by the standard-state
mark ⦵ are different quantities.

```latex
\Delta_\mathrm{r} G := \Delta_\mathrm{r} G^\ominus + R T \ln Q
E := -\frac{\Delta_\mathrm{r} G}{n F}
E^\ominus := -\frac{\Delta_\mathrm{r} G^\ominus}{n F}
E = E^\ominus - \frac{R T}{n F} \ln Q
```

```python
G0, R, T, Q, n, F = symbols("Delta_rG0 R T Q n F", positive=True)
E = -(G0 + R * T * log(Q)) / (n * F)
E0 = -G0 / (n * F)
assert simplify(E - (E0 - R * T / (n * F) * log(Q))) == 0
# for n = 2 at 298.15 K, a factor of 100 in Q lowers the potential by 59 mV
values = {R: 8.314462618, T: 298.15, F: 96485.33212, n: 2, Q: 100}
assert abs(float((E0 - E).subs(values)) - 0.0592) < 1e-4
expected = True
```

## The Daniell cell

**Level 4** · `chem.electrochemistry.daniell-cell` · standard

Zn | Zn²⁺ ‖ Cu²⁺ | Cu, with \(E^\circ = 1.10\) V, at \([\mathrm{Zn}^{2+}] = 0.10\) M and
\([\mathrm{Cu}^{2+}] = 0.0010\) M: the Nernst equation with the concentrations of the ions
written as chemists write them, a formula and its charge inside brackets. The cell gives
1.041 V.

```latex
R := 8.314462618
T := 298.15
F := 96485.33212
[\mathrm{Zn}^{2+}] := 0.10
[\mathrm{Cu}^{2+}] := 0.0010
1.10 - \frac{R T}{2 F} \ln \frac{[\mathrm{Zn}^{2+}]}{[\mathrm{Cu}^{2+}]}
```

```python
R, T, F = Rational("8.314462618"), Rational("298.15"), Rational("96485.33212")
Q = Rational(10, 100) / Rational(10, 10**4)  # [Zn2+] / [Cu2+]
expected = Rational(110, 100) - R * T / (2 * F) * log(Q)
assert Q == 100 and abs(float(expected) - 1.04084) < 1e-5  # volts
# Q = 100 lowers the potential by (RT/2F) ln 100 = (RT/F) ln 10, 59 mV
assert abs((1.10 - float(expected)) - 8.314462618 * 298.15 / 96485.33212 * np.log(10)) < 1e-12
```
