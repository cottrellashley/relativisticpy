# Spectroscopy

Molecular spectra in the chemist's units, wavenumbers in cm⁻¹: the vibration of HCl,
harmonic and anharmonic, its most populated rotational level, the absorption of a
conjugated polyene in the free-electron model, and the small population difference that
NMR detects. McQuarrie and Simon (1997) is *Physical Chemistry: A Molecular Approach*;
Herzberg (1950) is *Spectra of Diatomic Molecules*, 2nd edition, and Huber and Herzberg
(1979) is *Constants of Diatomic Molecules*. Constants are CODATA 2018 values.

## The vibrational wavenumber of HCl

**Level 2** · `chem.spectroscopy.hcl-wavenumber` · McQuarrie and Simon (1997), chapter 5; Huber and Herzberg (1979)

The harmonic wavenumber of ¹H³⁵Cl from its force constant, 516.3 N/m, and the reduced
mass of its atoms: \(\tilde\nu = (1/2\pi c)\sqrt{k/\mu}\), with \(c\) in cm/s, about
2991 cm⁻¹. The wavenumber is a name with an accent, \(\tilde\nu\), defined with \(:=\).

```latex
u := 1.66053906660 \times 10^{-27}
m_\mathrm{H} := 1.00782503 u
m_\mathrm{Cl} := 34.96885268 u
\mu := \frac{m_\mathrm{H} m_\mathrm{Cl}}{m_\mathrm{H} + m_\mathrm{Cl}}
k := 516.3
c := 2.99792458 \times 10^{10}
\tilde{\nu} := \frac{1}{2\pi c} \sqrt{\frac{k}{\mu}}
\tilde{\nu}
```

```python
u = Rational("1.66053906660e-27")  # kg, the atomic mass constant
m_H, m_Cl = Rational("1.00782503") * u, Rational("34.96885268") * u
mu = m_H * m_Cl / (m_H + m_Cl)
k, c = Rational("516.3"), 299792458 * 100  # N/m, cm/s
expected = sqrt(k / mu) / (2 * pi * c)
assert abs(float(mu / u) - 0.979593) < 1e-6
assert abs(float(expected) - 2990.946) < 0.1  # omega_e of H-35Cl, Huber and Herzberg
```

## The populations of nuclear spin states

**Level 2** · `chem.spectroscopy.nmr-populations` · standard

Protons in a 400 MHz spectrometer at 25 °C: the upper spin state holds the Boltzmann
fraction \(N_\beta = N_\alpha e^{-h\nu/k_\mathrm{B}T}\) of the lower, and the excess in
the lower, \((N_\alpha - N_\beta)/(N_\alpha + N_\beta) = \tanh(h\nu/2k_\mathrm{B}T)\), is
only \(3.2 \times 10^{-5}\): why NMR is insensitive.

```latex
h := 6.62607015 \times 10^{-34}
k_\mathrm{B} := 1.380649 \times 10^{-23}
\nu := 400 \times 10^{6}
T := 298.15
N_\beta := N_\alpha \exp\left(-\frac{h \nu}{k_\mathrm{B} T}\right)
\frac{N_\alpha - N_\beta}{N_\alpha + N_\beta}
```

```python
h, k_B = Rational("6.62607015e-34"), Rational("1.380649e-23")  # CODATA 2018, exact
x = h * 400 * 10**6 / (k_B * Rational("298.15"))
expected = (1 - exp(-x)) / (1 + exp(-x))
assert abs(float(expected - tanh(x / 2))) < 1e-18
assert abs(float(expected) / 3.2193e-5 - 1) < 1e-4
assert abs(float(expected) / float(x / 2) - 1) < 1e-8  # far above the splitting, h nu / 2 k T
```

## The absorption of butadiene in the free-electron model

**Level 3** · `chem.spectroscopy.free-electron` · McQuarrie and Simon (1997), chapter 3

The four π electrons of butadiene move freely in a box of length 578 pm, with levels
\(E_n = n^2 h^2 / 8 m_\mathrm{e} L^2\), two electrons to a level. The first absorption
lifts one from \(n = 2\) to \(n = 3\), at \(\lambda = hc/(E_3 - E_2)\): about 220 nm,
where 217 nm is observed. The levels are a family, \(E_n\), evaluated at 2 and 3.

```latex
h := 6.62607015 \times 10^{-34}
m_\mathrm{e} := 9.1093837015 \times 10^{-31}
c := 2.99792458 \times 10^{8}
L := 578 \times 10^{-12}
E_n := \frac{n^2 h^2}{8 m_\mathrm{e} L^2}
\lambda := \frac{h c}{E_3 - E_2}
\lambda
```

```python
h, m_e, c = Rational("6.62607015e-34"), Rational("9.1093837015e-31"), 299792458
L = Rational(578, 10**12)
E = lambda n: n**2 * h**2 / (8 * m_e * L**2)
expected = h * c / (E(3) - E(2))
N = 4  # pi electrons: the transition from level N/2 to N/2 + 1 is at 8 m c L^2 / (h (N + 1))
assert simplify(expected - 8 * m_e * c * L**2 / (h * (N + 1))) == 0
assert abs(float(expected) - 220.3e-9) < 0.1e-9  # metres
assert abs(float(expected) / 217e-9 - 1) < 0.02  # the observed band
```

## The most populated rotational level

**Level 3** · `chem.spectroscopy.rotational-maximum` · standard

The population of rotational level \(J\), \((2J + 1) e^{-hcBJ(J+1)/k_\mathrm{B}T}\), with
\(J\) treated as continuous, is greatest where its derivative vanishes, at
\(J_{\max} = \sqrt{k_\mathrm{B}T/2hcB} - \tfrac{1}{2}\): of the two roots, the assumption
\(J > 0\) keeps one. That root is positive because the program also takes
\(k_\mathrm{B}T > hcB\), the regime in which \(J\) may be treated as continuous. For HCl at
25 °C, with \(B = 10.59\) cm⁻¹, \(J_{\max} \approx 2.6\), and level 3 is the most populated.

```latex
h, c, k_\mathrm{B}, B, T > 0
k_\mathrm{B} T > h c B
J > 0
P(J) := (2J + 1) \exp\left(-\frac{h c B J(J + 1)}{k_\mathrm{B} T}\right)
solve\left(\frac{dP}{dJ} = 0, J\right)
```

```python
h, c, k_B, B, T, J = symbols("h c k_B B T J", positive=True)
P = (2 * J + 1) * exp(-h * c * B * J * (J + 1) / (k_B * T))
found = solve(Eq(diff(P, J), 0), J)
expected = [sqrt(k_B * T / (2 * h * c * B)) - Rational(1, 2)]
assert len(found) == 1 and simplify(found[0] - expected[0]) == 0
# HCl, B = 10.59 cm^-1, at 298.15 K: J_max = 2.63, and level 3 is the most populated
kT_hc = 1.380649e-23 * 298.15 / (6.62607015e-34 * 2.99792458e10)  # cm^-1
levels = np.arange(30)
populations = (2 * levels + 1) * np.exp(-10.59 * levels * (levels + 1) / kT_hc)
assert levels[np.argmax(populations)] == 3
assert abs(np.sqrt(kT_hc / (2 * 10.59)) - 0.5 - 2.628) < 1e-3
```

## The fundamental of an anharmonic oscillator

**Level 4** · `chem.spectroscopy.anharmonic` · Herzberg (1950), chapter III; Huber and Herzberg (1979)

Spectroscopists write the vibrational terms of a diatomic molecule as \(G(v) =
\omega_e(v + \tfrac{1}{2}) - \omega_e x_e (v + \tfrac{1}{2})^2\), where \(\omega_e x_e\) is
one constant, not a product. For ¹H³⁵Cl, \(\omega_e = 2990.9\) cm⁻¹ and \(\omega_e x_e =
52.8\) cm⁻¹, so the fundamental, \(G(1) - G(0) = \omega_e - 2\omega_e x_e\), is at
2885.3 cm⁻¹, where 2886 cm⁻¹ is observed.

```latex
\omega_e := 2990.9
\omega_e x_e := 52.8
G(v) := \omega_e \left(v + \tfrac{1}{2}\right) - \omega_e x_e \left(v + \tfrac{1}{2}\right)^2
G(1) - G(0)
```

```python
v = symbols("v")
omega_e, omega_e_x_e = Rational("2990.9"), Rational("52.8")  # cm^-1, rounded
G = omega_e * (v + Rational(1, 2)) - omega_e_x_e * (v + Rational(1, 2)) ** 2
expected = G.subs(v, 1) - G.subs(v, 0)
assert expected == omega_e - 2 * omega_e_x_e == Rational("2885.3")
assert abs(float(expected) - 2886) < 1  # the observed fundamental of HCl
```
