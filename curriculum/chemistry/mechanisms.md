# Mechanisms and chemical equations

Reactions built from elementary steps: consecutive reactions and their intermediate, the
steady state of an enzyme and of an activated molecule, and the chemical equation itself,
which chemists balance and from which, step by step, they write rate equations.
Levenspiel (1999) is *Chemical Reaction Engineering*, 3rd edition.

## The intermediate of consecutive reactions

**Level 3** · `chem.mechanisms.consecutive-maximum` · Levenspiel (1999), chapter 3

In A → R → S, with rate constants \(k_1\) and \(k_2\), the intermediate rises and falls,
\(C_\mathrm{R} = C_{\mathrm{A}0} k_1 (e^{-k_1 t} - e^{-k_2 t}) / (k_2 - k_1)\). It is
greatest where its derivative vanishes, at \(t = \ln(k_2/k_1) / (k_2 - k_1)\), a positive
time whichever constant is larger. The time is real and positive, as the program says: over
the complex numbers the derivative also vanishes at that time plus
\(2\pi i n / (k_2 - k_1)\) for every integer \(n\).

```latex
k_1, k_2, C_{\mathrm{A}0}, t > 0, \quad k_1 \neq k_2
C_\mathrm{R}(t) := \frac{k_1 C_{\mathrm{A}0}}{k_2 - k_1} \left(e^{-k_1 t} - e^{-k_2 t}\right)
solve\left(\frac{d C_\mathrm{R}}{d t} = 0, t\right)
```

```python
t, k_1, k_2, C0 = symbols("t k_1 k_2 C_A0", positive=True)
C_R = k_1 * C0 / (k_2 - k_1) * (exp(-k_1 * t) - exp(-k_2 * t))
found = solve(Eq(diff(C_R, t), 0), t)
assert len(found) == 1 and simplify(found[0] - log(k_2 / k_1) / (k_2 - k_1)) == 0
expected = [log(k_2 / k_1) / (k_2 - k_1)]
for a, b in ((1, 2), (2, 1), (0.3, 5)):  # positive, whichever constant is larger
    assert log(b / a) / (b - a) > 0
grid = np.linspace(0, 5, 500001)  # with k_1 = 1 and k_2 = 2 the maximum is at ln 2
assert abs(grid[np.argmax(np.exp(-grid) - np.exp(-2 * grid))] - np.log(2)) < 1e-4
```

## The steady state of an enzyme

**Level 4** · `chem.mechanisms.michaelis-menten` · Briggs and Haldane (1925), Biochem. J. 19, 338

In E + S ⇌ ES → E + P the complex forms at \(k_1[\mathrm{E}][\mathrm{S}]\), from the free
enzyme \([\mathrm{E}]_0 - [\mathrm{ES}]\), and is lost at \((k_{-1} + k_2)[\mathrm{ES}]\).
In the steady state the two balance, and the Michaelis–Menten concentration
\([\mathrm{ES}] = [\mathrm{E}]_0[\mathrm{S}] / (K_\mathrm{M} + [\mathrm{S}])\) satisfies
the balance: brackets that name concentrations.

```latex
K_\mathrm{M} := \frac{k_{-1} + k_2}{k_1}
[\mathrm{ES}] := \frac{[\mathrm{E}]_0 [\mathrm{S}]}{K_\mathrm{M} + [\mathrm{S}]}
k_1 \left([\mathrm{E}]_0 - [\mathrm{ES}]\right) [\mathrm{S}] = (k_{-1} + k_2) [\mathrm{ES}]
```

```python
k_1, k_m1, k_2, E0, S = symbols("k_1 k_m1 k_2 E_0 S", positive=True)
ES = symbols("ES")
K_M = (k_m1 + k_2) / k_1
steady = solve(Eq(k_1 * (E0 - ES) * S, (k_m1 + k_2) * ES), ES)
assert len(steady) == 1 and simplify(steady[0] - E0 * S / (K_M + S)) == 0
# the rate k_2 [ES] saturates at V_max = k_2 [E]_0, and is half of it at [S] = K_M
assert limit(k_2 * steady[0], S, oo) == k_2 * E0
assert simplify(steady[0].subs(S, K_M) - E0 / 2) == 0
expected = True
```

## A mechanism written with arrows

**Level 5** · `chem.mechanisms.lindemann` · Lindemann (1922), Trans. Faraday Soc. 17, 598

Lindemann's mechanism of a unimolecular reaction: molecules are activated by collisions,
and the activated ones decay. The rate constants sit on the arrows, mass action turns each
step into terms of the rate equations, and with \(\mathrm{A}^*\) in a steady state the
product forms at \(k_1 k_2 [\mathrm{A}]^2 / (k_{-1}[\mathrm{A}] + k_2)\). No design reads
a chemical equation as the rate equations it implies.

```latex
\mathrm{A} + \mathrm{A} \xrightleftharpoons[k_{-1}]{k_1} \mathrm{A}^* + \mathrm{A}
\mathrm{A}^* \xrightarrow{k_2} \mathrm{P}
\frac{\mathrm{d}[\mathrm{A}^*]}{\mathrm{d}t} = 0
\frac{\mathrm{d}[\mathrm{P}]}{\mathrm{d}t} = \frac{k_1 k_2 [\mathrm{A}]^2}{k_{-1} [\mathrm{A}] + k_2}
```

```python
k_1, k_m1, k_2, A, A_star = symbols("k_1 k_m1 k_2 A A_star", positive=True)
# mass action: A + A -> A* + A at k_1 [A]^2, back at k_-1 [A*][A], and A* -> P at k_2 [A*]
rate_A_star = k_1 * A**2 - k_m1 * A_star * A - k_2 * A_star
(steady,) = solve(Eq(rate_A_star, 0), A_star)
rate_P = k_2 * steady
assert simplify(rate_P - k_1 * k_2 * A**2 / (k_m1 * A + k_2)) == 0
# first order at high pressure, second order at low pressure
assert limit(rate_P / A, A, oo) == k_1 * k_2 / k_m1
assert limit(rate_P / A**2, A, 0) == k_1
expected = True
```

## Balancing a chemical equation

**Level 5** · `chem.mechanisms.balancing` · standard

The combustion of propane, with its stoichiometric coefficients unknown: conserving
carbon, hydrogen and oxygen, with \(a = 1\), gives
\(\mathrm{C_3H_8} + 5\,\mathrm{O_2} \rightarrow 3\,\mathrm{CO_2} + 4\,\mathrm{H_2O}\). A
chemical formula is not a product of symbols, and the arrow is not an equation between
them; no design reads either yet.

```latex
a\,\mathrm{C_3H_8} + b\,\mathrm{O_2} \longrightarrow c\,\mathrm{CO_2} + d\,\mathrm{H_2O}
a = 1
(a, b, c, d)
```

```python
# atoms in each species: columns C3H8, O2, CO2, H2O; rows C, H, O; products counted negative
composition = Matrix([[3, 0, -1, 0], [8, 0, 0, -2], [0, 2, -2, -1]])
(basis,) = composition.nullspace()  # one reaction: the coefficients are fixed up to a factor
expected = list(basis / basis[0])
assert expected == [1, 5, 3, 4]
a, b, c, d = expected
assert (3 * a, 8 * a, 2 * b) == (c, 2 * d, 2 * c + d)  # carbon, hydrogen, oxygen
```
