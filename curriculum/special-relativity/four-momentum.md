# Four-momentum

Four-vectors and their invariant products: the mass shell, the invariant mass of a pair,
and the conservation of four-momentum in collisions, from thresholds to phase space.
Units with \(c = 1\).

## Two upper indices are not a contraction

**Level 1** · `sr.four-momentum.two-upper-indices` · standard (Einstein's summation convention)

With a Lorentzian metric in force, an index summed twice in the upper position is wrong
notation, and must be refused rather than summed.

```latex
ds^2 = -dt^2 + dx^2 + dy^2 + dz^2
p^\mu := (E, p, 0, 0)
p^\mu p^\mu
```

```python
error = "IndexNotationError"
message = "upper index"
```

## The mass shell

**Level 2** · `sr.four-momentum.mass-shell` · Schutz (2009), section 2.4

The norm of the four-momentum is minus the mass squared; solved for the energy, it gives
both sheets of the mass shell.

```latex
ds^2 = -dt^2 + dx^2 + dy^2 + dz^2
p^\mu := (E, p, 0, 0)
solve(p_\mu p^\mu = -m^2, E)
```

```python
E, p, m = symbols("E p m")
eta = diag(-1, 1, 1, 1)
momentum = Matrix([E, p, 0, 0])
expected = solve(Eq((momentum.T * eta * momentum)[0], -(m**2)), E)
assert expected == [-sqrt(m**2 + p**2), sqrt(m**2 + p**2)]
```

## The invariant mass of two photons

**Level 2** · `sr.four-momentum.two-photon-mass` · standard

Two labelled four-vectors, \(k_1^\mu\) and \(k_2^\mu\), lowered as \(k_{1\mu}\) and
\(k_{2\mu}\): the label and the index share the subscript, as particle physicists write
them. Minus the square of their sum is the invariant mass squared of the pair,
\(2 E_1 E_2 (1 - \cos\theta)\).

```latex
ds^2 = -dt^2 + dx^2 + dy^2 + dz^2
k_1^\mu := (E_1, E_1, 0, 0)
k_2^\mu := (E_2, E_2 \cos\theta, E_2 \sin\theta, 0)
-(k_1^\mu + k_2^\mu)(k_{1\mu} + k_{2\mu})
```

```python
E1, E2, theta = symbols("E_1 E_2 theta")
eta = diag(-1, 1, 1, 1)
total = Matrix([E1, E1, 0, 0]) + Matrix([E2, E2 * cos(theta), E2 * sin(theta), 0])
expected = simplify(-(total.T * eta * total)[0])
assert simplify(expected - 2 * E1 * E2 * (1 - cos(theta))) == 0
```

## The threshold for making antiprotons

**Level 3** · `sr.four-momentum.antiproton-threshold` · Chamberlain, Segrè, Wiegand and Ypsilantis (1955), Phys. Rev. 100, 947

A proton strikes a proton at rest and makes \(p + p \to p + p + p + \bar p\). At threshold
the invariant mass of the pair is that of four protons at rest: the beam energy is
\(7m\), a kinetic energy of \(6m \approx 5.6\) GeV.

```latex
ds^2 = -dt^2 + dx^2 + dy^2 + dz^2
p_a^\mu := (E, \sqrt{E^2 - m^2}, 0, 0)
p_b^\mu := (m, 0, 0, 0)
P^\mu := p_a^\mu + p_b^\mu
solve(-P_\mu P^\mu = (4 m)^2, E)
```

```python
E, m = symbols("E m", positive=True)
eta = diag(-1, 1, 1, 1)
total = Matrix([E, sqrt(E**2 - m**2), 0, 0]) + Matrix([m, 0, 0, 0])
expected = solve(Eq(-(total.T * eta * total)[0], (4 * m) ** 2), E)
assert expected == [7 * m]
assert abs(6 * 0.93827208816 - 5.6) < 0.05  # GeV, the threshold of the Bevatron experiment
```

## Compton scattering

**Level 3** · `sr.four-momentum.compton` · Compton (1923), Phys. Rev. 21, 483

A photon scatters off an electron at rest. Conservation of four-momentum, with the
outgoing momenta primed, and the mass shell of the outgoing electron give the energy of
the scattered photon.

```latex
ds^2 = -dt^2 + dx^2 + dy^2 + dz^2
k^\mu := (E, E, 0, 0)
k'^\mu := (E', E' \cos\theta, E' \sin\theta, 0)
p^\mu := (m, 0, 0, 0)
p'^\mu := p^\mu + k^\mu - k'^\mu
solve(p'_\mu p'^\mu = -m^2, E')
```

```python
E, E_out, m, theta = symbols("E E_out m theta")
eta = diag(-1, 1, 1, 1)
k = Matrix([E, E, 0, 0])
k_out = Matrix([E_out, E_out * cos(theta), E_out * sin(theta), 0])
electron = Matrix([m, 0, 0, 0]) + k - k_out
# The terms in E'^2 cancel only through cos^2 + sin^2 = 1: the equation is linear in E'.
norm = trigsimp(expand((electron.T * eta * electron)[0]))
expected = solve(Eq(norm, -(m**2)), E_out)
# Compton's law, 1/E' - 1/E = (1 - cos(theta))/m: the shift of wavelength h/(mc) (1 - cos(theta))
assert len(expected) == 1
assert simplify(1 / expected[0] - 1 / E - (1 - cos(theta)) / m) == 0
```

## The Mandelstam variables sum to the masses

**Level 4** · `sr.four-momentum.mandelstam` · Mandelstam (1958), Phys. Rev. 112, 1344

Four-momenta written without components, as papers write them, with the metric
\((+,-,-,-)\) and \(p^2 = p \cdot p\): from conservation and the mass shells alone,
\(s + t + u\) is the sum of the four masses squared.

```latex
s = (p_1 + p_2)^2, \quad t = (p_1 - p_3)^2, \quad u = (p_1 - p_4)^2
p_1 + p_2 = p_3 + p_4, \quad p_i^2 = m_i^2
s + t + u = m_1^2 + m_2^2 + m_3^2 + m_4^2
```

```python
components = symbols("a0:4 b0:4 c0:4")
eta = diag(1, -1, -1, -1)
p1, p2, p3 = (Matrix(components[i : i + 4]) for i in (0, 4, 8))
p4 = p1 + p2 - p3


def square(vector):
    return (vector.T * eta * vector)[0]


s, t, u = square(p1 + p2), square(p1 - p3), square(p1 - p4)
masses = square(p1) + square(p2) + square(p3) + square(p4)
assert expand(s + t + u - masses) == 0
expected = True
```

## Two-body phase space

**Level 5** · `sr.four-momentum.two-body-phase-space` · PDG, Review of Particle Physics, "Kinematics"

The Lorentz-invariant phase space of two particles, an integral over two momenta against
a four-dimensional delta function, in the centre-of-momentum frame. It is
\(\pi |\mathbf{p}^*| / \sqrt{s}\), where \(|\mathbf{p}^*|\) is the momentum of either
particle; in the PDG's normalisation, with \((2\pi)^4\) and \((2\pi)^{-3}\) per particle,
\(\int d\Phi_2 = |\mathbf{p}^*| / (4\pi\sqrt{s})\).

```latex
P^\mu = (\sqrt{s}, 0, 0, 0), \quad E_i = \sqrt{|\mathbf{p}_i|^2 + m_i^2}
\int \frac{d^3 p_1}{2 E_1} \frac{d^3 p_2}{2 E_2} \, \delta^4(P - p_1 - p_2)
```

```python
s, m1, m2, p = symbols("s m_1 m_2 p", positive=True)
kallen = (s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2)
p_star = sqrt(kallen) / (2 * sqrt(s))
E1, E2 = sqrt(p**2 + m1**2), sqrt(p**2 + m2**2)
# The spatial delta sets p_2 = -p_1. Then d^3p_1 = 4 pi p^2 dp, and
# delta(sqrt(s) - E1 - E2) = delta(p - p*) / |d(E1 + E2)/dp|.
density = 4 * pi * p**2 / (4 * E1 * E2) / diff(E1 + E2, p)
expected = pi * sqrt(kallen) / (2 * s)
for values in ({s: 50, m1: 3, m2: 2}, {s: 7, m1: 1, m2: Rational(1, 2)}, {s: 4, m1: 0, m2: 0}):
    at = p_star.subs(values)
    assert abs(N((E1 + E2).subs(values).subs(p, at) - sqrt(values[s]))) < 1e-12  # on shell
    assert abs(N(density.subs(values).subs(p, at) - expected.subs(values))) < 1e-12
assert simplify(expected - pi * p_star / sqrt(s)) == 0
```
