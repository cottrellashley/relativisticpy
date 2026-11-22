# Time dilation, length contraction and the Doppler effect

What moving clocks, rods and light sources look like from another frame: the lifetime of
a fast muon, the ages of twins, the length of a moving rod, and the frequency of light
seen by an observer moving away from its source.

## Cosmic-ray muons live longer

**Level 1** · `sr.kinematics.muon-lifetime` · standard

Time dilation evaluated with numbers: a muon with a proper lifetime of 2.2 μs, moving at
\(0.995c\), lives about 22 μs in the laboratory.

```latex
\left. \frac{\tau}{\sqrt{1 - v^2/c^2}} \right|_{\tau = 2.2,\ v = 0.995 c}
```

```python
expected = 2.2 / np.sqrt(1 - 0.995**2)
assert abs(expected - 22.03) < 0.01
```

## The twin paradox

**Level 2** · `sr.kinematics.twin-paradox` · Langevin (1911)

The travelling twin's age is the proper time along her worldline. She travels out at
\(0.8c\) for five years and back for five, by the clocks at home, and ages six years: an
integral of a piecewise velocity.

```latex
v(t) := \begin{cases} \frac{4}{5} c & \text{if } t < 5 \\ -\frac{4}{5} c & \text{otherwise} \end{cases}
\int_0^{10} \sqrt{1 - \frac{v(t)^2}{c^2}}\, dt
```

```python
t, c = symbols("t c", positive=True)
v = Piecewise((Rational(4, 5) * c, t < 5), (-Rational(4, 5) * c, True))
expected = integrate(sqrt(1 - v**2 / c**2), (t, 0, 10))
assert expected == 6
```

## Length contraction, derived

**Level 3** · `sr.kinematics.length-contraction` · Einstein (1905), section 4

A rod at rest in the primed frame has its ends at \(x'_A\) and \(x'_B\). Measured at one
time \(t\) in the unprimed frame, where it is \(L\) long, its rest length \(L_0\) fixes
\(L = L_0 / \gamma\).

```latex
\gamma := \frac{1}{\sqrt{1 - v^2/c^2}}
x'_A := \gamma (x_A - v t)
x'_B := \gamma (x_A + L - v t)
solve(x'_B - x'_A = L_0, L)
```

```python
v, c, t, L, L0, xA = symbols("v c t L L_0 x_A")
gamma = 1 / sqrt(1 - v**2 / c**2)
expected = solve(Eq(gamma * (xA + L - v * t) - gamma * (xA - v * t), L0), L)
assert len(expected) == 1 and simplify(expected[0] - L0 / gamma) == 0
```

## The Doppler shift from four-vectors

**Level 3** · `sr.kinematics.doppler` · Einstein (1905), section 7; Schutz (2009), section 2.7

The frequency an observer measures is \(-k_\mu u^\mu\), for a photon of four-momentum
\(k^\mu\) and an observer of four-velocity \(u^\mu\). An observer moving with speed
\(v\) along the photon's direction measures it redshifted by
\(\sqrt{(1 - v)/(1 + v)}\) (\(c = 1\)).

```latex
ds^2 = -dt^2 + dx^2 + dy^2 + dz^2
k^\mu := (\omega, \omega, 0, 0)
u^\mu := \frac{1}{\sqrt{1 - v^2}} (1, v, 0, 0)
\omega_{\text{obs}} := -k_\mu u^\mu
\frac{\omega_{\text{obs}}}{\omega}
```

```python
omega, v = symbols("omega v")
eta = diag(-1, 1, 1, 1)
photon = Matrix([omega, omega, 0, 0])
observer = Matrix([1, v, 0, 0]) / sqrt(1 - v**2)
ratio = simplify(-(photon.T * eta * observer)[0] / omega)
expected = sqrt((1 - v) / (1 + v))
for speed in (Rational(1, 10), Rational(1, 2), Rational(9, 10)):
    assert abs(N((ratio - expected).subs(v, speed))) < 1e-12
```
