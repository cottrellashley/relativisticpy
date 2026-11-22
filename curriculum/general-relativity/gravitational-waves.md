# Gravitational waves

Linearised gravity about flat spacetime: plane waves, the transverse–traceless gauge, the
two polarisations and how they mix under a rotation, and the curvature a wave carries.

## A plane wave solves the wave equation

**Level 2** · `gr.waves.plane-wave` · Carroll (2004), section 7.4

In the TT gauge each component of the perturbation obeys \(\Box h_{\mu\nu} = 0\); a plane
wave moving along \(z\) does.

```latex
ds^2 = -dt^2 + dx^2 + dy^2 + dz^2
h := A \cos[\omega (t - z)]
\Box h = 0
```

```python
t, x, y, z, A, omega = symbols("t x y z A omega")
h = A * cos(omega * (t - z))
box = -diff(h, t, 2) + diff(h, x, 2) + diff(h, y, 2) + diff(h, z, 2)
expected = simplify(box) == 0
assert expected
```

## The transverse–traceless gauge

**Level 2** · `gr.waves.tt-gauge` · Carroll (2004), section 7.4

A wave along \(z\) with polarisations \(h_+\) and \(h_\times\) is transverse to its wave
vector, \(k^\mu h_{\mu\nu} = 0\), and traceless, \(h^\mu{}_\mu = 0\).

```latex
ds^2 = -dt^2 + dx^2 + dy^2 + dz^2
h_{\mu\nu} := \begin{pmatrix} 0 & 0 & 0 & 0 \\ 0 & h_+ & h_\times & 0 \\ 0 & h_\times & -h_+ & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix} \cos[\omega (t - z)]
k^\mu := [\omega, 0, 0, \omega]
k^\mu h_{\mu\nu} = 0
h^\mu{}_\mu = 0
```

```python
t, z, omega = symbols("t z omega")
hp, hx = symbols("h_+ h_\\times")
eta = diag(-1, 1, 1, 1)
h = Matrix([[0, 0, 0, 0], [0, hp, hx, 0], [0, hx, -hp, 0], [0, 0, 0, 0]]) * cos(omega * (t - z))
k = Matrix([omega, 0, 0, omega])
assert (k.T * eta * k)[0] == 0  # a null wave vector
transverse = (k.T * h) == zeros(1, 4)
traceless = simplify((eta.inv() * h).trace()) == 0
outputs = [transverse, traceless]
assert all(outputs)
```

## The polarisations under a rotation

**Level 3** · `gr.waves.polarisation-rotation` · Carroll (2004), section 7.4

Rotating the transverse plane by \(\psi\) mixes the polarisations by \(2\psi\): the mark of
a field of helicity two. The transformation law of a tensor, with a rotation matrix.

```latex
ds^2 = dx^2 + dy^2
h_{ij} := \begin{pmatrix} h_+ & h_\times \\ h_\times & -h_+ \end{pmatrix}
R^i{}_j := \begin{pmatrix} \cos\psi & \sin\psi \\ -\sin\psi & \cos\psi \end{pmatrix}
h'_{ij} := R_i{}^k R_j{}^l h_{kl}
h'_{xx}
```

```python
psi = symbols("psi")
hp, hx = symbols("h_+ h_\\times")
h = Matrix([[hp, hx], [hx, -hp]])
R = Matrix([[cos(psi), sin(psi)], [-sin(psi), cos(psi)]])
rotated = R * h * R.T  # h'_{ij} = R_i^k R_j^l h_{kl}, with the Euclidean metric
expected = simplify(rotated[0, 0])
assert simplify(expected - (hp * cos(2 * psi) + hx * sin(2 * psi))) == 0
```

## The curvature of a gravitational wave

**Level 4** · `gr.waves.linearised-riemann` · Carroll (2004), section 7.1

The Riemann tensor to first order in the perturbation, written with partial derivatives
in index notation, for a plus-polarised wave: \(R_{txtx} = -\frac{1}{2}\partial_t^2
h_{xx}\).

```latex
ds^2 = -dt^2 + dx^2 + dy^2 + dz^2
h_{\mu\nu} := h_+ \cos[\omega (t - z)] \begin{pmatrix} 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}
R_{\mu\nu\rho\sigma} := \frac{1}{2} \left( \partial_\rho \partial_\nu h_{\mu\sigma} + \partial_\sigma \partial_\mu h_{\nu\rho} - \partial_\sigma \partial_\nu h_{\mu\rho} - \partial_\rho \partial_\mu h_{\nu\sigma} \right)
R_{txtx}
```

```python
t, x, y, z, omega, eps = symbols("t x y z omega epsilon")
hp = symbols("h_+")
X = [t, x, y, z]
wave = hp * cos(omega * (t - z))
h = diag(0, wave, -wave, 0)


def linear(m, v, p, s):  # Carroll's linearised Riemann tensor
    d = lambda f, a, b: diff(f, X[a], X[b])
    return (d(h[m, s], p, v) + d(h[v, p], s, m) - d(h[m, p], s, v) - d(h[v, s], p, m)) / 2


expected = simplify(linear(0, 1, 0, 1))
assert simplify(expected - omega**2 * wave / 2) == 0
# the same from the exact curvature of eta + eps h, to first order in eps
g = diag(-1, 1, 1, 1) + eps * h
gi = g.inv()
n = 4
Gam = [
    [
        [
            sum(
                gi[a, d] * (diff(g[d, b], X[c]) + diff(g[d, c], X[b]) - diff(g[b, c], X[d]))
                for d in range(n)
            )
            / 2
            for c in range(n)
        ]
        for b in range(n)
    ]
    for a in range(n)
]
up = (
    diff(Gam[0][1][1], X[0])
    - diff(Gam[0][0][1], X[1])  # R^t_{xtx}
    + sum(Gam[0][0][e] * Gam[e][1][1] - Gam[0][1][e] * Gam[e][0][1] for e in range(n))
)
exact = g[0, 0] * up  # R_{txtx}, the metric being diagonal
assert simplify(series(exact, eps, 0, 2).removeO() - eps * expected) == 0
```
