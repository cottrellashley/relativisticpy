# Lorentz transformations

Boosts written as coordinate transformations and as matrices, the rapidity that makes
collinear boosts add, the composition of velocities that follows from them, and the
rotation hidden in two boosts that are not collinear.

## The rapidity of three fifths of the speed of light

**Level 1** · `sr.lorentz.rapidity` · Jackson (1999), section 11.3

The inverse hyperbolic tangent, written as a textbook prints it: the rapidity of a boost
with \(\beta = 3/5\) is \(\ln 2\).

```latex
\tanh^{-1} \frac{3}{5}
```

```python
rapidity = atanh(Rational(3, 5))
assert simplify(rapidity.rewrite(log) - log(2)) == 0  # tanh(ln 2) = 3/5
expected = log(2)
```

## Two velocities composed

**Level 1** · `sr.lorentz.velocity-addition` · Einstein (1905), section 5

Einstein's law for composing collinear velocities, evaluated with an evaluation bar: half
the speed of light on half the speed of light is four fifths of it.

```latex
\left. \frac{u + v}{1 + \frac{u v}{c^2}} \right|_{u = c/2,\ v = c/2}
```

```python
u, v, c = symbols("u v c")
expected = simplify(((u + v) / (1 + u * v / c**2)).subs({u: c / 2, v: c / 2}))
assert expected == Rational(4, 5) * c
```

## A boost preserves the Minkowski metric

**Level 2** · `sr.lorentz.preserves-metric` · Carroll (2004), section 1.3

The defining property of a Lorentz transformation in matrix form, with the transpose
written as a superscript \(T\).

```latex
\gamma := \frac{1}{\sqrt{1 - \beta^2}}
\eta := \begin{pmatrix} -1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}
\Lambda := \begin{pmatrix} \gamma & -\gamma\beta & 0 & 0 \\ -\gamma\beta & \gamma & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}
\Lambda^T \eta \Lambda = \eta
```

```python
beta = symbols("beta")
gamma = 1 / sqrt(1 - beta**2)
eta = diag(-1, 1, 1, 1)
boost = Matrix(
    [[gamma, -gamma * beta, 0, 0], [-gamma * beta, gamma, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
)
assert simplify(boost.T * eta * boost - eta) == zeros(4, 4)
expected = True
```

## Collinear boosts add their rapidities

**Level 2** · `sr.lorentz.rapidities-add` · Jackson (1999), section 11.3

A boost written with its rapidity \(\zeta\), and the matrix identity that makes
rapidities add: it holds only through the addition formulas of \(\cosh\) and \(\sinh\).

```latex
B(\zeta) := \begin{pmatrix} \cosh\zeta & -\sinh\zeta \\ -\sinh\zeta & \cosh\zeta \end{pmatrix}
B(\zeta_1) B(\zeta_2) = B(\zeta_1 + \zeta_2)
```

```python
z1, z2 = symbols("zeta_1 zeta_2")


def boost(z):
    return Matrix([[cosh(z), -sinh(z)], [-sinh(z), cosh(z)]])


difference = boost(z1) * boost(z2) - boost(z1 + z2)
assert difference.applyfunc(lambda entry: simplify(expand_trig(entry))) == zeros(2, 2)
expected = True
```

## The composition of velocities, derived

**Level 3** · `sr.lorentz.velocity-composition` · Einstein (1905), section 5; Griffiths (2013), section 12.1.3

A particle moves with \(x = ut\) in one frame; the Lorentz transformation, written with
primed coordinates, gives its velocity \(x'/t'\) in a frame moving with \(v\).

```latex
\gamma := \frac{1}{\sqrt{1 - v^2/c^2}}
x := u t
x' := \gamma (x - v t)
t' := \gamma \left(t - \frac{v x}{c^2}\right)
\frac{x'}{t'}
```

```python
u, v, c, t = symbols("u v c t")
gamma = 1 / sqrt(1 - v**2 / c**2)
x = u * t
expected = simplify(gamma * (x - v * t) / (gamma * (t - v * x / c**2)))
assert simplify(expected - (u - v) / (1 - u * v / c**2)) == 0
```

## The Lorentz condition in index notation

**Level 4** · `sr.lorentz.index-form` · Carroll (2004), section 1.3

The same condition as Carroll writes it, with \(\eta_{\mu\nu}\) the Minkowski metric
given by name and the transformation carrying primed indices.

```latex
\eta_{\mu\nu} = \operatorname{diag}(-1, +1, +1, +1)
\Lambda^{\mu'}{}_{\nu} = \begin{pmatrix} \cosh\phi & -\sinh\phi & 0 & 0 \\ -\sinh\phi & \cosh\phi & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}
\eta_{\rho\sigma} = \Lambda^{\mu'}{}_{\rho} \Lambda^{\nu'}{}_{\sigma} \eta_{\mu'\nu'}
```

```python
phi = symbols("phi")
eta = diag(-1, 1, 1, 1)
boost = Matrix(
    [[cosh(phi), -sinh(phi), 0, 0], [-sinh(phi), cosh(phi), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
)
# Lambda^{mu'}_rho Lambda^{nu'}_sigma eta_{mu'nu'} is the (rho, sigma) entry of boost^T eta boost
assert simplify(boost.T * eta * boost - eta) == zeros(4, 4)
expected = True
```

## Two perpendicular boosts make a rotation

**Level 5** · `sr.lorentz.wigner-rotation` · standard (the Thomas–Wigner rotation)

Two boosts at right angles compose to a boost and a rotation about the third axis. The
program states the decomposition with an unknown rotation and an unknown boost, and asks
for the angle: no design of the language solves for the factors of a decomposition.

```latex
\gamma(\beta) := \frac{1}{\sqrt{1 - \beta^2}}
B_x(\beta) := \begin{pmatrix} \gamma(\beta) & -\beta\gamma(\beta) & 0 & 0 \\ -\beta\gamma(\beta) & \gamma(\beta) & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}
B_y(\beta) := \begin{pmatrix} \gamma(\beta) & 0 & -\beta\gamma(\beta) & 0 \\ 0 & 1 & 0 & 0 \\ -\beta\gamma(\beta) & 0 & \gamma(\beta) & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}
B_y(\beta_2)\, B_x(\beta_1) = R_z(\theta)\, B(\mathbf{u})
\cos\theta
```

```python
import scipy.linalg

b1, b2 = symbols("beta_1 beta_2")
g1, g2 = 1 / sqrt(1 - b1**2), 1 / sqrt(1 - b2**2)
expected = (g1 + g2) / (1 + g1 * g2)


def boost(beta, axis):
    matrix = np.eye(4)
    g = 1 / np.sqrt(1 - beta**2)
    matrix[0, 0] = matrix[axis, axis] = g
    matrix[0, axis] = matrix[axis, 0] = -g * beta
    return matrix


# The polar decomposition M = R B of a Lorentz matrix: B = sqrt(M^T M) is a pure boost,
# and R is a rotation, whose spatial trace is 1 + 2 cos(theta).
for v1, v2 in [(0.3, 0.6), (0.8, 0.5), (0.95, 0.9)]:
    composed = boost(v2, 2) @ boost(v1, 1)
    pure = scipy.linalg.sqrtm(composed.T @ composed).real
    rotation = composed @ np.linalg.inv(pure)
    assert np.allclose(rotation.T @ rotation, np.eye(4)) and np.allclose(rotation[0], [1, 0, 0, 0])
    cosine = (np.trace(rotation[1:, 1:]) - 1) / 2
    assert abs(cosine - float(expected.subs({b1: v1, b2: v2}))) < 1e-9
```
