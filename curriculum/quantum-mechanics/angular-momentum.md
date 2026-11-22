# Angular momentum

Orbital angular momentum as differential operators on the sphere, with the spherical
harmonics \(Y_l^m\) as their eigenfunctions, and the addition of angular momenta: two
spins in the singlet, and the Clebsch–Gordan coefficients that couple \(l = 1\) to spin
one-half.

## \(L_z\) on an azimuthal phase

**Level 1** · `qm.angular-momentum.lz` · Griffiths (2005), section 4.3.2

\(L_z = -i\hbar\,\partial/\partial\phi\) acting on \(e^{im\phi}\): an eigenfunction, with
eigenvalue \(m\hbar\).

```latex
-i\hbar \frac{\partial}{\partial\phi} e^{im\phi}
```

```python
hbar, m, phi = symbols("hbar m phi")
expected = -I * hbar * diff(exp(I * m * phi), phi)
assert simplify(expected - m * hbar * exp(I * m * phi)) == 0
```

## \(L^2\) on a spherical harmonic

**Level 2** · `qm.angular-momentum.l-squared` · Griffiths (2005), section 4.3.2

\(L^2 = -\hbar^2 \left[\frac{1}{\sin\theta}\partial_\theta(\sin\theta\,\partial_\theta) +
\frac{1}{\sin^2\theta}\partial_\phi^2\right]\) on \(Y_2^1\), a function named with a
subscript and a superscript: the eigenvalue is \(l(l+1)\hbar^2 = 6\hbar^2\).

```latex
Y_2^1(\theta, \phi) := -\sqrt{\frac{15}{8\pi}} \sin\theta\cos\theta\, e^{i\phi}
-\hbar^2 \left[\frac{1}{\sin\theta}\frac{\partial}{\partial\theta}\left(\sin\theta \frac{\partial Y_2^1}{\partial\theta}\right) + \frac{1}{\sin^2\theta}\frac{\partial^2 Y_2^1}{\partial\phi^2}\right] = 6\hbar^2 Y_2^1
```

```python
theta, phi = symbols("theta phi", real=True)
hbar = symbols("hbar", positive=True)
Y = -sqrt(15 / (8 * pi)) * sin(theta) * cos(theta) * exp(I * phi)
assert simplify(Y - Ynm(2, 1, theta, phi).expand(func=True)) == 0  # Griffiths, Table 4.3
L2Y = -(hbar**2) * (
    diff(sin(theta) * diff(Y, theta), theta) / sin(theta) + diff(Y, phi, 2) / sin(theta) ** 2
)
assert simplify(L2Y - 2 * (2 + 1) * hbar**2 * Y) == 0
expected = True
```

## Two spins in the singlet

**Level 2** · `qm.angular-momentum.singlet` · Griffiths (2005), section 4.4.3

Operators on two spins are tensor products: \(\vec\sigma^{(1)}\cdot\vec\sigma^{(2)}\) on the
singlet \((|\uparrow\downarrow\rangle - |\downarrow\uparrow\rangle)/\sqrt{2}\), written as
a column, gives \(-3\), so its total spin is zero.

```latex
\sigma_x := \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
\sigma_y := \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}
\sigma_z := \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
\chi := \frac{1}{\sqrt{2}} \begin{pmatrix} 0 \\ 1 \\ -1 \\ 0 \end{pmatrix}
\left(\sigma_x \otimes \sigma_x + \sigma_y \otimes \sigma_y + \sigma_z \otimes \sigma_z\right) \chi = -3\chi
```

```python
from sympy.physics.quantum import TensorProduct

hbar = symbols("hbar", positive=True)
sigma = [Matrix([[0, 1], [1, 0]]), Matrix([[0, -I], [I, 0]]), Matrix([[1, 0], [0, -1]])]
chi = Matrix([0, 1, -1, 0]) / sqrt(2)
dot = sum((TensorProduct(s, s) for s in sigma), zeros(4, 4))
assert dot * chi == -3 * chi
# So S^2 = (hbar/2)^2 (sigma1 + sigma2)^2 = (hbar^2/4)(6 + 2 sigma1.sigma2) vanishes on it.
S_squared = hbar**2 / 4 * (6 * eye(4) + 2 * dot)
assert S_squared * chi == zeros(4, 1)
expected = True
```

## A Clebsch–Gordan coefficient

**Level 4** · `qm.angular-momentum.clebsch-gordan` · Griffiths (2005), section 4.4.3, Table 4.8

Coupling \(l = 1\) to spin one-half, in Sakurai's notation for the overlap of an
uncoupled and a coupled state,
\(\langle j_1 j_2; m_1 m_2 | j_1 j_2; j m\rangle\):
\(\langle 1, \tfrac12; 1, -\tfrac12 | 1, \tfrac12; \tfrac32, \tfrac12\rangle = 1/\sqrt{3}\).

```latex
\langle 1, \tfrac{1}{2}; 1, -\tfrac{1}{2} | 1, \tfrac{1}{2}; \tfrac{3}{2}, \tfrac{1}{2} \rangle
```

```python
from sympy.physics.quantum import TensorProduct
from sympy.physics.quantum.cg import CG

half = Rational(1, 2)


def lowering(j):  # J_- in the basis m = j, j - 1, ..., -j, in units of hbar
    ms = [j - k for k in range(int(2 * j) + 1)]
    return Matrix(len(ms), len(ms), lambda r, c: sqrt(j * (j + 1) - ms[c] * (ms[c] - 1)) if ms[r] == ms[c] - 1 else 0)


# Lower the stretched state |3/2, 3/2> = |1, 1>|1/2, 1/2> once, with J_- = L_- + S_-.
J_minus = TensorProduct(lowering(1), eye(2)) + TensorProduct(eye(3), lowering(half))
state = J_minus * TensorProduct(Matrix([1, 0, 0]), Matrix([1, 0]))
state = state / sqrt((state.T * state)[0])
coefficient = (TensorProduct(Matrix([1, 0, 0]), Matrix([0, 1])).T * state)[0]
assert coefficient == CG(1, 1, half, -half, Rational(3, 2), half).doit() == 1 / sqrt(3)
expected = coefficient
```
