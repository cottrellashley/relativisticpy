# The field tensor

The electric and magnetic fields as one antisymmetric tensor \(F_{\mu\nu}\): its two
Lorentz invariants, its dual, and how a boost mixes \(\mathbf{E}\) and \(\mathbf{B}\).
Each case says whose conventions it follows, since the signs of \(F\) depend on them.

## The first invariant of the field

**Level 2** · `em.field-tensor.invariant` · Jackson (1999), section 11.9; Carroll (2004), section 1.8

The field tensor written as a matrix of its components, as Carroll writes it in the
signature \((-,+,+,+)\) with \(c = 1\), and contracted with itself: \(2(B^2 - E^2)\).

```latex
ds^2 = -dt^2 + dx^2 + dy^2 + dz^2
F_{\mu\nu} := \begin{pmatrix} 0 & -E_x & -E_y & -E_z \\ E_x & 0 & B_z & -B_y \\ E_y & -B_z & 0 & B_x \\ E_z & B_y & -B_x & 0 \end{pmatrix}
F_{\mu\nu} F^{\mu\nu}
```

```python
Ex, Ey, Ez, Bx, By, Bz = symbols("E_x E_y E_z B_x B_y B_z")
F = Matrix([[0, -Ex, -Ey, -Ez], [Ex, 0, Bz, -By], [Ey, -Bz, 0, Bx], [Ez, By, -Bx, 0]])
eta = diag(-1, 1, 1, 1)
F_up = eta * F * eta
expected = expand(sum(F[a, b] * F_up[a, b] for a in range(4) for b in range(4)))
assert expected == 2 * (Bx**2 + By**2 + Bz**2 - Ex**2 - Ey**2 - Ez**2)
```

## The fields seen from a moving frame

**Level 2** · `em.field-tensor.boost` · Jackson (1999), section 11.10

The field tensor transforms with two factors of the boost,
\(F'^{\mu\nu} = \Lambda^\mu{}_\alpha \Lambda^\nu{}_\beta F^{\alpha\beta}\). In Jackson's
conventions (signature \((+,-,-,-)\), \(F^{i0} = E_i\), \(c = 1\)) its components are the
transformed fields: \(E'_y = \gamma(E_y - \beta B_z)\) and so on.

```latex
ds^2 = dt^2 - dx^2 - dy^2 - dz^2
\gamma := \frac{1}{\sqrt{1 - \beta^2}}
\Lambda^\mu{}_\nu := \begin{pmatrix} \gamma & -\gamma\beta & 0 & 0 \\ -\gamma\beta & \gamma & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}
F^{\alpha\beta} := \begin{pmatrix} 0 & -E_x & -E_y & -E_z \\ E_x & 0 & -B_z & B_y \\ E_y & B_z & 0 & -B_x \\ E_z & -B_y & B_x & 0 \end{pmatrix}
F'^{\mu\nu} := \Lambda^\mu{}_\alpha \Lambda^\nu{}_\beta F^{\alpha\beta}
F'^{\mu\nu}
```

```python
beta = symbols("beta")
gamma = 1 / sqrt(1 - beta**2)
Ex, Ey, Ez, Bx, By, Bz = symbols("E_x E_y E_z B_x B_y B_z")
F = Matrix([[0, -Ex, -Ey, -Ez], [Ex, 0, -Bz, By], [Ey, Bz, 0, -Bx], [Ez, -By, Bx, 0]])
boost = Matrix(
    [[gamma, -gamma * beta, 0, 0], [-gamma * beta, gamma, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
)
F_moving = boost * F * boost.T
# Jackson (11.148): E'_x = E_x, E'_y = gamma (E_y - beta B_z), E'_z = gamma (E_z + beta B_y),
# B'_x = B_x, B'_y = gamma (B_y + beta E_z), B'_z = gamma (B_z - beta E_y)
expected = {
    ("x", "t"): Ex,
    ("y", "t"): gamma * (Ey - beta * Bz),
    ("z", "t"): gamma * (Ez + beta * By),
    ("z", "y"): Bx,
    ("x", "z"): gamma * (By + beta * Ez),
    ("y", "x"): gamma * (Bz - beta * Ey),
}
names = {"t": 0, "x": 1, "y": 2, "z": 3}
for (row, column), value in expected.items():
    assert simplify(F_moving[names[row], names[column]] - value) == 0
assert expand(F_moving + F_moving.T) == zeros(4, 4)
```

## The scalar product of E and B is invariant

**Level 3** · `em.field-tensor.e-dot-b` · Griffiths (2013), section 12.3.2

Griffiths's rules for the fields in a frame moving with \(v\) along \(x\), written as
three-vectors with primes, and the claim that \(\mathbf{E} \cdot \mathbf{B}\) is the same
in both frames (SI units).

```latex
\gamma := \frac{1}{\sqrt{1 - v^2/c^2}}
\mathbf{E} := (E_x, E_y, E_z), \quad \mathbf{B} := (B_x, B_y, B_z)
\mathbf{E}' := \left(E_x,\ \gamma (E_y - v B_z),\ \gamma (E_z + v B_y)\right)
\mathbf{B}' := \left(B_x,\ \gamma \left(B_y + \frac{v}{c^2} E_z\right),\ \gamma \left(B_z - \frac{v}{c^2} E_y\right)\right)
\mathbf{E}' \cdot \mathbf{B}' = \mathbf{E} \cdot \mathbf{B}
```

```python
v, c = symbols("v c")
Ex, Ey, Ez, Bx, By, Bz = symbols("E_x E_y E_z B_x B_y B_z")
gamma = 1 / sqrt(1 - v**2 / c**2)
E = Matrix([Ex, Ey, Ez])
B = Matrix([Bx, By, Bz])
E_moving = Matrix([Ex, gamma * (Ey - v * Bz), gamma * (Ez + v * By)])
B_moving = Matrix([Bx, gamma * (By + v / c**2 * Ez), gamma * (Bz - v / c**2 * Ey)])
assert simplify(E_moving.dot(B_moving) - E.dot(B)) == 0
expected = True
```

## The second invariant, made with the dual tensor

**Level 4** · `em.field-tensor.dual-invariant` · Jackson (1999), section 11.9

The dual tensor, made with the Levi-Civita symbol, and the second invariant
\(\mathcal{F}_{\alpha\beta} F^{\alpha\beta} = -4\, \mathbf{E} \cdot \mathbf{B}\). Jackson's
conventions: signature \((+,-,-,-)\) and \(\epsilon^{0123} = +1\).

```latex
ds^2 = dt^2 - dx^2 - dy^2 - dz^2
\epsilon^{0123} = 1
F^{\alpha\beta} := \begin{pmatrix} 0 & -E_x & -E_y & -E_z \\ E_x & 0 & -B_z & B_y \\ E_y & B_z & 0 & -B_x \\ E_z & -B_y & B_x & 0 \end{pmatrix}
\mathcal{F}^{\alpha\beta} := \frac{1}{2} \epsilon^{\alpha\beta\gamma\delta} F_{\gamma\delta}
\mathcal{F}_{\alpha\beta} F^{\alpha\beta}
```

```python
Ex, Ey, Ez, Bx, By, Bz = symbols("E_x E_y E_z B_x B_y B_z")
g = diag(1, -1, -1, -1)
F = Matrix([[0, -Ex, -Ey, -Ez], [Ex, 0, -Bz, By], [Ey, Bz, 0, -Bx], [Ez, -By, Bx, 0]])
F_down = g * F * g
dual = Matrix(
    4,
    4,
    lambda a, b: Rational(1, 2)
    * sum(LeviCivita(a, b, c, d) * F_down[c, d] for c in range(4) for d in range(4)),
)
# Jackson (11.141): the dual is F with E replaced by B and B by -E
swap = {Ex: Bx, Ey: By, Ez: Bz, Bx: -Ex, By: -Ey, Bz: -Ez}
assert dual == F.subs(swap, simultaneous=True)
dual_down = g * dual * g
expected = expand(sum(dual_down[a, b] * F[a, b] for a in range(4) for b in range(4)))
assert expected == -4 * (Ex * Bx + Ey * By + Ez * Bz)
```
