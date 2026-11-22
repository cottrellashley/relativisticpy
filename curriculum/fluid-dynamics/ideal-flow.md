# Bernoulli's equation and potential flow

Flow without viscosity: Bernoulli's equation, which trades pressure for speed along a
streamline, and the two-dimensional potential flows of a complex potential
\(w(z) = \phi + i\psi\), whose derivative is the complex velocity \(u - iv\). The flow past
a circular cylinder, \(w = U(z + a^2/z)\), without circulation and with it. Acheson (1990)
is *Elementary Fluid Dynamics*.

## Torricelli's law

**Level 2** · `fluid.ideal.torricelli` · standard

Bernoulli's equation from the free surface of a tank, where the fluid is nearly at rest, to
a hole a depth \(h\) below it, both at atmospheric pressure, solved for the speed of the
jet: \(v = \sqrt{2gh}\). The speed is positive, as the program says, which picks one root.

```latex
\rho, g, h > 0
v > 0
solve\left(p_a + \rho g h = p_a + \frac{1}{2}\rho v^2, v\right)
```

```python
p_a, w = symbols("p_a w")
rho, g, h, v = symbols("rho g h v", positive=True)
expected = solve(Eq(p_a + rho * g * h, p_a + rho * v**2 / 2), v)
assert expected == [sqrt(2 * g * h)]
# without the sign, two roots: the negative one is not a speed
assert len(solve(Eq(p_a + rho * g * h, p_a + rho * w**2 / 2), w)) == 2
assert abs(float(expected[0].subs({g: 9.81, h: 5})) - 9.9045) < 1e-4  # m/s, 5 m down
```

## The Venturi meter

**Level 3** · `fluid.ideal.venturi` · standard

Continuity, \(A_1 v_1 = A_2 v_2\), and Bernoulli's equation along a horizontal tube that
narrows from the area \(A_1\) to \(A_2\), solved together for the speed and the pressure at the throat:
the pressure falls there by \(\frac12\rho v_1^2 (A_1^2/A_2^2 - 1)\).

```latex
solve\left([A_1 v_1 = A_2 v_2, p_1 + \frac{1}{2}\rho v_1^2 = p_2 + \frac{1}{2}\rho v_2^2], [v_2, p_2]\right)
```

```python
A_1, A_2, v_1, v_2, p_1, p_2, rho = symbols("A_1 A_2 v_1 v_2 p_1 p_2 rho")
equations = [Eq(A_1 * v_1, A_2 * v_2), Eq(p_1 + rho * v_1**2 / 2, p_2 + rho * v_2**2 / 2)]
expected = solve(equations, [v_2, p_2], dict=False)
throat_speed = A_1 * v_1 / A_2
throat_pressure = p_1 - rho * v_1**2 * (A_1**2 / A_2**2 - 1) / 2
assert len(expected) == 1
assert simplify(expected[0][0] - throat_speed) == 0
assert simplify(expected[0][1] - throat_pressure) == 0
# water at 2 m/s into a throat of half the area: the pressure falls by 6 kPa
assert throat_pressure.subs({rho: 1000, v_1: 2, A_1: 2, A_2: 1, p_1: 0}) == -6000
```

## The stagnation points of the flow past a cylinder

**Level 2** · `fluid.ideal.stagnation-points` · Acheson (1990), chapter 4

The complex velocity \(dw/dz = U(1 - a^2/z^2)\) of a uniform stream past a cylinder of
radius \(a\) vanishes at two points, the front and the back of the cylinder, \(z = \pm a\).

```latex
U, a > 0
w(z) := U\left(z + \frac{a^2}{z}\right)
solve\left(\frac{dw}{dz} = 0, z\right)
```

```python
z = symbols("z")
U, a = symbols("U a", positive=True)
w = U * (z + a**2 / z)
expected = solve(Eq(diff(w, z), 0), z)  # in the order SymPy gives them
assert expected == [-a, a]
assert all(abs(point) == a for point in expected)  # both on the cylinder, |z| = a
assert diff(w, z).subs(z, I * a) == 2 * U  # where the fluid is fastest, at the top
```

## The pressure on the cylinder

**Level 3** · `fluid.ideal.cylinder-pressure` · Acheson (1990), chapter 4

On the cylinder, \(z = a e^{i\theta}\), the speed is \(|w'(z)|\), and Bernoulli's equation
gives the pressure coefficient \(C_p = (p - p_\infty)/\frac12\rho U^2 = 1 - |w'|^2/U^2 = 1
- 4\sin^2\theta\): the modulus of a complex expression, for real \(\theta\).

```latex
U, a > 0
\theta \in \mathbb{R}
w(z) := U\left(z + \frac{a^2}{z}\right)
1 - \frac{\left|w'(a e^{i\theta})\right|^2}{U^2}
```

```python
z = symbols("z")
U, a, r = symbols("U a r", positive=True)
theta = symbols("theta", real=True)
w = U * (z + a**2 / z)
speed_squared = Abs(diff(w, z).subs(z, a * exp(I * theta))) ** 2
expected = 1 - 4 * sin(theta) ** 2
assert simplify((1 - speed_squared / U**2).rewrite(cos) - expected) == 0
# the same from the velocity potential U (r + a^2/r) cos(theta): on r = a only u_theta remains
phi = U * (r + a**2 / r) * cos(theta)
assert diff(phi, r).subs(r, a) == 0
u_theta = diff(phi, theta) / r
assert simplify(1 - (u_theta.subs(r, a) / U) ** 2 - expected) == 0
```

## d'Alembert's paradox

**Level 2** · `fluid.ideal.dalembert` · standard

The pressure \(p_\infty + \frac12\rho U^2(1 - 4\sin^2\theta)\) on a cylinder in potential
flow is the same at the back as at the front, so the drag, the integral of \(-p\cos\theta\)
round the cylinder, vanishes: d'Alembert's paradox.

```latex
p(\theta) := p_\infty + \frac{1}{2}\rho U^2\left(1 - 4\sin^2\theta\right)
-\int_0^{2\pi} p(\theta) \cos\theta \, a \, d\theta
```

```python
theta, p_inf, rho, U, a = symbols("theta p_infty rho U a")
p = p_inf + rho * U**2 * (1 - 4 * sin(theta) ** 2) / 2
expected = -integrate(p * cos(theta) * a, (theta, 0, 2 * pi))
assert expected == 0
# p(pi - theta) = p(theta), while cos(pi - theta) = -cos(theta): the back cancels the front
assert simplify(p.subs(theta, pi - theta) - p) == 0
assert -integrate(p * sin(theta) * a, (theta, 0, 2 * pi)) == 0  # and there is no lift either
```

## The lift on a cylinder with circulation

**Level 4** · `fluid.ideal.blasius-lift` · Acheson (1990), chapter 4

A clockwise circulation \(\Gamma\) round the cylinder adds \(\frac{i\Gamma}{2\pi}\ln z\) to
the complex potential. Blasius's theorem gives the force as a contour integral round the
cylinder, \(X - iY = \frac{i\rho}{2}\oint (dw/dz)^2\,dz = -i\rho U\Gamma\): a lift
\(\rho U \Gamma\), the Kutta–Joukowski theorem, and still no drag.

```latex
U, a > 0
w(z) := U\left(z + \frac{a^2}{z}\right) + \frac{i\Gamma}{2\pi}\ln z
\frac{i\rho}{2} \oint_{|z| = a} \left(\frac{dw}{dz}\right)^2 dz
```

```python
z, rho, Gamma, theta, p_inf = symbols("z rho Gamma theta p_infty")
U, a = symbols("U a", positive=True)
w = U * (z + a**2 / z) + I * Gamma / (2 * pi) * log(z)
expected = I * rho / 2 * 2 * pi * I * residue(expand(diff(w, z) ** 2), z, 0)
assert simplify(expected + I * rho * U * Gamma) == 0
# the same from the pressure on the cylinder, where the speed is |2U sin(theta) + Gamma/(2 pi a)|:
# the upward force is -int p sin(theta) a dtheta, and the drag vanishes
q = 2 * U * sin(theta) + Gamma / (2 * pi * a)
p = p_inf + rho * (U**2 - q**2) / 2
assert simplify(-integrate(p * sin(theta) * a, (theta, 0, 2 * pi)) - rho * U * Gamma) == 0
assert integrate(p * cos(theta) * a, (theta, 0, 2 * pi)) == 0
# and numerically round the circle, at U = 1.3, a = 0.7, Gamma = 2.1, rho = 1.2
t = np.linspace(0, 2 * np.pi, 4001)[:-1]
points = 0.7 * np.exp(1j * t)
dw_dz = 1.3 * (1 - 0.7**2 / points**2) + 2.1j / (2 * np.pi * points)
force = 1j * 1.2 / 2 * np.mean(dw_dz**2 * 1j * points) * 2 * np.pi
assert abs(force - (-1j * 1.2 * 1.3 * 2.1)) < 1e-12
```
