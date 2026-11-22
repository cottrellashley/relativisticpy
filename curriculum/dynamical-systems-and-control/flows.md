# Fixed points and stability of flows

Differential equations \(\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})\) studied through their fixed
points, as Strogatz does: linear stability on the line, the Jacobian of a planar system,
Lyapunov functions and conserved quantities, the pendulum, the Lorenz equations and the
birth of their chaos, and the period of a relaxation oscillation.

## Linear stability on the line

**Level 1** · `dyn.flows.stability-1d` · Strogatz (2015), section 2.4

A fixed point \(x^*\) of \(\dot x = f(x)\) is stable when \(f'(x^*) < 0\). For logistic growth,
\(f(x) = x(1 - x)\), the fixed point \(x^* = 1\) has \(f'(1) = -1\).

```latex
\left. \frac{d}{dx}\, x(1 - x) \right|_{x=1}
```

```python
x = symbols("x")
expected = diff(x * (1 - x), x).subs(x, 1)
assert expected == -1
```

## Rabbits versus sheep

**Level 2** · `dyn.flows.rabbits-sheep` · Strogatz (2015), section 6.4

The Lotka–Volterra model of competition, \(\dot x = x(3 - x - 2y)\), \(\dot y = y(2 - x - y)\).
At the coexistence point \((1, 1)\) the Jacobian, written as its matrix of partial
derivatives and evaluated there, has determinant \(-1\): a saddle, so one species wins.

```latex
f(x, y) := x(3 - x - 2y)
g(x, y) := y(2 - x - y)
J := \left. \begin{pmatrix} \frac{\partial f}{\partial x} & \frac{\partial f}{\partial y} \\ \frac{\partial g}{\partial x} & \frac{\partial g}{\partial y} \end{pmatrix} \right|_{x = 1, y = 1}
\det J
```

```python
x, y = symbols("x y")
F = Matrix([x * (3 - x - 2 * y), y * (2 - x - y)])
assert F.subs({x: 1, y: 1}) == zeros(2, 1)  # a fixed point
J = F.jacobian([x, y]).subs({x: 1, y: 1})
assert J == Matrix([[-1, -2], [-1, -1]])
assert J.eigenvals() == {-1 - sqrt(2): 1, -1 + sqrt(2): 1}  # one of each sign
expected = J.det()
assert expected == -1
```

## A Lyapunov function

**Level 2** · `dyn.flows.lyapunov-function` · Strogatz (2015), section 7.2

For \(\dot x = -x + 4y\), \(\dot y = -x - y^3\), the function \(V = x^2 + 4y^2\) decreases along
every trajectory: \(\dot V = \frac{\partial V}{\partial x}\dot x + \frac{\partial V}{\partial y}\dot y
= -2x^2 - 8y^4\), which rules out closed orbits.

```latex
f(x, y) := -x + 4y
g(x, y) := -x - y^3
V(x, y) := x^2 + 4y^2
\frac{\partial V}{\partial x} f(x, y) + \frac{\partial V}{\partial y} g(x, y)
```

```python
x, y = symbols("x y")
V = x**2 + 4 * y**2
expected = expand(diff(V, x) * (-x + 4 * y) + diff(V, y) * (-x - y**3))
assert expected == -2 * x**2 - 8 * y**4
```

## Linearising the pendulum

**Level 3** · `dyn.flows.pendulum` · Strogatz (2015), section 6.7

The damped pendulum \(\dot\theta = \omega\), \(\dot\omega = -b\omega - \sin\theta\) as a vector
field, its Jacobian \(\partial F/\partial(\theta, \omega)\) at the origin, and the
characteristic polynomial there: \(\lambda^2 + b\lambda + 1\), a claim, which holds.

```latex
F(\theta, \omega) := \begin{pmatrix} \omega \\ -b\omega - \sin\theta \end{pmatrix}
J := \left. \frac{\partial F}{\partial (\theta, \omega)} \right|_{\theta = 0, \omega = 0}
\det(\lambda I - J) = \lambda^2 + b\lambda + 1
```

```python
theta, omega, b, lam = symbols("theta omega b lambda")
F = Matrix([omega, -b * omega - sin(theta)])
J = F.jacobian([theta, omega]).subs({theta: 0, omega: 0})
assert J == Matrix([[0, 1], [-1, -b]])
assert expand((lam * eye(2) - J).det() - (lam**2 + b * lam + 1)) == 0
expected = True
```

## A conserved quantity of predator and prey

**Level 3** · `dyn.flows.lotka-volterra-invariant` · standard

For the predator–prey equations \(\dot x = x(\alpha - \beta y)\), \(\dot y = y(\delta x - \gamma)\),
the function \(V = \delta x - \gamma\ln x + \beta y - \alpha\ln y\) is constant along
trajectories, so the orbits are closed: its derivative along the flow vanishes
identically.

```latex
\dot{x} := x(\alpha - \beta y)
\dot{y} := y(\delta x - \gamma)
V(x, y) := \delta x - \gamma \ln x + \beta y - \alpha \ln y
\frac{\partial V}{\partial x} \dot{x} + \frac{\partial V}{\partial y} \dot{y} = 0
```

```python
from scipy.integrate import solve_ivp

x, y, alpha, beta, gamma, delta = symbols("x y alpha beta gamma delta", positive=True)
V = delta * x - gamma * log(x) + beta * y - alpha * log(y)
flow = diff(V, x) * x * (alpha - beta * y) + diff(V, y) * y * (delta * x - gamma)
assert simplify(flow) == 0
rates = (1.0, 0.5, 0.8, 0.3)  # alpha, beta, gamma, delta
field = lambda t, v: [v[0] * (rates[0] - rates[1] * v[1]), v[1] * (rates[3] * v[0] - rates[2])]
orbit = solve_ivp(field, (0, 30), [2.0, 1.0], rtol=1e-11, atol=1e-11).y
energy = (
    rates[3] * orbit[0]
    - rates[2] * np.log(orbit[0])
    + rates[1] * orbit[1]
    - rates[0] * np.log(orbit[1])
)
assert np.ptp(energy) < 1e-8
expected = True
```

## The origin of the Lorenz equations

**Level 3** · `dyn.flows.lorenz-origin` · Strogatz (2015), section 9.2

The Lorenz equations with \(\sigma = 10\), \(r = 28\) and \(b = \frac{8}{3}\). At the origin the
Jacobian has the eigenvalue \(-b\) and the roots of \(\lambda^2 + (\sigma + 1)\lambda +
\sigma(1 - r)\), one of them positive for \(r > 1\): the largest is
\(\frac{-11 + \sqrt{1201}}{2} = 11.83\), and the origin is a saddle.

```latex
\sigma := 10
r := 28
b := \frac{8}{3}
F(x, y, z) := \begin{pmatrix} \sigma(y - x) \\ r x - y - x z \\ x y - b z \end{pmatrix}
J := \left. \frac{\partial F}{\partial (x, y, z)} \right|_{x = 0, y = 0, z = 0}
\lambda_{\max}(J)
```

```python
x, y, z = symbols("x y z")
F = Matrix([10 * (y - x), 28 * x - y - x * z, x * y - Rational(8, 3) * z])
J = F.jacobian([x, y, z]).subs({x: 0, y: 0, z: 0})
eigenvalues = J.eigenvals()
assert eigenvalues == {Rational(-8, 3): 1, (-11 + sqrt(1201)) / 2: 1, (-11 - sqrt(1201)) / 2: 1}
expected = (-11 + sqrt(1201)) / 2
assert abs(max(np.linalg.eigvals(np.array(J.tolist(), float)).real) - float(expected)) < 1e-12
```

## The Hopf bifurcation of the Lorenz equations

**Level 4** · `dyn.flows.lorenz-hopf` · Strogatz (2015), section 9.2

The nontrivial fixed points \(C^\pm = (\pm\sqrt{b(r-1)}, \pm\sqrt{b(r-1)}, r - 1)\) lose their
stability where a pair of eigenvalues of the Jacobian crosses the imaginary axis,
\(\lambda = i\omega\): at \(r_H = \frac{\sigma(\sigma + b + 3)}{\sigma - b - 1} = \frac{470}{19}
= 24.74\) for \(\sigma = 10\) and \(b = \frac{8}{3}\). Solving \(\det(i\omega I - J) = 0\) for the
parameter and the frequency together.

```latex
\sigma := 10
b := \frac{8}{3}
r > 1
\omega > 0
x^* := \sqrt{b(r - 1)}
J := \begin{pmatrix} -\sigma & \sigma & 0 \\ 1 & -1 & -x^* \\ x^* & x^* & -b \end{pmatrix}
solve(\det(i\omega I - J) = 0, [r, \omega])
```

```python
r, omega = symbols("r omega", positive=True)
sigma, b = 10, Rational(8, 3)
x = sqrt(b * (r - 1))
J = Matrix([[-sigma, sigma, 0], [1, -1, -x], [x, x, -b]])
polynomial = expand((I * omega * eye(3) - J).det())
found = solve([re(polynomial), im(polynomial)], [r, omega], dict=True)
assert found == [{r: Rational(470, 19), omega: 4 * sqrt(2090) / 19}]
assert Rational(470, 19) == sigma * (sigma + b + 3) / (sigma - b - 1)  # the book's formula
at = np.array(J.subs(r, Rational(470, 19)).evalf().tolist(), float)
assert abs(min(abs(np.linalg.eigvals(at).real))) < 1e-10  # a pair on the imaginary axis
expected = [(Rational(470, 19), 4 * sqrt(2090) / 19)]
```

## The period of a relaxation oscillation

**Level 5** · `dyn.flows.relaxation-period` · Strogatz (2015), section 7.5

The van der Pol oscillator with strong damping, \(\mu \gg 1\), has a limit cycle of period
\(T \approx (3 - 2\ln 2)\mu\): slow drifts along two branches of the cubic nullcline and fast
jumps between them. A limit cycle and its period, named in words, are beyond any design
of the language yet.

```latex
\ddot{x} - \mu(1 - x^2)\dot{x} + x = 0
T(\mu) := \text{the period of the limit cycle}
\lim_{\mu \to \infty} \frac{T(\mu)}{\mu}
```

```python
from scipy.integrate import solve_ivp

x = symbols("x", positive=True)
# On a slow branch dx/dt = -x / (mu (x^2 - 1)); x goes from 2 to 1, twice per period.
expected = 2 * integrate((x**2 - 1) / x, (x, 1, 2))
assert simplify(expected - (3 - 2 * log(2))) == 0
mu = 20
field = lambda t, v: [v[1], mu * (1 - v[0] ** 2) * v[1] - v[0]]
orbit = solve_ivp(
    field, (0, 6 * mu), [2, 0], method="Radau", rtol=1e-8, atol=1e-8, dense_output=True
)
times = np.linspace(2 * mu, 6 * mu, 200001)
position = orbit.sol(times)[0]
crossings = times[1:][(position[:-1] < 0) & (position[1:] >= 0)]
period = np.mean(np.diff(crossings))
airy = 2.338107410  # the next term of the asymptotics, 3 a mu^(-1/3), a the first zero of Ai
assert abs(period / mu - float(expected) - 3 * airy * mu ** (-4 / 3)) < 0.02
```
