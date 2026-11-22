# Linear equations of second order, and systems

Equations with constant coefficients: the Wronskian of two solutions, the free harmonic
oscillator, the undamped oscillator forced at resonance, the response of a damped one, an
initial value problem with an impulse, and a system of two equations.

## The Wronskian

**Level 1** · `de.second-order.wronskian` · Boyce and DiPrima (2012), section 3.2

The determinant of two solutions and their derivatives, written with the functions' names
and primes. It never vanishes, so \(e^{-2t}\) and \(e^{-3t}\) are a fundamental set.

```latex
y_1(t) := e^{-2t}
y_2(t) := e^{-3t}
\begin{vmatrix} y_1 & y_2 \\ y_1' & y_2' \end{vmatrix}
```

```python
t = symbols("t")
y1, y2 = exp(-2 * t), exp(-3 * t)
expected = simplify(Matrix([[y1, y2], [diff(y1, t), diff(y2, t)]]).det())
assert expected == -exp(-5 * t)
```

## The harmonic oscillator

**Level 2** · `de.second-order.harmonic` · Taylor (2005), section 5.2

The free, undamped oscillator in the physicist's notation, with dots for derivatives in
time: \(x = x_0 \cos\omega t + (v_0/\omega) \sin\omega t\) solves
\(\ddot{x} + \omega^2 x = 0\), and starts at \(x_0\) with velocity \(v_0\).

```latex
x(t) := x_0 \cos \omega t + \frac{v_0}{\omega} \sin \omega t
\ddot{x} + \omega^2 x = 0
x(0) = x_0
\dot{x}(0) = v_0
```

```python
t, w, x0, v0 = symbols("t omega x_0 v_0")
x = x0 * cos(w * t) + v0 / w * sin(w * t)
assert simplify(diff(x, t, 2) + w**2 * x) == 0
assert x.subs(t, 0) == x0 and simplify(diff(x, t).subs(t, 0) - v0) == 0
outputs = [True, True, True]
```

## Resonance

**Level 2** · `de.second-order.resonance` · Boyce and DiPrima (2012), section 3.8

An undamped oscillator forced at its natural frequency \(\omega_0 = \sqrt{k/m}\) responds
with an amplitude that grows linearly in time. The book's solution, put into the equation
of motion as the book writes it.

```latex
k := m \omega_0^2
u(t) := \frac{F_0}{2 m \omega_0} t \sin \omega_0 t
m u'' + k u = F_0 \cos \omega_0 t
```

```python
t, m, w0, F0 = symbols("t m omega_0 F_0")
u = F0 / (2 * m * w0) * t * sin(w0 * t)
assert simplify(m * diff(u, t, 2) + m * w0**2 * u - F0 * cos(w0 * t)) == 0
expected = True
```

## The response of a damped oscillator

**Level 3** · `de.second-order.amplitude` · Boyce and DiPrima (2012), section 3.8

The amplitude \(R\) of the steady response to a force \(F_0 \cos \omega t\), and the
frequencies where it is stationary: zero, and \(\pm\omega_{\max}\), where
\(\omega_{\max}^2 = \omega_0^2 (1 - \gamma^2 / 2mk)\) and the amplitude peaks.

```latex
R(\omega) := \frac{F_0}{\sqrt{m^2 (\omega_0^2 - \omega^2)^2 + \gamma^2 \omega^2}}
solve(\frac{dR}{d\omega} = 0, \omega)
```

```python
w, m, g, w0, F0 = symbols("omega m gamma omega_0 F_0")
R = F0 / sqrt(m**2 * (w0**2 - w**2) ** 2 + g**2 * w**2)
expected = solve(Eq(diff(R, w), 0), w)
k = m * w0**2
assert expected[0] == 0 and expected[1] == -expected[2]
assert simplify(expected[2] ** 2 - w0**2 * (1 - g**2 / (2 * m * k))) == 0
```

## An impulse

**Level 3** · `de.second-order.impulse` · Boyce and DiPrima (2012), section 6.5

An oscillator at rest, struck at \(t = 5\): a Dirac delta forces it, the kind of problem
the Laplace transform is made for. Nothing moves until the impulse; then
\(y = \frac{2}{\sqrt{15}}\, u_5(t)\, e^{-(t - 5)/4} \sin\frac{\sqrt{15}}{4}(t - 5)\).

```latex
dsolve(2y'' + y' + 2y = \delta(t - 5), y(0) = 0, y'(0) = 0)
```

```python
t = symbols("t")
y = Function("y")
equation = Eq(2 * y(t).diff(t, 2) + y(t).diff(t) + 2 * y(t), DiracDelta(t - 5))
expected = dsolve(equation, y(t), ics={y(0): 0, y(t).diff(t).subs(t, 0): 0})
book = 2 / sqrt(15) * Heaviside(t - 5) * exp(-(t - 5) / 4) * sin(sqrt(15) * (t - 5) / 4)
assert all(abs(N((expected.rhs - book).subs(t, value))) < 1e-12 for value in (1, 6, 7.5, 12))
```

## A system of two equations

**Level 2** · `de.second-order.system` · Boyce and DiPrima (2012), section 7.5

\(\mathbf{x}' = A\mathbf{x}\): the combination of the solutions \(\boldsymbol{\xi} e^{rt}\)
made from the eigenvalues 3 and \(-1\) of \(A\) and their eigenvectors solves the system,
whatever its constants.

```latex
A := \begin{pmatrix} 1 & 1 \\ 4 & 1 \end{pmatrix}
\mathbf{x}(t) := c_1 \begin{pmatrix} 1 \\ 2 \end{pmatrix} e^{3t} + c_2 \begin{pmatrix} 1 \\ -2 \end{pmatrix} e^{-t}
\mathbf{x}' = A \mathbf{x}
```

```python
t, c1, c2 = symbols("t c_1 c_2")
A = Matrix([[1, 1], [4, 1]])
assert A.eigenvals() == {3: 1, -1: 1}
x = c1 * Matrix([1, 2]) * exp(3 * t) + c2 * Matrix([1, -2]) * exp(-t)
assert simplify(x.diff(t) - A * x) == zeros(2, 1)
expected = True
```
