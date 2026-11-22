# Integrating ordinary differential equations

Euler's method and the classical Runge–Kutta method, as recurrences on a grid of times,
checked against exact solutions: Burden and Faires's running example
\(y' = y - t^2 + 1\), \(y(0) = 0.5\), whose solution is \(y(t) = (t + 1)^2 - \frac{1}{2}e^t\),
and the convergence of Euler's method for \(y' = y\) to \(e\).

## Euler's method

**Level 2** · `num.ode.euler` · Burden and Faires (2011), section 5.2

Ten steps of \(w_{i+1} = w_i + h f(t_i, w_i)\) with \(h = 0.2\), the recurrence written as the
book writes it: \(w_{10} = 4.8657845\), against \(y(2) = 5.3054720\).

```latex
f(t, y) := y - t^2 + 1
h := 0.2
t_i := i h
w_0 := 0.5
w_{i+1} := w_i + h f(t_i, w_i)
w_{10}
```

```python
f = lambda t, y: y - t**2 + 1
h = Rational(1, 5)
w = Rational(1, 2)
for i in range(10):
    w = w + h * f(i * h, w)
expected = w
assert abs(float(expected) - 4.8657845) < 5e-8  # the book's table
t = symbols("t")
y = (t + 1) ** 2 - exp(t) / 2
assert simplify(diff(y, t) - (y - t**2 + 1)) == 0 and y.subs(t, 0) == Rational(1, 2)
assert abs(float(y.subs(t, 2)) - 5.3054720) < 5e-8
```

## The Runge–Kutta method of order four

**Level 3** · `num.ode.rk4` · Burden and Faires (2011), section 5.4

The classical Runge–Kutta method on the same problem, with the four stages \(k_1, \ldots,
k_4\) of each step defined before the step itself: \(w_{10} = 5.3053630\), within
\(1.1 \times 10^{-4}\) of \(y(2)\).

```latex
f(t, y) := y - t^2 + 1
h := 0.2
t_i := i h
w_0 := 0.5
k_1 := h f(t_i, w_i)
k_2 := h f\left(t_i + \frac{h}{2}, w_i + \frac{k_1}{2}\right)
k_3 := h f\left(t_i + \frac{h}{2}, w_i + \frac{k_2}{2}\right)
k_4 := h f(t_{i+1}, w_i + k_3)
w_{i+1} := w_i + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)
w_{10}
```

```python
from scipy.integrate import solve_ivp

f = lambda t, y: y - t**2 + 1
h = Rational(1, 5)
w = Rational(1, 2)
for i in range(10):
    t = i * h
    k1 = h * f(t, w)
    k2 = h * f(t + h / 2, w + k1 / 2)
    k3 = h * f(t + h / 2, w + k2 / 2)
    k4 = h * f(t + h, w + k3)
    w = w + (k1 + 2 * k2 + 2 * k3 + k4) / 6
expected = w
assert abs(float(expected) - 5.3053630) < 5e-8  # the book's table
exact = 9 - np.exp(2) / 2
assert abs(float(expected) - exact) < 1.1e-4
accurate = solve_ivp(f, (0, 2), [0.5], rtol=1e-12, atol=1e-12).y[0, -1]
assert abs(accurate - exact) < 1e-9
```

## Euler's method converges, at first order

**Level 4** · `num.ode.euler-convergence` · Burden and Faires (2011), section 5.2

Euler's method for \(y' = y\), \(y(0) = 1\), with \(N\) steps of \(h = 1/N\) gives
\(w_N = (1 + 1/N)^N\), which tends to \(y(1) = e\) with an error of order \(h\):
\(N(e - w_N) \to e/2\). The recurrence must be solved for a symbolic number of steps
before the limit is taken.

```latex
h := \frac{1}{N}
w_0 := 1
w_{i+1} := w_i + h w_i
\lim_{N \to \infty} N (e - w_N)
```

```python
N_ = symbols("N", positive=True, integer=True)
i = symbols("i", integer=True, nonnegative=True)
w = Function("w")
solution = rsolve(w(i + 1) - w(i) - w(i) / N_, w(i), {w(0): 1})
assert simplify(solution - (1 + 1 / N_) ** i) == 0
expected = limit(N_ * (E - solution.subs(i, N_)), N_, oo)
assert expected == E / 2
import mpmath

mpmath.mp.dps = 40
steps = mpmath.mpf(10) ** 6  # the next term of the error is -11e/(24N)
assert abs(steps * (mpmath.e - (1 + 1 / steps) ** steps) - mpmath.e / 2) < 2e-6
```
