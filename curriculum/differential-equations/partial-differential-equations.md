# Partial differential equations

The heat equation by separation of variables, d'Alembert's solution of the wave equation,
Laplace's equation in polar coordinates, and the fundamental solution of the heat
equation.

## The heat equation in a rod

**Level 3** · `de.pde.heat` · Boyce and DiPrima (2012), section 10.5

A rod 50 cm long, initially at 20 °C throughout, with its ends held at 0 °C. Separation of
variables gives the temperature as the modes \(e^{-n^2\pi^2\alpha^2 t/L^2}
\sin(n\pi x/L)\), weighted by the coefficients \(c_n\) of the sine series of the initial
temperature: \(80/n\pi\) for odd \(n\), and zero for even.

```latex
L := 50
f(x) := 20
c_n := \frac{2}{L} \int_0^L f(x) \sin\frac{n \pi x}{L} \, dx
c_1
c_2
c_3
```

```python
x = symbols("x")
c = lambda n: Rational(2, 50) * integrate(20 * sin(n * pi * x / 50), (x, 0, 50))
outputs = [c(1), c(2), c(3)]
assert outputs == [80 / pi, 0, 80 / (3 * pi)]
```

## D'Alembert's solution

**Level 3** · `de.pde.dalembert` · Arfken, Weber and Harris (2013), section 9.6

The solution of the wave equation for any initial displacement \(f\) and velocity \(g\),
both functions left unknown: it satisfies the equation and both initial conditions.

```latex
u(x, t) := \frac{1}{2}\left[f(x - ct) + f(x + ct)\right] + \frac{1}{2c} \int_{x - ct}^{x + ct} g(s) \, ds
\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}
u(x, 0) = f(x)
\left. \frac{\partial u}{\partial t} \right|_{t = 0} = g(x)
```

```python
x, t, c, s = symbols("x t c s")
f, g = Function("f"), Function("g")
u = (f(x - c * t) + f(x + c * t)) / 2 + Integral(g(s), (s, x - c * t, x + c * t)) / (2 * c)
assert simplify(diff(u, t, 2) - c**2 * diff(u, x, 2)) == 0
assert simplify(u.subs(t, 0).doit() - f(x)) == 0
assert simplify(diff(u, t).subs(t, 0).doit() - g(x)) == 0
outputs = [True, True, True]
```

## Laplace's equation in a disk

**Level 2** · `de.pde.laplace-disk` · Boyce and DiPrima (2012), section 10.8

The functions \(r^n \cos n\theta\) that separation of variables gives in a disk satisfy
Laplace's equation, written in polar coordinates.

```latex
u(r, \theta) := r^n \cos n\theta
\frac{\partial^2 u}{\partial r^2} + \frac{1}{r} \frac{\partial u}{\partial r} + \frac{1}{r^2} \frac{\partial^2 u}{\partial \theta^2} = 0
```

```python
r, theta, n = symbols("r theta n")
u = r**n * cos(n * theta)
assert simplify(diff(u, r, 2) + diff(u, r) / r + diff(u, theta, 2) / r**2) == 0
expected = True
```

## The fundamental solution of the heat equation

**Level 5** · `de.pde.heat-kernel` · Strauss (2008), section 2.4

The heat kernel solves the heat equation and starts as a point of heat: as \(t \to 0^+\)
it tends to the Dirac delta, a limit in the sense of distributions, not of numbers.

```latex
K(x, t) := \frac{1}{\sqrt{4\pi t}} e^{-\frac{x^2}{4t}}
\frac{\partial K}{\partial t} = \frac{\partial^2 K}{\partial x^2}
\lim_{t \to 0^+} K(x, t) = \delta(x)
```

```python
x = symbols("x", real=True)
t = symbols("t", positive=True)
K = exp(-x**2 / (4 * t)) / sqrt(4 * pi * t)
assert simplify(diff(K, t) - diff(K, x, 2)) == 0
assert simplify(integrate(K, (x, -oo, oo))) == 1  # its mass is one at every time
assert limit(K.subs(x, 1), t, 0, "+") == 0  # it vanishes away from the origin
smoothed = integrate(K * exp(-(x**2)), (x, -oo, oo))  # and samples a test function at 0
assert limit(simplify(smoothed), t, 0, "+") == 1
outputs = [True, True]
```
