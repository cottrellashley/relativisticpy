# Numerics

The symbolic results of RelativisticPy become numbers through
`relativisticpy.numeric`, which compiles expressions and tensors into NumPy
functions and integrates geodesics with SciPy. It needs the optional
dependencies:

```bash
pip install "relativisticpy[numeric]"
```

## NumPy functions

`lambdify` compiles a scalar or a tensor into a vectorised NumPy function of
the variables you name; every other symbol is given a value:

```python
>>> import numpy as np
>>> from relativisticpy import spacetimes
>>> from relativisticpy.numeric import evaluate, lambdify
>>> g = spacetimes.schwarzschild()
>>> kretschmann = lambdify(g.kretschmann, ["r"], M=1)
>>> kretschmann(np.array([2.0, 3.0, 6.0])).round(5)
array([0.75   , 0.06584, 0.00103])

```

A tensor compiles to a function returning an array of its components'
shape, with the variables' shape appended when they are arrays:

```python
>>> metric = lambdify(g, g.coordinates, M=1)
>>> metric(0.0, np.linspace(3, 10, 50), np.pi / 2, 0.0).shape
(4, 4, 50)

```

`evaluate` gives the value at one point, and names any value you forgot:

```python
>>> float(evaluate(g.christoffel, r=10, theta=np.pi / 2, M=1)[1, 0, 0])
0.008
>>> evaluate(g.christoffel, r=10, M=1)
Traceback (most recent call last):
  ...
ValueError: give a value for theta

```

## Geodesics

`geodesic` integrates the geodesic equation
\(\ddot x^a = -\Gamma^a{}_{bc}\,\dot x^b \dot x^c\) with
`scipy.integrate.solve_ivp`, from an initial position and four-velocity.
`four_velocity` completes a four-velocity from its spatial components, so
that \(g_{\mu\nu} u^\mu u^\nu = -1\) for a particle (or \(0\) for light).

A circular orbit stays circular:

```python
>>> from relativisticpy.numeric import four_velocity, geodesic
>>> start = [0, 10, np.pi / 2, 0]                       # t, r, θ, φ
>>> u = four_velocity(g, start, [0, 0, (1 / 700) ** 0.5], M=1)
>>> orbit = geodesic(g, start, u, (0, 1000), M=1)
>>> print(orbit["r"].min().round(6), orbit["r"].max().round(6))
10.0 10.0

```

An eccentric orbit precesses. Far from the black hole the advance of the
periapsis per orbit approaches the famous \(6\pi M / a(1 - e^2)\), which
accounts for Mercury's anomalous precession:

```python
>>> start = [0, 500, np.pi / 2, 0]                      # start at apoapsis
>>> circular = (1 / (500**2 * 497)) ** 0.5
>>> u = four_velocity(g, start, [0, 0, 0.9 * circular], M=1)
>>> orbit = geodesic(g, start, u, (0, 120000), M=1, samples=400001)
>>> r, phi = orbit["r"], orbit["phi"]
>>> apoapsis = np.where((r[1:-1] > r[:-2]) & (r[1:-1] > r[2:]))[0][0] + 1
>>> advance = phi[apoapsis] - 2 * np.pi
>>> a, e = (r.max() + r.min()) / 2, (r.max() - r.min()) / (r.max() + r.min())
>>> print(f"{advance:.3f} rad per orbit, {advance / (6 * np.pi / (a * (1 - e**2))):.2f} of the formula")
0.047 rad per orbit, 1.01 of the formula

```

The remaining percent is the higher-order correction the formula leaves out.

The four-velocity's norm is conserved along the orbit, a check on the
integration:

```python
>>> norms = [orbit.u[:, i] @ metric(*orbit.x[:, i]) @ orbit.u[:, i] for i in (0, 200000, 400000)]
>>> np.allclose(norms, -1)
True

```

`geodesic` returns a `Geodesic` with `tau`, the coordinates `x` and velocities
`u` (one row per coordinate), access by coordinate name (`orbit["r"]`), and
SciPy's full `solution`. `method`, `rtol` and `atol` go to `solve_ivp`; every
other keyword argument is a parameter of the metric, such as `M=1`.

## Light

Light follows null geodesics, with `four_velocity(..., kind="null")`. A ray
aimed closer than \(b = 3\sqrt{3}\,M\) to a black hole falls in. `stop` ends
an integration where a function of the position and velocity crosses zero, here
once the ray is inside the photon sphere at \(r = 3M\), and `stopped` says
whether that happened:

```python
>>> def ray(b, r0=1000.0):
...     start = [0, r0, np.pi / 2, 0]
...     inward = -(1 - (1 - 2 / r0) * b**2 / r0**2) ** 0.5
...     u = four_velocity(g, start, [inward, 0, b / r0**2], kind="null", M=1)
...     return geodesic(g, start, u, (0, 3000), M=1, stop=lambda x, u: x[1] - 2.5)
>>> ray(5.1).stopped, ray(5.3).stopped
(True, False)

```

[`examples/light.py`](https://github.com/cottrellashley/relativisticpy/blob/main/examples/light.py)
finds the critical impact parameter this way, and measures the bending of
starlight.

## Results from a workbook

Anything a workbook computes can go the same way: `wb.metric` is the metric in
force as a `Metric`, and tensor results are `Tensor` values.

```python
>>> from relativisticpy import Workbook
>>> wb = Workbook()
>>> wb.expr(r"g_{\mu\nu} := ReissnerNordstrom(M, Q)")
>>> horizon = wb.expr(r"solve(g_{tt} = 0, r)")
>>> horizon
[M - sqrt((M - Q)*(M + Q)), M + sqrt((M - Q)*(M + Q))]
>>> [round(evaluate(root, M=1, Q=0.5), 4) for root in horizon]
[0.134, 1.866]

```
