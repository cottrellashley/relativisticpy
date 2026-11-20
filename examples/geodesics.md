# Geodesics from a Lagrangian

A free particle in curved spacetime extremises
$L = g_{\mu\nu} \dot x^\mu \dot x^\nu$. Its Euler-Lagrange equations are the
geodesic equations.

```bash
relativisticpy examples/geodesics.md --render
```

$$
\begin{aligned}
g_{\mu\nu} &:= \mathrm{Schwarzschild}(M) \\
x^{\mu} &:= [t(\tau), r(\tau), \theta(\tau), \phi(\tau)]
\end{aligned}
$$

The Lagrangian, along the path:

$$
L := g_{\mu\nu} \dot{x}^\mu \dot{x}^\nu \Big|_{t = t(\tau),\; r = r(\tau),\; \theta = \theta(\tau),\; \phi = \phi(\tau)}
$$

The Euler-Lagrange equations, one per coordinate:

$$
EL_{\mu} := \frac{d}{d\tau} \frac{\partial L}{\partial \dot{x}^\mu} - \frac{\partial L}{\partial x^\mu}
$$

The $\theta$ equation, for example:

$$
EL_{\theta}
$$

Each is twice the geodesic equation
$\ddot x^\nu + \Gamma^\nu{}_{\alpha\beta}\, \dot x^\alpha \dot x^\beta = 0$
with its index lowered, which RelativisticPy confirms component by component:

$$
EL_{\mu} = \left. 2 g_{\mu\nu} \left( \ddot{x}^{\nu} + \Gamma^{\nu}_{\alpha\beta}\, \dot{x}^{\alpha} \dot{x}^{\beta} \right) \right|_{t = t(\tau),\; r = r(\tau),\; \theta = \theta(\tau),\; \phi = \phi(\tau)}
$$
