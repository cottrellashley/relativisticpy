# The Friedmann equations

A homogeneous, isotropic universe has the FLRW metric, with spatial curvature
$k$ and a scale factor $a(t)$ to be found:

```bash
relativisticpy examples/friedmann.md --render
```

$$
\begin{aligned}
Coordinates &:= [t, r, \theta, \phi] \\
ds^2 &= -dt^2 + a(t)^2 \left(\frac{dr^2}{1 - k r^2} + r^2 d\Omega^2\right)
\end{aligned}
$$

Fill it with a perfect fluid at rest, of density $\rho(t)$ and pressure $p(t)$:

$$
\begin{aligned}
u^{\mu} &:= [1, 0, 0, 0] \\
T^{\mu\nu} &:= (\rho(t) + p(t))\, u^{\mu} u^{\nu} + p(t)\, g^{\mu\nu}
\end{aligned}
$$

Einstein's equations $G_{\mu\nu} = 8 \pi G\, T_{\mu\nu}$ then hold component by
component. The time component is the first Friedmann equation,
$\left(\dot a / a\right)^2 = \frac{8\pi G}{3}\rho - k/a^2$:

$$
\begin{aligned}
F_1 &:= G_{tt} = 8 \pi G\, T_{tt} \\
\mathrm{solve}(F_1, \rho(t))
\end{aligned}
$$

and a radial component gives the second, for the acceleration $\ddot a$:

$$
\begin{aligned}
F_2 &:= G^{r}{}_{r} = 8 \pi G\, T^{r}{}_{r} \\
\mathrm{solve}(F_2, \ddot{a(t)})
\end{aligned}
$$

Energy conservation, $\nabla_\mu T^{\mu\nu} = 0$, is the continuity equation
$\dot\rho + 3 \frac{\dot a}{a}(\rho + p) = 0$:

$$
\nabla_{\mu} T^{\mu t}
$$
