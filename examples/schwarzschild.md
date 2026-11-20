# The Schwarzschild solution

Every equation on this page is both LaTeX, which GitHub typesets, and
RelativisticPy, which computes. To see the results typeset under each step:

```bash
relativisticpy examples/schwarzschild.md --render
```

## The ansatz

Outside a static, spherically symmetric star, spacetime can be written with two
unknown functions of the radius, $A(r)$ and $B(r)$:

$$
\begin{aligned}
\mathrm{Coordinates} &:= [t, r, \theta, \phi] \\
g_{\mu\nu} &:= \mathrm{diag}\left(-A(r),\; B(r),\; r^2,\; r^2 \sin^2\theta\right)
\end{aligned}
$$

## The field equations

In vacuum the Ricci tensor vanishes. Three of its components carry the
information we need:

$$
\begin{aligned}
E_0 &:= R_{tt} \\
E_1 &:= R_{rr} \\
E_2 &:= R_{\theta\theta}
\end{aligned}
$$

This combination of the first two involves only first derivatives, and fixes
$B$ in terms of $A$:

$$
\begin{aligned}
B_1 &:= \mathrm{RHS}\left(\mathrm{dsolve}\left((E_0 B(r) + E_1 A(r))\, r\, B(r),\; B(r)\right)\right) \\
B_1
\end{aligned}
$$

Substituting into the third leaves an equation for $A$ alone:

$$
\begin{aligned}
A_1 &:= \mathrm{RHS}\left(\mathrm{dsolve}\left(\mathrm{simplify}\left(E_2 \big|_{B(r) = B_1}\right),\; A(r)\right)\right) \\
A_1
\end{aligned}
$$

## The integration constants

Far from the star, the radial acceleration of a particle at rest must be
Newton's, $-GM/r^2$. With $A = 1 + C/r$ and $B = 1/A$:

$$
\begin{aligned}
g_{\mu\nu} &:= \mathrm{diag}\left(-(1 + \tfrac{C}{r}),\; \frac{1}{1 + \tfrac{C}{r}},\; r^2,\; r^2 \sin^2\theta\right) \\
\mathrm{solve}\left(\lim_{r \to \infty} r^2\, \Gamma^{r}_{tt} - G M,\; C\right)
\end{aligned}
$$

So $C = -2GM$, and spacetime around any spherical mass is the Schwarzschild
metric. Its curvature invariant shows that $r = 2GM$ is no singularity at all,
while $r = 0$ is:

$$
\begin{aligned}
g_{\mu\nu} &:= \mathrm{diag}\left(-(1 - \tfrac{2 G M}{r}),\; \frac{1}{1 - \tfrac{2 G M}{r}},\; r^2,\; r^2 \sin^2\theta\right) \\
R_{abcd} R^{abcd}
\end{aligned}
$$
