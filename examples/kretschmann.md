# The Kretschmann scalar, from scratch

Built-in tensors are a convenience, not a requirement. Here every step of the
calculation of $K = R_{abcd} R^{abcd}$ is written out by hand, in the
plain-text notation, and checked against the built-in result.

```bash
relativisticpy examples/kretschmann.md --render
```

Schwarzschild spacetime, with $G$ and $c$ kept explicit:

$$
\begin{aligned}
Coordinates &:= [t, r, \theta, \phi] \\
g_{\mu\nu} &:= \mathrm{diag}\left(-(1 - \frac{2 G M}{c^2 r}),\; \frac{1}{1 - \frac{2 G M}{c^2 r}},\; r^2,\; r^2 \sin^2\theta\right)
\end{aligned}
$$

The Christoffel symbols, from their definition:

$$
\Gamma^{a}_{c f} := \frac{1}{2} g^{a b} \left(\partial_{c} g_{b f} + \partial_{f} g_{b c} - \partial_{b} g_{c f}\right)
$$

The Riemann tensor, from the Christoffel symbols:

$$
Riemann^{a}_{m b n} := \partial_{b} \Gamma^{a}_{n m} - \partial_{n} \Gamma^{a}_{b m} + \Gamma^{a}_{b l} \Gamma^{l}_{n m} - \Gamma^{a}_{n l} \Gamma^{l}_{b m}
$$

Its indices raised and lowered by hand, then contracted:

$$
\begin{aligned}
Upper^{a f h s} &:= g^{s q} g^{h c} g^{f b} \, Riemann^{a}_{b c q} \\
Lower_{a f h s} &:= g_{a n} \, Riemann^{n}_{f h s} \\
K &:= \mathrm{simplify}\left(Upper^{a f h s} Lower_{a f h s}\right) \\
K
\end{aligned}
$$

The built-in Kretschmann scalar agrees:

$$
K - R_{abcd} R^{abcd}
$$
