# The Schwarzschild black hole

The vacuum outside a static, spherically symmetric mass: its metric, its curvature, and
the orbits around it.

## The time component of the metric

**Level 1** · `gr.schwarzschild.g-tt` · Carroll (2004), eq. (5.31)

A component of a metric given as a line element, by coordinate name.

```latex
ds^2 = -\left(1 - \frac{2GM}{r}\right) dt^2 + \left(1 - \frac{2GM}{r}\right)^{-1} dr^2 + r^2 d\Omega^2
g_{tt}
```

```python
G, M, r = symbols("G M r")
expected = -(1 - 2 * G * M / r)
```

## The Kretschmann scalar

**Level 2** · `gr.schwarzschild.kretschmann` · Carroll (2004), eq. (5.33)

The square of the Riemann tensor, contracted over every index pair.

```latex
g_{\mu\nu} := Schwarzschild(M)
R_{\alpha\beta\gamma\delta} R^{\alpha\beta\gamma\delta}
```

```python
M, r = symbols("M r")
expected = 48 * M**2 / r**6
```

## A vacuum solution

**Level 2** · `gr.schwarzschild.vacuum` · Carroll (2004), section 5.2

Einstein's equation in vacuum: the Ricci tensor vanishes.

```latex
g_{\mu\nu} := Schwarzschild(M)
R_{\mu\nu} = 0
```

```python
expected = True
```

## The radius of the photon sphere

**Level 3** · `gr.schwarzschild.photon-sphere` · Carroll (2004), eq. (5.69)

Light orbits in a circle where the effective potential of null geodesics is extremal.

```latex
V(r) := \frac{L^2}{2 r^2} \left(1 - \frac{2GM}{r}\right)
solve(\frac{dV}{dr} = 0, r)
```

```python
G, M = symbols("G M")
expected = [3 * G * M]
```

## A Killing vector of the metric

**Level 4** · `gr.schwarzschild.killing-time` · Carroll (2004), eq. (3.174)

The time translation \(\partial_t\) is a symmetry: its covariant derivative,
symmetrised, vanishes.

```latex
g_{\mu\nu} := Schwarzschild(M)
K^\mu := [1, 0, 0, 0]
\nabla_{(\mu} K_{\nu)} = 0
```

```python
expected = True
```
