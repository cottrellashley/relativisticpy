# Balls and spheres in n dimensions

The volume \(\pi^{n/2} / \Gamma(n/2 + 1)\) of the unit ball and the area
\(2\pi^{n/2} / \Gamma(n/2)\) of the unit sphere in \(\mathbb{R}^n\), the second from the
Gaussian integral, and the dimension in which the unit ball is largest. DLMF is the NIST
*Digital Library of Mathematical Functions*. The volume of the three-sphere of radius \(a\)
is `dg.forms.three-sphere-volume`.

## The volume of the five-dimensional ball

**Level 1** · `geom.balls.five-ball` · DLMF §5.19(iii)

\(V_n = \pi^{n/2} / \Gamma(n/2 + 1)\) at \(n = 5\), with \(\Gamma\) at a half-integer:
\(8\pi^2/15\).

```latex
\left. \frac{\pi^{n/2}}{\Gamma\left(\frac{n}{2} + 1\right)} \right|_{n = 5}
```

```python
expected = pi ** Rational(5, 2) / gamma(Rational(7, 2))
assert simplify(expected - 8 * pi**2 / 15) == 0
# In hyperspherical coordinates: the radial integral of r^4, times the area of the unit
# four-sphere, sin^3(a) sin^2(b) sin(c) integrated over its angles.
r, a, b, c, d = symbols("r a b c d")
area = integrate(
    sin(a) ** 3 * sin(b) ** 2 * sin(c), (a, 0, pi), (b, 0, pi), (c, 0, pi), (d, 0, 2 * pi)
)
assert integrate(r**4, (r, 0, 1)) * area == 8 * pi**2 / 15
```

## The largest unit ball

**Level 2** · `geom.balls.largest` · standard

The volume of the unit ball grows with the dimension up to \(n = 5\) and falls towards 0
after: the dimension that maximises it, written \(\arg\max\) over the natural numbers.

```latex
\arg\max_{n \in \mathbb{N}} \frac{\pi^{n/2}}{\Gamma\left(\frac{n}{2} + 1\right)}
```

```python
def V(n):
    return pi ** Rational(n, 2) / gamma(Rational(n, 2) + 1)


volumes = [V(n) for n in range(1, 41)]
expected = 1 + max(range(len(volumes)), key=lambda k: volumes[k])
assert expected == 5
assert V(4) < V(5) and V(6) < V(5)
# The ratio V(n + 1)/V(n) = sqrt(pi) Gamma(n/2 + 1)/Gamma(n/2 + 3/2) falls as n grows, and
# passes 1 between n = 4 and n = 5: the volumes rise to n = 5 and fall from then on.
ratios = [float(V(n + 1) / V(n)) for n in range(1, 400)]
assert all(p > q for p, q in zip(ratios, ratios[1:]))
assert ratios[3] > 1 > ratios[4]  # V(5)/V(4) > 1 > V(6)/V(5)
```

## The area of the unit sphere from the Gaussian integral

**Level 3** · `geom.balls.sphere-area` · standard

\(\left(\int e^{-x^2} dx\right)^n = \pi^{n/2}\) is the integral of \(e^{-|x|^2}\) over
\(\mathbb{R}^n\). In polar coordinates it is the area \(A_n\) of the unit sphere times a
radial integral, \(\Gamma(n/2)/2\) for \(n > 0\), so \(A_n = 2\pi^{n/2} / \Gamma(n/2)\).

```latex
n > 0
solve\left(\left(\int_{-\infty}^{\infty} e^{-x^2} \, dx\right)^n = A_n \int_0^\infty e^{-r^2} r^{n-1} \, dr, A_n\right)
```

```python
n, x, r = symbols("n x r", positive=True)
A = symbols("A_n")
gaussian = integrate(exp(-(x**2)), (x, -oo, oo)) ** n
radial = integrate(exp(-(r**2)) * r ** (n - 1), (r, 0, oo))
assert radial == gamma(n / 2) / 2
expected = solve(Eq(gaussian, A * radial), A)
area = 2 * pi ** (n / 2) / gamma(n / 2)
assert len(expected) == 1 and simplify(expected[0] - area) == 0
# The circle, the sphere and the three-sphere: 2 pi, 4 pi and 2 pi^2.
assert [area.subs(n, k) for k in (2, 3, 4)] == [2 * pi, 4 * pi, 2 * pi**2]
# And A_n = n V_n, with V_n = pi^(n/2) / Gamma(n/2 + 1).
assert simplify(gammasimp(area - n * pi ** (n / 2) / gamma(n / 2 + 1))) == 0
```
