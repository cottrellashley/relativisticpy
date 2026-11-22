# Distances in cosmology

The distances of an expanding universe, in the notation of Hogg (1999), "Distance measures
in cosmology", arXiv:astro-ph/9905116: the Hubble distance \(D_H = c/H_0\), the function
\(E(z) = H(z)/H_0\), the line-of-sight comoving distance
\(D_C = D_H\int_0^z dz'/E(z')\), the transverse comoving distance \(D_M\), and the
angular-diameter and luminosity distances \(D_A = D_M/(1 + z)\) and
\(D_L = (1 + z)D_M\).

## The comoving distance in an Einstein–de Sitter universe

**Level 2** · `astro.distances.comoving-distance` · Hogg (1999), arXiv:astro-ph/9905116, "Comoving distance (line-of-sight)"

The line-of-sight comoving distance for a flat universe of matter alone, where
\(E(z) = (1 + z)^{3/2}\): \(\frac{2c}{H_0}\left(1 - 1/\sqrt{1 + z}\right)\). The variable of
integration is primed, as Hogg writes it.

```latex
z \geq 0
E(z) := (1 + z)^{3/2}
D_C := \frac{c}{H_0} \int_0^z \frac{dz'}{E(z')}
D_C
```

```python
from scipy.integrate import quad

z, c, H_0, s = symbols("z c H_0 s", positive=True)
expected = 2 * c / H_0 * (1 - 1 / sqrt(1 + z))
assert simplify(c / H_0 * integrate((1 + s) ** Rational(-3, 2), (s, 0, z)) - expected) == 0
# Numerically, at z = 3 the distance is one Hubble distance; the horizon, at z -> oo, is two
assert abs(quad(lambda x: (1 + x) ** -1.5, 0, 3)[0] - 1) < 1e-12
assert limit(expected, z, oo) == 2 * c / H_0
```

## The largest angular-diameter distance

**Level 3** · `astro.distances.angular-diameter-maximum` · Hogg (1999), arXiv:astro-ph/9905116, "Angular diameter distance"

In a flat universe the angular-diameter distance is \(D_A = D_C/(1 + z)\). In the
Einstein–de Sitter universe it rises and then falls, so that an object of a given size
looks smallest at \(z = 5/4\).

```latex
z \geq 0
D_C(z) := \frac{2c}{H_0}\left(1 - \frac{1}{\sqrt{1 + z}}\right)
D_A(z) := \frac{D_C(z)}{1 + z}
solve\left(\frac{d}{dz} D_A(z) = 0, z\right)
```

```python
from scipy.optimize import minimize_scalar

z, c, H_0 = symbols("z c H_0", positive=True)
D_A = 2 * c / H_0 * (1 - 1 / sqrt(1 + z)) / (1 + z)
expected = solve(diff(D_A, z), z)
assert expected == [Rational(5, 4)]
# Numerically: the maximum of D_A, with c / H_0 = 1
best = minimize_scalar(
    lambda x: -2 * (1 - (1 + x) ** -0.5) / (1 + x), bounds=(0, 10), method="bounded"
)
assert abs(best.x - 1.25) < 1e-4
```

## The luminosity distance in an empty universe

**Level 3** · `astro.distances.milne-luminosity-distance` · Hogg (1999), arXiv:astro-ph/9905116, "Comoving distance (transverse)" and "Luminosity distance"

Hogg's transverse comoving distance \(D_M\) takes one of three forms by the sign of the
curvature parameter \(\Omega_k\), written as cases. In an empty universe, \(\Omega_k = 1\)
and \(E(z) = 1 + z\), and the luminosity distance \(D_L = (1 + z)D_M\) is
\(\frac{c}{H_0}\left(z + \frac{z^2}{2}\right)\).

```latex
z \geq 0
D_H := \frac{c}{H_0}
\Omega_k := 1
E(z) := 1 + z
D_C := D_H \int_0^z \frac{dz'}{E(z')}
D_M := \begin{cases} \frac{D_H}{\sqrt{\Omega_k}} \sinh\left(\sqrt{\Omega_k} \frac{D_C}{D_H}\right) & \Omega_k > 0 \\ D_C & \Omega_k = 0 \\ \frac{D_H}{\sqrt{|\Omega_k|}} \sin\left(\sqrt{|\Omega_k|} \frac{D_C}{D_H}\right) & \Omega_k < 0 \end{cases}
D_L := (1 + z) D_M
D_L
```

```python
z, c, H_0, s = symbols("z c H_0 s", positive=True)
D_H = c / H_0
D_C = D_H * integrate(1 / (1 + s), (s, 0, z))  # E(z) = 1 + z
assert simplify(D_C - D_H * log(1 + z)) == 0
D_M = D_H * sinh(D_C / D_H)  # the branch for Omega_k = 1 > 0
expected = D_H * (z + z**2 / 2)
assert simplify(((1 + z) * D_M - expected).rewrite(exp)) == 0
# Numerically at z = 2: (1 + z) sinh(ln(1 + z)) = 3 sinh(ln 3) = 4 = z + z^2/2
assert abs(3 * np.sinh(np.log(3)) - 4) < 1e-12
```
