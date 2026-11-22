# Gravitational collapse

How a cloud of gas collapses to form stars: the free-fall time of a uniform sphere released
from rest, and the Jeans mass, above which a cloud's thermal energy cannot hold it up
against its own gravity.

## The free-fall time

**Level 3** · `astro.collapse.free-fall-time` · Carroll and Ostlie (2007), chapter 12

A uniform sphere released from rest at radius \(r_0\) falls inward with the speed that
energy conservation gives, \(v = \sqrt{2GM(1/r - 1/r_0)}\), and reaches the centre after
\(\int_0^{r_0} dr/v\), a time that depends only on its density:
\(t_{\mathrm{ff}} = \sqrt{3\pi/32G\rho}\). The quantities are declared positive, as a list.

```latex
G, r_0, \rho > 0
M := \frac{4}{3}\pi r_0^3 \rho
v(r) := \sqrt{2 G M \left(\frac{1}{r} - \frac{1}{r_0}\right)}
t_{\mathrm{ff}} := \int_0^{r_0} \frac{dr}{v(r)}
t_{\mathrm{ff}}
```

```python
G, r_0, rho, r, u = symbols("G r_0 rho r u", positive=True)
M = Rational(4, 3) * pi * r_0**3 * rho
v = sqrt(2 * G * M * (1 / r - 1 / r_0))
expected = sqrt(3 * pi / (32 * G * rho))
assert simplify(integrate(1 / v, (r, 0, r_0)) - expected) == 0
# With r = r_0 sin^2 u, dr / v = 2 r_0^(3/2) sin^2 u du / sqrt(2 G M), for u from 0 to pi/2
substituted = 2 * r_0 ** Rational(3, 2) * sin(u) ** 2 / sqrt(2 * G * M)
assert simplify(integrate(substituted, (u, 0, pi / 2)) - expected) == 0
```

## The Jeans mass

**Level 3** · `astro.collapse.jeans-mass` · Carroll and Ostlie (2007), chapter 12

A uniform cloud of mass \(M\), temperature \(T\) and density \(\rho_0\) collapses when its
gravitational energy outweighs twice its thermal energy. The mass at which \(2K + U = 0\)
is the Jeans mass, \(M_J = (5kT/G\mu m_H)^{3/2}(3/4\pi\rho_0)^{1/2}\); every quantity in it
is positive, which makes it the one root.

```latex
M, T, \rho_0, k, G, \mu, m_H > 0
R := \left(\frac{3 M}{4 \pi \rho_0}\right)^{1/3}
K := \frac{3}{2} \frac{M}{\mu m_H} k T
U := -\frac{3}{5} \frac{G M^2}{R}
solve(2K + U = 0, M)
```

```python
M, T, rho_0, k, G, mu, m_H = symbols("M T rho_0 k G mu m_H", positive=True)
R = (3 * M / (4 * pi * rho_0)) ** Rational(1, 3)
K = Rational(3, 2) * M / (mu * m_H) * k * T  # (3/2) N k T, with N = M / mu m_H particles
U = -Rational(3, 5) * G * M**2 / R
jeans = (5 * k * T / (G * mu * m_H)) ** Rational(3, 2) * (3 / (4 * pi * rho_0)) ** Rational(1, 2)
roots = solve(Eq(2 * K + U, 0), M)
assert len(roots) == 1 and simplify(roots[0] - jeans) == 0
assert simplify((2 * K + U).subs(M, jeans)) == 0  # it balances the two energies
expected = [jeans]
```
