# Interacting fields

Perturbation theory in quantum field theory, from the Gaussian integral under every path
integral to Wick's theorem, Feynman rules read off a Lagrangian, a cross-section, a loop
integral in \(d\) dimensions and the running of a coupling. Most of it is the notation
of specialist texts and beyond any design of the language yet: functional integrals and
derivatives, time-ordered products, amplitudes and spin sums.

## A Gaussian integral with a source

**Level 1** · `qft.interactions.gaussian` · Zee (2010), chapter I.2

The integral under every path integral, in one variable, with a source \(J\):
\(\int dq\, e^{-\frac{1}{2}aq^2 + Jq} = \sqrt{2\pi/a}\, e^{J^2/2a}\), the differential
written first.

```latex
\int_{-\infty}^{\infty} dq \, e^{-\frac{1}{2} a q^2 + J q}
```

```python
q, J = symbols("q J", real=True)
a = symbols("a", positive=True)
computed = integrate(exp(-a * q**2 / 2 + J * q), (q, -oo, oo))
stated = sqrt(2 * pi / a) * exp(J**2 / (2 * a))  # Zee, chapter I.2
assert simplify(computed - stated) == 0
expected = stated
```

## A cross-section from its angular distribution

**Level 1** · `qft.interactions.solid-angle` · Peskin and Schroeder (1995), eq. (5.13)

The angular distribution of \(e^+e^- \to \mu^+\mu^-\) at high energy,
\(\frac{d\sigma}{d\Omega} = \frac{\alpha^2}{4s}(1 + \cos^2\theta)\), integrated over all
directions, with \(\int d\Omega\) as physicists write it: \(\sigma = 4\pi\alpha^2/3s\).

```latex
\int d\Omega \, \frac{\alpha^2}{4s} \left(1 + \cos^2\theta\right)
```

```python
theta, phi = symbols("theta phi", real=True)
alpha, s = symbols("alpha s", positive=True)
# d Omega = sin(theta) d theta d phi, over the whole sphere.
expected = integrate(
    alpha**2 / (4 * s) * (1 + cos(theta) ** 2) * sin(theta), (theta, 0, pi), (phi, 0, 2 * pi)
)
assert simplify(expected - 4 * pi * alpha**2 / (3 * s)) == 0  # Peskin (5.13)
```

## Wick's theorem for four fields

**Level 5** · `qft.interactions.wick` · Peskin and Schroeder (1995), eq. (4.40)

The vacuum expectation value of a time-ordered product of four free fields is the sum
over its three complete contractions, each a product of Feynman propagators.

```latex
\langle 0 | T\{\phi(x_1)\phi(x_2)\phi(x_3)\phi(x_4)\} | 0 \rangle = D_F(x_1 - x_2) D_F(x_3 - x_4) + D_F(x_1 - x_3) D_F(x_2 - x_4) + D_F(x_1 - x_4) D_F(x_2 - x_3)
```

```python
# The same combinatorics in a Gaussian integral over four variables with covariance D:
# four derivatives of exp(J.D.J/2) at J = 0 pair the variables in every way.
J = symbols("J1:5")
D = Matrix(4, 4, lambda i, j: Symbol(f"D{min(i, j) + 1}{max(i, j) + 1}"))
generating = exp((Matrix(J).T * D * Matrix(J))[0] / 2)
moment = diff(generating, *J).subs({j: 0 for j in J})
contractions = D[0, 1] * D[2, 3] + D[0, 2] * D[1, 3] + D[0, 3] * D[1, 2]
assert simplify(moment - contractions) == 0
assert factorial2(4 - 1) == 3  # (2n - 1)!! complete contractions of 2n fields
expected = True
```

## The propagator from the generating functional

**Level 5** · `qft.interactions.generating-functional` · Peskin and Schroeder (1995), section 9.2

The generating functional of a free field, an exponential of a double integral over
spacetime, and two functional derivatives of it: the two-point function is the Feynman
propagator.

```latex
Z[J] := Z_0 \exp\left[-\frac{1}{2} \int d^4x \, d^4y \, J(x) D_F(x - y) J(y)\right]
\left. \frac{1}{Z_0} \left(-i\frac{\delta}{\delta J(x_1)}\right) \left(-i\frac{\delta}{\delta J(x_2)}\right) Z[J] \right|_{J=0} = D_F(x_1 - x_2)
```

```python
# Spacetime made of two points: the double integral is the quadratic form J.D.J, and
# functional derivatives are partial derivatives with respect to J1 and J2.
J_1, J_2, D_11, D_12, D_22, Z_0 = symbols("J_1 J_2 D_11 D_12 D_22 Z_0")
J = Matrix([J_1, J_2])
D = Matrix([[D_11, D_12], [D_12, D_22]])  # D_F(x - y) is symmetric
Z = Z_0 * exp(-(J.T * D * J)[0] / 2)
two_point = (1 / Z_0 * (-I) ** 2 * diff(Z, J_1, J_2)).subs({J_1: 0, J_2: 0})
assert two_point == D_12
expected = True
```

## The vertex of φ⁴ theory

**Level 5** · `qft.interactions.phi4-vertex` · Peskin and Schroeder (1995), section 4.6

A Feynman rule read off a Lagrangian: at lowest order, the amplitude for two scalars to
scatter in \(\phi^4\) theory is the vertex, \(i\mathcal{M} = -i\lambda\), the \(4!\) ways
of attaching four lines cancelling the \(1/4!\) in the coupling.

```latex
\mathcal{L} := \frac{1}{2}(\partial_\mu\phi)^2 - \frac{1}{2}m^2\phi^2 - \frac{\lambda}{4!}\phi^4
i\mathcal{M}(\phi\phi \to \phi\phi)
```

```python
phi, lam = symbols("phi lambda")
L_interaction = -lam / factorial(4) * phi**4
# The vertex is i times the interaction Lagrangian differentiated once for each line.
expected = I * diff(L_interaction, phi, 4)
assert expected == -I * lam  # Peskin, section 4.6
```

## The cross-section of e⁺e⁻ → μ⁺μ⁻

**Level 5** · `qft.interactions.muon-pair` · Peskin and Schroeder (1995), section 5.1

From the Feynman rules of QED to a total cross-section: the amplitude of one photon's
exchange, its square averaged over the initial spins and summed over the final ones,
and the integral over directions, at energies far above both masses:
\(\sigma = 4\pi\alpha^2/3s = e^4/12\pi s\). Here \(e\) is the charge.

```latex
i\mathcal{M} = \bar{v}(p') (-ie\gamma^\mu) u(p) \left(\frac{-ig_{\mu\nu}}{q^2}\right) \bar{u}(k) (-ie\gamma^\nu) v(k')
\sigma := \int d\Omega \, \frac{1}{64\pi^2 s} \cdot \frac{1}{4} \sum_{\text{spins}} |\mathcal{M}|^2
\sigma
```

```python
E, theta, phi, e, s = symbols("E theta phi e s", positive=True)
sigma_pauli = [Matrix([[0, 1], [1, 0]]), Matrix([[0, -I], [I, 0]]), Matrix([[1, 0], [0, -1]])]
one, zero = eye(2), zeros(2, 2)
gamma_up = [Matrix(BlockMatrix([[zero, one], [one, zero]]))]
gamma_up += [Matrix(BlockMatrix([[zero, s_i], [-s_i, zero]])) for s_i in sigma_pauli]
g = diag(1, -1, -1, -1)
gamma_down = [g[mu, mu] * gamma_up[mu] for mu in range(4)]


def slash(vector):  # gamma^mu p_mu for p given with an upper index
    lower = g * Matrix(vector)
    return sum((gamma_up[mu] * lower[mu] for mu in range(4)), zeros(4, 4))


# Massless kinematics in the centre-of-mass frame: e-(p) e+(p') -> mu-(k) mu+(k').
p, p_bar = [E, 0, 0, E], [E, 0, 0, -E]
k, k_bar = [E, E * sin(theta), 0, E * cos(theta)], [E, -E * sin(theta), 0, -E * cos(theta)]
q_squared = (2 * E) ** 2
# Spin sums turn |M|^2 into traces: (e^4 / 4 q^4) tr[p'-slash g^mu p-slash g^nu] tr[k-slash g_mu k'-slash g_nu].
electron = [
    [(slash(p_bar) * gamma_up[mu] * slash(p) * gamma_up[nu]).trace() for nu in range(4)]
    for mu in range(4)
]
muon = [
    [(slash(k) * gamma_down[mu] * slash(k_bar) * gamma_down[nu]).trace() for nu in range(4)]
    for mu in range(4)
]
averaged = (
    e**4
    / (4 * q_squared**2)
    * sum(electron[mu][nu] * muon[mu][nu] for mu in range(4) for nu in range(4))
)
assert simplify(averaged - e**4 * (1 + cos(theta) ** 2)) == 0  # Peskin, section 5.1
sigma = integrate(
    averaged / (64 * pi**2 * q_squared) * sin(theta), (theta, 0, pi), (phi, 0, 2 * pi)
)
expected = e**4 / (12 * pi * s)
assert simplify(sigma.subs(E, sqrt(s) / 2) - expected) == 0
alpha = e**2 / (4 * pi)
assert simplify(expected - 4 * pi * alpha**2 / (3 * s)) == 0  # Peskin (5.13)
```

## A loop integral in d dimensions

**Level 5** · `qft.interactions.loop-integral` · Peskin and Schroeder (1995), eq. (A.44)

Dimensional regularisation: an integral over a loop momentum in \(d\) spacetime
dimensions, done by a Wick rotation, is
\(\frac{i}{(4\pi)^{d/2}}\Gamma(2 - \tfrac{d}{2})\Delta^{d/2 - 2}\), whose pole at
\(d = 4\) is the divergence that renormalisation removes.

```latex
\int \frac{d^d\ell}{(2\pi)^d} \frac{1}{(\ell^2 - \Delta)^2}
```

```python
ell, Delta, d = symbols("ell Delta d", positive=True)
# After the Wick rotation l^0 = i l_E^0, d^d l = i d^d l_E and l^2 - Delta = -(l_E^2 + Delta);
# the Euclidean integral is radial, over a sphere of area 2 pi^(d/2) / Gamma(d/2).
radial = integrate(ell ** (d - 1) / (ell**2 + Delta) ** 2, (ell, 0, oo), conds="none")
sphere = 2 * pi ** (d / 2) / gamma(d / 2)
computed = I * sphere / (2 * pi) ** d * radial
stated = I / (4 * pi) ** (d / 2) * gamma(2 - d / 2) * Delta ** (d / 2 - 2)  # Peskin (A.44)
for dimension in (Rational(1, 2), 1, Rational(3, 2), 3, Rational(7, 2)):  # where it converges
    assert abs(N((computed - stated).subs({d: dimension, Delta: Rational(7, 3)}))) < 1e-12
expected = stated
```

## The running coupling of φ⁴ theory

**Level 5** · `qft.interactions.running-coupling` · Peskin and Schroeder (1995), section 12.3

The renormalisation-group equation of \(\phi^4\) theory, with the one-loop beta function
\(3\bar\lambda^2/16\pi^2\) and the coupling \(\lambda\) at the scale \(M\): solved, the
coupling grows with the scale, \(\bar\lambda(p) = \lambda/(1 - \frac{3\lambda}{16\pi^2}\log\frac{p}{M})\).

```latex
\frac{d\bar{\lambda}}{d\log(p/M)} = \frac{3\bar{\lambda}^2}{16\pi^2}
\bar{\lambda}(M) = \lambda
\bar{\lambda}(p)
```

```python
t = symbols("t")  # t = log(p/M)
lam, p, M = symbols("lambda p M", positive=True)
running = Function("lambda_bar")
flow = Eq(running(t).diff(t), 3 * running(t) ** 2 / (16 * pi**2))
solution = dsolve(flow, running(t), ics={running(0): lam})
computed = solution.rhs.subs(t, log(p / M))
stated = lam / (1 - 3 * lam / (16 * pi**2) * log(p / M))  # Peskin, section 12.3
assert simplify(computed - stated) == 0
expected = stated
```
