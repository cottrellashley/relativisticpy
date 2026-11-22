# The Ising model and mean-field theory

The Ising chain solved exactly by its transfer matrix, with \(K = \beta J\) and no
field; then the mean-field theory of the ferromagnet and the critical exponents it
predicts.

## The eigenvalues of the transfer matrix

**Level 2** · `statmech.ising.transfer-eigenvalues` · Pathria and Beale (2011), section 13.2

The transfer matrix of the Ising chain, \(\langle s | T | s' \rangle = e^{K s s'}\),
and the roots of its characteristic equation: \(2 \sinh K\) and \(2 \cosh K\).

```latex
T := \begin{pmatrix} e^{K} & e^{-K} \\ e^{-K} & e^{K} \end{pmatrix}
solve(\det(T - \lambda I) = 0, \lambda)
```

```python
K, lam = symbols("K lambda")
T = Matrix([[exp(K), exp(-K)], [exp(-K), exp(K)]])
roots = solve((T - lam * eye(2)).det(), lam)
expected = [2 * sinh(K), 2 * cosh(K)]
assert len(roots) == 2
assert all(abs(complex((r - e).subs(K, 0.7).evalf())) < 1e-12 for r, e in zip(roots, expected))
```

## The free energy of the Ising chain

**Level 3** · `statmech.ising.free-energy` · Pathria and Beale (2011), section 13.2

The partition function of a ring of \(N\) spins is \(\lambda_1^N + \lambda_2^N\). In the
thermodynamic limit the larger eigenvalue alone survives, and \(-\beta f =
\ln(2 \cosh K)\).

```latex
\lambda_1 := 2 \cosh K
\lambda_2 := 2 \sinh K
Z_N := \lambda_1^N + \lambda_2^N
\lim_{N \to \infty} \frac{1}{N} \ln Z_N
```

```python
K = symbols("K", positive=True)
expected = log(2 * cosh(K))
# (1/N) ln(l1^N + l2^N) = ln l1 + (1/N) ln(1 + tanh(K)^N), and tanh K < 1.
for k in (0.1, 0.7, 2.0):
    l1, l2 = 2 * np.cosh(k), 2 * np.sinh(k)
    assert abs(np.logaddexp(4000 * np.log(l1), 4000 * np.log(l2)) / 4000 - np.log(l1)) < 1e-6
```

## A ring of four spins by contraction

**Level 3** · `statmech.ising.ring` · Pathria and Beale (2011), section 13.2

The partition function of four spins on a ring as the trace of the fourth power of the
transfer matrix, written as a contraction over the spin states:
\((2 \cosh K)^4 + (2 \sinh K)^4\).

```latex
\mathcal{S} := \mathbb{R}^{2}
a, b, c, d \in \mathcal{S}
T_{ab} := [[e^{K}, e^{-K}], [e^{-K}, e^{K}]]
T_{ab} T_{bc} T_{cd} T_{da}
```

```python
import itertools

K = symbols("K", positive=True)
states = itertools.product([1, -1], repeat=4)
expected = sum(exp(K * sum(s[i] * s[(i + 1) % 4] for i in range(4))) for s in states)
assert simplify((expected - (2 * cosh(K)) ** 4 - (2 * sinh(K)) ** 4).rewrite(exp)) == 0
```

## The Curie–Weiss law

**Level 4** · `statmech.meanfield.curie-weiss` · Pathria and Beale (2011), chapter 12

In mean-field theory the magnetisation solves its own equation, \(m = \tanh((zJm +
h)/kT)\). Differentiated implicitly at \(h = 0\) above the transition, where \(m = 0\),
it gives the susceptibility \(1 / (k(T - T_c))\), with \(T_c = zJ/k\).

```latex
m = \tanh\left(\frac{z J m + h}{k T}\right)
\left. \frac{\partial m}{\partial h} \right|_{h = 0, m = 0}
```

```python
z, J, k, T, h, m = symbols("z J k T h m", positive=True)
chi = symbols("chi")
u = (z * J * m + h) / (k * T)
# Differentiate m = tanh(u) with respect to h, where dm/dh = chi, then set h = m = 0.
equation = Eq(chi, (1 - tanh(u) ** 2) * (z * J * chi + 1) / (k * T))
expected = solve(equation.subs({h: 0, m: 0}), chi)[0]
assert simplify(expected - 1 / (k * T - z * J)) == 0
```

## The critical exponent of the order parameter

**Level 5** · `statmech.meanfield.critical-exponent` · Pathria and Beale (2011), chapter 12

Landau's free energy, its minimum, and the exponent \(\beta\) in \(m \sim (T_c -
T)^{\beta}\): an asymptotic relation that defines a number, \(\beta = \frac{1}{2}\) in
mean-field theory.

```latex
f(m) := a (T - T_c) m^2 + b m^4
f'(m) = 0
m \sim (T_c - T)^{\beta}
\beta
```

```python
a, b, m, t = symbols("a b m t", positive=True)  # below the transition, t = T_c - T > 0
f = a * (-t) * m**2 + b * m**4
minimum = solve(diff(f, m), m)  # the positive root
assert len(minimum) == 1 and simplify(minimum[0] ** 2 - a * t / (2 * b)) == 0
expected = limit(log(minimum[0]) / log(t), t, 0)
assert expected == Rational(1, 2)
```
