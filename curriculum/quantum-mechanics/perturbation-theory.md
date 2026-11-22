# Perturbation theory

The energies of a Hamiltonian \(H^0 + H'\) as a series in the perturbation: to first
order, the expectation of \(H'\) in the unperturbed state; to second order, a sum over
the other states, \(E_n^{(2)} = \sum_{k \ne n} |\langle k|H'|n\rangle|^2/(E_n^{(0)} -
E_k^{(0)})\). In the infinite well, the oscillator and hydrogen.

## First order: a step in the well

**Level 2** · `qm.perturbation.first-order` · Griffiths (2005), Example 6.1

A step of height \(V_0\) across the left half of the infinite well, written as Griffiths
writes a perturbation, \(H'\), and as a piecewise function: it raises the ground state by
\(V_0/2\).

```latex
\psi_1(x) := \sqrt{\frac{2}{a}} \sin\left(\frac{\pi x}{a}\right)
H'(x) := \begin{cases} V_0 & \text{if } x < a/2 \\ 0 & \text{otherwise} \end{cases}
\int_0^a \psi_1(x)^* H'(x) \psi_1(x) \, dx
```

```python
x = symbols("x", real=True)
a, V_0 = symbols("a V_0", positive=True)
psi_1 = sqrt(2 / a) * sin(pi * x / a)
H1 = Piecewise((V_0, x < a / 2), (0, True))
expected = simplify(integrate(psi_1 * H1 * psi_1, (x, 0, a)))
assert expected == V_0 / 2  # Griffiths, Example 6.1
```

## Second order: an oscillator in a uniform field

**Level 3** · `qm.perturbation.second-order` · Sakurai (2017), Problem 5.1

A perturbation \(bx\) on the oscillator connects the ground state to the first excited
state only: the matrix element, then the second-order shift built from it,
\(-b^2/2m\omega^2\), which is exact, as completing the square shows.

```latex
\psi_0(x) := \left(\frac{m\omega}{\pi\hbar}\right)^{1/4} e^{-\frac{m\omega x^2}{2\hbar}}
\psi_1(x) := \sqrt{\frac{2m\omega}{\hbar}}\, x\, \psi_0(x)
V_{10} := \int_{-\infty}^{\infty} \psi_1(x)^* \, b x \, \psi_0(x) \, dx
\Delta_0^{(2)} := \frac{|V_{10}|^2}{\frac{1}{2}\hbar\omega - \frac{3}{2}\hbar\omega}
\Delta_0^{(2)}
```

```python
x, b = symbols("x b", real=True)
m, omega, hbar = symbols("m omega hbar", positive=True)
psi_0 = (m * omega / (pi * hbar)) ** Rational(1, 4) * exp(-m * omega * x**2 / (2 * hbar))
psi_1 = sqrt(2 * m * omega / hbar) * x * psi_0
assert integrate(psi_1**2, (x, -oo, oo)) == 1
V_10 = integrate(psi_1 * b * x * psi_0, (x, -oo, oo))
assert simplify(V_10 - b * sqrt(hbar / (2 * m * omega))) == 0
computed = simplify(V_10**2 / (hbar * omega / 2 - 3 * hbar * omega / 2))
# Exactly, the field shifts the well's centre and lowers its bottom, every level alike.
V = m * omega**2 * x**2 / 2 + b * x
[centre] = solve(diff(V, x), x)
exact_shift = V.subs(x, centre)
assert simplify(exact_shift + b**2 / (2 * m * omega**2)) == 0  # Sakurai, Problem 5.1(b)
assert simplify(computed - exact_shift) == 0
expected = exact_shift
```

## The Stark effect in hydrogen

**Level 3** · `qm.perturbation.stark` · Griffiths (2005), Problem 6.36

The one matrix element of \(z\) that splits the \(n = 2\) level in an electric field,
\(\langle\psi_{200}|z|\psi_{210}\rangle = -3a\): wave functions built from radial
functions and spherical harmonics, then an integral over all of space.

```latex
R_{20}(r) := \frac{1}{\sqrt{2}} a^{-3/2} \left(1 - \frac{r}{2a}\right) e^{-\frac{r}{2a}}
R_{21}(r) := \frac{1}{\sqrt{24}} a^{-3/2} \frac{r}{a} e^{-\frac{r}{2a}}
\psi_{200} := R_{20}(r) \frac{1}{\sqrt{4\pi}}
\psi_{210} := R_{21}(r) \sqrt{\frac{3}{4\pi}} \cos\theta
\int_0^{2\pi} \int_0^{\pi} \int_0^{\infty} \psi_{200}^* \, r\cos\theta \, \psi_{210} \, r^2 \sin\theta \, dr \, d\theta \, d\phi
```

```python
r = symbols("r", positive=True)
theta, phi = symbols("theta phi", real=True)
a = symbols("a", positive=True)
R_20 = 1 / sqrt(2) * a ** Rational(-3, 2) * (1 - r / (2 * a)) * exp(-r / (2 * a))
R_21 = 1 / sqrt(24) * a ** Rational(-3, 2) * r / a * exp(-r / (2 * a))
psi_200 = R_20 * Ynm(0, 0, theta, phi).expand(func=True)
psi_210 = R_21 * Ynm(1, 0, theta, phi).expand(func=True)
limits = ((r, 0, oo), (theta, 0, pi), (phi, 0, 2 * pi))
assert integrate(psi_210**2 * r**2 * sin(theta), *limits) == 1
expected = integrate(psi_200 * r * cos(theta) * psi_210 * r**2 * sin(theta), *limits)
assert expected == -3 * a  # Griffiths, Problem 6.36: the level splits by +-3 e a E
```

## Second order: a sum over states

**Level 4** · `qm.perturbation.sum-over-states` · Landau and Lifshitz (1977), section 38

A cubic anharmonicity \(\lambda x^3\) on the oscillator, in Dirac notation with ladder
operators: the first-order shift vanishes, and the second order is a sum over every other
number state, of which \(|1\rangle\) and \(|3\rangle\) contribute,
\(-\tfrac{11}{8}\lambda^2\hbar^2/m^3\omega^4\).

```latex
\langle n' | n \rangle = \delta_{n'n}
a |n\rangle = \sqrt{n}\, |n - 1\rangle
a^\dagger |n\rangle = \sqrt{n + 1}\, |n + 1\rangle
x := \sqrt{\frac{\hbar}{2m\omega}} \left(a + a^\dagger\right)
E_n^{(0)} := \hbar\omega\left(n + \frac{1}{2}\right)
H' := \lambda x^3
\sum_{k \neq 0} \frac{|\langle k | H' | 0 \rangle|^2}{E_0^{(0)} - E_k^{(0)}}
```

```python
m, omega, hbar, lam = symbols("m omega hbar lambda", positive=True)
size = 12
# Matrix elements in the number basis: <k|a|j> = sqrt(j) when k = j - 1.
a = Matrix(size, size, lambda k, j: sqrt(j) if k == j - 1 else 0)
x = sqrt(hbar / (2 * m * omega)) * (a + a.T)
H1 = lam * x**3
E = lambda k: hbar * omega * (k + Rational(1, 2))
assert H1[0, 0] == 0  # no first-order shift
contributing = [k for k in range(1, size - 3) if H1[k, 0] != 0]
assert contributing == [1, 3]
computed = simplify(sum(H1[k, 0] ** 2 / (E(0) - E(k)) for k in range(1, size - 3)))
# Landau and Lifshitz: -(lambda^2/hbar omega)(hbar/m omega)^3 (15/4 (n + 1/2)^2 + 7/16) at n = 0.
stated = (
    -(lam**2)
    / (hbar * omega)
    * (hbar / (m * omega)) ** 3
    * (Rational(15, 4) * Rational(1, 4) + Rational(7, 16))
)
assert simplify(computed - stated) == 0
assert simplify(stated + Rational(11, 8) * lam**2 * hbar**2 / (m**3 * omega**4)) == 0
expected = stated
```
