# Time evolution

The Schrödinger equation moves a state with the unitary operator
\(U(t) = e^{-iHt/\hbar}\): an exponential of a matrix for a spin, a phase
\(e^{-iE_nt/\hbar}\) on each stationary state of a well, and, in Feynman's formulation,
a sum over every path between two points.

## The evolution operator is unitary

**Level 2** · `qm.time-evolution.unitarity` · Sakurai (2017), section 2.1

\(U(t) = e^{-iHt/\hbar}\) for a two-level Hamiltonian, an exponential of a matrix written
as a power of \(e\): \(U^\dagger U = 1\), so probability is conserved.

```latex
H := \frac{\hbar\omega}{2} \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
U(t) := e^{-iHt/\hbar}
U(t)^\dagger U(t) = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}
```

```python
t = symbols("t", real=True)
omega, hbar = symbols("omega hbar", positive=True)
H = hbar * omega / 2 * Matrix([[0, 1], [1, 0]])
U = (-I * H * t / hbar).exp()
assert simplify(U - Matrix([[cos(omega * t / 2), -I * sin(omega * t / 2)], [-I * sin(omega * t / 2), cos(omega * t / 2)]])) == zeros(2, 2)
assert simplify(U.H * U) == eye(2)
expected = True
```

## Spin precession

**Level 3** · `qm.time-evolution.precession` · Sakurai (2017), section 2.1

A spin in a magnetic field along \(z\), \(H = \omega S_z\), starts in the eigenstate of
\(S_x\): the state is evolved, then the expectation of \(S_x\) taken in it,
\(\langle S_x\rangle = \frac{\hbar}{2}\cos\omega t\). The spin precesses about the field.

```latex
S_x := \frac{\hbar}{2} \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
S_z := \frac{\hbar}{2} \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
H := \omega S_z
\chi(t) := e^{-iHt/\hbar} \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 1 \end{pmatrix}
\chi(t)^\dagger S_x \chi(t)
```

```python
t = symbols("t", real=True)
omega, hbar = symbols("omega hbar", positive=True)
S_x = hbar / 2 * Matrix([[0, 1], [1, 0]])
S_z = hbar / 2 * Matrix([[1, 0], [0, -1]])
chi = (-I * omega * S_z * t / hbar).exp() * Matrix([1, 1]) / sqrt(2)
computed = simplify((chi.H * S_x * chi)[0])
stated = hbar / 2 * cos(omega * t)  # Sakurai, section 2.1
assert simplify(computed - stated) == 0
expected = stated
```

## A superposition in the square well

**Level 3** · `qm.time-evolution.beats` · Griffiths (2005), Problem 2.5

Two stationary states of the infinite well, each turning with its own phase
\(e^{-iE_nt/\hbar}\), with \(E_n\) a sequence: \(\langle x\rangle\) oscillates at the
frequency \((E_2 - E_1)/\hbar = 3\pi^2\hbar/2ma^2\).

```latex
\psi_n(x) := \sqrt{\frac{2}{a}} \sin\left(\frac{n\pi x}{a}\right)
E_n := \frac{n^2\pi^2\hbar^2}{2ma^2}
\Psi(x, t) := \frac{1}{\sqrt{2}} \left(\psi_1(x) e^{-iE_1 t/\hbar} + \psi_2(x) e^{-iE_2 t/\hbar}\right)
\int_0^a x\, |\Psi(x, t)|^2 \, dx
```

```python
x, t = symbols("x t", real=True)
a, m, hbar = symbols("a m hbar", positive=True)
psi = lambda n: sqrt(2 / a) * sin(n * pi * x / a)
E = lambda n: n**2 * pi**2 * hbar**2 / (2 * m * a**2)
Psi = (psi(1) * exp(-I * E(1) * t / hbar) + psi(2) * exp(-I * E(2) * t / hbar)) / sqrt(2)
density = expand(Psi * conjugate(Psi))
assert simplify(integrate(density, (x, 0, a))) == 1
computed = integrate(x * density, (x, 0, a))
omega = (E(2) - E(1)) / hbar
stated = a / 2 * (1 - 32 / (9 * pi**2) * cos(omega * t))  # Griffiths, Problem 2.5(c)
assert simplify((computed - stated).rewrite(cos)) == 0
expected = stated
```

## The propagator of a free particle as a path integral

**Level 5** · `qm.time-evolution.path-integral` · Sakurai (2017), section 2.6

Feynman's sum over every path from \(x_a\) at \(t_a\) to \(x_b\) at \(t_b\), weighted by
\(e^{iS/\hbar}\), for a free particle: the propagator
\(\sqrt{m/2\pi i\hbar(t_b - t_a)}\,\exp[im(x_b - x_a)^2/2\hbar(t_b - t_a)]\). No design of
the language reads a functional integral yet.

```latex
\int_{x_a}^{x_b} \mathcal{D}[x(t)] \, \exp\left[\frac{i}{\hbar} \int_{t_a}^{t_b} \frac{1}{2} m \dot{x}^2 \, dt\right]
```

```python
x_0, x_1, x_2 = symbols("x_0 x_1 x_2", real=True)
x_a, x_b, t_a, t_b = symbols("x_a x_b t_a t_b", real=True)
m, hbar = symbols("m hbar", positive=True)
tau_1, tau_2 = symbols("tau_1 tau_2", positive=True)


def heat_kernel(x_end, x_start, tau):  # the path integral in imaginary time, tau = i t
    return sqrt(m / (2 * pi * hbar * tau)) * exp(-m * (x_end - x_start) ** 2 / (2 * hbar * tau))


# Slicing the paths: integrating over an intermediate point joins two slices into one,
# so the limit of many slices is the kernel of one.
joined = integrate(heat_kernel(x_2, x_1, tau_2) * heat_kernel(x_1, x_0, tau_1), (x_1, -oo, oo))
assert simplify(joined - heat_kernel(x_2, x_0, tau_1 + tau_2)) == 0
T = t_b - t_a
K = sqrt(m / (2 * pi * I * hbar * T)) * exp(I * m * (x_b - x_a) ** 2 / (2 * hbar * T))
assert simplify(heat_kernel(x_b, x_a, I * T) - K) == 0  # back to real time
assert simplify(I * hbar * diff(K, t_b) + hbar**2 / (2 * m) * diff(K, x_b, 2)) == 0  # it solves Schrodinger's equation
expected = K
```
