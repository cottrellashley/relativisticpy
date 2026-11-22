# Linear systems and control

The linear system \(\dot{\mathbf{x}} = A\mathbf{x} + B\mathbf{u}\), \(y = C\mathbf{x}\): its
classification by trace and determinant, its solution by the matrix exponential, its
transfer function and step response, controllability and observability by the ranks of
Kalman's matrices, a stability boundary, pole placement by Ackermann's formula, and the
Lyapunov equation.

## Trace and determinant

**Level 1** · `dyn.linear.trace-determinant` · Strogatz (2015), section 5.2

A planar linear system is classified by \(\tau = \operatorname{tr} A\) and
\(\Delta = \det A\): with \(\tau < 0\), \(\Delta > 0\) and \(\tau^2 - 4\Delta = 1 > 0\), this one is
a stable node.

```latex
A := \begin{pmatrix} 0 & 1 \\ -2 & -3 \end{pmatrix}
\left(\operatorname{tr} A\right)^2 - 4 \det A
```

```python
A = Matrix([[0, 1], [-2, -3]])
expected = A.trace() ** 2 - 4 * A.det()
assert expected == 1
assert A.eigenvals() == {-1: 1, -2: 1}  # real and negative: a stable node
```

## Percent overshoot

**Level 1** · `dyn.control.overshoot` · Franklin, Powell and Emami-Naeini (2015), chapter 3

A second-order system with damping ratio \(\zeta = 0.5\) overshoots its step by
\(100 e^{-\pi\zeta/\sqrt{1 - \zeta^2}} = 16.3\) percent.

```latex
100 \exp\left(-\frac{\pi \cdot 0.5}{\sqrt{1 - 0.5^2}}\right)
```

```python
from scipy.signal import lti, step

expected = 100 * exp(-pi / sqrt(3))
assert abs(float(expected) - 16.3) < 0.05
t, response = step(lti([1], [1, 1, 1]), N=200001, T=np.linspace(0, 20, 200001))
assert abs(100 * (response.max() - 1) - float(expected)) < 1e-3  # omega_n = 1, zeta = 1/2
```

## A step response

**Level 1** · `dyn.control.step-response` · Franklin, Powell and Emami-Naeini (2015), chapter 3

The response of \(\frac{1}{s + 1}\) to a unit step, \(\frac{1}{s}\), by the inverse Laplace
transform written with its book name: \(1 - e^{-t}\) for \(t > 0\).

```latex
t > 0
\mathcal{L}^{-1}\left\{\frac{1}{s(s + 1)}\right\}
```

```python
s = symbols("s")
t = symbols("t", positive=True)
expected = 1 - exp(-t)
assert simplify(inverse_laplace_transform(1 / (s * (s + 1)), s, t) - expected) == 0
assert simplify(laplace_transform(expected, t, s, noconds=True) - 1 / (s * (s + 1))) == 0
```

## The solution by the matrix exponential

**Level 2** · `dyn.linear.matrix-exponential` · Åström and Murray (2008), chapter 5

\(\mathbf{x}(t) = e^{At}\mathbf{x}(0)\): from \((1, 0)\), the system with eigenvalues \(-1\)
and \(-2\) moves along \((2e^{-t} - e^{-2t}, -2e^{-t} + 2e^{-2t})\).

```latex
A := \begin{pmatrix} 0 & 1 \\ -2 & -3 \end{pmatrix}
e^{A t} \begin{pmatrix} 1 \\ 0 \end{pmatrix}
```

```python
from scipy.linalg import expm

t = symbols("t")
A = Matrix([[0, 1], [-2, -3]])
expected = Matrix([2 * exp(-t) - exp(-2 * t), -2 * exp(-t) + 2 * exp(-2 * t)])
assert simplify((A * t).exp() * Matrix([1, 0]) - expected) == zeros(2, 1)
assert simplify(diff(expected, t) - A * expected) == zeros(2, 1)
numeric = expm(np.array([[0, 1], [-2, -3]], float) * 0.7) @ [1, 0]
assert np.allclose(numeric, [float(v) for v in expected.subs(t, Rational(7, 10))])
```

## Controllability

**Level 2** · `dyn.control.controllability` · Åström and Murray (2008), section 6.1

The system is controllable, every state reachable, when Kalman's matrix
\([B \;\; AB]\), the columns \(B\) and \(AB\) side by side, has full rank: here 2.

```latex
A := \begin{pmatrix} 0 & 1 \\ -2 & -3 \end{pmatrix}
B := \begin{pmatrix} 0 \\ 1 \end{pmatrix}
\operatorname{rank} \begin{pmatrix} B & AB \end{pmatrix}
```

```python
A = Matrix([[0, 1], [-2, -3]])
B = Matrix([0, 1])
expected = Matrix.hstack(B, A * B).rank()
assert expected == 2
assert np.linalg.matrix_rank(np.array(Matrix.hstack(B, A * B).tolist(), float)) == 2
```

## Observability

**Level 2** · `dyn.control.observability` · Åström and Murray (2008), section 7.1

The system is observable when the matrix with the rows \(C\) and \(CA\) has full rank. Two
decoupled modes measured through the first alone are not: the rank is 1, and the second
mode cannot be seen in \(y\).

```latex
A := \begin{pmatrix} 1 & 0 \\ 0 & 2 \end{pmatrix}
C := \begin{pmatrix} 1 & 0 \end{pmatrix}
\operatorname{rank} \begin{pmatrix} C \\ C A \end{pmatrix}
```

```python
A = diag(1, 2)
C = Matrix([[1, 0]])
expected = Matrix.vstack(C, C * A).rank()
assert expected == 1
```

## The transfer function of a state-space system

**Level 3** · `dyn.control.transfer-function` · Åström and Murray (2008), chapter 8

\(G(s) = C(sI - A)^{-1}B + D\), a single input and output, so a scalar function of \(s\):
\(\frac{1}{s^2 + 3s + 2}\), with poles at the eigenvalues of \(A\).

```latex
A := \begin{pmatrix} 0 & 1 \\ -2 & -3 \end{pmatrix}
B := \begin{pmatrix} 0 \\ 1 \end{pmatrix}
C := \begin{pmatrix} 1 & 0 \end{pmatrix}
D := 0
C (s I - A)^{-1} B + D
```

```python
from scipy.signal import ss2tf

s = symbols("s")
A = Matrix([[0, 1], [-2, -3]])
expected = simplify((Matrix([[1, 0]]) * (s * eye(2) - A).inv() * Matrix([0, 1]))[0])
assert expected == 1 / (s**2 + 3 * s + 2)
numerator, denominator = ss2tf([[0, 1], [-2, -3]], [[0], [1]], [[1, 0]], [[0]])
assert np.allclose(numerator, [[0, 0, 1]]) and np.allclose(denominator, [1, 3, 2])
```

## The critical gain

**Level 3** · `dyn.control.critical-gain` · Franklin, Powell and Emami-Naeini (2015), chapter 3

The closed loop with characteristic polynomial \(s^3 + 3s^2 + 3s + 1 + K\) is stable for
\(0 < K < 8\), as Routh's test says. At \(K = 8\) a pair of poles reaches the imaginary axis,
at \(s = \pm i\sqrt{3}\): the solution of \(p(i\omega) = 0\) for the gain and the frequency.

```latex
K > 0
\omega > 0
p(s) := s^3 + 3s^2 + 3s + 1 + K
solve(p(i\omega) = 0, [K, \omega])
```

```python
K, omega = symbols("K omega", positive=True)
s = symbols("s")
p = s**3 + 3 * s**2 + 3 * s + 1 + K
at = expand(p.subs(s, I * omega))
assert solve([re(at), im(at)], [K, omega], dict=True) == [{K: 8, omega: sqrt(3)}]
assert 3 * 3 > 1 + 7 and not 3 * 3 > 1 + 9  # Routh: a_1 a_2 > a_3 for a cubic
assert max(np.roots([1, 3, 3, 1 + 7.9]).real) < 0 < max(np.roots([1, 3, 3, 1 + 8.1]).real)
expected = [(8, sqrt(3))]
```

## Pole placement by Ackermann's formula

**Level 4** · `dyn.control.ackermann` · Franklin, Powell and Emami-Naeini (2015), chapter 7

The feedback \(u = -K\mathbf{x}\) that puts the poles of a double integrator at
\(-2 \pm 2i\), the roots of \(\alpha_c(s) = s^2 + 4s + 8\): Ackermann's formula
\(K = [0 \;\; 1]\,\mathcal{C}^{-1}\alpha_c(A)\), with \(\alpha_c\) a polynomial evaluated at a
matrix, its constant term times the identity. \(K = (8, 4)\).

```latex
A := \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}
B := \begin{pmatrix} 0 \\ 1 \end{pmatrix}
\alpha_c(s) := s^2 + 4s + 8
\begin{pmatrix} 0 & 1 \end{pmatrix} \begin{pmatrix} B & AB \end{pmatrix}^{-1} \alpha_c(A)
```

```python
from scipy.signal import place_poles

A = Matrix([[0, 1], [0, 0]])
B = Matrix([0, 1])
expected = Matrix([[0, 1]]) * Matrix.hstack(B, A * B).inv() * (A**2 + 4 * A + 8 * eye(2))
assert expected == Matrix([[8, 4]])
assert (A - B * expected).eigenvals() == {-2 - 2 * I: 1, -2 + 2 * I: 1}
gains = place_poles(
    np.array([[0, 1], [0, 0]], float), np.array([[0], [1]], float), [-2 + 2j, -2 - 2j]
)
assert np.allclose(gains.gain_matrix, [[8, 4]])
```

## The Lyapunov equation

**Level 4** · `dyn.linear.lyapunov-equation` · Khalil (2002), chapter 4

A linear system is asymptotically stable exactly when \(A^T P + PA = -Q\) has a positive
definite solution \(P\) for \(Q = I\): an equation for an unknown matrix.
\(P = \begin{pmatrix} 5/4 & 1/4 \\ 1/4 & 1/4 \end{pmatrix}\), positive definite.

```latex
A := \begin{pmatrix} 0 & 1 \\ -2 & -3 \end{pmatrix}
solve(A^T P + P A = -I, P)
```

```python
from scipy.linalg import solve_continuous_lyapunov

A = Matrix([[0, 1], [-2, -3]])
p, q, r = symbols("p q r")
P = Matrix([[p, q], [q, r]])
found = solve(list(A.T * P + P * A + eye(2)), [p, q, r])
expected = P.subs(found)
assert expected == Matrix([[Rational(5, 4), Rational(1, 4)], [Rational(1, 4), Rational(1, 4)]])
assert expected.is_positive_definite
numeric = solve_continuous_lyapunov(np.array(A.T.tolist(), float), -np.eye(2))
assert np.allclose(numeric, np.array(expected.tolist(), float))
```
