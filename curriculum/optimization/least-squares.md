# Least squares

Least squares as an optimisation problem: the minimiser of \(\|Ax - b\|_2^2\) written with
\(\arg\min\), the least-norm solution of an underdetermined system, the pseudoinverse of a
matrix that is not of full rank, and a Gauss–Newton step for a nonlinear fit.

## The pseudoinverse

**Level 2** · `opt.least-squares.pseudoinverse` · Strang (2016), chapter 7

\(A^{+}\), the Moore–Penrose pseudoinverse, for a matrix of rank one, whose \(A^T A\) has
no inverse: \(A^{+}b\) is the shortest of the least-squares solutions,
\((\frac{17}{70}, \frac{17}{35})\).

```latex
A := \begin{pmatrix} 1 & 2 \\ 2 & 4 \\ 3 & 6 \end{pmatrix}
\mathbf{b} := \begin{pmatrix} 1 \\ 2 \\ 4 \end{pmatrix}
A^{+} \mathbf{b}
```

```python
A = Matrix([[1, 2], [2, 4], [3, 6]])
b = Matrix([1, 2, 4])
assert A.rank() == 1 and (A.T * A).det() == 0
expected = A.pinv() * b
assert expected == Matrix([Rational(17, 70), Rational(17, 35)])
assert np.allclose(np.linalg.pinv(np.array(A.tolist(), float)) @ [1, 2, 4], [17 / 70, 17 / 35])
```

## A Gauss–Newton step

**Level 2** · `opt.least-squares.gauss-newton` · Nocedal and Wright (2006), chapter 10

Fitting \(y = a e^{bt}\) to the points \((0, 2)\), \((1, 4)\) and \((2, 10)\): the residuals
\(r\), their Jacobian \(J\), and one Gauss–Newton step \(-(J^T J)^{-1} J^T r\) from
\((a, b) = (1, 1)\), to \((1.7369, 0.8058)\).

```latex
r(a, b) := \begin{pmatrix} a - 2 \\ a e^{b} - 4 \\ a e^{2b} - 10 \end{pmatrix}
J(a, b) := \begin{pmatrix} 1 & 0 \\ e^{b} & a e^{b} \\ e^{2b} & 2a e^{2b} \end{pmatrix}
\begin{pmatrix} 1 \\ 1 \end{pmatrix} - \left(J(1, 1)^T J(1, 1)\right)^{-1} J(1, 1)^T r(1, 1)
```

```python
a, b = symbols("a b")
r = Matrix([a - 2, a * exp(b) - 4, a * exp(2 * b) - 10])
J = r.jacobian([a, b])
assert J == Matrix([[1, 0], [exp(b), a * exp(b)], [exp(2 * b), 2 * a * exp(2 * b)]])
at = {a: 1, b: 1}
expected = Matrix([1, 1]) - (J.subs(at).T * J.subs(at)).inv() * J.subs(at).T * r.subs(at)
assert np.allclose([float(v) for v in expected], [1.73691821, 0.80580806])
t, y = np.array([0.0, 1.0, 2.0]), np.array([2.0, 4.0, 10.0])
Jn = np.column_stack([np.exp(t), t * np.exp(t)])
step = np.linalg.lstsq(Jn, -(np.exp(t) - y), rcond=None)[0]
assert np.allclose(np.array([1.0, 1.0]) + step, [float(v) for v in expected])
```

## Least squares as a minimisation

**Level 3** · `opt.least-squares.argmin` · Boyd and Vandenberghe (2004), section 6.1

The parabola \(x_1 + x_2 t + x_3 t^2\) closest to five points, written as the problem
itself: the minimiser of \(\|A\mathbf{x} - \mathbf{b}\|_2^2\), which the normal equations
give as \((\frac{4}{7}, \frac{7}{10}, \frac{17}{14})\).

```latex
A := \begin{pmatrix} 1 & -2 & 4 \\ 1 & -1 & 1 \\ 1 & 0 & 0 \\ 1 & 1 & 1 \\ 1 & 2 & 4 \end{pmatrix}
\mathbf{b} := \begin{pmatrix} 4 \\ 1 \\ 1 \\ 2 \\ 7 \end{pmatrix}
\arg\min_{\mathbf{x}} \|A\mathbf{x} - \mathbf{b}\|_2^2
```

```python
A = Matrix([[1, -2, 4], [1, -1, 1], [1, 0, 0], [1, 1, 1], [1, 2, 4]])
b = Matrix([4, 1, 1, 2, 7])
expected = (A.T * A).LUsolve(A.T * b)
assert expected == Matrix([Rational(4, 7), Rational(7, 10), Rational(17, 14)])
fit = np.polyfit([-2, -1, 0, 1, 2], [4, 1, 1, 2, 7], 2)[::-1]
assert np.allclose(fit, [4 / 7, 7 / 10, 17 / 14])
```

## The least-norm solution

**Level 3** · `opt.least-squares.least-norm` · Boyd and Vandenberghe (2004), section 6.2

Two equations in three unknowns have a line of solutions, \((t, 1 - t, 1 + t)\); the
shortest is \(A^T(AA^T)^{-1}\mathbf{b} = (0, 1, 1)\), written as the problem that defines it.

```latex
A := \begin{pmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \end{pmatrix}
\mathbf{b} := \begin{pmatrix} 1 \\ 2 \end{pmatrix}
\arg\min_{\mathbf{x}} \|\mathbf{x}\|_2 \quad \text{subject to} \quad A\mathbf{x} = \mathbf{b}
```

```python
A = Matrix([[1, 1, 0], [0, 1, 1]])
b = Matrix([1, 2])
expected = A.T * (A * A.T).inv() * b
assert expected == Matrix([0, 1, 1])
t = symbols("t", real=True)
on_the_line = Matrix([t, 1 - t, 1 + t])
assert A * on_the_line == b
assert solve(diff((on_the_line.T * on_the_line)[0], t), t) == [0]
assert np.allclose(np.linalg.pinv(np.array(A.tolist(), float)) @ [1, 2], [0, 1, 1])
```
