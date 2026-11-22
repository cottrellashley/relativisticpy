# Dirac notation

Kets, bras and the operators made from them, as Sakurai, Griffiths and Nielsen and
Chuang write them. A ket is a state whatever basis it is written in; its bra is its
conjugate transpose, a bra against a ket is their inner product, and a ket against a bra
is an operator. Continuous bases, such as position and momentum, insert completeness as
an integral.

## An inner product in an orthonormal basis

**Level 4** · `qm.dirac.inner-product` · Griffiths (2005), Problem 3.22

Two kets written as combinations of orthonormal basis kets, and their inner product: the
bra of a ket conjugates its coefficients.

```latex
\langle m | n \rangle = \delta_{mn}
|\alpha\rangle := i|1\rangle - 2|2\rangle - i|3\rangle
|\beta\rangle := i|1\rangle + 2|3\rangle
\langle \alpha | \beta \rangle
```

```python
# The components of each ket in the basis |1>, |2>, |3>; the bra is the conjugate transpose.
alpha = Matrix([I, -2, -I])
beta = Matrix([I, 0, 2])
computed = (alpha.H * beta)[0]
assert computed == 1 + 2 * I  # Griffiths: <alpha|beta> = 1 + 2i
assert (beta.H * alpha)[0] == conjugate(computed)  # and <beta|alpha> is its conjugate
expected = computed
```

## The completeness relation

**Level 4** · `qm.dirac.completeness` · Nielsen and Chuang (2010), section 2.1.4

The outer products of the kets of an orthonormal basis, summed, give the identity:
\(\sum_i |i\rangle\langle i| = I\), here for the eigenbasis of \(X\), written as columns in
the computational basis.

```latex
|+\rangle := \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 1 \end{pmatrix}
|-\rangle := \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ -1 \end{pmatrix}
|+\rangle\langle+| + |-\rangle\langle-| = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}
```

```python
plus = Matrix([1, 1]) / sqrt(2)
minus = Matrix([1, -1]) / sqrt(2)
assert (plus.H * minus)[0] == 0 and (plus.H * plus)[0] == 1  # an orthonormal basis
assert plus * plus.H + minus * minus.H == eye(2)
expected = True
```

## A change of basis

**Level 2** · `qm.dirac.change-of-basis` · Sakurai (2017), section 1.5

\(S_z\) in the basis of the eigenkets of \(S_x\): the transformation matrix \(U\), whose
columns are the new base kets, takes an operator's matrix \(A\) to \(U^\dagger A U\), with
the Hermitian conjugate written as a dagger.

```latex
U := \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}
S_z := \frac{\hbar}{2} \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
U^\dagger S_z U
```

```python
hbar = symbols("hbar")
U = Matrix([[1, 1], [1, -1]]) / sqrt(2)
S_z = hbar / 2 * Matrix([[1, 0], [0, -1]])
assert U.H * U == eye(2)  # U is unitary
expected = U.H * S_z * U
# In the S_x basis S_z is off-diagonal: it swaps the two eigenkets of S_x.
assert expected == hbar / 2 * Matrix([[0, 1], [1, 0]])
```

## A wave function in momentum space

**Level 4** · `qm.dirac.momentum-wavefunction` · Sakurai (2017), section 1.7

A Gaussian wave packet taken to momentum space by inserting the completeness of the
position eigenkets: \(\langle p|\alpha\rangle = \int dx\, \langle p|x\rangle \langle
x|\alpha\rangle\), where \(\langle p|x\rangle\) is the conjugate of \(\langle x|p\rangle\).

```latex
\langle x | p \rangle := \frac{1}{\sqrt{2\pi\hbar}} e^{ipx/\hbar}
\langle x | \alpha \rangle := \frac{1}{\pi^{1/4} \sqrt{d}} \exp\left(ikx - \frac{x^2}{2d^2}\right)
\int_{-\infty}^{\infty} \langle p | x \rangle \langle x | \alpha \rangle \, dx
```

```python
x, p, k = symbols("x p k", real=True)
d, hbar = symbols("d hbar", positive=True)
x_p = exp(I * p * x / hbar) / sqrt(2 * pi * hbar)
x_alpha = exp(I * k * x - x**2 / (2 * d**2)) / (pi**Rational(1, 4) * sqrt(d))
computed = integrate(conjugate(x_p) * x_alpha, (x, -oo, oo))
stated = sqrt(d / (hbar * sqrt(pi))) * exp(-((p - hbar * k) ** 2) * d**2 / (2 * hbar**2))
assert simplify(computed - stated) == 0  # Sakurai: a Gaussian about hbar k, of width hbar/d
expected = stated
```
