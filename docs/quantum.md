# Quantum notation

> **For:** anyone who computes quantum mechanics. **You will learn:** Dirac's kets and
> bras, commutators, wavefunctions, and operators as values. **Before this:**
> [The language](language.md), and [Matrices](matrices.md).

```python
>>> from relativisticpy import Workbook

```

## Kets, bras and commutators

Dirac's notation, commutators and operators are read as books write them, by what they
are written with. As for [matrices](matrices.md), each reading is recorded, as
`Workbook.explain` shows, with a note where another was possible.

A ket a program defines as a column is that vector, and its bra is the conjugate
transpose: a bracket is their inner product, `\langle\phi|A|\psi\rangle` a row times a
matrix times a column, and `|a\rangle\langle b|` a matrix. A label may be a sign, an
expression or several, as in `|j, m\rangle`, and `|00\rangle`, of kets the program
defines, is `|0\rangle \otimes |0\rangle`. `[A, B]` of matrices or operators is their
commutator, AB - BA, and `\{A, B\}` their anticommutator; of anything else they are a
list and a set, as before.

```python
>>> qm = Workbook()
>>> qm.exe(r"""
...     |+\rangle := \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 1 \end{pmatrix}
...     |-\rangle := \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ -1 \end{pmatrix}
...     X := \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
...     Z := \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
...     \langle + | X | + \rangle
...     |+\rangle\langle+| + |-\rangle\langle-|
...     [X, Z]
... """)
[1, [[1, 0], [0, 1]], [[0, -2], [2, 0]]]

```

## Wavefunctions

A function of one variable is a wavefunction, and so is a bracket defined as
`\langle x|\alpha\rangle := ...`. A bracket of wavefunctions is an integral, of
\(\phi^* A \psi\), over the domain the program states for the variable, or over the whole
line, and angle brackets around an operator are its expectation value, in the one state
the program defines or the state a subscript names, as in `\langle x \rangle_\psi`,
divided by \(\langle\psi|\psi\rangle\), as a state need not be normalised. A function named
\(\phi\) or \(\Phi\), which books give potentials and phases too, is a state only where a
ket or a subscript names it.
Written the other way round, a bracket a program defines is its conjugate, its labels
real; and `\psi^*(x)` is the conjugate of \(\psi(x)\).

## Operators

An expression that ends in an operator with nothing to act on, as `-i\hbar\frac{d}{dx}`
does, is an operator. Defined, it is a value: it acts on the rest of the term after it,
as \(\frac{d}{dx}\) does, as in `\hat{p}\psi(x)` and `\hat{p} x \psi(x)`, and products
and powers compose it, as in `\hat{p}^2`. Defined with
`\pm` in its name, as in `a_\pm := ...`, it is two, one for each sign. A claim between
operators is decided by applying both sides to a test function, as a note says.

```python
>>> qm.exe(r"""
...     0 \leq x \leq a
...     \psi(x) := \sqrt{\frac{2}{a}} \sin\left(\frac{\pi x}{a}\right)
...     \hat{p} := -i\hbar \frac{d}{dx}
...     \langle\psi|\psi\rangle
...     \langle x \rangle_\psi
...     \langle \hat{p}^2 \rangle_\psi
...     a_\pm := \frac{1}{\sqrt{2\hbar m\omega}} \left(\mp i\hat{p} + m\omega x\right)
...     [a_-, a_+] = 1
... """)
[1, a/2, pi**2*hbar**2/a**2, True]

```

## Rules

Angle brackets hold what a program defines, as `\langle x^2 \rangle := ...` does, or the
bracket of two, as `\langle p, q \rangle := \int_{-1}^{1} p(x) q(x) \, dx` defines; around
two vectors they are their inner product. An `=` whose left side is written with
operators and kets that nothing defines states a rule for what follows, not a claim: a
commutator, as `[\hat{x}, \hat{p}] = i\hbar`, operators acting on a ket, as
`a|n\rangle = \sqrt{n}|n - 1\rangle`, or a bracket, as `\langle m|n\rangle =
\delta_{mn}`, whose labels stand for any label. The algebra of those operators and kets
reduces by the rules: operators act on kets, bras meet kets, and products are put in the
order the commutators give. The modulus of a bracket of them is the root of the bracket
times its conjugate, so that a sum over a label, as of `|\langle k|H'|0\rangle|^2`, is
taken.

```python
>>> rules = Workbook()
>>> rules.exe(r"""
...     \langle (1, 2), (3, 4) \rangle
...     \langle m | n \rangle = \delta_{mn}
...     a |n\rangle = \sqrt{n}\, |n - 1\rangle
...     a^\dagger |n\rangle = \sqrt{n + 1}\, |n + 1\rangle
...     \langle n | (a + a^\dagger)^2 | n \rangle
...     [\hat{x}, \hat{p}] = i\hbar
...     [\hat{x}^3, \hat{p}] = 3i\hbar \hat{x}^2
... """)
[11, 2*n + 1, True]

```

## Other notation

`\langle j_1, j_2; m_1, m_2|j_1, j_2; j, m\rangle` is a Clebsch–Gordan coefficient, as
Sakurai writes it, and the physics and braket packages' `\ket`, `\bra`, `\braket`,
`\ketbra`, `\expval` and `\mel` read as what they print.

**Next:** [Tensors and geometry](tensors.md), or [Beyond spacetime](spaces.md), where
an index ranges over the states of a qubit.
