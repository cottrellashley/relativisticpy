# Commutators and the uncertainty principle

Operators that do not commute: position and momentum, with momentum a derivative, the
components of angular momentum, and operator algebra from the canonical commutator alone.
Their commutators bound the product of uncertainties, and a Gaussian meets the bound.

## The canonical commutator

**Level 2** · `qm.commutators.canonical` · Griffiths (2005), section 2.3.1

\([x, p] = i\hbar\), with the momentum operator a derivative, shown by acting on a test
function as Griffiths does.

```latex
\hat{p} := \frac{\hbar}{i} \frac{d}{dx}
[x, \hat{p}] f(x) = i\hbar f(x)
```

```python
x, hbar = symbols("x hbar")
f = Function("f")
p = lambda g: hbar / I * diff(g, x)
commutator = x * p(f(x)) - p(x * f(x))
assert simplify(commutator - I * hbar * f(x)) == 0
expected = True
```

## The algebra of angular momentum

**Level 3** · `qm.commutators.angular-momentum` · Griffiths (2005), section 4.3.1

The components of orbital angular momentum as differential operators, composed:
\([L_x, L_y] = i\hbar L_z\).

```latex
L_x := -i\hbar \left( y \frac{\partial}{\partial z} - z \frac{\partial}{\partial y} \right)
L_y := -i\hbar \left( z \frac{\partial}{\partial x} - x \frac{\partial}{\partial z} \right)
L_z := -i\hbar \left( x \frac{\partial}{\partial y} - y \frac{\partial}{\partial x} \right)
[L_x, L_y] = i\hbar L_z
```

```python
x, y, z, hbar = symbols("x y z hbar")
f = Function("f")(x, y, z)  # a test function
L_x = lambda g: -I * hbar * (y * diff(g, z) - z * diff(g, y))
L_y = lambda g: -I * hbar * (z * diff(g, x) - x * diff(g, z))
L_z = lambda g: -I * hbar * (x * diff(g, y) - y * diff(g, x))
assert simplify(L_x(L_y(f)) - L_y(L_x(f)) - I * hbar * L_z(f)) == 0
expected = True
```

## A commutator from the canonical one

**Level 4** · `qm.commutators.power-of-x` · Griffiths (2005), Problem 3.13

Operator algebra without a representation: from \([\hat x, \hat p] = i\hbar\) alone,
\([\hat x^3, \hat p] = 3i\hbar \hat x^2\).

```latex
[\hat{x}, \hat{p}] = i\hbar
[\hat{x}^3, \hat{p}] = 3i\hbar\hat{x}^2
```

```python
from sympy.physics.quantum import Commutator, Operator

hbar = symbols("hbar")
X, P = Operator("x"), Operator("p")
# [x^3, p] = x^2 [x, p] + x [x, p] x + [x, p] x^2, then [x, p] = i hbar.
expanded = Commutator(X**3, P).expand(commutator=True)
reduced = expanded.subs(Commutator(P, X), -I * hbar).subs(Commutator(X, P), I * hbar)
assert simplify(reduced - 3 * I * hbar * X**2) == 0
# The same in the position representation, p = -i hbar d/dx, on a test function.
x = symbols("x")
f = Function("f")(x)
p = lambda g: -I * hbar * diff(g, x)
assert simplify(x**3 * p(f) - p(x**3 * f) - 3 * I * hbar * x**2 * f) == 0
expected = True
```

## The minimum-uncertainty wave packet

**Level 3** · `qm.commutators.gaussian-packet` · Griffiths (2005), section 3.5.2

The expectation values of \(x^2\) and \(p^2\) in a Gaussian, the second with momentum as
a derivative: their product meets the bound of the uncertainty principle,
\(\sigma_x \sigma_p = \hbar/2\).

```latex
\psi(x) := \left(\frac{2a}{\pi}\right)^{1/4} e^{-ax^2}
\langle x^2 \rangle := \int_{-\infty}^{\infty} x^2 |\psi(x)|^2 \, dx
\langle p^2 \rangle := \int_{-\infty}^{\infty} \psi^*(x) \left(-\hbar^2 \frac{d^2}{dx^2}\right) \psi(x) \, dx
\sqrt{\langle x^2 \rangle \langle p^2 \rangle}
```

```python
x = symbols("x", real=True)
a, hbar = symbols("a hbar", positive=True)
psi = (2 * a / pi) ** Rational(1, 4) * exp(-a * x**2)
assert integrate(psi**2, (x, -oo, oo)) == 1
# <x> and <p> vanish, so the uncertainties are the roots of <x^2> and <p^2>.
assert integrate(x * psi**2, (x, -oo, oo)) == 0
assert integrate(psi * (-I * hbar) * diff(psi, x), (x, -oo, oo)) == 0
x2 = integrate(x**2 * psi**2, (x, -oo, oo))
p2 = integrate(psi * (-(hbar**2)) * diff(psi, x, 2), (x, -oo, oo))
assert simplify(sqrt(x2 * p2) - hbar / 2) == 0
expected = hbar / 2
```
