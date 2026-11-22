# Complex numbers

Arithmetic with \(i\), the conjugate, Euler's formula and the roots of unity. Ahlfors (1979)
is *Complex Analysis*, 3rd edition.

## Arithmetic with i

**Level 1** · `complex.numbers.arithmetic` · Ahlfors (1979), ch. 1, §1.1, exercise 1

A quotient of complex numbers, squared.

```latex
\left( \frac{2 + i}{3 - 2i} \right)^2
```

```python
expected = expand(((2 + I) / (3 - 2 * I))**2)
assert expected == Rational(-33, 169) + Rational(56, 169) * I
```

## The conjugate

**Level 1** · `complex.numbers.conjugate` · Ahlfors (1979), ch. 1, §1.4

The conjugate written as a bar over a whole expression, \(\overline{z w}\).

```latex
\overline{(2 - 3i)(1 + i)}
```

```python
expected = conjugate(expand((2 - 3 * I) * (1 + I)))
assert expected == 5 + I
```

## Euler's formula

**Level 1** · `complex.numbers.euler` · Ahlfors (1979), ch. 2, §3.2

The exponential of an imaginary number, \(e^{i\theta}\), equals \(\cos\theta + i\sin\theta\).

```latex
e^{i\theta} = \cos\theta + i\sin\theta
```

```python
theta = symbols("theta")
assert simplify((exp(I * theta) - cos(theta) - I * sin(theta)).rewrite(exp)) == 0
expected = True
```

## The roots of unity

**Level 2** · `complex.numbers.roots-of-unity` · Ahlfors (1979), ch. 1, §2.2

The fifth roots of unity are the powers of \(\omega = e^{2\pi i/5}\), the vertices of a
regular pentagon; their sum is 0.

```latex
\omega := e^{2\pi i/5}
\sum_{k=0}^{4} \omega^k
```

```python
omega = exp(2 * pi * I / 5)
expected = simplify((omega**5 - 1) / (omega - 1))  # a geometric sum, with omega^5 = 1
assert expected == 0
assert abs(N(sum(omega**k for k in range(5)))) < 1e-15
```
