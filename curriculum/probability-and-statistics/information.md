# Entropy and divergence

Shannon's entropy and the Kullback–Leibler divergence, or relative entropy, in bits, as
information theory writes them.

## The entropy of a distribution

**Level 1** · `prob.information.entropy` · Cover and Thomas (2006), example 2.1.2

The entropy of a variable taking four values with probabilities \(\frac{1}{2},
\frac{1}{4}, \frac{1}{8}, \frac{1}{8}\), written term by term: \(\frac{7}{4}\) bits.

```latex
-\frac{1}{2} \log_2 \frac{1}{2} - \frac{1}{4} \log_2 \frac{1}{4} - \frac{1}{8} \log_2 \frac{1}{8} - \frac{1}{8} \log_2 \frac{1}{8}
```

```python
from scipy.stats import entropy

probabilities = [Rational(1, 2), Rational(1, 4), Rational(1, 8), Rational(1, 8)]
expected = simplify(-sum(p * log(p, 2) for p in probabilities))
assert expected == Rational(7, 4)
assert abs(entropy([0.5, 0.25, 0.125, 0.125], base=2) - 1.75) < 1e-12
```

## The relative entropy of two coins

**Level 1** · `prob.information.kl-bernoulli` · Cover and Thomas (2006), example 2.3.1

\(D(p \| q)\) for a fair coin \(p\) against a coin \(q\) that shows heads one time in
four, written out: \(1 - \frac{1}{2} \log_2 3 = 0.2075\) bits.

```latex
\frac{1}{2} \log_2 \frac{1/2}{3/4} + \frac{1}{2} \log_2 \frac{1/2}{1/4}
```

```python
from scipy.stats import entropy

expected = 1 - log(3) / (2 * log(2))
assert abs(float(expected) - entropy([0.5, 0.5], [0.75, 0.25], base=2)) < 1e-12
assert abs(float(expected) - 0.2075) < 1e-4  # the book's value
```

## The most uncertain coin

**Level 2** · `prob.information.binary-entropy-maximum` · Cover and Thomas (2006), example 2.1.1

The binary entropy function is greatest where its derivative vanishes: at \(p = 1/2\).

```latex
H(p) := -p \log_2 p - (1 - p) \log_2 (1 - p)
solve(H'(p) = 0, p)
```

```python
p = symbols("p")
H = -p * log(p, 2) - (1 - p) * log(1 - p, 2)
expected = solve(diff(H, p), p)
assert expected == [Rational(1, 2)]
assert H.subs(p, Rational(1, 2)) == 1  # one bit
```

## The relative entropy in the notation of information theory

**Level 4** · `prob.information.kl-notation` · Cover and Thomas (2006), section 2.3

The same two coins, as two distributions, and their relative entropy written
\(D(p \| q)\), whose definition is \(\sum_x p(x) \log \frac{p(x)}{q(x)}\).

```latex
p := \left[\frac{1}{2}, \frac{1}{2}\right]
q := \left[\frac{3}{4}, \frac{1}{4}\right]
D(p \| q)
```

```python
from scipy.stats import entropy

p, q = [Rational(1, 2), Rational(1, 2)], [Rational(3, 4), Rational(1, 4)]
expected = simplify(sum(a * log(a / b, 2) for a, b in zip(p, q)))
assert simplify(expected - (1 - log(3, 2) / 2)) == 0
assert abs(float(expected) - entropy([0.5, 0.5], [0.75, 0.25], base=2)) < 1e-12
```
