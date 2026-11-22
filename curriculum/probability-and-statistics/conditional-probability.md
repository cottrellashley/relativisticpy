# Conditional probability and Bayes' theorem

Probabilities conditioned on what is known, Bayes' theorem for events and for
densities, and conditional expectation, in the notation of probability books.

## A positive test for a rare disease

**Level 1** · `prob.bayes.rare-disease` · DeGroot and Schervish (2012), section 2.3

Bayes' theorem with numbers: a disease one person in 10,000 has, and a test that is 90
percent reliable. A positive result leaves a probability of only about 0.00090.

```latex
\frac{(0.9)(0.0001)}{(0.9)(0.0001) + (0.1)(0.9999)}
```

```python
prior, detection, false_alarm = Rational(1, 10000), Rational(9, 10), Rational(1, 10)
expected = detection * prior / (detection * prior + false_alarm * (1 - prior))
assert expected == Rational(9, 10008)
assert abs(float(expected) - 0.00090) < 5e-6  # the book's 0.00090
```

## The memoryless exponential distribution

**Level 2** · `prob.bayes.memoryless` · DeGroot and Schervish (2012), section 5.7

\(\Pr(X > s + t \mid X > s) = \Pr(X > t)\): the conditional probability, from the
survival function of the exponential distribution, does not depend on \(s\).

```latex
S(x) := e^{-\lambda x}
\frac{S(s + t)}{S(s)} = S(t)
```

```python
lam, s, t = symbols("lambda s t", positive=True)
assert simplify(exp(-lam * (s + t)) / exp(-lam * s) - exp(-lam * t)) == 0
expected = True
```

## Laplace's rule of succession

**Level 3** · `prob.bayes.rule-of-succession` · Wasserman (2004), section 11.2

Bayes' theorem for densities: a uniform prior on the probability \(p\) of success, the
likelihood of 7 successes in 10 trials, and the posterior mean
\(\int p\, \mathcal{L}(p) f(p)\, dp / \int \mathcal{L}(p) f(p)\, dp = (7 + 1)/(10 + 2)\).

```latex
f(p) := 1
\mathcal{L}(p) := p^{7} (1 - p)^{3}
\frac{\int_0^1 p \, \mathcal{L}(p) f(p) \, dp}{\int_0^1 \mathcal{L}(p) f(p) \, dp}
```

```python
p = symbols("p")
successes, trials = 7, 10
likelihood = p**successes * (1 - p) ** (trials - successes)
expected = integrate(p * likelihood, (p, 0, 1)) / integrate(likelihood, (p, 0, 1))
assert expected == Rational(successes + 1, trials + 2)  # the posterior mean (S + 1)/(n + 2)
```

## Bayes' theorem in the notation of conditional probability

**Level 4** · `prob.bayes.conditional-notation` · DeGroot and Schervish (2012), section 2.3

The rare disease again, with the probabilities named as the book names them,
\(\Pr(A \mid B_1)\) for the probability of a positive test \(A\) given the disease
\(B_1\).

```latex
\Pr(B_1) := 0.0001
\Pr(B_2) := 0.9999
\Pr(A \mid B_1) := 0.9
\Pr(A \mid B_2) := 0.1
\frac{\Pr(A \mid B_1) \Pr(B_1)}{\Pr(A \mid B_1) \Pr(B_1) + \Pr(A \mid B_2) \Pr(B_2)}
```

```python
expected = Rational(9, 10) * Rational(1, 10000) / (
    Rational(9, 10) * Rational(1, 10000) + Rational(1, 10) * Rational(9999, 10000)
)
assert abs(float(expected) - 0.00090) < 5e-6
```

## The law of total expectation

**Level 4** · `prob.bayes.total-expectation` · DeGroot and Schervish (2012), section 4.7

A Poisson number of trials, each a success with probability \(p\): a distribution given
conditionally on another random variable, and \(\mathbb{E}[X] =
\mathbb{E}[\mathbb{E}[X \mid N]] = \lambda p\).

```latex
N \sim \operatorname{Poisson}(\lambda)
X \mid N \sim \operatorname{Bin}(N, p)
\mathbb{E}[X]
```

```python
n = symbols("n", integer=True, nonnegative=True)
lam, p = symbols("lambda p", positive=True)
expected = simplify(summation(n * p * exp(-lam) * lam**n / factorial(n), (n, 0, oo)))
assert expected == lam * p
```
