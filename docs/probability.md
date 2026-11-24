# Random variables

> **For:** anyone who computes probabilities. **You will learn:** how to declare a random
> variable, and to write probabilities, expectations and variances as books do.
> **Before this:** [The language](language.md).

```python
>>> from relativisticpy import Workbook

```

## Declaring a random variable

`X \sim D` declares a random variable, as books write it: `\sim` between a name nothing
defines and a distribution, for the rest of the program. It shows nothing, and a note says
what it declared. Each distribution is written in the forms books give it, which say what its
arguments are: `\mathcal{N}(\mu, \sigma^2)` and `N(\mu, \sigma^2)` take the variance, as
most books write them, `\Gamma(k, \theta)` the shape and the scale, and
`\operatorname{Exp}(\lambda)` the rate. Called by its name in plain text, a distribution takes
its parameters as SymPy and SciPy do, `Normal(mu, sigma)` the standard deviation. The
[reference](reference.md#random-variables) lists them, with the binomial, Poisson, Bernoulli,
uniform, gamma, χ², Student's t, beta and geometric distributions.

## Probabilities and expectations

`\Pr(A)`, `\mathbb{P}(A)` and `P(X = 2)` are probabilities, `\mathbb{E}[X]` an expectation,
and `\operatorname{Var}(X)` and `\operatorname{Cov}(X, Y)` the variance and the covariance;
inside them `\mid` is the conditional bar. P alone, a pressure elsewhere, is a probability
only of an event of random variables, and E an expectation only of them. Without brackets,
`\mathbb{E}` and `\operatorname{Var}` act on the product written after them, as
`\operatorname{tr}` does, so that `\operatorname{Var} 3Z` is the variance of 3Z. SymPy
computes each exactly; where it finds no closed form in 5 seconds and the problem has no
parameters, SciPy computes it numerically, and a note says so:

```python
>>> rv = Workbook()
>>> rv.exe(r"""
...     N \sim \operatorname{Poisson}(\lambda)
...     P(N = 2)
...     Z \sim \mathcal{N}(0, 1)
...     \Pr(Z > 1)
...     \operatorname{Var}(3Z + 1)
...     \mathbb{E}[Z^4]
...     \operatorname{Var} 3Z
... """)
[lambda**2*exp(-lambda)/2, 1/2 - erf(sqrt(2)/2)/2, 9, 3, 9]

```

## Laws given by other random variables

A law may be given by another random variable, as a Poisson number of trials gives a
binomial number of successes, or for each value another takes. Given a random variable, an
expectation is a function of it, and the laws of total expectation and probability give the
rest:

```python
>>> rv.exe(r"""
...     X \mid N \sim \operatorname{Bin}(N, p)
...     \mathbb{E}[X \mid N]
...     \mathbb{E}[X]
...     \operatorname{Var}(X)
...     Y \sim \operatorname{Bernoulli}(1/2)
...     W \mid Y = 0 \sim \mathcal{N}(-1, 1)
...     W \mid Y = 1 \sim \mathcal{N}(1, 1)
...     \Pr(W > 0)
... """)
[p*N, lambda*p, lambda*p, 1/2]

```

## Samples and independence

Random variables declared in one statement with `\overset{\text{iid}}{\sim}`, or with
`\text{independent}` after the distribution, are independent; declared apart, they are taken
independent, and a note says so. `X_1, \ldots, X_n \overset{\text{iid}}{\sim} D` declares a
sample of n, whose members `X_i` a sum takes in turn. A capital letter nothing defines,
inside a probability, an expectation or a variance, is a random variable of a law not given,
as books write one, so that a rule that holds for every law is decided. The probabilities a
program gives events, as `\Pr(A \mid B_1) := 0.9`, are what it writes of them; an event of
a random variable it declares takes its probability from the law, and giving it another is
refused, as is defining a name declared a random variable:

```python
>>> sample = Workbook()
>>> sample.exe(r"""
...     X_1, \ldots, X_n \overset{\text{iid}}{\sim} \mathcal{N}(\mu, \sigma^2)
...     \bar{X} := \frac{1}{n} \sum_{i=1}^{n} X_i
...     \mathbb{E}[\bar{X}]
...     \operatorname{Var}(\bar{X})
...     \operatorname{Var}(a Y + b) = a^2 \operatorname{Var}(Y)
...     \Pr(B) := 0.01
...     \Pr(A \mid B) := 0.9
...     \Pr(A \mid B) \Pr(B)
... """)
[mu, sigma**2/n, True, 0.00900000000000000]

```

## Events of several variables

Events of several random variables that share one are not independent, and are not
computed as if they were. Of discrete variables, an event is summed over the values one of
them can take where it holds, and a variable an event gives a value takes it in the others.
What SymPy gives of several continuous variables together is kept only where a simulation
bears it out, and refused otherwise, saying so:

```python
>>> counts = Workbook()
>>> counts.exe(r"""
...     X \sim \operatorname{Poisson}(2)
...     Y \sim \operatorname{Poisson}(3)
...     P(X = 1 \mid X + Y = 3)
...     P(X = 1, X + Y = 3)
...     \mathbb{E}[X \mid X + Y = 3]
... """)
[54/125, 9*exp(-5), 6/5]

```

## Densities and generating functions

`f_X(x)`, `F_X(x)` and `F_X^{-1}(q)` are X's density, distribution function and quantile, and
`M_X(t)` and `\varphi_X(t)` its moment-generating and characteristic functions. Where a
probability holds only for some values of a parameter the program says nothing of, it is
given where the parameter is among the values the variable takes, as a book gives it, and a
note says so:

```python
>>> rv.exe(r"""
...     T \sim \operatorname{Exp}(\lambda)
...     \Pr(T > t)
...     f_Z(x)
...     F_Z^{-1}(0.975)
...     M_N(s)
... """)
[exp(-lambda*t), sqrt(2)*exp(-x**2/2)/(2*sqrt(pi)), 1.95996398454005, exp(lambda*(exp(s) - 1))]

```

A moment-generating function is the expectation of an exponential only where it is finite,
as where `t < \lambda` of the exponential distribution: past it the expectation is ∞, and of
a symbol a note says what is assumed. A mean or a variance that does not exist, as of
Student's t with one degree of freedom, Cauchy's law, is refused, and its second moment is ∞:

```python
>>> [step.assuming for step in rv.each("M_T(s)")]
[['s < λ']]
>>> rv.exe(r"""
...     V \sim \operatorname{Exp}(1)
...     \mathbb{E}[e^{V/2}]
...     \mathbb{E}[e^{2V}]
...     C \sim t_1
...     \mathbb{E}[C^2]
... """)
[2, oo, oo]

```

## Over data

Over data, `\mathbb{E}_t`, `\operatorname{Var}_t` and `\operatorname{Cov}_t` are statistics
along an index, as [Beyond spacetime](spaces.md#statistics) says; without an index, of data,
they are refused, saying so.

**Next:** [Optima](optima.md), or [Beyond spacetime](spaces.md#statistics), for
statistics of data.
