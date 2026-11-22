# Estimation and regression

Estimators computed from data given inline, maximum likelihood, least squares by the
normal equations, and the correlation matrix of a covariance matrix, in the notation of
statistics books.

## The sample variance

**Level 2** · `prob.estimation.sample-variance` · standard

The unbiased sample variance of eight observations, with the sample mean \(\bar{x}\) and
the observations \(x_i\) written as statistics writes them.

```latex
x := [2, 4, 4, 4, 5, 5, 7, 9]
n := 8
\bar{x} := \frac{1}{n} \sum_{i=1}^{n} x_i
\frac{1}{n - 1} \sum_{i=1}^{n} (x_i - \bar{x})^2
```

```python
data = [2, 4, 4, 4, 5, 5, 7, 9]
mean = Rational(sum(data), len(data))
expected = sum((value - mean) ** 2 for value in data) / (len(data) - 1)
assert expected == Rational(32, 7)
assert abs(float(expected) - np.var(data, ddof=1)) < 1e-12
```

## Least squares by the normal equations

**Level 2** · `prob.estimation.normal-equations` · Hastie, Tibshirani and Friedman (2009), eq. (3.6)

\(\hat\beta = (X^T X)^{-1} X^T y\) for a straight line through four points, with the
design matrix and the responses written out and the transpose written \(X^T\): the line
\(y = 3.5 + 1.4 x\).

```latex
X := \begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \\ 1 & 4 \end{pmatrix}
y := \begin{pmatrix} 6 \\ 5 \\ 7 \\ 10 \end{pmatrix}
(X^T X)^{-1} X^T y
```

```python
X = Matrix([[1, 1], [1, 2], [1, 3], [1, 4]])
y = Matrix([6, 5, 7, 10])
expected = (X.T * X).inv() * X.T * y
assert expected == Matrix([[Rational(7, 2)], [Rational(7, 5)]])
fit = np.linalg.lstsq(np.array(X.tolist(), float), np.array(y.tolist(), float).ravel(), rcond=None)
assert np.allclose(fit[0], [3.5, 1.4])
```

## The maximum likelihood estimator of an exponential rate

**Level 3** · `prob.estimation.mle-exponential` · DeGroot and Schervish (2012), section 7.5

The likelihood of \(n\) exponential observations whose sum is \(y\), its logarithm, and
the stationary point of the log-likelihood: \(\hat\theta = n / y\), one over the sample
mean.

```latex
L(\theta) := \theta^{n} e^{-\theta y}
\ell(\theta) := \ln L(\theta)
solve(\frac{d\ell}{d\theta} = 0, \theta)
```

```python
theta, n, y = symbols("theta n y", positive=True)
expected = solve(diff(log(theta**n * exp(-theta * y)), theta), theta)
assert expected == [n / y]
```

## The correlation matrix of a covariance matrix

**Level 2** · `prob.estimation.correlation-matrix` · Johnson and Wichern (2007), example 2.14

\(\rho = (V^{1/2})^{-1} \Sigma (V^{1/2})^{-1}\), where \(V^{1/2}\) is the square root of
the diagonal matrix of variances: the standard deviations on the diagonal.

```latex
\Sigma := \begin{pmatrix} 4 & 1 & 2 \\ 1 & 9 & -3 \\ 2 & -3 & 25 \end{pmatrix}
V := \operatorname{diag}(4, 9, 25)
\left(V^{1/2}\right)^{-1} \Sigma \left(V^{1/2}\right)^{-1}
```

```python
Sigma = Matrix([[4, 1, 2], [1, 9, -3], [2, -3, 25]])
root = diag(*[sqrt(Sigma[i, i]) for i in range(3)])
expected = root.inv() * Sigma * root.inv()
assert expected == Matrix([[30, 5, 6], [5, 30, -6], [6, -6, 30]]) / 30  # the book's rho
```
