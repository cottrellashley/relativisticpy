# Principal components and clustering

Unsupervised learning on tables from Python: the share of the variance the first
principal component explains, as an eigenvalue of the sample covariance, and k-means, its
centroids and its objective, in index notation over points, clusters and coordinates.

## The variance explained by the first principal component

**Level 2** · `ml.pca.explained-variance` · Jolliffe (2002), chapter 6

Eight objects measured twice, read from a CSV file into a pandas DataFrame. The first
principal component is the eigenvector of the sample covariance \(S\) with the largest
eigenvalue, and it explains the fraction \(\lambda_{\max}(S)/\operatorname{tr} S = 0.9894\) of
the total variance.

```python-scope
import pandas as pd

measurements = pd.read_csv("tables/measurements.csv")
```

```latex
X_{ta} := \texttt{measurements}
S_{ab} := \operatorname{Cov}_t(X_{ta}, X_{tb})
\frac{\lambda_{\max}(S)}{\operatorname{tr} S}
```

```python
import csv

with open("tables/measurements.csv") as file:
    data = np.array([[float(v) for v in row] for row in list(csv.reader(file))[1:]])
covariance = np.cov(data.T)
eigenvalues = np.linalg.eigvalsh(covariance)
expected = float(eigenvalues.max() / np.trace(covariance))
singular = np.linalg.svd(data - data.mean(axis=0), compute_uv=False)  # the same from the SVD
assert abs(singular[0] ** 2 / np.sum(singular**2) - expected) < 1e-12
assert abs(expected - 0.98944) < 5e-6
```

## The centroids of k-means

**Level 2** · `ml.kmeans.centroids` · Bishop (2006), section 9.1

Given the assignments \(r_{nk}\), one when point \(n\) is in cluster \(k\), each centroid is the
mean of its points, \(\mu_{ka} = \sum_n r_{nk} x_{na} / \sum_n r_{nk}\): \((\frac{4}{3},
\frac{4}{3})\) and \((\frac{19}{3}, \frac{20}{3})\). The points and the assignments come from
Python.

```python-scope
import pandas as pd

table = pd.read_csv("tables/points.csv")
points = table[["x", "y"]]
assignments = pd.get_dummies(table["cluster"]).astype(int)
```

```latex
x_{na} := \texttt{points}
r_{nk} := \texttt{assignments}
\mu_{ka} := \frac{\sum_n r_{nk} x_{na}}{\sum_n r_{nk}}
\mu_{ka}
```

```python
import csv

with open("tables/points.csv") as file:
    rows = [[int(v) for v in row] for row in list(csv.reader(file))[1:]]
expected = [
    [
        Rational(sum(r[c] for r in rows if r[2] == k), sum(1 for r in rows if r[2] == k))
        for c in (0, 1)
    ]
    for k in (0, 1)
]
assert expected == [[Rational(4, 3), Rational(4, 3)], [Rational(19, 3), Rational(20, 3)]]
points = np.array([r[:2] for r in rows], float)
labels = np.array([r[2] for r in rows])
assert np.allclose([points[labels == k].mean(axis=0) for k in (0, 1)], np.array(expected, float))
```

## The objective of k-means

**Level 3** · `ml.kmeans.objective` · Bishop (2006), section 9.1

The distortion \(J = \sum_n \sum_k r_{nk} \|\mathbf{x}_n - \boldsymbol{\mu}_k\|^2\), with the
squared distance written out over the coordinates: \(\frac{4}{3} + \frac{10}{3} = \frac{14}{3}\)
for these two clusters at their centroids.

```python-scope
import pandas as pd

table = pd.read_csv("tables/points.csv")
points = table[["x", "y"]]
assignments = pd.get_dummies(table["cluster"]).astype(int)
```

```latex
x_{na} := \texttt{points}
r_{nk} := \texttt{assignments}
\mu_{ka} := \frac{\sum_n r_{nk} x_{na}}{\sum_n r_{nk}}
J := \sum_n \sum_k r_{nk} \sum_a (x_{na} - \mu_{ka})^2
J
```

```python
import csv

with open("tables/points.csv") as file:
    rows = [[int(v) for v in row] for row in list(csv.reader(file))[1:]]
centroids = {}
for k in (0, 1):
    members = [r for r in rows if r[2] == k]
    centroids[k] = [Rational(sum(r[c] for r in members), len(members)) for c in (0, 1)]
expected = sum((r[0] - centroids[r[2]][0]) ** 2 + (r[1] - centroids[r[2]][1]) ** 2 for r in rows)
assert expected == Rational(14, 3)
points = np.array([r[:2] for r in rows], float)
labels = np.array([r[2] for r in rows])
within = sum(np.sum((points[labels == k] - points[labels == k].mean(axis=0)) ** 2) for k in (0, 1))
assert abs(within - 14 / 3) < 1e-12
```
