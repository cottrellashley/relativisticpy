r"""A portfolio of 500 assets, in index notation.

Ten years of daily returns on 500 assets, driven by five hidden factors and noise,
are bound to the tensor R_{ta}: t ranges over 2,520 days and a over 500 assets. From
there everything is notation, as it would be written on a page:

- the expected returns and the covariance, as expectations over the days;
- the tangency portfolio of Markowitz, computed with linear solves, never an inverse;
- risk as geometry, with the covariance as the metric of the assets;
- the spectrum of the correlation matrix against the Marchenko–Pastur law of random
  matrices, which tells the five real factors from the noise.

    python examples/portfolio.py

This needs NumPy and SciPy: pip install "relativisticpy[numeric]".
"""

import numpy as np

from relativisticpy import Workbook

DAYS, ASSETS, FACTORS = 2520, 500, 5

rng = np.random.default_rng(7)
factors = rng.normal(0, 0.01, (DAYS, FACTORS))
loadings = rng.normal(0, 1, (FACTORS, ASSETS))
loadings /= np.linalg.norm(loadings, axis=0)  # every asset equally exposed
returns = 0.0004 + 0.6 * factors @ loadings + rng.normal(0, 0.01, (DAYS, ASSETS))

wb = Workbook()
wb["R_{ta}"] = returns
wb.exe(r"""
    \mu_a := \mathbb{E}_t[R_{ta}]
    \Sigma_{ab} := \mathbb{E}_t[(R_{ta} - \mu_a)(R_{tb} - \mu_b)]
    m^a := \frac{(\Sigma^{-1})^{ab} \mu_b}{1_c (\Sigma^{-1})^{cd} \mu_d}
""")
tangency = wb["m"].to_numpy()
print(f"the tangency portfolio's weights sum to {tangency.sum():.6f}")

# Signal and noise: the spectrum of the correlation matrix.
wb["Z_{ta}"] = (returns - returns.mean(axis=0)) / returns.std(axis=0)
wb.exe(r"C_{ab} := \mathbb{E}_t[Z_{ta} Z_{tb}]")
spectrum = wb.expr(r"\operatorname{spec}(C^a{}_b)").to_numpy()
edge = (1 + np.sqrt(ASSETS / DAYS)) ** 2  # the largest eigenvalue of pure noise
real = int((spectrum > edge).sum())
print(f"{real} eigenvalues stand above the Marchenko–Pastur edge of {edge:.2f}")

# Risk as geometry: with the covariance as the metric of the assets, w^a w_a is a
# variance. (Raising an index of an asset uses the metric from now on.)
wb["w^a"] = np.full(ASSETS, 1 / ASSETS)
wb.exe(r"g_{ab} := \Sigma_{ab}")
variance = wb.expr(r"w^a w_a")
print(f"the variance of the equally weighted portfolio is {variance:.3e}")
