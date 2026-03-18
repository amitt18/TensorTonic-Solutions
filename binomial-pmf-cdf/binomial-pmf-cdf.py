import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    # Edge cases
    if p == 0:
        return (1.0 if k == 0 else 0.0,
                1.0 if k >= 0 else 0.0)

    if p == 1:
        return (1.0 if k == n else 0.0,
                1.0 if k >= n else 0.0)

    # PMF
    pmf = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

    # CDF (vectorized)
    i = np.arange(0, k + 1)
    cdf = np.sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i)))

    return float(pmf), float(cdf)