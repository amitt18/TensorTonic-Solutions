import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    Return None for invalid input.
    """
    X = np.array(X)

    # need at least 2 samples
    if X.ndim!=2 or X.shape[0] < 2:
        return None

    cov = np.cov(X, rowvar=False)
    return np.atleast_2d(cov)