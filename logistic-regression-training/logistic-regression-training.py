import numpy as np

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    
    X = np.array(X)
    y = np.array(y)

    N, D = X.shape

    # initialize parameters
    w = np.zeros(D)
    b = 0.0

    for _ in range(steps):

        # linear combination
        z = X @ w + b

        # sigmoid
        p = 1 / (1 + np.exp(-z))

        # gradients
        dw = (1/N) * (X.T @ (p - y))
        db = (1/N) * np.sum(p - y)

        # update
        w = w - lr * dw
        b = b - lr * db

    return w, b