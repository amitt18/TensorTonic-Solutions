import numpy as np

def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Perform one Nadam update step.
    """
    w = np.array(w)
    m = np.array(m)
    v = np.array(v)
    grad = np.array(grad)

    # update moments
    mt = beta1 * m + (1 - beta1) * grad
    vt = beta2 * v + (1 - beta2) * (grad ** 2)

    # Nadam Nesterov term
    m_nadam = beta1 * mt + (1 - beta1) * grad

    # parameter update
    wt = w - lr * m_nadam / (np.sqrt(vt) + eps)

    return wt, mt, vt