import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    param = np.array(param)
    grad = np.array(grad)
    m = np.array(m)
    v = np.array(v)
    mnew = beta1*m + (1-beta1)*grad
    vnew = beta2*v +(1-beta2)* (grad**2)
    mhat = mnew/(1-beta1**t)
    vhat = vnew/(1-beta2**t)
    pnew = param-lr*mhat/(np.sqrt(vhat)+eps)
    return (pnew,mnew,vnew)
    
    pass