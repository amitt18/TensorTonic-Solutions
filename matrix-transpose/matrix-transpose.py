import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    rows = len(A)
    cols = len(A[0])
    res = []

    for j in range (cols):
        new_res = []
        for i in range (rows):
            new_res.append(A[i][j])
        res.append(new_res)
        
    x = np.array(res)
    return x
    
