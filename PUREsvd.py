import numpy as np
import numpy.linalg as lin

def pureSVD(A, d):
    [u,s,v] = lin.svd(A)
    u = u[:, 0:d]
    s = s[0:d]
    v = v[:, 0:d]
    return  u,s,v

