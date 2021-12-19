import numpy as np
import numpy.linalg as lin
from time import perf_counter

def project(A):
    B  =np.zeros(A.shape)
    [m,n] = A.shape
    for i in range(m):
        for j in range(n):
            if(A[i,j] < 0 ):
                B[i,j] = 0
            if(A[i,j] > 5):
                B[i,j] = 5
            else:
                B[i,j] = A[i,j]
    return B

def pureSVD(A, d):
    [u,s,v] = lin.svd(A)
    u = u[:, 0:d]
    s = s[0:d]
    v = v[:, 0:d]

    return  (u @ np.diag(s)) , v

def loss(A, U,V):
    B = project(U@V.T)
    return np.linalg.norm(A - B, "fro")**2

def pureTest(A, dlist):
    for d in dlist:
        start =  perf_counter()
        U, V = pureSVD(A, d)
        print(U.shape, (V.T).shape)
        print("d:", d, " time", perf_counter()-start, " loss", loss(A,U,V))

