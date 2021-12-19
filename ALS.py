import numpy as np
import numpy.linalg as lin


def updateU(R, V, lama):
    [n, d] = V.shape
    [m, n] = R.shape
    U = np.zeros((m, d))
    for i in range(m):
        U[i, :] = R[i, :] @ V @ lin.inv(V.T @ V + lama * np.eye(d, d))
    return U


def updateV(R, U, lama):
    [m, d] = U.shape
    [m, n] = R.shape
    V = np.zeros((n, d))
    for i in range(n):
        V[i,:] = R[:,i]@  U @ lin.inv(U.T @ U + lama * np.eye(d, d))
    return V

def loss(A, U,V, lama):
    L = lin.norm(A - (U@V.T), "fro")**2
    L += lama * ( lin.norm(U, "fro")**2 + lin.norm(V, "fro")**2)
    return L


def ALS(R, d, lama, iters=100):
    [m, n] = R.shape
    U = 0.01 * np.random.randn(m, d)
    V = np.zeros((n, d))
    old = 0
    loss = 0
    for i in range(iters):
        V = updateV(R, U, lama)
        U = updateU(R, V, lama)
        old = loss
        loss = lin.norm(R - U @ (V.T))
        dif = abs(loss - old)
        if dif < 2e-2:
            break

    return U, V


