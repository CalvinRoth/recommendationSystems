import numpy as np
from numpy.linalg import norm


def matfro1(W):
    sum = 0
    for row in W:
        for col in row:
            sum += abs(sum)
    return sum


def score(A, W, beta, lama):
    t1 = 0.5 * norm(A - (A @ W), "fro") ** 2
    t2 = 0.5 * beta * norm(W, "fro") ** 2
    t3 = lama * matfro1(W)
    return t1 + t2 + t3


def coordinateDes(A, W, beta, i, lama):
    [n, m] = A.shape
    sol = np.zeros((n, 1))
    factor = np.linalg.inv(A + np.eye(n, m) * beta)
    for j in range(n):
        if (i == j):
            continue
        if (A(j, i) > lama):
            sol[j] = factor * (A(j, i) - lama)
        elif (abs(A(j, i)) < lama):
            sol[j] = 0
        else:  # Project to domain
            sol[j] = 0
    return 0


def optimSLIM(A, W, beta, lama, max_iters, tol):
    iters = 0
    [n, m] = A.shape
    while score(A, W, beta, lama) > tol and iters < max_iters:
        for i in range(m):
            W[:, i] = coordinateDes(A, W, beta, i, lama)
        iters += 1
    return W



def getRecs(A,W, k):
    [n, m] = A.shape
    # Sort and get top k of w_i that aren't in A[i]