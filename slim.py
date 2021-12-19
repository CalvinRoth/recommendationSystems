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

def softThreshold(x,y):
    if(x > y):
        return x - y
    if(x < -y):
        return x+y
    else:
        return 0

def vecvec(x,y, i):
    return np.dot(x,y) - (x[i] * y[i])



def coordinateDes(A, W, beta, i, lama):
    [m,n] = A.shape
    sol = np.zeros((n,))
    for j in range(n):
        if(j % 1000 == 0):
            print(j)
        if(i==j):
            continue
        total = 0
        for k in range(m):
            total += A[k,j] * (A[i,k] - vecvec(A[k,:], W, k))
        sol[j] = max(0, softThreshold(total, lama)/(1+beta))
    return sol


def optimSLIM(A, W, beta, lama, max_iters=100, tol=2e-2):
    iters = 0
    [m,n] = A.shape
    print("Starting")
    while score(A, W, beta, lama) > tol and iters < max_iters:
        for i in range(n):
            print(i)
            W[:, i] = coordinateDes(A, W[:,i], beta, i, lama)
        iters += 1
        print(score(A,W, beta, lama))
    return W
