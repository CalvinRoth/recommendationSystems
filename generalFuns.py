import numpy as np
import pandas as pd
import ALS
import PUREsvd
import slim
from time import perf_counter

def readData(path):
    df = pd.read_csv(path)
    n_users = max(df["userId"])
    n_movies = max(df["movieId"])
    A = np.zeros((n_users, n_movies))
    for index, row in df.iterrows():
        i = int(row["userId"]) - 1
        j = int(row["movieId"]) - 1
        if (row["rating"] >= 3):
            A[i, j] = 1
    A = A[:, ~np.all(A == 0, axis=0)]
    A = A[~np.all(A == 0, axis=1), :]
    return A

def readDataRaw(path):
    df = pd.read_csv(path)
    n_users = max(df["userId"])
    n_movies = max(df["movieId"])
    A = np.zeros((n_users, n_movies))
    for index, row in df.iterrows():
        i = int(row["userId"]) - 1
        j = int(row["movieId"]) - 1
        A[i, j] = 1
    A = A[:, ~np.all(A == 0, axis=0)]
    A = A[~np.all(A == 0, axis=1), :]
    return A

def partitionData(A):
    [n, _] = A.shape
    B = np.copy(A)
    preds = np.zeros((n,))
    for user in range(n):
        choices = np.where(A[user, :] > 3)[0]
        i = np.random.choice(np.nonzero(A[user, :])[0])
        preds[user] = i
        B[user, i] = 0
    return B, preds


def sortedRecs(A, predictMat, k):
    [n_users, n_items] = A.shape
    predicts = np.zeros((n_users, n_items))
    for user in range(n_users):
        v = np.where(A[user, :] == 0)
        vals = []
        for item in range(n_items):
            if item in v[0]:
                vals.append((item, predictMat[user, item]))
        v = sorted(vals, key=lambda x: x[1], reverse=True)
        row = np.array([i[0] for i in v])
        predicts[user, :] = row
    return predicts


def hitrate(trueMat, left_out, preds):
    n_users = trueMat.shape[0]
    hits = 0
    for i in range(n_users):
        if (left_out in preds[i, :]):
            hits += 1
    return hits / n_users




def ALStes():
    start = perf_counter()
    [u, s, v] = np.linalg.svd(training)
    total = np.sum(s)
    thres70 = total * 0.7
    thres90 = total * 0.9
    thres99 = total * 0.99
    count70 = 0

    total = 0
    i = 0
    while (total < thres70):
        total += s[i]
        count70 += 1
        i += 1

    count90 = count70
    while (total < thres90):
        total += s[i]
        count90 += 1
        i += 1
    count99 = count90
    while (total < thres99):
        total += s[i]
        count99 += 1
        i += 1
    stop = perf_counter() - start
    print("Time to computer Sizes", stop)
    start = perf_counter()
    U, V = ALS.ALS(training, count70, 1)
    loss1 = ALS.loss(A, U, V, 1)
    stop = perf_counter() - start
    print("Time for first ALS", stop, " Size", count70, " Loss", loss1)
    start = perf_counter()
    U, V = ALS.ALS(training, count90, 1)
    loss2 = ALS.loss(A, U, V, 1)
    stop = perf_counter() - start
    print("Time for 2nd trial", stop, " Size", count90, " Loss", loss2)
    start = perf_counter()
    U, V = ALS.ALS(training, count99, 1)
    loss3 = ALS.loss(A, U, V, 1)
    stop = perf_counter() - start
    print("Time for 3rd trial", stop, " Size", count99, " Loss", loss3)
    return (loss1, loss2, loss3)


path = "data/ratings.csv"
## Test for ALS
A = readData(path)
#training, left_out = partitionData(A)

# uncomment me to run test
#print(ALStes())


## Pure SVD
A = readDataRaw(path)
training, left_out = partitionData(A)
#
#PUREsvd.pureTest(A, [252,478, 578])
U,V = PUREsvd.pureSVD(training, 200)



