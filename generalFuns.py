import numpy as np
import pandas as pd
import ALS
import PUREsvd
import slim


def readData():
    df = pd.read_csv("~/classes/optim/recommendationSystems/data/ratings.csv")
    n_users = max(df["userId"])
    n_movies = max(df["movieId"])
    A = np.zeros((n_users, n_movies))
    for index, row in df.iterrows():
        i = int(row["userId"]) - 1
        j = int(row["movieId"]) - 1
        if (row["rating"] >= 3):
            A[i, j] = 1
    A = A[:, ~np.all(A == 0, axis=0)]
    A = A[~np.all(A==0, axis=1), :]
    return A


def partitionData(A):
    [n, _] = A.shape
    B = np.copy(A)
    preds = np.zeros((n,))
    for user in range(n):
        choices = np.nonzero(A[user,:])[0]
        i = np.random.choice(np.nonzero(A[user, :])[0] )
        preds[user] = i
        B[user, i] = 0
    return B, preds



def sortedRecs(A, predictMat, k):
    [n_users, n_items] = A.shape
    predicts = np.zeros((n_users, k))
    for user in range(n_users):
        v = np.where(A[n_users,:] == 0)
        vals = []
        for item in range(n_items):
            if item in v:
                vals.append((item, predictMat[user, item]))
        predicts[user, :] = sorted(vals, key = lambda x : x[1], reverse=True)[0:k]
    return predicts


def hitrate(trueMat, left_out, preds):
    n_users = trueMat.shape[0]
    hits = 0
    for i in range(n_users):
        if (left_out in preds[i, :]):
            hits += 1
    return hits / n_users


A = readData()
training, left_out = partitionData(A)

def ALStes():
    [u,s,v] = np.linalg.svd(training)
    total = np.sum(s)
    thres70 = total * 0.7
    thres90 = total * 0.9
    thres99 = total * 0.99
    count70 = 0

    total = 0
    i = 0
    while(total < thres70):
        total += s[i]
        count70 += 1
        i += 1

    count90 = count70
    while(total < thres90):
        total += s[i]
        count90 += 1
        i += 1
    count99 = count90
    while(total < thres99):
        total += s[i]
        count99 += 1
        i += 1




    U,V = ALS.ALS(training, count70, 1, 5)
    return U,V
    #ALS.ALS(training, count90, 1)
    #ALS.ALS(training, count99, 1)



u, v = ALStes()

