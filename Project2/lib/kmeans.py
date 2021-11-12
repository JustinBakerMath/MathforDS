import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean,cosine

def orig(X,Y):
    return np.linalg.norm(X-Y,axis=1)

def eucl(X,Y):
    ret = []
    for x in X:
        ret = ret + [euclidean(x,Y)]
    return np.array(ret).T

def cos(X,Y):
    ret = []
    for x in X:
        ret = ret + [cosine(x,Y)]
    return np.array(ret).T


def kmeans(X,k,distance=eucl,eps=1e-8,maxiters=100,seed=0):
    np.random.seed(seed)
    n,d = X.shape
    cent = np.random.rand(k,d)
    centk = cent.copy()
    for i in range(maxiters):
        dist = distance(X,cent[0,:]).reshape(-1,1)
        for class_ in range(1,k):
            dist = np.append(dist, distance(X,cent[class_,:]).reshape(-1,1), axis=1)
        classes = np.argmin(dist,axis=1)
        for class_ in classes:
            cent[class_,:] = np.mean(X[classes == class_,:],axis=0)
        if np.linalg.norm(cent - centk) < eps:
            print("Reached training criteria in {} iterations".format(i))
            break
        centk = cent.copy()

    if i == maxiters: print("Reached max iterations")
    return classes, cent, dist