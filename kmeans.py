from numpy import *
import numpy as np


def gen_sim(a, b):
    num = float(np.dot(a, b.T))
    denum = np.linalg.norm(a) * np.linalg.norm(b)
    if denum == 0:
        denum = 1
    cosn = num / denum
    sim = 0.5 + 0.5 * cosn
    return sim


def randcent(dataset, k):
    n = np.shape(dataset)[1]
    centroids = np.mat(np.zeros((k, n)))  # create centroid mat
    for j in range(n):  # create random cluster centers, within bounds of each dimension
        minj = min(dataset[:, j])
        rangej = float(max(dataset[:, j]) - minj)
        centroids[:, j] = np.mat(minj + rangej * np.random.rand(k, 1))
    return centroids


def kmeans(dataset, k, distmeas=gen_sim, createcent=randcent):
    m = np.shape(dataset)[0]
    clusterassment = np.mat(np.zeros((m, 2)))  # create mat to assign data points
    # to a centroid, also holds SE of each point
    centroids = createcent(dataset, k)
    clusterchanged = True
    while clusterchanged:
        clusterchanged = False
        for i in range(m):  # for each data point assign it to the closest centroid
            mindist = np.inf
            minindex = -1
            for j in range(k):
                distji = distmeas(centroids[j, :], dataset[i, :])
                if distji < mindist:
                    mindist = distji
                    minindex = j
            if clusterassment[i, 0] != minindex:
                clusterchanged = True
            clusterassment[i, :] = minindex, mindist ** 2
        # print centroids
        for cent in range(k):  # recalculate centroids
            if clusterassment[:, 0].any() == cent:
                ptsinclust = dataset[np.nonzero(clusterassment[:, 0])[0]]  # get all the point in this cluster
                centroids[cent, :] = np.mean(ptsinclust, axis=0)  # assign centroid to mean
    return centroids, clusterassment
