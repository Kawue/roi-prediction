import sys
import numpy as np
import scipy as sp
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial.distance import squareform

def cluster_routine(data, n_clusters, method):
    data = data.T
    if method == "AgglomerativeClustering":
        return agglomerativeclustering(data, n_clusters)
    elif method == "kMeans":
        return kmeans(data, n_clusters)
    else:
        raise ValueError("Method not implemented. Use either 'AgglomerativeClustering' or 'kMeans' or provide labels by loading a .npy or .csv file.")


########## Cluster Routines ##########
# Each method has to return a list of labels that begins with zero!

#return fclusterdata(data, t=nr_clusters, criterion="maxclust", metric=metric, method=method) - 1
def agglomerativeclustering(data, n_clusters, metric="cosine", linkage="average"):
    distance_matrix = squareform(sp.spatial.distance.pdist(data, metric=metric))
    print(data.shape)
    print(distance_matrix.shape)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage=linkage)
    labels = clustering.fit_predict(distance_matrix)
    return labels


def kmeans(data, n_clusters):
    clustering = KMeans(n_clusters=n_clusters)
    labels = clustering.fit_predict(data)
    return labels
