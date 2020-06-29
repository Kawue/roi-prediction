import sys
import numpy as np
import scipy as sp
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import fclusterdata, fcluster
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
    if n_clusters > 1:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage=linkage)
        labels = clustering.fit_predict(distance_matrix)
    elif n_clusters == -1:
        cond_dmatrix = squareform(distance_matrix)
        Z = linkage(cond_dmatrix, method="average", optimal_ordering=True)
        mean_dd = np.mean(Z[:,2])
        std_dd = np.std(Z[:,2])
        C = 1
        labels = fcluster(Z, t=C*std_dd+mean_dd, criterion="distance")
        labels = labels - 1
    else:
        raise ValueError("Choose n_clusters as -1 or greater than 1.")
    return labels


def kmeans(data, n_clusters):
    clustering = KMeans(n_clusters=n_clusters)
    labels = clustering.fit_predict(data)
    return labels