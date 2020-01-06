import numpy as np
import scipy as sp
import sklearn
from scipy.cluster.hierarchy import fclusterdata

def cluster_routine(data, n_clusters, method):
    if method == "AgglomerativeClustering":
        return agglomerativeclustering(data, n_clusters)
    elif method == "kMeans":
        return kmeans(data, n_clusters)
    else:
        raise ValueError("Method not implemented. Use either 'AgglomerativeClustering' or 'kMeans' or provide labels by loading a .npy or .csv file.")


#return fclusterdata(data, t=nr_clusters, criterion="maxclust", metric=metric, method=method) - 1
def agglomerativeclustering(data, n_clusters, metric="cosine", linkage="average"):
    distance_matrix = sp.spatial.distance.pdist(data, metric=metric)
    clustering = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage=linkage)
    labels = clustering.fit_predict(distance_matrix)
    return labels
    

def kmeans(data, n_clusters):
    clustering = sklearn.cluster.KMeans(n_clusters=n_clusters)
    labels = clustering.fit_predict(data)
    return labels