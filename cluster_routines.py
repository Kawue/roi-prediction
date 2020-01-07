import numpy as np
import scipy as sp
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import fclusterdata

def add_cluster_subparser(parser):
    subparser = parser.add_subparsers(title="Optional Procedures", dest="command")
    cluster_parser = subparser.add_parser("clustering", help="arguments for pre defined clustering procedures.")
    cluster_parser.add_argument("--cluster_method", required=True, choices=["AgglomerativeClustering", "kMeans"], help="Cluster method.")
    cluster_parser.add_argument("--n_clusters", required=False, type=int, help="Number of Clusters.")
    cluster_parser.add_argument("--save_clustering", default=None, required=False, type=str, help="Filename or path/filename to save a cluster result if a pre defined cluster method was used. Must contain .npy or .csv.")
    cluster_parser.add_argument("--metric", required=False, help="Metric for distance computation (see scipy's pdist).")
    cluster_parser.add_argument("--linkage", required=False, help="Linkage Method ('ward', 'complete', 'average', 'single') for AgglomerativeClustering(). 'ward' requires 'euclidean' as metric.")

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
    distance_matrix = sp.spatial.distance.pdist(data, metric=metric)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage=linkage)
    labels = clustering.fit_predict(distance_matrix)
    return labels
    

def kmeans(data, n_clusters):
    clustering = KMeans(n_clusters=n_clusters)
    labels = clustering.fit_predict(data)
    return labels