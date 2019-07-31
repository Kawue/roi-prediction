import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from roi_predictor import ROIpredictor
from msi_image_writer import MsiImageWriter

# Spatial Clustering
def spatial_clustering(dframe):
    labels = KMeans(n_clusters=8).fit_predict(dframe.T)
    return labels


# ROI prediction
def roi_prediction(dframe, labels):
    pred = ROIpredictor(aggregation_mode=3)
    pred.fit(dframe, labels, True, True)
    #pred.plot_rois(0,True)
    pred.get_index(None)
    plt.show()


# Spectral Clustering on each ROI
def spectral_clustering():
    pass


def plot_spatial_clusters(dframe, labels, savepath):
    writer = MsiImageWriter(dframe, savepath)
    writer.write_msi_clusters(labels)


def plot_rois():
    pass


def plot_segmentations():
    pass


if __name__ == "__main__":
    dframe = pd.read_hdf("C:\\Users\\kwuellems\\Desktop\\grinev2test\\barley101GrineV2.h5")
    print(dframe.shape)
    savepath = "C:\\Users\\kwuellems\\Desktop\\roipredictiontests"
    # First do a segmentation on the whole data set as comparison
    spectral_clustering()
    plot_segmentations()

    # Follow up with the whole workflow to show the advantage of roi limitation for segmentation
    labels = spatial_clustering(dframe)
    
    #plot_spatial_clusters(dframe, labels, savepath)
    
    roi_prediction(dframe, labels)
    #spectral_clustering()
    #plot_segmentations()
