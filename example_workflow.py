import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import KMeans
from roi_predictor import ROIpredictor
from msi_image_writer import MsiImageWriter
from sklearn.cluster import AgglomerativeClustering

def spectral_clustering_dframe(dframe, method):
    labels = method.fit_predict(dframe)
    labels = labels+1
    gx = np.array(dframe.index.get_level_values("grid_x")).astype(int)
    gy = np.array(dframe.index.get_level_values("grid_y")).astype(int)
    img = np.zeros((gy.max()+1, gx.max()+1))
    img[(gy,gx)] = labels
    plt.figure()
    plt.imshow(img, cmap="tab20")
    

# Spatial Clustering
def spatial_clustering(dframe, method):
    labels = method.fit_predict(dframe.T)
    return labels


# ROI prediction
def roi_prediction(dframe, labels):
    pred = ROIpredictor(aggregation_mode=3)
    pred.fit(dframe, labels, True, True)
    return pred
    


# Spectral Clustering on each ROI
def spectral_clustering_rois(dframe, method, pred):
    for k, rois in pred.get_rois().items():
        for roi in rois:
            gx, gy = pred.get_index(roi)
            dname = [dframe.index.get_level_values("dataset")[0]]*len(gx)
            roiframe = dframe.loc[zip(gx, gy, dname)]
            labels = method.fit_predict(roiframe)
            labels = labels+2
            #tab20 = plt.cm.tab20(np.linspace(0, 1, np.amax(labels)))
            dgx = np.array(dframe.index.get_level_values("grid_x")).astype(int)
            dgy = np.array(dframe.index.get_level_values("grid_y")).astype(int)
            img = np.zeros((dgy.max()+1, dgx.max()+1))
            img[(dgy,dgx)] = 1
            img[(gy,gx)] = labels
            plt.figure()
            plt.imshow(img, cmap="tab20")
        plt.show()



# Spectral Clustering on each MSER
def spectral_clustering_msers(dframe, method, pred):
    for k, mser in pred.get_msers().items():
            gx, gy = pred.get_index(mser)
            dname = [dframe.index.get_level_values("dataset")[0]]*len(gx)
            mserframe = dframe.loc[zip(gx, gy, dname)]
            labels = method.fit_predict(mserframe)
            labels = labels+2
            #tab20 = plt.cm.tab20(np.linspace(0, 1, np.amax(labels)))
            dgx = np.array(dframe.index.get_level_values("grid_x")).astype(int)
            dgy = np.array(dframe.index.get_level_values("grid_y")).astype(int)
            img = np.zeros((dgy.max()+1, dgx.max()+1))
            img[(dgy,dgx)] = 1
            img[(gy,gx)] = labels
            plt.figure()
            plt.imshow(img, cmap="tab20")



def plot_spatial_clusters(dframe, labels, savepath):
    writer = MsiImageWriter(dframe, savepath)
    writer.write_msi_clusters(labels)


def plot_rois():
    pass


def plot_segmentations():
    pass


if __name__ == "__main__":
    mpl.use('TkAgg')
    dframe = pd.read_hdf("C:\\Users\\kwuellems\\Desktop\\grinev2test\\barley101GrineV2.h5")
    print(dframe.shape)
    savepath = "C:\\Users\\kwuellems\\Desktop\\roipredictiontests"

    method = AgglomerativeClustering(n_clusters=4, affinity="euclidean", linkage="ward")

    # First do a segmentation on the whole data set as comparison
    spectral_clustering_dframe(dframe, method)
    
    # Follow up with the whole workflow to show the advantage of roi limitation for segmentation
    method = AgglomerativeClustering(n_clusters=8, affinity="euclidean", linkage="ward")
    labels = spatial_clustering(dframe, method)
    pred = roi_prediction(dframe, labels)

    method = AgglomerativeClustering(n_clusters=4, affinity="euclidean", linkage="ward")
    spectral_clustering_msers(dframe, method, pred)
    plt.show()
    spectral_clustering_rois(dframe, method, pred)
    #plot_segmentations()
