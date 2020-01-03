import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import KMeans
from roi_predictor import ROIpredictor
from msi_image_writer import MsiImageWriter
from sklearn.cluster import AgglomerativeClustering

def spectral_clustering_dframe(dframe, method, plot):
    labels = method.fit_predict(dframe)
    labels = labels+1
    gx = np.array(dframe.index.get_level_values("grid_x")).astype(int)
    gy = np.array(dframe.index.get_level_values("grid_y")).astype(int)
    img = np.zeros((gy.max()+1, gx.max()+1))
    img[(gy,gx)] = labels
    if plot:
        #plt.figure()
        #plt.imshow(img, cmap="tab20")
        plt.imsave(os.path.join(savepath,"pre_segmentation.png"), img, cmap="tab20")
        plt.close()
    

# Spatial Clustering
def spatial_clustering(dframe, method):
    labels = method.fit_predict(dframe.T)
    return labels


# ROI prediction
def roi_prediction(dframe, labels, aggregation_mode, method):
    pred = ROIpredictor(aggregation_mode=aggregation_mode)
    pred.fit(dframe, labels, method, True, True)
    return pred
    


# Spectral Clustering on each ROI
def spectral_clustering_rois(dframe, method, pred, plot):
    segmented_rois = []
    for k, rois in pred.get_rois().items():
        seg_rois = []
        for i, roi in enumerate(rois):
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
            if plot:
                #plt.figure()
                #plt.imshow(img, cmap="tab20")
                plt.imsave(os.path.join(savepath,"C%i"%k,"roi%i_segmentation.png"%i), img, cmap="tab20")
                plt.close()
                #plt.figure()
                plt.imsave(os.path.join(savepath,"C%i"%k,"roi%i.png"%i), roi)
                plt.close()
            seg_rois.append(img)
        segmented_rois.append(seg_rois)
    return segmented_rois



# Spectral Clustering on each MSER
def spectral_clustering_msers(dframe, method, pred, plot):
    segmented_mser = []
    for k, mser in pred.get_msers().items():
        gx, gy = pred.get_index(mser)
        dname = [dframe.index.get_level_values("dataset")[0]]*len(gx)
        mserframe = dframe.loc[zip(gx, gy, dname)]
        print(mserframe.shape)
        labels = method.fit_predict(mserframe)
        print(labels.shape)
        labels = labels+2
        #tab20 = plt.cm.tab20(np.linspace(0, 1, np.amax(labels)))
        dgx = np.array(dframe.index.get_level_values("grid_x")).astype(int)
        dgy = np.array(dframe.index.get_level_values("grid_y")).astype(int)
        img = np.zeros((dgy.max()+1, dgx.max()+1))
        img[(dgy,dgx)] = 1
        print(img.shape)
        img[(gy,gx)] = labels
        if plot:
            #plt.figure()
            #plt.imshow(img, cmap="tab20")
            # !!! MSER is not correct since other methods are not called MSER !!!
            plt.imsave(os.path.join(savepath,"C%i"%k,"mser_segmentation.png"), img, cmap="tab20")
            plt.close()
            #plt.figure()
            plt.imsave(os.path.join(savepath,"C%i"%k,"mser.png"), mser)
            plt.close()
        segmented_mser.append(img)
    return segmented_mser



def plot_spatial_clusters(dframe, labels, savepath):
    writer = MsiImageWriter(dframe, savepath)
    writer.write_msi_clusters(labels)
    plt.close("all")


def plot_rois():
    pass


def plot_segmentations():
    pass


if __name__ == "__main__":
    mpl.use('TkAgg')
    #dframe = pd.read_hdf("C:\\Users\\kwuellems\\Desktop\\grinev2test\\barley101GrineV2.h5")
    dframe = pd.read_hdf("C:\\Users\\kwuellems\\Desktop\\msi-measure-compare-datasets\\barley_101\\barley101.h5")
    savepath = "C:\\Users\\kwuellems\\Github\\roi_prediction\\testresults"
    print(dframe.shape)
    #rand = np.random.randint(0,101, dframe.size)
    #rand = rand/10000000000
    #rand = rand.reshape(dframe.shape)
    #dframe = dframe + rand
    dframe = dframe + 0.000000001
    print(dframe.shape)

    method = AgglomerativeClustering(n_clusters=6, affinity="cosine", linkage="average") #affinity="euclidean", linkage="ward"

    # First do a segmentation on the whole data set as comparison
    spectral_clustering_dframe(dframe=dframe, method=method, plot=True)
    
    # Follow up with the whole workflow to show the advantage of roi limitation for segmentation
    method = AgglomerativeClustering(n_clusters=8, affinity="cosine", linkage="average") #affinity="euclidean", linkage="ward"
    labels = spatial_clustering(dframe=dframe, method=method)
    plot_spatial_clusters(dframe=dframe, labels=labels, savepath=savepath)
    pred = roi_prediction(dframe=dframe, labels=labels, aggregation_mode=2, method="cv")
    
    method = AgglomerativeClustering(n_clusters=6, affinity="cosine", linkage="average") #affinity="euclidean", linkage="ward"
    segmented_msers = spectral_clustering_msers(dframe=dframe, method=method, pred=pred, plot=True)
    
    segmented_rois = spectral_clustering_rois(dframe=dframe, method=method, pred=pred, plot=True)
