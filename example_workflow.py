# Spatial Clustering
def spatial_clustering():
    pass


# ROI prediction
def roi_prediction():
    pass


# Spectral Clustering on each ROI
def spectral_clustering():
    pass


def plot_spatial_clusters():
    pass


def plot_rois():
    pass


def plot_segmentations():
    pass


if __name__ == "__main__":
    # First do a segmentation on the whole data set as comparison
    spectral_clustering()
    plot_segmentations()

    # Follow up with the whole workflow to show the advantage of roi limitation for segmentation
    spatial_clustering()
    plot_spatial_clusters()
    roi_prediction()
    plot_rois()
    spectral_clustering()
    plot_segmentations()
