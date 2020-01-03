import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu

def calc_mt_regions(img, thresholds):
    if type(thresholds) == int:
        # +2 because min is 0 and max is 255
        #thresholds = np.linspace(0, 255, thresholds+2)
        thresholds = threshold_multiotsu(img, classes=thresholds+1, nbins=256)
    elif type(thresholds) == list:
        if np.any([1 > x > 256 for x in thresholds]):
            raise ValueError("Tresholds must be in [1,255].")
        thresholds = thresholds
    else:
        raise ValueError("thresholds parameter needs to be of type list or int.")

    regions = []
    for t in thresholds:
        reg = np.zeros(img.shape)
        reg[np.where(img >= t)] = 1
        regions.append(reg)

    return np.sum(regions, axis=0), regions