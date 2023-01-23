import cv2
from skimage import img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
import skimage.filters as skif
from skimage.exposure import equalize_adapthist
from scipy.stats.mstats import winsorize
from skimage.segmentation import felzenszwalb

def calc_mser_cv(img, delta, min_area, max_area, binary_mask):
    binary_mask = binary_mask.astype(bool)
    if img.dtype != np.uint8:
        img = img_as_ubyte(img)
    #img = img.astype(float)
    #idx = np.where(~binary_mask)
    #for i,j in zip(idx[0],idx[1]):
    #    img[i,j] = np.nan
    #img = img.astype(np.uint8)

    img2 = equalize_adapthist(img)

    #img = prepro(img, binary_mask)
    #img = img_as_ubyte(img)
    #print(img.shape)


    delta = 5
    min_area = int(img.size * min_area)
    max_area = int(img.size * max_area)
    max_variation = 0.05
    min_diversity = 0.8
    mser = cv2.MSER_create(_delta=delta, _min_area=min_area, _max_area=max_area, _max_variation=max_variation, _min_diversity=min_diversity)
    regions, bboxes = mser.detectRegions(img)
    regions_list = []
    print(regions)
    for reg in regions:
        print(reg)
        img2 = np.zeros(img.shape)
        img2[(reg[:,1],reg[:,0])] = 1
        #if (img2 * binary_mask).sum() != 0:
        plt.figure()
        plt.imshow(img2)
        regions_list.append(img2)
    plt.show()
    return np.sum(regions_list, axis=0), regions_list

def prepro(img, binary_mask):
    p = np.percentile(img[binary_mask], 99)
    img[img > p] = p

    t = skif.threshold_otsu(img[binary_mask], nbins=256)
    print(t)
    print(np.percentile(img[binary_mask], 60))
    img[img < t] = 0
    img = skif.gaussian(img, sigma=0.8)
    img = equalize_adapthist(img)

    return img