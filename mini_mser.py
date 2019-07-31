import cv2
from skimage import img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt

def calc_mser(img):
    delta = 10
    min_area = 500
    max_area = 10000
    max_variation = 0.1
    mser = cv2.MSER_create(_delta=delta, _min_area=min_area, _max_area=max_area, _max_variation=max_variation)
    regions, bboxes = mser.detectRegions(img_as_ubyte(img))
    for reg in regions:
        img2 = np.zeros(img.shape)
        img2[(reg[:,1],reg[:,0])] = 1
        plt.imshow(img2)
        plt.show()
    
    asdsad
    print(bboxes)