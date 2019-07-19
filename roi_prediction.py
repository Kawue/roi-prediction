import numpy as np
import pandas as pd
import matplotlib.pyplot as pltmatrixre
from maximally_stable_regions import calc_mser


class ROIpredictor:
    def __init__(self, sequence_min=3, delta=1, min_area=0.01, max_area=0.8):
        self.sequence_min = sequence_min
        self.delta = delta
        self.min_area = min_area # 0.01
        self.max_area = max_area # measured_area_size = int(np.where(sum_img > 0)[0].size * 0.8)

    def fit(self, data, memberships):
        self.data = data
        self.memberships = memberships
        pass

    # Prediction of rois based on Maximally Stable Extended Regions, REMEMBER: There ist still an extension, important?
    def predict_rois(self):
        mser_img, stable_regions = calc_mser(img_as_ubyte(sum_img), sequence_min=self.sequence_min, delta=self.delta, min_area=self.max_area, max_area=self.max_area, all_maps=True)


    # Return list of rois ... format? maybe list of cutted data slices? Either frames or nparray, depending on input?
    def get_rois(self):
        pass

    # Return a list of (x,y) tuples for each roi
    def get_index(self):
        pass

    # Plot each roi in a separate image. Sample area in background and roi area in foreground
    def plot_rois(self):
        img = np.zeros()
        pass