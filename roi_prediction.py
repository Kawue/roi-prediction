import numpy as np
import pandas as pd
import matplotlib.pyplot as pltmatrixre
from maximally_stable_regions import calc_mser
from skimage import img_as_ubyte


class ROIpredictor:
    def __init__(self, aggregation_mode, sequence_min=3, delta=1, min_area=0.01, max_area=0.8):
        self.aggregation_mode = aggregation_mode
        self.sequence_min = sequence_min
        self.delta = delta
        self.min_area = min_area # 0.01
        self.max_area = max_area # measured_area_size = int(np.where(sum_img > 0)[0].size * 0.8)

    def fit(self, dframe, memberships):
        self.dframe = dframe
        self.memberships = memberships
        self.gx = self.dframe.index.get_level_values("grid_x")
        self.gy = self.dframe.index.get_level_values("grid_y")
        self.binary_mask = np.zeros((np.amax(self.gy)+1, np.amax(self.gx)+1))
        self.binary_mask[(self.gy, self.gx)] = 1
        self.data_dict = {}
        for memb in set(memberships):
            self.data_dict[memb] = {}
            grpframe = self.dframe.loc[:, np.where(memberships==memb)]
            self.data_dict[memb]["grpframe"] = grpframe
            grpimgs = np.array([self.build_img(grpframe[col]) for col in grpframe.columns])
            self.data_dict[memb]["grpimgs"] = grpimgs
            grpimg = self.aggregation_img(grpimgs, self.aggregation_mode)
            self.data_dict[memb]["grpimg"] = grpimg
            mser, rois = self.predict_rois(grpimg)
            self.data_dict[memb]["rois"] = rois
            self.data_dict[memb]["mser"] = mser

            

    

    # Prediction of rois based on Maximally Stable Extended Regions, REMEMBER: There ist still an extension, important?
    def predict_rois(self, img):
        mser_img, stable_regions = calc_mser(img_as_ubyte(img), sequence_min=self.sequence_min, delta=self.delta, min_area=self.max_area, max_area=self.max_area, all_maps=True)
        return mser_img, stable_regions

    
    def aggregation_img(self, grp, aggregation_mode, normalization=True):
        if aggregation_mode == 1:
            img = np.sum(grp, axis=0)
        elif aggregation_mode == 2:
            img = np.mean(grp, axis=0)
        elif aggregation_mode == 3:
            img = np.max(grp, axis=0)
        elif aggregation_mode == 4:
            img = np.min(grp, axis=0)
        elif aggregation_mode == 5:
            img = np.median(grp, axis=0)
        elif aggregation_mode == 6:
            img = np.prod(grp, axis=0)

        if normalization:
            ma = np.amax(img)
            mi = np.amin(img)
            img = (img - mi) / (ma - mi)

        return img

    def build_img(self, intensities):
        img = np.zeros((np.amax(self.gy)+1, np.amax(self.gx)+1))
        img[(self.gy, self.gx)] = intensities
        return img

    '''
    def approximate_binary_mask(self):
        mask = np.sum()
        mask[mask > 0] = 1
        return mask
    '''



    # Some getter to access data
    def get_rois(self):
        return {memb: self.data_dict[memb]["rois"] for memb in self.data_dict}

    def get_msers(self):
        return {memb: self.data_dict[memb]["mser"] for memb in self.data_dict}

    def get_aggregation_images(self):
        return {memb: self.data_dict[memb]["grpimg"] for memb in self.data_dict}

    def get_image_groups(self):
        return {memb: self.data_dict[memb]["grpimgs"] for memb in self.data_dict}

    def get_group_dframe(self):
        return {memb: self.data_dict[memb]["grpframe"] for memb in self.data_dict}

    # Return a list of (x,y) tuples for each roi
    def get_index(self):
        pass



    # Plot each roi in a separate image. Sample area in background and roi area in foreground
    def plot_rois(self):
        img = np.zeros()
        pass