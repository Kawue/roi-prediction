import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from maximally_stable_regions import calc_mser
from skimage import img_as_ubyte
from mini_mser import calc_mser as cmser
import os
from skimage.morphology import binary_closing, square, disk, binary_opening

class ROIpredictor:
    def __init__(self, aggregation_mode, sequence_min=3, delta=1, min_area=100, max_area=2000):
        mpl.use("TkAgg")

        self.aggregation_mode = aggregation_mode
        self.sequence_min = sequence_min
        self.delta = delta
        self.min_area = min_area # 0.01
        self.max_area = max_area # measured_area_size = int(np.where(sum_img > 0)[0].size * 0.8)

    def fit(self, dframe, memberships, normalize, close):
        self.dframe = dframe
        self.memberships = memberships
        self.gx = np.array(self.dframe.index.get_level_values("grid_x"))
        self.gy = np.array(self.dframe.index.get_level_values("grid_y"))
        self.binary_mask = np.zeros((np.amax(self.gy)+1, np.amax(self.gx)+1))
        self.binary_mask[(self.gy, self.gx)] = 1
        self.data_dict = {}
        for memb in set(memberships):
            self.data_dict[memb] = {}
            grpframe = self.dframe.iloc[:, np.where(memberships==memb)[0]]
            self.data_dict[memb]["grpframe"] = grpframe
            grpimgs = [self.build_img(grpframe[col]) for col in grpframe.columns]
            if normalize:
                grpimgs = [(img - img.min())/(img.max()-img.min()) for img in grpimgs]
            grpimgs = np.array(grpimgs)
            self.data_dict[memb]["grpimgs"] = grpimgs
            grpimg = self.aggregation_img(grpimgs, self.aggregation_mode)
            self.data_dict[memb]["grpimg"] = grpimg
            mser, rois = self.predict_rois(grpimg)
            if close:
                closed_rois = []
                for roi in rois:
                    closed_rois.append(binary_closing(binary_opening(binary_closing(roi, selem=disk(3)), selem=square(3)), selem=square(3)))
                self.data_dict[memb]["rois"] = np.array(closed_rois)
                self.data_dict[memb]["mser"] = np.sum(np.array(closed_rois), axis=0)
            else:
                self.data_dict[memb]["rois"] = rois
                self.data_dict[memb]["mser"] = mser



    # Prediction of rois based on Maximally Stable Extended Regions, REMEMBER: There ist still an extension, important?
    def predict_rois(self, img):
        #es werden keine mser gefunden, nochmal in mser code gucken
        mser_img, stable_regions = calc_mser(img_as_ubyte(img), sequence_min=self.sequence_min, delta=self.delta, min_area=self.min_area, max_area=self.max_area, all_maps=True)
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
    def get_index(self, roi):
        sample = roi[(self.gy, self.gx)]
        roi_gx = self.gx[np.where(sample > 0)[0]]
        roi_gy = self.gy[np.where(sample > 0)[0]]
        roi_tuples = np.array(list(zip(roi_gy, roi_gx)))
        return roi_gx, roi_gy



    # Plot each roi in a separate image. Sample area in background and roi area in foreground
    def plot_rois(self, grp_idx, mser, savepath):
        if type(grp_idx) == int:
            grp_idx = [grp_idx]
        elif grp_idx is None:
            grp_idx = list(self.data_dict.keys)
        else:
            if type(grp_idx) != list:
                raise ValueError("Wrong type for grp_idx!")
        
        for idx in grp_idx:
            title = "Group %i "%(idx)
            plt.figure()
            plt.title(title + "MSER")
            #plt.imshow(self.data_dict[idx]["mser"])
            if savepath:
                clusterpath = os.path.join(savepath, "C%i"%idx)
                if not os.path.isdir(clusterpath):
                    os.makedirs(clusterpath)
                plt.imsave("C%i-MSER"%idx, self.data_dict[idx]["mser"])
            for i, roi in enumerate(self.data_dict[idx]["rois"]):
                plt.figure()
                plt.title(title + "ROI %i"%(i))
                #plt.imshow(self.binary_mask + roi)
                if savepath:
                    plt.imsave("C%i-ROI%i"%(idx,i), self.binary_mask + roi)
            plt.show()


    def create_base_image(self):
        img = np.zeros((self.gy.max()+1, self.gx.max()+1))
        img[(self.gy, self.gx)] = 1
        return img

    '''
    def approximate_binary_mask(self):
        mask = np.sum()
        mask[mask > 0] = 1
        return mask
    '''


multi thresholding als rois und summe als mser als alternative auf agglomeriertem image