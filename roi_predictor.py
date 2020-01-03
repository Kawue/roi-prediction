import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import AutoVivificationDict, create_empty_img, create_img, normalization
from active_contours import calc_ac
from dimension_reduction_regions import DimensionReductionRegions


from maximally_stable_regions import calc_mser
from multi_threshold_regions import calc_mt_regions

from skimage import img_as_ubyte
from mini_mser import calc_mser as cmser
import os
from skimage.morphology import binary_closing, square, disk, binary_opening

class ROIpredictor:
    def __init__(self, dframe):
        self.dframe = dframe
        self.gx = np.array(self.dframe.index.get_level_values("grid_x")).astype(int)
        self.gy = np.array(self.dframe.index.get_level_values("grid_y")).astype(int)
        self.width = np.amax(self.gx)
        self.height = np.amax(self.gy)
        self.binary_image = self._create_img(1) #"Sample/Measured Area"

    def fit_clustering(self, memberships, normalize):
        self.memberships = memberships
        self.data_dict = AutoVivificationDict()
        for memb in set(memberships):
            grpframe = self.dframe.iloc[:, np.where(self.memberships==memb)[0]]
            self.data_dict[memb]["grpframe"] = grpframe
            if normalize:
                grpimgs = np.array([self._normalization(self._create_img(grpframe[col])) for col in grpframe.columns])
            else:
                grpimgs = np.array([self._create_img(grpframe[col]) for col in grpframe.columns])
            self.data_dict[memb]["grpimgs"] =  grpimgs
            self.data_dict[memb]["grpimg"]["mean"] = self.aggregation_img(grpimgs, 1)
            self.data_dict[memb]["grpimg"]["median"] = self.aggregation_img(grpimgs, 2)
            self.data_dict[memb]["grpimg"]["sum"] = self.aggregation_img(grpimgs, 3)
            self.data_dict[memb]["grpimg"]["prod"] = self.aggregation_img(grpimgs, 4)
            self.data_dict[memb]["grpimg"]["max"] = self.aggregation_img(grpimgs, 5)
            self.data_dict[memb]["grpimg"]["min"] = self.aggregation_img(grpimgs, 6)


    def fill_holes(self, roi, method):
        if method == "morphology":
            return binary_closing(binary_opening(binary_closing(roi, selem=disk(3)), selem=square(3)), selem=square(3))
        elif method == "cv":
            return calc_ac(roi, method)
        elif method == "mcv":
            return calc_ac(roi, method)
        elif method == "ac":
            return calc_ac(roi, method)
        elif method == "mgac":
            return calc_ac(roi, method)
        else:
            raise ValueError("Function 'fill_holes()' takes only 'morphology','cv','mcv','ac' or 'mgac' as method parameter.")


    
    def aggregation_img(self, grp, aggregation_mode, normalization=True):
        if aggregation_mode == 1:
            img = np.mean(grp, axis=0)
        elif aggregation_mode == 2:
            img = np.median(grp, axis=0)
        elif aggregation_mode == 3:
            img = np.sum(grp, axis=0)
        elif aggregation_mode == 4:
            img = np.prod(grp, axis=0)
        elif aggregation_mode == 5:
            img = np.max(grp, axis=0)
        elif aggregation_mode == 6:
            img = np.min(grp, axis=0)

        if normalization:
            img = self._normalization(img)

        return img

    
    def predict_rois(self, method, kwargs):
        if method == "dr":
            region_sum, regions = self._dr_routine(**kwargs)
        elif method == 'mt':
            region_sum, regions = self._mt_routine(**kwargs)
        elif method == 'mser':
            region_sum, regions = self._mser_routine(**kwargs)
        else:
            raise ValueError("Available methods for predict_rois() are 'dr' for dimension reduction, 'mt' for multi threshold or 'mser' for maximally stable extendable regions.")
        
        return region_sum, regions
        

    
    def _mt_routine(self, img=None, thresholds=None):
        if img is None or thresholds is None:
            raise ValueError("For multi threshold roi prediction an image and a list of thresholds or number of classes must be provided.")
        img = img_as_ubyte(img)
        region_sum, regions = calc_mt_regions(img, thresholds)
        return region_sum, regions


    def _dr_routine(self, pred_method, dr_method=None, components=None, embedding_nr=None, n_neighbors=None, radius=None, expansion_factor=None):
        if dr_method is None:
            raise ValueError("For dimension reduction roi prediction the dimension reduction method must be provided.")
        drr = DimensionReductionRegions(self.dframe, dr_method=dr_method, components=components)
        if pred_method == "components":
            if embedding_nr is None:
                raise ValueError("For the 'components' submethod in dimension reduction roi prediction 'embedding_nr' must be provided.")
            region_sum, regions = drr.embedding_regions(embedding_nr=embedding_nr)
        elif pred_method == "knn":
            if components is None or n_neighbors is None or radius is None:
                raise ValueError("For the 'knn' submethod in dimension reduction roi prediction 'components', 'n_neighbors' and 'radius' must be provided.")
            region_sum, regions = drr.cc_regions(selected_components=components, n_neighbors=n_neighbors, radius=radius, expansion_factor=expansion_factor)
        elif pred_method == "interactive":
            if components is None:
                raise ValueError("For the 'interactive' submethod in dimension reduction roi prediction 'components' must be provided.")
            region_sum, regions = drr.interactive_regions(selected_components=components)
        return region_sum, regions

    
    def _mser_routine(self, img, sequence_min, delta, min_area, max_area):
        if img is None or sequence_min is None or delta is None or min_area is None or max_area is None:
            raise ValueError("For multi threshold roi prediction an image, 'sequence_min', 'delta', 'min_area' and 'max_area' must be provided.")
        img = img_as_ubyte(img)
        # Prediction of rois based on Maximally Stable Extended Regions, REMEMBER: There ist still an extension, important?
        region_sum, regions = calc_mser(img, sequence_min=sequence_min, delta=delta, min_area=min_area, max_area=max_area, all_maps=True)
        return region_sum, regions






    
########## Utility Functions ##########

    # https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
    def _normalization(self, x, nmin=0, nmax=1):
        return normalization(x)

    def _create_empty_img(self, rgba):
        return create_empty_img(self.height, self.width, rgba)

    def _create_img(self, intensities):
        return create_img(intensities, self.gy, self.gx, self.height, self.width)
    


########## Getters ##########

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