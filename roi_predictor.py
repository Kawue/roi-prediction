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
from mini_mser import calc_mser_cv as calc_mser_cv
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
        self.memberships = None

    # Fit a given clustering
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


    # Fill wholes in a RoI image
    def fill_holes(self, results_dict, method):
        for resultskey, result_dict in results_dict.items():
            for regionkey, region in result_dict["regions_dict"].items():
                if method == "morphology":
                    filledregion = binary_closing(binary_opening(binary_closing(region, selem=disk(3)), selem=square(3)), selem=square(3))
                elif method == "cv":
                    filledregion = calc_ac(region, method)
                elif method == "mcv":
                    filledregion = calc_ac(region, method)
                elif method == "mgac":
                    filledregion = calc_ac(region, method)
                elif method == "contours_low":
                    filledregion = calc_ac(region, method)
                elif method == "contours_high":
                    filledregion = calc_ac(region, method)
                else:
                    raise ValueError("Function 'fill_holes()' takes only 'morphology', 'cv', 'mcv', 'mgac', 'contours_low' or 'contours_high' as method parameter.")
                results_dict[resultskey]["regions_dict"][regionkey] = filledregion
            region_sum = np.sum([region for key, region in results_dict[resultskey]["regions_dict"].items()], axis=0)
            results_dict[resultskey] = {"region_sum": region_sum, "regions_dict": result_dict["regions_dict"]}
        return results_dict


    # Aggregate images of a clustered image stack
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

    
    # Call a RoI prediction method
    def predict_rois(self, method, kwargs):
        if method == "dr":
            results_dict = self._dr_routine(**kwargs)
        elif method == 'mt':
            results_dict = self._mt_routine(**kwargs)
        elif method == 'mser':
            results_dict = self._mser_routine(**kwargs)
        else:
            raise ValueError("Available methods for predict_rois() are 'dr' for dimension reduction, 'mt' for multi threshold or 'mser' for maximally stable extendable regions.")
        return results_dict





########## Prediction Routines ##########
  
    def _mt_routine(self, images=None, thresholds=None):
        if images is None or thresholds is None:
            raise ValueError("For multi threshold roi prediction an image and a list of thresholds or number of classes must be provided.")
        results_dict = {}
        for idx, img in enumerate(images):
            img = img_as_ubyte(img)
            region_sum, regions = calc_mt_regions(img, thresholds)
            regions_dict = {}
            for i, region in enumerate(regions):
                regions_dict[i] = region
            results_dict[idx] = {"region_sum": region_sum, "regions_dict": regions_dict}
        return results_dict


    def _dr_routine(self, pred_method, dr_method=None, components=None, embedding_nr=None, components_method=None, n_neighbors=None, radius=None, expansion_factor=None):
        results_dict = {}
        if dr_method is None:
            raise ValueError("For dimension reduction roi prediction the dimension reduction method must be provided.")
        drr = DimensionReductionRegions(self.dframe, dr_method=dr_method, components=components, embedding_nr=embedding_nr)
        if pred_method == "component_pred":
            if embedding_nr is None or components_method is None:
                raise ValueError("For the 'components' submethod in dimension reduction roi prediction at lest one 'embedding_nr' and a method ('cv', 'mcv', 'mgac', 'contours_low' or 'contours_high') must be provided.")
            region_sum, regions = drr.embedding_regions(method=components_method, embedding_nr=embedding_nr)
        elif pred_method == "knn_pred":
            if components is None or n_neighbors is None or radius is None:
                raise ValueError("For the 'knn' submethod in dimension reduction roi prediction 'components', 'n_neighbors' and 'radius' must be provided.")
            if len(embedding_nr) != 2:
                raise ValueError("embedding_nr must provide exact two components (two integers) to use cc_regions().")
            region_sum, regions = drr.cc_regions(selected_components=embedding_nr, n_neighbors=n_neighbors, radius=radius, expansion_factor=expansion_factor)
        elif pred_method == "interactive_pred":
            if components is None:
                raise ValueError("For the 'interactive' submethod in dimension reduction roi prediction 'components' must be provided.")
            if len(embedding_nr) != 2:
                raise ValueError("embedding_nr must provide exact two components (two integers) to use interactive_regions().")
            region_sum, regions = drr.interactive_regions(selected_components=embedding_nr)
        elif pred_method == "return_drr":
            return drr
        else:
            raise ValueError("Error in _dr_routine().")
        results_dict[0] = {"region_sum": region_sum, "regions_dict": regions}
        return results_dict

    
    def _mser_routine(self, images, mser_method, sequence_min, delta, min_area, max_area):
        if images is None or sequence_min is None or delta is None or min_area is None or max_area is None:
            raise ValueError("For multi threshold roi prediction an image, 'sequence_min', 'delta', 'min_area' and 'max_area' must be provided.")
        results_dict = {}
        for idx, img in enumerate(images):
            img = img_as_ubyte(img)
            # Prediction of rois based on Maximally Stable Extended Regions, REMEMBER: There ist still an extension, important?
            region_sum, regions = calc_mser(img, sequence_min=sequence_min, delta=delta, min_area=min_area, max_area=max_area, all_maps=True, binary_mask=self.binary_image)
            #region_sum, regions = calc_mser_cv(img, delta=delta, min_area=min_area, max_area=max_area, binary_mask=self.binary_image)
            regions_dict = {}
            if mser_method == "level":
                for lvl in range(0, int(np.amax(region_sum))):
                    img = self._create_empty_img(False)
                    img[np.where(region_sum > lvl)] = 1
                    regions_dict[lvl] = img
            elif mser_method == "all":
                for i, region in enumerate(regions):
                    regions_dict[i] = region
            else:
                raise ValueError("mser_method must be 'level' or 'all'.")
            results_dict[idx] = {"region_sum": region_sum, "regions_dict": regions_dict}
        return results_dict





########## Utility Functions ##########

    # https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
    def _normalization(self, x, nmin=0, nmax=1):
        return normalization(x)

    def _create_empty_img(self, rgba):
        return create_empty_img(self.height, self.width, rgba)

    def _create_img(self, intensities):
        return create_img(intensities, self.gy, self.gx, self.height, self.width)
    




########## Getters ##########
    def get_aggregation_images(self, method):
        if self.memberships is None:
            raise ValueError("fit_clustering() was not called.")
        return {memb: self.data_dict[memb]["grpimg"][method] for memb in self.data_dict}

    def get_image_groups(self):
        if self.memberships is None:
            raise ValueError("fit_clustering() was not called.")
        return {memb: self.data_dict[memb]["grpimgs"] for memb in self.data_dict}

    def get_group_dframe(self):
        if self.memberships is None:
            raise ValueError("fit_clustering() was not called.")
        return {memb: self.data_dict[memb]["grpframe"] for memb in self.data_dict}