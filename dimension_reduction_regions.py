import numpy as np
import pandas as pd
from skimage import img_as_ubyte
from skimage import filters as skfilters
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components
from msi_dimension_reducer import PCA, NMF, LDA, ICA, UMAP, TSNE
from utils import create_empty_img, create_img, normalization
from active_contours import calc_ac
from interactive_dimreduce_regions import InteractiveDimensionReductionRegionsSelector

import matplotlib.pyplot as plt


class DimensionReductionRegions():
    dr_dict = {
            "pca": PCA,
            "nmf": NMF,
            "ica": ICA,
            "lda": LDA,
            "tsne": TSNE,
            "umap": UMAP
        }
    def __init__(self, data, dr_method, components, embedding_nr, y_pixels=None, x_pixels=None):
        self.nr_components = components
        self.selected_components = embedding_nr
        self.data = data

        if isinstance(data, pd.DataFrame):
            self.gx = np.array(self.data.index.get_level_values("grid_x")).astype(int)
            self.gy = np.array(self.data.index.get_level_values("grid_y")).astype(int)
            self.width = np.amax(self.gx)
            self.height = np.amax(self.gy)
        else:
            if y_pixels is None or x_pixels is None:
                raise ValueError("If data is not a DataFrame, x and y pixels must be provided.")
            else:
                self.gx = x_pixels
                self.gy = y_pixels
                self.width = np.amax(self.gx)
                self.height = np.amax(self.gy)

        dimred = self.dr_dict[dr_method](data, self.nr_components)
        self.embedding = dimred.perform()[:, self.selected_components]


    def embedding_images(self, embedding_nr):
        embedding = self.embedding
        img = create_img(embedding[:,embedding_nr], self.gy, self.gx, self.height, self.width)
        return img


    def embedding_regions(self, method, embedding_nr):
        regions_dict={}
        for nr in embedding_nr:
            embedding_img = self.embedding_images(embedding_nr[nr])
            embedding_uint = img_as_ubyte(normalization(embedding_img))
            embedding_uint_bin = (embedding_uint > skfilters.threshold_otsu(embedding_uint)).astype(int)
            if method == "cv":
                region = calc_ac(embedding_uint_bin, "cv", init_img=embedding_img)
            elif method == "mcv":
                region = calc_ac(embedding_uint_bin, "mcv", init_img=embedding_img)
            elif method == "mgac":
                region = calc_ac(embedding_uint_bin, "mgac", init_img=embedding_img)
            elif method == "contours_low":
                region = calc_ac(embedding_img, "contours_low", init_img=embedding_uint_bin)
            elif method == "contours_high":
                region = calc_ac(embedding_img, "contours_high", init_img=embedding_uint_bin)
            regions_dict[nr] = region
        region_sum = np.sum([img for key, img in regions_dict.items()], axis=0)
        return region_sum, regions_dict

    def cc_regions(self, selected_components=None, n_neighbors=None, radius=None, expansion_factor=None):
        embedding = self.embedding
        '''
        if embedding.shape[1] == 2:
            selected_components = [0,1]
        elif selected_components is not None:
            if type(selected_components) == list:
                selected_components = selected_components
            else:
                raise ValueError("selected_components parameter must be od type list.")
        else:
            raise ValueError("'DimensionReductionRegions' was initialized with more than two components. Provide selected_components as list.")
        '''
        #embedding = embedding[:, selected_components]

        if n_neighbors is None:
            n_neighbors = 0

        if radius is None:
            radius = 0

        if expansion_factor is not None:
            embedding = embedding**expansion_factor * np.sign(embedding)

        nn = NearestNeighbors(n_neighbors=n_neighbors, radius=radius)
        nn.fit(embedding)

        knn = nn.kneighbors_graph()
        knn_nr_components, knn_cc_labels = connected_components(knn, directed=False)
        print("Number of knn components: %i"%(knn_nr_components))
        knn_cc = [tuple(np.where(knn_cc_labels==lbl)[0]) for lbl in range(max(knn_cc_labels)+1)]

        rnn = nn.radius_neighbors_graph()
        rnn_nr_components, rnn_cc_labels = connected_components(rnn, directed=False)
        print("Number of rnn components: %i"%(rnn_nr_components))
        rnn_cc = [tuple(np.where(rnn_cc_labels==lbl)[0]) for lbl in range(max(rnn_cc_labels)+1)]

        regions_idx = list(set(knn_cc + rnn_cc))

        print("Total number of components: %i"%(len(regions_idx)))

        regions = []
        
        for idx in regions_idx:
            idx = list(idx)
            img = create_empty_img(self.height, self.width, False)
            img[(self.gy[idx], self.gx[idx])] = 1
            regions.append(img)
        regions = np.array(regions)
        regions_dict = {}
        for idx, region in enumerate(regions):
            regions_dict[idx] = region
        region_sum = np.sum(regions, axis=0)

        return region_sum, regions_dict


    def interactive_regions(self, selected_components=None):
        embedding = self.embedding
        '''
        if embedding.shape[1] == 2:
            selected_components = [0,1]
        elif selected_components is not None:
            if type(selected_components) == list:
                selected_components = selected_components
            else:
                raise ValueError("selected_components parameter must be od type list.")
        else:
            raise ValueError("'DimensionReductionRegions' was initialized with more than two components. Provide selected_components as list.")
        '''
        embedding = embedding[:, selected_components]

        if isinstance(self.data, pd.DataFrame):
            data = self.data.values
        else:
            data = self.data

        plot = InteractiveDimensionReductionRegionsSelector(data, embedding, self.gy, self.gx)
        plot.plot()

        regions = np.array(plot.return_rois())
        regions_dict = {}
        for idx, region in enumerate(regions):
                regions_dict[idx] = region
        region_sum = np.sum(regions, axis=0)

        return region_sum, regions_dict