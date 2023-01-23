from skimage.segmentation import chan_vese, morphological_chan_vese, morphological_geodesic_active_contour, active_contour, inverse_gaussian_gradient
from skimage.measure import find_contours, label, regionprops
from skimage.morphology import convex_hull_image, binary_erosion
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
def calc_ac(img, method, init_img=None):
    if init_img is None:
        init_img = img
    if method == "cv":
        region = chan_vese(img, mu=1, tol=0.05, max_iter=5000, init_level_set=init_img)
    elif method == "mcv":
        region = morphological_chan_vese(img, iterations=1000, lambda1=1.0, lambda2=1.0, smoothing=2, init_level_set=init_img)
    elif method == "ac":
        raise ValueError("Not implemented!")
    elif method == "mgac":
        gimg = inverse_gaussian_gradient(img)
        region = morphological_geodesic_active_contour(gimg, iterations=1000, smoothing=2, balloon=0, init_level_set=img)
    elif method == "contours_low":
        # For this method init_img needs to be the intensity image, while img is binary.
        lbl = label(img)
        props = regionprops(lbl, intensity_image=init_img)
        filled_contours = np.zeros_like(img)
        for prop in props:
            #sr,sc,er,ec = prop.bbox
            contour_target = np.zeros_like(init_img)
            contour_target[prop.coords[:,0], prop.coords[:,1]] += init_img[prop.coords[:,0], prop.coords[:,1]]
            if prop.area > 4:
                snake = find_contours(contour_target, level=(np.amax(contour_target) - np.amin(contour_target)) / 2, fully_connected='low', positive_orientation='low')
                snake_img = np.zeros_like(contour_target)
                if len(snake) > 0:
                    max_snake_idx = np.argmax([len(x) for x in snake])
                    snake_img = np.zeros_like(contour_target)
                    snake_coords = snake[max_snake_idx].astype(int)
                    snake_img[snake_coords[:,0], snake_coords[:,1]] += 1
                    filled_contours += ndimage.morphology.binary_fill_holes(snake_img)
        region = filled_contours
    elif method == "contours_high":
        # For this method init_img needs to be the intensity image, while img is binary.
        lbl = label(img)
        props = regionprops(lbl, intensity_image=init_img)
        filled_contours = np.zeros_like(img)
        for prop in props:
            #sr,sc,er,ec = prop.bbox
            contour_target = np.zeros_like(init_img)
            contour_target[prop.coords[:,0], prop.coords[:,1]] += init_img[prop.coords[:,0], prop.coords[:,1]]
            if prop.area > 4:
                snake = find_contours(contour_target, level=(np.amax(contour_target) - np.amin(contour_target)) / 2, fully_connected='high', positive_orientation='high')
                snake_img = np.zeros_like(contour_target)
                if len(snake) > 0:
                    max_snake_idx = np.argmax([len(x) for x in snake])
                    snake_img = np.zeros_like(contour_target)
                    snake_coords = snake[max_snake_idx].astype(int)
                    snake_img[snake_coords[:,0], snake_coords[:,1]] += 1
                    filled_contours += ndimage.morphology.binary_fill_holes(snake_img)
        region = filled_contours
    else:
        raise ValueError("Method needs to be 'cv', 'mcv', 'mgac', 'contours_low' or 'contours_high'.")
    
    return region