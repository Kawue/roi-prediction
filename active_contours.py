from skimage.segmentation import chan_vese, morphological_chan_vese, morphological_geodesic_active_contour, active_contour, inverse_gaussian_gradient
from skimage.measure import find_contours
from skimage.morphology import convex_hull_image, binary_erosion
import matplotlib.pyplot as plt
import numpy as np
def calc_ac(img, method):
    if method == "cv":
        region = chan_vese(img, mu=0.1, tol=0.0005, max_iter=5000)#, init_level_set=img
    elif method == "mcv":
        region = morphological_chan_vese(img, iterations=1000, lambda1=1.0, lambda2=1.0, smoothing=1, init_level_set=img)
    elif method == "ac":
        snake = find_contours(img, level=1)
        hull_img = convex_hull_image(img)
        shrinked_hull = binary_erosion(hull_img).astype(int)
        snake = np.array(list(zip(*np.where((hull_img - shrinked_hull) == 1))))
        print(snake)
        opt_snake = np.round(active_contour(img, snake, alpha=0.01, beta=0.1, w_line=0, w_edge=1, gamma=0.01, coordinates='rc'), 0).astype(int)
        snake_img = np.zeros_like(img)
        print(opt_snake)
        snake_img[(opt_snake[:,0], opt_snake[:,1])] = 1
        region = snake_img
    elif method == "mgac":
        gimg = inverse_gaussian_gradient(img)
        region = morphological_geodesic_active_contour(gimg, iterations=1000, smoothing=1, balloon=0, init_level_set=img)
    elif method == "props":
        lbl = label(img)
        props = regionprops(lbl)
        ?????
    else:
        raise ValueError("Method needs to be 'cv', 'mcv', 'ac' or 'mgac'.")
    
    return region


    