import sys
import argparse
import pandas as pd
from roi_predictor import ROIpredictor

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-r", "--readpath", type=str, required=True, help="Path to h5 file.")
parser.add_argument("-s", "--savepath", type=str, required=True, help="Path to save results.")

parser.add_argument("-p", "--regionprediction", type=str, required=True, choices=["dr", "mt", "mser"], help="Method to predict regions of interest. dr: Dimension Reduction, mt: Multi-Threshold, mser: Maximally Stable Extended Regions.")

parser.add_argument("--drpred", default=None, type=str, required="dr" in sys.argv, choices=["components", "knn", "interactive"], help="Dimension Reduction prediction submethod. Only available if -p equals 'dr'.")
parser.add_argument("--drmethod", default=None, type=str, required="dr" in sys.argv, choices=["pca", "nmf", "lda", "ica", "umap", "tsne"], help="Dimension Reduction method. Only available if -p equals 'dr'.")
parser.add_argument("--list", default=None, required="components" in sys.argv, action='store_true', help="If True --components expects a list of integers and a single integer otherwise. Only available if --drpred equals 'components'.")
parser.add_argument("--components", default=None, type=int, nargs="+", required="components" in sys.argv, help="Number of components in list or single integer format.") # check for [0] or whole list depending on --lists
parser.add_argument("--neighbors", default=None, type=float, required="knn" in sys.argv, help="Number of neighbors for the kNN graph. Zero will result in no impact.")
parser.add_argument("--radius", default=None, type=float, required="knn" in sys.argv, help="Radius for the rNN graph. Zero will result in no impact.")

parser.add_argument("--nr_classes", type=int, default=False, required="mt" in sys.argv, help="Number of classes for multi-otsus thresholding.")
parser.add_argument("--classes", type=int, default=False, nargs="+", required="mt" in sys.argv, help="List of thresholds for multi thresholding.")

parser.add_argument("-mserm", default=None, type=str, required="mser" in sys.argv, choices=["sum", "level", "all"], help="MSER region selecton method.")
parser.add_argument("--sequence_min", default=None, type=int, required="mser" in sys.argv, help="Minimum number of maximally extended images to cover a region.")
parser.add_argument("--delta", default=None, type=int, required="mser" in sys.argv, help="Step size to search for region sequences [1-255].")
parser.add_argument("--min_area", default=None, type=float, required="mser" in sys.argv, help="Minimal area size of a region to be accepted. Smaller than one will be considered as percentage.") # check if between 0 and 1 or above 1, above one casts to int
parser.add_argument("--max_area", default=None, type=float, required="mser" in sys.argv, help="Maximal area size of a region to be accepted. Smaller than one will be considered as percentage.") # check if between 0 and 1 or above 1, above one casts to int

parser.add_argument("-c", "--contour", type=str, required=True, choices=["cv", "mcv", "ac", "mgac", "morphology", "props"], help="Active Contour Method.")
parser.add_argument("--fill_holes", required=False, action='store_true', help="Applies hole filling method.")

args=parser.parse_args()

if "mt" in sys.argv:
    if not args.nrClasses and not args.classes:
        raise ValueError("If -p equals 'mt', then --nr_classes or --classes must be given.")
    if args.nr_classes and args.classes:
        raise ValueError("Choose either --nr_classes or --classes. Both are not allowed.")

if "mser" in sys.argv:
    if args.sequence_min < 3:
        raise ValueError("--sequence_min has to be bigger than two.")
    if not 0 < args.delta < 256:
        raise ValueError("--delta must be in range of [1,255]")

if args.nr_classes:
    mt_classes = args.nr_classes
elif args.classes:
    mt_classes = args.classes
else:
    mt_classes = None


dframe = pd.read_hdf(args.readpath)
roi_pred = ROIpredictor(dframe)

kwargs = {}

if mt_classes is not None:
    kwargs["thresholds"] = mt_classes

if args.drmethod is not None:
    kwargs["dr_method"] = args.drmethod

if args.drpred is not None:
    kwargs["pred_method"] = args.drpred

if args.components is not None:
    kwargs["components"] = args.components

#if args.embedding_nr is not None:
#    kwargs["embedding_nr"] = args.embedding_nr

if args.neighbors is not None:
    kwargs["n_neighbors"] = args.neighbors

if args.radius is not None:
    kwargs["radius"] = args.radius

#if args.expansion_factor is not None:
#    kwargs["expansion_factor"] = args.expansion_factor

if args.sequence_min is not None:
    kwargs["sequence_min"] = args.sequence_min

if args.delta is not None:
    kwargs["delta"] = args.delta

if args.min_area is not None:
    kwargs["min_area"] = args.min_area

if args.max_area is not None:
    kwargs["max_area"] = args.max_area

#if args.img is not None:
#    kwargs["img"] = args.img


if args.regionprediction == "dr":
    region_sum, regions = roi_pred.predict_rois("dr", kwargs)
elif args.regionprediction == "mt":
    region_sum, regions = roi_pred.predict_rois("mt", kwargs)
elif args.regionprediction == "mser":
    region_sum, regions = roi_pred.predict_rois("mser", kwargs)
else:
    raise ValueError("-p must be 'dr', 'mt' or 'mser'.")

if args.fill_holes:
    roi = regions[0]
    fill_method = args.contour
    filled_roi = roi_pred.fill_holes(roi, fill_method)

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(regions[0])
plt.figure()
plt.imshow(filled_roi)
plt.show()








#https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.find_contours
#https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
#https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label
