import sys
import argparse
import numpy as np
import pandas as pd
import csv
from roi_predictor import ROIpredictor
from cluster_routines import cluster_routine, add_cluster_subparser
from save_procedure import save_procedure

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-r", "--readpath", type=str, required=True, help="Path to h5 file.")
parser.add_argument("-s", "--savepath", type=str, required=True, help="Path to save results.")

parser.add_argument("-p", "--regionprediction", type=str, required=True, choices=["dr", "mt", "mser"], help="Method to predict regions of interest. dr: Dimension Reduction, mt: Multi-Threshold, mser: Maximally Stable Extended Regions.")

parser.add_argument("--drpred", default=None, type=str, required="dr" in sys.argv, choices=["component_pred", "knn_pred", "interactive_pred", "return_drr"], help="Dimension Reduction prediction submethod. Only available if -p equals 'dr'.")
parser.add_argument("--drmethod", default=None, type=str, required="dr" in sys.argv, choices=["pca", "nmf", "lda", "ica", "umap", "tsne"], help="Dimension Reduction method. Only available if -p equals 'dr'.")
parser.add_argument("--components", default=None, type=int, nargs="+", required="components" in sys.argv, help="Number of components in list or single integer format.")
parser.add_argument("--embedding_nr", default=None, type=int,  nargs="+", required="components" in sys.argv, help="Number of components to use. All of --components per default.")
parser.add_argument("--components_method", default=None, type=str, required="component_pred" in sys.argv, choices=["cv", "mcv", "mgac", "contours_low", "contours_high"], help="Method for the dimension reduction components region prediction.")
parser.add_argument("--neighbors", default=None, type=float, required="knn" in sys.argv, help="Number of neighbors for the kNN graph. Zero will result in no impact.")
parser.add_argument("--radius", default=None, type=float, required="knn" in sys.argv, help="Radius for the rNN graph. Zero will result in no impact.")
parser.add_argument("--expansion_factor", default=None, type=float, required=False, help="Factor for the exponential increase of distances between data points in the calculated embedding.")

parser.add_argument("--nr_classes", type=int, default=False, required="mt" in sys.argv, help="Number of classes for multi-otsus thresholding.")
parser.add_argument("--classes", type=int, default=False, nargs="+", required="mt" in sys.argv, help="List of thresholds for multi thresholding.")

parser.add_argument("--mser_method", default=None, type=str, required="mser" in sys.argv, choices=["all", "level"], help="MSER region selecton method.")
parser.add_argument("--sequence_min", default=None, type=int, required="mser" in sys.argv, help="Minimum number of maximally extended images to cover a region. Has to be at least three.")
parser.add_argument("--delta", default=None, type=int, required="mser" in sys.argv, help="Step size to search for region sequences [1-255].")
parser.add_argument("--min_area", default=None, type=float, required="mser" in sys.argv, help="Minimal area size of a region to be accepted. Smaller than one will be considered as percentage.") # check if between 0 and 1 or above 1, above one casts to int
parser.add_argument("--max_area", default=None, type=float, required="mser" in sys.argv, help="Maximal area size of a region to be accepted. Smaller than one will be considered as percentage.") # check if between 0 and 1 or above 1, above one casts to int

parser.add_argument("-c", "--contour", type=str, required=True, choices=["cv", "mcv", "mgac", "morphology", "contours_low", "contours_high"], help="Active Contour Method.")
parser.add_argument("--fill_holes", required=False, action='store_true', help="Applies hole filling method.")

parser.add_argument("--load_clusterlabels", required=False, type=str, help="Path to a saved clustering (.npy or .csv).")
parser.add_argument("--delimiter", default=None, required=False, type=str, help="Type of .csv delimiter.")

parser.add_argument("--cluster_labels", default=[-1], type=int,  nargs="+", required=False, help="Choose the labels of the preceding clustering that should be considered. To select all labels use -1.")
parser.add_argument("--show_cluster_labels", required=False, action='store_true', help="Shows the cluster labels and quits the script. We advice to use --save_clustering if a pre defined cluster method is used!")
parser.add_argument("--aggregation_mode", default=None, type=str, choices=['mean', 'median', 'sum', 'prod', 'max', 'min'], required="clustering", help="Method to aggregate clustered image stacks. Choose from 'mean', 'median', 'sum', 'prod', 'max' or 'min'. Required for mt or mser.")

parser.add_argument("--save_roi_pixels", required=False, action='store_true', help="Save pixel positions of detected RoIs as .csv.")
parser.add_argument("--save_roi_plots", required=False, action='store_true', help="Save plots of selected RoIs.")
parser.add_argument("--save_roi_as_array", required=False, action='store_true', help="Save RoI images as .npy.")

add_cluster_subparser(parser)

args=parser.parse_args()

# Needed if clustering subparser is not used
try:
    args.cluster_method
except:
    args.cluster_method = None

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
    '''
    if len(args.components) == 1:
        kwargs["components"] = int(args.components[0])
    elif len(args.components) > 1:
        kwargs["components"] = args.components
    else:
        raise ValueError("--components has to be at least one integer.")
    '''

if args.embedding_nr is not None:
    kwargs["embedding_nr"] = args.embedding_nr

if args.components_method is not None:
    kwargs["components_method"] = args.components_method

if args.neighbors is not None:
    kwargs["n_neighbors"] = args.neighbors

if args.radius is not None:
    kwargs["radius"] = args.radius

if args.expansion_factor is not None:
    kwargs["expansion_factor"] = args.expansion_factor

if args.mser_method is not None:
    kwargs["mser_method"] = args.mser_method

if args.sequence_min is not None:
    kwargs["sequence_min"] = args.sequence_min

if args.delta is not None:
    kwargs["delta"] = args.delta

if args.min_area is not None:
    kwargs["min_area"] = args.min_area

if args.max_area is not None:
    kwargs["max_area"] = args.max_area


if args.embedding_nr is None and args.cluster_labels is None:
    raise ValueError("To use mt or mser provide either embedding_nr or cluster_labels.")



if args.regionprediction != "dr":
    images = []
    if (args.load_clusterlabels is not None) ^ (args.cluster_method is not None):
        if args.cluster_method is not None:
            cluster_labels = cluster_routine(dframe.values, args.n_clusters, args.cluster_method)
            if args.save_clustering is not None:
                if ".npy" in args.save_clustering:
                    np.save(args.save_clustering, cluster_labels)
                elif ".csv" in args.save_clustering:
                    np.savetxt(args.save_clustering, cluster_labels, delimiter=",")
                else:
                    raise ValueError("File must be .npy or .csv.")

        if args.load_clusterlabels is not None:
            if args.save_clustering is not None:
                print("Stored clustering was loaded from file. --save_clustering will be ignored!")
            if ".npy" in args.clustering:
                cluster_labels = np.load(args.clustering)
                if cluster_labels.ndim != 1:
                    raise ValueError("Loaded .npy cluster labels file is not one dimensional.")
            if ".csv" in args.clustering:
                if args.delimiter:
                    delimiter = args.delimiter
            cluster_labels = []
            with open(args.clustering, "w") as csvfile: 
                if args.delimiter is None:
                    dialect = csv.Sniffer().sniff(csvfile.read())
                    csvfile.seek(0)
                    delimiter = dialect.delimiter
                else:
                    delimiter = args.delimiter
                reader = csv.reader(csvfile, delimiter=delimiter)
                for row in reader:
                    cluster_labels.append(row)
            if len(cluster_labels) == 0:
                raise ValueError(".csv file seems to be empty.")
            if len(cluster_labels) > 1:
                raise ValueError("Labels in .csv file must be encoded in one line.")
            else:
                cluster_labels = cluster_labels[0]

        roi_pred.fit_clustering(memberships=cluster_labels, normalize=True)
        
        if args.show_cluster_labels:
            print("The selected Cluster method applied the following labels: " + str(roi_pred.memberships))
            exit(0)

        if args.cluster_labels[0] == -1:
            cluster_labels = cluster_labels
        else:
            cluster_labels = args.cluster_labels
        for lbl in sorted(list(set(cluster_labels))):
            images.append(roi_pred.data_dict[lbl]["grpimg"][args.aggregation_mode])
    else:
        raise ValueError("Choose one between a pre defined cluster method with --cluster_method or load a cluster method with --load_clusterlabels.")
    kwargs["images"] = images
else:
    pass
    '''
    if args.drmethod is None or args.components is None:
        raise ValueError("To use mt or mser on dimension reduction embedding images, drmethod and components must be provided.")
    drr = roi_pred._dr_routine("return_drr", args.drmethod, args.components)
    for nr in args.embedding_nr:
        images.append(drr.embedding_images(nr))
    '''






if args.regionprediction == "dr":
    results_dict = roi_pred.predict_rois("dr", kwargs)
elif args.regionprediction == "mt":
    results_dict = roi_pred.predict_rois("mt", kwargs)
elif args.regionprediction == "mser":
    results_dict = roi_pred.predict_rois("mser", kwargs)
else:
    raise ValueError("-p must be 'dr', 'mt' or 'mser'.")

if args.fill_holes:
    fill_method = args.contour
    filled_roi = roi_pred.fill_holes(results_dict, fill_method)


save_procedure(results_dict, args, sys.argv)




    









#https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.find_contours
#https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
#https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label
