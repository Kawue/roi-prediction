from sys import argv
import os
import argparse
import numpy as np
import pandas as pd
import csv
from roi_predictor import ROIpredictor
from cluster_routines import cluster_routine
from save_procedure import save_procedure
from dimension_reduction_regions import DimensionReductionRegions
from utils import normalization

subparser = ["--dimensionreduction", "--multithreshold", "--mser", "--clustering", "--save"]
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--fullhelp", required=False, action='store_true', help="Prints full help of all subparsers.")
parser.add_argument("--subhelp", required=False, choices=subparser, help="Prints help of specified subparsers.")
parser.add_argument("-r", "--readpath", type=str, required=True, help="Path to h5 file.")
parser.add_argument("-s", "--savepath", type=str, required=True, help="Path to save results.")
parser.add_argument("-p", "--regionprediction", type=str, required=True, choices=["dr", "mt", "mser"], help="Method to predict regions of interest. dr: Dimension Reduction, mt: Multi-Threshold, mser: Maximally Stable Extended Regions.")
parser.add_argument("--fill_holes", required=False, action='store_true', help="Applies hole filling method.")
parser.add_argument("-c", "--contour", type=str, required="--fill_holes" in argv, choices=["cv", "mcv", "mgac", "morphology", "contours_low", "contours_high"], help="Active Contour Method for hole filling.")

parser.add_argument("--dimensionreduction", required=False, action='store_true', help="Flag to make the dimension reduction options available.")
parser.add_argument("--multithreshold", required=False, action='store_true', help="Flag to make the multi-threshold options available.")
parser.add_argument("--mser", required=False, action='store_true', help="Flag to make the mser options available.")
parser.add_argument("--clustering", required=("--mser" in argv or "--multithreshold" in argv) and "--drmethod" not in argv, action='store_true', help="Flag to make the clustering options available (needed for mt and mser).")
parser.add_argument("--save", required=False, action='store_true', help="Flag to make the save options available.")

if "--fullhelp" in argv:
    argv  += subparser
else:
    if "--subhelp" in argv:
        argv += [x for x in subparser if x in argv]


if "-h" not in argv and "--fullhelp" not in argv and "--subhelp" not in argv:
    prevent_parser_error = sum(["--dimensionreduction" in argv, "--multithreshold" in argv, "--mser" in argv])
    if prevent_parser_error > 1:
        raise ValueError("Choose only one RoI prediction method from: '--dimensionreduction', '--multithreshold', '--mser'.")
    if prevent_parser_error == 0:
        raise ValueError("Choose one of the RoI prediction methods from: '--dimensionreduction', '--multithreshold', '--mser'.")

if "--dimensionreduction" in argv:
    parser.add_argument("--drpred", default=None, type=str, required="--dimensionreduction" in argv, choices=["component_pred", "knn_pred", "interactive_pred"], help="Dimension Reduction prediction submethod. Only available if -p equals 'dr'.")
    parser.add_argument("--drmethod", default=None, type=str, required="--dimensionreduction" in argv, choices=["pca", "nmf", "lda", "ica", "umap", "tsne"], help="Dimensionreduction: Dimension Reduction method.")
    parser.add_argument("--components", default=None, type=int, required="--dimensionreduction" in argv, help="Dimensionreduction: Number of components in list or single integer format.")
    parser.add_argument("--embedding_nr", default=None, type=int,  nargs="+", required="--dimensionreduction" in argv, help="Dimensionreduction: Number of components to use. -1 uses all components.")
    parser.add_argument("--components_method", default=None, type=str, required="component_pred" in argv, choices=["cv", "mcv", "mgac", "contours_low", "contours_high"], help="Dimensionreduction: Method for the dimension reduction components region prediction.")
    parser.add_argument("--neighbors", default=None, type=int, required="knn" in argv, help="Dimensionreduction: Number of neighbors for the kNN graph. Zero will result in no impact.")
    parser.add_argument("--radius", default=None, type=float, required="knn" in argv, help="Dimensionreduction: Radius for the rNN graph. Zero will result in no impact.")
    parser.add_argument("--expansion_factor", default=None, type=float, required=False, help="Dimensionreduction: Factor for the exponential increase of distances between data points in the calculated embedding.")

if "--multithreshold" in argv:
    parser.add_argument("--nr_classes", type=int, default=False, required="--multithreshold" in argv and "--classes" not in argv, help="Multithreshold: Number of classes for multi-otsus thresholding.")
    parser.add_argument("--classes", type=int, default=False, nargs="+", required="--multithreshold" in argv and "--nr_classes" not in argv, help="Multithreshold: List of thresholds for multi thresholding.")
    parser.add_argument("--drmethod_mt", default=None, type=str, required=False in argv, choices=["pca", "nmf", "lda", "ica", "umap", "tsne"], help="Dimensionreduction: Dimension Reduction method. (If active, 'mt' will be executed on the embedding images.)")
    parser.add_argument("--components_mt", default=None, type=int, required="--drmethod" in argv, help="Dimensionreduction: Number of components in list or single integer format. (Only required in combination with dr-method.)")
    parser.add_argument("--embedding_nr_mt", default=None, type=int,  nargs="+", required="--drmethod" in argv, help="Dimensionreduction: Number of components to use. -1 uses all components. (Only required in combination with dr-method.)")

if "--mser" in argv:
    parser.add_argument("--mser_method", default=None, type=str, required=True, choices=["all", "level"], help="MSER: MSER region selecton method.")
    parser.add_argument("--sequence_min", default=None, type=int, required="--mser" in argv, help="MSER: Minimum number of maximally extended images to cover a region. Has to be at least three.")
    parser.add_argument("--delta", default=None, type=int, required="--mser" in argv, help="MSER: Step size to search for region sequences [1-255].")
    parser.add_argument("--min_area", default=None, type=float, required="--mser" in argv, help="MSER: Minimal area size of a region to be accepted. Smaller than one will be considered as percentage.") # check if between 0 and 1 or above 1, above one casts to int
    parser.add_argument("--max_area", default=None, type=float, required="--mser" in argv, help="MSER: Maximal area size of a region to be accepted. Smaller than one will be considered as percentage.") # check if between 0 and 1 or above 1, above one casts to int
    parser.add_argument("--drmethod_mser", default=None, type=str, required=False in argv, choices=["pca", "nmf", "lda", "ica", "umap", "tsne"], help="Dimensionreduction: Dimension Reduction method. (If active, 'mt' will be executed on the embedding images.)")
    parser.add_argument("--components_mser", default=None, type=int, required="--drmethod" in argv, help="Dimensionreduction: Number of components in list or single integer format. (Only required in combination with dr-method.)")
    parser.add_argument("--embedding_nr_mser", default=None, type=int,  nargs="+", required="--drmethod" in argv, help="Dimensionreduction: Number of components to use. -1 uses all components. (Only required in combination with dr-method.)")

if "--clustering" in argv:
    parser.add_argument("--aggregation_mode", default=None, type=str, choices=['mean', 'median', 'sum', 'prod', 'max', 'min'], required="clustering" in argv, help="Clustering: Method to aggregate clustered image stacks. Choose from 'mean', 'median', 'sum', 'prod', 'max' or 'min'. Required for mt or mser.")
    parser.add_argument("--cluster_method", required="--clustering" in argv and "--load_clusterlabels" not in argv, choices=["AgglomerativeClustering", "kMeans"], help="Clustering: Cluster method.")
    parser.add_argument("--n_clusters", required="--cluster_method" in argv, type=int, help="Clustering: Number of Clusters.")
    parser.add_argument("--save_clustering", default=None, required=False, type=str, help="Clustering: Path + filename to save a cluster result if a pre defined cluster method was used. Must contain .npy or .csv.")
    parser.add_argument("--metric", required=False, help="Clustering: Metric for distance computation (see scipy's pdist).")
    parser.add_argument("--linkage", required=False, help="Clustering: Linkage Method ('ward', 'complete', 'average', 'single') for AgglomerativeClustering(). 'ward' requires 'euclidean' as metric.")
    parser.add_argument("--load_clusterlabels", required=False, type=str, help="Clustering: Path to a saved clustering (.npy or .csv).")
    parser.add_argument("--delimiter", default=None, required=False, type=str, help="Clustering: Type of .csv delimiter.")
    parser.add_argument("--cluster_labels", default=[-1], type=int, nargs="+", required=False, help="Clustering: Choose the labels of the preceding clustering that should be considered. To select all labels use -1.")
    parser.add_argument("--show_cluster_labels", required=False, action='store_true', help="Clustering: Shows the cluster labels and quits the script. We advice to use --save_clustering if a pre defined cluster method is used!")

if "--save" in argv:
    parser.add_argument("--save_pixels", required=False, action='store_true', help="Save: Save pixel positions of detected RoIs as .csv.")
    parser.add_argument("--save_plots", required=False, action='store_true', help="Save: Save plots of selected RoIs.")
    parser.add_argument("--save_array", required=False, action='store_true', help="Save: Save RoI images as .npy.")


if "--fullhelp" in argv or "--subhelp" in argv:
    parser.print_help()
    exit(0)

args=parser.parse_args()

if not os.path.exists(args.savepath):
    os.makedirs(args.savepath)


# Needed if clustering subparser is not used
try:
    args.cluster_method
except:
    args.cluster_method = None

if "mt" in argv:
    if not args.nr_classes and not args.classes:
        raise ValueError("If -p equals 'mt', then --nr_classes or --classes must be given.")
    if args.nr_classes and args.classes:
        raise ValueError("Choose either --nr_classes or --classes. Both are not allowed.")

if "mser" in argv:
    if args.sequence_min < 3:
        raise ValueError("--sequence_min has to be bigger than two.")
    if not 0 < args.delta < 256:
        raise ValueError("--delta must be in range of [1,255]")


dframe = pd.read_hdf(args.readpath)
roi_pred = ROIpredictor(dframe)


kwargs = {}

if "--dimensionreduction" in argv:
    if args.drmethod is not None:
        kwargs["dr_method"] = args.drmethod

    if args.drpred is not None:
        kwargs["pred_method"] = args.drpred
        '''
        if len(args.components) == 1:
            kwargs["components"] = int(args.components[0])
        elif len(args.components) > 1:
            kwargs["components"] = args.components
        else:
            raise ValueError("--components has to be at least one integer.")
        '''
    if args.embedding_nr is not None:
        if -1 in args.embedding_nr:
            kwargs["embedding_nr"] = list(range(args.components))
        else:
            kwargs["embedding_nr"] = args.embedding_nr
    
    if args.components is not None:
        kwargs["components"] = args.components

    if args.components_method is not None:
        kwargs["components_method"] = args.components_method

    if args.neighbors is not None:
        kwargs["n_neighbors"] = args.neighbors

    if args.radius is not None:
        kwargs["radius"] = args.radius

    if args.expansion_factor is not None:
        kwargs["expansion_factor"] = args.expansion_factor


if "--multithreshold" in argv:
    if args.nr_classes:
        mt_classes = args.nr_classes
    elif args.classes:
        mt_classes = args.classes
    else:
        mt_classes = None
        
    if mt_classes is not None:
        kwargs["thresholds"] = mt_classes

    if args.components_mt is not None:
        kwargs["components"] = args.components_mt

    if args.embedding_nr_mt is not None:
        if -1 in args.embedding_nr_mt:
            kwargs["embedding_nr"] = list(range(args.components_mt))
        else:
            kwargs["embedding_nr"] = args.embedding_nr_mt

    if args.drmethod_mt is not None:
        kwargs["dr_method"] = args.drmethod_mt


    


if "--mser" in argv:
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

    if args.drmethod_mser is not None:
        kwargs["dr_method"] = args.drmethod_mser

    if args.embedding_nr_mser is not None:
        if -1 in args.embedding_nr_mser:
            kwargs["embedding_nr"] = list(range(args.components_mser))
        else:
            kwargs["embedding_nr"] = args.embedding_nr_mser
    
    if args.components_mser is not None:
        kwargs["components"] = args.components_mser



#hier fehlt noch, dass man auch drr embedding als images für mser oder mt haben kann
if args.regionprediction != "dr":
    try:
        dr_flag = args.drmethod_mser
    except:
        dr_flag = args.drmethod_mt
    if dr_flag is not None:
        drr = DimensionReductionRegions(data=dframe, dr_method=kwargs["dr_method"], components=kwargs["components"], embedding_nr=kwargs["embedding_nr"])
        images = [normalization(drr.embedding_images(i)) for i in kwargs["embedding_nr"]]
    else:
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

            if -1 in args.cluster_labels:
                cluster_labels = cluster_labels
            else:
                cluster_labels = args.cluster_labels
            for lbl in sorted(list(set(cluster_labels))):
                images.append(roi_pred.data_dict[lbl]["grpimg"][args.aggregation_mode])
        else:
            raise ValueError("Choose one between a pre defined cluster method with --cluster_method or load a cluster method with --load_clusterlabels.")
    kwargs["images"] = images
    
    
    






if args.regionprediction == "dr":
    results_dict = roi_pred.predict_rois("dr", kwargs)
else:
    if "dr_method" in kwargs:
        del kwargs["dr_method"]
    if "embedding_nr" in kwargs:
        del kwargs["embedding_nr"]
    if "components" in kwargs:
        del kwargs["components"]

    if args.regionprediction == "mt":
        results_dict = roi_pred.predict_rois("mt", kwargs)
    elif args.regionprediction == "mser":
        results_dict = roi_pred.predict_rois("mser", kwargs)
    else:
        raise ValueError("-p must be 'dr', 'mt' or 'mser'.")

if args.fill_holes:
    fill_method = args.contour
    filled_roi = roi_pred.fill_holes(results_dict, fill_method)

if "--save" in argv:
    save_procedure(results_dict, args, argv)




    









#https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.find_contours
#https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
#https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label
