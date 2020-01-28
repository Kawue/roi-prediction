import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def save_procedure(results_dict, args, argv):
    filename = os.path.basename(args.readpath).split(".h5")[0]
    # Save command line call as backup
    with open(os.path.join(args.savepath, "commandline" + ".txt"), "w") as f:
        print("python " + " ".join(argv), file=f)

    # filename, args.regionprediction, label, fileextension
    region_sum_str = "{}_{}_label{}_regionsum.{}"
    # filename, args.regionprediction, label, region_idx, fileextension
    region_str = "{}_{}_label{}_region{}.{}"
    
    # keys in results_dict are equivalent to cluster labels if clustering was applied. Otherwise its just an index.
    for label, result_dict in results_dict.items():
        # Save every sum of regions images
        region_sum = result_dict["region_sum"]
        if args.save_pixels:
            with open(os.path.join(args.savepath, region_sum_str.format(filename, args.regionprediction, label, "csv")), "w") as f:
                fieldnames = ["row_pixel", "column_pixel", "level"]
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=",")
                writer.writeheader()
                pixels = np.where(region_sum > 0)
                for row, col in list(zip(*pixels)):
                    level = region_sum[row,col]
                    writer.writerow({"row_pixel": row, "column_pixel": col, "level": level})
        if args.save_plots:
            plt.imsave(os.path.join(args.savepath, region_sum_str.format(filename, args.regionprediction, label, "png")), region_sum)
        if args.save_array:
            np.save(os.path.join(args.savepath, region_sum_str.format(filename, args.regionprediction, label, "npy")), region_sum)

        # Save every region images
        regions_savepath = os.path.join(args.savepath, filename + "_roi_images")
        if not os.path.exists(regions_savepath):
            os.makedirs(regions_savepath)

        regions_dict = result_dict["regions_dict"]
        if args.save_pixels:
            region_csv_str = "{}_{}_label{}.{}"
            with open(os.path.join(regions_savepath, region_csv_str.format(filename, args.regionprediction, label, "csv")), "w") as f:
                fieldnames = ["region_key", "row_pixel", "column_pixel"]
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=",")
                writer.writeheader()
                for region_idx, region in regions_dict.items():
                    pixels = np.where(region > 0)
                    for row, col in list(zip(*pixels)) :
                        writer.writerow({"region_key": region_idx, "row_pixel": row, "column_pixel": col})

                    if args.save_plots:
                        plt.imsave(os.path.join(regions_savepath, region_str.format(filename, args.regionprediction, label, region_idx, "png")), region)
                    if args.save_array:
                        np.save(os.path.join(regions_savepath, region_str.format(filename, args.regionprediction, label, region_idx, "npy")), region)