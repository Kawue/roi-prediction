import numpy as np
import scipy as sp
from os import listdir, makedirs
from os.path import join, exists
import matplotlib.pyplot as plt
from imageio import imread, imwrite
from skimage import img_as_float, img_as_ubyte
from skimage.morphology import opening, closing, disk
import skimage.filters as skif
from sklearn.metrics import silhouette_score
from scipy.stats.mstats import winsorize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from kneed import KneeLocator
from sklearn.metrics import calinski_harabaz_score, davies_bouldin_score, silhouette_score
#from cv2_rolling_ball import subtract_background_rolling_ball

from measures.gradient_helper import gradient_map, magnitude_map, direction_map

from measures.ims_similarity_preprocessing import pxpxvariationreduction
from measures.maximally_stable_regions import calc_mser
from measures.scale_space_helper import scale_space_gauss, scale_space_dog


def read_images(dirpath, padding):
    imgs = []
    labels = []
    for filename in listdir(dirpath):
        if len(filename.split(".")) > 1:
            img = imread(join(dirpath, filename), pilmode="L")
            img = img_as_float(img)
            img = np.pad(img, padding, "constant", constant_values=0)
            imgs.append(img)
            filename = filename.split(".")[0]
            labels.append(filename)
    return imgs, labels


def preprocess_images(imgs, index_mask, mask_img):
    pp_imgs = []
    for X in imgs:
        Xs = pxpxvariationreduction(X, index_mask, mask_img, "otsus")
        Xs = pxpxvariationreduction(Xs, index_mask, mask_img, "scalespace")
        winsorize(Xs, limits=(0, 0.01), inplace=True)
        #Xs = Xs*mask_img
        pp_imgs.append(Xs)
    return pp_imgs

def preprocess_scalespace(ppimgs, method, steps, index_mask, mask_img):
    scalespace_list = [[] for i in range(steps)]
    for X in ppimgs:
        if method == "gauss":
            img_space = scale_space_gauss(X, downscale=1, steps=steps-1, sigma=0.8)
        elif method == "dog":
            img_space = scale_space_dog(X, downscale=1, steps=steps-1, sigma=0.8)
        else:
            raise ValueError("Method has to be 'gauss' or 'dog'.")
        for idx, img in enumerate(img_space):
            scalespace_list[idx].append(img)
    return np.array(scalespace_list)


# Clustering
def cluster(dmatrix, nr_cluster, method_name, savepath):
    cond_dmatrix = squareform(dmatrix)
    Z = linkage(cond_dmatrix, method="average", optimal_ordering=True)
    #memb = fcluster(Z, t=nr_cluster, criterion="maxclust")
    #memb = fcluster(Z, t=1.1, criterion="inconsistent")
    #memb = fcluster(Z, t=0.3, criterion="distance")
    if nr_cluster == "auto":
        t_cluster = 1.0
        #memb = fcluster(Z, t=t_cluster, criterion="inconsistent")
        memb = fcluster(Z, t=0.3, criterion="distance")
    else:    
        memb = fcluster(Z, t=nr_cluster, criterion="maxclust")
    with open(join(savepath, method_name) + ".txt", "a") as txt:
        txt.write("Clustering Method: Hierarchical Clustering (Scipy)\n")
        if nr_cluster == "auto":
            txt.write("Number of Clusters was set to scipys inconsistent criterion with a threshold of: %d .\n" % t_cluster)
        else:
            txt.write("Number of Clusters %d decided by the median of optimal Silhouette Score, Within Cluster Sum of Squares, Between Cluster Sum of Squares of all similarity techniques.\n" % nr_cluster)
        txt.write("\n")
    return memb, Z


# Save clustered images into one folder per cluster
def save_clustered_images(imgs, labels, memb, savepath):
    nr_cluster = np.amax(memb)
    for i in range(1, nr_cluster+1):
        idx = np.where(memb == i)[0]
        for j in idx:
            if not exists(join(savepath, "c" + str(i))):
                makedirs(join(savepath, "c" + str(i)))
            plt.imsave(join(savepath, "c" + str(i) , labels[j] + ".png"), imgs[j], vmin=0, vmax=1)


# List of array with similarity values within clusters, cluster labels, method name
def boxplot(c_arrays, memb, savepath, method_name):
    fig, ax = plt.subplots()
    plt.boxplot(c_arrays, whis=[5,95], showmeans=True, meanline=True)
    plt.title("Intra-Cluster Distance Values: " + method_name)
    ax.set_xticklabels(["%s\n$v$=%d\n$n$=%d" % (i, len(v), np.where(memb==i)[0].size) for i,v in enumerate(c_arrays, start=1)])
    plt.xlabel("Clusters")
    plt.ylabel("Distance Values")
    plt.savefig(join(savepath, "boxplot-" + method_name))


# Prepare arrays containing the similarity measure for each pair of one cluster once.
def prep_cluster_similarity_array(dmatrix, memb):
    c_arrays = []
    for i in range(1, np.amax(memb)+1):
        idx = np.where(memb == i)[0]
        sub_dmatrix = dmatrix[idx[:, None], idx]
        c_arrays.append(squareform(sub_dmatrix))
    return c_arrays


# Rescale similarities / distances in a range of [0,1]
def rescale_similarities(dmatrix, distance):
    if not isinstance(distance, bool):
        raise ValueError("Distance parameter has to be boolen. Choose True if a distance matrix is used and False if a similarity matrix is used.")
    mmax = np.amax(dmatrix)
    mmin = np.amin(dmatrix)
    dmatrix = (dmatrix-mmin) / (mmax-mmin)
    return dmatrix

# Bring k-medoids output on par with hierarchical clustering output
def process_kmedoids_memb(centroids):
    codes = {val: idx for idx, val in enumerate(set(centroids), start=1)}
    return np.array([codes[val] for val in centroids])

# Bring dbscan output on par with hierarchical clustering output
def process_dbscan_memb(labels):
    labels = np.array(labels)
    noise_idx = np.where(labels==-1)[0]
    for label, noise_idx in enumerate(noise_idx, start=np.amax(labels)+1):
        labels[noise_idx] = label
    return labels+1



def grad_img(img):
    dy, dx = gradient_map(img)
    return magnitude_map(dy, dx)

def ori_img(img):
    dy, dx = gradient_map(img)
    return direction_map(dy, dx) + 180



def cluster_evaluation_stats(dmatrix, c_arrays, memb, pp_imgs, method_name, eval_object):
    cl_object = {}

    pp_imgs = np.array(pp_imgs)

    ########## Statistics ##########

    # Calculate minimal and maximal similarity, i.e. range of values
    sim_vals = squareform(dmatrix)
    min_sim = np.amin(sim_vals)
    max_sim = np.amax(sim_vals)
    #range_sim = (min_sim, max_sim)

    mean_dists = np.array([np.mean(x) for x in c_arrays if x.size > 0])
    median_dists = np.array([np.median(x) for x in c_arrays if x.size > 0])
    std_dists = np.array([np.std(x) for x in c_arrays if x.size > 0])
    cluster_sizes = np.array([len(np.where(memb==i)[0]) for i in range(1, np.amax(memb)+1)])
    min_cl_size = np.amin(cluster_sizes)
    max_cl_size = np.amax(cluster_sizes)
    #mean_cl_size = np.mean(cluster_sizes)
    #median_cl_size = np.median(cluster_sizes)
    std_cl_size = np.std(cluster_sizes)

    # Calculate minimal and maximal similarity across all clusterings, i.e. range of clustered values
    #min_cl_sim = np.amin(np.concatenate(c_arrays))
    max_cl_sim = np.amax(np.concatenate(c_arrays))
    #range_cl_sim = (min_cl_sim, max_cl_sim)
    mean_dist_cl = np.mean(mean_dists)
    median_dist_cl = np.median(median_dists)
    mean_std_dist_cl = np.mean(std_dists)
    mean_std_dist_cl_norm_size = np.mean(std_dists/cluster_sizes[np.where(cluster_sizes > 1)])

    mads = []
    msds = []
    pp_imgs_mag = np.array([grad_img(img) for img in pp_imgs])
    mads_mag = []
    msds_mag = []
    pp_imgs_ori = np.array([ori_img(img) for img in pp_imgs])
    mads_ori = []
    msds_ori = []
    for grp in range(1, np.amax(memb)+1):
        grp_imgs = pp_imgs[np.where(memb == grp)]
        #mean_img = np.mean(grp_imgs, axis=0)
        #median_img = np.median(grp_imgs, axis=0)
        std_img = np.std(grp_imgs, axis=0)
        mad_img = sp.stats.median_absolute_deviation(grp_imgs, axis=0, center=np.mean)

        sgl_idx = np.where(np.sum(grp_imgs, axis=0) > 0)
        
        msd = np.mean(std_img[sgl_idx])
        msds.append(msd)

        #mad = np.mean([np.abs(mean_img - img)[sgl_idx] for img in grp_imgs])
        mad = np.mean(mad_img[sgl_idx])
        mads.append(mad)


        ############### Magnitude ##################
        grp_imgs_mag = pp_imgs_mag[np.where(memb == grp)]
        #mean_img = np.mean(grp_imgs, axis=0)
        #median_img = np.median(grp_imgs, axis=0)
        std_img_mag = np.std(grp_imgs_mag, axis=0)
        mad_img_mag = sp.stats.median_absolute_deviation(grp_imgs_mag, axis=0, center=np.mean)

        sgl_idx_mag = np.where(np.sum(grp_imgs_mag, axis=0) > 0)
        
        msd_mag = np.mean(std_img_mag[sgl_idx_mag])
        msds_mag.append(msd_mag)

        #mad = np.mean([np.abs(mean_img - img)[sgl_idx] for img in grp_imgs])
        mad_mag = np.mean(mad_img_mag[sgl_idx_mag])
        mads_mag.append(mad_mag)


        ############### Orientation ##################
        grp_imgs_ori = pp_imgs_ori[np.where(memb == grp)]
        #mean_img = np.mean(grp_imgs, axis=0)
        #median_img = np.median(grp_imgs, axis=0)
        std_img_ori = np.std(grp_imgs_ori, axis=0)
        mad_img_ori = sp.stats.median_absolute_deviation(grp_imgs_ori, axis=0, center=np.mean)

        sgl_idx_ori = np.where(np.sum(grp_imgs_ori, axis=0) > 0)
        
        msd_ori = np.mean(std_img_ori[sgl_idx_ori])
        msds_ori.append(msd_ori)

        #mad = np.mean([np.abs(mean_img - img)[sgl_idx] for img in grp_imgs])
        mad_ori = np.mean(mad_img_ori[sgl_idx_ori])
        mads_ori.append(mad_ori)

    mads = np.array(mads)
    msds = np.array(msds)

    mads_mag = np.array(mads_mag)
    msds_mag = np.array(msds_mag)

    mads_ori = np.array(mads_ori)
    msds_ori = np.array(msds_ori)

    max_msd = np.amax(msds)
    max_mad = np.amax(mads)

    max_msd_mag = np.amax(msds_mag)
    max_mad_mag = np.amax(mads_mag)

    max_msd_ori = np.amax(msds_ori)
    max_mad_ori = np.amax(mads_ori)

    mean_msd = np.mean(msds)
    msd_idx = np.where(cluster_sizes > 1)[0]
    mean_msd_norm = np.mean(msds[msd_idx])
    #mean_msd_norm_size = np.mean(msds[msd_idx]/cluster_sizes[msd_idx])

    mean_msd_mag = np.mean(msds_mag)
    msd_mag_idx = np.where(cluster_sizes > 1)[0]
    mean_msd_mag_norm = np.mean(msds_mag[msd_mag_idx])

    mean_msd_ori = np.mean(msds_ori)
    msd_ori_idx = np.where(cluster_sizes > 1)[0]
    mean_msd_ori_norm = np.mean(msds_ori[msd_ori_idx])



    mean_mad = np.mean(mads)
    mad_idx = np.where(cluster_sizes > 1)[0]
    mean_mad_norm = np.mean(mads[mad_idx])
    #mean_mad_norm_size = np.mean(mads[mad_idx]/cluster_sizes[mad_idx])

    mean_mad_mag = np.mean(mads_mag)
    mad_mag_idx = np.where(cluster_sizes > 1)[0]
    mean_mad_mag_norm = np.mean(mads_mag[mad_mag_idx])

    mean_mad_ori = np.mean(mads_ori)
    mad_ori_idx = np.where(cluster_sizes > 1)[0]
    mean_mad_ori_norm = np.mean(mads_ori[mad_ori_idx])


    ########## Clusterindices ##########
    shape2d = (pp_imgs.shape[0], pp_imgs.shape[1]*pp_imgs.shape[2])
    shs = silhouette_score(dmatrix, labels=memb, metric="precomputed") # 1: indicate best assignments, -1: indicate wrong assignments, 0: indicate overlapping clusters
    chs = calinski_harabaz_score(pp_imgs.reshape(shape2d), memb) # higher better
    dbs = davies_bouldin_score(pp_imgs.reshape(shape2d), memb) # lower better
    wcss_dist = wcss(dmatrix, memb)
    bcss_dist = bcss(dmatrix, memb)
    #fstat_dist = fstat(dmatrix, memb)
    #wcss_sample = wcss_samples(pp_imgs, memb)
    #bcss_sample = bcss_samples(pp_imgs, memb)
    #fstat_sample = fstat_samples(pp_imgs, memb)


    #cl_object["mean_cl_size"] = mean_cl_size
    #cl_object["median_cl_size"] = median_cl_size
    cl_object["std_cl_size"] = std_cl_size
    cl_object["min_cl_size"] = min_cl_size
    cl_object["max_cl_size"] = max_cl_size
    cl_object["min_dist(l)"] = min_sim
    cl_object["max_dist(h)"] = max_sim
    #cl_object["min_cl_dist(l)"] = min_cl_sim
    cl_object["max_cl_dist(l)"] = max_cl_sim
    cl_object["mean_dist_cl(l)"] = mean_dist_cl
    cl_object["median_dist_cl(l)"] = median_dist_cl
    cl_object["mean_std_dist_cl(l)"] = mean_std_dist_cl
    cl_object["mean_std_dist_cl_norm_size(l)"] = mean_std_dist_cl_norm_size
    cl_object["mean_msd(l)"] = mean_msd
    cl_object["mean_msd_norm(l)"] = mean_msd_norm
    #cl_object["mean_msd_norm_size(l)"] = mean_msd_norm_size
    cl_object["max_msd(l)"] = max_msd
    cl_object["mean_mad(l)"] = mean_mad
    cl_object["mean_mad_norm(l)"] = mean_mad_norm
    #cl_object["mean_mad_norm_size(l)"] = mean_mad_norm_size
    cl_object["max_mad(l)"] = max_mad
    cl_object["silhouette_score(h)"] = shs
    cl_object["calinski_harabaz_score(h)"] = chs
    cl_object["davies_bouldin_score(l)"] = dbs
    cl_object["wcss_dist(l)"] = wcss_dist
    cl_object["bcss_dist(h)"] = bcss_dist
    #cl_object["fstat_dist(h)"] = fstat_dist
    #cl_object["wcss_sample(l)"] = wcss_sample
    #cl_object["bcss_sample(h)"] = bcss_sample
    #cl_object["fstat_sample(h)"] = fstat_sample


    cl_object["mean_msd_mag(l)"] = mean_msd_mag
    cl_object["mean_msd_mag_norm(l)"] = mean_msd_mag_norm
    cl_object["max_msd_mag(l)"] = max_msd_mag
    cl_object["mean_mad_mag(l)"] = mean_mad_mag
    cl_object["mean_mad_mag_norm(l)"] = mean_mad_mag_norm
    cl_object["max_mad_mag(l)"] = max_mad_mag


    cl_object["mean_msd_ori(l)"] = mean_msd_ori
    cl_object["mean_msd_ori_norm(l)"] = mean_msd_ori_norm
    cl_object["max_msd_ori(l)"] = max_msd_ori
    cl_object["mean_mad_ori(l)"] = mean_mad_ori
    cl_object["mean_mad_ori_norm(l)"] = mean_mad_ori_norm
    cl_object["max_mad_ori(l)"] = max_mad_ori


    eval_object[method_name] = cl_object

    return eval_object

#savepath needs to direct directly to the filename
def write_statistical_evaluation(savepath, method_names, stat_names, eval_object):
    #nr_stats = len(eval_object[method_names[0]])+1
    #method_names = method_names

    max_stat_len = max([len(x) for x in stat_names])
    full_row_tab_nr = max_stat_len//4 + 1

    with open(savepath, "a") as txt:
        #txt.write(" " + ("\t"*max_stat_len))
        txt.write("\t"*full_row_tab_nr)
        #max_name_len = max([len(x) for x in method_names])
        # Column Tabs
        for name in method_names:
            #tabs_col = "\t" * ((max_name_len // len(name)))
            tabs_col = "\t" * int(np.around(len(name)/4))
            #if len(name) % 4 == 0:
            #    tabs_col += "\t"
            #    tabs_col = tabs_col[:-1]
            #tabs_col += " " * 4
            if ((len(name)/4) - int(len(name)/4)) > 0.5 or len(name)%4 == 0:
                tabs_col += "\t"
            #if len(name) == 8:
            #    tabs = "\t" * ((max_name_len // len(name))-1)
            txt.write(name+tabs_col)
        txt.write("\n")
        
        for stat in stat_names:
            tab_nr_row = (full_row_tab_nr-len(stat)//4)
            #tab_nr = (3-(len(stat)//8))
            #if len(stat) == 8:
            #    tab_nr = (2-(len(stat)//8))
            tabs_row = "\t"*(tab_nr_row)
            txt.write(stat+tabs_row)
            # Field separator tabs
            for name in method_names:
                name = name.replace(" ", "_").lower()
                #tabs_field = "\t" * ((max_name_len // len(name)) + 1)
                tabs_field = "\t" * (int(np.around(len(name)/4)))
                #if len(name) % 8 == 0:
                #    tabs_field += "\t"
                #    tabs_field = tabs_field[:-1]
                #tabs_field += " " * 4
                if ((len(name)/4) - int(len(name)/4)) > 0.5 or len(name)%4 == 0:
                    tabs_field += "\t"
                #print("#####")
                #print(name)
                #print(eval_object[name][stat])
                #print("#####")
                if not np.isfinite(eval_object[name][stat]):
                    eval_object[name][stat] = -999.99
                if len(name) > 4 and len(str(int(eval_object[name][stat]))) < 4:
                    tabs_field += "\t" * (len(name)//4 - len(str(int(eval_object[name][stat])))//4)
                #if len(name) == 8:
                #    tabs_field = "\t" * ((max_name_len // len(name)))
                #tab_nr = (len(name)//8) + 1
                #tabs = "\t"*(tab_nr)
                #if len(str(int(eval_object[name][stat]))) > 1 or len(str(np.around(eval_object[name][stat], 5))) > 3:
                #    if len(str(int(eval_object[name][stat]))) > 1 and len(str(np.around(eval_object[name][stat], 5))) > 3:
                #        txt.write(str(np.around(eval_object[name][stat], 5))+tabs_field[:-2])
                #    else:
                #        txt.write(str(np.around(eval_object[name][stat], 5))+tabs_field[:-1])
                #else:
                overlength = len(str(np.around(eval_object[name][stat], 5))) // 4
                if overlength != 0:
                    txt.write(str(np.around(eval_object[name][stat], 5))+tabs_field[:-overlength])
                else:
                    txt.write(str(np.around(eval_object[name][stat], 5))+tabs_field)
            txt.write("\n")


def cluster_evaluation(c_arrays, memb, pp_imgs, savepath, method_name):
    # mean, median, std in similarity values
    # Singleton cluster will appear as NaN, since their c_array is empty.
    mean_dists = [np.mean(x) for x in c_arrays]
    median_dists = [np.median(x) for x in c_arrays]
    std_dists = [np.std(x) for x in c_arrays]
    cluster_sizes = [len(np.where(memb==i)[0]) for i in range(1, np.amax(memb)+1)]

    with open(join(savepath, method_name) + ".txt", "a") as txt:
        txt.write("Statistical intra cluster analysis. Singletons will appear as NaN.\n")
        for idx, (size, mean, median, std) in enumerate(zip(cluster_sizes, mean_dists, median_dists, std_dists)):
            txt.write("Cluster %d:\nSize: %d, Mean: %f, Median: %f, Std: %f\n" % (idx + 1, size, mean, median, std))
        txt.write("\n\n")
        txt.write("Statistical cluster image stack analysis:\n")
        txt.write("Color scales of mean, median and max aggregation images are bounded in [0,1], while std aggregation images are bounded in [0,0.5], as 0.5 is the maximum standard deviation for values in [0,1].\n")
        txt.write("Hull abstraction is created by Otsu binarizing and opening(closing()) with a disk struct of radius 2.\n")
        txt.write("Region abtraction is created by the maximally stable extremal regions (MSER) algorithm.")
        txt.write("Mean standard deviation (MSD): Standard deviation for each Pixel across the cluster image stack.\n")
        txt.write("Mean absolute deviation (MAD): Mean absolute deviation of every image within the cluster image stack from the clusters mean aggregation image.\n")
        txt.write("MSD and MAD are calculate over every pixel with signal > 0 in at least one image.\n")
        txt.write("\n")

    # mean, median, std difference from representation image
    pp_imgs = np.array(pp_imgs)
    mads = []
    msds = []
    for grp in range(1, np.amax(memb)+1):
        grp_imgs = pp_imgs[np.where(memb == grp)]
        mean_img = np.mean(grp_imgs, axis=0)
        median_img = np.median(grp_imgs, axis=0)
        std_img = np.std(grp_imgs, axis=0)
        max_img = np.max(grp_imgs, axis=0)
        t_otsu = skif.threshold_otsu(max_img)
        hull_img = np.zeros(max_img.shape)
        hull_img[np.where(max_img > t_otsu)] = 1
        sum_img = np.sum(grp_imgs, axis=0)
        sum_img = (sum_img - np.amin(sum_img)) / (np.amax(sum_img) - np.amin(sum_img))
        measured_area_size = int(np.where(sum_img > 0)[0].size * 0.8)
        mser_img, stable_regions = calc_mser(img_as_ubyte(sum_img), 3, 1, 0.01, measured_area_size, True)

        sgl_idx = np.where(np.sum(grp_imgs, axis=0) > 0)
        
        msd = np.mean(std_img[sgl_idx])
        msds.append(msd)
        
        plt.figure()
        plt.title("Mean Aggregation Image (Cluster %d)" % grp)
        #plt.imshow(mean_img, vmin=0, vmax=1)
        plt.imsave(join(savepath, "c" + str(grp), "mean_img_c" + str(grp) + "-" + method_name + ".png"), mean_img, vmin=0, vmax=1)
        

        plt.figure()
        plt.title("Median Aggregation Image (Cluster %d)" % grp)
        #plt.imshow(median_img, vmin=0, vmax=1)
        plt.imsave(join(savepath, "c" + str(grp), "median_img_c" + str(grp) + "-" + method_name + ".png"), median_img, vmin=0, vmax=1)
        
        plt.figure()
        plt.title("Max Aggregation Image (Cluster %d)" % grp)
        #plt.imshow(max_img, vmin=0, vmax=1)
        plt.imsave(join(savepath, "c" + str(grp), "max_img_c" + str(grp) + "-" + method_name + ".png"), max_img, vmin=0, vmax=1)

        plt.figure()
        plt.title("Std Aggregation Image (Cluster %d)" % grp)
        #plt.imshow(std_img, vmin=0, vmax=0.5)
        plt.imsave(join(savepath, "c" + str(grp), "std_img_c" + str(grp) + "-" + method_name + ".png"), std_img, vmin=0, vmax=0.5)

        plt.figure()
        plt.title("Aggregation Hull Abstraction (Cluster %d)" % grp)
        plt.imsave(join(savepath, "c" + str(grp), "hull_img_c" + str(grp) + "-" + method_name + ".png"), opening(closing(hull_img, disk(2)), disk(2)))

        plt.figure()
        plt.title("Summation Image (Cluster %d)" % grp)
        plt.imsave(join(savepath, "c" + str(grp), "sum_img_c" + str(grp) + "-" + method_name + ".png"), sum_img)

        plt.figure()
        plt.title("Maximally Stable Extremal Regions Abtraction Image (Cluster %d)" % grp)
        plt.imsave(join(savepath, "c" + str(grp), "mser_img_c" + str(grp) + "-" + method_name + ".png"), mser_img)

        mad = np.mean([np.abs(mean_img - img)[sgl_idx] for img in grp_imgs])
        mads.append(mad)

        with open(join(savepath, method_name) + ".txt", "a") as txt:
            txt.write("Cluster %d:\n" % grp)
            txt.write("MSD: %f, MAD: %f\n" % (msd, mad))
            txt.write("\n")

        plt.close("all")
            
    
    
    plt.figure()
    plt.title("Mean Standard Deviations per Cluster")
    plt.plot(range(1, len(msds) + 1), msds)
    plt.savefig(join(savepath, "msds.png"))

    plt.figure()
    plt.title("Mean Absolute Deviations from Mean Cluster Image per Cluster")
    plt.plot(range(1, len(mads) + 1), mads)
    plt.savefig(join(savepath, "mads.png"))

    plt.close("all")


# Scatterplots for each pair of measures
def scatterplot(listA, listB, labels, method_nameX, method_nameY, savepath):
    if not labels:
        labels = np.arange(0, squareform(listA).shape[0])

    fig, ax = plt.subplots()
    plt.title("Distance Scatterplot")
    plt.plot(listA, listB, "bo")
    plt.plot([0,1], [0,1], "r")
    plt.xlabel(method_nameX)
    plt.ylabel(method_nameY)

    for lbl in range(len(listA)):
        idx = np.triu_indices_from(np.zeros((len(labels), len(labels))), k=1)
        tuples = np.array(range(len(labels))) * np.ones(len(labels))[:,None]
        tuples = np.dstack((tuples.T, tuples))
        tuples = tuples[idx[0], idx[1], :]
        ##if len(labels) > 0:
        # Use annotate only for interactive exploration. It clutters too much to save.
        #ax.annotate((labels[int(tuples[lbl][0])], labels[int(tuples[lbl][1])]), (scatterlists[0][lbl], scatterlists[1][lbl]))
        ##else:
        ##    ax.annotate(tuples[lbl], (scatterlists[0][lbl], scatterlists[1][lbl]))

    plt.savefig(join(savepath, "scatterplot-" + method_nameX + "_" + method_nameY + ".png"))

    plt.close("all")

################### Cluster Number Evaluation Measures ###################
# To maximize
def silhouette(dmatrix,memb):
	return silhouette_score(dmatrix, memb,  metric="precomputed")

# To minimize
def wcss(dmatrix, memb):
    css = []
    for i in range(1, np.amax(memb)+1):
        idx = np.where(memb == i)[0]
        c_vals = squareform(dmatrix[idx[:,None], idx])
        c_mean = np.mean(c_vals)
        if np.isnan(c_mean):
            css.append(0)
        else:
            css.append(np.sum((c_vals - c_mean)**2))
    return np.sum(css)

# To maximize
def bcss(dmatrix, memb):
    gm = np.mean(squareform(dmatrix))
    c_means = []
    c_lens = []
    for i in range(1, np.amax(memb)+1):
        idx = np.where(memb == i)[0]
        c_vals = squareform(dmatrix[idx[:,None], idx])
        c_means.append(np.mean(c_vals))
        c_lens.append(len(c_vals))
    c_means = np.array(c_means)
    c_lens = np.array(c_lens)    
    c_means[np.isnan(c_means)] = 0
    return np.sum(c_lens * (c_means - gm)**2)

# To maximize
def fstat(dmatrix, memb):
    return bcss(dmatrix, memb)/wcss(dmatrix, memb)

# To minimize
def wcss_samples(samples, memb):
    samples = np.array(samples)
    css = []
    for i in range(1, np.amax(memb)+1):
        idx = np.where(memb == i)[0]
        c_samples = samples[idx]
        c_mean = np.mean(c_samples, axis=0)
        css.append(np.sum((c_samples - c_mean)**2))
    return np.sum(css)/(len(samples)-np.amax(memb))

# To maximize
def bcss_samples(samples, memb):
    sampels = np.array(samples)
    gm = np.mean(samples, axis=0)
    c_means = []
    c_lens = []
    for i in range(1, np.amax(memb)+1):
        idx = np.where(memb == i)[0]
        c_samples = samples[idx]
        c_means.append(np.mean(c_samples, axis=0))
        c_lens.append(len(c_samples))
    c_means = np.array(c_means)
    c_lens = np.array(c_lens)
    return np.sum(c_lens[:,None,None] * (c_means - gm)**2) / (np.amax(memb)-1)

# To maximize
def fstat_samples(samples, memb):
    return bcss_samples(samples, memb)/wcss_samples(samples, memb)


# Try to estimate the best number of clusters by optimizing silhoutte, wcss and bcss
def calc_cluster_number(dmatrix, labels, method_name, savepath):
    cond_dmatrix = squareform(dmatrix)
    Z = linkage(cond_dmatrix, method="average", optimal_ordering=True)
    min_cl = 2
    sil_scores = []
    wcss_scores = []
    bcss_scores = []
    for i in range(min_cl,len(labels)-1):
        memb = fcluster(Z, t=i, criterion="maxclust")
        sil_scores.append(silhouette(dmatrix, memb))
        wcss_scores.append(wcss(dmatrix, memb))
        bcss_scores.append(bcss(dmatrix, memb))

    kneedle = KneeLocator(range(min_cl,len(labels)-1), wcss_scores, curve="convex", direction="decreasing")
    
    sil_nr = np.argmax(sil_scores) + min_cl
    wcss_nr = kneedle.knee
    bcss_nr = np.argmax(bcss_scores) + min_cl
    
    plt.figure()
    plt.title("Silhouetten Plot: " + method_name)
    plt.plot(range(min_cl,len(labels)-1), sil_scores)
    plt.savefig(join(savepath, "silhoutteplot_" + method_name + ".png"))
    plt.figure()
    plt.title("wcss_scores")
    plt.plot(range(min_cl,len(labels)-1), wcss_scores)
    plt.savefig(join(savepath, "wcss_" + method_name + ".png"))
    plt.figure()
    plt.title("bcss_scores")
    plt.plot(range(min_cl,len(labels)-1), bcss_scores)
    plt.savefig(join(savepath, "bcss_" + method_name + ".png"))

    return sil_nr, wcss_nr, bcss_nr