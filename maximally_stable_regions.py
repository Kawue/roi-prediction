import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import regionprops, label

# Currently Restricted to uInt8 Values!

# If the Image has a large background region, e.g. ims barley seed, the max_area parameter must be respectively small.
# E.g. if the background is 50% of the image and max_area is > 0.5 the whole foreground will be included, since it is smaller than the max_area parameter.

# Own implementation to find maximally stable regions
# Original Paper: "Robust wide-baseline stereo from maximally stable extremal regions"
def calc_nested_regions(img, delta, min_area, max_area):
    img = img.copy()

    delta = delta

    if 0 < min_area < 1:
        min_area = int(img.size * min_area)
    else:
        min_area = min_area

    if 0 < max_area <= 1:
        max_area = int(img.size * max_area)
    else:
        max_area = max_area

    img_dict = {}
    nested_regions = { -np.inf: [np.zeros(img.shape)] }
    counter = 0
    
    img_crop = np.full(img.shape, 255)

    # Cut away low intensities first
    for i in range(0,256,delta):
        img_crop[np.where(img == i)] = 0
        image_copy = np.zeros(img.shape) + img_crop
        labels, num = label(image_copy, connectivity=1, return_num=True)
        img_dict[i] = {
            "image": image_copy,
            "regions": labels,
            "number": num
        }
    
    for i in range(0,256,delta):
        update_dict = {}
        for region in regionprops(img_dict[i]["regions"]):
            if min_area < region.area < max_area:
                t = np.zeros(img.shape)
                t[region.coords[:,0], region.coords[:,1]] = 1
                for key in nested_regions:
                    if ((nested_regions[key][-1] - t) > -1).all():
                        # Needed because the update dict begins empty at each loop.
                        try:
                            update_dict[key].append(t)
                        except:
                            update_dict[key] = [t]
                        break
                else:
                    update_dict[counter] = [t]
                    counter += 1

        # Check if update is needed.
        # There is no update needed when the region area is outside of min/max area, which results in an empty update dict.
        if update_dict:
            key_adjust = max(max(update_dict.keys()),max(nested_regions.keys())) + 1
            # Update
            for key in update_dict:
                if key in nested_regions:
                    if len(update_dict[key]) > 1:
                        for reg in update_dict[key]:
                            update_region = nested_regions[key][:]
                            update_region.append(reg)
                            nested_regions[key_adjust] = update_region
                            key_adjust += 1
                        del nested_regions[key]
                    else:
                        nested_regions[key].extend(update_dict[key])
                else:
                    nested_regions[key] = update_dict[key][:]
            counter = key_adjust + 1
    else:
        del nested_regions[-np.inf]
    return nested_regions


def calc_stability(bigger_region, current_region, smaller_region):
    return (bigger_region.sum() - smaller_region.sum()) / current_region.sum()


def calc_mser(img, sequence_min, delta, min_area, max_area, all_maps=False):
    img = img.copy()

    #print("ATTENTION, depending on the background size, the max_area parameter should be chosen respectively small!")
    if img.dtype != np.uint8:
        raise ValueError("This method currently needs uint8 format!")

    # The sequence length needs to be at least 3 to evaluate the stability.
    sequence_min = sequence_min
    stability_dict = {}
    nested_regions = calc_nested_regions(img, delta, min_area, max_area)

    for key, regions in nested_regions.items():
        if len(regions) >= sequence_min:
            stabilities = []
            # Regions becoming smaller as the list goes on.
            # The reverse case is true in the original MSER paper.
            for idx, _ in enumerate(regions[1:-1], start=1):
                stability = calc_stability(regions[idx-1], regions[idx], regions[idx+1])
                stabilities.append(stability)
            stability_dict[key] = stabilities
    
    
    #TODO: What happens in minimum valley and does it matter which image to take? Take the first minimum, as it has the largest covered area?
    stable_regions = []
    for key, value in stability_dict.items():
        idx = np.argmin(value)
        stable_regions.append(nested_regions[key][idx])

    stable_regions = np.array(stable_regions)
    #TODO: How to solve dublicated stable regions, i.e. when two or more series have their stable image before they divided
    if len(stable_regions) > 0:        
        # Sum all stable regions to combine information and visualize stability intensity.
        mser_img = np.sum(stable_regions, axis=0)
        if all_maps:
            return mser_img, stable_regions
        else:
            return mser_img
    else:
        print("ATTENTION! There are no stable regions in this image. There ist most likely not enough color or intensity information!")
        print("")
        return np.zeros(img.shape), []



'''
def maximally_stable_extremal_regions(img, sample_mask):
    # 1. Calculate Binary Images for all thresholds t in 0, ..., 255 with step s
    # 2. Find regions and save them in sets
    # 3. Regions are functions of intensity
    # 4. MSERs are regions of local minima in the rate of change
    delta = 1
    min_factor = 0.05
    max_factor = 0.95
    min_area = int(np.ceil(img.size * min_factor))
    max_area = int(np.ceil(img.size * max_factor))
    variation = 0.01

    mser = cv2.MSER_create(_delta=delta, _min_area=min_area, _max_area=max_area, _max_variation=variation)
    regions, boundingboxes = mser.detectRegions(img)

    # MSER also detects stable background areas.
    # To remove those the background is removed via masking.
    # To remove areas that are too small after background removal, deselect those via thresholding.
    mask_size = np.where(sample_mask > 0)[0].size
    mask_threshold = 0.05
    abs_threshold = int(np.ceil(mask_size * mask_threshold))

    breakpoint = None

    for idx, array in enumerate(regions):
        mser_img = np.zeros(img.shape)
        mser_img[array[:,1], array[:,0]] = 1

        mser_img = mser_img * sample_mask

        mser_size = np.where(mser_img > 0)[0].size

        # TODO: Check, not sure if this is always the case.
        # Assumption: When the mser area equals the whole image, this equals pixel value 0
        # Therefore everything with index before this point is stable on the inverted image (low values) and above is stable on the image (high values) (NOT SURE IF THIS HOLDS!)
        if mser_size == mask_size:
            breakpoint = idx

        print(mser_size)

        #Das passt nicht immer.
        #Den Teil des Codes mit dem breakpoint auslagern und als funktion minimieren, scheint sinnvoller

        if mser_size > 3400:
            plt.figure()
            plt.imshow(sample_mask)
            plt.figure()
            plt.imshow(mser_img)
        
        if  mser_size > abs_threshold and breakpoint != None:
            plt.figure()
            plt.imshow(mser_img)
    
    plt.show()
'''