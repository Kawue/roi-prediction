import numpy as np

class AutoVivificationDict(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


########## Utility Functions ##########
# https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
def normalization(x, nmin=0, nmax=1):
    pmin = np.amin(x)
    pmax = np.amax(x)
    x = (nmax-nmin) * ((x-pmin) / (pmax-pmin)) + nmin
    return x

def create_empty_img(height, width, rgba):
    if rgba:
        return np.zeros((height+1, width+1, 4))
    else:
        return np.zeros((height+1, width+1))

def create_img(intensities, y_pixels, x_pixels, height, width):
    img = create_empty_img(height, width, False)
    img[(y_pixels, x_pixels)] = intensities
    return img