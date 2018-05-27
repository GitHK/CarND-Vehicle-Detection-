import cv2
import matplotlib.image as mpimg
import numpy as np
from skimage.feature import hog

COLORS_SPACE_MAP = {
    'RGB': lambda x: np.copy(x),
    'HSV': lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2HSV),
    'LUV': lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2LUV),
    'HLS': lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2HLS),
    'YUV': lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2YUV),
    'YCrCb': lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2YCrCb),
}


def convert_color(img, color_space='YCrCb'):
    """ Converts image in selected color space"""
    return COLORS_SPACE_MAP[color_space](img)


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Hog Feature and Visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    return hog(img, orientations=orient,
               pixels_per_cell=(pix_per_cell, pix_per_cell),
               cells_per_block=(cell_per_block, cell_per_block),
               visualise=vis, feature_vector=feature_vec,
               # block_norm="L2-Hys" # car detection performs better with default block norm
               )


def extract_features(images, color_space='YCrCb', spatial_size=(32, 32), hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel='ALL', spatial_feat=True, hist_feat=True,
                     hog_feat=True):
    features = []
    for file in images:
        file_features = []

        image = mpimg.imread(file)

        feature_image = convert_color(image, color_space)

        if spatial_feat is True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)

        if hist_feat is True:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)

        if hog_feat is True:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))

    return features
