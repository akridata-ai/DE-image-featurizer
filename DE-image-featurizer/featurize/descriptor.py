"""https://www.sciencedirect.com/science/article/pii/S0030402620306690"""
import warnings
import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern

# To support numpy reshape ops on tensors in `patchify` module:
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from .patchify import patchify


def shi_tomasi_keypoints(gray_image, max_keypoints=100, threshold=0.001, min_distance=7):
    """
    The shi_tomasi_keypoints function takes in a grayscale image and returns an array of keypoints.
    The function uses the OpenCV implementation of Shi-Tomasi corner detection to find the strongest corners
    in the image. The function also sets some parameters for how many keypoints are found, as well as
    how strong they are at detecting a corner.
    If no shi-tomasi coners are found we return random keypoints.
    Parameters
    ----------
    gray_image : np.array
        2D grayscale image to be used for corner detection
    max_keypoints : int
        Set the maximum number of keypoints to be found in the image
    threshold : float
        Determine the minimum quality of corner points
    min_distance : int
        Specify the minimum distance between any two corners detected
    Returns
    -------
    list
        List of keypoints in np.array
    """
    if threshold <= 0:
        warnings.warn("Cannot set threshold value <= 0 in Shi Tomasi Keypoints function. "
                      "Setting value to 0.001")
        threshold = 0.001

    corners = cv2.goodFeaturesToTrack(gray_image,
                                      maxCorners=max_keypoints,
                                      qualityLevel=threshold,
                                      minDistance=min_distance)
    if corners is None:
        warnings.warn("No keypoints found by SIFT or Shi-Tomasi, returning random keypoints.")
        height_pt = np.random.randint(0, gray_image.shape[0], (max_keypoints,))
        width_pt = np.random.randint(0, gray_image.shape[1], (max_keypoints,))
        return [cv2.KeyPoint(float(x[1]), float(x[0]), 13) for x in zip(height_pt, width_pt)]

    corners = np.int0(corners)
    return [cv2.KeyPoint(float(x[1]), float(x[0]), 13) for x in corners.reshape(-1, 2)]


class HOG:
    def __init__(self, orientations=8, pixels_per_cell=(16, 16),
                 cells_per_block=(1, 1), grid_size=(1, 1)):
        """
        References
        ----------
        https://github.com/canxkoz/HOG-Classifier/blob/caf20c5fe427983a5a373cce0bb299a98d75e8f4/HOG.py
        Parameters
        ----------
        orientations : int
            Specify the number of orientation bins that the gradient information will be split up into in the histogram
        pixels_per_cell : tuple
            Specify the number of pixels in the cel
        cells_per_block : tuple
            Determine the size of the block
        grid_size : tuple
            Determine the grid size of the patches
        """
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.grid_size = grid_size

    def hog_featurizer(self, image):
        """
        The hog_featurizer function takes in an image and returns a histogram of oriented
        gradients (HOG) feature vector for that image.
        Parameters
        ----------
        image: np.array
            RGB image to be converted having dtype uint8
        Returns
        -------
        np.array
            The histogram of oriented gradients
        """
        fd = hog(image,
                 orientations=self.orientations,
                 pixels_per_cell=self.pixels_per_cell,
                 cells_per_block=self.cells_per_block,
                 feature_vector=True,
                 visualize=False,
                 multichannel=True)
        return fd

    def fit_transform(self, images):
        """
        The fit_transform function takes in a list of images and returns the feature vectors for each image.
        Parameters
        ----------
        images : list or array of ndarrays of length n_images
            Images to calculate features over.
            If passing a single image, pass as array of shape (1, ...) or as list [img_arr].
            
        Returns
        -------
        patch_feature_arr: ndarray of shape (n_images, grid_rows, grid_cols, n_features)
            HOG features for each patch in each input image.
            Traditionally, HOG descriptors are ndarrays. We call `hog` with `feature_vector=True`,
                which flattens the descriptor into a 1d array. This is done for each patch, as identified by
                self.grid_size.
        """
        features_all_images = []
        for image in images:
            image_patches = patchify(image, self.grid_size)
            patch_feature_arr = []
            for patch_row in image_patches:
                for patch_col in patch_row:
                    patch_feature_arr.append(self.hog_featurizer(patch_col))

            patch_feature_arr = np.stack(patch_feature_arr)
            patch_feature_arr = patch_feature_arr.reshape(1, *self.grid_size, *patch_feature_arr.shape[1:])
            features_all_images.append(patch_feature_arr)
            
        return np.concatenate(features_all_images, axis=0)


class LBP:
    def __init__(self, n_bins=256, n_points=8, radius=1, method='default', grid_size=(1, 1)):
        """
        References
        ----------
        https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_local_binary_pattern.html
        Parameters
        ----------
        n_points : int
            Number of circularly symmetric neighbour set points (quantization of the angular space).
        radius : int
            Radius of circle (spatial resolution of the operator).
        method : str
            Method to determine the pattern.
        grid_size : tuple
            Determine the grid size of the patches
        """
        self.n_bins = n_bins
        self.grid_size = grid_size
        self.n_points = n_points
        self.radius = radius
        self.method = method

    def lbp_featurizer(self, image):
        """
        The lbp_featurizer function takes an image as input and returns a histogram of the local binary pattern
        of the image. The number of points is set to 8, radius is set to 1, method is set to 'uniform', and normalize
        is set to True.
        References
        ----------
        https://stackoverflow.com/questions/51239715/compare-the-lbp-in-python
        Parameters
        ----------
        image : np.array
            Extract the channels from the image
        Returns
        -------
        list
            A list of histograms for each channel in the image
        """
        channels = image.shape[-1]

        histogram = []
        for ch in range(channels):
            lbp = local_binary_pattern(image[:, :, ch], self.n_points, self.radius, method=self.method)
            hist, _ = np.histogram(lbp, bins=np.arange(self.n_bins), density=True)
            histogram.append(hist)

        return histogram

    def fit_transform(self, images):
        """
        The fit_transform function takes in a list of images and returns the LBP features for each image.
        The output is a numpy array with shape (num_images, grid_size[0], grid_size[1], num_features).
        Parameters
        ----------
        images: np.array or list
            Pass the list of images to be processed by the fit_transform function
        Returns
        -------
        list
            A list of arrays of shape (grid_size[0], grid_size[1], 3, patch_shape[0]*patch_shape[1])
        """
        features = []
        for image in images:
            image_patches = patchify(image, self.grid_size)
            patch_feature_arr = []
            for patch_row in image_patches:
                for patch_col in patch_row:
                    patch_feature_arr.append(self.lbp_featurizer(patch_col))

            patch_feature_arr = np.stack(patch_feature_arr)
            patch_feature_arr = patch_feature_arr.reshape(*self.grid_size, *patch_feature_arr.shape[1:])
            features.append(patch_feature_arr)
        return features


class ORB:
    def __init__(self, num_features=32, grid_size=(1, 1), fastThreshold=31, edgeThreshold=31):
        """
        Parameters
        ----------
        num_features : int
            Set the number of ORB features to be extracted for each image
        grid_size : tuple
            Determine the grid size of the patches
        fastThreshold : int
            Determine the FAST threshold
        edgeThreshold : int
            Determine the harris corner threshold

        """
        self.grid_size = grid_size
        self.num_features = num_features
        self.orb = cv2.ORB_create(nfeatures=self.num_features,
                                  fastThreshold=fastThreshold,
                                  edgeThreshold=edgeThreshold)

    def orb_featurizer(self, image):
        """
        The orb_featurizer function takes an image as input and returns a list of ORB descriptors.
        The number of descriptors is limited by the num_features parameter (defaults to 32).
        Parameters
        ----------
        image : np.array
            Pass in the grayscale image to be featurized of shape (height, width)
        Returns
        -------
        np.array
            Feature matrix of the image
        """
        _, descriptors = self.orb.detectAndCompute(image, None)
        return descriptors

    def fit_transform(self, images):
        """
        The fit_transform function takes in a list of RGB images and returns the features for each image.
        The features are constructed by taking the ORB descriptors from each patch in an image,
        and then concatenating those together into one feature vector. The result is a matrix where
        each row corresponds to an image and contains all of its feature vectors.
        Parameters
        ----------
        images : Pass the images to the fit_transform function
        Returns
        -------
        list
            A list of arrays of shape (grid_size[0], grid_size[1], feature_matrix)
        """
        features = []
        for image in images:
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image_patches = patchify(grayscale_image, self.grid_size)
            patch_feature_arr = []
            for patch_row in image_patches:
                for patch_col in patch_row:
                    patch_feature_arr.append(self.orb_featurizer(patch_col))

            patch_feature_arr = np.stack(patch_feature_arr)
            patch_feature_arr = patch_feature_arr.reshape(*self.grid_size, *patch_feature_arr.shape[1:])
            features.append(patch_feature_arr)
        return features


class SIFT:
    def __init__(self, num_features=32, grid_size=(1, 1)):
        """
        Parameters
        ----------
        num_features : int
            Set the number of SIFT features to be extracted for each image
        grid_size : tuple
            Determine the grid size of the patches
        """
        self.grid_size = grid_size
        self.num_features = num_features
        self.sift = cv2.SIFT_create(nfeatures=self.num_features)

    def sift_featurizer(self, image):
        """
        The sift_featurizer function takes an image as input and returns a list of SIFT descriptors.
        The number of descriptors is limited by the num_features parameter (defaults to 32).
        Parameters
        ----------
        image : np.array
            Pass in the grayscale image to be featurized of shape (height, width)
        Returns
        -------
        np.array
            Feature matrix of the image
        """
        _, descriptors = self.sift.detectAndCompute(image, None)
        if descriptors is None:
            warnings.warn('No SIFT features found returning Shi-Tomasi features.')
            key_points = shi_tomasi_keypoints(image, max_keypoints=self.num_features)
            _, descriptors = self.sift.compute(image, key_points)

        elif len(descriptors) < self.num_features:
            warnings.warn("SIFT wasn't able to generate specified number of features. "
                          "Padding feature array using Shi-Tomasi for desired number of features")
            padding = self.num_features - len(descriptors)
            key_points = shi_tomasi_keypoints(image, max_keypoints=padding)
            _, descriptors_pad = self.sift.compute(image, key_points)
            descriptors = np.concatenate((descriptors, descriptors_pad), axis=0)

        else:
            descriptors = descriptors[:self.num_features]
        return descriptors

    def fit_transform(self, images):
        """
        The fit_transform function takes in a list of RGB images and returns the features for each image.
        The features are constructed by taking the SIFT descriptors from each patch in an image,
        and then concatenating those together into one feature vector. The result is a matrix where
        each row corresponds to an image and contains all of its feature vectors.
        Parameters
        ----------
        images : Pass the images to the fit_transform function
        Returns
        -------
        list
            A list of arrays of shape (grid_size[0], grid_size[1], feature_matrix)
        """
        features = []
        for image in images:
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image_patches = patchify(grayscale_image, self.grid_size)
            patch_feature_arr = []
            for patch_row in image_patches:
                for patch_col in patch_row:
                    patch_feature_arr.append(self.sift_featurizer(patch_col))

            patch_feature_arr = np.stack(patch_feature_arr)
            patch_feature_arr = patch_feature_arr.reshape(*self.grid_size, *patch_feature_arr.shape[1:])
            features.append(patch_feature_arr)
        return features
