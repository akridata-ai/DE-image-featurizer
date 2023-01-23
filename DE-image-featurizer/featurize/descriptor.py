"""https://www.sciencedirect.com/science/article/pii/S0030402620306690"""
import numpy as np
from skimage.feature import hog, local_binary_pattern

from patchify import patchify


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
        images : np.array or list
            Images for featurization

        Returns
        -------
        np.array
            A numpy array with the same shape as images, but instead of having a single channel
            for each image, it has multiple channels
        """
        for image in images:
            image_patches = patchify(image, self.grid_size)
            patch_feature_arr = []
            for patch_row in image_patches:
                for patch_col in patch_row:
                    patch_feature_arr.append(self.hog_featurizer(patch_col))

            patch_feature_arr = np.stack(patch_feature_arr)
            patch_feature_arr = patch_feature_arr.reshape(*self.grid_size, *patch_feature_arr.shape[1:])
            return patch_feature_arr


class LBP:
    def __init__(self, n_bins=257, n_points=8, radius=1, method='default', grid_size=(1, 1)):
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
        of the image. The number of points is set to 1280, radius is set to 160, method is set to 'uniform', and normalize
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
            A list of arrays of shape (grid_size[0], grid_size[0], 3, patch_shape[0]*patch_shape[0])
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
