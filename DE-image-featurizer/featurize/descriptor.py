"""https://www.sciencedirect.com/science/article/pii/S0030402620306690"""
import numpy as np
from skimage.feature import hog, local_binary_pattern

from patchify import patchify


class HOG:
    def __init__(self, n_bin=1280, orientations=8, pixels_per_cell=(16, 16),
                 cells_per_block=(1, 1), grid_size=(1, 1), normalize=True):
        """
        References
        ----------
        https://github.com/canxkoz/HOG-Classifier/blob/caf20c5fe427983a5a373cce0bb299a98d75e8f4/HOG.py


        Parameters
        ----------
        n_bin : int
            Specify the number of bins that will be used in the histogram
        orientations : int
            Specify the number of orientation bins that the gradient information will be split up into in the histogram
        pixels_per_cell : tuple
            Specify the number of pixels in the cel
        cells_per_block : tuple
            Determine the size of the block
        grid_size : tuple
            Determine the grid size of the patches
        normalize : bool
            Normalize the histogram
        """
        self.n_bin = n_bin
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.grid_size = grid_size
        self.normalize = normalize

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
                 block_norm='L2-Hys',
                 channel_axis=-1)
        hist, _ = np.histogram(fd, bins=self.n_bin)

        if self.normalize:
            hist = np.array(hist) / np.sum(hist)

        return hist

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
    def __init__(self, n_points=1280, radius=160, method='uniform', grid_size=(1, 1), normalize=True):
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
        normalize : bool
            Normalize the features
        """
        self.grid_size = grid_size
        self.n_points = n_points
        self.radius = radius
        self.method = method
        self.normalize = normalize

    def lbp_featurizer(self, image):
        """
        The lbp_featurizer function takes an image as input and returns a histogram of the local binary pattern
        of the image. The number of points is set to 1280, radius is set to 160, method is set to 'uniform', and normalize
        is set to True.

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
            lbp = local_binary_pattern(image[:, :, ch], P=self.n_points, R=self.radius, method=self.method)
            n_bins = int(lbp.max() + 1)
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

            if self.normalize:
                hist = np.array(hist) / np.sum(hist)

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
        np.array
        """
        for image in images:
            image_patches = patchify(image, self.grid_size)
            patch_feature_arr = []
            for patch_row in image_patches:
                for patch_col in patch_row:
                    patch_feature_arr.append(self.lbp_featurizer(patch_col))

            patch_feature_arr = np.stack(patch_feature_arr)
            patch_feature_arr = patch_feature_arr.reshape(*self.grid_size, *patch_feature_arr.shape[1:])
            return patch_feature_arr
