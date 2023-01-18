"""Color-space based featurizers"""
import cv2
import numpy as np
from patchify import patchify


class ColorFeaturizer:
    def __init__(self, color_mode='HSV', grid_size=(1, 1), n_bins=255,
                 hist_range=[(0, 180), (0, 255), (0, 255)], multivariate=False):
        """
        Parameters
        ----------
        color_mode : str
            Specify the color space that we want to use for histogram calculation
        grid_size : tuple
            Grid size for image patches
        n_bins : int
            Specify the number of bins to use for each channel
        hist_range : list
            Specify the number of bins to use for each color channel. Defaults to HSV ranges.
        multivariate : bool
            Specify whether the histogram is univariate or multivariate
        """
        self.color_mode = color_mode.upper()
        self.cv2_color_mode_name = f'COLOR_RGB2{self.color_mode}'
        self.grid_size = grid_size
        self.n_bins = n_bins
        self.hist_range = hist_range
        self.featurizer = self.univariate_histogram_featurizer

        if multivariate:
            self.featurizer = self.multivariate_histogram_featurizer

    def convert_image(self, image):
        """
        The convert_image function converts an image from one color space to another. If the
        desired color space is RGB, then no conversion is performed, and we return the original
        image. Otherwise, we use OpenCV to convert our input RGB images into other color spaces.

        Parameters
        ----------
        image : np.array
            RGB image to be converted having dtype uint8

        Returns
        --------
        np.array
            The converted image.
        """
        if self.color_mode == 'RGB':  # RGB2RGB case
            return image
        else:
            cvt_image = cv2.cvtColor(image, getattr(cv2, self.cv2_color_mode_name))
            return cvt_image

    def univariate_histogram_featurizer(self, image):
        """
        The univariate_histogram_featurizer function takes in an image and returns a histogram
        of the intensities for each channel. The histogram is computed by taking the number of
        pixels that fall into each bin, where bins are defined by n_bins and range is defined
        by hist_range. The function then returns this as a feature vector.

        Parameters
        ----------
        image : np.array
            Calculate the histogram of each channel in the image.

        Returns
        -------
        np.array
            A feature vector of the histogram of pixel values for each channel in the image
        """
        channels = image.shape[-1]
        pixel_arr = image.reshape((-1, channels))
        histogram = []
        for ch, rng in zip(range(channels), self.hist_range):
            hist_channel, _ = np.histogram(pixel_arr[:, ch], bins=self.n_bins, range=rng)
            histogram.append(hist_channel)
        return np.dstack(histogram).squeeze()

    def multivariate_histogram_featurizer(self, image):
        """
        The multivariate_histogram_featurizer function takes in an image and returns a
        histogram of the pixel intensities of each channel. The histogram is computed by taking
        the number of combination pixels in N dimensions that fall into each bin. The function
        then returns this as a feature vector.

        Parameters
        ----------
        image : np.array
            Calculate the histogram of each channel in the image.

        Returns
        -------
        np.array
            A feature vector of the histogram of pixel values for each channel in the image.
        """
        channels = image.shape[-1]
        pixel_arr = image.reshape((-1, channels))
        histogram, _ = np.histogramdd(pixel_arr, bins=self.n_bins, range=self.hist_range)
        return histogram

    def fit_transform(self, images):
        """
        The fit_transform function takes in a list of images and returns the feature array.
        The feature array is constructed by taking each image, converting it to desired color space,
        and then breaking it into patches. Each patch is then featurized (the output of our Featurizer class)
        and those features are assembled into a 3D tensor that represents the original image.

        Parameters
        ----------
        images : np.array
            Pass in the images that are to be featurized

        Returns
        -------
        np.array
            Patch-wise feature array.
        """
        for image in images:
            cvt_image = self.convert_image(image)
            image_patches = patchify(cvt_image, self.grid_size)
            patch_feature_arr = []
            for patch_row in image_patches:
                for patch_col in patch_row:
                    patch_feature_arr.append(self.featurizer(patch_col))

            patch_feature_arr = np.stack(patch_feature_arr)
            patch_feature_arr = patch_feature_arr.reshape(*self.grid_size, *patch_feature_arr.shape[1:])
            return patch_feature_arr
