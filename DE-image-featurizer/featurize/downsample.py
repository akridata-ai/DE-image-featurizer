import cv2
from patchify import patchify
import numpy as np


class DownSample:
    def __init__(self, size=(16, 16), grid_size=(1, 1), interpolation=cv2.INTER_AREA):
        """
        Parameters
        ----------
        size : tuple
            Set down-sampling size
        grid_size : tuple
            Set the patch grid size
        interpolation : cv2 enum
            Specify the interpolation method. Defaults as cv2.INTER_AREA.
        """
        self.size = size
        self.grid_size = grid_size
        self.interpolation = interpolation

    def resize(self, image):
        """
        The resize function takes an image and resizes it to the size specified in the constructor.
        The interpolation parameter specifies what type of interpolation is used when resizing,
        and can be any of the following: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA,
        cv2.INTER_CUBIC or cv2.INTER_LANCZOS4.

        Parameters
        ----------
        image : np.array
            Pass the image to be resized

        Returns
        -------
        np.array
            The resized image as a flattened array.
        """
        return cv2.resize(image, self.size, interpolation=self.interpolation).flatten()

    def fit_transform(self, images):
        """
        The fit_transform function takes in a list of images and returns the feature vectors for each image.
        The feature vector is simply the flattened array of all patches in the downsampled image.

        Parameters
        ----------
        images : np.array
            Pass the images to be transformed

        Returns
        -------
        np.array
            A numpy array of shape (grid_size[0], grid_size[0], 3, patch_shape[0]*patch_shape[0])
        """
        for image in images:
            image_patches = patchify(image, self.grid_size)
            patch_feature_arr = []
            for patch_row in image_patches:
                for patch_col in patch_row:
                    patch_feature_arr.append(self.resize(patch_col))

            patch_feature_arr = np.stack(patch_feature_arr)
            patch_feature_arr = patch_feature_arr.reshape(*self.grid_size, *patch_feature_arr.shape[1:])
            return patch_feature_arr
