import warnings
from enum import Enum
from functools import partial

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageEnhance, Image
from skimage import filters
from skimage.filters import threshold_otsu, threshold_local


def global_thresholding(image, binary=False, invert=False):
    """
    References
    ----------
    https://en.wikipedia.org/wiki/Otsu's_method
    https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.threshold_otsu

    Parameters
    ----------
    image : np.array
        Array with ndim == 2
    binary : bool
        Returns a boolean image after thresholding
    invert : bool
        Inverts the boolean thresholding filter

    Returns
    -------
    np.array
        Processed image with dtype unit8 or bool
    """
    global_thresh = threshold_otsu(image)
    if invert:
        threshed = image < global_thresh  # binary thresholding
    else:
        threshed = image > global_thresh  # binary thresholding
    if not binary:
        threshed = image * threshed
    return threshed


def local_thresholding(image, block_size=7, offset=0, binary=False, invert=False):
    """
    References
    ----------
    https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.threshold_local

    Parameters
    ----------
    image : np.array (2D)
        Array with ndim == 2
    block_size : int
        Odd size of pixel neighborhood which is used to calculate the threshold value (e.g. 3, 5, 7, …)
    offset : float
        Constant subtracted from weighted mean of neighborhood to calculate the local threshold value.
    binary : bool
        Returns a boolean image after thresholding
    invert : bool
        Inverts the boolean thresholding filter

    Returns
    -------
    np.array
        Processed image with dtype unit8 or bool
    """
    local_thresh = threshold_local(image, block_size=block_size, offset=offset)
    if invert:
        threshed = image < local_thresh  # binary thresholding
    else:
        threshed = image > local_thresh  # binary thresholding
    if not binary:
        threshed = image * threshed
    return threshed


def hysteresis_thresholding(image, low=.35, high=.50):
    """
    References
    ----------
    https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.apply_hysteresis_threshold

    Parameters
    ----------
    image : np.array
        Array with ndim == 2
    low : float
        Percentile filter for edges; clips filtered edges with thresholds
    high : float
        Percentile filter for edges; clips filtered edges with thresholds

    Returns
    -------
    np.array
        Processed image with dtype unit8
    """
    edges = filters.sobel(image)
    low = np.percentile(edges, low * 100)
    high = np.percentile(edges, high * 100)
    threshed = filters.apply_hysteresis_threshold(edges, low, high)
    return threshed.astype('uint8') * 255


def gamma_correction(image, gamma=5):
    """
    References
    ----------
    https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_log_gamma.html

    Parameters
    ----------
    image : np.array
        Array with ndim == 2
    gamma : float
        A gamma of (i) 0 is darker image; (ii) 1.0 gives the original image; (iii) > 1.0 brighter images

    Returns
    -------
    np.array
        Processed image with dtype unit8
    """
    image = Image.fromarray(image)
    enhancer = ImageEnhance.Contrast(image)
    return np.array(enhancer.enhance(gamma))


def log_correction(image):
    """
    References
    ----------
    https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_log_gamma.html

    Parameters
    ----------
    image : np.array
        Array with ndim == 2

    Returns
    -------
    np.array
        Processed image with dtype unit8
    """
    # Apply log transform.
    c = 255 / (np.log(1 + 1e-5 + np.max(image)))
    log_transformed = c * np.log(1 + image)

    log_transformed = np.array(log_transformed, dtype=np.uint8)  # can be removed
    return log_transformed


def histogram_equalization(image):
    """
    References
    ----------
    https://en.wikipedia.org/wiki/Histogram_equalization
    https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html

    Parameters
    ----------
    image : np.array
        Array with ndim == 2

    Returns
    -------
    np.array
        Processed image with dtype unit8
    """
    return cv2.equalizeHist(image)


def clahe_equalization(image, max_clip=40.0):
    """
    References
    ----------
    https://en.wikipedia.org/wiki/Histogram_equalization
    https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html

    Parameters
    ----------
    image : np.array
        Array with ndim == 2
    max_clip : float
        Threshold for contrast limiting.

    Returns
    -------
    np.array
        Processed image with dtype unit8
    """
    clahe = cv2.createCLAHE(clipLimit=max_clip)
    return clahe.apply(image)


def sobel_edge_highlighting(image):
    """
    References
    ----------
    https://en.wikipedia.org/wiki/Sobel_operator

    Parameters
    ----------
    image : np.array
        Array with ndim == 2

    Returns
    -------
    np.array
        Processed image with dtype unit8
    """
    edges = filters.sobel(image)
    edges = edges * 255

    image += edges.astype(image.dtype)
    image = np.clip(image, a_min=0, a_max=255)
    return image


def color_mapping(image, cmap='viridis'):
    """
    Parameters
    ----------
    image : np.array
        Array with ndim == 2
    cmap : string
        Colormaps available in matplotlib. Check here: https://matplotlib.org/stable/tutorials/colors/colormaps.html

    Returns
    -------
    np.array
        Processed 3D (RGB) image with dtype unit8
    """
    image = image
    cm = plt.get_cmap(cmap)
    colored_image = cm(image) * 255.0
    colored_image = colored_image.astype('uint8')

    # Obtain a 4-channel image (R,G,B,A) in float [0, 1] and clip it to RGB
    return colored_image[:, :, :3]


class PreProcessors(Enum):
    GlobalThresholding = partial(global_thresholding)
    LocalThresholding = partial(local_thresholding)
    HysteresisThresholding = partial(hysteresis_thresholding)
    GammaCorrection = partial(gamma_correction)
    LogCorrection = partial(log_correction)
    HistogramEqualization = partial(histogram_equalization)
    CLAHE = partial(clahe_equalization)
    ColorMapping = partial(color_mapping)
    SobelEdgeHighlighting = partial(sobel_edge_highlighting)

    def __init__(self, processor):
        """
        Parameters
        ----------
        processor : partial
        """
        self.processor = processor

    @staticmethod
    def validate_image_format(image):
        """
        Parameters
        ----------
        image : np.array

        Returns
        -------
        np.array
        """
        if image.ndim > 3:
            raise Exception("Found ndim > 3; This method expects ndim==2 or ndim==3 (grayscale images).")

        if image.ndim == 3:
            warnings.warn("Found ndim == 3; Clipping image to 2 dimensions for grayscale images")
            image = image[:, :, 0]

        if image.dtype != np.uint8:
            warnings.warn("This method expects uint8 datatype")

        return np.squeeze(image)

    def fit_transform(self, image, **args):
        """
        Parameters
        ----------
        image : np.array
            Image of 2-dim or 3-dim array

        Returns
        -------
        np.array
            Processed image with dtype unit8
        """
        image = self.validate_image_format(image)
        return self.processor(image, **args)
