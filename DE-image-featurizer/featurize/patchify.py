"""This py file contains the patchify function that is used to get sub-tiles of the images"""
import numpy as np


def patchify(image, grid_shape):
    """
    The patchify function takes an image and a grid shape (in the form of a tuple) as input.
    It then divides the image into tiles of equal size, given by grid_shape.
    The function returns a numpy array containing all these tiles in a grid.

    Parameters
    ----------
    image: ndarray of shape (imrows, imcols, n_channels)
        Image to be converted into tiles.
            
    grid_shape: 2-tuple of int
        Grid to use for tiling, as (grows, gcols)

    Returns
    -------
    tiled_image: ndarray of shape (grows, gcols, imrows//grows, imcols//gcols, ...)
        Tiled image.
        
    Examples
    --------
    Let's tile a simple numpy array.
    >>> img = np.arange(24, dtype=int).reshape(4, 6)
    >>> tiled_img = patchify(img, (2, 2))
    >>> tiled_img[0, 0]
    array([[0, 1, 2],
           [6, 7, 8]])
    >>> tiled_img[0, 1]
    array([[ 3,  4,  5],
           [ 9, 10, 11]])
    """
    image_shape = image.shape

    if (image_shape[0] % grid_shape[0] != 0) or (image_shape[1] % grid_shape[1] != 0):
        raise ValueError(
            "Image rows and columns should be integer multiples of grid rows and cols.")

    # For grayscale images, add an extra axis at the end to have consistent logic later.
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        added_axis = True
    else:
        added_axis = False

    imrows, imcols = image_shape[0], image_shape[1]
    grows, gcols = grid_shape[0], grid_shape[1]
    
    # We'll use numpy reshapes to do this.
    # 1. Tile along columns.
    tiled_image = image.reshape((1, imrows, gcols, imcols//gcols, -1))
    # 2. Columns are already tiled. Move this away so that rows can be tiled next.
    tiled_image = np.swapaxes(tiled_image, 1, 2)
    # Rows in original image will be manipulated if we do another reshape now.
    # 3. Tile rows,
    tiled_image = tiled_image.reshape((grows, gcols, imrows//grows, imcols//gcols, -1))
    
    # If we added an axis previously for grayscale images, get rid of it.
    if added_axis:
        tiled_image = tiled_image.squeeze(axis=-1)
    
    return tiled_image

if __name__ == "__main__":
    import doctest
    print("Running doctests...")
    doctest.testmod()    
