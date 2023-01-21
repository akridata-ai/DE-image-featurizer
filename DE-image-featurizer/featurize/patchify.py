"""This py file contains the patchify function that is used to get sub-tiles of the images"""
import numpy as np


def patchify(image, grid_shape):
    """
    The patchify function takes an image and a grid shape (in the form of a tuple) as input.
    It then divides the image into tiles of equal size, given by grid_shape.
    The function returns a numpy array containing all these tiles in a grid.

    Args:
        image: Pass the image to be tiled
        grid_shape=(6: Define the number of tiles in each direction
        6): Specify the size of the patches

    Returns:
        A numpy array of shape (6, 6, tile_size[0], tile_size[0], 3), where the first two dimensions are the row and column index of each patch

    """
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    image_shape = image.shape

    tile_size = (image_shape[0] // grid_shape[0], image_shape[1] // grid_shape[1])
    tiled_image = np.empty((*grid_shape, *tile_size, image_shape[-1]), dtype='uint8')

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            cropped_img = image[i * tile_size[0]:(i + 1) * tile_size[0],
                          j * tile_size[1]:(j + 1) * tile_size[1], :]
            tiled_image[i][j] = cropped_img

    return tiled_image.squeeze()
