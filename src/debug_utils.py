import math
import numpy as np
from keras.preprocessing.image import load_img
from data import construct_img_path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: Could not load matplotlib")

TEST_DIR = "../input/test/"


def tile_arrays(arrays, grid_shape):
    """
    Arguments:
        arrays -- 3d (or 4d if color) array of images (channels last) to tile into a grid
        grid_shape -- the shape of the output grid.
                      REQUIRED: the number of cells be greater than arrays.shape[0]
    Returns:
        The arrays in the grid of input shape
    """
    assert arrays.ndim in {3, 4}
    full_shape = (grid_shape[0] * arrays.shape[1], grid_shape[1] * arrays.shape[2])
    if arrays.ndim == 3:
        grid = np.empty(full_shape)
    elif arrays.ndim == 4:
        grid = np.empty(full_shape + (arrays.shape[-1],))
    else:
        raise ValueError("Array is incorrect dim of %s. Must be dimension 3 or 4" % arrays.ndim)
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            cell_ind = i * grid_shape[1] + j
            if cell_ind < arrays.shape[0]:
                grid[i * arrays.shape[1]:(i + 1) * arrays.shape[1],
                     j * arrays.shape[2]:(j + 1) * arrays.shape[2]] = arrays[cell_ind]
    return grid


def plot_tiles(arrays, grid_shape, figsize=None, ion=False):
    if ion:
        plt.ion()
    # If no input figure size use 8x8 figure size for each grid cell
    figsize = (grid_shape[0] * 8, grid_shape[1] * 8)
    plt.figure(figsize=figsize)
    if arrays.ndim == 3:
        plt.imshow(tile_arrays(arrays, grid_shape), cmap='gray')
    else:
        plt.imshow(tile_arrays(arrays, grid_shape))

    if ion:
        plt.pause(1)
        plt.close()
    else:
        plt.show()


def infer_grid_shape(tiles):
    s2 = int(math.ceil(math.sqrt(tiles)))
    if s2 % 2 != 0:
        s2 -= 1
    s1 = int(math.ceil(tiles / s2))
    if s2 < s1:
        return (s2, s1)
    return (s1, s2)


def plot_img_mask_tiled(imgs, masks, grid_shape=None, figsize=None, ion=False):
    assert imgs.shape[:-1] == masks.shape, "Image and masks shapes do not match!"
    merged = np.empty((imgs.shape[0] * 2,) + imgs.shape[1:])
    merged[:-1:2] = imgs
    merged[1::2] = masks[..., np.newaxis]
    if grid_shape is None:
        grid_shape = infer_grid_shape(merged.shape[0])
    plot_tiles(merged, grid_shape, figsize=figsize, ion=ion)


def examine_prediction(long_id, input_img, output_mask, resized_mask, orig_imgsize=(1918, 1280), ion=False):
    if ion:
        plt.ion()
    fig, ax = plt.subplots(2, 2, figsize=[20, 20])
    orig_img = load_img(construct_img_path(long_id, TEST_DIR, mask=False))
    assert orig_img.size == orig_imgsize, "Loaded image does not match orig_imgsize."
    # Display the images
    ax[0, 0].imshow(orig_img)
    ax[0, 1].imshow(resized_mask, cmap='gray')
    ax[1, 0].imshow(input_img)
    ax[1, 1].imshow(output_mask, cmap='gray')
    if ion:
        plt.pause(1)
        plt.clf()
    else:
        plt.show()
