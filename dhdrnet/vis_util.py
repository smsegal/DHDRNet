import matplotlib.pyplot as plt
import numpy as np
from more_itertools import interleave
from mpl_toolkits.axes_grid1 import ImageGrid


def show_image_pair(im1: np.ndarray, im2: np.ndarray):
    fig = plt.figure(figsize=(12, 12))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.1)

    for ax, im in zip(grid, [im1, im2]):
        ax.imshow(im)


def show_exp_group(*images):
    images = list(images)
    num_im = len(images)
    fig = plt.figure(figsize=(12 * num_im, 12))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.1)


def show_image_groups(*image_groups):
    image_groups = list(image_groups)
    group1 = image_groups[0]
    interleaved = interleave(*image_groups)
    fig = plt.figure(figsize=(30, 30))
    grid = ImageGrid(
        fig, 111, nrows_ncols=(len(group1), len(image_groups)), axes_pad=0.1
    )

    for ax, im in zip(grid, interleaved):
        ax.imshow(im)
