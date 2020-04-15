import matplotlib.pyplot as plt
import numpy as np
from dhdrnet import image_loader
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


def view_data_sample(dataset, idx=None):
    if not idx:
        idx = np.random.choice(len(dataset))
    sample = dataset[idx]
    gt_fig, gt_ax = plt.subplots()
    gt_ax.imshow(sample["ground_truth"].swapaxes(-1, 0))
    gt_ax.set_title("ground_truth")

    mx_fig, mx_ax = plt.subplots()
    mx_ax.imshow(image_loader.clip_hdr(sample["mid_exposure"].swapaxes(-1, 0)))
    mx_ax.set_title("mid_exposure")

    exp_grid = ImageGrid(
        plt.figure(figsize=(30, 30)), 111, nrows_ncols=(1, 4), axes_pad=0.1
    )
    for ax, im in zip(exp_grid, sample["exposures"]):
        ax.imshow(im)
