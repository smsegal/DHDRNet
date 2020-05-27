import matplotlib.pyplot as plt
import numpy as np
from more_itertools import interleave, flatten
from mpl_toolkits.axes_grid1 import ImageGrid

from dhdrnet import image_loader


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


def get_pred_dist(stats_df, categories, type, save_plots=False):
    grouped = dict()
    for cat, ev_stops in categories.items():
        selected_cols = list(flatten([[c for c in stats_df.columns if c.endswith(f"_{ev}")] for ev in ev_stops]))
        grouped[cat] = stats_df.loc[:, ["name", *selected_cols]]

    for ev, stats in grouped.items():
        type_stats = stats.loc[
                     :, ["name", *[c for c in stats.columns if c.startswith(type)]]
                     ]
        type_stats = type_stats.rename(lambda c: c.split("_")[-1], axis="columns")
        if "ssim" in type:
            type_stats[f"optimal_{type}"] = (
                type_stats.loc[:, f"-{ev}.0":f"{ev}.0"].idxmax(axis=1).apply(float)
            )
        else:
            type_stats[f"optimal_{type}"] = (
                type_stats.loc[:, f"-{ev}.0":f"{ev}.0"].idxmin(axis=1).apply(float)
            )

        plt.figure()
        ax = type_stats[f"optimal_{type}"].value_counts(sort=False, normalize=True).plot(
            kind="bar", title=f"Prediction Distribution EV Range [-{ev},{ev}]",
            figsize=(15, 15)
        )
        ax.set_xlabel(f"EV Choices for {type}")
        ax.set_ylabel("Frequency")

        if save_plots:
            plt.savefig(f"distribution_ev{ev}_{type}")
        plt.show()
