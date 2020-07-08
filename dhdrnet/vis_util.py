import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from more_itertools import flatten, interleave
from mpl_toolkits.axes_grid1 import ImageGrid

from dhdrnet import image_loader


def show_image_pair(im1: np.ndarray, im2: np.ndarray, title=None):
    fig = plt.figure(figsize=(12, 8))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.1)

    for ax, im in zip(grid, [im1, im2]):
        ax.imshow(im)

    if title:
        fig.suptitle(title)


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
        selected_cols = list(
            flatten(
                [
                    [c for c in stats_df.columns if c.endswith(f"_{ev}")]
                    for ev in ev_stops
                ]
            )
        )
        grouped[cat] = stats_df.loc[:, ["name", *selected_cols]]

    # fig, ax = plt.subplots(1, len(grouped))
    # fig.tight_layout()

    for i, (ev, stats) in enumerate(grouped.items()):
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
    return type_stats
    # type_stats[f"optimal_{type}"].value_counts(sort=True, normalize=True).plot(
    #     kind="bar",
    #     title=f"{type} [-{ev},{ev}]",
    #     figsize=(10, 5),
    #     ax=ax[i],
    # )
    # ax[i].set_ylabel("Frequency")

    # plt.subplots_adjust(top=0.85, wspace=0.4)

    # if save_plots:
    #     plt.savefig(f"distribution_{type}")
    # plt.show()


from collections import defaultdict


def columns_with(df, name):
    """Returns view of df with columns that have the substring name in their column name"""
    selected_columns = [c for c in df.columns if name in c]
    return selected_columns


def get_metric_cat_groups(df: pd.DataFrame, categories):
    metrics = ("mse", "ssim", "ms_ssim")
    columns_to_groups = defaultdict(dict)
    for metric in metrics:
        for ev, cat in categories.items():
            for c in cat:
                for col in df.columns:
                    if col.startswith(metric) and col.endswith(str(c)):
                        columns_to_groups[metric][col] = ev

    return columns_to_groups
