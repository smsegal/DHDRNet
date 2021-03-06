from itertools import repeat

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
from more_itertools import flatten
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
from skimage.color import gray2rgb

from dhdrnet import image_loader
from dhdrnet.util import DATA_DIR


def rgb_bgr_swap(image: np.ndarray) -> np.ndarray:
    return image[..., [2, 1, 0]]


def show_image_pair(im1: np.ndarray, im2: np.ndarray, title=None, labels=None):
    fig = plt.figure(figsize=(12, 8))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.1)

    for ax, im, label in zip(grid, [im1, im2], labels):
        ax.imshow(im)
        ax.set_xlabel(label)

    if title:
        fig.suptitle(title)


def show_image_grid(images, img_labels=None, num_per_row=2, title=None):
    num_images = len(images)
    num_rows = num_images // num_per_row
    fig = plt.figure(figsize=(12 * num_per_row, 8 * num_rows))  # 4 x 3/2 ratio
    grid = ImageGrid(fig, 111, nrows_ncols=(num_rows, num_per_row), axes_pad=0.1)

    if img_labels is None:
        img_labels = repeat("")

    for (ax, im, label) in zip(grid, images, img_labels):
        ax.imshow(im)
        ax.set_xlabel(label)

    if title:
        fig.suptitle(title)

    plt.show()


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


def show_predictions(predictions, model_name, image_names, ev_options):
    for pred, img_name in zip(predictions, image_names):
        ev = ev_options[pred]
        pred_img = Image.open(DATA_DIR / "exposures" / f"{img_name}[{ev}].png")
        baseline_img = Image.open(DATA_DIR / "exposures" / f"{img_name}[0.0].png")
        show_image_pair(
            baseline_img,
            pred_img,
            title=f"{img_name} {model_name} (Input + Predicted)",
            labels=["Input", "Predicted"],
        )

        lower = np.minimum(ev, 0.0)
        upper = np.maximum(ev, 0.0)
        reconstructed = Image.open(
            DATA_DIR / "fusions" / f"{img_name}[{lower}][{upper}].png"
        )
        ground_truth = Image.open(DATA_DIR / "ground_truth" / f"{img_name}.png")
        show_image_pair(
            ground_truth,
            reconstructed,
            title=f"{img_name} {model_name} (Ground Truth + Reconstructed)",
            labels=["Ground Truth", "Reconstructed"],
        )


def recolour_image(weight_maps, colours):
    # weight maps are grayscale

    colour_mapping = mc.get_named_colors_mapping()
    colour_vals = np.array([mc.to_rgb(colour_mapping[cname]) for cname in colours])
    bwms = (binarize_weights(wm) for wm in weight_maps)

    # ensuring that the 0 values are translated to white we want to
    # add 1 to each (0,0,0) of the binarized wm so instead of figuring
    # out whatever weird broadcasting method might exist or looping
    # over slowly, just -1 from the colours we are weighting by,
    # multiply in the shifted colours, and then add 1 to the whole image
    coloured_wms = np.array(
        [
            (c[np.newaxis, np.newaxis, ...] - 1) * wm + 1
            for c, wm in zip(colour_vals, map(gray2rgb, bwms))
        ]
    )
    fused = np.ones_like(coloured_wms[0])  # shape: (NxMx3)

    # want the colour of the fused image to be that of largest
    # magnitude of the original weight maps
    brightest = np.max(weight_maps, axis=0)
    for wm, c in zip(weight_maps, colour_vals):
        fused[wm >= brightest] = c

    return coloured_wms, fused


def binarize_weights(wm):
    wm_float = wm.copy()
    mid = wm_float.mean()
    wm_float[wm_float < mid] = 0.0
    wm_float[wm_float >= mid] = 1.0
    return wm_float
