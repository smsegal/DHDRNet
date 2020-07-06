import argparse
import datetime
import operator as op
from collections import defaultdict
from functools import partial, reduce
from pathlib import Path
from typing import Collection, List

import cv2 as cv
import exifread
import numpy as np
import pandas as pd
import rawpy
import torch
from more_itertools.more import distinct_combinations
from itertools import product
from perceptual_similarity import PerceptualLoss
from perceptual_similarity.util.util import im2tensor

from skimage.metrics import mean_squared_error, structural_similarity
from tqdm.contrib.concurrent import thread_map

from dhdrnet.util import ROOT_DIR


def main(args):
    generator = GenAllPairs(
        raw_path=Path(args.raw_path),
        out_path=Path(args.out_path),
        store_path=Path(args.store_path),
        single_threaded=args.single_thread,
    )
    generator()


class GenAllPairs:
    def __init__(
        self, raw_path: Path, out_path: Path, store_path: Path, single_threaded: bool
    ):
        self.exposures = np.arange(-3, 6.25, 0.25)
        self.raw_path = raw_path
        self.out_path = out_path
        self.exp_out_path = self.out_path / "exposures"
        self.reconstructed_out_path = self.out_path / "reconstructions"
        self.gt_path = self.out_path / "ground_truth"

        self.image_names = [rf.stem for rf in self.raw_path.iterdir()]

        self.metricfuncs = {
            "mse": mean_squared_error,
            "ssim": partial(structural_similarity, multichannel=True),
            "perceptual": PerceptualMetric(),
        }
        self.metrics = list(self.metricfuncs.keys())

        self.exp_out_path.mkdir(parents=True, exist_ok=True)
        self.reconstructed_out_path.mkdir(parents=True, exist_ok=True)
        self.gt_path.mkdir(parents=True, exist_ok=True)

        self.store_path = store_path
        if store_path.is_file():
            self.store = pd.read_csv(
                store_path, usecols=["name", "metric", "ev1", "ev2", "score"]
            )
        else:
            self.store = pd.DataFrame(
                data=None, columns=["name", "metric", "ev1", "ev2", "score"]
            )
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.single_threaded = single_threaded

    def __call__(self):
        if self.single_threaded:
            self.stats_dispatch()
        else:
            self.stats_dispatch_parallel()
        print(f"computed all stats, saved in {self.store}")

    def stats_dispatch(self):
        stats = dict()
        for i, image_name in enumerate(self.image_names):
            stats = reduce(nested_dict_merge, [stats, self.compute_stats(image_name)])
        return stats

    def stats_dispatch_parallel(self):
        stats = reduce(
            nested_dict_merge, thread_map(self.compute_stats, self.image_names)
        )
        return stats

    def compute_stats(self, img_name):
        stats = defaultdict(list)
        ev_combinations = distinct_combinations(self.exposures, r=2)

        options_df = pd.DataFrame.from_records(
            product([img_name], ev_combinations, self.metrics), columns=["name", "ev", "metric"]
        )
        options_df["ev1"] = options_df["ev"].apply(lambda d: d[0])
        options_df["ev2"] = options_df["ev"].apply(lambda d: d[1])
        im_stats = self.store[self.store["name"] == img_name]

        remaining: pd.DataFrame = pd.concat([im_stats, options_df]).drop_duplicates(
            keep=False, subset=["name", "ev1", "ev2", "metric"]
        ).drop(columns=["ev", "score"])

        for name, ev1, ev2, metric in remaining.itertuples(index=False):
            ground_truth = self.get_ground_truth(name)
            reconstruction = self.get_reconstruction(name, ev1, ev2)
            stats["name"].append(img_name)
            stats["metric"].append(metric)
            stats["ev1"].append(ev1)
            stats["ev2"].append(ev2)
            stats["score"].append(
                self.metricfuncs[metric](ground_truth, reconstruction)
            )

        if len(remaining):
            df = pd.DataFrame.from_dict(stats)
            if self.store_path.exists():
                header = None
            else:
                header = df.columns

            df.to_csv(self.store_path, mode="a", header=header)
        return stats

    def get_exposures(self, image_name, exposures):
        raw_fp = self.raw_path / f"{image_name}.dng"
        computed_exposures = []
        for ev in exposures:
            image_path = self.exp_out_path / f"{image_name}[{ev}].png"
            if image_path.exists():
                # print(f"reading previously generated: {image_path}")
                computed_exposures.append(ev)
                yield cv.imread(str(image_path))

        exposures = set(exposures) - set(computed_exposures)
        if len(exposures) > 0:
            for image, ev in zip(exposures_from_raw(raw_fp, exposures), exposures):
                yield image
                cv.imwrite(str(self.exp_out_path / f"{image_name}[{ev}].png"), image)

    def get_ground_truth(self, image_name):
        gt_fp = self.gt_path / f"{image_name}.png"
        if gt_fp.exists():
            gt_img = cv.imread(str(gt_fp))
        else:
            image_inputs = self.get_exposures(image_name, self.exposures)
            gt_img = fuse(*image_inputs)
            cv.imwrite(str(gt_fp), gt_img)
        return gt_img

    def get_reconstruction(self, name, ev1, ev2):
        rec_path = self.reconstructed_out_path / f"{name}[{ev1}][{ev2}].png"
        if rec_path.exists():
            rec_img = cv.imread(str(rec_path))
        else:
            im1, im2 = self.get_exposures(name, [ev1, ev2])
            rec_img = fuse(im1, im2)
            cv.imwrite(str(rec_path), rec_img)
        return rec_img


def exposures_from_raw(raw_path: Path, exposures: Collection, for_opencv=True):
    # opencv needs color channels in BGR vs RGB for idk... everything? else
    if for_opencv:
        channel_swapper = [2, 1, 0]
    else:
        channel_swapper = [0, 1, 2]

    baseline_ev = ev_from_exif(raw_path)

    with rawpy.imread(str(raw_path)) as raw:
        black_levels = raw.black_level_per_channel
        raw_orig = raw.raw_image.copy()

        # tiled to add to the right channels of the bayer image
        black_levels_tiled = np.tile(black_levels, (raw_orig.shape // np.array([1, 4])))
        raw_im = np.maximum(raw_orig, black_levels_tiled) - black_levels_tiled

        for exposure in exposures:
            # adjust exposures to account for the given baseline
            im = raw_im * (2 ** (exposure - baseline_ev))
            im = im + black_levels_tiled
            im = np.minimum(im, (2 ** 16) - 1)
            raw.raw_image[:, :] = im
            postprocessed = raw.postprocess(use_camera_wb=True, no_auto_bright=True)[
                :, :, channel_swapper
            ]
            newsize = tuple(postprocessed.shape[:2] // np.array([8]))
            yield cv.resize(postprocessed, dsize=newsize, interpolation=cv.INTER_AREA)


def ev_from_exif(img_path: Path):
    tags = exifread.process_file(img_path.open("rb"), details=False)
    fnumber = tags["EXIF FNumber"].values[0].num
    shutter_speed_ratio = tags["EXIF ShutterSpeedValue"].values[0]
    return (2 * np.log2(fnumber)) + np.log2(
        shutter_speed_ratio.den / shutter_speed_ratio.num
    )


def iso_from_exif(img_path: Path):
    tags = exifread.process_file(img_path.open("rb"), details=False)
    return tags["EXIF ISOSpeedRatings"].values[0]


def nested_dict_merge(d1, d2):
    merged = dict()

    # base case, have lists as leaves
    if type(d1) == list:
        return d1 + d2

    combined_keys = set(reduce(op.add, (list(d.keys()) for d in (d1, d2))))
    for k in combined_keys:
        if k in d1 and k in d2:
            merged[k] = nested_dict_merge(d1[k], d2[k])
        elif k in d1:
            merged[k] = d1[k]
        elif k in d2:
            merged[k] = d2[k]
    return merged


def _ff(images):
    return cv.createMergeMertens().process(images)


def fuse(*images: List[np.ndarray]) -> np.ndarray:
    merged = _ff(images)
    merged = np.clip(merged * 255, 0, 255).astype("uint8")
    return merged


def PerceptualMetric(model="net-lin", net="alex"):
    model = PerceptualLoss(
        model=model, net=net, use_gpu=torch.cuda.is_available(), gpu_ids=[0]
    )

    def perceptual_loss_metric(ima, imb):
        ima_t, imb_t = map(im2tensor, [ima, imb])
        d = model.forward(ima_t, imb_t).detach().cpu().numpy().flatten()[0]
        return d

    return perceptual_loss_metric


def _present(df, name, metric, ev1, ev2):
    if len(df) > 0:
        record = df[
            (df["metric"] == metric)
            & (df["ev1"] == ev1)
            & (df["ev2"] == ev2)
            & (df["name"] == name)
        ]
        return len(record)
    else:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate stats and images")

    parser.add_argument("--out-path", "-o", help="where to save the processed files")
    parser.add_argument("--raw-path", help="location of raw files")
    parser.add_argument(
        "--store-path",
        help="file to store data in (created if does not exist)",
        default="store",
    )
    parser.add_argument(
        "--single-thread", help="single threaded mode", action="store_true"
    )

    args = parser.parse_args()
    main(args)
