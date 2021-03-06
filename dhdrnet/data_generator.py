import argparse
import operator as op
from collections import defaultdict
from functools import partial, reduce
from itertools import product
from pathlib import Path
from typing import Callable, Collection, Dict, List, Optional

import cv2 as cv
import exifread
import numpy as np
import pandas as pd
import rawpy
import torch
from lpips import LPIPS as LPIPS_orig
from lpips import im2tensor
from more_itertools import collapse, flatten, one
from pandas import DataFrame
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)
from torch import nn
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from dhdrnet.util import DATA_DIR


def main(args):
    generator = DataGenerator(
        raw_path=Path(args.raw_path),
        out_path=Path(args.out_path),
        store_path=Path(args.store_path),
        exp_max=args.exp_max,
        exp_min=args.exp_min,
        exp_step=args.exp_step,
        multithreaded=args.single_thread,
        image_names=args.image_names,
    )
    if args.updown:
        print("Computing UpDown Strategy")
        generator.updown_strategy()
    else:
        print(
            f"Computing Images from EV {args.exp_min} to {args.exp_max} with stepsize of {args.exp_step}"
        )
        data = generator()
        data.to_csv(args.store_path)


class DataGenerator:
    def __init__(
        self,
        raw_path: Path,
        out_path: Path,
        store_path: Optional[Path] = None,
        compute_scores=True,
        exp_min: float = -3,
        exp_max: float = 6,
        exp_step: float = 0.25,
        multithreaded: bool = True,
        image_names=None,
        metrics: List[str] = ["rmse", "psnr", "ssim", "lpips"],
    ):
        self.exposures: np.ndarray = np.linspace(
            exp_min, exp_max, int((exp_max - exp_min) / exp_step + 1)
        )
        self.raw_path = raw_path
        self.out_path = out_path
        self.exp_out_path = self.out_path / "exposures"
        self.reconstructed_out_path = self.out_path / "fusions"
        self.fused_out_path = self.out_path / "fusions"
        self.updown_out_path = self.out_path / "updown"
        self.gt_out_path = self.out_path / "ground_truth"

        if image_names is None:
            self.image_names = [p.stem for p in (DATA_DIR / "dngs").iterdir()]
        elif isinstance(image_names, List):
            self.image_names = image_names
        else:
            self.image_names = list(flatten(pd.read_csv(image_names).to_numpy()))

        if compute_scores:
            all_metric_fns = {
                "rmse": rmse,
                "psnr": peak_signal_noise_ratio,
                "ssim": partial(structural_similarity, multichannel=True),
                "lpips": LPIPS(),
            }
            self.metric_fns: Dict[str, Callable] = {
                k: v for k, v in all_metric_fns.items() if k in metrics
            }

            self.metrics = metrics

        self.exp_out_path.mkdir(parents=True, exist_ok=True)
        self.reconstructed_out_path.mkdir(parents=True, exist_ok=True)
        self.updown_out_path.mkdir(parents=True, exist_ok=True)
        self.gt_out_path.mkdir(parents=True, exist_ok=True)
        self.fused_out_path.mkdir(parents=True, exist_ok=True)

        if store_path:
            self.store_path: Path = store_path
            self.store: DataFrame
            if store_path.is_file():
                self.store = pd.read_csv(
                    store_path, usecols=["name", "metric", "ev1", "ev2", "score"]
                )
            else:
                self.store = DataFrame(
                    data=None, columns=["name", "metric", "ev1", "ev2", "score"]
                )
                self.store_path.parent.mkdir(parents=True, exist_ok=True)

            self.store.to_csv(self.store_path, index=False)

        self.multithreaded = multithreaded

    def __call__(self):
        if self.multithreaded:
            stats = self.stats_dispatch_parallel()
        else:
            stats = self.stats_dispatch()

        stats_df = DataFrame.from_dict(stats)
        print(f"computed all stats, saved in {self.store_path}")
        return stats_df

    def stats_dispatch(self):
        stats = dict()
        for image_name in self.image_names:
            stats = reduce(nested_dict_merge, [stats, self.compute_stats(image_name)])
        return stats

    def stats_dispatch_parallel(self):
        stats = reduce(
            nested_dict_merge, thread_map(self.compute_stats, self.image_names)
        )
        return stats

    def updown_strategy(self):
        stats_df = self.compute_updown(self.image_names)
        stats_df.to_csv(self.store_path)

    def compute_updown(self, image_names):
        records = []
        for name in tqdm(image_names, total=len(image_names)):
            for ev in range(1, 6):
                updown_img = self.get_updown(name, ev)
                ground_truth = self.get_ground_truth(name)
                for metric in self.metrics:
                    score = self.metric_fns[metric](ground_truth, updown_img)
                    records.append((name, metric, ev, score))

        stats = pd.DataFrame.from_records(
            records, index="name", columns=["name", "metric", "ev", "score"]
        )
        return stats

    def compute_stats(self, img_name):
        stats = defaultdict(list)
        ev_options = [
            *self.exposures[self.exposures < 0],
            *self.exposures[self.exposures > 0],
        ]

        ground_truth = self.get_ground_truth(img_name)
        options = product(ev_options, self.metrics)
        for ev, metric in options:
            reconstruction = self.get_reconstruction(img_name, 0.0, ev)
            stats["name"].append(img_name)
            stats["metric"].append(metric)
            stats["ev1"].append(0.0)
            stats["ev2"].append(ev)
            stats["score"].append(self.metric_fns[metric](ground_truth, reconstruction))

        df = pd.DataFrame.from_dict(stats)
        df.to_csv(self.store_path, mode="a", header=None, index=False)
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

    def get_ground_truth(self, name):
        return self.get_fused(
            name,
            ev_list=[-4.0, -2.0, 0.0, 2.0, 4.0],
            out_path=self.gt_out_path,
            out_name=f"{name}.png",
        )

    def get_updown(self, name, ev):
        return self.get_fused(name, ev_list=[-ev, 0, ev], out_path=self.updown_out_path)

    def get_reconstruction(self, name, ev1, ev2):
        return self.get_fused(
            name, ev_list=[ev1, ev2], out_path=self.reconstructed_out_path
        )

    def fuse_fn(self, images) -> np.ndarray:
        return cv.createMergeMertens().process(images)

    def get_fused(
        self,
        name: str,
        ev_list: List[float],
        out_path: Path = None,
        out_name: str = None,
    ) -> np.ndarray:
        ev_list = sorted(ev_list)
        if out_path is None:
            out_path = self.fused_out_path
        if out_name is None:
            ev_in_name = reduce(op.add, [f"[{ev}]" for ev in ev_list])
            out_name = f"{name}{ev_in_name}.png"

        fused_path = out_path / out_name
        if fused_path.exists():
            fused_im = cv.imread(str(fused_path))
            # print(f"reading fused file: {fused_path}")
        else:
            # print(f"generating fused file: {fused_path}", sys.stderr)
            images = self.get_exposures(name, ev_list)
            fused_im = self.fuse_fn([im.astype("float32") for im in images])
            fused_im = np.clip(fused_im * 255, 0, 255).astype("uint8")

            cv.imwrite(str(fused_path), fused_im)
        return fused_im


def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))


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

            # uhh I think this might be swapping w/h
            newsize = tuple(postprocessed.shape[:2] // np.array([8]))[::-1]
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
    if isinstance(d1, list):
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


class LPIPS:
    def __init__(self, net: str = "alex") -> None:
        self.net: str = net
        self.usegpu: bool = torch.cuda.is_available()
        self.model: nn.Module = LPIPS_orig(net=net, spatial=False)
        if self.usegpu:
            self.model = self.model.cuda()

    def __call__(self, ima: torch.Tensor, imb: torch.Tensor) -> torch.Tensor:
        ima_t, imb_t = map(im2tensor, [ima, imb])
        if self.usegpu:
            ima_t = ima_t.cuda()
            imb_t = imb_t.cuda()

        dist = one(collapse(self.model.forward(ima_t, imb_t).data.cpu().numpy()))
        return dist


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Generate stats and images")

#     parser.add_argument("--out-path", "-o", help="where to save the processed files")
#     parser.add_argument("--raw-path", help="location of raw files")
#     parser.add_argument(
#         "--store-path",
#         help="file to store data in (created if does not exist)",
#         default="store",
#     )

#     parser.add_argument("--image-names", default=None)

#     parser.add_argument("--exp-min", default=-3)
#     parser.add_argument("--exp-max", default=6)
#     parser.add_argument("--exp-step", default=0.25)
#     parser.add_argument(
#         "--multithread", help="single threaded mode", action="store_true"
#     )
#     parser.add_argument(
#         "--updown", help="compute the updown strategy", action="store_true"
#     )

#     args = parser.parse_args()
#     main(args)
