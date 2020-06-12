import argparse
from collections import defaultdict
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from itertools import repeat, combinations
from pathlib import Path
from typing import Collection, List

import cv2 as cv
import numpy as np
import pandas as pd
import rawpy
from PIL import Image
from more_itertools import flatten
# from sewar import mse, msssim, ssim
from skimage.metrics import mean_squared_error, structural_similarity
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from dhdrnet.util import ROOT_DIR


def main(args):
    generator = GenAllPairs(
        ev_maximums=range(4, 8),
        raw_path=Path(args.raw_path),
        gt_path=Path(args.gt_path),
        out_path=Path(args.out_path),
    )
    generator(skip_exp=args.skip_exp)

from functools import partial
class GenAllPairs:
    def __init__(self, ev_maximums, raw_path: Path, gt_path: Path, out_path: Path):
        self.exposure_groups = [np.linspace(-ev, ev, 5) for ev in ev_maximums]
        self.exposures = set(sorted(flatten(self.exposure_groups)))
        self.raw_path = raw_path
        self.gt_path = gt_path
        self.out_path = out_path
        self.exp_out_path = self.out_path / "exposures"
        self.reconstructed_out_path = self.out_path / "reconstructions"
        self.gt_out_path = self.out_path / "ground_truth"

        self.metricfuncs = {"mse": mean_squared_error, "ssim": partial(structural_similarity, multichannel=True)}
        self.metrics = list(self.metricfuncs.keys())

        self.exp_out_path.mkdir(parents=True, exist_ok=True)
        self.reconstructed_out_path.mkdir(parents=True, exist_ok=True)
        self.gt_out_path.mkdir(parents=True, exist_ok=True)

        self._fusefunc = cv.createMergeMertens().process

        self._storepath = ROOT_DIR / "precomputed_data" / "store.h5"

        self.store = pd.HDFStore(str(self._storepath))
        self.store_key = "fusion_stats"
        self.stats = pd.DataFrame(
            data=None, columns=["name", "metric", "ev_a", "ev_b", "score"]
        )
        self.store.put(self.store_key, self.stats, format="table")

    def __call__(self, skip_exp=False):
        if not skip_exp:
            computed_paths = self.gen_exposures_dispatch()
            print("computed all raws")
        else:
            print("skipped computing exp images, assumed already done")
        stats = self.compute_stats()
        print(f"computed all stats, saved in {self._storepath}")

    def gen_exposures_dispatch(self):
        raw_files = list(self.raw_path.iterdir())
        return process_map(self.create_needed_exposures, raw_files, chunksize=40)

    def compute_stats(self):
        stats = defaultdict(list)
        raw_files = list(self.raw_path.iterdir())
        with ThreadPoolExecutor() as executor:
            for e_idx, ev_group in enumerate(self.exposure_groups):
                for (gt, raw_fp) in tqdm(executor.map(self.get_ground_truth, raw_files, repeat(e_idx, len(raw_files)), chunksize=40),
                                         total=len(raw_files)):
                    name = raw_fp.stem
                    img_pool = [
                        (cv.imread(str(self.exp_out_path / f"{name}[{ev}].png")), ev)
                        for ev in ev_group
                    ]

                    img_pairs = list(combinations(img_pool, r=2))

                    # for ((im_a, ev_a), (im_b, ev_b)), metric in img_metric_product:
                    for reconstruction, ev_a, ev_b in tqdm(
                            executor.map(self.get_reconstruction, repeat(name), img_pairs, chunksize=40),
                            total=len(img_pairs),
                    ):
                        for metric in self.metrics:
                            stats["name"].append(name)
                            stats["metric"].append(metric)
                            stats["ev_a"].append(ev_a)
                            stats["ev_b"].append(ev_b)
                            stats["score"].append(self.metricfuncs[metric](gt, reconstruction))

                    df = pd.DataFrame.from_dict(stats)
                    self.store.append(self.store_key, df)
                    print(f"appended {name} to store!")

    def create_needed_exposures(self, raw_fp):
        computed_exposures = []
        for ev in self.exposures:
            image_path = self.out_path / "exposures" / f"{raw_fp.stem}[{ev}].png"
            if image_path.exists():
                print(f"skipping previously generated: {image_path}")
                computed_exposures.append(ev)

        exposures = self.exposures - set(computed_exposures)
        for image, ev in self.exposures_from_raw(raw_fp, exposures):
            image.save(self.exp_out_path / f"{raw_fp.stem}[{ev}].png")
        return raw_fp

    def get_ground_truth(self, raw_fp, exp_group):
        img_name = raw_fp.stem
        gt_fp = self.gt_path / f"{img_name}.png"
        if gt_fp.exists():
            gt_img = cv.imread(str(gt_fp))
        else:
            image_inputs = [
                cv.imread(str(self.exp_out_path / f"{img_name}[{ev}].png"))
                for ev in self.exposure_groups[exp_group]
            ]
            gt_img = self.fuse(*image_inputs)
            cv.imwrite(str(gt_fp), gt_img)
        return gt_img, raw_fp

    def get_reconstruction(self, *args):
        name, ((im_a, ev_a), (im_b, ev_b)) = args
        rec_path = self.reconstructed_out_path / f"{name}[{ev_a}][{ev_b}].png"
        if rec_path.exists():
            rec_img = np.array(Image.open(rec_path))
        else:
            rec_img = self.fuse(im_a, im_b)
            cv.imwrite(str(rec_path), rec_img)
        return rec_img, ev_a, ev_b

    def fuse(self, *images: List[np.ndarray]) -> np.ndarray:
        merged = self._fusefunc(images)
        merged = np.clip(merged * 255, 0, 255).astype("uint8")
        return merged

    @staticmethod
    def exposures_from_raw(raw_path: Path, exposures: Collection):
        with rawpy.imread(str(raw_path)) as raw:
            black_levels = raw.black_level_per_channel
            raw_orig = raw.raw_image.copy()

            # tiled to add to the right channels of the bayer image
            black_levels_tiled = np.tile(
                black_levels, (raw_orig.shape // np.array([1, 4]))
            )
            raw_im = np.maximum(raw_orig, black_levels_tiled) - black_levels_tiled

            for exposure in exposures:
                im = raw_im * (2 ** exposure)
                im = im + black_levels_tiled
                im = np.minimum(im, 2 ** 16 - 1)
                raw.raw_image[:, :] = im
                postprocessed = raw.postprocess(use_camera_wb=True, no_auto_bright=True)
                yield Image.fromarray(postprocessed, "RGB"), exposure


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate stats and images")

    parser.add_argument("raw_path", help="location of raw files")
    parser.add_argument("gt_path", help="location of ground truth (merged) files")
    parser.add_argument("--skip-exp", action="store_true")
    parser.add_argument("--out-path", "-o", help="where to save the processed files")

    args = parser.parse_args()
    main(args)
