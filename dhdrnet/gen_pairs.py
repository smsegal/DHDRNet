import argparse
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Collection, List

import cv2 as cv
import numpy as np
import pandas as pd
import rawpy
from more_itertools import distinct_combinations, flatten
from PIL import Image
from sewar import mse, msssim, ssim
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

        self.metricfuncs = {"mse": mse, "ssim": ssim, "ms_ssim": msssim}
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
        store.put(self.store_key, self.stats)

    def __call__(self, skip_exp=False):
        if not skip_exp:
            computed_paths = self.gen_exposures_dispatch()
            print("computed all raws")
        else:
            print("skipped computing exp images, assumed already done")
        stats = self.compute_stats(self.raw_path)
        print(f"computed all stats, saved in {self._storepath}")

    def gen_exposures_dispatch(self):
        raw_files = list(self.raw_path.iterdir())
        return process_map(self.create_needed_exposures, raw_files, chunksize=40)

    def compute_stats(self):
        stats = defaultdict(list)
        for i, (gt, name) in enumerate(
            process_map(self.get_ground_truth, self.raw_files, chunksize=40)
        ):
            stat_getter = self.image_stats(gt)

            img_pool = (
                (Image.open(self.exp_out_path / f"{name}[{ev}].png"), ev)
                for ev in ev_group
            )
            for ev_group in self.exposure_groups:
                img_pairs = distinct_combinations(img_pool, r=2,)
                for ((im_a, ev_a), (im_b, ev_b), metric) in product(
                    img_pool, self.metrics
                ):
                    reconstruction = self.fuse(im_a, im_b)
                    stats["name"].append(name)
                    stats["metric"].append(metric)
                    stats["ev_a"].append(ev_a)
                    stats["ev_b"].append(ev_b)
                    stats["score"].append(self.metricfuncs[metric](gt, reconstruction))

            if i % 20 == 0:
                df = pd.DataFrame.from_dict(stats)
                self.store.append(self.store_key, df)

    def create_needed_exposures(self, raw_fp):
        computed_exposures = []
        for ev in self.exposures:
            image_path = self.out_path / "exposures" / f"{raw_fp.stem}[{ev}].png"
            if image_path.exists():
                print(f"skipping previously generated: {image_path}")
                computed_exposures.append(ev)

        exposures = self.exposures - computed_exposures
        for image, ev in self.exposures_from_raw(raw_fp, exposures):
            image.save(self.exp_out_path / f"{raw_fp.stem}[{ev}].png")
        return raw_fp

    def get_ground_truth(self, raw_fp):
        img_name = raw_fp.stem
        gt_fp = self.gt_path / f"{img_name}.png"
        if gt_fp.exists():
            return Image.open(gt_fp)
        else:
            gt_img = self._generate_gt(img_name)
            gt_img.save(self.gt_out_path / f"{img_name}.png")
            return gt_img, img_name

    def _generate_gt(self, img_name, exp_group):
        image_inputs = [
            self.exp_out_path / f"{img_name}[{ev}].png"
            for ev in self.exposure_groups[exp_group]
        ]
        return self.fuse(image_inputs)

    def fuse(self, images: List[np.ndarray]) -> np.ndarray:
        merged = self._fusefunc(images)
        merged_rgb = merged  # [:, :, [2, 1, 0]]
        return merged_rgb

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

    parser.add_argument("raw_dir", help="location of raw files")
    parser.add_argument("gt_dir", help="location of ground truth (merged) files")
    parser.add_argument("--out-dir", "-o", help="where to save the processed files")

    args = parser.parse_args()
    main(args)
