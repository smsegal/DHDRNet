import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations, product
from pathlib import Path
from typing import Callable, Collection, List

import colour as co
import cv2 as cv
import numpy as np
import rawpy
from more_itertools import flatten
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import dhdrnet.image_loader as il
from dhdrnet.image_loader import gen_multi_exposures
from dhdrnet.reconstruction import mertens_fuse, reconstruction_stats
from dhdrnet.util import append_csv, get_project_root, suppress

DATA_DIR = get_project_root() / "data"


def main():
    # Let's load the dng images and adjust the exposure on the fly.
    raw_imgs = list((DATA_DIR / "dngs").iterdir())
    gt_imgs = list((DATA_DIR / "merged").iterdir())
    gen_all_fuse_options([mertens_fuse], raw_imgs, gt_imgs, DATA_DIR / "all_processed")
    print("Done!")


def gen(
    image_paths: Collection[Path],
    processed_dir: Path,
    fuse_fun: Callable[[Collection[Path]], Collection[Path]],
) -> Collection[Path]:
    image_paths = list(image_paths)

    print(f"Generating multi exposure files in {processed_dir} (Skipping existing)")

    processed_dir.mkdir(exist_ok=True)
    multi_exposure_paths = list(processed_dir.iterdir())
    if len(multi_exposure_paths) == 0:
        multi_exposure_paths = il.parallel_write_multi_exp(image_paths, processed_dir)

    grouped_ldr_paths = list(il.multi_exp_paths(image_paths, processed_dir))
    print("Generating merged files")
    return list(il.parallel_cv_fused(fuse_fun, grouped_ldr_paths))


def gen_optimal_pairs(fuse_func, raw_dir, gt_dir, remaining, out_dir, logname):
    raw_gt = [
        (raw_dir / f"{entry}.dng", gt_dir / f"{entry}.png") for entry in remaining
    ]
    ev_max = [4, 5, 6, 7]
    ev_ranges = [np.linspace(-ev, ev, 5) for ev in ev_max]
    all_csv_keys = [
        f"{metric}_[{ev_a}][{ev_b}]"
        for metric, (ev_a, ev_b) in flatten(
            [
                product(["mse", "ssim", "ms_ssim"], combinations(ev_range, 2))
                for ev_range in ev_ranges
            ]
        )
    ]
    with ThreadPoolExecutor() as executor:
        for raw, gt in tqdm(raw_gt):
            future = executor.submit(
                optimal_fusion_stats, fuse_func, ev_ranges, raw, gt, out_dir
            )
            records = future.result()
            append_csv(records, out_dir / logname, fieldnames=["name", *all_csv_keys])


class GenAllPairs:
    def __init__(self, ev_maximums, raw_path: Path, gt_path: Path, out_path: Path):
        self.exposures = set(
            sorted(flatten(np.linspace(-ev, ev, 5) for ev in ev_maximums))
        )
        self.raw_path = raw_path
        self.gt_path = gt_path
        self.out_path = out_path
        self.exp_out_path = self.out_path / "exposures"
        self.reconstructed_path = self.out_path / "reconstructions"

        self.exp_out_path.mkdir(parents=True, exist_ok=True)
        self.reconstructed_path.mkdir(parents=True, exist_ok=True)

        # self.store = pd.HDFStore(ROOT_DIR / "precomputed_data" / "store.h5")
        # self.stats = pd.DataFrame(data=None, columns=["name", "metric", "ev a", "ev b"])

    def gen_exposures_dispatch(self):
        raw_files = list(self.raw_path.iterdir())
        return process_map(self.create_needed_exposures, raw_files, chunksize=40)

    def compute_stats(self):
        pass

    def __call__(self):
        computed_paths = self.gen_exposures_dispatch()
        print("computed all raws")

    def create_needed_exposures(self, raw_fp):
        computed_exposures = set()
        for ev in self.exposures:
            image_path = self.out_path / "exposures" / f"{raw_fp.stem}[{ev}].png"
            if image_path.exists():
                print(f"skipping previously generated: {image_path}")
                computed_exposures.add(ev)

        exposures = self.exposures - computed_exposures
        for image, ev in exposures_from_raw(raw_fp, exposures):
            image.save(self.out_path / "exposures" / f"{raw_fp.stem}[{ev}].png")
        return raw_fp


def image_statistics(ground_truth):
    mertens_merger = cv.createMergeMertens()

    def mertens_fuse(images: List[np.ndarray]) -> np.ndarray:
        merged = mertens_merger.process(images)
        merged_rgb = merged  # [:, :, [2, 1, 0]]
        return merged_rgb

    def compute_stats(ima, exa, imb, exb):
        reconstruction = mertens_fuse(ima, imb)
        return reconstruction_stats(reconstruction, ground_truth)


def exposures_from_raw(raw_path: Path, exposures: Collection):
    with rawpy.imread(str(raw_path)) as raw:
        black_levels = raw.black_level_per_channel
        raw_orig = raw.raw_image.copy()

        # tiled to add to the right channels of the bayer image
        black_levels_tiled = np.tile(black_levels, (raw_orig.shape // np.array([1, 4])))
        raw_im = np.maximum(raw_orig, black_levels_tiled) - black_levels_tiled

        for exposure in exposures:
            im = raw_im * (2 ** exposure)
            im = im + black_levels_tiled
            im = np.minimum(im, 2 ** 16 - 1)
            raw.raw_image[:, :] = im
            postprocessed = raw.postprocess(use_camera_wb=True, no_auto_bright=True)
            yield Image.fromarray(postprocessed, "RGB"), exposure


def optimal_images(fuse_func, ev_range, raw, gt, out_dir):
    for ev in ev_range:
        pass


def optimal_fusion_stats(fuse_func, ev_ranges, raw, gt, out_dir):
    logs = {"name": gt.stem}
    gt_img = co.read_image(gt)

    for ev_range in ev_ranges:
        exposures = exposures_from_raw(raw, ev_range)
        ev_exposures = zip(ev_range, exposures)
        for (ev_a, a), (ev_b, b) in combinations(ev_exposures, 2):
            if (out_dir / f"{gt.stem}_[{ev_a}][{ev_b}].png").exists():
                print("")  # newline since progress bar gets in the way
                print(
                    f"image {gt.stem}_[{ev_a}][{ev_b}].png already exists, skipped generation"
                )
                # fused_img = np.array(cv.imread(str(out_dir / f"{gt.stem}_[{ev_a}][{ev_b}].png"))).astype(np.uint8)
                fused_img = co.read_image(out_dir / f"{gt.stem}_[{ev_a}][{ev_b}].png")
            else:
                fused_img = fuse_func([a, b])
                with suppress(err=True, out=True):
                    co.write_image(
                        fused_img,
                        out_dir / f"{gt.stem}_[{ev_a}][{ev_b}].png",
                        bit_depth="float32",
                    )

            mse, ssim, ms_ssim = reconstruction_stats(fused_img, gt_img)
            logs.update(
                {
                    f"mse_[{ev_a}][{ev_b}]": mse,
                    f"ssim_[{ev_a}][{ev_b}]": ssim,
                    f"ms_ssim_[{ev_a}][{ev_b}]": ms_ssim,
                }
            )
    return logs


def gen_all_fuse_options(
    fuse_funcs, raw_images: List[Path], gt_images: List[Path], out_dir,
):
    all_ev_steps = [5]  # range(5, 10)
    sorted_raw = sorted(raw_images, key=lambda p: p.stem)
    sorted_gt = sorted(gt_images, key=lambda p: p.stem)
    ev_max = [4, 5, 6, 7]
    all_combinations = list(
        product(fuse_funcs, ev_max, all_ev_steps, zip(sorted_raw, sorted_gt))
    )
    with ThreadPoolExecutor(max_workers=None) as executor:
        for a in all_combinations:
            future = executor.submit(fuse_exposures, out_dir, *a)
            print(f"computing: {future.result()}")


def fuse_exposures(out_dir, fuse_func, ev_max, ev_steps, raw_gt):
    raw, gt = raw_gt
    ev_range = list(np.linspace(-ev_max, ev_max, ev_steps))
    exposures = gen_multi_exposures(raw, *ev_range)
    mid_exp = exposures.pop(len(exposures) // 2)
    ev_range.remove(0)
    ev_dir = out_dir / f"max_ev{ev_max}"
    ev_dir.mkdir(parents=True, exist_ok=True)
    for ev, exp in zip(ev_range, exposures):
        dest = ev_dir / f"{raw.stem}_ev{ev}.png"
        if dest.exists():  # don't recompute
            continue

        fused = fuse_func([mid_exp, exp])

        # save the results to disk
        co.write_image(fused, dest)

    return raw.stem


def write_log(dest, opt_log):
    with open(f"{dest}.json", "w") as f:
        json.dump(opt_log, f)


if __name__ == "__main__":
    co.utilities.filter_warnings()

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    parser.add_argument("raw_dir", help="location of raw files")
    parser.add_argument("gt_dir", help="location of ground truth (merged) files")
    parser.add_argument("--out-dir", "-o", help="where to save the processed files")
    parser.add_argument("--log-name", "-l", help="log to record stats in")

    group.add_argument(
        "--gen-opt", "-go", help="generate with baseline EV0", action="store_true"
    )
    group.add_argument(
        "--gen-baseline", "-gb", help="generate with baseline EV0", action="store_true"
    )

    args = parser.parse_args()
    if args.gen_opt:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(exist_ok=True)
        raw_dir = Path(args.raw_dir)
        gt_dir = Path(args.gt_dir)
        logname = args.log_name
        # if (out_dir / logname).exists():
        #     print("existing records found, excluding existing records from computation")
        #     stats_file = read_stats_from_file(out_dir / logname)
        #     remaining = find_remaining(stats_file, gt_dir, 121)
        # else:
        # now just skipping over files already genned, but still computing stats as some might have been missed
        # remaining = [gt.stem for gt in gt_dir.iterdir()]
        #
        # gen_optimal_pairs(
        #     mertens_fuse, raw_dir, gt_dir, remaining, out_dir, args.log_name
        # )
        gen = GenAllPairs(
            ev_maximums=range(4, 8), gt_path=gt_dir, raw_path=raw_dir, out_path=out_dir
        )
        gen()
