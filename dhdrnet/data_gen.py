import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations, product
from pathlib import Path
from typing import Callable, Collection, List

import colour as co
import numpy as np
from more_itertools import flatten
from tqdm import tqdm

import dhdrnet.image_loader as il
from dhdrnet.image_loader import gen_multi_exposures
from dhdrnet.reconstruction import (
    mertens_fuse,
    reconstruction_stats,
)
from dhdrnet.util import get_project_root, append_csv, suppress

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


def optimal_fusion_stats(fuse_func, ev_ranges, raw, gt, out_dir):
    logs = {"name": gt.stem}
    gt_img = co.read_image(gt)

    for ev_range in ev_ranges:
        exposures = gen_multi_exposures(raw, *ev_range)
        ev_exposures = zip(ev_range, exposures)
        for (ev_a, a), (ev_b, b) in combinations(ev_exposures, 2):
            if (out_dir / f"{gt.stem}_[{ev_a}][{ev_b}].png").exists():
                print("")  # newline since progress bar gets in the way
                print(f"image {gt.stem}_[{ev_a}][{ev_b}].png already exists, skipped generation")
                # fused_img = np.array(cv.imread(str(out_dir / f"{gt.stem}_[{ev_a}][{ev_b}].png"))).astype(np.uint8)
                fused_img = co.read_image(out_dir / f"{gt.stem}_[{ev_a}][{ev_b}].png")
            else:
                fused_img = fuse_func([a, b])
                with suppress(err=True, out=True):
                    co.write_image(
                        fused_img, out_dir / f"{gt.stem}_[{ev_a}][{ev_b}].png", bit_depth="float32"
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
        remaining = [gt.stem for gt in gt_dir.iterdir()]

        gen_optimal_pairs(
            mertens_fuse, raw_dir, gt_dir, remaining, out_dir, args.log_name
        )

# if __name__ == "__main__":
#     IS_SCRIPT = True
#     parser = argparse.ArgumentParser(
#         description="generates synthetic HDR image composed of LDR images of the same scene"
#     )
#     parser.add_argument(
#         "-m",
#         "--method",
#         help="Fusion method to use",
#         choices=["debevec", "mertens", "robertson", "all"],
#     )
#     parser.add_argument("-o", "--out-dir", default=str(DATA_DIR / "merged"))
#     parser.add_argument("--input-dir", default=str(DATA_DIR / "dngs"))
#     parser.add_argument("-n", "--dry-run", action="store_true")

#     args = parser.parse_args()
#     fuse_method = FuseMethod[str(args.method).capitalize()]
#     out_dir = (Path(".") / args.out_dir).resolve()
#     processed_dir = out_dir.parent / "processed"
#     in_dir = Path(args.input_dir).resolve()
#     dry_run = args.dry_run

#     print(
#         f"""
# Using the following settings:
#  Merged Image Output Directory:     {out_dir}
#  Exposure Adjusted Image Directory: {processed_dir}
#  Merge Method:                      {str(fuse_method)}
#  Dry Run? {"Yes" if dry_run else "No"}
#         """
#     )

#     if not dry_run:
#         main(in_dir, out_dir, fuse_method)
