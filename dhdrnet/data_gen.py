import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import islice, product
from pathlib import Path
from typing import Callable, Collection, List

import colour as co
import numpy as np

import dhdrnet.image_loader as il
from dhdrnet.cv_fuse import FuseMethod
from dhdrnet.image_loader import gen_multi_exposures
from dhdrnet.reconstruction import reconstruction_stats
from dhdrnet.util import get_project_root

IS_SCRIPT = False
DATA_DIR = get_project_root() / "data"


def main(in_dir: Path, out_dir: Path, fuse_method: FuseMethod):
    fuse_fun = partial(il.cv_fuse, method=fuse_method, out_dir=out_dir)
    out_dir.mkdir(exist_ok=True)
    processed_paths = gen(
        image_paths=in_dir.iterdir(),
        processed_dir=out_dir.parent / "processed",
        fuse_fun=fuse_fun,
    )
    print(processed_paths)


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


def gen_all_fuse_options(
    fuse_funcs, raw_images: List[Path], gt_images: List[Path], out_dir,
):
    all_ev_steps = [5]  # range(5, 10)
    sorted_raw = sorted(raw_images, key=lambda p: p.stem)
    sorted_gt = sorted(gt_images, key=lambda p: p.stem)
    ev_max = [6]
    all_combinations = list(
        product(fuse_funcs, ev_max, all_ev_steps, zip(sorted_raw, sorted_gt))
    )
    print(len(all_combinations))
    with ThreadPoolExecutor(max_workers=11) as executor:
        results = executor.map(lambda a: fuse_exposures(out_dir, *a), all_combinations)

    opt_log = {k: v for k, v in results}

    return opt_log


def fuse_exposures(out_dir, fuse_func, ev_max, ev_steps, raw_gt):
    raw, gt = raw_gt
    ev_range = list(np.linspace(-ev_max, ev_max, ev_steps))
    exposures = gen_multi_exposures(raw, *ev_range)
    mid_exp = exposures.pop(len(exposures) // 2)
    mid_exp_range = ev_range.pop(len(ev_range) // 2)

    out_dir.mkdir(parents=True, exist_ok=True)
    logs = []
    for ev, exp in zip(ev_range, exposures):
        fused = fuse_func([mid_exp, exp])

        # save the results to disk
        co.write_image(fused, out_dir / f"{raw.stem}_ev{ev}.png")

        # compute stats
        mse, ssim, ms_ssim = reconstruction_stats(fused, co.read_image(gt))
        logs.append({ev: {"mse": mse, "ssim": ssim, "ms_ssim": ms_ssim,}})

    return (raw.stem, logs)


def write_log(dest, opt_log):
    with open(f"{dest}.json", "w") as f:
        json.dump(opt_log, f)


if __name__ == "__main__":
    IS_SCRIPT = True
    parser = argparse.ArgumentParser(
        description="generates synthetic HDR image composed of LDR images of the same scene"
    )
    parser.add_argument(
        "-m",
        "--method",
        help="Fusion method to use",
        choices=["debevec", "mertens", "robertson", "all"],
    )
    parser.add_argument("-o", "--out-dir", default=str(DATA_DIR / "merged"))
    parser.add_argument("--input-dir", default=str(DATA_DIR / "dngs"))
    parser.add_argument("-n", "--dry-run", action="store_true")

    args = parser.parse_args()
    fuse_method = FuseMethod[str(args.method).capitalize()]
    out_dir = (Path(".") / args.out_dir).resolve()
    processed_dir = out_dir.parent / "processed"
    in_dir = Path(args.input_dir).resolve()
    dry_run = args.dry_run

    print(
        f"""
Using the following settings:
 Merged Image Output Directory:     {out_dir}
 Exposure Adjusted Image Directory: {processed_dir}
 Merge Method:                      {str(fuse_method)}
 Dry Run? {"Yes" if dry_run else "No"}
        """
    )

    if not dry_run:
        main(in_dir, out_dir, fuse_method)
