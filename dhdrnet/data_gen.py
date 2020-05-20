import argparse
from functools import partial
from pathlib import Path
from typing import Callable, Collection, List

import dhdrnet.image_loader as il
from dhdrnet.cv_fuse import FuseMethod
from dhdrnet.util import get_project_root

IS_SCRIPT = False
DATA_DIR = get_project_root() / "data"


def main(in_dir: Path, out_dir: Path, fuse_method: FuseMethod):
    fuse_fun = partial(il.cv_fuse, method=fuse_method, out_dir=out_dir)
    out_dir.mkdir(exist_ok=True)
    print(
        gen(
            image_paths=in_dir.iterdir(),
            processed_dir=out_dir.parent / "processed",
            fuse_fun=fuse_fun,
        )
    )


def gen(
    image_paths: Collection[Path],
    processed_dir: Path,
    fuse_fun: Callable[[Collection[Path]], Collection[Path]],
) -> Collection[Path]:
    image_paths = list(image_paths)

    print(f"Generating multi exposure files in {processed_dir} (Skipping existing)")

    # processed_dir.mkdir(exist_ok=True)
    # multi_exposure_paths = list(processed_dir.iterdir())
    # if len(multi_exposure_paths) == 0:
    #     multi_exposure_paths = il.parallel_write_multi_exp(image_paths, processed_dir)

    # grouped_ldr_paths = list(il.multi_exp_paths(image_paths, processed_dir))
    # print("Generating merged files")
    # return list(il.parallel_cv_fused(fuse_fun, grouped_ldr_paths))


def gen_all_fuse(
    fuse_funcs, raw_images: List[Path], gt_images: List[Path], out_dir,
):
    all_ev_steps = range(5, 10)
    all_combinations = product(fuse_funcs, all_ev_steps, zip(raw_images, gt_images))
    raw: Path
    gt: Path
    for ff, ev_steps, (raw, gt) in all_combinations:
        exposures = get_multi_exposures(raw, ev_steps)
        exp_choices = exposures[np.random.choice(range(len(exposures)), 2)]
        fused: Image = ff(exp_choices)
        # save the results to disk
        fused.save(out_dir / f"{raw.stem}.png")


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
