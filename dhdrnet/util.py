import os
import random
from collections import defaultdict
from collections.abc import Iterable as It
from itertools import product
from math import ceil
from pathlib import Path
from subprocess import CalledProcessError, check_output
from typing import DefaultDict, Iterator, List, Mapping, Set, TypeVar

import colour_hdri as ch
import numpy as np
from PIL import Image

T = TypeVar("T")


class Indexable(Mapping[int, T]):
    def __init__(self, *args, **kwargs):
        super(self, *args, **kwargs)


root_env_keys = ("IDE_PROJECT_ROOTS", "DHDR_ROOT")


def get_project_root() -> Path:
    for key in root_env_keys:
        if key in os.environ:
            return Path(os.environ[key])
    try:
        git_root = (
            check_output(["git", "rev-parse", "--show-toplevel"])
                .decode("utf-8")
                .strip()
        )
        return Path(git_root).absolute()
    except CalledProcessError:
        # not in a git repo
        # out of options here, might want to just fail at this point
        return Path.cwd().parent.parent


ROOT_DIR: Path = get_project_root()
if "DHDR_DATA_DIR" in os.environ:
    DATA_DIR = Path(str(os.environ["DHDR_DATA_DIR"]))
else:
    DATA_DIR = ROOT_DIR / "data"
MODEL_DIR: Path = ROOT_DIR / "models"


def create_train_test_split(data_dir: Path, train_split=0.9, dry_run=False):
    files: Set = set((data_dir / "dngs").iterdir())

    train_size = round(train_split * len(files))
    train: Set = set(random.sample(files, k=train_size))

    test = files - train

    if not dry_run:
        for name, split in {"train": train, "test": test}.items():
            split_file = ROOT_DIR / f"{name}.txt"
            split_file.write_text("\n".join([fname.stem for fname in split]))

    return files, train, test


def split_data(data_dir: Path, root_dir: Path) -> DefaultDict[Path, List[Path]]:
    """
    End file hierarchy should look like:
    DATA_DIR /
    {train, test} /
    {dngs, merged, processed}
    """
    splits = ["train", "test"]
    data_splits = [
        set(f.read_text().splitlines())
        for f in [root_dir / f"{split}.txt" for split in splits]
    ]

    source_dest_map: DefaultDict[Path, List[Path]] = defaultdict(list)
    for source_dir in [data_dir / st for st in ["dngs", "merged", "processed"]]:
        for split_list, split_name in zip(data_splits, splits):
            target_dir = data_dir / split_name / source_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            source_dest_map[source_dir].append(target_dir)
            for name in split_list:
                source_files: Iterator[Path] = source_dir.glob(f"{name}.*")
                for source_file in source_files:
                    target = target_dir / source_file.name
                    target.symlink_to(source_file)
    return source_dest_map


def flatten(items, ignore_types=(str, bytes)):
    for x in items:
        if isinstance(x, It) and not isinstance(x, ignore_types):
            yield from flatten(x)
        else:
            yield x


def get_img_name(path: Path) -> str:
    """
    path: needs to be of the form $DATA_DIR/../../{IMG_NAME}.{EXP_LEVEL}.{ext}
    """
    return path.stem.split(".")[0]


def get_mid_exp(path: Path):
    name = get_img_name(path)
    possible_exps = path.parent.glob(f"{name}.0.*")
    return next(possible_exps)


def min_centercrop(images: List):
    shapes = (im.shape for im in images)
    return centercrop(images, map(min, zip(*shapes)))


def centercrop(images, shape):
    minw, minh = shape
    all_cropped = []
    for im in images:
        w, h = im.shape[:-1]
        cropped = im[
                  ceil((w - minw) / 2): ceil(w - ((w - minw) / 2)),
                  ceil((h - minh) / 2): ceil(h - ((h - minh) / 2)),
                  ]
        all_cropped.append(cropped)
    return all_cropped


# compute all fusion steps with different EVs -- grid search essentially
def get_multi_exposures(raw, ev_steps):
    exposures = []
    img = Image.open(raw)
    for exposure in ev_steps:
        ch.adjust_exposure(img.data, exposure)
    return exposures


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


def compute_metadata(ev):
    f = 1
    iso = 100
    exposure_time = (f ** 2) / (2 ** ev)
    metadata = ch.Metadata(f_number=f, iso=iso, exposure_time=exposure_time)
    return metadata


def norm_zero_one(a):
    min_a = np.min(a)
    return (a - min_a) / (np.max(a) - min_a)
