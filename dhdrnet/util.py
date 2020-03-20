import random
from collections import defaultdict
from collections.abc import Iterable as It
from pathlib import Path

# End file hierarchy should look like:
# DATA_DIR /
# {train, test, val} /
# {dngs, merged, processed}
from pprint import pprint
from subprocess import check_output
from typing import Set

import numpy as np
import torch


def get_project_root() -> Path:
    git_root = (
        check_output(["git", "rev-parse", "--show-toplevel"]).decode("utf-8").strip()
    )
    return Path(git_root).absolute()


ROOT_DIR: Path = get_project_root()
DATA_DIR: Path = ROOT_DIR / "data"


def get_train_test(data_dir: Path) -> torch.utils.data.Dataset:
    pass


def create_train_test_split(data_dir: Path, train_split=0.9, val_split=0.2):
    files: Set = set((data_dir / "dngs").iterdir())

    train_size = int(train_split * len(files))
    train: Set = set(random.sample(files, k=train_size))
    test = files - train

    val_size = int(val_split * len(train))
    val = set(random.sample(train, k=val_size))
    train = train - val

    print(len(files))
    print(len(train))
    print(len(test))
    print(len(val))
    assert len(files) == len(train) + len(test) + len(val)

    for name, split in {"train": train, "test": test, "val": val}.items():
        with open(get_project_root() / f"{name}.txt", "w") as f:
            for fp in split:
                print(fp.stem, file=f)


def split_dataset(
    root_dir: Path = ROOT_DIR, data_dir: Path = DATA_DIR, dry_run: bool = False
):
    ds_splits = ["train", "test", "val"]
    data_subdirs = ["dngs", "merged", "processed"]
    source_dest_map = defaultdict(list)
    # Making sure that the dataset split directories exist
    for ds_dir in [(data_dir / f"{split}") for split in ds_splits]:
        for subdir, subname in [((ds_dir / subdir), subdir) for subdir in data_subdirs]:
            subdir.mkdir(parents=True, exist_ok=True)
            source_dest_map[data_dir / subname].append(subdir)
    pprint(source_dest_map)

    # getting each file that contains listings
    # and reading into dict keyed by train,test,val
    ds_files = {split: [] for split in ds_splits}
    for f, split in [(root_dir / f"{f}.txt", f) for f in ds_splits]:
        if f.is_file():
            ds_files[split] = set(f.read_text().splitlines())

    for source, dests in source_dest_map.items():
        source_files = list(source.iterdir())
        for split, names in ds_files.items():
            to_move = flatten(
                [[sf for name in names if name in sf.name] for sf in source_files]
            )
            print(f"{source}: {split}")
            # print(len(to_move))
            # print("\n")
            # pprint(to_move[:10])
            for orig in to_move:
                dest = data_dir / split / source.name
                print(f"{orig} --> {dest / orig.name}")
                # return
                try:
                    (dest / orig.name).symlink_to(orig)
                except FileExistsError as fee:
                    print(fee)


def flatten(items, ignore_types=(str, bytes)):
    for x in items:
        if isinstance(x, It) and not isinstance(x, ignore_types):
            yield from flatten(x)
        else:
            yield x
