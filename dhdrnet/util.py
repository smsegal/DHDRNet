import random
from pathlib import Path
from subprocess import check_output
import torch
import numpy as np
from typing import Set


def get_project_root() -> Path:
    git_root = (
        check_output(["git", "rev-parse", "--show-toplevel"]).decode("utf-8").strip()
    )
    return Path(git_root).absolute()


DATA_DIR = get_project_root() / "data"


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
        with open(f"{name}.txt", "w") as f:
            for fp in split:
                print(fp.name, file=f)
