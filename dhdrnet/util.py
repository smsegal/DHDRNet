import random
from pathlib import Path
from subprocess import check_output
import torch
import numpy as np


def get_project_root() -> Path:
    git_root = (
        check_output(["git", "rev-parse", "--show-toplevel"]).decode("utf-8").strip()
    )
    return Path(git_root).absolute()


DATA_DIR = get_project_root() / "data"


def get_train_test(data_dir: Path) -> torch.utils.data.Dataset:
    pass


def create_train_test_split(data_dir: Path, train_split=0.9, val_split=0.2):
    files = set((data_dir / "dngs").iterdir())
    train = random.sample(files, k=train_split * len(files))
    test = files - train
    val = random.sample(train, k=val_split * len(train))
    for name, split in {"train": train, "test": test, "val": val}.items():
        with open(f"{name}.txt", "w") as f:
            for fp in split:
                print(str(fp), file=f)
