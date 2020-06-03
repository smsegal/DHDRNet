from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class HDRDataset(Dataset):
    """HDR image dataset"""

    def __init__(
        self, gt_dir: Path, exp_dir: Path, transform=None,
    ):
        self.gt_dir = gt_dir
        self.exp_dir = exp_dir
        self.gt_paths = list(self.gt_dir.iterdir())
        self.transform = transform

    def __len__(self):
        return len(self.gt_paths)

    # a single sample from the training set will return a dict:
    # { "exposures": [exp], "ground_truth": image, "name": filename}
    # input to our network is middle-exposure image + randomly exposed image
    # ground_truth is the _gold standard_ merged image
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        gt_path = self.gt_paths[idx]
        gt_image = Image.open(gt_path)

        img_name = gt_path.stem
        exposure_paths = sorted(
            self.exp_dir.glob(f"{img_name}*"), key=lambda p: int(p.stem.split(".")[-1])
        )

        # mid exposure is only one that matches with a 0 at the end
        mid_exposure_path = [ep for ep in exposure_paths if ep.stem.endswith("0")][0]
        mid_exposure = Image.open(mid_exposure_path)

        if self.transform:
            gt_image = self.transform(gt_image)
            mid_exposure = self.transform(mid_exposure)

        return {
            "exposure_paths": exposure_paths,
            "mid_exposure": mid_exposure,
            "ground_truth": gt_image,
        }

    @staticmethod
    def collate_fn(batch):
        exposure_paths = [b["exposure_paths"] for b in batch]
        mid_exposure = torch.stack([b["mid_exposure"] for b in batch])
        ground_truth = torch.stack([b["ground_truth"] for b in batch])
        return exposure_paths, mid_exposure, ground_truth


import numpy as np


class LUTDataset(Dataset):
    def __init__(
        self,
        choice_path: Path,
        stats_path: Path,
        img_dir,
        ev_range,
        transform,
        metric="mse",
    ):
        self.img_dir = img_dir
        self.transform = transform
        self.ev_cat = str(ev_range)
        self.opt_choices = (
            pd.read_csv(choice_path)
            .set_index("name")
            .sort_index()
            .groupby("metric")
            .get_group(metric)
        )
        self.stats = pd.read_csv(stats_path).set_index("name")
        self.names = pd.Series(self.opt_choices.index)

        self._length = len(self)  # cache it
        self.labelindices = _ev_to_index(ev_range)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        if index >= self._length:
            raise IndexError()

        img_name = self.names[index]
        ev = self.opt_choices.loc[img_name, self.ev_cat]
        label = self.labelindices[ev]
        mid_exp = self.transform(Image.open(self.img_dir / f"{img_name}.png"))
        return mid_exp, label


def _ev_to_index(ev_range):
    labels = {}
    for i, ev in enumerate(np.linspace(-ev_range, ev_range, 5)):
        if ev == 0:
            continue
        if ev > 0:
            v = i - 1
        else:
            v = i
        labels[ev] = v
    return labels
