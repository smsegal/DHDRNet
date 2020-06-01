from pathlib import Path

import numpy as np
import torch
from PIL import Image
from more_itertools import flatten
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


import pandas as pd
from dhdrnet.vis_util import get_metric_cat_groups


class LUTDataset(Dataset):
    def __init__(self, fname: Path, ev, img_dir, transform):
        self.dir = dir
        self.img_dir = img_dir
        self.transform = transform

        all_data = pd.read_csv(fname).set_index("name")
        ev_categories = {
            ev_max: np.linspace(-ev_max, ev_max, 5) for ev_max in range(ev, ev + 1)
        }
        metric_categories = list(
            flatten(get_metric_cat_groups(all_data, ev_categories).values())
        )

        self.names = all_data.index
        self.mse_data = np.array(
            all_data[metric_categories]
                .loc[:, lambda df: [c for c in df.columns if c.startswith("mse")]]
                .rename(columns=lambda c: c.split("_")[-1])
        )

        self.ssim_data = np.array(
            all_data[metric_categories]
                .loc[:, lambda df: [c for c in df.columns if c.startswith("ssim")]]
                .rename(columns=lambda c: c.split("_")[-1])
        )

        self.ms_ssim_data = np.array(
            all_data[metric_categories]
                .loc[:, lambda df: [c for c in df.columns if c.startswith("ms_ssim")]]
                .rename(columns=lambda c: c.split("_")[-1])
        )

    def __len__(self):
        return len(self.mse_data)

    def __getitem__(self, idx):
        mid_exp = Image.open(self.img_dir / f"{self.names[idx]}.png")
        mid_exp = self.transform(mid_exp)

        mse_data = torch.as_tensor(self.mse_data[idx])
        return (
            mid_exp,
            mse_data,
            self.ssim_data[idx],
            self.ms_ssim_data[idx],
        )

    @staticmethod
    def collate_fn(batch):
        mid_exposure = torch.stack([b[0] for b in batch])
        mse_data = torch.stack([b[1] for b in batch])
        ssim_data = torch.stack([b[2] for b in batch])
        ms_ssim_data = torch.stack([b[3] for b in batch])
        return mid_exposure, mse_data, ssim_data, ms_ssim_data
