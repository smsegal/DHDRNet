from pathlib import Path
from typing import Collection

import colour_hdri as ch
import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset
from PIL import Image

from dhdrnet.image_loader import read_hdr


class HDRDataset(Dataset):
    """HDR image dataset"""

    def __init__(
        self,
        gt_dir: Path,
        raw_dir: Path,
        exposure_levels: Collection = np.linspace(-4, 4, 5),
        transforms=None,
    ):
        self.gt_dir = gt_dir
        self.raw_dir = raw_dir

        self.gt_paths = list(self.gt_dir.iterdir())
        self.exposure_levels = exposure_levels
        self.transforms = transforms

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
        raw_image = Image.open(
            self.raw_dir / f"{gt_path.stem}.dng"
        )  # read_hdr(self.raw_dir / f"{gt_path.stem}.dng")
        gt_image = Image.open(gt_path)

        if self.transforms:
            raw_image = self.transforms(raw_image)
            gt_image = self.transforms(gt_image)

        exposures = [
            ch.adjust_exposure(raw_image, exp_level)
            for exp_level in self.exposure_levels
        ]
        return {
            "exposures": exposures,
            "ground_truth": gt_image,
        }
