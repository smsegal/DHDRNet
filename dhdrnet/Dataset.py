from pathlib import Path
from typing import Collection

import colour_hdri as ch
import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset

from dhdrnet.image_loader import read_hdr


class HDRDataset(Dataset):
    """HDR image dataset"""

    def __init__(
        self,
        root_dir: Path,
        raw_dir: Path,
        exposure_levels: Collection = np.linspace(-4, 4, 5),
    ):
        self.root_dir = root_dir
        self.raw_dir = raw_dir

        self.gt_paths = list(self.root_dir.iterdir())
        self.exposure_levels = exposure_levels

    def __len__(self):
        return len(self.gt_paths)

    # a single sample from the training set will return a dict:
    # { "exposures": [exp], "ground_truth": image, "name": filename}
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        gt_path = self.gt_paths[idx]
        raw_image = read_hdr(self.raw_dir / f"{gt_path.stem}.dng")
        gt_image = io.imread(gt_path)
        exposures = [
            ch.adjust_exposure(raw_image, exp_level)
            for exp_level in self.exposure_levels
        ]
        return {
            "exposures": exposures,
            "ground_truth": gt_image,
            "name": gt_path.stem,
        }
