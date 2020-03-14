from pathlib import Path

import torch
from skimage import io
from torch.utils.data import Dataset

import dhdrnet.image_loader as il


class HDRDataset(Dataset):
    """HDR image dataset"""

    def __init__(self, root_dir: Path, exp_dir: Path):
        self.root_dir = root_dir
        self.exp_dir = exp_dir

        self.gt_paths = list(self.root_dir.iterdir())
        self.exp_paths = list(self.exp_dir.iterdir())

    def __len__(self):
        return len(self.gt_paths)

    # a single sample from the training set will return a dict:
    # { "exposures": [exp], "ground_truth": image, "name": filename}
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        gt_path = self.gt_paths[idx]
        exposures = [
            io.imread(str(ip)) for ip in il.get_exposures(self.exp_dir, gt_path)
        ]
        return {
            "exposures": exposures,
            "ground_truth": io.imread(str(gt_path)),
            "name": gt_path.stem,
        }
