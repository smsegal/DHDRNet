from pathlib import Path

import numpy as np
import pandas as pd
import torch
from deprecated import deprecated
from more_itertools.more import one
from more_itertools.recipes import flatten
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dhdrnet.data_generator import DataGenerator


class LUTDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        exposure_path: Path,
        raw_dir: Path,
        name_list: Path,
        transform=transforms.ToTensor(),
        metric="mse",
    ):
        self.exposure_path = exposure_path
        self.transform = transform

        names = flatten(pd.read_csv(name_list).to_numpy())

        by_ev = df.pivot_table(index="name", columns=["metric", "ev"], values="score")
        by_ev = by_ev.loc[by_ev.index.intersection(names)]

        exp_min = -3
        exp_max = 6
        exp_step = 0.25
        evs: np.ndarray = np.linspace(
            exp_min, exp_max, int((exp_max - exp_min) / exp_step + 1)
        )
        self.evs = np.array([*evs[evs < 0], *evs[evs > 0]])

        self.ev_indices = {ev: i for (i, ev) in enumerate(self.evs)}

        self.opt_choices = by_ev[metric].idxmin(axis=1)
        self.metric = metric
        self.data = by_ev
        self.names = pd.Series(self.data.index)

        self.generator = DataGenerator(
            raw_path=raw_dir,
            out_path=exposure_path.parent,
            store_path=None,
            compute_scores=False,
        )

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError()

        label_idx = self.ev_indices[self.opt_choices[index]]
        img_name = self.names[index]
        mid_exp_bgr = one(self.generator.get_exposures(img_name, [0.0]))
        mid_exp_rgb = Image.fromarray(mid_exp_bgr[:, :, [2, 1, 0]])
        mid_exp = self.transform(mid_exp_rgb)
        return mid_exp, label_idx, img_name


class RCDataset(LUTDataset):
    def __getitem__(self, index):
        mid_exp, label, name = super().__getitem__(index)
        gt_image = self.generator.get_ground_truth(name)
        gt_image = Image.fromarray(gt_image[..., [2, 1, 0]])
        gt_image = self.transform(gt_image)
        # fused_images  (
        #     self.generator.get_reconstruction(name, 0.0, ev) for ev in self.evs
        # )
        # fused_images = (Image.fromarray(fi) for fi in fused_images)
        # fused_images = torch.stack([self.transform(fi) for fi in fused_images])
        return mid_exp, gt_image, name, label


class HistogramDataset(LUTDataset):
    """Dataset class for feeding histograms to a
    fully connected network as a baseline"""

    def __init__(self, bins: int = 100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bins = bins

    def __getitem__(self, index: int):
        mid_ev_image, label_idx, name = super().__getitem__(index)
        return torch.histc(mid_ev_image, bins=self.bins), label_idx, name


@deprecated
class HDRDataset(Dataset):
    """HDR image dataset"""

    def __init__(
        self,
        gt_dir: Path,
        exp_dir: Path,
        transform=None,
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
