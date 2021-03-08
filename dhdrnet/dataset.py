from pathlib import Path
from typing import Collection, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from more_itertools.more import one
from more_itertools.recipes import flatten
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dhdrnet.data_generator import DataGenerator
from dhdrnet.util import evpairs_to_classes


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

        by_ev = pd.pivot_table(
            data=df, index="name", columns=["metric", "ev"], values="score"
        )
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


class CachingDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        image_names: Optional[Iterable[str]] = None,
        exposure_values: Iterable[float] = [-4, -2, 0, 2, 4],
        metric="psnr",
        transform=transforms.ToTensor(),
    ):
        self.data_dir = data_dir
        self.exposure_values = exposure_values
        self.metric = metric
        self.transform = transform
        self.evpairs_to_class = evpairs_to_classes(exposure_values)

        if image_names is not None:
            potential_image_paths = (
                (data_dir / f"dngs/{name}.dng") for name in image_names
            )
            self.image_paths = [ip for ip in potential_image_paths if ip.exists()]
        else:
            self.image_paths = list((data_dir / "dngs").iterdir())

        self._len = len(self.image_paths)

        self.data_generator = DataGenerator(
            raw_path=data_dir / "dngs",
            out_path=data_dir,
            compute_scores=False,
            multithreaded=True,
        )

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index):
        image_name = self.image_paths[index].stem
        exposure_images = torch.stack(
            [
                self.transform(img)
                for img in (
                    self.data_generator.get_exposures(
                        image_name, exposures=self.exposure_values
                    )
                )
            ],
            dim=0,
        )
        score, best_ev_pair = self.data_generator.get_best_evs(
            image_name, self.exposure_values, metric=self.metric
        )
        evpair_class = self.evpairs_to_class[best_ev_pair]
        return exposure_images, evpair_class, score


class RCDataset(LUTDataset):
    def __getitem__(self, index):
        mid_exp, label, name = super().__getitem__(index)
        gt_image = self.generator.get_ground_truth(name)
        gt_image = Image.fromarray(gt_image[..., [2, 1, 0]])
        gt_image = self.transform(gt_image)
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
