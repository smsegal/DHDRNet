from pathlib import Path

import pandas as pd
import torch
from more_itertools.more import one
from more_itertools.recipes import flatten
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dhdrnet.gen_pairs import GenAllPairs
from dhdrnet.util import centercrop, min_shape


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
        df = df.set_index("name")

        # ev 0
        baseline_df = df[(df["ev1"] == 0) | (df["ev2"] == 0)].copy()
        baseline_df["ev"] = baseline_df[["ev1", "ev2"]].apply(
            lambda evs: [e for e in evs if e != 0][0], axis=1
        )
        baseline_df = baseline_df.drop(columns=["ev1", "ev2"])

        by_ev = baseline_df.pivot_table(
            index="name", columns=["metric", "ev"], values="score"
        )
        by_ev = by_ev.loc[by_ev.index.intersection(names)]

        self.evs = sorted(baseline_df["ev"].unique())
        self.ev_indices = {ev: i for (i, ev) in enumerate(self.evs)}

        self.opt_choices = by_ev[metric].idxmin(axis=1)
        self.metric = metric
        self.data = by_ev
        self.names = pd.Series(self.data.index)

        self.generator = GenAllPairs(
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
        ground_truth = self.generator.get_ground_truth(name)
        return mid_exp, ground_truth, name

    @staticmethod
    def collate_fn(batch):
        mid_exposures, ground_truths, names = zip(*batch)
        minw, minh, _minc = min_shape(mid_exposures)
        mid_exp_t = torch.stack(
            [torch.tensor(im) for im in centercrop(mid_exposures, [minw, minh])]
        )
        ground_truth_t = torch.stack(
            [torch.tensor(im) for im in centercrop(ground_truths, [minw, minh])]
        )
        return mid_exp_t, ground_truth_t, names


class HistogramDataset(LUTDataset):
    """Dataset class for feeding histograms to a
    fully connected network as a baseline"""

    def __init__(self, bins: int = 100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bins = bins

    def __getitem__(self, index: int):
        mid_ev_image, label_idx, stats = super().__getitem__(index)
        return torch.histc(mid_ev_image, bins=self.bins), label_idx, stats


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
