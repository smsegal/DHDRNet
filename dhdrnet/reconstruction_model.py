from typing import List, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models
import torchvision.utils as tu
from more_itertools import collapse

from dhdrnet.Dataset import RCDataset
from dhdrnet.model import DHDRNet
from dhdrnet.gen_pairs import GenAllPairs
from dhdrnet.util import DATA_DIR, ROOT_DIR
from PIL import Image
import numpy as np
import sys

from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
)

figdir = ROOT_DIR / "figures"


class RCNet(DHDRNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.Dataset = RCDataset
        self.feature_extractor = models.mobilenet_v2(
            pretrained=False, num_classes=self.num_classes
        )
        # self.feature_extractor.classifier = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(
        #         self.feature_extractor.last_channel,
        #         self.feature_extractor.last_channel // 2,
        #     ),
        #     nn.BatchNorm1d(self.feature_extractor.last_channel // 2),
        #     nn.Linear(self.feature_extractor.last_channel // 2, self.num_classes),
        # )
        self.criterion = nn.MSELoss()

        # duplicated code grosssss
        exp_min = -3
        exp_max = 6
        exp_step = 0.25
        self.exposures: np.ndarray = np.linspace(
            exp_min, exp_max, int((exp_max - exp_min) / exp_step + 1)
        )

    def common_step(self, batch):
        mid_exposures, ground_truth_images, names, _labels = batch
        predicted_ev_idx = torch.argmax(self(mid_exposures), dim=1)
        predicted_ev = self.exposures[predicted_ev_idx.cpu().numpy()]
        predicted_fused = torch.stack(
            [
                self.transform(
                    Image.fromarray(self.generator.get_reconstruction(name, 0.0, pev))
                )
                for pev, name in zip(predicted_ev, names)
            ]
        )
        predicted_fused = predicted_fused.to(device="cuda:0")
        predicted_fused.requires_grad_(True)
        # print(f"{all_fused_images.shape=}")
        # print(f"{predicted_ev_idx.shape=}")
        # print(f"{predicted_fused.shape=}")
        # print(f"{ground_truth_images.shape=}")

        # sys.exit()

        # objective = sum(
        #     [
        #         sum([sum(fuse(x) + fuse(e) + 2 * fuse(gt)) for e in EV])
        #         for x in images
        #     ]
        # )
        # I want to pick the best ev in EV
        # but how to translate from PDF to objective function?

        loss = F.mse_loss(predicted_fused, ground_truth_images)
        return loss

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=8,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=8,
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return x
