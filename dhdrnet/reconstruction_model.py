from typing import List, Union

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models

from dhdrnet.dataset import RCDataset
from dhdrnet.model import DHDRNet
from dhdrnet.util import ROOT_DIR

figdir = ROOT_DIR / "figures"


class RCNet(DHDRNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.Dataset = RCDataset
        self.feature_extractor = models.mobilenet_v2(
            pretrained=False, num_classes=self.num_classes
        )
        self.feature_extractor.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(
                self.feature_extractor.last_channel,
                self.feature_extractor.last_channel // 2,
            ),
            nn.BatchNorm1d(self.feature_extractor.last_channel // 2),
            nn.Linear(self.feature_extractor.last_channel // 2, self.num_classes),
        )
        self.criterion = nn.MSELoss()

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
