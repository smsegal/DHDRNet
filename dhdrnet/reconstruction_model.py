from typing import List, Union

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models

from dhdrnet.Dataset import RCDataset, RCDataset2
from dhdrnet.model import DHDRNet


class RCNet(DHDRNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.Dataset = RCDataset
        self.feature_extractor = models.mobilenet_v2(pretrained=False)
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

    def common_step(self, batch):
        mid_exposures, y_true, names = batch
        y_pred = F.softmax(self(mid_exposures), dim=0)

        loss = F.mse_loss(y_pred, y_true)
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


class RCNet2(RCNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Dataset = RCDataset2

    def common_step(self, batch):
        mid_exposures, mse, names = batch
        y_pred = self(mid_exposures)

        loss = F.mse_loss(y_pred, mse)
        return loss
