from typing import List, Union

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models

from dhdrnet.Dataset import RCDataset
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
            nn.Softmax(dim=0),
        )
        self.criterion = nn.MSELoss()

    def common_step(self, batch):
        mid_exposures, y_true, names = batch
        y_pred = self(mid_exposures)

        loss = F.mse_loss(y_pred, y_true)
        # print(f"{mid_exposures.dtype=}")
        # print(f"computed {loss=} with \n\n{y_pred=} \n\n{y_true=}")
        return loss

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
            # collate_fn=RCDataset.collate_fn,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=8,
            # collate_fn=RCDataset.collate_fn,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=8,
            # collate_fn=RCDataset.collate_fn,
        )

    def forward(self, x):
        # features = self.feature_extractor.features(x)
        # print(f"{features.shape=}")
        x = self.feature_extractor(x)
        return x
