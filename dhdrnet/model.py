from math import ceil
from typing import Dict, List, Union

import pandas as pd
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import models
from torchvision.transforms import (
    Compose,
    TenCrop,
    Lambda,
    ToTensor,
    RandomCrop,
    RandomChoice,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)

from dhdrnet.Dataset import LUTDataset
from dhdrnet.util import DATA_DIR, ROOT_DIR


class DHDRNet(LightningModule):  # pylint: disable=too-many-ancestors
    def __init__(self, learning_rate=1e-3, batch_size=8, use_tencrop=True, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.use_tencrop = use_tencrop
        self.save_hyperparameters()

    def prepare_data(self):
        if self.use_tencrop:
            transform = Compose(
                [
                    TenCrop(350),
                    Lambda(
                        lambda crops: torch.stack([ToTensor()(crop) for crop in crops])
                    ),
                ]
            )
        else:
            transform = Compose(
                [
                    RandomCrop(350),
                    RandomChoice(
                        [RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.5)]
                    ),
                    ToTensor(),
                ]
            )

        data_df = pd.read_csv(
            ROOT_DIR / "precomputed_data" / "store_finer_2020-07-06.csv"
        )
        trainval_data = LUTDataset(
            df=data_df,
            exposure_path=DATA_DIR / "correct_exposures" / "exposures",
            raw_dir=DATA_DIR / "dngs",
            name_list=ROOT_DIR / "train.txt",
            transform=transform,
        )

        test_data = LUTDataset(
            df=data_df,
            exposure_path=DATA_DIR / "correct_exposures" / "exposures",
            raw_dir=DATA_DIR / "dngs",
            name_list=ROOT_DIR / "test.txt",
            transform=transform,
        )
        train_val_ratio = 0.8
        train_len = ceil(train_val_ratio * len(trainval_data))
        val_len = len(trainval_data) - train_len
        train_data, val_data = random_split(trainval_data, lengths=(train_len, val_len))
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_data, batch_size=self.batch_size, pin_memory=True, num_workers=8
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_data, batch_size=self.batch_size, pin_memory=True, num_workers=8
        )

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

    def common_step(self, batch):
        mid_exposure, label, stats = batch
        if self.use_tencrop:
            bs, ncrops, c, h, w = mid_exposure.size()
            outputs_crops = self(mid_exposure.view(-1, c, h, w))
            outputs = outputs_crops.view(bs, ncrops, -1).mean(1)
        else:
            outputs = self(mid_exposure)

        loss = self.criterion(outputs, label)
        return loss, stats

    def training_step(self, batch, batch_idx) -> Dict[str, Union[Dict, Tensor]]:
        loss, stats = self.common_step(batch)
        logs = {"train_loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx) -> Dict[str, Union[Dict, Tensor]]:
        loss, stats = self.common_step(batch)
        logs = {"val_loss": loss}
        return {"val_loss": loss, "log": logs}

    def validation_epoch_end(
        self, outputs: List[Dict[str, Tensor]],
    ) -> Dict[str, Union[Dict, Tensor]]:
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_epoch_loss": avg_loss}
        logs = {"val_loss": avg_loss, "log": tensorboard_logs}
        return logs

    def test_step(self, batch, batch_idx) -> Dict[str, Union[Dict, Tensor]]:
        loss, stats = self.common_step(batch)
        logs = {"test_loss": loss}
        return {"test_loss": loss, "log": logs}

    def test_epoch_end(self, outputs) -> Dict[str, Union[Dict, Tensor]]:
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        tensorboard_logs = {"test_loss": avg_loss}
        return {"test_loss": avg_loss, "log": tensorboard_logs}


class DHDRMobileNet(DHDRNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        num_classes = 36
        self.feature_extractor = models.mobilenet_v2(pretrained=False)
        # self.feature_extractor.eval()
        self.feature_extractor.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(self.feature_extractor.last_channel, num_classes)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # features = self.feature_extractor.features(x)
        # print(f"{features.shape=}")
        x = self.feature_extractor(x)
        return x


class DHDRSqueezeNet(DHDRNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = 36
        self.inner_model = models.squeezenet1_1(
            pretrained=False, num_classes=num_classes
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.inner_model(x)
        return x
