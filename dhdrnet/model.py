from math import ceil
from typing import List, Union

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
from torchvision import models
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)

from dhdrnet.Dataset import LUTDataset
from dhdrnet.squeezenet import squeezenet1_1
from dhdrnet.util import DATA_DIR, ROOT_DIR


class DHDRNet(pl.LightningModule):
    def __init__(
        self, learning_rate=1e-3, batch_size=8, want_summary=False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_classes = 36
        self.save_hyperparameters()

    def print_summary(self, model, size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        summary(model, (3, 300, 300))
        del model

    def prepare_data(self):
        transform = Compose(
            [
                Resize((300, 300)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),
            ]
        )

        data_df = pd.read_csv(ROOT_DIR / "precomputed_data" / "store_current.csv")
        trainval_data = LUTDataset(
            df=data_df,
            exposure_path=DATA_DIR / "correct_exposures" / "exposures",
            raw_dir=DATA_DIR / "dngs",
            name_list=ROOT_DIR / "precomputed_data" / "train_current.csv",
            transform=transform,
        )

        test_data = LUTDataset(
            df=data_df,
            exposure_path=DATA_DIR / "correct_exposures" / "exposures",
            raw_dir=DATA_DIR / "dngs",
            name_list=ROOT_DIR / "precomputed_data" / "test_current.csv",
            transform=transform,
        )
        train_val_ratio = 0.9
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
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer)
        return [optimizer], [scheduler]

    def common_step(self, batch):
        inputs, labels, stats = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        return loss

    def training_step(self, batch, batch_idx) -> pl.TrainResult:
        loss = self.common_step(batch)
        result = pl.TrainResult(minimize=loss)
        result.log("train_loss", loss)
        return result

    # def validation_step(self, batch, batch_idx) -> pl.EvalResult:
    #     loss = self.common_step(batch)
    #     result = pl.EvalResult(loss)
    #     result.log("val_loss", loss)
    #     return result


    def test_step(self, batch, batch_idx) -> pl.EvalResult:
        loss = self.common_step(batch)
        result = pl.EvalResult(loss)
        result.log("test_loss", loss)
        return result


class DHDRMobileNet_v1(DHDRNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.feature_extractor = models.mobilenet_v2(pretrained=False)
        self.feature_extractor.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feature_extractor.last_channel, self.num_classes),
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class DHDRMobileNet_v2(DHDRNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.feature_extractor = models.mobilenet_v2(pretrained=False)
        self.feature_extractor.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_extractor.last_channel, self.num_classes),
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # features = self.feature_extractor.features(x)
        # print(f"{features.shape=}")
        x = self.feature_extractor(x)
        return x


class DHDRMobileNet_v3(DHDRNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.feature_extractor = models.mobilenet_v2(pretrained=False)
        # self.feature_extractor.eval()
        self.feature_extractor.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(
                self.feature_extractor.last_channel,
                self.feature_extractor.last_channel // 2,
            ),
            nn.BatchNorm1d(self.feature_extractor.last_channel // 2),
            nn.Linear(self.feature_extractor.last_channel // 2, self.num_classes),
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # features = self.feature_extractor.features(x)
        # print(f"{features.shape=}")
        x = self.feature_extractor(x)
        return x


class DHDRSqueezeNet(DHDRNet):
    def __init__(self, want_summary=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.inner_model = squeezenet1_1(pretrained=False, num_classes=self.num_classes)

        if want_summary:
            super().print_summary(self.inner_model, size=(1, 300, 300))

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.inner_model(x)
        return x


class DHDRSimple(DHDRNet):
    def __init__(self, want_summary=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = self._simple_model(self.num_classes)

        if want_summary:
            super().print_summary(self.model, size=(1, 300, 300))

        self.criterion = nn.CrossEntropyLoss()

    def _simple_model(self, num_classes=100):
        model = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            *[
                nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(24),
                nn.ReLU(),
                nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
                nn.Dropout2d(p=0.5),
                nn.BatchNorm2d(24),
                nn.ReLU(),
                nn.AvgPool2d(2),
            ]
            * 2,
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(1944, num_classes)
        )
        return model

    def forward(self, x):
        return self.model(x)
