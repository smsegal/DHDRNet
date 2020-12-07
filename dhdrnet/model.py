from typing import Dict, List, Union, Callable

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models

from dhdrnet.Dataset import LUTDataset
from dhdrnet.squeezenet import squeezenet1_1


class DHDRNet(LightningModule):
    def __init__(self, learning_rate=1e-3, batch_size=8, want_summary=False):
        super().__init__()
        self.num_classes = 36
        self.learning_rate = learning_rate
        self.criterion: Callable
        self.batch_size = batch_size
        self.Dataset = LUTDataset
        self.save_hyperparameters()

    def print_summary(self, model, size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        summary(model, (3, 300, 300))
        del model

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "scheduler": scheduler,
            "monitor": "val_loss",
        }

    def common_step(self, batch):
        mid_exposure, label, image_name = batch
        outputs = self(mid_exposure)

        loss = self.criterion(outputs, label)
        return loss

    def training_step(self, batch, *_rest) -> Tensor:
        loss = self.common_step(batch)
        self.log("training_loss", loss)
        return loss

    def validation_step(self, batch, *_rest) -> Tensor:
        loss = self.common_step(batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx) -> Tensor:
        loss = self.common_step(batch)
        self.log("test_loss", loss)
        return loss


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
            self.print_summary(self.inner_model, size=(1, 300, 300))

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.inner_model(x)
        return x


class DHDRSimple(DHDRNet):
    def __init__(self, want_summary=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = self._simple_model(self.num_classes)

        if want_summary:
            self.print_summary(self.model, size=(1, 300, 300))

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
            nn.Linear(1944, self.num_classes)
        )
        return model

    def forward(self, x):
        return self.model(x)
