from math import ceil
from typing import List, Union, Dict

from pytorch_lightning import LightningModule
from torch import Tensor
import torch.nn as nn
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models

from dhdrnet.Dataset import LUTDataset
from dhdrnet.util import DATA_DIR, ROOT_DIR


class DHDRNet(LightningModule):
    def __init__(self):
        super(DHDRNet, self).__init__()

        num_classes = 4
        self.inner_model = models.squeezenet1_1(
            pretrained=False, num_classes=num_classes
        )
        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = False
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, x):
        x = self.inner_model(x)
        return x

    def prepare_data(self):
        transform = transforms.Compose(
            [
                transforms.CenterCrop((360, 474)),  # min dimensions along DS
                transforms.ToTensor(),
            ]
        )
        trainval_data = LUTDataset(
            choice_path=ROOT_DIR
            / "precomputed_data"
            / "store_exposure_correct_cleaned.csv",
            exposure_path=DATA_DIR / "correct_exposures" / "exposures",
            name_list=ROOT_DIR / "train.txt",
            transform=transform,
        )

        test_data = LUTDataset(
            choice_path=ROOT_DIR
            / "precomputed_data"
            / "store_exposure_correct_cleaned.csv",
            exposure_path=DATA_DIR / "correct_exposures" / "exposures",
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
        return DataLoader(self.train_data, batch_size=16, num_workers=8)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_data, batch_size=8, num_workers=8)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_data, batch_size=8, num_workers=8)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def common_step(self, batch):
        mid_exposure, label, stats = batch
        outputs = self(mid_exposure)
        _, preds = torch.max(outputs, 1)

        loss = self.criterion(preds, label)
        # loss = 1 - ssim_score
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
        self, outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
    ) -> Dict[str, Union[Dict, Tensor]]:
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_epoch_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx) -> Dict[str, Union[Dict, Tensor]]:
        loss = self.common_step(batch)
        logs = {"test_loss": loss}
        return {"test_loss": loss, "log": logs}

    def test_epoch_end(self, outputs) -> Dict[str, Union[Dict, Tensor]]:
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        tensorboard_logs = {"test_loss": avg_loss}
        return {"test_loss": avg_loss, "log": tensorboard_logs}
