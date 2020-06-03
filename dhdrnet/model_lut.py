from math import ceil
from typing import List, Union, Dict

from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models

from dhdrnet.Dataset import LUTDataset
from dhdrnet.unet_components import *
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
        self.classifier = nn.LogSoftmax()
        self.criterion = nn.NLLLoss(reduction="mean")

    def forward(self, x):
        x = self.inner_model(x)
        x = self.classifier(x)
        return x

    def prepare_data(self):
        transform = transforms.Compose(
            [
                transforms.CenterCrop((360, 474)),  # min dimensions along DS
                transforms.ToTensor(),
            ]
        )
        trainval_data = LUTDataset(
            ROOT_DIR / "precomputed_data" / "train.choices.csv",
            ROOT_DIR / "precomputed_data" / "train.stats.csv",
            img_dir=DATA_DIR / "merged",
            ev_range=4.0,  # keys are floats converted to strings (dumb i know)
            transform=transform,
        )

        test_data = LUTDataset(
            ROOT_DIR / "precomputed_data" / "test.choices.csv",
            ROOT_DIR / "precomputed_data" / "test.stats.csv",
            img_dir=DATA_DIR / "merged",
            ev_range=4.0,
            transform=transform,
        )
        train_val_ratio = 0.8
        train_len = ceil(train_val_ratio * len(trainval_data))
        val_len = ceil((1 - train_val_ratio) * len(trainval_data))
        train_data, val_data = random_split(trainval_data, lengths=(train_len, val_len))
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.train_data, batch_size=16, num_workers=8
        )  # collate_fn=LUTDataset.collate_fn)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_data, batch_size=8, num_workers=8
        )  # collate_fn=LUTDataset.collate_fn)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_data, batch_size=8, num_workers=8
        )  # collate_fn=LUTDataset.collate_fn)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def common_step(self, batch):
        mid_exposure, label = batch
        outputs = self(mid_exposure)
        # _, preds = torch.max(outputs, 1)

        loss = self.criterion(outputs, label)
        # loss = 1 - ssim_score
        return loss

    def training_step(self, batch, batch_idx) -> Dict[str, Union[Dict, Tensor]]:
        loss = self.common_step(batch)
        logs = {"train_loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx) -> Dict[str, Union[Dict, Tensor]]:
        loss, ssim_score = self.common_step(batch)
        logs = {"val_loss": loss}
        return {"val_loss": loss, "log": logs}

    def validation_epoch_end(
            self, outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
    ) -> Dict[str, Union[Dict, Tensor]]:
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_epoch_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx) -> Dict[str, Union[Dict, Tensor]]:
        loss, ssim_score, *_ = self.common_step(batch)
        logs = {"test_loss": loss}
        return {"test_loss": loss, "log": logs}

    def test_epoch_end(self, outputs) -> Dict[str, Union[Dict, Tensor]]:
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        tensorboard_logs = {"test_loss": avg_loss}
        return {"test_loss": avg_loss, "log": tensorboard_logs}
