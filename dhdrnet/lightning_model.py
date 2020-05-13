from math import ceil, floor
from typing import List, Union, Dict

from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torch.optim import Adam
import torch
from torch.nn import functional as F
from torch import nn, Tensor
from pytorch_msssim import ssim, ms_ssim

from dhdrnet.Dataset import HDRDataset, collate_fn
from dhdrnet.util import DATA_DIR
from reconstruction import reconstruct_hdr_from_pred
from dhdrnet.unet_components import *


# parts of network taken from https://github.com/milesial/Pytorch-UNet according to license

class DHDRNet(LightningModule):
    def __init__(self, bilinear=False):
        super(DHDRNet, self).__init__()
        num_classes = 4
        self.model = models.resnet50(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        # n_channels = 3
        # n_classes = 4
        # self.bilinear = bilinear
        #
        # self.inc = DoubleConv(n_channels, 64)
        # self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        # factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)
        # self.up1 = Up(1024, 512 // factor, bilinear)
        # self.up2 = Up(512, 256 // factor, bilinear)
        # self.up3 = Up(256, 128 // factor, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        # self.outc = OutConv(64, n_classes)

    def forward(self, x):
        preds = self.model(x)
        return preds

    # x1 = self.inc(x)
    # x2 = self.down1(x1)
    # x3 = self.down2(x2)
    # x4 = self.down3(x3)
    # x5 = self.down4(x4)
    # x = self.up1(x5, x4)
    # x = self.up2(x, x3)
    # x = self.up3(x, x2)
    # x = self.up4(x, x1)
    # logits = self.outc(x)
    # return logits

    def prepare_data(self):
        transform = transforms.Compose(
            [
                transforms.CenterCrop((360, 474)),  # min dimensions along DS
                transforms.ToTensor(),
            ]
        )
        HDRData = HDRDataset(DATA_DIR / "train" / "merged",
                             DATA_DIR / "train" / "processed",
                             transform=transform)
        test_data = HDRDataset(DATA_DIR / "test" / "merged",
                               DATA_DIR / "test" / "processed",
                               transform=transform)
        train_val_ratio = 0.8
        train_len = ceil(train_val_ratio * len(HDRData))
        val_len = floor((1 - train_val_ratio) * len(HDRData))
        train_data, val_data = random_split(HDRData, lengths=[train_len, val_len])
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.train_data, batch_size=8, collate_fn=collate_fn, num_workers=4)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_data, batch_size=4, collate_fn=collate_fn, num_workers=4)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_data, batch_size=4, collate_fn=collate_fn, num_workers=4)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        exposure_paths, mid_exposure, ground_truth = batch
        outputs = self(mid_exposure)
        _, preds = torch.max(outputs, 1)
        reconstructed_hdr = reconstruct_hdr_from_pred(
            exposure_paths, ground_truth, preds
        ).type_as(mid_exposure)
        loss = F.mse_loss(reconstructed_hdr, ground_truth)
        ssim_score = ssim(reconstructed_hdr, ground_truth)
        logs = {'train_loss': loss, "train_sim":ssim_score}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        exposure_paths, mid_exposure, ground_truth = batch
        outputs = self(mid_exposure)
        _, preds = torch.max(outputs, 1)
        reconstructed_hdr = reconstruct_hdr_from_pred(
            exposure_paths, ground_truth, preds
        ).type_as(mid_exposure)
        loss = F.mse_loss(reconstructed_hdr, ground_truth)
        ssim_score = ssim(reconstructed_hdr, ground_truth)
        logs = {"val_loss": loss, "val_ssim": ssim_score}
        return {"val_loss": loss, 'log': logs}

    def validation_epoch_end(
            self,
            outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
    ) -> Dict[str, Dict[str, Tensor]]:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_epoch_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        exposure_paths, mid_exposure, ground_truth = batch
        outputs = self(mid_exposure)
        _, preds = torch.max(outputs, 1)
        reconstructed_hdr = reconstruct_hdr_from_pred(
            exposure_paths, ground_truth, preds
        ).type_as(mid_exposure)
        loss = F.mse_loss(reconstructed_hdr, ground_truth)
        ssim_score = ssim(reconstructed_hdr, ground_truth)
        logs = {"test_loss": loss, "test_ssim": ssim_score}
        return {"test_loss": loss, 'log': logs}

    def test_epoch_end(
            self,
            outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
    ) -> Dict[str, Dict[str, Tensor]]:
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, "log": tensorboard_logs}
