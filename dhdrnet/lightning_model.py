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



class DHDRNet(LightningModule):
    def __init__(self, bilinear=False):
        super(DHDRNet, self).__init__()
        num_classes = 4
        self.feature_extractor = models.resnet50(pretrained=True)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        num_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x

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
        return DataLoader(self.train_data, batch_size=16, collate_fn=collate_fn, num_workers=8)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_data, batch_size=8, collate_fn=collate_fn, num_workers=8)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_data, batch_size=8, collate_fn=collate_fn, num_workers=8)

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
        logs = {'train_loss': loss, "train_sim": ssim_score}
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
