from math import ceil, floor
from typing import List, Union, Dict

import torchvision
from pytorch_msssim import ssim
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models

from dhdrnet.Dataset import HDRDataset, collate_fn
from dhdrnet.colour_utils import YPbPrColorSpace
from dhdrnet.reconstruction import reconstruct_hdr_from_pred
from dhdrnet.unet_components import *
from dhdrnet.util import DATA_DIR


class DHDRNet(LightningModule):
    def __init__(self):
        super(DHDRNet, self).__init__()
        self.colour_space = YPbPrColorSpace()

        num_classes = 4
        self.inner_model = models.squeezenet1_1(pretrained=False, num_classes=num_classes)
        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = False

        # num_features = self.inner_model.classifier.in_features
        # self.inner_model.classifier = nn.Linear(num_features, num_classes)

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
        HDRData = HDRDataset(
            DATA_DIR / "train" / "merged",
            DATA_DIR / "train" / "processed",
            transform=transform,
        )
        test_data = HDRDataset(
            DATA_DIR / "test" / "merged",
            DATA_DIR / "test" / "processed",
            transform=transform,
        )
        train_val_ratio = 0.8
        train_len = ceil(train_val_ratio * len(HDRData))
        val_len = floor((1 - train_val_ratio) * len(HDRData))
        train_data, val_data = random_split(HDRData, lengths=(train_len, val_len))
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.train_data, batch_size=16, collate_fn=collate_fn, num_workers=8
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_data, batch_size=8, collate_fn=collate_fn, num_workers=8
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_data, batch_size=8, collate_fn=collate_fn, num_workers=8
        )

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def common_step(self, batch):
        exposure_paths, mid_exposure, ground_truth = batch
        outputs = self(mid_exposure)
        _, preds = torch.max(outputs, 1)
        reconstructed_hdr = reconstruct_hdr_from_pred(
            exposure_paths, ground_truth, preds
        ).type_as(mid_exposure)

        rc_yuv, gt_yuv = (
            self.colour_space.from_rgb(im_batch)
            for im_batch in [reconstructed_hdr, ground_truth]
        )

        loss = F.mse_loss(reconstructed_hdr, ground_truth)
        ssim_score = ssim(reconstructed_hdr, ground_truth)
        # loss = 1 - ssim_score
        return loss, ssim_score, reconstructed_hdr, ground_truth

    def training_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        loss, ssim_score, *_ = self.common_step(batch)
        logs = {"train_loss": loss, "train_sim": ssim_score}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        loss, ssim_score, reconstructed_hdr, ground_truth = self.common_step(batch)
        grid_rec = torchvision.utils.make_grid(reconstructed_hdr, normalize=True)
        grid_gt = torchvision.utils.make_grid(ground_truth, normalize=True)
        self.logger.experiment.add_image("reconstructed_hdr", grid_rec, batch_idx)
        self.logger.experiment.add_image("ground_truth", grid_gt, batch_idx)
        logs = {"val_loss": loss, "val_ssim": ssim_score}
        return {"val_loss": loss, "log": logs}

    def validation_epoch_end(
            self, outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
    ) -> Dict[str, Dict[str, Tensor]]:
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_epoch_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        loss, ssim_score, *_ = self.common_step(batch)
        logs = {"test_loss": loss, "test_ssim": ssim_score}
        return {"test_loss": loss, "log": logs}

    def test_epoch_end(
            self, outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
    ) -> Dict[str, Dict[str, Tensor]]:
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        tensorboard_logs = {"test_loss": avg_loss}
        return {"test_loss": avg_loss, "log": tensorboard_logs}
