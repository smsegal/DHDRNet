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
        trainval_data = LUTDataset(
            ROOT_DIR / "precomputed_data" / "train.csv",
            ev=4,
            img_dir=DATA_DIR / "merged",
            transform=transform,
        )

        test_data = LUTDataset(
            ROOT_DIR / "precomputed_data" / "test.csv",
            ev=4,
            img_dir=DATA_DIR / "merged",
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
        return DataLoader(self.train_data, batch_size=16, num_workers=8)  # collate_fn=LUTDataset.collate_fn)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_data, batch_size=8, num_workers=8)  # collate_fn=LUTDataset.collate_fn)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_data, batch_size=8, num_workers=8)  # collate_fn=LUTDataset.collate_fn)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def common_step(self, batch):
        mid_exposure, mse_data, ssim_data, ms_ssim_data = batch
        outputs = self(mid_exposure)
        _, preds = torch.max(outputs, 1)
        # reconstructed_hdr = reconstruct_hdr_from_pred(
        #     exposure_paths, ground_truth, preds
        # ).type_as(mid_exposure)
        preds.requires_grad_(True)
        loss = torch.index_select(mse_data, dim=0, index=preds)
        ssim_score = ssim_data[preds]
        # loss = 1 - ssim_score
        return loss, ssim_score

    def training_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        loss, ssim_score, *_ = self.common_step(batch)
        logs = {"train_loss": loss, "train_ssim": ssim_score}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        loss, ssim_score = self.common_step(batch)
        # grid_rec = torchvision.utils.make_grid(reconstructed_hdr, normalize=True)
        # grid_gt = torchvision.utils.make_grid(ground_truth, normalize=True)
        # self.logger.experiment.add_image("reconstructed_hdr", grid_rec, batch_idx)
        # self.logger.experiment.add_image("ground_truth", grid_gt, batch_idx)
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
