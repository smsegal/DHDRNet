import pytorch_lightning as pl
from math import ceil
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dhdrnet.util import DATA_DIR, ROOT_DIR
from dhdrnet.dataset import LUTDataset
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd


class HDRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        precomputed_data_dir: Path = ROOT_DIR / "precomputed_data",
        batch_size: int = 20,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.precomputed_data_dir = precomputed_data_dir
        self.batch_size = batch_size
        self.train_val_ratio = 0.9
        self.transform = transforms.Compose(
            [
                transforms.Resize((300, 300)),
                transforms.ToTensor(),
            ]
        )

    def setup(self, stage: Optional[str] = None):
        self.df: pd.DataFrame = pd.read_csv(
            self.precomputed_data_dir / "store_current.csv"
        )

        if stage == "fit" or stage is None:
            hdr_full = LUTDataset(
                df=self.df,
                exposure_path=(self.data_dir / "exposures"),
                raw_dir=self.data_dir / "dngs",
                name_list=self.precomputed_data_dir / "train_current.csv",
                transform=self.transform,
            )
            train_len = ceil(self.train_val_ratio * len(hdr_full))
            self.hdr_train, self.hdr_val = random_split(
                hdr_full, lengths=(train_len, len(hdr_full) - train_len)
            )
            self.dims = tuple(self.hdr_train[0][0].shape)

        if stage == "test" or stage is None:
            self.hdr_test = LUTDataset(
                df=self.df,
                exposure_path=(self.data_dir / "exposures"),
                raw_dir=self.data_dir / "dngs",
                name_list=self.precomputed_data_dir / "test_current.csv",
                transform=self.transform,
            )
            self.dims = tuple(self.hdr_test[0][0].shape)

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.hdr_train,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.hdr_val, batch_size=self.batch_size, pin_memory=True, num_workers=8
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.hdr_test, batch_size=self.batch_size, pin_memory=True, num_workers=8
        )
