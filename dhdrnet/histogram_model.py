from math import ceil

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split

from dhdrnet.Dataset import HistogramDataset
from dhdrnet.model import DHDRNet
from dhdrnet.util import DATA_DIR, ROOT_DIR


class HistogramNet(DHDRNet):
    def __init__(self, bins=100, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bins = bins

        self.model = nn.Linear(self.bins, self.num_classes)
        # self.model = nn.Sequential(
        #     nn.Linear(self.bins, self.bins),
        #     nn.ReLU(),
        #     nn.Linear(self.bins, self.bins),
        #     nn.ReLU(),
        #     nn.Linear(self.bins, self.num_classes),
        #     nn.ReLU(),
        # )
        self.criterion = F.cross_entropy

    def forward(self, x):
        return self.model(x)

    def prepare_data(self):
        data_df = pd.read_csv(ROOT_DIR / "precomputed_data" / "store_current.csv")
        trainval_data = HistogramDataset(
            df=data_df,
            exposure_path=DATA_DIR / "correct_exposures" / "exposures",
            raw_dir=DATA_DIR / "dngs",
            name_list=ROOT_DIR / "precomputed_data" / "train_current.csv",
            bins=self.bins,
        )

        test_data = HistogramDataset(
            df=data_df,
            exposure_path=DATA_DIR / "correct_exposures" / "exposures",
            raw_dir=DATA_DIR / "dngs",
            name_list=ROOT_DIR / "precomputed_data" / "test_current.csv",
            bins=self.bins,
        )

        train_val_ratio = 0.9
        train_len = ceil(train_val_ratio * len(trainval_data))
        val_len = len(trainval_data) - train_len
        train_data, val_data = random_split(trainval_data, lengths=(train_len, val_len))
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
