import torch
from torch import nn
from torchvision import models

from dhdrnet.gen_pairs import GenAllPairs
from dhdrnet.model import DHDRNet
from dhdrnet.util import DATA_DIR
import dhdrnet.util as util
import numpy as np


class RCNet(DHDRNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.feature_extractor = models.mobilenet_v2(pretrained=False)
        self.gen = GenAllPairs(
            raw_path=DATA_DIR / "dngs",
            out_path=DATA_DIR / "correct_exposures",
            store_path=None,
            compute_scores=False,
        )

        exp_min = -3
        exp_max = 6
        exp_step = 0.25
        self.exposures = np.linspace(
            exp_min, exp_max, int((exp_max - exp_min) / exp_step + 1)
        )

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
        self.criterion = nn.MSELoss()

    def get_reconstructions(self, names, evs):
        reconstructions = [
            self.gen.get_reconstruction(name, "0.0", ev)[:,:,[2,1,0]]
            for name, ev in zip(names, evs)
        ]
        reconstructions = util.centercrop(reconstructions,)
        stacked = torch.stack(reconstructions)
        return stacked

    def common_step(self, batch):
        mid_exposures, _, names = batch
        outputs = self(mid_exposures)
        _, predicted_ev_idx = torch.max(outputs, 1)
        predicted_evs = [self.exposures[ev_idx] for ev_idx in predicted_ev_idx]
        reconstructions = self.get_reconstructions(names, predicted_evs)

        loss = self.criterion(reconstructions, ground_truth)
        return loss

    def forward(self, x):
        # features = self.feature_extractor.features(x)
        # print(f"{features.shape=}")
        x = self.feature_extractor(x)
        return x
