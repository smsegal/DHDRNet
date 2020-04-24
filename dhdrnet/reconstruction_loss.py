from enum import Enum
from pathlib import Path
from typing import Collection, List, Callable, Union

import cv2 as cv
import numpy as np
import torch
from torch import nn, Tensor
from dhdrnet import util
from dhdrnet.util import get_mid_exp

FuseMethod = Enum("FuseMethod", "Debevec Robertson Mertens All")


class ReconstructionLoss(nn.Module):
    fuse_fun: Callable[[List[Tensor]], Tensor]

    def __init__(self, fusemethod: FuseMethod, transforms: Callable):
        super(ReconstructionLoss, self).__init__()
        self.fuse_fun = {
            FuseMethod.Debevec: self.debevec_fuse,
            FuseMethod.Mertens: self.mertens_fuse,
            FuseMethod.Robertson: self.robertson_fuse,
        }[fusemethod]
        self.transforms = transforms

    def forward(self, inputs):
        with torch.no_grad():
            pred_paths, ground_truth = inputs
            fused_batch = []
            for pred_p in pred_paths:
                mid_exp_p = get_mid_exp(pred_p)
                mid_exp = cv.imread(str(mid_exp_p))
                predicted = cv.imread(str(pred_p))
                fused = self.transforms(np.array(self.fuse_fun([mid_exp, predicted])))
                print(f"{fused.shape=}")
                fused_batch.append(fused)
            reconstructed_hdr = torch.stack(fused_batch)
        l2 = nn.MSELoss()
        print("the below shapes should match")
        print(f"{ground_truth.shape=}")
        print(f"{reconstructed_hdr.shape=}")
        return l2(reconstructed_hdr, ground_truth)

    def mertens_fuse(self, images: List[np.ndarray]) -> Tensor:
        mertens_merger = cv.createMergeMertens()
        return torch.tensor(clip_hdr(mertens_merger.process(images)))

    def debevec_fuse(self, images: List[torch.Tensor]) -> Tensor:
        debevec_merger = cv.createMergeDebevec()
        return self._cv_fuse(images, debevec_merger.process, "debevec")

    def robertson_fuse(self, images: List[torch.Tensor]) -> Tensor:
        robertson_merger = cv.createMergeRobertson()
        return self._cv_fuse(images, robertson_merger.process, "robertson")

    def _cv_fuse(
        self, images: List[np.ndarray], fuse_func: Callable, method: str,
    ) -> Tensor:
        exposure_levels = list(range(len(images)))
        exp_min = np.min(exposure_levels)
        exp_max = np.max(exposure_levels)
        exp_normed_shift = (exposure_levels - exp_min + 1) / (exp_max - exp_min)
        tonemap = cv.createTonemap(gamma=2.2)
        hdr = fuse_func(images, times=exp_normed_shift.copy())
        result = tonemap.process(hdr.copy())

        return clip_hdr(result)


def clip_hdr(fused):
    return np.clip(fused * 255, 0, 255).astype("uint8")
