from enum import Enum
from pathlib import Path
from typing import List, Collection, Callable

import cv2 as cv
import numpy as np
import torch

FuseMethod = Enum("FuseMethod", "Debevec Robertson Mertens All")


class CVFuse:
    def __init__(self, method: FuseMethod):
        self.fuse_fun = {
            FuseMethod.Debevec: self.debevec_fuse,
            FuseMethod.Mertens: self.mertens_fuse,
            FuseMethod.Robertson: self.robertson_fuse,
        }[method]

    def __call__(self, images: List[torch.Tensor]):
        # swap channels to openCV layout from pytorch
        channel_swapped = (i.permute(5, 3, 2, 1).cpu().numpy() for i in images)
        unbatched = zip(*channel_swapped)
        return list(map(self.fuse_fun, unbatched))

    @staticmethod
    def mertens_fuse(images: List[torch.Tensor]) -> np.ndarray:
        mertens_merger = cv.createMergeMertens()
        return mertens_merger.process(images)

    def debevec_fuse(self, images: Collection[Path]) -> np.ndarray:
        debevec_merger = cv.createMergeDebevec()
        return self._cv_fuse(images, debevec_merger.process, "debevec")

    def robertson_fuse(self, images: Collection[Path]) -> np.ndarray:
        robertson_merger = cv.createMergeRobertson()
        return self._cv_fuse(images, robertson_merger.process, "robertson")

    @staticmethod
    def _cv_fuse(
            images: Collection[Path], fuse_func: Callable, method: str,
    ) -> np.ndarray:
        loaded_images = [cv.imread(str(img_path)) for img_path in images]
        exposure_levels = [int(image.name.split(".")[1]) for image in images]
        exp_min = np.min(exposure_levels)
        exp_max = np.max(exposure_levels)
        exp_normed_shift = (exposure_levels - exp_min + 1) / (exp_max - exp_min)
        tonemap = cv.createTonemap(gamma=2.2)
        hdr = fuse_func(loaded_images, times=exp_normed_shift.copy())
        result = tonemap.process(hdr.copy())

        return result
