from enum import Enum

import numpy as np
import cv2 as cv
from torch import nn

FuseMethod = Enum("FuseMethod", "Debevec Robertson Mertens All")


class ReconstructionLoss(nn.Module):
    def __init__(self, fusemethod: FuseMethod):
        super(ReconstructionLoss, self).__init__()
        self.fuse_fun = {
            FuseMethod.Debevec: self.debevec_fuse,
            FuseMethod.Mertens: self.mertens_fuse,
            FuseMethod.Robertson: self.robertson_fuse,
        }[fusemethod]

    def forward(self, inputs, target):
        with torch.no_grad():
            reconstructed_hdr = self.fuse_fun(inputs)
        l2 = nn.MSELoss()
        return l2(target, reconstructed_hdr)

    def mertens_fuse(self, images: List[torch.Tensor]) -> Path:
        mertens_merger = cv.createMergeMertens()
        return clip_hdr(mertens_merger.process(images))

    def debevec_fuse(self, images: Collection[Path]) -> Path:
        debevec_merger = cv.createMergeDebevec()
        return self._cv_fuse(images, debevec_merger.process, "debevec")

    def robertson_fuse(self, images: Collection[Path]) -> Path:
        robertson_merger = cv.createMergeRobertson()
        return self._cv_fuse(images, robertson_merger.process, "robertson")

    def _cv_fuse(
        self, images: Collection[Path], fuse_func: Callable, method: str,
    ) -> Path:
        loaded_images = [cv.imread(str(img_path)) for img_path in images]
        exposure_levels = [int(image.name.split(".")[1]) for image in images]
        exp_min = np.min(exposure_levels)
        exp_max = np.max(exposure_levels)
        exp_normed_shift = (exposure_levels - exp_min + 1) / (exp_max - exp_min)
        tonemap = cv.createTonemap(gamma=2.2)
        hdr = fuse_func(loaded_images, times=exp_normed_shift.copy())
        result = tonemap.process(hdr.copy())

        return clip_hdr(result)


class CVFuse:
    def __call__(self, images: List[torch.Tensor]):
        shapes = [i.shape for i in images]
        reshaped = [i.permute(0, 3, 2, 1) for i in images]
        return self.fuse_fun(reshaped)


def clip_hdr(fused):
    return np.clip(fused * 255, 0, 255).astype("uint8")
