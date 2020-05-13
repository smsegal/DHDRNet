from enum import Enum
from typing import List

import cv2 as cv
import numpy as np
import torch

from dhdrnet import util
from dhdrnet.util import get_mid_exp

FuseMethod = Enum("FuseMethod", "Debevec Robertson Mertens All")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def clip_hdr(fused):
    return np.clip(fused * 255, 0, 255).astype("uint8")


def get_predicted_exps(exposures, preds):
    """
    exposures: shape = [batch_size x num_exposures x channels x width x height]
    preds: shape = [batch_size] <-- one prediction per batch
    """
    shifted = shift_preds(preds)
    return [exposure[pred] for exposure, pred in zip(exposures, shifted)]


def mertens_fuse(images: List[np.ndarray]) -> np.ndarray:
    mertens_merger = cv.createMergeMertens()
    return clip_hdr(mertens_merger.process(images))


def shift_preds(preds):
    preds[preds >= 2] += 1
    return preds


def reconstruct_hdr_from_pred(exposure_paths, ground_truth, preds):
    with torch.no_grad():
        selected_exposures = get_predicted_exps(exposure_paths, preds)
        fused_batch = []
        for pred_p in selected_exposures:
            mid_exp_p = get_mid_exp(pred_p)
            mid_exp = cv.imread(str(mid_exp_p))
            predicted = cv.imread(str(pred_p))
            fused = torch.tensor(mertens_fuse([mid_exp, predicted]), dtype=torch.float)

            fused_batch.append(fused)
        # last two entries of shape are w,h for a torch.tensor
    centercrop = util.centercrop(fused_batch, ground_truth.shape[2:])
    reconstruction = torch.stack(centercrop).permute(0, 3, 1, 2)
    reconstruction.requires_grad_()
    return reconstruction
