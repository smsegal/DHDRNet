from typing import List

import cv2 as cv
import numpy as np
import torch
from pytorch_msssim import ssim, ms_ssim
from torch.nn import functional as F

from dhdrnet import util
from dhdrnet.util import get_mid_exp


def get_predicted_exps(exposures, preds):
    """
    exposures: shape = [batch_size x num_exposures x channels x width x height]
    preds: shape = [batch_size] <-- one prediction per batch
    """
    shifted = shift_preds(preds)
    predicted = [exposure[pred] for exposure, pred in zip(exposures, shifted)]
    return predicted


def mertens_fuse(images: List[np.ndarray]) -> np.ndarray:
    mertens_merger = cv.createMergeMertens()
    merged = mertens_merger.process(images)

    # colour channels are BGR for some stupid reason in OpenCV
    merged_rgb = merged  # [:, :, [2, 1, 0]]
    return merged_rgb


def shift_preds(preds):
    preds[preds >= 2] += 1
    return preds


def reconstruction_stats(reconstructed, ground_truth):
    reconstructed_t = torch.tensor(reconstructed).permute(2, 0, 1)[None, :]
    ground_truth_t = torch.tensor(ground_truth).permute(2, 0, 1)[None, :]
    mse = F.mse_loss(reconstructed_t, ground_truth_t)
    ssim_score = ssim(reconstructed_t, ground_truth_t)
    ms_ssim_score = ms_ssim(reconstructed_t, ground_truth_t)
    return float(mse), float(ssim_score), float(ms_ssim_score)


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
