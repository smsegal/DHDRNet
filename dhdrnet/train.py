import copy
import time
from datetime import datetime
from enum import Enum
from typing import List

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms

import dhdrnet.util as util
from dhdrnet.Dataset import HDRDataset, collate_fn
from dhdrnet.image_loader import clip_hdr
from dhdrnet.util import DATA_DIR, MODEL_DIR, get_mid_exp

DEBUG = False


class Phase(Enum):
    TRAIN = "train"
    EVAL = "val"


data_transforms = {
    Phase.TRAIN: transforms.Compose(
        [
            transforms.CenterCrop((360, 474)),  # min dimensions along DS
            transforms.ToTensor(),
        ]
    ),
    Phase.EVAL: transforms.Compose(
        [
            transforms.CenterCrop((360, 474)),  # min dimensions along DS
            transforms.ToTensor(),
        ]
    ),
}
datasets = {
    split: HDRDataset(
        DATA_DIR / split.value / "merged",
        DATA_DIR / split.value / "processed",
        transforms=data_transforms[split],
    )
    for split in Phase
}
batch_sizes = {
    Phase.TRAIN: 4,
    Phase.EVAL: 8,  # double the batch size for val cause jeremy howard said so
}
dataloaders = {
    split: torch.utils.data.DataLoader(
        datasets[split],
        batch_size=batch_sizes[split],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    for split in Phase
}
dataset_sizes = {x: len(datasets[x]) for x in Phase}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def mertens_fuse(images: List[np.ndarray]) -> np.ndarray:
    mertens_merger = cv.createMergeMertens()
    return clip_hdr(mertens_merger.process(images))


def main(debug: bool = None):
    if debug is not None:
        DEBUG = debug
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    num_classes = 4  # number of exposures

    model.fc = nn.Linear(num_features, num_classes)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    epochs = 100
    steps_per_epoch = 100
    trained = train(
        model,
        criterion,
        optimizer,
        exp_lr_scheduler,
        num_epochs=epochs,
        steps_per_epoch=steps_per_epoch,
    )

    timestamp = datetime.now().strftime("%m-%d-%R")
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": trained.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": criterion,
        },
        MODEL_DIR / f"dhdr_{timestamp}.pt",
    )


def train(model, loss_fun, optimizer, scheduler, num_epochs, steps_per_epoch=100):
    since = time.time()

    best_weights = copy.deepcopy(model.state_dict())
    best_loss = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        for phase in Phase:
            if phase == Phase.TRAIN:
                model.train()  # set to training mode
                for step in range(steps_per_epoch):
                    # iterate over data
                    running_loss = fit(
                        dataloaders, model, phase, loss_fun, device, optimizer
                    )
                    scheduler.step()
            else:
                model.eval()

            epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_correct.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f}")

            # deep copy model if eval time
            if phase == Phase.EVAL and epoch_loss > best_loss:
                best_loss = epoch_loss
                best_weights = copy.deepcopy(model.state_dict())

            if epoch % 10 == 0:
                timestamp = datetime.now().strftime("%m-%d-%R")
                torch.save(
                    model.state_dict(),
                    MODEL_DIR / "checkpoints" / f"dhdr_checkpoint_{timestamp}_epoch{epoch}.pt",
                )

        print()  # line sep

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    # print(f"Best val Acc: {best_acc:4f}")

    model.load_state_dict(best_weights)
    return model


def fit(dataloaders, model, phase, loss_fun, device, optimizer):
    running_loss = 0.0
    running_correct = 0
    for batch in dataloaders[phase]:
        exposure_paths, mid_exposure, ground_truth = batch
        mid_exposure = mid_exposure.to(torch.float32).to(device)
        ground_truth = ground_truth.to(torch.float32).to(device)

        optimizer.zero_grad()

        # forwards pass
        with torch.set_grad_enabled(phase == Phase.TRAIN):
            outputs = model(mid_exposure)
            _, preds = torch.max(outputs, 1)
            reconstructed_hdr = reconstruct_hdr_from_pred(
                exposure_paths, ground_truth, preds
            )
            loss = loss_fun(reconstructed_hdr.to(device), ground_truth)

            # backwards + optim in training
            if phase == Phase.TRAIN:
                loss.backward()
                optimizer.step()

        # stats
        running_loss += loss.item() * mid_exposure.size(0)
        # running_correct += torch.sum( == ground_truth.data)
        return running_loss  # , running_correct


def reconstruct_hdr_from_pred(exposure_paths, ground_truth, preds):
    with torch.no_grad():
        selected_exposures = get_predicted_exps(exposure_paths, preds)
        fused_batch = []
        for pred_p in selected_exposures:
            mid_exp_p = get_mid_exp(pred_p)
            mid_exp = cv.imread(str(mid_exp_p))
            predicted = cv.imread(str(pred_p))
            fused = torch.tensor(mertens_fuse([mid_exp, predicted]), dtype=torch.float)

            # print(f"{fused.shape=}")
            # print(f"{type(fused)=}")
            fused_batch.append(fused)
        # last two entries of shape are w,h for a torch.tensor
    centercrop = util.centercrop(fused_batch, ground_truth.shape[2:])
    reconstruction = torch.stack(centercrop).permute(0, 3, 1, 2)
    reconstruction.requires_grad_()
    return reconstruction


def get_predicted_exps(exposures, preds):
    """
    exposures: shape = [batch_size x num_exposures x channels x width x height]
    preds: shape = [batch_size] <-- one prediction per batch
    """
    shifted = shift_preds(preds)
    return [exposure[pred] for exposure, pred in zip(exposures, shifted)]


def shift_preds(preds):
    preds[preds >= 2] += 1
    return preds


if __name__ == "__main__":
    main()
