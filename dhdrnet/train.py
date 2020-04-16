import copy
import time
from enum import Enum

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms

from dhdrnet.Dataset import HDRDataset
from dhdrnet.reconstruction_loss import FuseMethod, ReconstructionLoss
from dhdrnet.util import get_project_root

DATA_DIR = get_project_root() / "data"
DEBUG = False


class Phase(Enum):
    TRAIN = "train"
    EVAL = "val"


data_transforms = {
    Phase.TRAIN: transforms.Compose(
        [transforms.CenterCrop((360, 474))]  # min dimensions along DS
    ),
    Phase.EVAL: transforms.Compose(
        [transforms.CenterCrop((360, 474))]  # min dimensions along DS
    ),
}
datasets = {
    split: HDRDataset(
        DATA_DIR / split.value / "merged",
        DATA_DIR / split.value / "dngs",
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
        datasets[split], batch_size=batch_sizes[split], shuffle=True, num_workers=0,
    )
    for split in Phase
}
dataset_sizes = {x: len(datasets[x]) for x in Phase}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(debug: bool = None):
    if debug is not None:
        DEBUG = debug
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    num_classes = 4  # number of exposures

    model.fc = nn.Linear(num_features, num_classes)
    model.to(device)

    criterion = ReconstructionLoss(FuseMethod.Mertens)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train(model, criterion, optimizer, exp_lr_scheduler, num_epochs=25)


def train(model, loss_fun, optimizer, scheduler, num_epochs):
    since = time.time()

    best_weights = copy.deepcopy(model.state_dict())
    best_loss = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        for phase in Phase:
            if phase == Phase.TRAIN:
                model.train()  # set to training mode
            else:
                model.eval()

            # iterate over data
            running_loss, running_correct = fit(
                dataloaders, model, phase, loss_fun, device, optimizer
            )

            if phase == Phase.TRAIN:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_correct.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f}")

            # deep copy model if eval time
            if phase == Phase.EVAL and epoch_loss > best_loss:
                best_loss = epoch_loss
                best_weights = copy.deepcopy(model.state_dict())

        print()  # line sep

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed %60:.0f}s")
    # print(f"Best val Acc: {best_acc:4f}")

    model.load_state_dict(best_weights)
    return model


def fit(dataloaders, model, phase, loss_fun, device, optimizer):
    running_loss = 0.0
    running_correct = 0
    for batch in dataloaders[phase]:
        exposures, mid_exposure, ground_truth = batch.values()
        if DEBUG:
            for bn, bv in batch.items():
                if type(bv) is not list:
                    print(f"{bn}: {bv.shape}")
                else:
                    for e in bv:
                        print(f"{e.shape=}")
            return
        mid_exposure = mid_exposure.to(torch.float32).to(device)
        ground_truth = ground_truth.to(torch.float32).to(device)

        optimizer.zero_grad()

        # forwards pass
        with torch.set_grad_enabled(phase == Phase.TRAIN):
            outputs = model(mid_exposure)
            _, preds = torch.max(outputs, 1)

            with torch.no_grad():
                selected_exposures = get_predicted_exps(exposures, preds)

            loss = loss_fun([mid_exposure, selected_exposures], ground_truth)

            # backwards + optim in training
            if phase == Phase.TRAIN:
                loss.backward()
                optimizer.step()

        # stats
        running_loss += loss.item() * mid_exposure.size(0)
        # running_correct += torch.sum( == ground_truth.data)
        return running_loss  # , running_correct


def get_predicted_exps(exposures, preds):
    """
    exposures: shape = [batch_size x num_exposures x channels x width x height]
    preds: shape = [batch_size] <-- one prediction per batch
    """
    selected_exposures = []
    print(f"{exposures.shape=}")
    for exposure, pred in zip(exposures, preds):
        predicted_exposure = exposure[pred]
        print(f"{predicted_exposure.shape=}")
        selected_exposures.append(predicted_exposure)
    selected_exposures_t = torch.stack(selected_exposures)
    print(f"{selected_exposures_t.shape=}")
    return selected_exposures_t


if __name__ == "__main__":
    main()
