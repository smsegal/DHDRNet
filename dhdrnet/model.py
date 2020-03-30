import copy
import time
from enum import Enum
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
from skimage import io, transform
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models, transforms

from dhdrnet.Dataset import HDRDataset
from dhdrnet.image_loader import CVFuse, FuseMethod
from dhdrnet.util import Indexable, get_project_root

DATA_DIR = get_project_root() / "data"


class Phase(Enum):
    TRAIN = "train"
    EVAL = "val"


data_transforms = {
    Phase.TRAIN: transforms.Compose(
        [transforms.CenterCrop((360, 474))]  # min dimensions along DS
    ),
    Phase.EVAL: transforms.Compose(
        [transforms.CenterCrop((360, 474))]  # min dimensions along DS
    )
}
datasets = {
    split: HDRDataset(
        DATA_DIR / split.value / "merged",
        DATA_DIR / split.value / "dngs",
        transforms=data_transforms[split],
    )
    for split in Phase
}
dataloaders = {
    split: torch.utils.data.DataLoader(
        datasets[split], batch_size=4, shuffle=True, num_workers=4,
    )
    for split in Phase
}
dataset_sizes = {x: len(datasets[x]) for x in Phase}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    num_classes = 5  # number of exposures

    model.fc = nn.Linear(num_features, num_classes)

    fuser = CVFuse(FuseMethod.Mertens)
    criterion = ReconstructionLoss(fuser)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train(model, criterion, optimizer, exp_lr_scheduler, num_epochs=25)


class ReconstructionLoss(nn.Module):
    def __init__(self, fuse_fun):
        super(ReconstructionLoss, self).__init__()
        self.fuse_fun = fuse_fun

    def forward(self, input, target):
        pass


def define_model(pretrained=True):
    resnext = models.resnext50_32x4d(pretrained=pretrained)
    num_classes = 5  # each of the exposure levels
    return resnext


def train(model, loss_fun, optimizer, scheduler, num_epochs):
    since = time.time()

    best_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        for phase in Phase:
            if phase == Phase.TRAIN:
                model.train()  # set to training mode
            else:
                model.eval()

            # iterate over data
            running_loss, running_correct = data_opt(
                dataloaders, model, phase, loss_fun, device, optimizer
            )

            if phase == Phase.TRAIN:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_correct.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy model if eval time
            if phase == Phase.EVAL and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_weights = copy.deepcopy(model.state_dict())

        print()  # line sep

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed %60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    model.load_state_dict(best_weights)
    return model


def data_opt(dataloaders, model, phase, loss_fun, device, optimizer):
    running_loss = 0.0
    running_correct = 0
    for inputs, labels, names in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # forwards pass
        with torch.set_grad_enabled(phase == Phase.TRAIN):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_fun(outputs, labels)

            # backwards + optim in training
            if phase == Phase.TRAIN:
                loss.backward()
                optimizer.step()

        # stats
        running_loss += loss.item() * inputs.size(0)
        running_correct += torch.sum(preds == labels.data)
        return running_loss, running_correct


if __name__ == "__main__":
    main()
