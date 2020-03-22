import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
from skimage import io, transform
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models

from dhdrnet.Dataset import HDRDataset
from dhdrnet.util import get_project_root

DATA_DIR = get_project_root() / "data"
# dataloader =


def main():
    model = define_model()

    datasets = {
        split: HDRDataset(DATA_DIR / split / "merged", DATA_DIR / split / "dngs")
        for split in ["train", "val"]
    }
    dataloaders = {
        split: torch.utils.data.DataLoader(
            datasets[split], batch_size=4, shuffle=True, num_workers=4
        )
        for split in ["train", "val"]
    }

    train(model, None, None, None, 25)


def define_model(pretrained=True):
    resnext = models.resnext50_32x4d(pretrained=pretrained)
    num_classes = 5  # each of the exposure levels
    return resnext


def train(model, loss, optimizer, scheduler, num_epochs):
    since = time.time()

    best_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # set to training mode
            else:
                model.eval()

            running_loss = 0.0
            running_correct = 0

            #iterate over data
            for inputs, labels in dataloaders


if __name__ == "__main__":
    main()
