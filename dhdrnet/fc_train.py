import time
from functools import partial
from scipy.special import comb
from multiprocessing import cpu_count
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import fire
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.tensor import Tensor
from torch.types import Device
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm, trange

from dhdrnet.dataset import CachingDataset
from dhdrnet.free_choice_model import FreeChoiceDHDRNet
from dhdrnet.util import get_valid_exp_path


def fit_epoch(
    epoch: int,
    model: nn.Module,
    device: Device,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    validate: Callable,
    val_steps: int,  # validate every val_steps steps
    writer: SummaryWriter,
) -> Tuple[torch.Tensor, List, float]:
    model.train()
    time_start = time.time()
    last_idx = len(loader)

    train_losses = []
    val_losses = []
    val_counter = 0
    train_iter = tqdm(loader, desc="Train", leave=False)
    for idx, (input, target, score) in enumerate(train_iter):
        last_batch = idx == last_idx

        input, target = input.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        train_losses.append(loss)
        optimizer.step()
        writer.add_scalar("Loss/train", loss)

        # if use_cuda:
        #     torch.cuda.synchronize()

        train_duration = time_start - time.time()

        train_iter.set_postfix({"train_loss": loss.item(), "batch_idx": idx})
        if idx % val_steps == 0:
            val_loss, val_duration = validate(
                model=model, device=device, loss_fn=loss_fn
            )
            writer.add_scalar("Loss/val", val_loss)
            val_losses.append(val_loss)
            val_counter += 1

    epoch_duration = time.time() - time_start
    epoch_train_loss = torch.mean(torch.tensor(train_losses))
    return epoch_train_loss, val_losses, epoch_duration


def validate(
    model: nn.Module, device: Device, loader: DataLoader, loss_fn: nn.Module, scheduler
) -> Tuple[Tensor, float]:
    model.eval()
    with torch.no_grad():
        val_loss = []
        time_start = time.time()
        duration = 0
        val_iter = tqdm(loader, desc="Val", leave=False)
        for idx, (input, target, score) in enumerate(val_iter):
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = loss_fn(output, target)
            val_loss.append(loss)

            val_iter.set_postfix({"val_loss": loss.item(), "batch_idx": idx})

        duration = time.time() - time_start
        avg_val_loss = torch.mean(torch.tensor(val_loss))
    model.train()
    scheduler.step(avg_val_loss)
    return avg_val_loss, duration


def fit(
    model: nn.Module,
    device: Device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    writer: SummaryWriter,
):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer)
    loss_fn = nn.CrossEntropyLoss()

    epoch_iter = trange(
        num_epochs,
        desc="Epochs",
    )
    for epoch in epoch_iter:
        epoch_loss, val_losses, epoch_duration = fit_epoch(
            epoch,
            model,
            device,
            train_loader,
            loss_fn,
            optimizer,
            val_steps=50,
            validate=partial(
                validate,
                loader=val_loader,
                scheduler=scheduler,
            ),
            writer=writer,
        )

        epoch_iter.set_postfix(
            {"epoch_train_loss": epoch_loss.item(), "epoch_duration": epoch_duration},
        )
        epoch_val_loss = torch.mean(torch.tensor(val_losses))
        writer.add_scalar("Loss/epoch_train", epoch_loss, epoch)
        writer.add_scalar("Loss/epoch_val", epoch_val_loss, epoch)
    writer.close()
    return {"train_loss_final": epoch_loss, "val_loss_final": epoch_val_loss}


def test_model(
    model: nn.Module, device: Device, loader: DataLoader, loss_fn: nn.Module
):
    model.eval()
    with torch.no_grad():
        test_loss = []
        test_iter = tqdm(loader, desc="Val", leave=False)
        for idx, (input, target, score) in enumerate(test_iter):
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = loss_fn(output, target)
            test_loss.append(loss)

            test_iter.set_postfix({"test_loss": loss.item(), "batch_idx": idx})

        avg_test_loss = torch.mean(torch.tensor(test_loss))
        return avg_test_loss


def get_loaders(ds: CachingDataset, batch_size: int, trainval_split: float):
    train_len = int(len(ds) * trainval_split)
    val_len = len(ds) - train_len
    train_ds, val_ds = random_split(ds, [train_len, val_len])
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cpu_count(),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=cpu_count(),
        pin_memory=True,
    )
    return train_loader, val_loader


def train(
    num_epochs: int,
    data_dir: Path = Path("./data"),
    device: Optional[Device] = None,
    batch_size: int = 4,
    lr: float = 1e-3,
    test: bool = False,
    train_file: Path = Path("./precomputed_data/train_current.csv"),
    test_file: Path = Path("./precomputed_data/test_current.csv"),
    log_dir: Path = Path("./logs"),
    experiment_name: Optional[str] = None,
):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu") if device is None else device

    exposure_values = [-5, -3, -1, 0, 1, 3, 5]
    num_classes = comb(len(exposure_values), 2)
    model = FreeChoiceDHDRNet(num_classes)

    train_image_names = pd.read_csv(train_file, squeeze=True, dtype=str).to_list()
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
        ]
    )
    train_ds = CachingDataset(
        data_dir=data_dir,
        image_names=train_image_names,
        exposure_values=exposure_values,
        metric="psnr",
        transform=transform,
    )

    train_loader, val_loader = get_loaders(train_ds, batch_size, trainval_split=0.8)

    log_dir = get_valid_exp_path(log_dir, experiment_name)
    writer = SummaryWriter(log_dir=log_dir)

    train_results = fit(model, device, train_loader, val_loader, num_epochs, lr, writer)

    print(f"Training Done! Final Train Loss: {train_results['train_loss_final']}")
    print(f"Final Val Loss: {train_results['val_loss_final']}")

    if test:
        print("Evaluating on test set...")

        test_image_names = pd.read_csv(test_file, squeeze=True, dtype=str).to_list()
        test_ds = CachingDataset(
            data_dir=data_dir,
            image_names=test_image_names,
            exposure_values=exposure_values,
            metric="psnr",
            transform=transform,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cpu_count(),
            pin_memory=True,
        )
        test_results = test_model(
            model, device, test_loader, loss_fn=nn.CrossEntropyLoss()
        )
        print(f"Testing Done! Final Test Loss: {test_results}")


if __name__ == "__main__":
    fire.Fire(train)
