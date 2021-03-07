from functools import partial
from pathlib import Path
import time
from multiprocessing import cpu_count
from typing import Callable, List, Optional, Tuple, Union

import timm
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.tensor import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm


class FreeChoiceDHDRNet(nn.Module):
    """num_classes will be n choose 2 for n = number of exposure values in consideration"""

    def __init__(self, num_classes: int):
        super().__init__()
        backbone = timm.create_model(
            "mobilenetv3_large_100",
            pretrained=True,
        )

        # freeze inner layers and just fine-tune the classifier
        for param in backbone.parameters():
            param.requires_grad = False

        backbone.classifier = nn.Linear(backbone.num_features, num_classes)

        self.backbone = backbone

        # now the idea is to use a 3d Convolution of the given images (given in EV order)
        # as input, and then output a class corresponding to the two
        # best exposure values
        self.inconv = nn.Conv3d(in_channels=7, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.inconv(x)
        x = torch.squeeze(x)
        x = self.backbone(x)
        return x


def train_epoch(
    epoch: int,
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    validate: Callable,
    val_steps: int,  # validate every val_steps steps
    writer: SummaryWriter,
    use_cuda: bool = False,
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

        if use_cuda:
            input, target = input.cuda(), target.cuda()

        optimizer.zero_grad()

        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        train_losses.append(loss)
        optimizer.step()
        writer.add_scalar("Loss/train", loss)

        if use_cuda:
            torch.cuda.synchronize()

        train_duration = time_start - time.time()

        train_iter.set_postfix({"train_loss": loss.item(), "batch_idx": idx})
        if idx % val_steps == 0:
            val_loss, val_duration = validate(model=model, use_cuda=use_cuda)
            writer.add_scalar("Loss/val", val_loss)
            val_losses.append(val_loss)
            val_counter += 1

    epoch_duration = time.time() - time_start
    epoch_train_loss = torch.mean(torch.tensor(train_losses))
    return epoch_train_loss, val_losses, epoch_duration


def validate(
    model: nn.Module, loader, loss_fn, scheduler, use_cuda=False
) -> Tuple[Tensor, float]:
    model.eval()
    with torch.no_grad():
        val_loss = []
        time_start = time.time()
        duration = 0
        val_iter = tqdm(loader, desc="Val", leave=False)
        for idx, (input, target, score) in enumerate(val_iter):
            if use_cuda:
                input, target = input.cuda(), target.cuda()
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
    train_ds: Dataset,
    val_ds: Dataset,
    batch_size: int,
    num_epochs: int,
    learning_rate: Optional[float] = None,
    exp_name: str = "0",
    logdir: Union[Path, str] = "../logs",
):
    optimizer = Adam(
        model.parameters(),
        lr=(learning_rate if learning_rate is not None else 1e-3),
    )
    scheduler = ReduceLROnPlateau(optimizer)
    loss_fn = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=f"{logdir}/exp_{exp_name}")

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

    epoch_iter = trange(
        num_epochs,
        desc="Epochs",
    )
    for epoch in epoch_iter:
        epoch_loss, val_losses, epoch_duration = train_epoch(
            epoch,
            model,
            train_loader,
            loss_fn,
            optimizer,
            val_steps=50,
            validate=partial(
                validate,
                loader=val_loader,
                loss_fn=loss_fn,
                scheduler=scheduler,
            ),
            use_cuda=torch.cuda.is_available(),
            writer=writer,
        )

        epoch_iter.set_postfix(
            {"epoch_train_loss": epoch_loss.item(), "epoch_duration": epoch_duration},
        )
        writer.add_scalar("Loss/epoch_train", epoch_loss, epoch)
        writer.add_scalar("Loss/epoch_val", torch.mean(torch.tensor(val_losses)), epoch)
    writer.close()
