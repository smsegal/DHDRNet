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

data_dir = get_project_root() / "data"
# dataloader =

def define_model(pretrained=True):
    resnext = models.resnext50_32x4d(pretrained=pretrained)
