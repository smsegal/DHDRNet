import torch
import torchvision as tv
from torchvision import models
from skimage import io, transform
import numpy as np


def define_model(num_classes: int, pretrained=True):
    resnext = models.resnext50_32x4d(pretrained=pretrained)
