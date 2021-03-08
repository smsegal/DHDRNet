import timm
import torch
import torch.nn as nn


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
