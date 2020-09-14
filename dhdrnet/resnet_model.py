import torchvision.models as models
from torch import nn
from torch.nn import functional as F

from dhdrnet.model import DHDRNet


class DHDRResnet(DHDRNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.backbone = models.resnet18(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, self.num_classes),
        )
        self.criterion = F.cross_entropy

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
