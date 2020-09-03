from pytorch_lightning import LightningModule
from dhdrnet.Dataset import LUTDataset
from dhdrnet.model import DHDRNet


class HistogramNet(DHDRNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
