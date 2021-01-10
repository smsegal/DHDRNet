from dhdrnet.data_module import HDRDataModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import fire
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from dhdrnet.histogram_model import HistogramNet
from typing import Dict, Optional
from pathlib import Path
from dhdrnet import model as models
from dhdrnet import resnet_model
from dhdrnet import reconstruction_model
import pytorch_lightning as pl


ModelClass: Dict[str, models.DHDRNet] = {
    "squeeze": models.DHDRSqueezeNet,
    "mobile_v1": models.DHDRMobileNet_v1,
    "mobile_v2": models.DHDRMobileNet_v2,
    "mobile_v3": models.DHDRMobileNet_v3,
    "simple": models.DHDRSimple,
    "hist": HistogramNet,
    "resnet": resnet_model.DHDRResnet,
    "reconstruction": reconstruction_model.RCNet,
}


def train(
    backbone: str,
    checkpoint_path: Path,
    batch_size: int = 20,
    learning_rate: float = 1e-3,
    gpus: int = 1,
    resume_path: Optional[Path] = None,
    summarize: bool = False,
    auto_scale_batch: Optional[str] = None,
    auto_learning_rate: bool = False,
    stage: str = "train",
):
    Model = ModelClass[backbone]
    model = Model(
        batch_size=batch_size, learning_rate=learning_rate, want_summary=summarize
    )

    data_module = HDRDataModule()

    trainer = pl.Trainer(
        gpus=gpus,
        auto_lr_find=auto_learning_rate,
        auto_scale_batch_size=auto_scale_batch,
        checkpoint_callback=True,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=15),
            ModelCheckpoint(
                dirpath=Path(checkpoint_path) / backbone,
                save_last=True,
                monitor="val_loss",
            ),
        ],
        resume_from_checkpoint=resume_path,
    )

    if stage == "train":
        trainer.fit(model, datamodule=data_module)

    trainer.test()


if __name__ == "__main__":
    fire.Fire(train)
