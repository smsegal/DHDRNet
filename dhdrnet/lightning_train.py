from argparse import ArgumentParser
from pathlib import Path

from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from dhdrnet.lightning_model import DHDRNet
from dhdrnet.model_lut import DHDRNet as LUTNet


def main(hparams=None):
    if hparams.method == "lut":
        Model = LUTNet
    elif hparams.method == "reconstruction":
        Model = DHDRNet

    if (checkpoint_path := Path(hparams.checkpoint_path)).exists():
        model = Model.load_from_checkpoint(checkpoint_path=str(checkpoint_path))
    else:
        model = Model()

    logger = loggers.TensorBoardLogger("logs/")

    checkpoint_loss_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        filepath="checkpoints/dhdr_{method}-{epoch}-{val_loss:.2f}-{val_ssim:.2f}",
    )

    trainer = Trainer(
        gpus=hparams.gpus, logger=logger, checkpoint_callback=checkpoint_loss_callback
    )

    if not hparams.test_only:
        trainer.fit(model)

    trainer.test(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--method", choices=["reconstruction", "lut"])
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("-c", "--checkpoint-path")
    args = parser.parse_args()
    print(args)
    main(args)
