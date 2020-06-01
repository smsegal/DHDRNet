from argparse import ArgumentParser

from pytorch_lightning import loggers, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from dhdrnet.lightning_model import DHDRNet
from dhdrnet.model_lut import DHDRNet as LUTNet


def main(hparams=None):
    if hparams.method == "lut":
        model = LUTNet()
    elif hparams.method == "reconstruction":
        model = DHDRNet()

    logger = loggers.TensorBoardLogger("logs/")

    checkpoint_loss_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        filepath="checkpoints/dhdr-{epoch}-{val_loss:.2f}-{val_ssim:.2f}",
    )

    trainer = Trainer(
        gpus=hparams.gpus, logger=logger, checkpoint_callback=checkpoint_loss_callback
    )

    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--method", choices=["reconstruction", "lut"])
    args = parser.parse_args()
    main(args)
