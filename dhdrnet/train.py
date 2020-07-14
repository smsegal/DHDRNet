import datetime
from argparse import ArgumentParser

from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from dhdrnet.model import DHDRMobileNet, DHDRSqueezeNet
from dhdrnet.util import ROOT_DIR


def main(hparams=None):
    if hparams.backbone == "squeeze":
        Model = DHDRSqueezeNet
    elif hparams.backbone == "mobile":
        Model = DHDRMobileNet

    if checkpoint_path := hparams.checkpoint_path:
        model = Model.load_from_checkpoint(checkpoint_path=str(checkpoint_path))
    else:
        model = Model()

    logger = loggers.TensorBoardLogger(ROOT_DIR / "logs/")

    checkpoint_loss_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        prefix=hparams.backbone,
        filepath="checkpoints/dhdr-{epoch}-{val_loss:.2f}",
    )

    if checkpoint_path and not hparams.test_only:
        trainer = Trainer(
            gpus=hparams.gpus,
            logger=logger,
            checkpoint_callback=checkpoint_loss_callback,
            resume_from_checkpoint=checkpoint_path,
        )
    else:
        trainer = Trainer(
            gpus=hparams.gpus,
            logger=logger,
            weights_summary="full",
            auto_lr_find=True,
            auto_scale_batch_size="binsearch",
            checkpoint_callback=checkpoint_loss_callback,
            max_epochs=2000,
        )

    if not hparams.test_only:
        trainer.fit(model)
        trainer.save_checkpoint(
            f"checkpoints/dhdr_final{datetime.datetime.now().isoformat()}.ckpt"
        )

    trainer.test(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--backbone", choices=["mobile", "squeeze"])
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("-c", "--checkpoint-path", default=None)
    args = parser.parse_args()
    print(args)
    main(args)
