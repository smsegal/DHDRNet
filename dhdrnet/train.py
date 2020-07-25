import datetime
from argparse import ArgumentParser

from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from dhdrnet import model as models
from dhdrnet.util import ROOT_DIR


def main(hparams=None):
    if hparams.backbone == "squeeze":
        Model = models.DHDRSqueezeNet
    elif hparams.backbone == "mobile_v1":
        Model = models.DHDRMobileNet_v1
    elif hparams.backbone == "mobile_v2":
        Model = models.DHDRMobileNet_v2
    elif hparams.backbone == "mobile_v3":
        Model = models.DHDRMobileNet_v3
    elif hparams.backbone == "simple":
        Model = models.DHDRSimple

    checkpoint_path = hparams.checkpoint_path
    model = Model(
        batch_size=int(hparams.batch_size),
        learning_rate=int(hparams.learning_rate),
        want_summary=hparams.summary,
    )

    timestamp = datetime.datetime.now().isoformat()
    logger = loggers.TensorBoardLogger(
        ROOT_DIR / "logs", name=f"dhdr_{hparams.backbone}"
    )

    checkpoint_loss_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        prefix=hparams.backbone,
        filepath=ROOT_DIR / "checkpoints/dhdr-{epoch}-{val_loss:.2f}",
    )

    early_stopping = EarlyStopping("val_loss", patience=15)

    trainer = Trainer(
        gpus=hparams.gpus,
        logger=logger,
        weights_summary=None,
        auto_lr_find=hparams.auto_lr,
        auto_scale_batch_size="binsearch" if hparams.auto_batch else None,
        checkpoint_callback=checkpoint_loss_callback,
        early_stop_callback=early_stopping,
        resume_from_checkpoint=checkpoint_path if checkpoint_path else None,
        max_epochs=2000,
    )

    if not hparams.test_only:
        trainer.fit(model)
        trainer.save_checkpoint(
            ROOT_DIR / f"checkpoints/dhdr_{hparams.backbone}_final{timestamp}.ckpt"
        )

    trainer.test(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus", default=None)
    parser.add_argument(
        "--backbone",
        choices=["mobile_v1", "mobile_v2", "mobile_v3", "squeeze", "simple"],
    )
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("-c", "--checkpoint-path", default=None)
    parser.add_argument(
        "--summary", help="print a summary of model weights", action="store_true"
    )
    parser.add_argument("--auto-batch", action="store_true")
    parser.add_argument("--batch-size", default=24)
    parser.add_argument("--auto-lr", action="store_true")
    parser.add_argument("--learning-rate", "-l", default=1e-3)
    args = parser.parse_args()
    print(args)
    main(args)
