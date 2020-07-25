import datetime
from argparse import ArgumentParser

from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from dhdrnet import model as models
from dhdrnet.util import ROOT_DIR


def main(hparams=None):
    if hparams.backbone == "squeeze":
        Model = models.DHDRSqueezeNet
    elif hparams.backbone == "mobile":
        Model = models.DHDRMobileNet_v3
    elif hparams.backbone == "simple":
        Model = models.DHDRSimple

    if checkpoint_path := hparams.checkpoint_path:
        model = Model.load_from_checkpoint(checkpoint_path=str(checkpoint_path))
    else:
        model = Model(
            use_tencrop=hparams.tencrop,
            batch_size=200,
            learning_rate=1e-3,
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

    early_stopping = EarlyStopping("val_loss", patience=5)

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
            weights_summary=None,
            auto_lr_find=True,
            auto_scale_batch_size="power" if hparams.auto_batch else None,
            checkpoint_callback=checkpoint_loss_callback,
            early_stop_callback=early_stopping,
            max_epochs=2000,
        )

    if not hparams.test_only:
        trainer.fit(model)
        trainer.save_checkpoint(f"checkpoints/dhdr_final{timestamp}.ckpt")

    trainer.test(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--backbone", choices=["mobile", "squeeze", "simple"])
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--tencrop", action="store_true")
    parser.add_argument("-c", "--checkpoint-path", default=None)
    parser.add_argument(
        "--summary", help="print a summary of model weights", action="store_true"
    )
    parser.add_argument("--auto-batch", action="store_true")
    args = parser.parse_args()
    print(args)
    main(args)
