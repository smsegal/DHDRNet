from dhdrnet.lightning_model import DHDRNet
from pytorch_lightning import loggers, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch


def main(hparams=None):
    model = DHDRNet()
    logger = loggers.TensorBoardLogger("logs/")

    num_gpu = 1 if torch.cuda.is_available() else 0

    checkpoint_loss_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        filepath="checkpoints/dhdr-{epoch}-{val_loss:.2f}-{val_ssim:.2f}",
    )

    trainer = Trainer(
        gpus=num_gpu, logger=logger, checkpoint_callback=checkpoint_loss_callback
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
