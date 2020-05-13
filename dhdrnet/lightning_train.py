from dhdrnet.lightning_model import DHDRNet
from pytorch_lightning import loggers, Trainer


def main(hparams=None):
    model = DHDRNet()
    logger = loggers.TensorBoardLogger("logs/")
    trainer = Trainer(gpus=1, logger=logger)
    trainer.fit(model)


if __name__ == "__main__":
    main()
