from pytorch_lightning import Trainer

def main(hparams=None):


    logger = loggers.TensorBoardLogger("logs/")
    trainer = Trainer(
        gpus=hparams.gpus, 
    )
