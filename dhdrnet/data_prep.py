from dhdrnet.util import ROOT_DIR, DATA_DIR
import pandas as pd
from pathlib import Path


def split_precomputed_dataset(fp: Path, ratio: float):
    """
   Splits the given csv file of the dataset into train, test portions
   """
    df = pd.read_csv(fp)
    df.set_index("name")

    train = df.sample(frac=ratio)
    test = df.loc[df.index.difference(train.index)]
    return train, test
