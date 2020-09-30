# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: DHDRNet
#     language: python
#     name: dhdrnet
# ---

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline

# +
from dhdrnet.Dataset import RCDataset
from dhdrnet.util import DATA_DIR, ROOT_DIR

from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
)

test_data = LUTDataset(
    df=pd.read_csv(ROOT_DIR / "precomputed_data" / "store_current.csv"),
    exposure_path=DATA_DIR / "correct_exposures" / "exposures",
    raw_dir=DATA_DIR / "dngs",
    name_list=ROOT_DIR / "precomputed_data" / "test_current.csv",
    transform=Compose([Resize((300, 300)), ToTensor()]),
)
# -

df = test_data.data
df

# +
import random as rand
# rand_ev = rand.choices(range(0,len(df["mse"].columns)), k=len(df))
# df.take(rand_ev,axis=0)
rand_sel = dict()
for name, data in df.iterrows():
    ev = rand.choice(df["mse"].columns)
    rand_sel[name] = (ev, data["mse"][ev])
    

mean_err = np.mean([x[1] for x in rand_sel.values()])
mean_err
# -




