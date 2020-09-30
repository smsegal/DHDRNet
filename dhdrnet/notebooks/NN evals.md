---
jupyter:
  jupytext:
    formats: ipynb,md,py:percent
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: DHDRNet
    language: python
    name: dhdrnet
---

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

from dhdrnet.util import DATA_DIR, ROOT_DIR

print(DATA_DIR)
```

```python
from dhdrnet.Dataset import RCDataset

from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
)

test_data = RCDataset(
    df=pd.read_csv(ROOT_DIR / "precomputed_data" / "store_current.csv"),
    exposure_path=DATA_DIR / "correct_exposures" / "exposures",
    raw_dir=DATA_DIR / "dngs",
    name_list=ROOT_DIR / "precomputed_data" / "test_current.csv",
    transform=Compose([Resize((300, 300)), ToTensor()]),
)
test_data.data
```

```python
import torch.nn.functional as F
import torch

errors = test_data.data["mse"]
err_t = torch.tensor(errors.to_numpy())
emax, _ = err_t.max(dim=1, keepdim=True)
emin, _ = err_t.min(dim=1, keepdim=True)
# err_norm = (err_t - emin) / (emax - emin)
# print(emax.shape)
# print(err_t.shape)
err_inv = emax - err_t
error_probabilities = (err_inv / err_inv.sum(dim=1, keepdim=True)).numpy()

err_df = pd.DataFrame(error_probabilities, index=errors.index, columns=errors.columns)
err_df = pd.concat([err_df, errors], keys=("prob", "mse"))
err_df
```

```python
from random import randint

ind = randint(0, len(err_df.loc["prob"]) -1)
print(ind)
s = err_df.loc["prob"].iloc[ind].transpose()
y = err_df.loc["mse"].iloc[ind].transpose()
x = s.index
print(x)
plt.scatter(x, y)
```

```python
mean_mse = err_df.loc["mse"].aggregate("mean", axis=0).to_numpy()
mse_prob = 1 - (mean_mse / mean_mse.max())
plt.scatter(x=errors.columns, y=F.softmax(torch.tensor(mse_prob), dim=0))
```

```python
from IPython.utils import io
from pytorch_lightning import Trainer

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
gpus = "0" if torch.cuda.is_available() else None

trainer = Trainer(gpus=gpus, progress_bar_refresh_rate=0)


def load_test_model(ckpt, model_cls):
    model = model_cls.load_from_checkpoint(checkpoint_path=str(ckpt))
    model.eval()

    with io.capture_output(stdout=True, stderr=True) as _captured:
        test_score = trainer.test(model)["test_loss"]
    return test_score, model, ckpt.stem


def get_best_model(model_cls, backbone: str):
    test_scores = dict()
    for ckpt in (ROOT_DIR / "checkpoints").glob(f"*{backbone}*.ckpt"):
        test_scores[str(ckpt.stem)] = load_test_model(ckpt, model_cls)

    best_model = min(test_scores.values(), key=lambda x: x[0])
    return best_model
```

```python
from dhdrnet.reconstruction_model import RCNet
from dhdrnet.histogram_model import HistogramNet
from dhdrnet.model import DHDRMobileNet_v3, DHDRSqueezeNet
from dhdrnet.resnet_model import DHDRResnet
from dhdrnet.Dataset import LUTDataset, RCDataset
from pytorch_lightning import seed_everything


seed_everything(19)
if all(
    model_name in globals()
    for model_name in ["rcnet_name", "mobile_name", "resnet_name", "squeeze_name"]
):
    print("model names defined")
    rcnet_score, rcnet_model, _ = load_test_model(rc_name, RCNet)
    mobile_score, mobile_model, _ = load_test_model(mobile_name, DHDRMobileNet_v3)
    resnet_score, resnet_model, _ = load_test_model(resnet_name, DHDRResnet)
    squeeze_score, squeeze_model, _ = load_test_model(squeeze_name, DHDRSqueezeNet)
else:
    print("finding best models")
    rcnet_score, rcnet_model, rc_name = get_best_model(RCNet, "reconstruction")
    mobile_score, mobile_model, mobile_name = get_best_model(
        DHDRMobileNet_v3, "mobile_v3"
    )
    resnet_score, resnet_model, resnet_name = get_best_model(DHDRResnet, "resnet")
    squeeze_score, squeeze_model, squeeze_name = get_best_model(
        DHDRSqueezeNet, "squeeze"
    )

print(f"{mobile_score=}")
print(f"{resnet_score=}")
print(f"{squeeze_score=}")
print(f"{rcnet_score=}")
```

```python
from torch.utils.data import DataLoader
from more_itertools import flatten, one, collapse

evs = torch.tensor(test_data.evs)


def get_ev(evs, indices):
    return [evs[i] for i in indices]


def get_rec_predictions(model, batch, k=4):
    X, y_true, names = batch
    y_pred = model(X.to(device))
    _, pred_ev_idx = torch.topk(y_pred, k, dim=1)
    pred_ev = evs[pred_ev_idx]

    true_ev_idx = torch.argmax(y_true, dim=1)
    true_ev = evs[true_ev_idx]
    return zip(names, pred_ev.numpy(), true_ev.numpy())


def get_ce_predictions(model, batch, k=4):
    X, y_true_idx, names = batch
    y_pred = model(X.to(device))
    _, pred_ev_idx = torch.topk(y_pred, k, dim=1)
    pred_ev = evs[pred_ev_idx]

    true_ev = evs[y_true_idx]
    return zip(names, pred_ev.numpy(), true_ev.numpy())


def topk_accuracy(model, evaluator, dataloader, k=4):
    model.eval()
    names, pred_evs, true_evs = zip(
        *flatten((evaluator(model, batch, k) for batch in dataloader))
    )

    c = 0
    for predicted_evs, true_ev in zip(pred_evs, true_evs):
        if true_ev in predicted_evs:
            c += 1

    return 100.0 * c / len(names)
```

```python
rc_data = test_data
reconstruction_loader = DataLoader(rc_data, batch_size=70, num_workers=8)

# rcnet_model = RCNet.load_from_checkpoint(
#     str(ROOT_DIR / "checkpoints" / "reconstructiondhdr-epoch=173-val_loss=0.00.ckpt")
# ).to(device)

lut_data = LUTDataset(
    df=pd.read_csv(ROOT_DIR / "precomputed_data" / "store_current.csv"),
    exposure_path=DATA_DIR / "correct_exposures" / "exposures",
    raw_dir=DATA_DIR / "dngs",
    name_list=ROOT_DIR / "precomputed_data" / "test_current.csv",
    transform=Compose([Resize((300, 300)), ToTensor()]),
)
lut_loader = DataLoader(lut_data, batch_size=70, num_workers=8)
```

```python
from collections import defaultdict
from pprint import pprint

model_loader_pairs = {
    "Reconstruction": (rcnet_model, get_rec_predictions, reconstruction_loader),
    "ResNet-18": (resnet_model, get_ce_predictions, lut_loader),
    "MobileNet-v2": (mobile_model, get_ce_predictions, lut_loader),
    "SqueezeNet": (squeeze_model, get_ce_predictions, lut_loader),
}

model_topk_scores = defaultdict(list)

for k in range(1, 8):
    for name, args in model_loader_pairs.items():
        score = topk_accuracy(*args, k=k)
        model_topk_scores[name].append(score)

pprint(model_topk_scores)
```

```python
from pathlib import Path

figdir = Path(ROOT_DIR / "figures")
kdf = pd.DataFrame(model_topk_scores, index=range(1, 8))
ax = kdf.plot(
    grid="both", title="Top-K Accuracy", xlabel="K", ylabel="% Accuracy", figsize=(5, 5)
)
ax.figure.savefig(figdir / "topk_plot.pdf")
```

```python
kdf.to_latex(figdir / "topk_table.tex")
```

Ok, this is all great stuff, now what about comparing to say random?

What do we need to do that?

* Test dataset
* randomly select an exposure as secondary choice
* compare the overall MSE of this sample

```python

```
