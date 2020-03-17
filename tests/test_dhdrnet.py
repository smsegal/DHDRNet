import numpy as np

from dhdrnet import __version__
from dhdrnet.Dataset import HDRDataset
from dhdrnet.util import get_project_root


def test_version():
    assert __version__ == "0.1.0"


DATA_DIR = get_project_root() / "data"


def test_sample_load():
    dataset = HDRDataset(
        gt_dir=DATA_DIR / "merged" / "mertens", raw_dir=DATA_DIR / "dngs"
    )
    sample_size = 100
    for i in np.random.choice(len(dataset), size=sample_size):
        sample = dataset[i]
        exposures, gt, _name = sample.values()
        exposure_shapes = [e.shape for e in exposures]
        for e_shape in exposure_shapes:
            assert e_shape == gt.shape
